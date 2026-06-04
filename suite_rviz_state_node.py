#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone RViz visualization node for the live robot + end-effector.

This node is meant to run *as soon as Gazebo is up* (independently from the MPC
tracking controller ``run_tracking_controller.py``). It reconstructs the robot
configuration from Gazebo / MAVROS state and the arm joint states, runs
Pinocchio forward kinematics, and publishes:

  /suite_mpc/robot_markers   (MarkerArray: per-link visual mesh / primitive)
  /suite_mpc/ee_axes         (MarkerArray: EE XYZ axes arrows) + TF "suite_mpc_ee"

It deliberately duplicates only the *visualization* portion of the tracking
controller so the user can see the robot in RViz without launching the MPC node.

Parameters (private ~):
  ~urdf_path        (str)   absolute path to the robot URDF (free-flyer base)
  ~robot_name       (str)   Gazebo model name used in /gazebo/model_states
  ~arm_enabled      (bool)  whether the robot has an arm (EE axes + joints)
  ~ee_frame_name    (str)   Pinocchio frame for the EE (default "gripper_link")
  ~odom_source      (str)   "gazebo" | "mavros"
  ~use_simulation   (bool)  True -> /arm_controller/joint_states, else /joint_states
  ~viz_frame_id     (str)   fixed frame for markers (default "map")
  ~publish_hz       (float) marker publish rate (default 20)
  ~ee_axis_length   (float) EE axis arrow length (m)
  ~ee_axis_diameter (float) EE axis arrow shaft diameter (m)
"""

import xml.etree.ElementTree as ET

import numpy as np
import rospy
import tf2_ros
import pinocchio as pin

from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


def _rpy_xyz_to_se3(origin_el):
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin_el is not None:
        if origin_el.get("xyz"):
            xyz = [float(v) for v in origin_el.get("xyz").split()]
        if origin_el.get("rpy"):
            rpy = [float(v) for v in origin_el.get("rpy").split()]
    R = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
    return pin.SE3(R, np.asarray(xyz, dtype=float))


class SuiteRvizStateNode:
    def __init__(self):
        self.urdf_path = str(rospy.get_param("~urdf_path", ""))
        self.robot_name = str(rospy.get_param("~robot_name", "s500_uam"))
        self.arm_enabled = bool(rospy.get_param("~arm_enabled", True))
        self.ee_frame_name = str(rospy.get_param("~ee_frame_name", "gripper_link"))
        self.odom_source = str(rospy.get_param("~odom_source", "gazebo")).strip().lower()
        self.use_simulation = bool(rospy.get_param("~use_simulation", True))
        self.viz_frame_id = str(rospy.get_param("~viz_frame_id", "map"))
        self.publish_hz = float(rospy.get_param("~publish_hz", 20.0))
        self.ee_axis_len = float(rospy.get_param("~ee_axis_length", 0.15))
        self.ee_axis_diam = float(rospy.get_param("~ee_axis_diameter", 0.012))

        if not self.urdf_path:
            rospy.logfatal("[suite_viz] ~urdf_path is required.")
            raise SystemExit(1)

        # ── Pinocchio model (free-flyer base) ────────────────────────────────
        self.robot_model = pin.buildModelFromUrdf(
            self.urdf_path, pin.JointModelFreeFlyer()
        )
        self.robot_data = self.robot_model.createData()
        self.nq = self.robot_model.nq
        self.arm_joint_number = max(0, self.nq - 7)

        self.ee_frame_id = None
        if self.arm_enabled:
            fid = self.robot_model.getFrameId(self.ee_frame_name)
            if 0 <= int(fid) < len(self.robot_model.frames):
                self.ee_frame_id = int(fid)
            else:
                rospy.logwarn(
                    f"[suite_viz] EE frame '{self.ee_frame_name}' not found; "
                    "EE axes disabled."
                )

        # neutral configuration; base identity, joints zero
        self.q = pin.neutral(self.robot_model)
        self._have_base = False

        self._build_robot_visual_specs()

        # ── Publishers ───────────────────────────────────────────────────────
        self.robot_markers_pub = rospy.Publisher(
            "/suite_mpc/robot_markers", MarkerArray, queue_size=2, latch=True
        )
        self.ee_axes_pub = rospy.Publisher(
            "/suite_mpc/ee_axes", MarkerArray, queue_size=2
        )
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ── Subscribers ──────────────────────────────────────────────────────
        if self.odom_source == "mavros":
            rospy.Subscriber(
                "/mavros/local_position/odom", Odometry, self._mavros_odom_cb,
                queue_size=10,
            )
        else:
            rospy.Subscriber(
                "/gazebo/model_states", ModelStates, self._gazebo_state_cb,
                queue_size=10,
            )
        if self.arm_enabled and self.arm_joint_number > 0:
            if self.use_simulation:
                rospy.Subscriber(
                    "/arm_controller/joint_states", JointState,
                    self._arm_state_sim_cb, queue_size=10,
                )
            else:
                rospy.Subscriber(
                    "/joint_states", JointState, self._arm_state_cb, queue_size=10,
                )

        # 清除 tracking node 留下的 latched marker（否则 Kill node 后 RViz 会卡在旧位姿）
        self._publish_clear_markers(rospy.Time.now())

        hz = max(self.publish_hz, 1.0)
        self._timer = rospy.Timer(rospy.Duration(1.0 / hz), self._on_timer)
        rospy.loginfo(
            f"[suite_viz] up: robot='{self.robot_name}', nq={self.nq}, "
            f"arm={self.arm_enabled}, odom={self.odom_source}, "
            f"specs={len(self._robot_visual_specs)}"
        )

    # =====================================================================
    # URDF visual parsing
    # =====================================================================
    def _build_robot_visual_specs(self):
        self._robot_visual_specs = []
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
        except Exception as e:
            rospy.logwarn(f"[suite_viz] URDF parse failed: {e}")
            return
        rm = self.robot_model
        n_mesh = 0
        for link in root.findall("link"):
            link_name = link.get("name")
            if not link_name:
                continue
            fid = rm.getFrameId(link_name)
            if int(fid) < 0 or int(fid) >= len(rm.frames):
                continue
            for visual in link.findall("visual"):
                geom = visual.find("geometry")
                if geom is None:
                    continue
                origin = _rpy_xyz_to_se3(visual.find("origin"))
                rgba = [0.8, 0.8, 0.8, 1.0]
                mat = visual.find("material")
                if mat is not None and mat.find("color") is not None:
                    c = mat.find("color").get("rgba")
                    if c:
                        rgba = [float(v) for v in c.split()]
                spec = {"frame_id": int(fid), "origin": origin, "rgba": rgba}
                mesh = geom.find("mesh")
                box = geom.find("box")
                sphere = geom.find("sphere")
                cylinder = geom.find("cylinder")
                if mesh is not None and mesh.get("filename"):
                    scale = [1.0, 1.0, 1.0]
                    if mesh.get("scale"):
                        scale = [float(v) for v in mesh.get("scale").split()]
                    spec.update({"kind": "mesh", "uri": mesh.get("filename"), "scale": scale})
                    n_mesh += 1
                elif box is not None and box.get("size"):
                    spec.update({"kind": "box", "size": [float(v) for v in box.get("size").split()]})
                elif sphere is not None and sphere.get("radius"):
                    spec.update({"kind": "sphere", "radius": float(sphere.get("radius"))})
                elif cylinder is not None:
                    spec.update({
                        "kind": "cylinder",
                        "radius": float(cylinder.get("radius", 0.05)),
                        "length": float(cylinder.get("length", 0.1)),
                    })
                else:
                    continue
                self._robot_visual_specs.append(spec)
        rospy.loginfo(
            f"[suite_viz] visual specs: {len(self._robot_visual_specs)} ({n_mesh} mesh)"
        )

    # =====================================================================
    # State callbacks
    # =====================================================================
    def _gazebo_state_cb(self, msg: ModelStates):
        try:
            idx = msg.name.index(self.robot_name)
        except ValueError:
            rospy.logwarn_throttle(
                5.0, f"[suite_viz] '{self.robot_name}' not in /gazebo/model_states"
            )
            return
        pose = msg.pose[idx]
        self.q[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.q[3:7] = [
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ]
        self._have_base = True

    def _mavros_odom_cb(self, msg: Odometry):
        pose = msg.pose.pose
        self.q[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.q[3:7] = [
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ]
        self._have_base = True

    def _arm_state_sim_cb(self, msg: JointState):
        nj = self.arm_joint_number
        if len(msg.position) >= nj:
            self.q[7 : 7 + nj] = list(msg.position[:nj])

    def _arm_state_cb(self, msg: JointState):
        nj = self.arm_joint_number
        if len(msg.position) >= nj:
            # real-robot ordering convention (matches run_tracking_controller)
            self.q[7 : 7 + nj] = [msg.position[-1], msg.position[-2]][:nj]

    # =====================================================================
    # Publishing
    # =====================================================================
    def _publish_clear_markers(self, stamp) -> None:
        """DELETEALL：去掉其他节点 latched 的旧 robot/EE marker。"""
        for ns, pub in (
            ("suite_mpc_robot", self.robot_markers_pub),
            ("suite_mpc_ee_axes", self.ee_axes_pub),
        ):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.viz_frame_id
            m.ns = ns
            m.id = 0
            m.action = Marker.DELETEALL
            pub.publish(MarkerArray(markers=[m]))

    def _on_timer(self, _event):
        if not self._have_base:
            return
        stamp = rospy.Time.now()
        q = np.asarray(self.q, dtype=float)
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        self._publish_robot_markers(stamp)
        if self.ee_frame_id is not None:
            self._publish_ee_axes(stamp)

    def _publish_robot_markers(self, stamp):
        if not self._robot_visual_specs:
            return
        data = self.robot_data
        arr = MarkerArray()
        for i, spec in enumerate(self._robot_visual_specs):
            oMv = data.oMf[spec["frame_id"]] * spec["origin"]
            quat = pin.Quaternion(oMv.rotation)
            t = oMv.translation
            m = Marker()
            m.header.frame_id = self.viz_frame_id
            m.header.stamp = stamp
            m.ns = "suite_mpc_robot"
            m.id = i
            m.action = Marker.ADD
            m.pose.position.x = float(t[0])
            m.pose.position.y = float(t[1])
            m.pose.position.z = float(t[2])
            m.pose.orientation.x = float(quat.x)
            m.pose.orientation.y = float(quat.y)
            m.pose.orientation.z = float(quat.z)
            m.pose.orientation.w = float(quat.w)
            rgba = spec["rgba"]
            m.color.r, m.color.g, m.color.b, m.color.a = (
                float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]),
            )
            m.lifetime = rospy.Duration(0)
            kind = spec["kind"]
            if kind == "mesh":
                m.type = Marker.MESH_RESOURCE
                m.mesh_resource = spec["uri"]
                m.mesh_use_embedded_materials = spec["uri"].lower().endswith(".dae")
                sc = spec["scale"]
                m.scale.x, m.scale.y, m.scale.z = float(sc[0]), float(sc[1]), float(sc[2])
            elif kind == "box":
                m.type = Marker.CUBE
                sz = spec["size"]
                m.scale.x, m.scale.y, m.scale.z = float(sz[0]), float(sz[1]), float(sz[2])
            elif kind == "sphere":
                m.type = Marker.SPHERE
                d = 2.0 * spec["radius"]
                m.scale.x = m.scale.y = m.scale.z = d
            elif kind == "cylinder":
                m.type = Marker.CYLINDER
                d = 2.0 * spec["radius"]
                m.scale.x = m.scale.y = d
                m.scale.z = spec["length"]
            arr.markers.append(m)
        self.robot_markers_pub.publish(arr)

    def _publish_ee_axes(self, stamp):
        data = self.robot_data
        oMf = data.oMf[self.ee_frame_id]
        origin = oMf.translation
        R = oMf.rotation
        L = self.ee_axis_len
        arr = MarkerArray()
        axis_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for ax in range(3):
            tip = origin + R[:, ax] * L
            m = Marker()
            m.header.frame_id = self.viz_frame_id
            m.header.stamp = stamp
            m.ns = "suite_mpc_ee_axes"
            m.id = ax
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0  # arrow geometry is point-defined; keep pose valid
            p0 = Point(); p0.x, p0.y, p0.z = float(origin[0]), float(origin[1]), float(origin[2])
            p1 = Point(); p1.x, p1.y, p1.z = float(tip[0]), float(tip[1]), float(tip[2])
            m.points = [p0, p1]
            m.scale.x = self.ee_axis_diam
            m.scale.y = self.ee_axis_diam * 2.0
            m.scale.z = self.ee_axis_diam * 2.5
            cr, cg, cb = axis_colors[ax]
            m.color.r, m.color.g, m.color.b, m.color.a = cr, cg, cb, 1.0
            m.lifetime = rospy.Duration(0)
            arr.markers.append(m)
        self.ee_axes_pub.publish(arr)

        try:
            quat = pin.Quaternion(R)
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = self.viz_frame_id
            tf_msg.child_frame_id = "suite_mpc_ee"
            tf_msg.transform.translation.x = float(origin[0])
            tf_msg.transform.translation.y = float(origin[1])
            tf_msg.transform.translation.z = float(origin[2])
            tf_msg.transform.rotation.x = float(quat.x)
            tf_msg.transform.rotation.y = float(quat.y)
            tf_msg.transform.rotation.z = float(quat.z)
            tf_msg.transform.rotation.w = float(quat.w)
            self._tf_broadcaster.sendTransform(tf_msg)
        except Exception as e:
            rospy.logdebug_throttle(5.0, f"[suite_viz] EE TF broadcast failed: {e}")


def main():
    rospy.init_node("suite_rviz_state_node")
    SuiteRvizStateNode()
    rospy.spin()


if __name__ == "__main__":
    main()
