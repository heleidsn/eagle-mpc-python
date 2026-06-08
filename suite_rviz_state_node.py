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
  ~rotor_spin_enable          (bool)  enable propeller spin animation
  ~rotor_follow_actuator      (bool)  drive spin rate from PX4 throttle to match Gazebo
  ~rotor_max_rot_velocity     (float) motor maxRotVelocity from SDF (default 1100)
  ~rotor_velocity_slowdown_sim(float) rotorVelocitySlowdownSim from SDF (default 10)
  ~rotor_actuator_topic       (str)   fallback ActuatorControl topic (default /mavros/target_actuator_control)
  ~rotor_throttle_index       (int)   collective throttle channel index in ActuatorControl (PX4: 3)
  ~rotor_spin_rate            (float) fallback fixed spin rate (rad/s) when no actuator data
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

try:
    from mavros_msgs.msg import ActuatorControl, AttitudeTarget, State
    _HAVE_MAVROS_MSGS = True
except Exception:  # pragma: no cover - mavros_msgs 不可用时降级为定速
    ActuatorControl = None
    AttitudeTarget = None
    State = None
    _HAVE_MAVROS_MSGS = False


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
        # 桨叶旋转（视觉效果）：绕旋翼 link 局部 z 轴旋转。
        #
        # 默认让 RViz 桨叶转速 **与 Gazebo 一致**：Gazebo 的 gazebo_motor_model 把
        # 可视关节按  ref_motor_rot_vel / rotorVelocitySlowdownSim  旋转（即真实电机
        # 转速 ÷ slowdown），而 ref_motor_rot_vel ≈ 油门 · maxRotVelocity。所以这里
        # 取 PX4 的归一化油门 [0,1]，经同一套电机模型换算得到与 Gazebo 一致的转速：
        #     spin_rate = throttle · rotor_max_rot_velocity / rotor_velocity_slowdown_sim
        # 这些默认值取自 s500_uam SDF（maxRotVelocity=1100, slowdown=10）。
        #
        # 油门来源（取最新到达者，按可靠性多路订阅）：
        #   1) /mavros/setpoint_raw/attitude        (AttitudeTarget.thrust，本控制器上行)
        #   2) /mavros/setpoint_raw/target_attitude (AttitudeTarget.thrust，PX4 下行，几乎各模式都有)
        #   3) /mavros/target_actuator_control      (ActuatorControl，mavros 默认常不发)
        #
        # 拿不到 actuator 数据（或 rotor_follow_actuator=False）时退回定速 rotor_spin_rate。
        # 注意混叠：两叶桨视觉周期 π，每帧转角须 < 90° 才不“倒转”，即
        # spin_rate < π/2 · publish_hz（20Hz 时约 31 rad/s），超过会看起来变慢/倒转，
        # 这与 Gazebo 在高帧率下的表现差异由渲染帧率决定，属正常现象。
        self.rotor_spin_enable = bool(rospy.get_param("~rotor_spin_enable", True))
        self.rotor_spin_rate = float(rospy.get_param("~rotor_spin_rate", 25.0))  # rad/s 回退定速
        self.rotor_link_prefix = str(rospy.get_param("~rotor_link_prefix", "rotor"))
        # 与 Gazebo 一致的电机模型参数（来自 SDF）
        self.rotor_follow_actuator = bool(
            rospy.get_param("~rotor_follow_actuator", True)
        )
        self.rotor_max_rot_velocity = float(
            rospy.get_param("~rotor_max_rot_velocity", 1100.0)
        )
        self.rotor_velocity_slowdown_sim = float(
            rospy.get_param("~rotor_velocity_slowdown_sim", 10.0)
        )
        self.rotor_actuator_topic = str(
            rospy.get_param("~rotor_actuator_topic", "/mavros/target_actuator_control")
        )
        # actuator group-0 中油门（collective）所在通道，PX4 多旋翼为 index 3
        self.rotor_throttle_index = int(
            rospy.get_param("~rotor_throttle_index", 3)
        )
        self._spin_angle = 0.0
        self._spin_last_t = None
        self._throttle = None       # 最近一次归一化油门 [0,1]
        self._throttle_stamp = 0.0  # 接收时刻（s），用于超时回退
        self._armed = None          # /mavros/state.armed；None=未知（无 mavros）

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
        # 订阅 PX4 归一化油门以匹配 Gazebo 桨叶转速（多路，取最新到达者）
        if (
            self.rotor_spin_enable
            and self.rotor_follow_actuator
            and _HAVE_MAVROS_MSGS
        ):
            rospy.Subscriber(
                "/mavros/setpoint_raw/attitude", AttitudeTarget,
                self._att_thrust_cb, queue_size=10,
            )
            rospy.Subscriber(
                "/mavros/setpoint_raw/target_attitude", AttitudeTarget,
                self._att_thrust_cb, queue_size=10,
            )
            rospy.Subscriber(
                self.rotor_actuator_topic, ActuatorControl,
                self._actuator_cb, queue_size=10,
            )
            # 解锁状态门控：未解锁时桨叶停转
            rospy.Subscriber(
                "/mavros/state", State, self._state_cb, queue_size=10,
            )
        elif self.rotor_follow_actuator and not _HAVE_MAVROS_MSGS:
            rospy.logwarn(
                "[suite_viz] mavros_msgs 不可用，桨叶转速回退为定速 "
                f"{self.rotor_spin_rate:g} rad/s"
            )

        # 清除 tracking node 留下的 latched marker（否则 Kill node 后 RViz 会卡在旧位姿）
        self._publish_clear_markers(rospy.Time.now())

        hz = max(self.publish_hz, 1.0)
        self._timer = rospy.Timer(rospy.Duration(1.0 / hz), self._on_timer)
        n_spin = sum(1 for s in self._robot_visual_specs if s.get("spin_dir", 0) != 0)
        if self.rotor_follow_actuator and _HAVE_MAVROS_MSGS:
            hover = 0.5 * self.rotor_max_rot_velocity / max(
                self.rotor_velocity_slowdown_sim, 1e-6
            )
            spin_desc = (
                f"gazebo-match(thr·{self.rotor_max_rot_velocity:g}/"
                f"{self.rotor_velocity_slowdown_sim:g}, ~{hover:g}rad/s@0.5thr)"
            )
        else:
            spin_desc = f"fixed@{self.rotor_spin_rate:g}rad/s"
        rospy.loginfo(
            f"[suite_viz] up: robot='{self.robot_name}', nq={self.nq}, "
            f"arm={self.arm_enabled}, odom={self.odom_source}, "
            f"specs={len(self._robot_visual_specs)}, "
            f"rotors={n_spin} spin={'on' if (self.rotor_spin_enable and n_spin) else 'off'} "
            f"{spin_desc}"
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
                # 桨叶旋转方向：link 名以 rotor 前缀开头者参与旋转；ccw/cw 反向旋转。
                spin_dir = 0
                if link_name.startswith(self.rotor_link_prefix):
                    spin_dir = 1
                spec["link_name"] = link_name
                mesh = geom.find("mesh")
                box = geom.find("box")
                sphere = geom.find("sphere")
                cylinder = geom.find("cylinder")
                if mesh is not None and mesh.get("filename"):
                    scale = [1.0, 1.0, 1.0]
                    if mesh.get("scale"):
                        scale = [float(v) for v in mesh.get("scale").split()]
                    uri = mesh.get("filename")
                    spec.update({"kind": "mesh", "uri": uri, "scale": scale})
                    if spin_dir != 0:
                        ul = uri.lower()
                        if "ccw" in ul:
                            spin_dir = 1
                        elif "cw" in ul:
                            spin_dir = -1
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
                spec["spin_dir"] = int(spin_dir)
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

    def _actuator_cb(self, msg: "ActuatorControl"):
        """PX4 group-0 actuator control：取 collective 油门通道（默认 index 3）。
        ACTUATOR_CONTROL_TARGET 中该值为归一化油门，多旋翼范围约 [0,1]。"""
        idx = self.rotor_throttle_index
        if 0 <= idx < len(msg.controls):
            self._set_throttle(msg.controls[idx])

    def _att_thrust_cb(self, msg: "AttitudeTarget"):
        """AttitudeTarget.thrust 已是归一化油门 [0,1]（上行或 PX4 下行均可）。"""
        self._set_throttle(msg.thrust)

    def _set_throttle(self, value) -> None:
        self._throttle = float(np.clip(value, 0.0, 1.0))
        self._throttle_stamp = rospy.Time.now().to_sec()

    def _state_cb(self, msg: "State"):
        self._armed = bool(msg.armed)

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

    def _current_spin_rate(self, now_t: float) -> float:
        """返回当前桨叶可视转速 (rad/s)。

        跟随 actuator 时复用 Gazebo gazebo_motor_model 的换算：
            spin_rate = throttle · maxRotVelocity / rotorVelocitySlowdownSim
        与 Gazebo 中可视关节速度 ref_motor_rot_vel / slowdown 一致。

        门控（跟随模式下）：
          * 未解锁（/mavros/state.armed=False）→ 0，桨叶停转；
          * 没有最新油门（>0.5s 超时，例如 PX4 还没起 / 已锁定）→ 0；
        只有当 rotor_follow_actuator=False（显式关闭跟随）时才用定速
        rotor_spin_rate，避免未解锁时桨叶仍旋转。
        """
        if self.rotor_follow_actuator:
            if self._armed is False:
                return 0.0
            if (
                self._throttle is not None
                and (now_t - self._throttle_stamp) < 0.5
            ):
                slowdown = max(self.rotor_velocity_slowdown_sim, 1e-6)
                return self._throttle * self.rotor_max_rot_velocity / slowdown
            return 0.0
        return self.rotor_spin_rate

    def _on_timer(self, _event):
        if not self._have_base:
            return
        stamp = rospy.Time.now()
        # 推进桨叶视觉旋转角（按真实经过时间）。转速优先取与 Gazebo 一致的
        # 油门换算值，拿不到 actuator 数据时回退为定速 rotor_spin_rate。
        if self.rotor_spin_enable:
            now_t = stamp.to_sec()
            rate = self._current_spin_rate(now_t)
            if rate != 0.0:
                if self._spin_last_t is None:
                    self._spin_last_t = now_t
                dt = now_t - self._spin_last_t
                self._spin_last_t = now_t
                if 0.0 < dt < 1.0:
                    self._spin_angle = (self._spin_angle + rate * dt) % (2.0 * np.pi)
            else:
                self._spin_last_t = now_t
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
            oMf_i = data.oMf[spec["frame_id"]]
            spin_dir = spec.get("spin_dir", 0)
            if self.rotor_spin_enable and spin_dir != 0:
                ang = self._spin_angle * float(spin_dir)
                ca, sa = np.cos(ang), np.sin(ang)
                Rz = np.array(
                    [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float
                )
                spin = pin.SE3(Rz, np.zeros(3))
                oMv = oMf_i * spin * spec["origin"]
            else:
                oMv = oMf_i * spec["origin"]
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
