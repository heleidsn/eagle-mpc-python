#!/usr/bin/env python3
"""Gazebo wrench disturbance + RViz marker helpers for ros_tracking GUI."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# Pinocchio / ROS optional at import time (GUI may run without rospy).
try:
    import rospy
    from gazebo_msgs.msg import LinkStates
    from gazebo_msgs.srv import ApplyBodyWrench, BodyRequest, GetLinkState
    from geometry_msgs.msg import Point, Vector3, Wrench
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    ROS_OK = True
except ImportError:
    ROS_OK = False

# Gazebo /groundtruth 将 model_states 直接当作 map；与 suite_rviz 一致。
MARKER_FRAME_ID = "map"

DISTURBANCE_MARKER_TOPIC = "/suite_mpc/disturbance_markers"
DISTURBANCE_CMD_TOPIC = "/suite_mpc/disturbance_cmd"
DISTURBANCE_CONFIG_PARAM = "/suite_mpc/disturbance_config"
BASE_LINK = "base_link"
EE_LINK = "gripper_link"


def scoped_link_name(model_name: str, link_name: str) -> str:
    return f"{model_name}::{link_name}"


def resolve_link(target: str) -> str:
    t = str(target).strip().lower()
    if t in ("ee", "ee_link", "gripper", "gripper_link"):
        return EE_LINK
    return BASE_LINK


def apply_gazebo_wrench(
    model_name: str,
    link_name: str,
    force: Sequence[float],
    torque: Sequence[float],
    *,
    frame: str = "world",
    timeout: float = 3.0,
) -> Tuple[bool, str]:
    """Apply persistent wrench via /gazebo/apply_body_wrench.

    frame: ``world`` — wrench in map/world frame; ``link`` — wrench in link frame.
    """
    if not ROS_OK:
        return False, "rospy/gazebo_msgs not available"
    f = np.asarray(force, dtype=float).reshape(3)
    tq = np.asarray(torque, dtype=float).reshape(3)
    body = scoped_link_name(model_name, link_name)
    # Gazebo: empty/world/map = inertial frame; link frame = scoped link name.
    ref_frame = body if str(frame).lower() == "link" else ""
    try:
        rospy.wait_for_service("/gazebo/apply_body_wrench", timeout=float(timeout))
        svc = rospy.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)
        wrench = Wrench(
            force=Vector3(float(f[0]), float(f[1]), float(f[2])),
            torque=Vector3(float(tq[0]), float(tq[1]), float(tq[2])),
        )
        resp = svc(
            wrench=wrench,
            reference_point=Point(0.0, 0.0, 0.0),
            reference_frame=ref_frame,
            body_name=body,
            start_time=rospy.Time(0),
            # duration < 0: continuous; duration = 0: do nothing (per srv doc).
            duration=rospy.Duration(-1),
        )
        ok = bool(getattr(resp, "success", False))
        msg = str(getattr(resp, "status_message", "OK" if ok else "FAIL"))
        return ok, msg
    except Exception as e:
        return False, str(e)


def clear_gazebo_wrenches(
    model_name: str,
    link_name: str = BASE_LINK,
    *,
    timeout: float = 3.0,
) -> Tuple[bool, str]:
    if not ROS_OK:
        return False, "rospy/gazebo_msgs not available"
    body = scoped_link_name(model_name, link_name)
    try:
        rospy.wait_for_service("/gazebo/clear_body_wrenches", timeout=float(timeout))
        svc = rospy.ServiceProxy("/gazebo/clear_body_wrenches", BodyRequest)
        resp = svc(body_name=body)
        ok = bool(getattr(resp, "success", True))
        msg = str(getattr(resp, "status_message", "OK" if ok else "FAIL"))
        return ok, msg
    except Exception as e:
        return False, str(e)


def _quat_rotate(qx: float, qy: float, qz: float, qw: float, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion (x,y,z,w)."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    # Hamilton product q * [v,0] * q^{-1}
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=float,
    )


def wrench_vectors_world(
    force: Sequence[float],
    torque: Sequence[float],
    quat_xyzw: Sequence[float],
    *,
    frame: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Express force/torque in world frame for RViz arrows."""
    f = np.asarray(force, dtype=float).reshape(3)
    tq = np.asarray(torque, dtype=float).reshape(3)
    if str(frame).lower() == "link" and quat_xyzw is not None:
        q = np.asarray(quat_xyzw, dtype=float).reshape(4)
        f_w = _quat_rotate(q[0], q[1], q[2], q[3], f)
        t_w = _quat_rotate(q[0], q[1], q[2], q[3], tq)
        return f_w, t_w
    return f, tq


def query_link_pose(
    model_name: str,
    link_name: str,
    *,
    timeout: float = 2.0,
) -> Tuple[bool, str, Optional[np.ndarray], Optional[np.ndarray]]:
    """同步查询 link 位姿（world/map 系），不依赖 Subscriber 回调。"""
    if not ROS_OK:
        return False, "rospy/gazebo_msgs not available", None, None
    scoped = scoped_link_name(model_name, link_name)
    try:
        rospy.wait_for_service("/gazebo/get_link_state", timeout=float(timeout))
        svc = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        resp = svc(link_name=scoped, reference_frame="")
        if not bool(getattr(resp, "success", False)):
            msg = str(getattr(resp, "status_message", "get_link_state failed"))
            return False, msg, None, None
        pose = resp.link_state.pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat = np.array(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            dtype=float,
        )
        return True, "OK", pos, quat
    except Exception as e:
        return False, str(e), None, None


def marker_stamp() -> "rospy.Time":
    """RViz + use_sim_time 下用 Time(0) 避免 GUI 墙钟与仿真时钟不一致导致不显示。"""
    return rospy.Time(0)


def find_link_pose(
    link_states: "LinkStates", scoped_name: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    names = list(link_states.name)
    try:
        idx = names.index(scoped_name)
    except ValueError:
        # tolerate short link name match
        idx = None
        for i, n in enumerate(names):
            if n.endswith(f"::{scoped_name.split('::')[-1]}") or n == scoped_name:
                idx = i
                break
        if idx is None:
            return None
    p = link_states.pose[idx].position
    o = link_states.pose[idx].orientation
    pos = np.array([p.x, p.y, p.z], dtype=float)
    quat = np.array([o.x, o.y, o.z, o.w], dtype=float)
    return pos, quat


def _scaled_arrow(
    vec: np.ndarray,
    *,
    meters_per_unit: float,
    min_len: float = 0.08,
    max_len: float = 2.0,
) -> Tuple[Optional[np.ndarray], float]:
    """Return (unit_dir * length, magnitude)."""
    mag = float(np.linalg.norm(vec))
    if mag < 1e-6:
        return None, 0.0
    length = float(np.clip(mag * meters_per_unit, min_len, max_len))
    direction = vec / mag
    return direction * length, mag


def _arrow_marker(
    mid: int,
    start: np.ndarray,
    vec: np.ndarray,
    *,
    color: Tuple[float, float, float, float],
    ns: str,
    frame_id: str,
    magnitude: float,
    marker_lifetime: float,
    shaft_d_base: float = 0.012,
    shaft_d_gain: float = 0.0025,
) -> Optional["Marker"]:
    mag = float(np.linalg.norm(vec))
    if mag < 1e-6:
        return None
    end = start + vec
    shaft_d = float(shaft_d_base + shaft_d_gain * min(magnitude, 40.0))
    head_d = shaft_d * 2.2
    head_len = max(0.06, min(0.2, 0.35 * mag))
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = marker_stamp()
    m.ns = ns
    m.id = int(mid)
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.points = [
        Point(float(start[0]), float(start[1]), float(start[2])),
        Point(float(end[0]), float(end[1]), float(end[2])),
    ]
    m.scale.x = shaft_d
    m.scale.y = head_d
    m.scale.z = head_len
    m.color = ColorRGBA(color[0], color[1], color[2], color[3])
    m.lifetime = rospy.Duration(float(marker_lifetime))
    return m


def _text_marker(
    mid: int,
    pos: np.ndarray,
    text: str,
    *,
    frame_id: str,
    marker_lifetime: float,
    z_offset: float = 0.14,
    char_height: float = 0.08,
) -> "Marker":
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = marker_stamp()
    m.ns = "suite_mpc_dist_label"
    m.id = int(mid)
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2]) + float(z_offset)
    m.pose.orientation.w = 1.0
    m.scale.z = float(char_height)
    m.color = ColorRGBA(1.0, 0.95, 0.85, 0.98)
    m.text = str(text)
    m.lifetime = rospy.Duration(float(marker_lifetime))
    return m


def build_disturbance_markers(
    pos_world: np.ndarray,
    force_world: np.ndarray,
    torque_world: np.ndarray,
    *,
    frame_id: str = MARKER_FRAME_ID,
    marker_lifetime: float = 0.25,
    force_m_per_n: float = 0.08,
    torque_m_per_nm: float = 0.12,
    force_cmd: Optional[Sequence[float]] = None,
    torque_cmd: Optional[Sequence[float]] = None,
    link_name: str = BASE_LINK,
) -> "MarkerArray":
    """Build force (red) and torque (blue) arrows; length ∝ |F|/|τ|."""
    arr = MarkerArray()
    if not ROS_OK:
        return arr

    f_vec, f_mag = _scaled_arrow(
        np.asarray(force_world, dtype=float), meters_per_unit=force_m_per_n
    )
    t_vec, t_mag = _scaled_arrow(
        np.asarray(torque_world, dtype=float), meters_per_unit=torque_m_per_nm
    )

    if f_vec is not None:
        m_f = _arrow_marker(
            0,
            pos_world,
            f_vec,
            color=(1.0, 0.15, 0.1, 0.95),
            ns="suite_mpc_dist_force",
            frame_id=frame_id,
            magnitude=f_mag,
            marker_lifetime=marker_lifetime,
        )
        if m_f is not None:
            arr.markers.append(m_f)

    if t_vec is not None:
        m_t = _arrow_marker(
            1,
            pos_world + np.array([0.0, 0.0, 0.05]),
            t_vec,
            color=(0.2, 0.45, 1.0, 0.9),
            ns="suite_mpc_dist_torque",
            frame_id=frame_id,
            magnitude=t_mag,
            marker_lifetime=marker_lifetime,
            shaft_d_base=0.01,
            shaft_d_gain=0.002,
        )
        if m_t is not None:
            arr.markers.append(m_t)

    if f_mag > 1e-6 or t_mag > 1e-6:
        anchor_r = float(0.035 + 0.004 * min(f_mag, 25.0))
        s = Marker()
        s.header.frame_id = frame_id
        s.header.stamp = marker_stamp()
        s.ns = "suite_mpc_dist_anchor"
        s.id = 2
        s.type = Marker.SPHERE
        s.action = Marker.ADD
        s.pose.position.x = float(pos_world[0])
        s.pose.position.y = float(pos_world[1])
        s.pose.position.z = float(pos_world[2])
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = anchor_r
        s.color = ColorRGBA(1.0, 0.6, 0.0, 0.85)
        s.lifetime = rospy.Duration(float(marker_lifetime))
        arr.markers.append(s)

        # 文字标签：Disturbance + 力/力矩大小
        f_cmd = np.asarray(force_cmd if force_cmd is not None else force_world, dtype=float)
        t_cmd = np.asarray(torque_cmd if torque_cmd is not None else torque_world, dtype=float)
        f_n = float(np.linalg.norm(f_cmd))
        t_nm = float(np.linalg.norm(t_cmd))
        lines = ["Disturbance", f"link: {link_name}"]
        if f_n > 1e-6:
            lines.append(f"F = {f_n:.2f} N")
        if t_nm > 1e-6:
            lines.append(f"τ = {t_nm:.2f} N·m")
        label_h = 0.09 if len(lines) <= 2 else 0.075
        arr.markers.append(
            _text_marker(
                3,
                pos_world,
                "\n".join(lines),
                frame_id=frame_id,
                marker_lifetime=marker_lifetime,
                z_offset=0.12 + 0.04 * min(f_n, 20.0),
                char_height=label_h,
            )
        )

    return arr


def markers_for_disturbance_cfg(
    cfg: dict,
    *,
    marker_lifetime: float = 0.25,
) -> Tuple[bool, str, "MarkerArray"]:
    """按扰动配置查询位姿并生成 RViz MarkerArray。"""
    ok, msg, pos, quat = query_link_pose(cfg["model"], cfg["link"])
    if not ok or pos is None:
        return False, msg, MarkerArray()
    f_w, t_w = wrench_vectors_world(
        cfg["force"], cfg["torque"], quat, frame=cfg.get("frame", "world")
    )
    arr = build_disturbance_markers(
        pos,
        f_w,
        t_w,
        frame_id=MARKER_FRAME_ID,
        marker_lifetime=marker_lifetime,
        force_cmd=cfg.get("force"),
        torque_cmd=cfg.get("torque"),
        link_name=str(cfg.get("link", BASE_LINK)),
    )
    if not arr.markers:
        return False, "zero wrench (no markers)", arr
    return True, "OK", arr


def delete_disturbance_markers(frame_id: str = MARKER_FRAME_ID) -> "MarkerArray":
    arr = MarkerArray()
    if not ROS_OK:
        return arr
    for ns in (
        "suite_mpc_dist_force",
        "suite_mpc_dist_torque",
        "suite_mpc_dist_anchor",
        "suite_mpc_dist_label",
    ):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = marker_stamp()
        m.ns = ns
        m.action = Marker.DELETEALL
        arr.markers.append(m)
    return arr
