#!/usr/bin/env python3
"""
ROS 空中操作臂闭环跟踪控制节点 — run_tracking_controller.py

与 run_controller.py 的区别
───────────────────────────
| run_controller.py           | run_tracking_controller.py            |
|-----------------------------|---------------------------------------|
| eagle_mpc.RailMpc (C++ lib) | UAMCrocoddylStateTrackingMPC /        |
|                             | UAMEEPoseTrackingCrocoddylMPC (纯 Py) |
| 固定 RailMpc 代价结构        | 可调权重 w_state_track / w_ee_pos 等   |
| 轨迹参考来自 YAML 求解       | 轨迹来自 GUI 规划导出 npz 或 YAML     |
| 与数值仿真共用同一类          | 专为 ROS 实时控制设计，无仿真 plant   |

控制器模式（~controller_mode）
──────────────────────────────
croc_full_state  : Crocoddyl 全状态跟踪 (build_shooting_problem_along_plan)
croc_ee_pose     : Crocoddyl EE 位姿跟踪 (build_shooting_problem_along_ee_ref)
px4              : 发送 PositionTarget（位置/速度/加速度/yaw）给 PX4 内部控制器
geometric        : 节点内置 geometric controller（直接输出 body_rate + thrust）

参考轨迹来源（~trajectory_source）
──────────────────────────────────
suite_npz        : 从 GUI 导出的 last_suite_plan.npz 加载 t_plan / x_plan
yaml             : 使用 eagle_mpc_debugger temp_trajectory.yaml 求解结果

发布/订阅接口（与 run_controller.py 保持一致）
──────────────────────────────────────────────
订阅: /gazebo/model_states 或 /mavros/local_position/odom (base 状态)
      /arm_controller/joint_states 或 /joint_states           (机械臂)
      /mavros/state                                           (PX4 模式)
发布: /mavros/setpoint_raw/attitude  (croc_*: body_rate + thrust → PX4)
      /mavros/setpoint_raw/local     (px4: PositionTarget)
      /desired_joint_states          (机械臂关节指令)
      /mpc/state                     (调试状态)

使用示例
────────
# Gazebo 仿真（从 GUI npz 加载轨迹，全状态 Crocoddyl 跟踪）
rosrun eagle_mpc_debugger run_tracking_controller.py \\
    _trajectory_source:=suite_npz \\
    _suite_plan_path:=/path/to/last_suite_plan.npz \\
    _controller_mode:=croc_full_state \\
    _odom_source:=gazebo

# 实机（从 YAML 加载，EE 位姿跟踪）
rosrun eagle_mpc_debugger run_tracking_controller.py \\
    _controller_mode:=croc_ee_pose \\
    _odom_source:=mavros \\
    _dt_mpc:=0.05 \\
    _horizon:=25

启动跟踪（OFFBOARD 且已解锁后）
───────────────────────────────
rosservice call /start_tracking  # 开始跟踪
rosservice call /stop_tracking   # 停止跟踪（悬停）
rosservice call /save_data       # 保存录制数据
"""

from __future__ import annotations

import os
import sys
import time
import math
import csv
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── ROS ──────────────────────────────────────────────────────────────────────
import rospy
from nav_msgs.msg import Odometry, Path as RosPath
from geometry_msgs.msg import PoseStamped, Vector3, Pose, Point, Quaternion, Twist, Vector3 as GeoVec3
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray, Header
from mavros_msgs.msg import State, AttitudeTarget, PositionTarget
from mavros_msgs.srv import SetMode, SetModeRequest
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, GetModelStateRequest
from std_srvs.srv import Trigger, TriggerResponse
from eagle_mpc_msgs.msg import MpcState

# ── visualization ─────────────────────────────────────────────────────────────
try:
    from eagle_mpc_viz import WholeBodyStatePublisher, WholeBodyTrajectoryPublisher
    _WHOLEBODY_VIZ_OK = True
except ImportError:
    _WHOLEBODY_VIZ_OK = False
    WholeBodyStatePublisher = None
    WholeBodyTrajectoryPublisher = None

# tf utilities
from tf.transformations import quaternion_matrix, euler_from_quaternion

# ── scripts/ (Crocoddyl MPC classes) ─────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import pinocchio as pin
import crocoddyl

from s500_uam_crocoddyl_state_tracking_mpc import (
    UAMCrocoddylStateTrackingMPC,
    UAMEEPoseTrackingCrocoddylMPC,
    EETrackingWeights,
    interp_full_state_piecewise,
    interp_ref_pose,
    default_hover_nominal,
)
from s500_uam_trajectory_planner import (
    S500UAMTrajectoryPlanner,
    compute_ee_kinematics_along_trajectory,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_suite_npz(path: str) -> Dict[str, Any]:
    """Load plan exported by run_ros_tracking.export_suite_plan_npz."""
    data = np.load(path, allow_pickle=True)
    t_plan = np.asarray(data["t_plan"], dtype=float).ravel()
    x_plan = np.asarray(data["x_plan"], dtype=float)
    dt_ms = int(np.asarray(data.get("dt_traj_opt_ms", 50)).item())
    kind = "unknown"
    if "kind" in data.files:
        try:
            kind = str(np.asarray(data["kind"]).item())
        except Exception:
            kind = "unknown"
    velocity_frame = "unknown"
    if "velocity_frame" in data.files:
        try:
            velocity_frame = str(np.asarray(data["velocity_frame"]).item()).strip().lower() or "unknown"
        except Exception:
            velocity_frame = "unknown"
    u_plan = None
    if "u_plan" in data.files:
        u = np.asarray(data["u_plan"], dtype=float)
        if u.ndim == 2 and u.shape[0] > 0:
            u_plan = u
    ddp_plan = None
    if "ddp_plan" in data.files:
        ddp = np.asarray(data["ddp_plan"], dtype=float)
        if ddp.ndim == 2 and ddp.shape[0] == t_plan.shape[0] and ddp.shape[1] == 3:
            ddp_plan = ddp
    return {
        "kind": kind,
        "velocity_frame": velocity_frame,
        "t_plan": t_plan,
        "x_plan": x_plan,
        "dt_traj_opt_ms": dt_ms,
        "u_plan": u_plan,
        "ddp_plan": ddp_plan,
    }


def _load_yaml_trajectory(dt_traj_opt_ms: int, use_squash: bool = True):
    """Load + solve trajectory from eagle_mpc_debugger temp_trajectory.yaml."""
    try:
        import rospkg
        import yaml
        import eagle_mpc

        rospack = rospkg.RosPack()
        pkg = rospack.get_path("eagle_mpc_debugger")
        yaml_path = os.path.join(pkg, "config/yaml/trajectories/temp_trajectory.yaml")

        traj = eagle_mpc.Trajectory()
        traj.autoSetup(yaml_path)
        problem = traj.createProblem(dt_traj_opt_ms, use_squash, "IntegratedActionModelEuler")

        if use_squash:
            solver = eagle_mpc.SolverSbFDDP(problem, traj.squash)
        else:
            solver = crocoddyl.SolverBoxFDDP(problem)

        solver.convergence_init = 1e-6
        solver.solve([], [], 100)

        xs = [np.array(x, dtype=float).ravel() for x in solver.xs]
        n = len(xs)
        t_plan = np.arange(n, dtype=float) * (dt_traj_opt_ms / 1000.0)
        x_plan = np.vstack(xs)
        us = [np.array(u, dtype=float).ravel() for u in solver.us]
        u_plan = np.vstack(us) if us else None
        rospy.loginfo(f"Loaded YAML trajectory: {n} states, T={t_plan[-1]:.2f}s")
        return {"t_plan": t_plan, "x_plan": x_plan, "dt_traj_opt_ms": dt_traj_opt_ms, "u_plan": u_plan}
    except Exception as e:
        rospy.logerr(f"Failed to load YAML trajectory: {e}")
        raise


def _compute_ee_vel_refs(
    t_ref: np.ndarray, p_ref: np.ndarray, yaw_ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference EE reference velocity arrays (dp/dt, dyaw/dt)."""
    n = len(t_ref)
    dp = np.zeros((n, 3), dtype=float)
    dyaw = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        dt = t_ref[i + 1] - t_ref[i - 1]
        if dt > 1e-10:
            dp[i] = (p_ref[i + 1] - p_ref[i - 1]) / dt
            dyaw[i] = (yaw_ref[i + 1] - yaw_ref[i - 1]) / dt
    if n >= 2:
        dt0 = t_ref[1] - t_ref[0]
        if dt0 > 1e-10:
            dp[0] = (p_ref[1] - p_ref[0]) / dt0
            dyaw[0] = (yaw_ref[1] - yaw_ref[0]) / dt0
        dtn = t_ref[-1] - t_ref[-2]
        if dtn > 1e-10:
            dp[-1] = (p_ref[-1] - p_ref[-2]) / dtn
            dyaw[-1] = (yaw_ref[-1] - yaw_ref[-2]) / dtn
    return dp, dyaw


# ─────────────────────────────────────────────────────────────────────────────
# Main controller class
# ─────────────────────────────────────────────────────────────────────────────

class SuiteTrackingController:
    """
    ROS 节点：使用 scripts/ 下的 Crocoddyl Python MPC 对 aerial manipulator
    进行在线闭环跟踪控制。

    状态来自 ROS 话题（Gazebo 或 MAVROS），控制指令发布到 MAVROS。
    MPC 每个控制周期重新构建 shooting problem 并求解（ZOH 持续到下一个求解完成）。
    """

    def __init__(self):
        rospy.init_node("suite_tracking_controller", anonymous=False, log_level=rospy.INFO)
        np.set_printoptions(precision=4, suppress=True)

        # ── 基础参数 ───────────────────────────────────────────────────────────
        self.robot_name = rospy.get_param("~robot_name", "s500_uam")
        self.controller_mode = rospy.get_param("~controller_mode", "croc_full_state")
        self.trajectory_source = rospy.get_param("~trajectory_source", "suite_npz")
        self.trajectory_name = str(rospy.get_param("~trajectory_name", "trajectory"))
        self.odom_source = rospy.get_param("~odom_source", "gazebo")
        self.use_simulation = rospy.get_param("~use_simulation", True)
        self.arm_enabled = rospy.get_param("~arm_enabled", True)
        self.arm_control_mode = rospy.get_param("~arm_control_mode", "position")
        if str(self.robot_name).strip().lower() == "s500":
            self.arm_enabled = False

        # 控制速率
        self.control_rate = rospy.get_param("~control_rate", 50.0)  # Hz
        self.dt_control = 1.0 / self.control_rate

        # MPC 参数
        self.dt_mpc = rospy.get_param("~dt_mpc", 0.05)       # s
        self.horizon = rospy.get_param("~horizon", 25)
        self.mpc_max_iter = rospy.get_param("~mpc_max_iter", 60)

        # 全状态跟踪权重
        self.w_state_track = rospy.get_param("~w_state_track", 10.0)
        self.w_state_reg = rospy.get_param("~w_state_reg", 0.1)
        self.w_control = rospy.get_param("~w_control", 1e-3)
        self.w_terminal_track = rospy.get_param("~w_terminal_track", 100.0)
        self.w_pos = rospy.get_param("~w_pos", 1.0)
        self.w_att = rospy.get_param("~w_att", 1.0)
        self.w_joint = rospy.get_param("~w_joint", 1.0)
        self.w_vel = rospy.get_param("~w_vel", 1.0)
        self.w_omega = rospy.get_param("~w_omega", 1.0)
        self.w_joint_vel = rospy.get_param("~w_joint_vel", 1.0)
        self.w_u_thrust = rospy.get_param("~w_u_thrust", 1.0)
        self.w_u_joint_torque = rospy.get_param("~w_u_joint_torque", 1.0)

        # EE 位姿跟踪权重
        self.ee_w_pos = rospy.get_param("~ee_w_pos", 10.0)
        self.ee_w_rot_rp = rospy.get_param("~ee_w_rot_rp", 1.0)
        self.ee_w_rot_yaw = rospy.get_param("~ee_w_rot_yaw", 1.0)
        self.ee_w_vel_lin = rospy.get_param("~ee_w_vel_lin", 1.0)
        self.ee_w_vel_ang_rp = rospy.get_param("~ee_w_vel_ang_rp", 0.5)
        self.ee_w_vel_ang_yaw = rospy.get_param("~ee_w_vel_ang_yaw", 0.5)
        self.ee_w_u = rospy.get_param("~ee_w_u", 1e-3)
        self.ee_w_terminal = rospy.get_param("~ee_w_terminal", 3.0)

        # PX4 指令限制
        self.max_thrust_total = rospy.get_param("~max_thrust", 7.43 * 4)  # N
        self.max_angular_velocity = rospy.get_param("~max_angular_velocity", math.radians(120))
        self.min_thrust_cmd = rospy.get_param("~min_thrust_cmd", 0.0)
        self.max_thrust_cmd = rospy.get_param("~max_thrust_cmd", 1.0)
        # 内置 geometric controller 增益
        self.geo_kp_pos = float(rospy.get_param("~geo_kp_pos", 4.0))
        self.geo_kd_vel = float(rospy.get_param("~geo_kd_vel", 2.5))
        self.geo_kR = float(rospy.get_param("~geo_kR", 4.0))
        self.geo_kOmega = float(rospy.get_param("~geo_kOmega", 0.35))
        self.geo_max_tilt_deg = float(rospy.get_param("~geo_max_tilt_deg", 35.0))

        # 轨迹来源参数
        self.suite_plan_path = rospy.get_param("~suite_plan_path", "")
        self.dt_traj_opt_ms = rospy.get_param("~dt_traj_opt_ms", 50)

        # ── 运行状态标志 ───────────────────────────────────────────────────────
        self.trajectory_started = False
        self.traj_finished = False
        self.controller_start_time = None

        self.px4_state = State()
        self.arm_state = JointState()

        # ── 加载轨迹 & 构建 MPC ───────────────────────────────────────────────
        self._traj_data: Dict[str, Any] = {}
        self.t_plan: Optional[np.ndarray] = None
        self.x_plan: Optional[np.ndarray] = None
        self.ddp_plan: Optional[np.ndarray] = None
        self.t_ref_ee: Optional[np.ndarray] = None
        self.p_ref_ee: Optional[np.ndarray] = None
        self.yaw_ref_ee: Optional[np.ndarray] = None
        self.dp_ref_ee: Optional[np.ndarray] = None
        self.dyaw_ref_ee: Optional[np.ndarray] = None

        # ── RViz 路径可视化参数 ────────────────────────────────────────────────
        self.viz_path_publish_hz     = float(rospy.get_param("~viz_path_publish_hz", 15.0))
        self.viz_path_min_pos_step   = float(rospy.get_param("~viz_path_min_position_step", 0.0))
        self.uav_actual_path_max_len = int(rospy.get_param("~uav_actual_path_max_length", 3000))
        self.ee_actual_path_max_len  = int(rospy.get_param("~ee_actual_path_max_length",  3000))
        _viz_hz = max(self.viz_path_publish_hz, 0.5)
        self._viz_path_min_period    = 1.0 / _viz_hz
        self._last_viz_pub_wall_t    = 0.0

        # 缓存固定参考路径（轨迹加载后构建一次，必须在 _load_trajectory 之前声明）
        self._cached_ref_path:     Optional[RosPath] = None
        self._cached_ee_plan_path: Optional[RosPath] = None
        # 每次 MPC 求解后暂存 horizon 路径（staging → throttled publish）
        self._staged_mpc_path:     Optional[RosPath] = None
        # 累积实际路径缓冲
        self._uav_actual_path_msg: RosPath = RosPath()
        self._uav_actual_path_msg.header.frame_id = "map"
        self._ee_actual_path_msg:  RosPath = RosPath()
        self._ee_actual_path_msg.header.frame_id  = "map"
        self._last_uav_path_pos_sample: Optional[np.ndarray] = None
        self._last_ee_path_pos_sample:  Optional[np.ndarray] = None

        self._load_trajectory()
        self._build_mpc()
        # 轨迹和 MPC 均就绪后构建 RViz 可视化缓存（EE FK 需要 self.mpc）
        self._rebuild_cached_viz_paths()

        # ── 初始状态（从规划起点） ────────────────────────────────────────────
        self.state = self._match_state_dim(np.asarray(self.x_plan[0], dtype=float).copy())
        self.arm_joint_number = self.mpc.robot_model.nq - 7  # nq = 7(base) + n_arm

        # WarmStart 缓存（shift-warm-start）
        self._xs_guess: Optional[List[np.ndarray]] = None
        self._us_guess: Optional[List[np.ndarray]] = None
        self._u_hold = self._hover_thrust_cmd()

        # ── Regulation 模式（MPC 镇定到用户设定目标） ─────────────────────────
        # 节点启动时默认处于 regulation 模式，目标 = x_plan[0]
        # 调用 /start_tracking 才切换到轨迹跟踪；/stop_tracking 切回 regulation
        self._regulating: bool = True
        self._reg_target: np.ndarray = self._match_state_dim(
            np.asarray(self.x_plan[0], dtype=float).copy(), zero_vel=True
        )
        # False: use default start/end target policy; True: keep user-set regulation target.
        self._reg_target_locked: bool = False
        self._reg_xs_guess: Optional[List[np.ndarray]] = None
        self._reg_us_guess: Optional[List[np.ndarray]] = None

        # ── 录制数据 ─────────────────────────────────────────────────────────
        self.recording_enabled = False
        self.recorded_data: Dict[str, list] = {
            "time": [], "position": [], "velocity": [], "orientation": [],
            "angular_velocity": [],
            "arm_joint_positions": [], "arm_joint_velocities": [],
            "mpc_control": [], "mpc_cost": [], "mpc_solve_time": [],
            "body_rate_commands": [], "thrust_command": [],
            "arm_joint_commands": [], "reference_position": [],
            "reference_orientation": [], "reference_velocity": [],
            "reference_angular_velocity": [], "reference_arm_positions": [],
        }
        self._active_tracking_tag: Optional[str] = None
        self.solve_times: deque = deque(maxlen=200)

        # ── ROS 话题 ─────────────────────────────────────────────────────────
        self._init_subscribers()
        self._init_publishers()
        self._init_services()

        # ── Gazebo set_model_state 服务代理（仅 use_simulation 时初始化） ─────
        self._gazebo_set_state: Optional[rospy.ServiceProxy] = None
        if self.use_simulation:
            try:
                rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)
                self._gazebo_set_state = rospy.ServiceProxy(
                    "/gazebo/set_model_state", SetModelState
                )
                rospy.loginfo("Connected to /gazebo/set_model_state")
            except Exception as e:
                rospy.logwarn(f"/gazebo/set_model_state unavailable: {e}")

        # ── 控制定时器 ───────────────────────────────────────────────────────
        self._thread_lock = threading.Lock()
        self.timer = rospy.Timer(rospy.Duration(self.dt_control), self._control_callback)

        rospy.loginfo(
            f"SuiteTrackingController ready | mode={self.controller_mode} "
            f"| traj={self.trajectory_source} | rate={self.control_rate} Hz "
            f"| dt_mpc={self.dt_mpc}s H={self.horizon} | arm={self.arm_enabled}"
        )

    # =========================================================================
    # 初始化：轨迹 / MPC
    # =========================================================================

    def _load_trajectory(self):
        """从 suite npz 或 YAML 加载参考轨迹。"""
        if self.trajectory_source == "suite_npz":
            if not self.suite_plan_path:
                raise ValueError("~suite_plan_path 不能为空（~trajectory_source:=suite_npz）")
            path = os.path.abspath(os.path.expanduser(self.suite_plan_path))
            rospy.loginfo(f"Loading suite npz: {path}")
            self._traj_data = _load_suite_npz(path)
        else:
            rospy.loginfo("Loading trajectory from YAML (temp_trajectory.yaml)")
            self._traj_data = _load_yaml_trajectory(self.dt_traj_opt_ms)

        self.t_plan = self._traj_data["t_plan"]
        self.x_plan = self._traj_data["x_plan"]
        self.ddp_plan = self._traj_data.get("ddp_plan")
        traj_kind = str(self._traj_data.get("kind", "unknown"))
        vel_frame = str(self._traj_data.get("velocity_frame", "unknown"))
        rospy.loginfo(
            f"Suite plan metadata: kind={traj_kind}, velocity_frame={vel_frame}. "
            "full_croc/full_acados body-frame velocities are used directly (no extra conversion)."
        )

        # EE 参考（用于 croc_ee_pose 模式，从 FK 建立）
        if self.controller_mode == "croc_ee_pose":
            self._build_ee_ref_from_full_state()

        rospy.loginfo(
            f"Trajectory loaded: {len(self.t_plan)} states, T={self.t_plan[-1]:.2f}s"
        )
        # 注意：_rebuild_cached_viz_paths() 需要 self.mpc，
        # 因此放到 _build_mpc() 之后在 __init__ 中调用，此处不再调用。

    def _build_ee_ref_from_full_state(self):
        """从全状态轨迹 FK 计算 EE 参考（位置 + yaw），用于 EE 位姿跟踪模式。"""
        s500_yaml_path, urdf_path = self._model_paths_for_robot()
        planner = S500UAMTrajectoryPlanner(
            s500_yaml_path=s500_yaml_path,
            urdf_path=urdf_path,
        )
        if planner.ee_frame_id is None:
            raise ValueError("Current robot model has no EE frame 'gripper_link'.")
        rm = planner.robot_model
        data = rm.createData()
        eid = planner.ee_frame_id

        ee_pos, _, ee_rpy, _ = compute_ee_kinematics_along_trajectory(
            self.x_plan, rm, data, eid
        )
        self.t_ref_ee = self.t_plan.copy()
        self.p_ref_ee = ee_pos.copy()
        self.yaw_ref_ee = ee_rpy[:, 2].copy()
        self.dp_ref_ee, self.dyaw_ref_ee = _compute_ee_vel_refs(
            self.t_ref_ee, self.p_ref_ee, self.yaw_ref_ee
        )
        rospy.loginfo("EE reference built from full-state FK.")

    def _model_paths_for_robot(self) -> Tuple[str, str]:
        root = Path(__file__).resolve().parent
        s500_yaml = str(root / "config" / "yaml" / "multicopter" / "s500.yaml")
        robot = str(self.robot_name).strip().lower()
        if robot == "s500":
            urdf = str(root / "models" / "urdf" / "s500_simple.urdf")
        else:
            urdf = str(root / "models" / "urdf" / "s500_uam_simple.urdf")
        return s500_yaml, urdf

    def _compute_x_nominal(self, mpc_obj) -> np.ndarray:
        """
        Build a dimension-safe nominal state for MPC regularization.
        Priority: x_plan[0] (if compatible) -> default_hover_nominal() (if compatible) -> neutral+zero-vel.
        """
        expected_dim = int(mpc_obj.nq + mpc_obj.nv)

        if self.x_plan is not None and len(self.x_plan) > 0:
            x0 = np.asarray(self.x_plan[0], dtype=float).reshape(-1)
            if x0.size == expected_dim:
                return x0.copy()

        x_def = np.asarray(default_hover_nominal(), dtype=float).reshape(-1)
        if x_def.size == expected_dim:
            return x_def.copy()

        # Fallback for model mismatch (e.g., s500 13D vs s500_uam 17D)
        x_nom = np.zeros(expected_dim, dtype=float)
        try:
            x_nom[: mpc_obj.nq] = pin.neutral(mpc_obj.robot_model)
        except Exception:
            pass
        if expected_dim >= 3:
            x_nom[2] = 1.0
        rospy.logwarn(
            f"Nominal state dimension mismatch: using neutral fallback ({expected_dim}D)."
        )
        return x_nom

    def _match_state_dim(self, x: np.ndarray, zero_vel: bool = False) -> np.ndarray:
        """Resize/crop a state vector to current MPC model dimension (nq+nv)."""
        x_in = np.asarray(x, dtype=float).reshape(-1)
        nq = int(self.mpc.nq)
        nx = int(self.mpc.nq + self.mpc.nv)
        if x_in.size == nx:
            out = x_in.copy()
        else:
            out = np.zeros(nx, dtype=float)
            if x_in.size > 0:
                out[: min(nx, x_in.size)] = x_in[: min(nx, x_in.size)]
            # Prefer a valid neutral configuration when input dimension mismatches.
            try:
                q0 = np.asarray(pin.neutral(self.mpc.robot_model), dtype=float).reshape(-1)
                out[: min(nq, q0.size)] = q0[: min(nq, q0.size)]
            except Exception:
                pass
            if nq >= 3 and out[2] == 0.0:
                out[2] = 1.0
            rospy.logwarn_throttle(
                1.0,
                f"State dimension mismatch fixed: input={x_in.size}, expected={nx}.",
            )
        if zero_vel and out.size > nq:
            out[nq:] = 0.0
        return out

    @staticmethod
    def _safe_token(text: str) -> str:
        s = str(text or "").strip().lower()
        out = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_"):
                out.append(ch)
            elif ch in (" ", "/", "\\", "."):
                out.append("_")
        tok = "".join(out).strip("_")
        return tok or "trajectory"

    def _clear_recorded_data(self) -> None:
        for k in list(self.recorded_data.keys()):
            self.recorded_data[k].clear()
        self.solve_times.clear()

    def _tracking_results_dir(self) -> Path:
        out_dir = Path(__file__).resolve().parent / "tracking_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _compose_tracking_file_tag(self) -> str:
        traj = self._safe_token(self.trajectory_name)
        mode = self._safe_token(self.controller_mode)
        return f"{traj}__{mode}"

    def _compute_tracking_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        t = np.asarray(self.recorded_data.get("time", []), dtype=float).reshape(-1)
        pos = np.asarray(self.recorded_data.get("position", []), dtype=float)
        pos_ref = np.asarray(self.recorded_data.get("reference_position", []), dtype=float)
        quat = np.asarray(self.recorded_data.get("orientation", []), dtype=float)
        quat_ref = np.asarray(self.recorded_data.get("reference_orientation", []), dtype=float)
        solve_ms = np.asarray(self.recorded_data.get("mpc_solve_time", []), dtype=float).reshape(-1)
        stats["samples"] = float(t.size)
        stats["duration_s"] = float(t[-1] - t[0]) if t.size >= 2 else 0.0
        if pos.ndim == 2 and pos_ref.ndim == 2 and pos.shape == pos_ref.shape and pos.shape[0] > 0:
            pe = np.linalg.norm(pos - pos_ref, axis=1)
            stats["pos_rmse_m"] = float(np.sqrt(np.mean(pe ** 2)))
            stats["pos_mean_m"] = float(np.mean(pe))
            stats["pos_max_m"] = float(np.max(pe))
        if quat.ndim == 2 and quat_ref.ndim == 2 and quat.shape == quat_ref.shape and quat.shape[0] > 0:
            dot = np.sum(quat * quat_ref, axis=1)
            dot = np.clip(np.abs(dot), 0.0, 1.0)
            ang = 2.0 * np.arccos(dot)
            stats["att_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(ang ** 2))))
            stats["att_mean_deg"] = float(np.degrees(np.mean(ang)))
            stats["att_max_deg"] = float(np.degrees(np.max(ang)))
        if solve_ms.size > 0:
            stats["solve_ms_mean"] = float(np.mean(solve_ms))
            stats["solve_ms_p95"] = float(np.percentile(solve_ms, 95))
            stats["solve_ms_max"] = float(np.max(solve_ms))
        return stats

    def _save_tracking_csv_and_stats(self, tag: str) -> Tuple[Path, Path]:
        out_dir = self._tracking_results_dir()
        csv_path = out_dir / f"{tag}.csv"
        txt_path = out_dir / f"{tag}_stats.txt"
        t = np.asarray(self.recorded_data.get("time", []), dtype=float).reshape(-1)
        pos = np.asarray(self.recorded_data.get("position", []), dtype=float)
        vel = np.asarray(self.recorded_data.get("velocity", []), dtype=float)
        omega = np.asarray(self.recorded_data.get("angular_velocity", []), dtype=float)
        quat = np.asarray(self.recorded_data.get("orientation", []), dtype=float)
        u = np.asarray(self.recorded_data.get("mpc_control", []), dtype=float)
        bcmd = np.asarray(self.recorded_data.get("body_rate_commands", []), dtype=float)
        thrust_cmd = np.asarray(self.recorded_data.get("thrust_command", []), dtype=float).reshape(-1)
        pos_ref = np.asarray(self.recorded_data.get("reference_position", []), dtype=float)
        vel_ref = np.asarray(self.recorded_data.get("reference_velocity", []), dtype=float)
        omega_ref = np.asarray(self.recorded_data.get("reference_angular_velocity", []), dtype=float)
        quat_ref = np.asarray(self.recorded_data.get("reference_orientation", []), dtype=float)
        solve_ms = np.asarray(self.recorded_data.get("mpc_solve_time", []), dtype=float).reshape(-1)
        n = int(t.size)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = [
                "time",
                "px", "py", "pz",
                "qx", "qy", "qz", "qw",
                "vx_b", "vy_b", "vz_b",
                "wx_b", "wy_b", "wz_b",
                "ref_px", "ref_py", "ref_pz",
                "ref_qx", "ref_qy", "ref_qz", "ref_qw",
                "ref_vx_b", "ref_vy_b", "ref_vz_b",
                "ref_wx_b", "ref_wy_b", "ref_wz_b",
                "u0", "u1", "u2", "u3", "u4", "u5",
                "cmd_wx", "cmd_wy", "cmd_wz", "cmd_thrust_norm",
                "solve_ms",
            ]
            w.writerow(header)
            for i in range(n):
                row = [float(t[i])]
                row.extend(pos[i].tolist() if pos.ndim == 2 and i < pos.shape[0] else [float("nan")] * 3)
                row.extend(quat[i].tolist() if quat.ndim == 2 and i < quat.shape[0] else [float("nan")] * 4)
                row.extend(vel[i].tolist() if vel.ndim == 2 and i < vel.shape[0] else [float("nan")] * 3)
                row.extend(omega[i].tolist() if omega.ndim == 2 and i < omega.shape[0] else [float("nan")] * 3)
                row.extend(pos_ref[i].tolist() if pos_ref.ndim == 2 and i < pos_ref.shape[0] else [float("nan")] * 3)
                row.extend(quat_ref[i].tolist() if quat_ref.ndim == 2 and i < quat_ref.shape[0] else [float("nan")] * 4)
                row.extend(vel_ref[i].tolist() if vel_ref.ndim == 2 and i < vel_ref.shape[0] else [float("nan")] * 3)
                row.extend(omega_ref[i].tolist() if omega_ref.ndim == 2 and i < omega_ref.shape[0] else [float("nan")] * 3)
                if u.ndim == 2 and i < u.shape[0]:
                    ui = u[i].tolist()
                    row.extend(ui[:6] + [float("nan")] * max(0, 6 - len(ui)))
                else:
                    row.extend([float("nan")] * 6)
                row.extend(bcmd[i].tolist() if bcmd.ndim == 2 and i < bcmd.shape[0] else [float("nan")] * 3)
                row.append(float(thrust_cmd[i]) if i < thrust_cmd.size else float("nan"))
                row.append(float(solve_ms[i]) if i < solve_ms.size else float("nan"))
                w.writerow(row)

        stats = self._compute_tracking_stats()
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(f"trajectory_name: {self.trajectory_name}\n")
            f.write(f"controller_mode: {self.controller_mode}\n")
            f.write(f"robot_name: {self.robot_name}\n")
            f.write(f"odom_source: {self.odom_source}\n")
            f.write(f"samples: {int(stats.get('samples', 0.0))}\n")
            for k in (
                "duration_s",
                "pos_rmse_m", "pos_mean_m", "pos_max_m",
                "att_rmse_deg", "att_mean_deg", "att_max_deg",
                "solve_ms_mean", "solve_ms_p95", "solve_ms_max",
            ):
                if k in stats:
                    f.write(f"{k}: {stats[k]:.6f}\n")
        return csv_path, txt_path

    def _build_mpc(self):
        """构建 Crocoddyl MPC 实例（与 scripts/ 仿真脚本相同的类）。"""
        s500_yaml_path, urdf_path = self._model_paths_for_robot()
        if self.controller_mode in ("croc_full_state", "px4", "geometric"):
            self.mpc = UAMCrocoddylStateTrackingMPC(
                s500_yaml_path=s500_yaml_path,
                urdf_path=urdf_path,
                dt_mpc=self.dt_mpc,
                horizon=self.horizon,
                w_state_track=self.w_state_track,
                w_state_reg=self.w_state_reg,
                w_control=self.w_control,
                w_terminal_track=self.w_terminal_track,
                w_pos=self.w_pos,
                w_att=self.w_att,
                w_joint=self.w_joint,
                w_vel=self.w_vel,
                w_omega=self.w_omega,
                w_joint_vel=self.w_joint_vel,
                w_u_thrust=self.w_u_thrust,
                w_u_joint_torque=self.w_u_joint_torque,
                use_thrust_constraints=True,
            )
            # 全状态模式下 mpc_reg 直接复用 mpc（避免重复构建）
            self.mpc_reg = self.mpc
        elif self.controller_mode == "croc_ee_pose":
            if not self.arm_enabled:
                raise ValueError(
                    "controller_mode 'croc_ee_pose' requires arm/EE model; "
                    "use 'croc_full_state', 'px4', or 'geometric' for s500."
                )
            ee_weights = EETrackingWeights(
                w_pos=self.ee_w_pos,
                w_rot_rp=self.ee_w_rot_rp,
                w_rot_yaw=self.ee_w_rot_yaw,
                w_vel_lin=self.ee_w_vel_lin,
                w_vel_ang_rp=self.ee_w_vel_ang_rp,
                w_vel_ang_yaw=self.ee_w_vel_ang_yaw,
                w_u=self.ee_w_u,
                w_terminal_scale=self.ee_w_terminal,
                w_state_reg=self.w_state_reg,
                w_state_track=self.w_state_track,
            )
            self.mpc = UAMEEPoseTrackingCrocoddylMPC(
                s500_yaml_path=s500_yaml_path,
                urdf_path=urdf_path,
                dt_mpc=self.dt_mpc,
                horizon=self.horizon,
                ee_weights=ee_weights,
                use_thrust_constraints=True,
            )
            # EE-pose 模式下，regulation 需要单独的全状态 MPC
            # （build_shooting_problem_along_plan 仅支持 full-state 模式）
            self.mpc_reg = UAMCrocoddylStateTrackingMPC(
                s500_yaml_path=s500_yaml_path,
                urdf_path=urdf_path,
                dt_mpc=self.dt_mpc,
                horizon=self.horizon,
                w_state_track=self.w_state_track,
                w_state_reg=self.w_state_reg,
                w_control=self.w_control,
                w_terminal_track=self.w_terminal_track,
                w_pos=self.w_pos,
                w_att=self.w_att,
                w_joint=self.w_joint,
                w_vel=self.w_vel,
                w_omega=self.w_omega,
                w_joint_vel=self.w_joint_vel,
                w_u_thrust=self.w_u_thrust,
                w_u_joint_torque=self.w_u_joint_torque,
                use_thrust_constraints=True,
            )
        else:
            raise ValueError(
                f"Unknown controller_mode: {self.controller_mode!r}. "
                "Use 'croc_full_state', 'croc_ee_pose', 'px4', or 'geometric'."
            )
        self.x_nom = self._compute_x_nominal(self.mpc_reg)

        # 从 s500_config 提取推力参数（用于归一化）
        p = self.mpc.s500_config["platform"]
        self._n_rotors = int(p["n_rotors"])
        self._single_max_thrust = float(p["max_thrust"])
        self._total_max_thrust_cfg = self._n_rotors * self._single_max_thrust
        if self.max_thrust_total <= 0:
            self.max_thrust_total = self._total_max_thrust_cfg
        rospy.loginfo(
            f"MPC built: mode={self.controller_mode}, nq={self.mpc.nq}, "
            f"nu={self.mpc.nu}, max_thrust={self.max_thrust_total:.1f} N"
        )

    # =========================================================================
    # 初始化：ROS 接口
    # =========================================================================

    def _init_subscribers(self):
        rospy.Subscriber("/mavros/state", State, self._mav_state_cb)

        if self.odom_source == "gazebo":
            rospy.Subscriber(
                "/gazebo/model_states", ModelStates, self._gazebo_state_cb
            )
        else:
            rospy.Subscriber(
                "/mavros/local_position/odom", Odometry, self._mavros_odom_cb
            )

        if self.use_simulation:
            rospy.Subscriber(
                "/arm_controller/joint_states", JointState, self._arm_state_sim_cb
            )
        else:
            rospy.Subscriber("/joint_states", JointState, self._arm_state_cb)

        # Regulation 目标话题（GUI 或外部工具发布）
        # 消息格式：Float64MultiArray.data = [x, y, z, yaw_deg, j1_deg, j2_deg]
        from std_msgs.msg import Float64MultiArray
        rospy.Subscriber(
            "~/regulation_target", Float64MultiArray, self._regulation_target_cb
        )

    def _init_publishers(self):
        self.body_rate_thrust_pub = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10
        )
        self.mavros_setpoint_raw_pub = rospy.Publisher(
            "/mavros/setpoint_raw/local", PositionTarget, queue_size=10
        )
        self.yaw_pub = rospy.Publisher("/reference/yaw", Float32, queue_size=10)
        self.arm_control_pub = rospy.Publisher(
            "/desired_joint_states", JointState, queue_size=10
        )
        # 仿真单关节话题
        from std_msgs.msg import Float64
        self.joint1_pub = rospy.Publisher(
            "/arm_controller/joint_1_controller/command", Float64, queue_size=10
        )
        self.joint2_pub = rospy.Publisher(
            "/arm_controller/joint_2_controller/command", Float64, queue_size=10
        )
        self.debug_pub = rospy.Publisher("/suite_mpc/state", MpcState, queue_size=10)

        # ── RViz 路径话题（与 run_controller.py 相同名称） ────────────────────
        # 参考轨迹（整条，固定）
        self.ref_traj_path_pub = rospy.Publisher(
            "/reference/current_trajectory", RosPath, queue_size=5, latch=True
        )
        # MPC horizon 预测路径（每个控制步更新）
        self.mpc_planned_path_pub = rospy.Publisher(
            "/mpc/current_planned_path", RosPath, queue_size=5
        )
        # 飞机实际轨迹（累积）
        self.uav_actual_path_pub = rospy.Publisher(
            "uav_path", RosPath, queue_size=5
        )
        # EE 规划轨迹（固定）
        self.ee_planned_path_pub = rospy.Publisher(
            "/ee/planned_trajectory", RosPath, queue_size=5, latch=True
        )
        # EE 实际轨迹（累积）
        self.ee_actual_path_pub = rospy.Publisher(
            "/ee/actual_trajectory", RosPath, queue_size=5
        )

        # ── WholeBody 可视化（eagle_mpc_viz，可选） ───────────────────────────
        self._wb_current_pub  = None
        self._wb_target_pub   = None
        self._wb_planned_pub  = None
        if _WHOLEBODY_VIZ_OK:
            try:
                rm  = self.mpc.robot_model
                pp  = self.mpc.s500_config["platform"]
                # eagle_mpc_viz 使用 platform_params 对象；这里传入轻量 namespace
                class _PP:
                    pass
                _pp = _PP()
                _pp.n_rotors = int(pp["n_rotors"])
                # WholeBodyStatePublisher / TrajectoryPublisher 只需 robot_model & platform_params
                self._wb_current_pub = WholeBodyStatePublisher(
                    "whole_body_state_current", rm, _pp, frame_id="world"
                )
                self._wb_target_pub = WholeBodyStatePublisher(
                    "whole_body_state_target", rm, _pp, frame_id="world"
                )
                self._wb_planned_pub = WholeBodyTrajectoryPublisher(
                    "whole_body_partial_trajectory_current", rm, _pp, frame_id="world"
                )
                rospy.loginfo("WholeBody visualization publishers initialized.")
            except Exception as e:
                rospy.logwarn(f"WholeBody viz init failed (non-fatal): {e}")
                self._wb_current_pub = None
                self._wb_target_pub  = None
                self._wb_planned_pub = None

    def _init_services(self):
        rospy.Service("start_tracking", Trigger, self._svc_start_tracking)
        rospy.Service("stop_tracking", Trigger, self._svc_stop_tracking)
        rospy.Service("save_data", Trigger, self._svc_save_data)
        rospy.Service("update_trajectory", Trigger, self._svc_update_trajectory)
        rospy.Service("reset_to_initial", Trigger, self._svc_reset_to_initial)
        rospy.Service("set_regulation_target", Trigger, self._svc_set_regulation_target)
        rospy.Service("update_controller_params", Trigger, self._svc_update_controller_params)

    # =========================================================================
    # 状态回调（与 run_controller.py 逻辑相同）
    # =========================================================================

    def _mav_state_cb(self, msg: State):
        self.px4_state = msg

    def _arm_state_cb(self, msg: JointState):
        """实机：/joint_states（真实关节顺序 -2/-1）。"""
        self.arm_state = msg
        nj = self.arm_joint_number
        self.state[7 : 7 + nj] = [msg.position[-1], msg.position[-2]]
        self.state[-nj:] = [msg.velocity[-1], msg.velocity[-2]]

    def _arm_state_sim_cb(self, msg: JointState):
        """仿真：/arm_controller/joint_states（顺序 0/1）。"""
        self.arm_state = msg
        nj = self.arm_joint_number
        self.state[7 : 7 + nj] = list(msg.position[:nj])
        self.state[-nj:] = list(msg.velocity[:nj])

    def _gazebo_state_cb(self, msg: ModelStates):
        try:
            idx = msg.name.index(self.robot_name)
        except ValueError:
            rospy.logerr_throttle(5.0, f"Robot '{self.robot_name}' not in Gazebo model_states")
            return
        pose = msg.pose[idx]
        twist = msg.twist[idx]
        self.state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.state[3:7] = [
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ]
        nj = self.arm_joint_number
        quat = self.state[3:7]
        R = quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])[:3, :3]
        v_world = np.array([twist.linear.x, twist.linear.y, twist.linear.z], dtype=float)
        w_world = np.array([twist.angular.x, twist.angular.y, twist.angular.z], dtype=float)
        self.state[7 + nj : 10 + nj] = R.T @ v_world   # body-frame linear vel
        self.state[10 + nj : 13 + nj] = R.T @ w_world  # body-frame angular vel

    def _mavros_odom_cb(self, msg: Odometry):
        pose = msg.pose.pose
        twist = msg.twist.twist
        self.state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.state[3:7] = [
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ]
        nq = self.mpc.nq
        self.state[nq : nq + 3] = [twist.linear.x, twist.linear.y, twist.linear.z]
        self.state[nq + 3 : nq + 6] = [twist.angular.x, twist.angular.y, twist.angular.z]

    # =========================================================================
    # 主控制回调
    # =========================================================================

    def _control_callback(self, event):
        now = rospy.Time.now()

        # ── 时间管理 ──────────────────────────────────────────────────────────
        if self.trajectory_started and not self.traj_finished:
            t_elapsed = (now - self.controller_start_time).to_sec()
            t_total = float(self.t_plan[-1] - self.t_plan[0])
            if t_elapsed >= t_total:
                self.traj_finished = True
                self.recording_enabled = False
                # 轨迹自然完成 → 锁定在终点继续跟踪，等待用户点击 /stop_tracking
                rospy.loginfo(
                    "Trajectory tracking finished. Holding at end-point. "
                    "Call /stop_tracking to enter regulation mode."
                )
        elif self.traj_finished:
            t_elapsed = float(self.t_plan[-1] - self.t_plan[0])
        else:
            t_elapsed = 0.0

        # ── PX4 / Geometric 高层控制模式（不走 Crocoddyl 在线求解）────────────
        if self.controller_mode in ("px4", "geometric"):
            self._control_callback_high_level(t_elapsed)
            return

        # ── MPC 求解 ──────────────────────────────────────────────────────────
        x_now = self.state.copy()

        # ── Regulation 模式：MPC 镇定到 _reg_target ──────────────────────────
        if self._regulating:
            # Keep reference policy deterministic in regulation mode.
            if not self._reg_target_locked:
                self._reg_target = self._default_reg_target()
            t0 = time.perf_counter()
            u_cmd, xs_next = self._solve_mpc_regulate(x_now)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            self.solve_times.append(solve_ms)
            rospy.logdebug_throttle(
                1.0, f"[regulate] MPC solve: {solve_ms:.1f} ms | target={self._reg_target[:3].tolist()}"
            )
            if u_cmd is not None:
                self._u_hold = u_cmd
            self._publish_body_rate_thrust(self._u_hold, xs_next)
            if self.arm_enabled:
                self._publish_arm_cmd(xs_next)
            self._publish_debug(0.0, solve_ms, x_ref_override=self._reg_target)
            # Regulation 模式下也追加实际路径（方便观察归位过程）
            self._append_uav_actual_path_point()
            if self.arm_enabled:
                self._append_ee_actual_path_point()
            if u_cmd is not None and self._reg_xs_guess is not None:
                self._stage_mpc_planned_path(self._reg_xs_guess)
            self._maybe_flush_viz_publishes(xs_next)
            return

        t0 = time.perf_counter()
        u_cmd, xs_next = self._solve_mpc(x_now, t_elapsed)
        solve_ms = (time.perf_counter() - t0) * 1000.0
        self.solve_times.append(solve_ms)
        rospy.logdebug_throttle(
            1.0, f"MPC solve: {solve_ms:.1f} ms | t_elapsed={t_elapsed:.2f}s"
        )

        if u_cmd is not None:
            self._u_hold = u_cmd

        # ── 发布控制指令 ──────────────────────────────────────────────────────
        self._publish_body_rate_thrust(self._u_hold, xs_next)
        if self.arm_enabled:
            self._publish_arm_cmd(xs_next)
        self._publish_debug(t_elapsed, solve_ms)

        # ── RViz 可视化 ───────────────────────────────────────────────────────
        # 实际路径只在 tracking 激活时累积，避免起始点与 x_plan[0] 不符
        if self.trajectory_started:
            self._append_uav_actual_path_point()
            if self.arm_enabled:
                self._append_ee_actual_path_point()
        if u_cmd is not None and self._xs_guess is not None:
            self._stage_mpc_planned_path(self._xs_guess)
        # 降频批量发布（15 Hz 默认，避免高频序列化）
        self._maybe_flush_viz_publishes(xs_next)

        # ── 数据录制 ──────────────────────────────────────────────────────────
        if self.recording_enabled:
            self._record(t_elapsed, self._u_hold, xs_next, solve_ms)

    # =========================================================================
    # MPC 求解核心
    # =========================================================================

    def _solve_mpc(
        self, x_now: np.ndarray, t_elapsed: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        构建 shooting problem 并求解，返回 (u_opt, xs_next) 或 (None, None)。

        xs_next: MPC 规划的下一时刻状态（用于从 xs[1] 读取 body rate 参考）。
        """
        t_start_plan = float(self.t_plan[0])
        t_query = t_start_plan + t_elapsed

        try:
            if self.controller_mode == "croc_full_state":
                prob = self.mpc.build_shooting_problem_along_plan(
                    x_now,
                    self.x_nom,
                    t_query,
                    self.t_plan,
                    self.x_plan,
                )
            else:  # croc_ee_pose
                prob = self.mpc.build_shooting_problem_along_ee_ref(
                    x_now,
                    t_query,
                    self.t_ref_ee,
                    self.p_ref_ee,
                    self.yaw_ref_ee,
                    self.dp_ref_ee,
                    self.dyaw_ref_ee,
                    t_plan=self.t_plan,
                    x_plan=self.x_plan,
                )

            solver = crocoddyl.SolverBoxFDDP(prob)
            solver.convergence_init = 1e-9
            solver.convergence_stop = 1e-7
            try:
                solver.setCallbacks([])
            except Exception:
                pass

            # Warm start
            if self._xs_guess is None:
                xs_init = [x_now.copy() for _ in range(self.horizon + 1)]
                us_init = [self._hover_thrust_cmd() for _ in range(self.horizon)]
            else:
                xs_init = self._xs_guess
                us_init = self._us_guess
                xs_init[0] = x_now.copy()

            solver.solve(xs_init, us_init, self.mpc_max_iter)

            u_opt = np.array(solver.us[0], dtype=float).copy()

            # Shift warm start for next iteration
            H = self.horizon
            self._xs_guess = (
                [np.array(solver.xs[i + 1], dtype=float).copy() for i in range(H)]
                + [np.array(solver.xs[-1], dtype=float).copy()]
            )
            self._xs_guess[0] = x_now.copy()
            self._us_guess = (
                [np.array(solver.us[i + 1], dtype=float).copy() for i in range(H - 1)]
                + [np.array(solver.us[-1], dtype=float).copy()]
            )

            xs_next = np.array(solver.xs[1], dtype=float).copy()
            return u_opt, xs_next

        except Exception as e:
            rospy.logwarn_throttle(1.0, f"MPC solve failed: {e}")
            self._xs_guess = None
            self._us_guess = None
            return None, None

    def _solve_mpc_regulate(
        self, x_now: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        将 _reg_target 作为常值参考，构建点镇定 shooting problem 并求解。
        用于 regulation 模式（启动时、stop 后、reset 后均进入此模式）。
        使用独立的 warm-start 缓存（_reg_xs_guess / _reg_us_guess）。
        """
        x_tgt = self._reg_target
        H     = self.horizon
        dt    = self.dt_mpc

        # 常值参考轨迹：H+1 个节点，全部等于 _reg_target
        t_reg = np.array([float(i) * dt for i in range(H + 1)], dtype=float)
        x_reg = np.array([x_tgt.copy() for _ in range(H + 1)], dtype=float)

        try:
            # 始终用独立的全状态 MPC 进行 regulation（EE-pose 模式下 self.mpc 不支持此接口）
            prob = self.mpc_reg.build_shooting_problem_along_plan(
                x_now,
                self.x_nom,
                t_reg[0],   # t_query = 0 → 全 horizon 均参考 x_tgt
                t_reg,
                x_reg,
            )

            solver = crocoddyl.SolverBoxFDDP(prob)
            solver.convergence_init = 1e-9
            solver.convergence_stop = 1e-7
            try:
                solver.setCallbacks([])
            except Exception:
                pass

            # Warm start
            if self._reg_xs_guess is None:
                xs_init = [x_now.copy() for _ in range(H + 1)]
                us_init = [self._hover_thrust_cmd() for _ in range(H)]
            else:
                xs_init = self._reg_xs_guess
                us_init = self._reg_us_guess
                xs_init[0] = x_now.copy()

            solver.solve(xs_init, us_init, self.mpc_max_iter)

            u_opt = np.array(solver.us[0], dtype=float).copy()

            # Shift warm start
            self._reg_xs_guess = (
                [np.array(solver.xs[i + 1], dtype=float).copy() for i in range(H)]
                + [np.array(solver.xs[-1], dtype=float).copy()]
            )
            self._reg_xs_guess[0] = x_now.copy()
            self._reg_us_guess = (
                [np.array(solver.us[i + 1], dtype=float).copy() for i in range(H - 1)]
                + [np.array(solver.us[-1], dtype=float).copy()]
            )

            xs_next = np.array(solver.xs[1], dtype=float).copy()
            return u_opt, xs_next

        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Regulate MPC solve failed: {e}")
            self._reg_xs_guess = None
            self._reg_us_guess = None
            return None, None

    def _make_reg_target(
        self,
        x: float, y: float, z: float,
        yaw_deg: float,
        j1_deg: float, j2_deg: float,
    ) -> np.ndarray:
        """
        将 (x, y, z, yaw_deg, j1_deg, j2_deg) 转为 MPC 全状态向量。
        姿态：roll=0, pitch=0, yaw=yaw_deg；速度全为 0。
        """
        from tf.transformations import quaternion_from_euler
        nq = self.mpc.nq
        nv = self.mpc.nv
        x_tgt = np.zeros(nq + nv, dtype=float)
        x_tgt[0] = x
        x_tgt[1] = y
        x_tgt[2] = z
        quat = quaternion_from_euler(0.0, 0.0, math.radians(yaw_deg))  # [qx,qy,qz,qw]
        x_tgt[3:7] = quat
        nj = self.arm_joint_number
        if nj >= 1:
            x_tgt[7]      = math.radians(j1_deg)
        if nj >= 2:
            x_tgt[7 + 1]  = math.radians(j2_deg)
        return x_tgt

    def _regulation_target_cb(self, msg) -> None:
        """
        接收来自 GUI 或外部工具的 regulation 目标。
        Float64MultiArray.data = [x, y, z, yaw_deg, j1_deg, j2_deg]
        目标更新后立即重置 warm-start 缓存，以便 MPC 重新规划。
        """
        if self.mpc is None:
            return
        d = list(msg.data)
        if len(d) < 6:
            rospy.logwarn(f"[regulation_target] 期望 6 个数值，收到 {len(d)}")
            return
        try:
            new_target = self._make_reg_target(
                x=float(d[0]), y=float(d[1]), z=float(d[2]),
                yaw_deg=float(d[3]),
                j1_deg=float(d[4]), j2_deg=float(d[5]),
            )
            self._reg_target    = new_target
            self._reg_target_locked = True
            self._reg_xs_guess  = None   # 目标改变，丢弃旧 warm-start
            self._reg_us_guess  = None
            # 只有在未进行跟踪（trajectory_started=False）时才立即激活 regulation
            # 轨迹结束后（traj_finished=True）仍在终点保持跟踪，等待 /stop_tracking
            if not self.trajectory_started:
                self._regulating = True
            rospy.loginfo(
                f"[regulation_target] 更新目标: "
                f"x={d[0]:.2f} y={d[1]:.2f} z={d[2]:.2f} "
                f"yaw={d[3]:.1f}° j1={d[4]:.1f}° j2={d[5]:.1f}°"
                + (" (regulation 已激活)" if not self.trajectory_started else " (tracking 中，下次 stop 后生效)")
            )
        except Exception as e:
            rospy.logerr(f"[regulation_target] 解析失败: {e}")

    # =========================================================================
    # PX4 / Geometric 高层参考发布
    # =========================================================================

    def _sample_ref_state(self, t_elapsed: float) -> np.ndarray:
        t_query = float(self.t_plan[0]) + float(t_elapsed)
        return interp_full_state_piecewise(
            t_query, self.t_plan, self.x_plan, self.mpc.robot_model
        )

    def _linear_vel_world_from_state(self, x_ref: np.ndarray) -> np.ndarray:
        nq = self.mpc.nq
        vel_body = np.asarray(x_ref[nq : nq + 3], dtype=float)
        quat = np.asarray(x_ref[3:7], dtype=float)
        R = quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])[:3, :3]
        return R @ vel_body

    def _sample_ref_kinematics(
        self, t_elapsed: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        从 x_plan 插值采样高层参考：
        pos_world, vel_world, acc_world, yaw。
        """
        x_ref = self._sample_ref_state(t_elapsed)
        pos = np.asarray(x_ref[0:3], dtype=float).copy()
        vel = self._linear_vel_world_from_state(x_ref)

        if (
            self.ddp_plan is not None
            and isinstance(self.ddp_plan, np.ndarray)
            and self.ddp_plan.ndim == 2
            and self.ddp_plan.shape[0] == len(self.t_plan)
            and self.ddp_plan.shape[1] == 3
        ):
            t_query = float(self.t_plan[0]) + float(t_elapsed)
            tq = float(np.clip(t_query, float(self.t_plan[0]), float(self.t_plan[-1])))
            acc = np.array(
                [
                    np.interp(tq, self.t_plan, self.ddp_plan[:, 0]),
                    np.interp(tq, self.t_plan, self.ddp_plan[:, 1]),
                    np.interp(tq, self.t_plan, self.ddp_plan[:, 2]),
                ],
                dtype=float,
            )
        else:
            dt = max(float(self.dt_control), 1e-3)
            t_prev = max(0.0, float(t_elapsed) - dt)
            t_next = min(float(self.t_plan[-1] - self.t_plan[0]), float(t_elapsed) + dt)
            if t_next > t_prev + 1e-6:
                x_prev = self._sample_ref_state(t_prev)
                x_next = self._sample_ref_state(t_next)
                v_prev = self._linear_vel_world_from_state(x_prev)
                v_next = self._linear_vel_world_from_state(x_next)
                acc = (v_next - v_prev) / (t_next - t_prev)
            else:
                acc = np.zeros(3, dtype=float)

        quat = np.asarray(x_ref[3:7], dtype=float)
        yaw = float(euler_from_quaternion([quat[0], quat[1], quat[2], quat[3]])[2])
        return pos, vel, acc, yaw

    def _control_callback_high_level(self, t_elapsed: float) -> None:
        if self._regulating:
            if not self._reg_target_locked:
                self._reg_target = self._default_reg_target()
            # PX4 / Geometric 下 regulation 使用常值目标（而非 Crocoddyl regulate 求解）
            x_ref = np.asarray(self._reg_target, dtype=float).copy()
            pos_ref = np.asarray(x_ref[0:3], dtype=float)
            vel_ref = np.zeros(3, dtype=float)
            acc_ref = np.zeros(3, dtype=float)
            q_ref = np.asarray(x_ref[3:7], dtype=float)
            yaw_ref = float(euler_from_quaternion([q_ref[0], q_ref[1], q_ref[2], q_ref[3]])[2])
        else:
            x_ref = self._sample_ref_state(t_elapsed)
            pos_ref, vel_ref, acc_ref, yaw_ref = self._sample_ref_kinematics(t_elapsed)
            if self.traj_finished:
                vel_ref = np.zeros(3, dtype=float)
                acc_ref = np.zeros(3, dtype=float)

        if self.controller_mode == "px4":
            self._publish_mavros_setpoint_raw(pos_ref, vel_ref, acc_ref, yaw_ref, 0.0)
        elif self.controller_mode == "geometric":
            self._publish_geometric_bodyrate_thrust(pos_ref, vel_ref, acc_ref, yaw_ref)

        # 机械臂维持参考轨迹角度
        if self.arm_enabled:
            self._publish_arm_cmd(x_ref)

        if self._regulating:
            self._publish_debug(t_elapsed, 0.0, x_ref_override=self._reg_target)
        else:
            self._publish_debug(t_elapsed, 0.0)
        if self.trajectory_started:
            self._append_uav_actual_path_point()
            if self.arm_enabled:
                self._append_ee_actual_path_point()
        self._maybe_flush_viz_publishes(None)

        if self.recording_enabled:
            self._record(
                t_elapsed,
                np.zeros(self.mpc.nu, dtype=float),
                x_ref,
                0.0,
            )
            if self.controller_mode == "px4":
                self.recorded_data["body_rate_commands"].append([0.0, 0.0, 0.0])
                self.recorded_data["thrust_command"].append(0.0)

    # =========================================================================
    # 发布控制指令
    # =========================================================================

    def _publish_body_rate_thrust(
        self,
        u: np.ndarray,
        xs_next: Optional[np.ndarray],
    ):
        """
        将 Crocoddyl MPC 控制量 u=[T1..T4, tau_j1, tau_j2] 转换并发布：
        - 总推力归一化 → att_msg.thrust
        - MPC 规划的下一时刻 body angular rate → att_msg.body_rate
        """
        nq = self.mpc.nq

        # ── 总推力 ─────────────────────────────────────────────────────────
        total_thrust_N = float(np.sum(u[: self._n_rotors]))
        thrust_normalized = np.clip(
            total_thrust_N / self.max_thrust_total,
            self.min_thrust_cmd,
            self.max_thrust_cmd,
        )

        # ── Body angular rate（来自 MPC xs[1]） ───────────────────────────
        # s500: nq+nv = 13, and body rates are at [nq+3 : nq+6], so equality is valid.
        if xs_next is not None and xs_next.size >= nq + 6:
            roll_rate = float(xs_next[nq + 3])
            pitch_rate = float(xs_next[nq + 4])
            yaw_rate = float(xs_next[nq + 5])
        else:
            # 若 MPC 求解失败，用上一次的参考状态估算
            t_query = 0.0
            if self.trajectory_started and self.controller_start_time is not None:
                t_query = (rospy.Time.now() - self.controller_start_time).to_sec()
            x_ref = interp_full_state_piecewise(
                float(self.t_plan[0]) + t_query, self.t_plan, self.x_plan, self.mpc.robot_model
            )
            roll_rate = float(x_ref[nq + 3])
            pitch_rate = float(x_ref[nq + 4])
            yaw_rate = float(x_ref[nq + 5])

        # 限幅
        roll_rate = float(np.clip(roll_rate, -self.max_angular_velocity, self.max_angular_velocity))
        pitch_rate = float(np.clip(pitch_rate, -self.max_angular_velocity, self.max_angular_velocity))
        yaw_rate = float(np.clip(yaw_rate, -self.max_angular_velocity, self.max_angular_velocity))

        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        att_msg.body_rate = Vector3(roll_rate, pitch_rate, yaw_rate)
        att_msg.thrust = thrust_normalized

        if self.recording_enabled:
            self.recorded_data["body_rate_commands"].append([roll_rate, pitch_rate, yaw_rate])
            self.recorded_data["thrust_command"].append(thrust_normalized)

        self.body_rate_thrust_pub.publish(att_msg)

    def _publish_mavros_setpoint_raw(
        self,
        pos_world: np.ndarray,
        vel_world: np.ndarray,
        acc_world: np.ndarray,
        yaw: float,
        yaw_rate: float,
    ) -> None:
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = PositionTarget.IGNORE_YAW_RATE
        msg.position = Point(float(pos_world[0]), float(pos_world[1]), float(pos_world[2]))
        msg.velocity.x = float(vel_world[0])
        msg.velocity.y = float(vel_world[1])
        msg.velocity.z = float(vel_world[2])
        msg.acceleration_or_force.x = float(acc_world[0])
        msg.acceleration_or_force.y = float(acc_world[1])
        msg.acceleration_or_force.z = float(acc_world[2])
        msg.yaw = float(yaw)
        msg.yaw_rate = float(yaw_rate)
        self.mavros_setpoint_raw_pub.publish(msg)

    def _publish_reference_yaw(self, yaw: float) -> None:
        yaw_msg = Float32()
        yaw_msg.data = float(yaw)
        self.yaw_pub.publish(yaw_msg)

    @staticmethod
    def _vee(M: np.ndarray) -> np.ndarray:
        return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)

    def _publish_geometric_bodyrate_thrust(
        self,
        pos_ref: np.ndarray,
        vel_ref: np.ndarray,
        acc_ref: np.ndarray,
        yaw_ref: float,
    ) -> None:
        """
        内置 geometric controller:
          (p,v) 误差 -> 期望加速度 -> 期望姿态 -> body rate + thrust。
        """
        nq = self.mpc.nq
        p = np.asarray(self.state[0:3], dtype=float)
        v_body = np.asarray(self.state[nq : nq + 3], dtype=float)
        quat = np.asarray(self.state[3:7], dtype=float)
        R = quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])[:3, :3]
        v = R @ v_body

        e_p = p - np.asarray(pos_ref, dtype=float)
        e_v = v - np.asarray(vel_ref, dtype=float)
        e3 = np.array([0.0, 0.0, 1.0], dtype=float)
        a_des = (
            np.asarray(acc_ref, dtype=float)
            - self.geo_kp_pos * e_p
            - self.geo_kd_vel * e_v
            + 9.81 * e3
        )

        # 倾角限幅
        a_xy = np.linalg.norm(a_des[:2])
        a_z = max(1e-3, float(a_des[2]))
        tilt = math.atan2(a_xy, a_z)
        tilt_max = math.radians(max(1.0, self.geo_max_tilt_deg))
        if tilt > tilt_max and a_xy > 1e-6:
            scale = math.tan(tilt_max) * a_z / a_xy
            a_des[0] *= scale
            a_des[1] *= scale

        if np.linalg.norm(a_des) < 1e-6:
            a_des = 9.81 * e3
        b3_des = a_des / np.linalg.norm(a_des)
        b1_yaw = np.array([math.cos(yaw_ref), math.sin(yaw_ref), 0.0], dtype=float)
        b2_des = np.cross(b3_des, b1_yaw)
        n_b2 = np.linalg.norm(b2_des)
        if n_b2 < 1e-6:
            b2_des = np.array([-math.sin(yaw_ref), math.cos(yaw_ref), 0.0], dtype=float)
        else:
            b2_des /= n_b2
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack([b1_des, b2_des, b3_des])

        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = self._vee(e_R_mat)
        omega = np.asarray(self.state[nq + 3 : nq + 6], dtype=float)
        rate_cmd = -self.geo_kR * e_R - self.geo_kOmega * omega
        rate_cmd = np.clip(rate_cmd, -self.max_angular_velocity, self.max_angular_velocity)

        mass = float(pin.computeTotalMass(self.mpc.robot_model))
        thrust_N = float(mass * np.dot(a_des, R[:, 2]))
        thrust_cmd = float(
            np.clip(thrust_N / self.max_thrust_total, self.min_thrust_cmd, self.max_thrust_cmd)
        )

        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        att_msg.body_rate = Vector3(float(rate_cmd[0]), float(rate_cmd[1]), float(rate_cmd[2]))
        att_msg.thrust = thrust_cmd
        self.body_rate_thrust_pub.publish(att_msg)
        self._publish_reference_yaw(float(yaw_ref))

        if self.recording_enabled:
            self.recorded_data["body_rate_commands"].append(rate_cmd.tolist())
            self.recorded_data["thrust_command"].append(thrust_cmd)

    def _publish_arm_cmd(self, xs_next: Optional[np.ndarray]):
        """
        根据 arm_control_mode 发布关节指令。
        位置参考 = MPC 规划下一时刻状态 xs[1] 的关节角。
        """
        from std_msgs.msg import Float64

        nj = self.arm_joint_number
        if xs_next is not None and xs_next.size >= 7 + nj:
            ref_j = xs_next[7 : 7 + nj]
            ref_jdot = xs_next[7 + nj + 6 : 7 + nj + 6 + nj] if xs_next.size >= 7 + nj + 6 + nj else np.zeros(nj)
        else:
            # 降级：用规划中该时刻的关节值
            t_elapsed = 0.0
            if self.trajectory_started and self.controller_start_time is not None:
                t_elapsed = (rospy.Time.now() - self.controller_start_time).to_sec()
            x_ref = interp_full_state_piecewise(
                float(self.t_plan[0]) + t_elapsed, self.t_plan, self.x_plan, self.mpc.robot_model
            )
            ref_j = x_ref[7 : 7 + nj]
            ref_jdot = np.zeros(nj)

        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [f"joint_{i+1}" for i in range(nj)]

        if self.arm_control_mode == "position":
            joint_msg.position = list(ref_j)
            joint_msg.velocity = [0.0] * nj
            joint_msg.effort = [0.0] * nj
        elif self.arm_control_mode == "position_velocity":
            joint_msg.position = list(ref_j)
            joint_msg.velocity = list(ref_jdot)
            joint_msg.effort = [0.0] * nj
        elif self.arm_control_mode == "velocity":
            joint_msg.position = [0.0] * nj
            joint_msg.velocity = list(ref_jdot)
            joint_msg.effort = [0.0] * nj
        else:
            joint_msg.position = list(ref_j)
            joint_msg.velocity = [0.0] * nj
            joint_msg.effort = [0.0] * nj

        self.arm_control_pub.publish(joint_msg)

        # 仿真单关节话题
        if self.use_simulation and nj >= 2:
            from std_msgs.msg import Float64 as F64
            self.joint1_pub.publish(F64(float(ref_j[0])))
            self.joint2_pub.publish(F64(float(ref_j[1])))

        if self.recording_enabled:
            self.recorded_data["arm_joint_commands"].append(list(ref_j))

    def _publish_debug(
        self,
        t_elapsed: float,
        solve_ms: float,
        x_ref_override: Optional[np.ndarray] = None,
    ):
        """发布 /suite_mpc/state 调试话题。"""
        try:
            msg = MpcState()
            msg.header.stamp = rospy.Time.now()
            msg.state = self.state.tolist()

            if x_ref_override is not None:
                x_ref = np.asarray(x_ref_override, dtype=float).flatten()
            else:
                x_ref = interp_full_state_piecewise(
                    float(self.t_plan[0]) + t_elapsed, self.t_plan, self.x_plan, self.mpc.robot_model
                )
            msg.state_ref = x_ref.tolist()
            nq = self.mpc.nq
            err = pin.difference(self.mpc.robot_model, x_ref[:nq], self.state[:nq])
            msg.state_error = err.tolist()
            msg.solving_time = float(solve_ms / 1000.0)
            msg.u_mpc = list(self._u_hold)
            self.debug_pub.publish(msg)
        except Exception:
            pass

    # =========================================================================
    # RViz 路径可视化（仿照 run_controller.py）
    # =========================================================================

    def _rebuild_cached_viz_paths(self):
        """
        将完整规划轨迹（x_plan）转为 RosPath，一次性构建并缓存。
        同时计算 EE 规划路径（需要 Pinocchio FK）。
        由 _load_trajectory() 在轨迹加载后调用一次。
        """
        self._cached_ref_path     = None
        self._cached_ee_plan_path = None

        if self.x_plan is None or len(self.x_plan) == 0:
            return

        stamp = rospy.Time.now()

        # ── UAV base 参考路径 ─────────────────────────────────────────────
        ref_msg = RosPath()
        ref_msg.header.stamp    = stamp
        ref_msg.header.frame_id = "map"
        for x in self.x_plan:
            ps = PoseStamped()
            ps.header = ref_msg.header
            ps.pose.position.x    = float(x[0])
            ps.pose.position.y    = float(x[1])
            ps.pose.position.z    = float(x[2])
            ps.pose.orientation.x = float(x[3])
            ps.pose.orientation.y = float(x[4])
            ps.pose.orientation.z = float(x[5])
            ps.pose.orientation.w = float(x[6])
            ref_msg.poses.append(ps)
        self._cached_ref_path = ref_msg

        # ── EE 规划路径（Pinocchio FK） ───────────────────────────────────
        if self.arm_enabled:
            try:
                rm   = self.mpc.robot_model
                data = rm.createData()
                eid  = self.mpc.ee_frame_id
                ee_msg = RosPath()
                ee_msg.header.stamp    = stamp
                ee_msg.header.frame_id = "map"
                for x in self.x_plan:
                    nq = rm.nq
                    q  = np.asarray(x[:nq], dtype=float)
                    pin.forwardKinematics(rm, data, q)
                    pin.updateFramePlacements(rm, data)
                    t_ee = data.oMf[eid].translation
                    R_ee = data.oMf[eid].rotation
                    quat_ee = pin.Quaternion(R_ee)
                    ps = PoseStamped()
                    ps.header = ee_msg.header
                    ps.pose.position.x    = float(t_ee[0])
                    ps.pose.position.y    = float(t_ee[1])
                    ps.pose.position.z    = float(t_ee[2])
                    ps.pose.orientation.x = float(quat_ee.x)
                    ps.pose.orientation.y = float(quat_ee.y)
                    ps.pose.orientation.z = float(quat_ee.z)
                    ps.pose.orientation.w = float(quat_ee.w)
                    ee_msg.poses.append(ps)
                if ee_msg.poses:
                    self._cached_ee_plan_path = ee_msg
            except Exception as e:
                rospy.logwarn(f"EE planned path build failed: {e}")

        n_ref = len(self._cached_ref_path.poses) if self._cached_ref_path else 0
        n_ee  = len(self._cached_ee_plan_path.poses) if self._cached_ee_plan_path else 0
        rospy.loginfo(
            f"Viz caches built: UAV ref={n_ref} pts"
            + (f", EE plan={n_ee} pts" if n_ee else "")
        )

    def _stage_mpc_planned_path(self, xs_guess: List[np.ndarray]) -> None:
        """将本次 MPC solver 规划的 horizon 状态序列暂存为 RosPath（不立即发布）。"""
        if not xs_guess:
            return
        msg = RosPath()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = "map"
        for x in xs_guess:
            if len(x) < 7:
                continue
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x    = float(x[0])
            ps.pose.position.y    = float(x[1])
            ps.pose.position.z    = float(x[2])
            ps.pose.orientation.x = float(x[3])
            ps.pose.orientation.y = float(x[4])
            ps.pose.orientation.z = float(x[5])
            ps.pose.orientation.w = float(x[6])
            msg.poses.append(ps)
        self._staged_mpc_path = msg if msg.poses else None

    def _append_uav_actual_path_point(self) -> None:
        """将当前 UAV base 位姿追加到实际路径缓冲。"""
        if self.state is None or len(self.state) < 7:
            return
        pos = np.array(self.state[0:3], dtype=float)
        if (self.viz_path_min_pos_step > 0.0
                and self._last_uav_path_pos_sample is not None
                and np.linalg.norm(pos - self._last_uav_path_pos_sample) < self.viz_path_min_pos_step):
            return
        self._last_uav_path_pos_sample = pos.copy()

        ps = PoseStamped()
        ps.header.stamp    = rospy.Time.now()
        ps.header.frame_id = "map"
        ps.pose.position.x    = float(self.state[0])
        ps.pose.position.y    = float(self.state[1])
        ps.pose.position.z    = float(self.state[2])
        ps.pose.orientation.x = float(self.state[3])
        ps.pose.orientation.y = float(self.state[4])
        ps.pose.orientation.z = float(self.state[5])
        ps.pose.orientation.w = float(self.state[6])
        self._uav_actual_path_msg.header.stamp    = ps.header.stamp
        self._uav_actual_path_msg.header.frame_id = "map"
        self._uav_actual_path_msg.poses.append(ps)
        if len(self._uav_actual_path_msg.poses) > self.uav_actual_path_max_len:
            self._uav_actual_path_msg.poses = (
                self._uav_actual_path_msg.poses[-self.uav_actual_path_max_len:]
            )

    def _append_ee_actual_path_point(self) -> None:
        """将当前 EE 位姿（Pinocchio FK）追加到实际路径缓冲。"""
        try:
            rm   = self.mpc.robot_model
            data = rm.createData()
            eid  = self.mpc.ee_frame_id
            nq   = rm.nq
            q    = np.asarray(self.state[:nq], dtype=float)
            pin.forwardKinematics(rm, data, q)
            pin.updateFramePlacements(rm, data)
            t_ee   = data.oMf[eid].translation
            R_ee   = data.oMf[eid].rotation
            quat_e = pin.Quaternion(R_ee)
        except Exception:
            return

        pos = np.array(t_ee, dtype=float)
        if (self.viz_path_min_pos_step > 0.0
                and self._last_ee_path_pos_sample is not None
                and np.linalg.norm(pos - self._last_ee_path_pos_sample) < self.viz_path_min_pos_step):
            return
        self._last_ee_path_pos_sample = pos.copy()

        ps = PoseStamped()
        ps.header.stamp    = rospy.Time.now()
        ps.header.frame_id = "map"
        ps.pose.position.x    = float(t_ee[0])
        ps.pose.position.y    = float(t_ee[1])
        ps.pose.position.z    = float(t_ee[2])
        ps.pose.orientation.x = float(quat_e.x)
        ps.pose.orientation.y = float(quat_e.y)
        ps.pose.orientation.z = float(quat_e.z)
        ps.pose.orientation.w = float(quat_e.w)
        self._ee_actual_path_msg.header.stamp    = ps.header.stamp
        self._ee_actual_path_msg.header.frame_id = "map"
        self._ee_actual_path_msg.poses.append(ps)
        if len(self._ee_actual_path_msg.poses) > self.ee_actual_path_max_len:
            self._ee_actual_path_msg.poses = (
                self._ee_actual_path_msg.poses[-self.ee_actual_path_max_len:]
            )

    def _maybe_flush_viz_publishes(self, xs_next: Optional[np.ndarray]) -> None:
        """
        限速批量发布所有 RViz 路径话题（默认 15 Hz），避免在高频控制回调中
        每步序列化大型 Path 消息。同时在此处更新 WholeBody 可视化。
        """
        now = time.time()
        if now - self._last_viz_pub_wall_t < self._viz_path_min_period:
            return
        self._last_viz_pub_wall_t = now
        stamp = rospy.Time.now()

        # 1. 参考轨迹（固定，latch 已处理；每次刷新 stamp 后重发保持活跃）
        if self._cached_ref_path is not None:
            self._cached_ref_path.header.stamp = stamp
            self.ref_traj_path_pub.publish(self._cached_ref_path)

        # 2. MPC horizon 预测路径
        if self._staged_mpc_path is not None and self._staged_mpc_path.poses:
            self._staged_mpc_path.header.stamp = stamp
            self.mpc_planned_path_pub.publish(self._staged_mpc_path)

        # 3. UAV 实际路径
        if self._uav_actual_path_msg.poses:
            self._uav_actual_path_msg.header.stamp = stamp
            self.uav_actual_path_pub.publish(self._uav_actual_path_msg)

        # 4. EE 规划 & 实际路径
        if self.arm_enabled:
            if self._cached_ee_plan_path is not None and self._cached_ee_plan_path.poses:
                self._cached_ee_plan_path.header.stamp = stamp
                self.ee_planned_path_pub.publish(self._cached_ee_plan_path)
            if self._ee_actual_path_msg.poses:
                self._ee_actual_path_msg.header.stamp = stamp
                self.ee_actual_path_pub.publish(self._ee_actual_path_msg)

        # 5. WholeBody 可视化（eagle_mpc_viz）
        if _WHOLEBODY_VIZ_OK and self._wb_current_pub is not None:
            try:
                nq  = self.mpc.nq
                q   = self.state[:nq].tolist()
                v   = self.state[nq:].tolist()
                tau = self._u_hold.tolist()
                self._wb_current_pub.publish(stamp, q, v, tau)
                if xs_next is not None:
                    q_ref = xs_next[:nq].tolist()
                    v_ref = xs_next[nq:].tolist()
                    self._wb_target_pub.publish(stamp, q_ref, v_ref, tau)
                if self._staged_mpc_path is not None and self._xs_guess is not None:
                    ts = [float(i) * self.dt_mpc for i in range(len(self._xs_guess))]
                    qs = [np.asarray(x[:nq], dtype=float).tolist() for x in self._xs_guess]
                    vs = [np.asarray(x[nq:], dtype=float).tolist() for x in self._xs_guess]
                    self._wb_planned_pub.publish(ts, qs, vs)
            except Exception as e:
                rospy.logdebug_throttle(5.0, f"WholeBody viz publish error: {e}")

    def _clear_actual_paths(self) -> None:
        """清空实际路径缓冲（reset 时调用）。"""
        self._uav_actual_path_msg.poses = []
        self._ee_actual_path_msg.poses  = []
        self._last_uav_path_pos_sample  = None
        self._last_ee_path_pos_sample   = None
        empty = RosPath()
        empty.header.stamp    = rospy.Time.now()
        empty.header.frame_id = "map"
        self.uav_actual_path_pub.publish(empty)
        if self.arm_enabled:
            self.ee_actual_path_pub.publish(empty)

    # =========================================================================
    # 工具方法
    # =========================================================================

    def _hover_thrust_cmd(self) -> np.ndarray:
        """悬停时各转子推力（等推力悬停）。"""
        mass = float(pin.computeTotalMass(self.mpc.robot_model))
        T_hover = mass * 9.81 / self._n_rotors
        nu = self.mpc.nu
        u = np.zeros(nu, dtype=float)
        u[: self._n_rotors] = T_hover
        return u

    def _default_reg_target(self) -> np.ndarray:
        """
        Regulation 默认参考策略：
        - 未开始 tracking（traj_finished=False）: 轨迹起点 x_plan[0]
        - tracking 结束后（traj_finished=True）: 轨迹终点 x_plan[-1]
        """
        if self.x_plan is None or len(self.x_plan) == 0:
            return self._match_state_dim(np.asarray(self.state, dtype=float).copy(), zero_vel=True)
        idx = -1 if bool(self.traj_finished) else 0
        x_tgt = np.asarray(self.x_plan[idx], dtype=float).copy()
        # Regulation is a point stabilization task; clear velocity targets.
        return self._match_state_dim(x_tgt, zero_vel=True)

    # =========================================================================
    # 数据录制
    # =========================================================================

    def _record(self, t: float, u: np.ndarray, xs_next: Optional[np.ndarray], solve_ms: float):
        s = self.state
        nq = self.mpc.nq
        nj = self.arm_joint_number
        self.recorded_data["time"].append(t)
        self.recorded_data["position"].append(s[0:3].tolist())
        self.recorded_data["velocity"].append(s[nq : nq + 3].tolist())
        self.recorded_data["angular_velocity"].append(s[nq + 3 : nq + 6].tolist())
        self.recorded_data["orientation"].append(s[3:7].tolist())
        self.recorded_data["arm_joint_positions"].append(s[7 : 7 + nj].tolist())
        self.recorded_data["arm_joint_velocities"].append(s[-nj:].tolist() if nj > 0 else [])
        self.recorded_data["mpc_control"].append(u.tolist())
        self.recorded_data["mpc_solve_time"].append(solve_ms)

        x_ref = interp_full_state_piecewise(
            float(self.t_plan[0]) + t, self.t_plan, self.x_plan, self.mpc.robot_model
        )
        self.recorded_data["reference_position"].append(x_ref[0:3].tolist())
        self.recorded_data["reference_orientation"].append(x_ref[3:7].tolist())
        self.recorded_data["reference_velocity"].append(x_ref[nq : nq + 3].tolist())
        self.recorded_data["reference_angular_velocity"].append(x_ref[nq + 3 : nq + 6].tolist())
        self.recorded_data["reference_arm_positions"].append(x_ref[7 : 7 + nj].tolist())

    # =========================================================================
    # ROS 服务
    # =========================================================================

    def _svc_start_tracking(self, req) -> TriggerResponse:
        if self.px4_state.mode != "OFFBOARD":
            return TriggerResponse(False, "Must be in OFFBOARD mode")
        if not self.px4_state.armed:
            return TriggerResponse(False, "Must be armed")
        if self.trajectory_started:
            return TriggerResponse(False, "Tracking already started")
        # 退出 regulation 模式，切换到轨迹跟踪
        self._regulating = False
        self._reg_target_locked = False
        self.trajectory_started = True
        self.traj_finished = False
        self.controller_start_time = rospy.Time.now()
        self.trajectory_name = str(rospy.get_param("~trajectory_name", self.trajectory_name))
        self._xs_guess = None
        self._us_guess = None
        self._clear_recorded_data()
        self._active_tracking_tag = self._compose_tracking_file_tag()
        self.recording_enabled = True
        self._clear_actual_paths()   # 重置实际路径，使实际轨迹从跟踪起点开始
        rospy.loginfo("Tracking started! (regulation mode off)")
        return TriggerResponse(True, "Tracking started")

    def _svc_stop_tracking(self, req) -> TriggerResponse:
        self.trajectory_started = False
        self.traj_finished = True
        self.recording_enabled = False
        # 停止跟踪后切回 regulation 模式，目标固定为轨迹终点。
        self._reg_target   = self._default_reg_target()
        self._reg_target_locked = False
        self._reg_xs_guess = None
        self._reg_us_guess = None
        self._regulating   = True
        tgt = self._reg_target
        rospy.loginfo(
            f"Tracking stopped. Switched to regulation at trajectory end-point "
            f"x={tgt[0]:.2f} y={tgt[1]:.2f} z={tgt[2]:.2f}"
        )
        try:
            tag = self._active_tracking_tag or self._compose_tracking_file_tag()
            csv_path, txt_path = self._save_tracking_csv_and_stats(tag)
            self._active_tracking_tag = None
            return TriggerResponse(
                True,
                f"Tracking stopped, holding position. Saved: {csv_path.name}, {txt_path.name}",
            )
        except Exception as e:
            return TriggerResponse(
                False,
                f"Tracking stopped, but save failed: {e}",
            )

    def _svc_save_data(self, req) -> TriggerResponse:
        try:
            out_dir = Path(__file__).resolve().parent / "results" / "suite_tracking"
            out_dir.mkdir(parents=True, exist_ok=True)
            filepath = out_dir / f"tracking_{self._safe_token(self.controller_mode)}.npz"
            save_data = {}
            for k, v in self.recorded_data.items():
                if v:
                    save_data[k] = np.array(v)
                else:
                    save_data[k] = np.array([])
            np.savez_compressed(str(filepath), **save_data)
            tag = self._active_tracking_tag or self._compose_tracking_file_tag()
            csv_path, txt_path = self._save_tracking_csv_and_stats(tag)
            rospy.loginfo(f"Data saved: {filepath} | {csv_path} | {txt_path}")
            return TriggerResponse(True, f"Saved to {filepath}, {csv_path}, {txt_path}")
        except Exception as e:
            return TriggerResponse(False, f"Save failed: {e}")

    def _svc_update_trajectory(self, req) -> TriggerResponse:
        """
        /update_trajectory 服务：
        重新加载轨迹并重建 MPC/缓存，不重启节点进程。
        """
        try:
            with self._thread_lock:
                self.trajectory_name = str(rospy.get_param("~trajectory_name", self.trajectory_name))
                self._load_trajectory()
                self._build_mpc()
                self.arm_joint_number = self.mpc.robot_model.nq - 7
                self._rebuild_cached_viz_paths()
                self._u_hold = self._hover_thrust_cmd()

                # Reset warm starts under the new trajectory/model.
                self._xs_guess = None
                self._us_guess = None
                self._reg_xs_guess = None
                self._reg_us_guess = None

                # Keep safe: switch to regulation at updated default target.
                self.trajectory_started = False
                self.traj_finished = False
                self.controller_start_time = None
                self._reg_target = self._default_reg_target()
                self._reg_target_locked = False
                self._regulating = True
                self._clear_actual_paths()

            msg = (
                f"Trajectory reloaded: {len(self.t_plan)} points, "
                f"mode={self.controller_mode}, regulation target reset."
            )
            rospy.loginfo(f"[update_trajectory] {msg}")
            return TriggerResponse(True, msg)
        except Exception as e:
            rospy.logerr(f"[update_trajectory] failed: {e}")
            return TriggerResponse(False, f"Error: {e}")

    # ── reset_to_initial ────────────────────────────────────────────────────

    def _svc_reset_to_initial(self, req) -> TriggerResponse:
        """
        /reset_to_initial 服务：将 regulation 目标重置为 x_plan[0]，
        切换到 regulation 模式（MPC 驱动机器人回到轨迹初始状态）。
        持续保持 regulation 模式直到 /start_tracking 显式调用。
        """
        if self.x_plan is None or self.mpc is None:
            return TriggerResponse(False, "Trajectory / MPC not loaded yet.")

        # ── 1. 停止轨迹跟踪，重置标志 ───────────────────────────────────────
        self.trajectory_started = False
        self.traj_finished = False
        self.controller_start_time = None
        self.recording_enabled = False
        self._xs_guess = None
        self._us_guess = None
        self._u_hold = self._hover_thrust_cmd()

        # ── 2. 将 regulation 目标设为 x_plan[0]，切换到 regulation 模式 ──────
        self._reg_target   = self._default_reg_target()
        self._reg_target_locked = False
        self._reg_xs_guess = None
        self._reg_us_guess = None
        self._regulating   = True

        # ── 3. 清空 RViz 实际路径缓冲 ────────────────────────────────────────
        self._clear_actual_paths()

        x0 = self._reg_target
        msg = (
            f"Reset started: regulation driving to x_plan[0]={x0[:3].tolist()}. "
            f"Call /start_tracking to begin trajectory tracking."
        )
        rospy.loginfo(f"[reset_to_initial] {msg}")
        return TriggerResponse(True, msg)

    def _svc_set_regulation_target(self, req) -> TriggerResponse:
        """
        /set_regulation_target 服务：从 ROS param ~regulation_target_data 读取目标，
        设置为新的 regulation 目标并切换到 regulation 模式。
        param 格式：[x, y, z, yaw_deg, j1_deg, j2_deg]（6 个浮点数）
        """
        if self.mpc is None:
            return TriggerResponse(False, "MPC not initialized yet.")
        try:
            data = rospy.get_param("~regulation_target_data", None)
            if data is None:
                return TriggerResponse(False, "Param ~regulation_target_data not set.")
            data = [float(v) for v in data]
            if len(data) < 6:
                return TriggerResponse(False, f"Expected 6 values, got {len(data)}.")
            new_target = self._make_reg_target(
                x=data[0], y=data[1], z=data[2],
                yaw_deg=data[3], j1_deg=data[4], j2_deg=data[5],
            )
            self._reg_target   = new_target
            self._reg_target_locked = True
            self._reg_xs_guess = None
            self._reg_us_guess = None
            if not self.trajectory_started:
                self._regulating = True
            status = "regulation 已激活" if self._regulating else "tracking 中（下次 stop 后生效）"
            msg = (
                f"Regulation target set: x={data[0]:.2f} y={data[1]:.2f} z={data[2]:.2f} "
                f"yaw={data[3]:.1f}° j1={data[4]:.1f}° j2={data[5]:.1f}° | {status}"
            )
            rospy.loginfo(f"[set_regulation_target] {msg}")
            return TriggerResponse(True, msg)
        except Exception as e:
            return TriggerResponse(False, f"Error: {e}")

    def _svc_update_controller_params(self, req) -> TriggerResponse:
        """
        /update_controller_params 服务：
        从 ROS param ~controller_update_data 读取控制器参数并在线更新。
        支持切换 controller_mode 以及更新 MPC / geometric 参数，无需重启节点。
        """
        try:
            cfg = rospy.get_param("~controller_update_data", None)
            if not isinstance(cfg, dict):
                return TriggerResponse(False, "Param ~controller_update_data must be a dict.")

            with self._thread_lock:
                new_mode = str(cfg.get("controller_mode", self.controller_mode)).strip()
                allowed = {"croc_full_state", "croc_ee_pose", "px4", "geometric"}
                if new_mode not in allowed:
                    return TriggerResponse(False, f"Invalid controller_mode: {new_mode}")
                self.controller_mode = new_mode

                # 通用参数
                self.control_rate = float(cfg.get("control_rate", self.control_rate))
                self.control_rate = max(1.0, self.control_rate)
                self.dt_control = 1.0 / self.control_rate

                self.dt_mpc = float(cfg.get("dt_mpc", self.dt_mpc))
                self.horizon = int(cfg.get("horizon", self.horizon))
                self.mpc_max_iter = int(cfg.get("mpc_max_iter", self.mpc_max_iter))
                self.dt_mpc = max(1e-3, self.dt_mpc)
                self.horizon = max(2, self.horizon)
                self.mpc_max_iter = max(1, self.mpc_max_iter)

                # full-state / EE 权重
                self.w_state_track = float(cfg.get("w_state_track", self.w_state_track))
                self.w_state_reg = float(cfg.get("w_state_reg", self.w_state_reg))
                self.w_control = float(cfg.get("w_control", self.w_control))
                self.w_terminal_track = float(cfg.get("w_terminal_track", self.w_terminal_track))
                self.w_pos = float(cfg.get("w_pos", self.w_pos))
                self.w_att = float(cfg.get("w_att", self.w_att))
                self.w_joint = float(cfg.get("w_joint", self.w_joint))
                self.w_vel = float(cfg.get("w_vel", self.w_vel))
                self.w_omega = float(cfg.get("w_omega", self.w_omega))
                self.w_joint_vel = float(cfg.get("w_joint_vel", self.w_joint_vel))
                self.w_u_thrust = float(cfg.get("w_u_thrust", self.w_u_thrust))
                self.w_u_joint_torque = float(cfg.get("w_u_joint_torque", self.w_u_joint_torque))

                self.ee_w_pos = float(cfg.get("ee_w_pos", self.ee_w_pos))
                self.ee_w_rot_rp = float(cfg.get("ee_w_rot_rp", self.ee_w_rot_rp))
                self.ee_w_rot_yaw = float(cfg.get("ee_w_rot_yaw", self.ee_w_rot_yaw))
                self.ee_w_vel_lin = float(cfg.get("ee_w_vel_lin", self.ee_w_vel_lin))
                self.ee_w_vel_ang_rp = float(cfg.get("ee_w_vel_ang_rp", self.ee_w_vel_ang_rp))
                self.ee_w_vel_ang_yaw = float(cfg.get("ee_w_vel_ang_yaw", self.ee_w_vel_ang_yaw))
                self.ee_w_u = float(cfg.get("ee_w_u", self.ee_w_u))
                self.ee_w_terminal = float(cfg.get("ee_w_terminal", self.ee_w_terminal))

                # geometric 增益
                self.geo_kp_pos = float(cfg.get("geo_kp_pos", self.geo_kp_pos))
                self.geo_kd_vel = float(cfg.get("geo_kd_vel", self.geo_kd_vel))
                self.geo_kR = float(cfg.get("geo_kR", self.geo_kR))
                self.geo_kOmega = float(cfg.get("geo_kOmega", self.geo_kOmega))
                self.geo_max_tilt_deg = float(cfg.get("geo_max_tilt_deg", self.geo_max_tilt_deg))

                # 限幅参数（可选）
                self.max_angular_velocity = float(
                    cfg.get("max_angular_velocity", self.max_angular_velocity)
                )
                self.min_thrust_cmd = float(cfg.get("min_thrust_cmd", self.min_thrust_cmd))
                self.max_thrust_cmd = float(cfg.get("max_thrust_cmd", self.max_thrust_cmd))

                # 若切到 EE 模式，确保 EE 参考存在
                if self.controller_mode == "croc_ee_pose":
                    self._build_ee_ref_from_full_state()

                # 统一重建 MPC（含模式切换与参数更新）
                self._build_mpc()
                self.arm_joint_number = self.mpc.robot_model.nq - 7
                self._u_hold = self._hover_thrust_cmd()

                # 清空 warm-start，避免新参数下使用旧缓存
                self._xs_guess = None
                self._us_guess = None
                self._reg_xs_guess = None
                self._reg_us_guess = None

                # 若维度变化导致目标不兼容，回退到 x_plan[0]
                if self._reg_target.shape[0] != self.mpc.nq + self.mpc.nv:
                    self._reg_target = self._default_reg_target()
                    self._reg_target_locked = False

            msg = (
                f"Controller params updated: mode={self.controller_mode}, "
                f"dt_mpc={self.dt_mpc:.3f}, H={self.horizon}, iter={self.mpc_max_iter}, "
                f"rate={self.control_rate:.1f}Hz"
            )
            rospy.loginfo(f"[update_controller_params] {msg}")
            return TriggerResponse(True, msg)
        except Exception as e:
            rospy.logerr(f"[update_controller_params] failed: {e}")
            return TriggerResponse(False, f"Error: {e}")

    def _publish_initial_joints(self, x0: np.ndarray, nj: int) -> None:
        """将 x0 中的初始关节角发布到仿真/实机关节控制话题。"""
        from std_msgs.msg import Float64

        if nj < 1:
            return

        ref_j = x0[7 : 7 + nj]

        # 实机 desired_joint_states
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [f"joint_{i+1}" for i in range(nj)]
        joint_msg.position = list(ref_j)
        joint_msg.velocity = [0.0] * nj
        joint_msg.effort = [0.0] * nj
        self.arm_control_pub.publish(joint_msg)

        # 仿真单关节话题
        if self.use_simulation:
            if nj >= 1:
                self.joint1_pub.publish(Float64(float(ref_j[0])))
            if nj >= 2:
                self.joint2_pub.publish(Float64(float(ref_j[1])))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        node = SuiteTrackingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("SuiteTrackingController terminated.")
