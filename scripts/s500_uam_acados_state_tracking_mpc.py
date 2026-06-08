#!/usr/bin/env python3
"""
Acados NMPC closed-loop tracking along a full-state plan (t_plan, x_plan).

Cost weights are driven by the same GUI parameters as Crocoddyl full-state tracking
(w_pos, w_att, w_joint, w_vel, w_omega, w_joint_vel, w_control, w_u_thrust,
w_u_joint_torque, w_state_track, w_terminal_track).

Plant integration reuses the CasADi RK4 helper from s500_uam_closed_loop_plant /
s500_uam_ee_snap_tracking_mpc (same as Acados EE tracking).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from acados_runtime import preload_acados_shared_libs

    preload_acados_shared_libs()
    from acados_template import AcadosOcp, AcadosOcpSolver

    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

try:
    import casadi as ca
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError as e:
    PINOCCHIO_AVAILABLE = False
    _pin_err = e

try:
    from pinocchio import casadi as cpin

    from s500_uam_acados_model import _quat_prod, _quat_to_R, load_s500_config
    from s500_uam_acados_trajectory import STATE_LIMITS

    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    _deps_err = e

from s500_uam_crocoddyl_state_tracking_mpc import interp_full_state_piecewise
from s500_uam_closed_loop_plant import CasadiRK4Plant
from s500_uam_ee_snap_tracking_mpc import (
    EE_FRAME_NAME,
    REPO_ROOT,
    _acados_cpu_time_s,
    _acados_nlp_iterations,
    _make_f_expl_fun,
    hover_thrust_controls,
    rollout_nominal_trajectory,
    set_solver_initial_guess,
    shift_solver_initial_guess,
)

SCRIPT_DIR = Path(__file__).resolve().parent


class _AcadosTrackMpcShim:
    """Minimal holder so ``crocoddyl_closed_loop_to_ee_tracking_res`` can reuse EE plots."""

    def __init__(self, robot_model: pin.Model, ee_frame_id: int):
        self.robot_model = robot_model
        self._planner = type("_PlannerShim", (), {"ee_frame_id": int(ee_frame_id)})()


def croc_tracking_weights_to_W_R(
    *,
    n_arm: int = 2,
    w_pos: float = 1.0,
    w_att: float = 1.0,
    w_joint: float = 1.0,
    w_vel: float = 1.0,
    w_omega: float = 1.0,
    w_joint_vel: float = 1.0,
    w_control: float = 1e-3,
    w_u_thrust: float = 1.0,
    w_u_joint_torque: float = 1.0,
    w_state_track: float = 10.0,
    w_terminal_track: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map Crocoddyl full-state tracking GUI weights to Acados NONLINEAR_LS W / W_e / R.

    State cost layout (robot-aware, matches ``_state_to_y_np``):
    [pos(3), yaw, roll, pitch, j*(n_arm), v_lin(3), ω(3), j̇*(n_arm)].
    """
    na = int(n_arm)
    s = float(w_state_track)
    diag = [float(w_pos) * s] * 3 + [float(w_att) * s] * 3
    diag += [float(w_joint) * s] * na
    diag += [float(w_vel) * s] * 3 + [float(w_omega) * s] * 3
    diag += [float(w_joint_vel) * s] * na
    W_state = np.diag(diag)
    r_thrust = float(w_control) * float(w_u_thrust)
    r_torque = float(w_control) * float(w_u_joint_torque) * 10000.0
    R = np.diag([r_thrust] * 4 + [r_torque] * na)
    W_e = W_state * float(w_terminal_track)
    return W_state, R, W_e


def _coerce_state_limits(state_limits: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Merge user/GUI overrides with ``STATE_LIMITS`` defaults."""
    out = {k: float(v) for k, v in STATE_LIMITS.items()}
    if state_limits:
        for key in out:
            if key in state_limits and state_limits[key] is not None:
                out[key] = float(state_limits[key])
    return out


def _build_robot_state_bounds(
    *,
    n_arm: int,
    control_mode: str,
    state_limits: Dict[str, float],
    min_thrust: float,
    max_thrust: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Box bounds on the MPC state vector (robot q/v [+ u_act for act1])."""
    v_max = float(state_limits["v_max"])
    om_max = float(state_limits["omega_max"])
    j_max = float(state_limits["j_angle_max"])
    jv_max = float(state_limits["j_vel_max"])
    na = int(n_arm)

    robot_lbx = np.concatenate(
        [
            np.array([-8.0, -8.0, 0.05, -1.0, -1.0, -1.0, -1.0]),
            np.array([-j_max] * na),
            np.array([-v_max, -v_max, -v_max, -om_max, -om_max, -om_max]),
            np.array([-jv_max] * na),
        ]
    )
    robot_ubx = np.concatenate(
        [
            np.array([8.0, 8.0, 6.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([j_max] * na),
            np.array([v_max, v_max, v_max, om_max, om_max, om_max]),
            np.array([jv_max] * na),
        ]
    )
    if control_mode == "actuator_first_order":
        uact_lbx = np.concatenate([np.full(4, min_thrust), np.full(2, -2.0)])
        uact_ubx = np.concatenate([np.full(4, max_thrust), np.full(2, 2.0)])
        return np.concatenate([robot_lbx, uact_lbx]), np.concatenate([robot_ubx, uact_ubx])
    return robot_lbx, robot_ubx


def _apply_solver_state_bounds(
    solver: "AcadosOcpSolver", N: int, lbx: np.ndarray, ubx: np.ndarray
) -> None:
    """Push updated state box bounds to shooting stages 0..N-1 (GUI-tunable limits).

    Terminal stage N has no ``idxbx`` box constraints in this OCP (dimension 0), so skip it.
    """
    lbx = np.asarray(lbx, dtype=float).flatten()
    ubx = np.asarray(ubx, dtype=float).flatten()
    for i in range(int(N)):
        solver.constraints_set(i, "lbx", lbx, api="new")
        solver.constraints_set(i, "ubx", ubx, api="new")


def _default_urdf_path() -> str:
    return str(REPO_ROOT / "models" / "urdf" / "s500_uam_simple.urdf")


def _infer_urdf_path(x_plan: np.ndarray, urdf_path: Optional[str] = None) -> str:
    if urdf_path is not None:
        return str(urdf_path)
    ncol = int(x_plan.shape[1]) if getattr(x_plan, "ndim", 1) == 2 else int(x_plan.size)
    if ncol <= 13:
        return str(REPO_ROOT / "models" / "urdf" / "s500_simple.urdf")
    return _default_urdf_path()


def _build_robot_acados_model(urdf_path: str):
    """Robot-aware AcadosModel: s500 (nq=7, nu=4) or s500_uam (nq=9, nu=6)."""
    from acados_template import AcadosModel

    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    cfg = load_s500_config()
    platform = cfg["platform"]
    cm_cf = platform["cm"] / platform["cf"]
    rotors = platform["$rotors"]

    nq, nv = model.nq, model.nv
    n_arm = nq - 7
    if n_arm < 0:
        raise ValueError(f"Unexpected nq={nq} (<7)")
    n_thrust = 4
    nu = n_thrust + n_arm

    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv)
    u = ca.SX.sym("u", nu)

    thrusts = u[:n_thrust]
    arm_tau = u[n_thrust:] if n_arm > 0 else None

    Fz = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
    Mx = My = Mz = 0.0
    for i, r in enumerate(rotors):
        pos = r["translation"]
        spin = r["spin_direction"][0]
        T = thrusts[i]
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T
    tau_base = ca.vertcat(0, 0, Fz, Mx, My, Mz)
    tau = ca.vertcat(tau_base, arm_tau) if n_arm > 0 else tau_base

    a = cpin.aba(cmodel, cdata, q, v, tau)

    quat = q[3:7]
    v_lin = v[:3]
    v_ang = v[3:6]
    R = _quat_to_R(quat)
    pos_dot = ca.mtimes(R, v_lin)
    quat_dot = 0.5 * _quat_prod(quat, ca.vertcat(v_ang[0], v_ang[1], v_ang[2], 0))
    if n_arm > 0:
        q_dot = ca.vertcat(pos_dot, quat_dot, v[6 : 6 + n_arm])
    else:
        q_dot = ca.vertcat(pos_dot, quat_dot)

    x = ca.vertcat(q, v)
    x_dot = ca.vertcat(q_dot, a)

    acados_model = AcadosModel()
    acados_model.name = "s500_uam_track" if n_arm > 0 else "s500_track"
    acados_model.x = x
    acados_model.u = u
    acados_model.xdot = ca.SX.sym("xdot", x.rows())
    acados_model.f_impl_expr = acados_model.xdot - x_dot
    acados_model.f_expl_expr = x_dot
    return acados_model, model, nq, nv, nu


def _build_cost_y_exprs(x_sym, u_sym, nq: int, n_arm: int):
    quat = x_sym[3:7]
    roll, pitch, yaw = _quat_to_euler_zyx_ca(quat)
    pieces = [x_sym[0:3], ca.vertcat(yaw, roll, pitch)]
    if n_arm > 0:
        pieces.append(x_sym[7 : 7 + n_arm])
    pieces.append(x_sym[nq : nq + 6])
    if n_arm > 0:
        pieces.append(x_sym[nq + 6 : nq + 6 + n_arm])
    cost_y_e = ca.vertcat(*pieces)
    cost_y = ca.vertcat(cost_y_e, u_sym)
    return cost_y, cost_y_e


def create_full_state_tracking_mpc_solver(
    N: int,
    dt: float,
    *,
    w_pos: float = 1.0,
    w_att: float = 1.0,
    w_joint: float = 1.0,
    w_vel: float = 1.0,
    w_omega: float = 1.0,
    w_joint_vel: float = 1.0,
    w_control: float = 1e-3,
    w_u_thrust: float = 1.0,
    w_u_joint_torque: float = 1.0,
    w_state_track: float = 10.0,
    w_terminal_track: float = 100.0,
    max_iter: int = 40,
    control_mode: str = "direct",
    urdf_path: Optional[str] = None,
    state_limits: Optional[Dict[str, float]] = None,
) -> tuple:
    if not ACADOS_AVAILABLE:
        raise ImportError("acados_template not installed")
    if not PINOCCHIO_AVAILABLE:
        raise ImportError(f"pinocchio/casadi required: {_pin_err}")
    if not DEPS_OK:
        raise ImportError(f"project deps: {_deps_err}")

    urdf_path = str(urdf_path or _default_urdf_path())
    n_arm_probe = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer()).nq - 7

    if control_mode == "direct":
        acados_model, pin_model, nq, nv, nu = _build_robot_acados_model(urdf_path)
        robot_tag = "uam" if n_arm_probe > 0 else "s500"
        code_subdir = f"s500_{robot_tag}_full_state_track_mpc"
    elif control_mode == "actuator_first_order":
        if n_arm_probe <= 0:
            raise ValueError("actuator_first_order is only supported for s500_uam (with arm)")
        from s500_uam_acados_model import build_acados_model_actuator_first_order

        acados_model, pin_model, nq, nv, nu, _meta = build_acados_model_actuator_first_order(
            urdf_path=urdf_path,
        )
        code_subdir = "s500_uam_full_state_track_mpc_act1"
    else:
        raise ValueError("control_mode must be 'direct' or 'actuator_first_order'")

    n_arm = nq - 7
    W_state, R, W_e = croc_tracking_weights_to_W_R(
        n_arm=n_arm,
        w_pos=w_pos,
        w_att=w_att,
        w_joint=w_joint,
        w_vel=w_vel,
        w_omega=w_omega,
        w_joint_vel=w_joint_vel,
        w_control=w_control,
        w_u_thrust=w_u_thrust,
        w_u_joint_torque=w_u_joint_torque,
        w_state_track=w_state_track,
        w_terminal_track=w_terminal_track,
    )

    ocp = AcadosOcp()
    ocp.model = acados_model
    nx = nq + nv + (6 if control_mode == "actuator_first_order" else 0)

    cost_y, cost_y_e = _build_cost_y_exprs(ocp.model.x, ocp.model.u, nq, n_arm)
    ocp.model.cost_y_expr = cost_y
    ocp.model.cost_y_expr_e = cost_y_e
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))
    ocp.cost.W_e = W_e
    ny = int(cost_y.shape[0])
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(cost_y_e.shape[0])

    ocp.dims.N = int(N)
    ocp.solver_options.tf = float(N) * float(dt)
    ocp.solver_options.nlp_solver_max_iter = int(max_iter)
    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = int(N)

    cfg = load_s500_config()
    platform = cfg["platform"]
    min_thrust = platform["min_thrust"]
    max_thrust = platform["max_thrust"]
    limits = _coerce_state_limits(state_limits)
    om_max = limits["omega_max"]
    j_max = limits["j_angle_max"]
    robot_lbx, robot_ubx = _build_robot_state_bounds(
        n_arm=n_arm,
        control_mode=control_mode,
        state_limits=limits,
        min_thrust=min_thrust,
        max_thrust=max_thrust,
    )

    if control_mode == "direct":
        ocp.constraints.lbu = np.array([min_thrust] * 4 + [-2.0] * n_arm)
        ocp.constraints.ubu = np.array([max_thrust] * 4 + [2.0] * n_arm)
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx = robot_lbx
        ocp.constraints.ubx = robot_ubx
    else:
        ocp.constraints.lbu = np.array(
            [-2.0 * om_max, -2.0 * om_max, -2.0 * om_max, 4.0 * min_thrust, -j_max, -j_max]
        )
        ocp.constraints.ubu = np.array(
            [2.0 * om_max, 2.0 * om_max, 2.0 * om_max, 4.0 * max_thrust, j_max, j_max]
        )
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx = robot_lbx
        ocp.constraints.ubx = robot_ubx

    ocp.constraints.idxbu = np.arange(nu)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = (
        "ERK" if control_mode == "actuator_first_order" else "IRK"
    )
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0
    if hasattr(ocp.solver_options, "qp_solver_iter_max"):
        ocp.solver_options.qp_solver_iter_max = max(50, int(max_iter) * 2)

    code_export_dir = REPO_ROOT / "c_generated_code" / code_subdir
    code_export_dir.mkdir(parents=True, exist_ok=True)
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(code_export_dir / "ocp.json")
    ocp.constraints.x0 = np.zeros(nx)

    # 非必要不重新生成代码：generate/build 置 False + check_reuse_possible，acados 用整个
    # OCP 的哈希判断是否可复用已编译代码——权重/N/dt/模型/约束/求解器选项任一改变都会
    # 触发重生成，否则跳过 codegen+build（仅加载已有 .so 并重建 solver，很快）。
    solver = AcadosOcpSolver(
        ocp,
        json_file=str(code_export_dir / "ocp.json"),
        generate=False,
        build=False,
        verbose=False,
        check_reuse_possible=True,
    )
    # 复用已生成代码时，C 代码里 bake 的 W 是上次生成时的值；运行时按当前权重重新下发
    # W / W_e，确保即便跳过 codegen，权重修改也立即生效（不依赖哈希是否覆盖 W）。
    W_full = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))
    for i in range(int(N)):
        solver.cost_set(i, "W", W_full)
    solver.cost_set(int(N), "W", W_e)
    _apply_solver_state_bounds(solver, int(N), robot_lbx, robot_ubx)
    return solver, acados_model, pin_model, nq, nv, nu, limits


def _quat_to_euler_zyx_ca(quat):
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    roll = ca.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = 2 * (qw * qy - qz * qx)
    sinp = ca.fmin(1, ca.fmax(-1, sinp))
    pitch = ca.asin(sinp)
    yaw = ca.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


def _quat_to_euler_zyx_np(qx, qy, qz, qw):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


def _state_to_y_np(x_ref: np.ndarray, nq: int, n_arm: int) -> np.ndarray:
    """Map full state to nonlinear-LS output (no controls). Layout matches ``_build_cost_y_exprs``."""
    x_ref = np.asarray(x_ref, dtype=float).flatten()
    roll, pitch, yaw = _quat_to_euler_zyx_np(x_ref[3], x_ref[4], x_ref[5], x_ref[6])
    parts = [x_ref[0:3], np.array([yaw, roll, pitch], dtype=float)]
    if n_arm > 0:
        parts.append(x_ref[7 : 7 + n_arm])
    parts.append(x_ref[nq : nq + 6])
    if n_arm > 0:
        parts.append(x_ref[nq + 6 : nq + 6 + n_arm])
    return np.concatenate(parts)


def _state_to_yref(x_ref: np.ndarray, u_ref: np.ndarray, nq: int, n_arm: int) -> np.ndarray:
    return np.concatenate(
        [_state_to_y_np(x_ref, nq, n_arm), np.asarray(u_ref, dtype=float).flatten()]
    )


def _state_to_yref_e(x_ref: np.ndarray, nq: int, n_arm: int) -> np.ndarray:
    return _state_to_y_np(x_ref, nq, n_arm)


def _resolve_ee_frame_id(pin_model: "pin.Model") -> int:
    """Return a valid EE frame id. For s500 (no gripper) fall back to the base frame.

    EE channels are blanked downstream for s500, so the exact fallback only needs to be
    a valid index that keeps ``compute_ee_kinematics_along_trajectory`` from crashing.
    """
    nframes = len(pin_model.frames)
    fid = int(pin_model.getFrameId(EE_FRAME_NAME))
    if 0 <= fid < nframes:
        return fid
    for name in ("base_link", "root_joint", "universe"):
        cand = int(pin_model.getFrameId(name))
        if 0 <= cand < nframes:
            return cand
    return max(0, nframes - 1)


def _bodyrate_from_horizon_states(
    x_horizon: Optional[List[np.ndarray]],
    lookahead_s: float,
    dt_mpc: float,
    nq: int,
) -> Optional[np.ndarray]:
    """从 MPC horizon 状态在 lookahead 处线性插值得到机体角速度设定点 (3,)。

    与 ROS run_tracking_controller._bodyrate_from_horizon 等价：horizon 状态位于
    0, dt_mpc, 2·dt_mpc, ...；取 t=lookahead_s 处的体角速度（x[nq+3:nq+6]）按相邻
    格点线性插值。lookahead=dt_mpc 退化为 xs[1]，lookahead=0 取 xs[0]。
    """
    if x_horizon is None or len(x_horizon) < 2 or dt_mpc <= 1e-9:
        return None

    def _rates(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).flatten()
        return np.array([x[nq + 3], x[nq + 4], x[nq + 5]], dtype=float)

    n_seg = len(x_horizon) - 1
    s = min(max(float(lookahead_s) / float(dt_mpc), 0.0), float(n_seg))
    i0 = int(math.floor(s))
    if i0 >= n_seg:
        i0 = n_seg - 1
    frac = s - float(i0)
    if x_horizon[i0].size < nq + 6 or x_horizon[i0 + 1].size < nq + 6:
        return None
    return (1.0 - frac) * _rates(x_horizon[i0]) + frac * _rates(x_horizon[i0 + 1])


def _quat_to_yaw_np(q: np.ndarray) -> float:
    """四元数 [x,y,z,w] → 偏航角 yaw (rad)。"""
    x, y, z, w = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _geometric_baseline_command(
    p: np.ndarray,
    v_world: np.ndarray,
    R: np.ndarray,
    omega_body: np.ndarray,
    p_ref: np.ndarray,
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    yaw_ref: float,
    mass: float,
    gravity: float,
    *,
    kp_pos: float,
    kd_vel: float,
    kR: float,
    kOmega: float,
    max_tilt_deg: float,
) -> tuple:
    """SE3 几何控制律（移植 example/l1_geometric_tracking_sim.geometric_baseline）。

    返回 (thrust_N, body_rate_cmd)：总推力（N）与机体角速度设定点（rad/s）。
    与 ROS run_tracking_controller 的 geometric 一致：位置/速度 PD 得期望比力 a_des，
    构造期望姿态 R_des，姿态误差 e_R 经 kR/kOmega 得体角速度指令。
    """
    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    e_p = np.asarray(p, dtype=float) - np.asarray(p_ref, dtype=float)
    e_v = np.asarray(v_world, dtype=float) - np.asarray(v_ref, dtype=float)
    a_des = np.asarray(a_ref, dtype=float) - kp_pos * e_p - kd_vel * e_v + gravity * e3

    a_xy = float(np.linalg.norm(a_des[:2]))
    a_z = max(1e-3, float(a_des[2]))
    tilt = math.atan2(a_xy, a_z)
    tilt_max = math.radians(max(1.0, float(max_tilt_deg)))
    if tilt > tilt_max and a_xy > 1e-6:
        scale = math.tan(tilt_max) * a_z / a_xy
        a_des[0] *= scale
        a_des[1] *= scale
    if float(np.linalg.norm(a_des)) < 1e-6:
        a_des = gravity * e3
    b3_des = a_des / float(np.linalg.norm(a_des))
    b1_yaw = np.array([math.cos(yaw_ref), math.sin(yaw_ref), 0.0], dtype=float)
    b2_des = np.cross(b3_des, b1_yaw)
    n_b2 = float(np.linalg.norm(b2_des))
    if n_b2 < 1e-6:
        b2_des = np.array([-math.sin(yaw_ref), math.cos(yaw_ref), 0.0], dtype=float)
    else:
        b2_des = b2_des / n_b2
    b1_des = np.cross(b2_des, b3_des)
    R_des = np.column_stack([b1_des, b2_des, b3_des])

    e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
    e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]], dtype=float)
    rate_cmd = -kR * e_R - kOmega * np.asarray(omega_body, dtype=float)
    thrust_N = float(mass * np.dot(a_des, R[:, 2]))
    return thrust_N, rate_cmd


def run_closed_loop_track_full_state_plan_acados(
    x0: np.ndarray,
    t_plan: np.ndarray,
    x_plan: np.ndarray,
    T_sim: float,
    sim_dt: float,
    control_dt: float,
    dt_mpc: float,
    N: int,
    *,
    w_pos: float = 1.0,
    w_att: float = 1.0,
    w_joint: float = 1.0,
    w_vel: float = 1.0,
    w_omega: float = 1.0,
    w_joint_vel: float = 1.0,
    w_control: float = 1e-3,
    w_u_thrust: float = 1.0,
    w_u_joint_torque: float = 1.0,
    w_state_track: float = 10.0,
    w_terminal_track: float = 100.0,
    mpc_max_iter: int = 40,
    mpc_log_interval: int = 0,
    control_mode: str = "direct",
    urdf_path: Optional[str] = None,
    state_limits: Optional[Dict[str, float]] = None,
    disturbance: Optional[Dict[str, Any]] = None,
    l1: Optional[Dict[str, Any]] = None,
    ctbr: Optional[Dict[str, Any]] = None,
    baseline: str = "acados",
    geometric: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Receding-horizon Acados tracking of ``(t_plan, x_plan)`` with RK4 plant.

    ``disturbance`` / ``l1`` (dict, 仅 direct 控制模式生效)：在闭环 plant 上注入
    外界力/力矩、变化扰动、总推力估计偏差、桨叶气动阻力、状态估计误差，并用
    ros_tracking 的 L1 自适应（scripts/l1_adaptive.py）在线估计与补偿。
    """
    t_plan = np.asarray(t_plan, dtype=float).flatten()
    x_plan = np.asarray(x_plan, dtype=float)
    if x_plan.ndim != 2 or len(t_plan) != len(x_plan):
        raise ValueError("x_plan must be 2D with same length as t_plan")

    urdf_path = _infer_urdf_path(x_plan, urdf_path)
    solver, acados_model, pin_model, nq, nv, nu, limits = create_full_state_tracking_mpc_solver(
        N,
        dt_mpc,
        w_pos=w_pos,
        w_att=w_att,
        w_joint=w_joint,
        w_vel=w_vel,
        w_omega=w_omega,
        w_joint_vel=w_joint_vel,
        w_control=w_control,
        w_u_thrust=w_u_thrust,
        w_u_joint_torque=w_u_joint_torque,
        w_state_track=w_state_track,
        w_terminal_track=w_terminal_track,
        max_iter=mpc_max_iter,
        control_mode=control_mode,
        urdf_path=urdf_path,
        state_limits=state_limits,
    )
    n_arm = nq - 7
    n_robot = nq + nv
    if x_plan.shape[1] < n_robot:
        raise ValueError(
            f"x_plan has {x_plan.shape[1]} state columns but robot needs {n_robot} "
            f"(urdf={urdf_path})"
        )

    f_fun = _make_f_expl_fun(acados_model)
    nx = n_robot + (6 if control_mode == "actuator_first_order" else 0)

    # ── Sim-only disturbances + L1 adaptive augmentation (direct control only) ──
    from sim_disturbance import (
        GRAVITY,
        DisturbanceParams,
        DisturbedRK4Plant,
        apply_thrust_bias,
        build_rotor_allocation,
        corrupt_state_estimate,
        external_force_torque,
        _active as _dist_active,
        _quat_to_R_np,
    )

    dist_params = DisturbanceParams.from_dict(disturbance)
    l1_cfg = dict(l1) if l1 else {}
    l1_on = bool(l1_cfg.get("enabled", False)) and control_mode == "direct"
    # 补偿来源：'adaptive'（L1 在线估计）或 'oracle'（用扰动真值，假设可精确测量），
    # 两者共用同一补偿管线（dFz/tilt、CTBR、位置通道），用于解耦评估补偿环节性能。
    comp_mode = str(l1_cfg.get("mode", "adaptive")).strip().lower()
    oracle_on = l1_on and comp_mode == "oracle"
    dist_on = dist_params.any_enabled() and control_mode == "direct"

    # ── CTBR 内环（总推力 + 前瞻角速度设定点 → 角速度 PID → 分配 → 电机一阶）──
    ctbr_cfg = dict(ctbr) if ctbr else {}
    ctbr_on = bool(ctbr_cfg.get("enabled", False)) and control_mode == "direct"
    ctbr_lookahead_s = float(ctbr_cfg.get("lookahead_ms", dt_mpc * 1000.0)) / 1000.0
    ctbr_kp = np.array([
        float(ctbr_cfg.get("kp_rp", 12.0)),
        float(ctbr_cfg.get("kp_rp", 12.0)),
        float(ctbr_cfg.get("kp_yaw", 4.0)),
    ], dtype=float)
    ctbr_ki = np.array([
        float(ctbr_cfg.get("ki_rp", 0.0)),
        float(ctbr_cfg.get("ki_rp", 0.0)),
        float(ctbr_cfg.get("ki_yaw", 0.0)),
    ], dtype=float)
    ctbr_kd = np.array([
        float(ctbr_cfg.get("kd_rp", 0.0)),
        float(ctbr_cfg.get("kd_rp", 0.0)),
        float(ctbr_cfg.get("kd_yaw", 0.0)),
    ], dtype=float)
    ctbr_max_rate = float(ctbr_cfg.get("max_rate", 8.0))
    ctbr_max_torque = float(ctbr_cfg.get("max_torque", 0.0))   # 0 = 不限
    ctbr_motor_tau = float(ctbr_cfg.get("motor_tau", 0.02))
    ctbr_int_limit = float(ctbr_cfg.get("int_limit", 2.0))

    # ── baseline 控制器：acados NMPC 或 geometric（SE3 几何，仅机体、只走 CTBR）──
    baseline_kind = str(baseline or "acados").strip().lower()
    geo_cfg = dict(geometric) if geometric else {}
    geometric_on = (baseline_kind == "geometric") and control_mode == "direct"
    geo_kp_pos = float(geo_cfg.get("kp_pos", 4.0))
    geo_kd_vel = float(geo_cfg.get("kd_vel", 2.5))
    geo_kR = float(geo_cfg.get("kR", 4.0))
    geo_kOmega = float(geo_cfg.get("kOmega", 0.35))
    geo_max_tilt = float(geo_cfg.get("max_tilt_deg", 35.0))
    if geometric_on:
        # 几何 baseline 没有直接四旋翼力指令，强制走 CTBR 内环。
        ctbr_on = True

    n_rotors = 4
    nominal_mass = float(pin.computeTotalMass(pin_model))
    cm_cf_alloc = float(load_s500_config()["platform"]["cm"] / load_s500_config()["platform"]["cf"])
    rotors_cfg = load_s500_config()["platform"]["$rotors"]
    A_alloc = build_rotor_allocation(rotors_cfg, cm_cf_alloc)
    try:
        A_inv = np.linalg.inv(A_alloc)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A_alloc)

    l1_aug = None
    if l1_on:
        from l1_adaptive import L1AdaptiveAugmentation, L1Params

        l1_keys = {
            "as_gain", "wc_xy", "wc_z", "tilt_gain", "max_accel_xy", "max_accel_z",
            "max_sigma", "use_pos_feedback", "k_pos_i_xy", "k_pos_i_z",
            "k_pos_p_xy", "k_pos_p_z", "max_pos_integral_xy", "max_pos_integral_z",
        }
        l1p = L1Params(enabled=True)
        for k in l1_keys:
            if k in l1_cfg and l1_cfg[k] is not None:
                setattr(l1p, k, l1_cfg[k])
        l1_aug = L1AdaptiveAugmentation(l1p)

    if dist_params.any_plant_disturbance():
        plant = DisturbedRK4Plant(f_fun, pin_model, sim_dt, nq, nv, nu)
    else:
        plant = CasadiRK4Plant(f_fun, sim_dt, nu)
    est_rng = np.random.default_rng(int(dist_params.est_seed))

    sim_dt = float(sim_dt)
    control_dt = float(control_dt)
    if control_dt < sim_dt - 1e-15:
        raise ValueError("control_dt must be >= sim_dt")
    mpc_stride = max(1, int(round(control_dt / sim_dt)))

    cfg = load_s500_config()
    plat = cfg["platform"]
    if control_mode == "actuator_first_order":
        from s500_uam_acados_model import nominal_command_hover

        u_hover = nominal_command_hover(
            pin_model, np.asarray(x0, dtype=float).flatten()[:n_robot],
            plat["min_thrust"], plat["max_thrust"],
        )
    else:
        u_hover = hover_thrust_controls(
            pin_model, nu, plat["min_thrust"], plat["max_thrust"]
        )

    n_steps = max(1, int(round(float(T_sim) / sim_dt)))
    t_log = np.zeros(n_steps + 1, dtype=float)
    x_log = np.zeros((n_steps + 1, nx), dtype=float)
    u_log = np.zeros((n_steps, nu), dtype=float)
    dist_force_log = np.zeros((n_steps, 3), dtype=float)
    dist_torque_body_log = np.zeros((n_steps, 3), dtype=float)
    dist_torque_world_log = np.zeros((n_steps, 3), dtype=float)
    l1_force_log = np.zeros((n_steps, 3), dtype=float)
    l1_sigma_log = np.zeros((n_steps, 3), dtype=float)
    l1_aac_log = np.zeros((n_steps, 3), dtype=float)
    l1_al1_log = np.zeros((n_steps, 3), dtype=float)
    # 控制分解日志：u_b(MPC baseline) / u_ac(L1 纯补偿) / u_p(位置通道)，
    # 三者满足 clip(u_b + u_ac + u_p) + thrust_bias = u_real(=u_log)。
    u_baseline_log = np.zeros((n_steps, nu), dtype=float)
    u_l1_delta_log = np.zeros((n_steps, nu), dtype=float)
    u_pos_delta_log = np.zeros((n_steps, nu), dtype=float)
    # CTBR 内环日志：角速度设定点（机体系，含 L1 tilt 增广）。
    ctbr_rate_sp_log = np.zeros((n_steps, 3), dtype=float)
    min_thrust = float(plat["min_thrust"])
    max_thrust = float(plat["max_thrust"])

    x = np.asarray(x0, dtype=float).flatten().copy()
    if x.size < nx:
        x_pad = np.zeros(nx, dtype=float)
        x_pad[: x.size] = x
        x = x_pad
    t_log[0] = 0.0
    x_log[0] = x

    x_prev: Optional[List[np.ndarray]] = None
    u_prev: Optional[List[np.ndarray]] = None
    mpc_solve_steps: List[int] = []
    mpc_iters: List[int] = []
    mpc_wall_s: List[float] = []
    u_apply = u_hover.copy()
    l1_thrust_delta = np.zeros(nu, dtype=float)
    # L1 推力增量的两路分解（a_l1 vs a_pos），其和恒等于 l1_thrust_delta。
    l1_thrust_delta_l1 = np.zeros(nu, dtype=float)
    l1_thrust_delta_pos = np.zeros(nu, dtype=float)
    # CTBR 内环状态：电机一阶滞后输出、角速度 PID 的积分/上一拍误差、当前推力指令。
    u_motor = u_hover.copy()
    ctbr_u_target = u_hover.copy()
    ctbr_int = np.zeros(3, dtype=float)
    ctbr_prev_err: Optional[np.ndarray] = None
    ctbr_rate_sp = np.zeros(3, dtype=float)
    # L1 预测器输入：上一拍"名义施加"比力加速度（世界系），必须含 L1 竖直注入，
    # 否则补偿推力会被误并入 σ̂，导致稳态估计仅为真值一半（见 _l1_predictor_accel）。
    l1_last_a_applied = np.zeros(3, dtype=float)

    for k in range(n_steps):
        t_k = k * sim_dt
        plant.on_pre_step(t_k, k)
        do_mpc = k % mpc_stride == 0

        # 控制器看到的状态估计（含量测噪声）。
        x_meas = (
            corrupt_state_estimate(dist_params, t_k, x, nq, nv, est_rng)
            if dist_on and dist_params.est_enable
            else x
        )

        if do_mpc:
            if geometric_on:
                # ── geometric baseline（SE3 几何，机体 s500，只走 CTBR）──────────
                # 由 x_plan 参考（世界系 pos/vel/acc + yaw）算总推力与角速度设定点。
                R_meas_g = _quat_to_R_np(x_meas[3:7])
                xr_now = interp_full_state_piecewise(t_k, t_plan, x_plan, pin_model)
                # 参考加速度数值微分步长：用控制周期（geometric 每 control_dt 更新一次），
                # 与 MPC 离散化无关，故不再依赖 dt_mpc。
                h_ref = max(control_dt, sim_dt)
                xr_p = interp_full_state_piecewise(t_k + h_ref, t_plan, x_plan, pin_model)
                use_central = (t_k - h_ref) >= 0.0
                xr_m = interp_full_state_piecewise(
                    max(0.0, t_k - h_ref), t_plan, x_plan, pin_model
                )

                def _v_world_of(xr):
                    Rr = _quat_to_R_np(np.asarray(xr[3:7], dtype=float))
                    return Rr @ np.asarray(xr[nq : nq + 3], dtype=float)

                p_ref = np.asarray(xr_now[0:3], dtype=float)
                v_ref = _v_world_of(xr_now)
                if use_central:
                    a_ref = (_v_world_of(xr_p) - _v_world_of(xr_m)) / (2.0 * h_ref)
                else:
                    a_ref = (_v_world_of(xr_p) - v_ref) / h_ref
                yaw_ref = _quat_to_yaw_np(np.asarray(xr_now[3:7], dtype=float))
                p_now = np.asarray(x_meas[0:3], dtype=float)
                v_world_now = R_meas_g @ np.asarray(x_meas[nq : nq + 3], dtype=float)
                omega_now = np.asarray(x_meas[nq + 3 : nq + 6], dtype=float)
                T_geo, omega_geo = _geometric_baseline_command(
                    p_now, v_world_now, R_meas_g, omega_now,
                    p_ref, v_ref, a_ref, yaw_ref, nominal_mass, GRAVITY,
                    kp_pos=geo_kp_pos, kd_vel=geo_kd_vel, kR=geo_kR,
                    kOmega=geo_kOmega, max_tilt_deg=geo_max_tilt,
                )
                u_apply = u_hover.copy()
                T_baseline = float(T_geo)
                ctbr_omega_base = omega_geo
                x_prev = None
            else:
                if k == 0 or x_prev is None:
                    x_roll = rollout_nominal_trajectory(f_fun, x_meas, u_hover, dt_mpc, N, nq)
                    set_solver_initial_guess(solver, x_meas, x_roll, u_hover, N)
                else:
                    shift_solver_initial_guess(solver, x_meas, x_prev, u_prev, N)

                solver.constraints_set(0, "lbx", x_meas, api="new")
                solver.constraints_set(0, "ubx", x_meas, api="new")

                xr_now = interp_full_state_piecewise(t_k, t_plan, x_plan, pin_model)
                for i in range(N):
                    ti = t_k + i * dt_mpc
                    xr = interp_full_state_piecewise(ti, t_plan, x_plan, pin_model)
                    yref = _state_to_yref(xr, u_hover, nq, n_arm)
                    solver.cost_set(i, "yref", yref, api="new")
                xrN = interp_full_state_piecewise(t_k + N * dt_mpc, t_plan, x_plan, pin_model)
                solver.cost_set(N, "yref", _state_to_yref_e(xrN, nq, n_arm), api="new")

                t0 = time.perf_counter()
                status = int(solver.solve())
                wall_s = time.perf_counter() - t0
                n_iter = _acados_nlp_iterations(solver)

                mpc_solve_steps.append(k)
                mpc_iters.append(n_iter)
                mpc_wall_s.append(wall_s)

                if mpc_log_interval > 0 and len(mpc_solve_steps) % mpc_log_interval == 0:
                    print(
                        f"[acados track] t={t_k:.3f} status={status} iter={n_iter} wall={wall_s*1e3:.1f} ms"
                    )

                x_prev = [solver.get(i, "x") for i in range(N + 1)]
                u_prev = [solver.get(i, "u") for i in range(N)]
                u_apply = np.asarray(solver.get(0, "u"), dtype=float).flatten().copy()
                T_baseline = float(np.sum(u_apply[:n_rotors]))
                ctbr_omega_base = None

            # ── L1 自适应：估计集总扰动并映射为四路推力增量（control rate） ──
            if l1_aug is not None:
                R_meas = _quat_to_R_np(x_meas[3:7])
                b3 = R_meas[:, 2]
                v_world = R_meas @ x_meas[nq : nq + 3]
                T_cmd = float(T_baseline)
                pos_err = (x_meas[0:3] - np.asarray(xr_now[0:3], dtype=float))
                if oracle_on:
                    # Oracle：用扰动真值（世界系加速度）= F_dist_world / m，假设可精确测量。
                    # 与 plant 侧 dist_force_log 一致：外力 + 推力估计偏差等效力。
                    F_ext_o, _Mb_o, _Mw_o = external_force_torque(
                        dist_params, t_k, R_meas, v_world, float(T_baseline), n_rotors
                    )
                    dT_o = 0.0
                    if dist_on and _dist_active(
                        t_k, dist_params.thrust_enable,
                        dist_params.thrust_t0, dist_params.thrust_t1,
                    ):
                        dT_o = (
                            (dist_params.thrust_scale - 1.0) * float(T_baseline)
                            + dist_params.thrust_bias
                        )
                    sigma_true = (F_ext_o + dT_o * b3) / max(nominal_mass, 1e-6)
                    a_ac = l1_aug.step_oracle(
                        control_dt, sigma_true, pos_err_world=pos_err
                    )
                else:
                    # 预测器输入用上一拍含 L1 竖直注入的实际加速度，使 σ̂ = 真实外部扰动。
                    a_ac = l1_aug.step(
                        control_dt, v_world, l1_last_a_applied, pos_err_world=pos_err
                    )
                dFz = nominal_mass * float(np.dot(a_ac, b3))
                a_nom = (T_cmd / max(nominal_mass, 1e-6)) * b3 - np.array([0.0, 0.0, GRAVITY])
                f_des = a_nom + a_ac + np.array([0.0, 0.0, GRAVITY])
                nfd = float(np.linalg.norm(f_des))
                M_body = np.zeros(3)
                if nfd > 1e-6 and l1_aug.params.tilt_gain > 0.0:
                    b3_des = f_des / nfd
                    e_tilt = np.cross(b3, b3_des)
                    M_body = R_meas.T @ (float(l1_aug.params.tilt_gain) * e_tilt)
                dthr = A_inv @ np.array([dFz, M_body[0], M_body[1], 0.0])
                l1_thrust_delta = np.zeros(nu, dtype=float)
                l1_thrust_delta[:n_rotors] = dthr

                # ── 推力增量分解：L1 纯补偿(a_l1) 与 位置通道(a_pos) ──────────
                # 竖直分量按比力投影线性可分；倾转力矩用 a_l1-only 的期望比力方向
                # 计算 M_l1，剩余 M_pos = M_body - M_l1，保证两路之和恒等于总增量。
                a_l1_w = np.asarray(l1_aug.a_l1, dtype=float)
                a_pos_w = np.asarray(l1_aug.a_pos, dtype=float)
                dFz_l1 = nominal_mass * float(np.dot(a_l1_w, b3))
                dFz_pos = nominal_mass * float(np.dot(a_pos_w, b3))
                M_l1 = np.zeros(3)
                if nfd > 1e-6 and l1_aug.params.tilt_gain > 0.0:
                    f_des_l1 = a_nom + a_l1_w + np.array([0.0, 0.0, GRAVITY])
                    nfd_l1 = float(np.linalg.norm(f_des_l1))
                    if nfd_l1 > 1e-6:
                        b3_des_l1 = f_des_l1 / nfd_l1
                        M_l1 = R_meas.T @ (
                            float(l1_aug.params.tilt_gain) * np.cross(b3, b3_des_l1)
                        )
                M_pos = M_body - M_l1
                l1_thrust_delta_l1 = np.zeros(nu, dtype=float)
                l1_thrust_delta_pos = np.zeros(nu, dtype=float)
                l1_thrust_delta_l1[:n_rotors] = A_inv @ np.array(
                    [dFz_l1, M_l1[0], M_l1[1], 0.0]
                )
                l1_thrust_delta_pos[:n_rotors] = A_inv @ np.array(
                    [dFz_pos, M_pos[0], M_pos[1], 0.0]
                )
                # 为下一拍更新预测器输入：实际总推力 = baseline + L1 竖直注入 dFz。
                # 水平补偿靠倾转实现，其效果已体现在测量姿态 b3 中，故不再显式 +a_ac。
                T_actual = T_cmd + dFz
                l1_last_a_applied = (
                    (T_actual / max(nominal_mass, 1e-6)) * b3
                    - np.array([0.0, 0.0, GRAVITY])
                )

            # ── CTBR 内环：总推力 + 前瞻角速度设定点 → 角速度 PID → 分配 ──────
            if ctbr_on:
                R_c = _quat_to_R_np(x_meas[3:7])
                b3_c = R_c[:, 2]
                T_cmd_c = float(T_baseline)
                # 角速度设定点：geometric 用几何律直接给；acados 用 MPC horizon 前瞻插值。
                if ctbr_omega_base is not None:
                    omega_sp = np.asarray(ctbr_omega_base, dtype=float).copy()
                else:
                    omega_sp = _bodyrate_from_horizon_states(
                        x_prev, ctbr_lookahead_s, dt_mpc, nq
                    )
                if omega_sp is None:
                    omega_sp = np.asarray(x_meas[nq + 3 : nq + 6], dtype=float).copy()
                # L1 增广（镜像 ROS run_tracking_controller._l1_augment_attitude_target）：
                # dFz→总推力，tilt→角速度设定点。a_ac≈0 时必须跳过——旧实现用
                # b3_des=(a_ac+g)/|a_ac+g|，在 a_ac=0 时退化为世界竖直 [0,0,1]，
                # 会持续把机体往“水平”拉，与 MPC 轨迹倾角对抗（无扰动也 ~4cm）。
                if l1_aug is not None:
                    a_ac = np.asarray(l1_aug.a_ac, dtype=float).reshape(3)
                    if np.any(np.abs(a_ac) > 1e-9):
                        T_cmd_c += nominal_mass * float(np.dot(a_ac, b3_c))
                        # 与 direct 路径一致：期望比力 = (T_baseline/m)·b3 + a_ac
                        # （不是 a_ac+g，后者在 a_ac=0 时错误地指向世界竖直）。
                        f_des = (
                            (float(T_baseline) / max(nominal_mass, 1e-6)) * b3_c
                            + a_ac
                        )
                        n_des = float(np.linalg.norm(f_des))
                        if n_des > 1e-6 and l1_aug.params.tilt_gain > 0.0:
                            b3_des = f_des / n_des
                            e_tilt = np.cross(b3_c, b3_des)
                            omega_sp = omega_sp + float(
                                l1_aug.params.tilt_gain
                            ) * e_tilt
                if ctbr_max_rate > 0.0:
                    omega_sp = np.clip(omega_sp, -ctbr_max_rate, ctbr_max_rate)
                ctbr_rate_sp = omega_sp.copy()
                # 角速度 PID（机体系，dt=control_dt）。
                omega_meas = np.asarray(x_meas[nq + 3 : nq + 6], dtype=float)
                err = omega_sp - omega_meas
                ctbr_int = ctbr_int + control_dt * err
                if ctbr_int_limit > 0.0:
                    ctbr_int = np.clip(ctbr_int, -ctbr_int_limit, ctbr_int_limit)
                if ctbr_prev_err is None:
                    derr = np.zeros(3, dtype=float)
                else:
                    derr = (err - ctbr_prev_err) / max(control_dt, 1e-9)
                ctbr_prev_err = err
                # 期望角加速度（机体系）：增益解释为闭环带宽 [1/s]，ω̇_des=Kp·e+Ki·∫+Kd·ė。
                w_cmd = ctbr_kp * err + ctbr_ki * ctbr_int + ctbr_kd * derr
                if ctbr_max_torque > 0.0:
                    w_cmd = np.clip(w_cmd, -ctbr_max_torque, ctbr_max_torque)
                # 控制有效性 B=∂(角加速度)/∂T_j：对 f_fun 有限差分，与 plant 完全一致，
                # 避免解析分配矩阵 A_alloc 的 roll/pitch 力矩符号与 plant 相反导致正反馈发散。
                x_eff = np.asarray(x_meas, dtype=float).flatten()
                if x_eff.size < nx:
                    _xp = np.zeros(nx, dtype=float)
                    _xp[: x_eff.size] = x_eff
                    x_eff = _xp
                xdot0 = np.asarray(f_fun(x_eff, u_apply)).flatten()
                B = np.zeros((3, n_rotors), dtype=float)
                for _j in range(n_rotors):
                    uu = u_apply.copy()
                    uu[_j] += 1.0
                    xdj = np.asarray(f_fun(x_eff, uu)).flatten()
                    B[:, _j] = (xdj[nq + 3 : nq + 6] - xdot0[nq + 3 : nq + 6])
                A_mix = np.vstack([np.ones((1, n_rotors)), B])
                b_mix = np.concatenate([[T_cmd_c], w_cmd])
                try:
                    rotors = np.linalg.solve(A_mix, b_mix)
                except np.linalg.LinAlgError:
                    rotors = np.linalg.lstsq(A_mix, b_mix, rcond=None)[0]
                ctbr_u_target = u_apply.copy()
                ctbr_u_target[:n_rotors] = np.clip(rotors, min_thrust, max_thrust)

        if k < n_steps:
            if ctbr_on:
                # CTBR：电机一阶滞后（sim 率）逼近 CTBR 指令；机械臂关节直通。
                alpha_m = (
                    1.0 - math.exp(-sim_dt / max(ctbr_motor_tau, 1e-9))
                    if ctbr_motor_tau > 1e-9
                    else 1.0
                )
                u_motor[:n_rotors] = (
                    u_motor[:n_rotors]
                    + alpha_m * (ctbr_u_target[:n_rotors] - u_motor[:n_rotors])
                )
                if nu > n_rotors:
                    u_motor[n_rotors:] = ctbr_u_target[n_rotors:]
                u_cmd = u_motor.copy()
                u_cmd[:n_rotors] = np.clip(u_cmd[:n_rotors], min_thrust, max_thrust)
            else:
                # 控制命令 = MPC 输出 + L1 增量；plant 侧再叠加推力估计偏差。
                u_cmd = u_apply.copy()
                u_cmd[:n_rotors] = np.clip(
                    u_cmd[:n_rotors] + l1_thrust_delta[:n_rotors], min_thrust, max_thrust
                )
            if dist_on and dist_params.thrust_enable:
                u_real, _dT = apply_thrust_bias(dist_params, t_k, u_cmd, n_rotors)
            else:
                u_real = u_cmd
            u_log[k] = u_real
            # 控制分解记录（保持与控制率一致：跨 stride 复用最近一次的增量）。
            if ctbr_on:
                u_baseline_log[k] = ctbr_u_target
                u_l1_delta_log[k] = 0.0
                u_pos_delta_log[k] = 0.0
                ctbr_rate_sp_log[k] = ctbr_rate_sp
            else:
                u_baseline_log[k] = u_apply
                u_l1_delta_log[k] = l1_thrust_delta_l1
                u_pos_delta_log[k] = l1_thrust_delta_pos

            if isinstance(plant, DisturbedRK4Plant):
                R_now = _quat_to_R_np(x[3:7])
                b3_now = R_now[:, 2]
                v_world_now = R_now @ x[nq : nq + 3]
                T_real = float(np.sum(u_real[:n_rotors]))
                F_ext, M_body_ext, M_world_ext = external_force_torque(
                    dist_params, t_k, R_now, v_world_now, T_real, n_rotors
                )
                w_body = np.zeros(6)
                w_body[:3] = R_now.T @ F_ext
                w_body[3:6] = M_body_ext
                plant.set_tau_ext_body(w_body)
                # 真值扰动力（世界系）：外力 + 推力估计偏差等效力（相对 MPC 基线 u_apply）。
                dT_base = 0.0
                if _dist_active(
                    t_k, dist_params.thrust_enable, dist_params.thrust_t0, dist_params.thrust_t1
                ):
                    T_base = float(np.sum(u_apply[:n_rotors]))
                    dT_base = (dist_params.thrust_scale - 1.0) * T_base + dist_params.thrust_bias
                dist_force_log[k] = F_ext + dT_base * b3_now
                dist_torque_body_log[k] = M_body_ext
                dist_torque_world_log[k] = M_world_ext

            if l1_aug is not None:
                l1_force_log[k] = l1_aug.disturbance_force_world(nominal_mass)
                l1_sigma_log[k] = l1_aug.sigma_hat
                l1_aac_log[k] = l1_aug.a_ac
                l1_al1_log[k] = l1_aug.a_l1

            x = plant.step(x, u_real)
            t_log[k + 1] = (k + 1) * sim_dt
            x_log[k + 1] = x

        if progress_cb is not None:
            progress_cb(k + 1, n_steps)

    ee_id = _resolve_ee_frame_id(pin_model)
    shim = _AcadosTrackMpcShim(pin_model, ee_id)

    return {
        "track_mode": "full_state_trajectory",
        "t_plan": t_plan,
        "x_plan": x_plan,
        "time": t_log,
        "states": x_log[:, :n_robot],
        "controls": u_log,
        "n_inner": mpc_stride,
        "sim_dt": sim_dt,
        "control_dt": control_dt,
        "mpc_solve_steps": mpc_solve_steps,
        "mpc_iters": mpc_iters,
        "mpc_wall_s": mpc_wall_s,
        "mpc": shim,
        "control_mode": control_mode,
        "dt_mpc": dt_mpc,
        "horizon": N,
        "state_limits": limits,
        "disturbance_active": bool(dist_on),
        "l1_active": bool(l1_on),
        "comp_mode": ("oracle" if oracle_on else "adaptive"),
        "oracle_active": bool(oracle_on),
        "nominal_mass": nominal_mass,
        "dist_force_world": dist_force_log,
        "dist_torque_body": dist_torque_body_log,
        "dist_torque_world": dist_torque_world_log,
        "l1_force_world": l1_force_log,
        "l1_sigma": l1_sigma_log,
        "l1_a_ac": l1_aac_log,
        "l1_a_l1": l1_al1_log,
        "controls_real": u_log,
        "u_baseline": u_baseline_log,
        "u_l1_delta": u_l1_delta_log,
        "u_pos_delta": u_pos_delta_log,
        "rotor_alloc": A_alloc,
        "n_rotors": n_rotors,
        "nq": nq,
        "nv": nv,
        "ctbr_active": bool(ctbr_on),
        "ctbr_rate_sp": ctbr_rate_sp_log,
        "baseline": baseline_kind,
        "geometric_active": bool(geometric_on),
    }


def acados_closed_loop_to_ee_tracking_res(out: Dict[str, Any]) -> Dict[str, Any]:
    from s500_uam_crocoddyl_state_tracking_mpc import crocoddyl_closed_loop_to_ee_tracking_res

    return crocoddyl_closed_loop_to_ee_tracking_res(out)
