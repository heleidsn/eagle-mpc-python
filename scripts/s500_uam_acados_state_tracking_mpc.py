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

import json
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
    from s500_uam_acados_trajectory import (
        STATE_LIMITS,
        _acados_ocp_generate_build_flags,
    )

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


def arm_joint_inertia_scales(
    pin_model, n_arm: int, q_ref: Optional[np.ndarray] = None
) -> np.ndarray:
    """各臂关节按质量矩阵对角元归一化的权重缩放系数（动能度量，均值=1）。

    串联臂各关节的等效惯量可差数倍（s500_uam: M[j1]≈4·M[j2]），给 j1/j2 同一权重
    会扭曲优化优先级、恶化 QP 条件数。这里用动能度量 W_i ∝ M_ii：惯量大的关节误差
    权重更高（更"贵"）。归一化到均值=1 → master ``w_joint`` 量级不变，且等惯量时
    退化为全 1（与原行为一致）。M 与构型弱相关，取中性构型评估即可。
    """
    na = int(n_arm)
    if na <= 0:
        return np.ones(0, dtype=float)
    nq = int(pin_model.nq)
    nv = int(pin_model.nv)
    if q_ref is None:
        q = pin.neutral(pin_model)
    else:
        q = np.asarray(q_ref, dtype=float).reshape(-1)[:nq].copy()
    data = pin_model.createData()
    M = pin.crba(pin_model, data, q)
    M = np.triu(M) + np.triu(M, 1).T
    diag = np.array([M[nv - na + i, nv - na + i] for i in range(na)], dtype=float)
    diag = np.clip(diag, 1e-12, None)
    mean = float(np.mean(diag))
    if mean <= 0.0:
        return np.ones(na, dtype=float)
    return diag / mean


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
    joint_weight_scales: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map Crocoddyl full-state tracking GUI weights to Acados NONLINEAR_LS W / W_e / R.

    State cost layout (robot-aware, matches ``_state_to_y_np``):
    [pos(3), yaw, roll, pitch, j*(n_arm), v_lin(3), ω(3), j̇*(n_arm)].

    ``joint_weight_scales``：长度 n_arm 的逐关节权重缩放（惯量缩放开启时由
    ``arm_joint_inertia_scales`` 提供），None 则全 1（各关节同权重，原行为）。
    位置与速度通道用同一缩放，保持度量一致。
    """
    na = int(n_arm)
    if joint_weight_scales is None:
        jsc = np.ones(na, dtype=float)
    else:
        jsc = np.asarray(joint_weight_scales, dtype=float).reshape(-1)
        if jsc.size != na:
            jsc = np.ones(na, dtype=float)
    s = float(w_state_track)
    diag = [float(w_pos) * s] * 3 + [float(w_att) * s] * 3
    diag += [float(w_joint) * s * float(jsc[i]) for i in range(na)]
    diag += [float(w_vel) * s] * 3 + [float(w_omega) * s] * 3
    diag += [float(w_joint_vel) * s * float(jsc[i]) for i in range(na)]
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


def _build_robot_acados_model(urdf_path: str, wholebody: bool = False):
    """Robot-aware AcadosModel: s500 (nq=7, nu=4) or s500_uam (nq=9, nu=6).

    ``wholebody=True`` 时把扰动参数从 6 维（base 等效 wrench）扩成 ``6+n_arm`` 维
    （whole-body）：前 6 维仍是世界系 base wrench（模型内旋回机体注入 floating
    joint），后 ``n_arm`` 维是关节空间扰动力矩，直接叠加到对应臂关节广义力通道，
    无坐标变换。用于动量观测器可直接给出整段 ``τ_ext`` (nv 维) 时不丢弃臂关节分量。
    """
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
    # disturbance-aware MPC：扰动参数 p_dist。默认 6 维 = 世界系 base wrench
    #   [F_world(3), M_world(3)]（N, N·m），经 ABA 以机体系广义力注入 base floating
    # joint 的力/力矩通道，使 MPC 在预测时已知扰动、规划最优倾角/推力/力矩（offset-free
    # MPC）。抓取负载在 ee 处的等效效应（重力 + 杠杆臂力矩）即以此 base 等效 wrench 表示。
    # wholebody=True 时再追加 n_arm 维关节空间扰动力矩 p_dist[6:]，直接叠加臂关节通道。
    # 后嵌（bolt-on）模式下调用方令 p_dist=0，模型退化为名义动力学。
    n_dist = 6 + (n_arm if (wholebody and n_arm > 0) else 0)
    p_dist_w = ca.SX.sym("p_dist_w", n_dist)

    thrusts = u[:n_thrust]
    arm_tau = u[n_thrust:] if n_arm > 0 else None

    quat = q[3:7]
    v_lin = v[:3]
    v_ang = v[3:6]
    R = _quat_to_R(quat)

    Fz = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
    Mx = My = Mz = 0.0
    for i, r in enumerate(rotors):
        pos = r["translation"]
        spin = r["spin_direction"][0]
        T = thrusts[i]
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T
    # 世界系扰动 wrench 转机体系并叠加到 base 广义力（floating joint 力/力矩均机体系）。
    F_dist_b = ca.mtimes(R.T, p_dist_w[:3])
    M_dist_b = ca.mtimes(R.T, p_dist_w[3:6])
    tau_base = ca.vertcat(
        F_dist_b[0], F_dist_b[1], Fz + F_dist_b[2],
        Mx + M_dist_b[0], My + M_dist_b[1], Mz + M_dist_b[2],
    )
    if n_arm > 0:
        # whole-body：关节空间扰动力矩直接叠加到臂关节广义力（无坐标变换）。
        arm_total = arm_tau + p_dist_w[6 : 6 + n_arm] if n_dist > 6 else arm_tau
        tau = ca.vertcat(tau_base, arm_total)
    else:
        tau = tau_base

    a = cpin.aba(cmodel, cdata, q, v, tau)

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
    acados_model.p = p_dist_w
    acados_model.f_impl_expr = acados_model.xdot - x_dot
    acados_model.f_expl_expr = x_dot
    return acados_model, model, nq, nv, nu


def build_plant_f_fun(urdf_path: str, load_params=None, ee_frame_id: Optional[int] = None):
    """构造 plant 用的 CasADi 显式动力学 f(x,u)（numpy 可调用），无扰动参数。

    与 MPC 名义模型同构（同一推力分配/ABA），但当 ``load_params`` 含启用的模拟负载时，
    把负载惯量刚性附着到 EE 帧再建动力学 → plant 比 MPC 名义模型多出负载效应（模型失配
    即真实扰动）。``ee_frame_id`` 缺省时按 EE_FRAME_NAME 解析。
    """
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    if load_params is not None and getattr(load_params, "load_enable", False):
        from sim_disturbance import augment_pin_model_with_load

        fid = ee_frame_id if ee_frame_id is not None else int(model.getFrameId(EE_FRAME_NAME))
        model = augment_pin_model_with_load(model, fid, load_params)

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    cfg = load_s500_config()
    platform = cfg["platform"]
    cm_cf = platform["cm"] / platform["cf"]
    rotors = platform["$rotors"]

    nq, nv = model.nq, model.nv
    n_arm = nq - 7
    n_thrust = 4
    nu = n_thrust + n_arm

    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv)
    u = ca.SX.sym("u", nu)
    thrusts = u[:n_thrust]
    arm_tau = u[n_thrust:] if n_arm > 0 else None

    quat = q[3:7]
    v_lin = v[:3]
    v_ang = v[3:6]
    R = _quat_to_R(quat)

    Fz = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
    Mx = My = Mz = 0.0
    for i, r in enumerate(rotors):
        pos = r["translation"]
        spin = r["spin_direction"][0]
        T = thrusts[i]
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T
    tau_base = ca.vertcat(0.0, 0.0, Fz, Mx, My, Mz)
    tau = ca.vertcat(tau_base, arm_tau) if n_arm > 0 else tau_base

    a = cpin.aba(cmodel, cdata, q, v, tau)
    pos_dot = ca.mtimes(R, v_lin)
    quat_dot = 0.5 * _quat_prod(quat, ca.vertcat(v_ang[0], v_ang[1], v_ang[2], 0))
    if n_arm > 0:
        q_dot = ca.vertcat(pos_dot, quat_dot, v[6 : 6 + n_arm])
    else:
        q_dot = ca.vertcat(pos_dot, quat_dot)
    x_dot = ca.vertcat(q_dot, a)
    return ca.Function("f_plant", [ca.vertcat(q, v), u], [x_dot])


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


# codegen 时 NLP/QP 迭代上界（运行时可 options_set 下调，避免 max_iter 改动触发重编）。
_ACADOS_CODEGEN_NLP_MAX_ITER = 80
_ACADOS_CODEGEN_QP_ITER_MAX = 100


def _align_ocp_with_cached_json(ocp: "AcadosOcp", json_path: Path) -> bool:
    """把参与 OCP 哈希、但运行时可改的字段对齐到已缓存 ocp.json，避免无谓 codegen。

    权重 W/W_e、状态/输入约束界、NLP/QP 迭代上限在加载后仍会用当前参数覆盖。
    """
    if not json_path.is_file():
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    cost = data.get("cost") or {}
    if cost.get("W"):
        W_cached = np.array(cost["W"], dtype=float)
        cur_w = getattr(ocp.cost, "W", None)
        if cur_w is not None and np.asarray(cur_w).shape == W_cached.shape:
            ocp.cost.W = W_cached
    if cost.get("W_e"):
        W_e_cached = np.array(cost["W_e"], dtype=float)
        cur_we = getattr(ocp.cost, "W_e", None)
        if cur_we is not None and np.asarray(cur_we).shape == W_e_cached.shape:
            ocp.cost.W_e = W_e_cached

    solver_opts = data.get("solver_options") or {}
    for key in ("nlp_solver_max_iter", "qp_solver_iter_max"):
        if key in solver_opts and hasattr(ocp.solver_options, key):
            setattr(ocp.solver_options, key, solver_opts[key])

    cst = data.get("constraints") or {}
    for key in ("lbx", "ubx", "lbu", "ubu"):
        vals = cst.get(key)
        if vals is not None and len(vals) > 0 and hasattr(ocp.constraints, key):
            setattr(ocp.constraints, key, np.array(vals, dtype=float))
    return True


def _apply_runtime_solver_options(
    solver: "AcadosOcpSolver", max_iter: int, *, solver_type: str = "SQP"
) -> None:
    """在已编译上界内下调 NLP 迭代次数（SQP 专用；RTI 每步固定 1 次 QP）。"""
    if str(solver_type or "SQP").upper() == "SQP_RTI":
        return
    want = int(max_iter)
    try:
        baked = int(solver.__solver_options.get("nlp_solver_max_iter", want))
    except Exception:
        baked = want
    if want <= baked:
        try:
            solver.options_set("nlp_solver_max_iter", want)
        except Exception:
            pass


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
    solver_type: str = "SQP",
    integrator: Optional[str] = None,
    obs_wholebody: bool = False,
    inertia_scaling: bool = False,
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
        acados_model, pin_model, nq, nv, nu = _build_robot_acados_model(
            urdf_path, wholebody=bool(obs_wholebody)
        )
        robot_tag = "uam" if n_arm_probe > 0 else "s500"
        wb_tag = "_wb" if (obs_wholebody and n_arm_probe > 0) else ""
        _integ_tag = str(
            integrator or ("ERK" if control_mode == "actuator_first_order" else "IRK")
        ).strip().upper()
        _stype_tag = str(solver_type or "SQP").strip().upper()
        if _stype_tag not in ("SQP", "SQP_RTI"):
            _stype_tag = "SQP"
        # 求解器/积分器写入子目录名，避免 SQP+IRK 与 SQP_RTI+ERK 共用目录互相覆盖 .so。
        code_subdir = (
            f"s500_{robot_tag}_full_state_track_mpc{wb_tag}"
            f"_{_stype_tag.lower()}_{_integ_tag.lower()}"
        )
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
    joint_weight_scales = (
        arm_joint_inertia_scales(pin_model, n_arm)
        if (inertia_scaling and n_arm > 0)
        else None
    )
    if joint_weight_scales is not None:
        print(
            "[acados track] inertia scaling on: joint weight scales = "
            + ", ".join(f"{v:.3f}" for v in joint_weight_scales)
        )
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
        joint_weight_scales=joint_weight_scales,
    )

    ocp = AcadosOcp()
    ocp.model = acados_model
    nx = nq + nv + (6 if control_mode == "actuator_first_order" else 0)
    # disturbance-aware MPC：direct 模型含扰动 wrench 参数 p_dist_w（默认 6 维世界系 base
    # wrench；whole-body 时 6+n_arm 维，追加关节扰动力矩），默认 0（名义）。闭环每拍按估计
    # wrench 下发到各 stage；后嵌模式保持 0。actuator_first_order 模型无参数。
    if control_mode == "direct":
        n_dist_param = 6 + (n_arm if (obs_wholebody and n_arm > 0) else 0)
        ocp.parameter_values = np.zeros(n_dist_param)

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
    # 编译进 C 的迭代上界取固定较大值；实际 max_iter 在加载后 options_set 下调。
    codegen_nlp_iter = max(_ACADOS_CODEGEN_NLP_MAX_ITER, int(max_iter))
    codegen_qp_iter = max(_ACADOS_CODEGEN_QP_ITER_MAX, int(max_iter) * 2)
    ocp.solver_options.nlp_solver_max_iter = codegen_nlp_iter
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
    # 积分器：默认 actuator_first_order 用 ERK，direct 用 IRK（隐式，更稳但更慢）。
    # 可经 integrator 覆盖为 "ERK" 提速（动力学非刚性时通常够用且快很多）。
    _integ = (integrator or ("ERK" if control_mode == "actuator_first_order" else "IRK"))
    ocp.solver_options.integrator_type = str(_integ).upper()
    # 求解器类型：SQP（多迭代到收敛，精确）或 SQP_RTI（实时迭代，每步 1 个 QP，快很多，
    # 暖启动下跟踪通常足够好）。RTI 下 nlp_solver_max_iter 无意义（固定单次）。
    _stype = str(solver_type or "SQP").upper()
    if _stype not in ("SQP", "SQP_RTI"):
        _stype = "SQP"
    ocp.solver_options.nlp_solver_type = _stype
    ocp.solver_options.print_level = 0
    if hasattr(ocp.solver_options, "qp_solver_iter_max"):
        ocp.solver_options.qp_solver_iter_max = codegen_qp_iter

    code_export_dir = REPO_ROOT / "c_generated_code" / code_subdir
    code_export_dir.mkdir(parents=True, exist_ok=True)
    json_path = code_export_dir / "ocp.json"
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(json_path)
    ocp.constraints.x0 = np.zeros(nx)

    # 权重/约束界/max_iter 会进入 OCP 哈希；对齐已缓存 json 后再比哈希，可避免每次仿真
    # 因调权重或状态限位而触发 ~15s 的 codegen+build（结构未变时应 <0.1s 加载 .so）。
    _align_ocp_with_cached_json(ocp, json_path)

    gen, bld = _acados_ocp_generate_build_flags(code_export_dir, acados_model.name)
    t_solver0 = time.perf_counter()
    solver = AcadosOcpSolver(
        ocp,
        json_file=str(json_path),
        generate=gen,
        build=bld,
        verbose=False,
        check_reuse_possible=True,
    )
    solver_setup_s = time.perf_counter() - t_solver0
    if solver_setup_s > 1.0 or getattr(solver, "generated", False):
        tag = "codegen+build" if getattr(solver, "generated", False) else "build"
        print(
            f"[acados track] OCP {tag} {solver_setup_s:.1f}s "
            f"dir={code_subdir} (结构/哈希变化；同结构二次运行应 <1s)"
        )
    # 运行时覆盖：权重、状态界、NLP 迭代（不必重编）。
    W_full = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))
    for i in range(int(N)):
        solver.cost_set(i, "W", W_full)
    solver.cost_set(int(N), "W", W_e)
    _apply_solver_state_bounds(solver, int(N), robot_lbx, robot_ubx)
    _apply_runtime_solver_options(solver, int(max_iter), solver_type=_stype)
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


def _cost_term_keys(n_arm: int) -> List[str]:
    """Cost analysis 分项名（与 _build_cost_y_exprs / W 布局对应，含臂时多关节项）。"""
    keys = ["pos", "att"]
    if n_arm > 0:
        keys.append("joint")
    keys += ["vel", "omega"]
    if n_arm > 0:
        keys.append("joint_vel")
    keys.append("u_thrust")
    if n_arm > 0:
        keys.append("u_torque")
    keys.append("terminal")
    return keys


def _acados_cost_term_breakdown(
    solver: "AcadosOcpSolver",
    N: int,
    nq: int,
    n_arm: int,
    W_state_diag: np.ndarray,
    R_diag: np.ndarray,
    w_terminal_track: float,
    yref_running: List[np.ndarray],
    yref_terminal: Optional[np.ndarray],
) -> Dict[str, float]:
    """按 NONLINEAR_LS 布局拆解本拍 MPC 的加权代价（0.5·rᵀWr，分项求和）。

    running 段（stage 0..N-1）：y=[pos,att,joint?,vel,omega,joint_vel?,u_thrust,u_torque?]；
    terminal 段（stage N）：仅状态项，权重 ×w_terminal_track，单列为 'terminal'。
    各分项之和 ≈ solver.get_cost()，但 get_cost 作为 total 单独记录以保精确。
    """
    na = int(n_arm)
    sy = 12 + 2 * na  # 状态 y 维度
    W_run = np.concatenate([np.asarray(W_state_diag, float), np.asarray(R_diag, float)])
    W_e_diag = np.asarray(W_state_diag, float) * float(w_terminal_track)
    grp = {k: 0.0 for k in _cost_term_keys(na)}

    for i in range(int(N)):
        xi = np.asarray(solver.get(i, "x"), dtype=float).flatten()
        ui = np.asarray(solver.get(i, "u"), dtype=float).flatten()
        yi = np.concatenate([_state_to_y_np(xi, nq, na), ui])
        ref = (
            np.asarray(yref_running[i], dtype=float).flatten()
            if i < len(yref_running)
            else np.zeros_like(yi)
        )
        m = min(yi.size, ref.size, W_run.size)
        wsq = W_run[:m] * (yi[:m] - ref[:m]) ** 2
        grp["pos"] += 0.5 * float(np.sum(wsq[0:3]))
        grp["att"] += 0.5 * float(np.sum(wsq[3:6]))
        off = 6
        if na > 0:
            grp["joint"] += 0.5 * float(np.sum(wsq[off : off + na]))
            off += na
        grp["vel"] += 0.5 * float(np.sum(wsq[off : off + 3]))
        grp["omega"] += 0.5 * float(np.sum(wsq[off + 3 : off + 6]))
        wc = wsq[sy:]
        grp["u_thrust"] += 0.5 * float(np.sum(wc[0:4]))
        if na > 0:
            grp["joint_vel"] += 0.5 * float(np.sum(wsq[off + 6 : off + 6 + na]))
            grp["u_torque"] += 0.5 * float(np.sum(wc[4 : 4 + na]))

    xN = np.asarray(solver.get(int(N), "x"), dtype=float).flatten()
    yN = _state_to_y_np(xN, nq, na)
    refN = (
        np.asarray(yref_terminal, dtype=float).flatten()
        if yref_terminal is not None
        else np.zeros_like(yN)
    )
    me = min(yN.size, refN.size, W_e_diag.size)
    grp["terminal"] = 0.5 * float(np.sum(W_e_diag[:me] * (yN[:me] - refN[:me]) ** 2))
    return grp


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


def _tau_applied_generalized(
    u_apply: np.ndarray,
    A_alloc: np.ndarray,
    nv: int,
    nu: int,
    n_rotors: int,
) -> np.ndarray:
    """名义施加广义力（机体系 rotor wrench + 臂关节力矩），与 aba / 动量观测器输入一致。"""
    w_rotor = A_alloc @ np.asarray(u_apply[:n_rotors], dtype=float)
    tau_app = np.zeros(nv, dtype=float)
    tau_app[2] = float(w_rotor[0])
    tau_app[3:6] = np.asarray(w_rotor[1:4], dtype=float)
    if nu > n_rotors and nv > 6:
        n_extra = min(nu - n_rotors, nv - 6)
        tau_app[6 : 6 + n_extra] = np.asarray(
            u_apply[n_rotors : n_rotors + n_extra], dtype=float
        )
    return tau_app


def _fill_sigma_mom_wb(
    sigma_for_mpc: np.ndarray,
    mom_obs,
    R_bw: np.ndarray,
    n_dist_param: int,
    n_arm: int,
    *,
    arm_only: bool = False,
) -> None:
    """动量观测器 → MPC 扰动参数。arm_only=True 时只填 p[6:]（臂关节）。"""
    if not arm_only:
        w_w = mom_obs.base_wrench_world(R_bw)
        sigma_for_mpc[:6] = np.asarray(w_w, dtype=float).reshape(6)
    if n_dist_param > 6 and n_arm > 0:
        arm_ext = np.asarray(mom_obs.arm_torque_ext(), dtype=float).reshape(-1)
        n_take = min(n_dist_param - 6, int(n_arm), arm_ext.size)
        sigma_for_mpc[6 : 6 + n_take] = arm_ext[:n_take]


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
    a_ff_world: Optional[np.ndarray] = None,
) -> tuple:
    """SE3 几何控制律（移植 example/l1_geometric_tracking_sim.geometric_baseline）。

    返回 (thrust_N, body_rate_cmd)：总推力（N）与机体角速度设定点（rad/s）。
    与 ROS run_tracking_controller 的 geometric 一致：位置/速度 PD 得期望比力 a_des，
    构造期望姿态 R_des，姿态误差 e_R 经 kR/kOmega 得体角速度指令。

    a_ff_world：L1 等扰动补偿的前馈加速度（世界系, m/s²），直接并入期望比力
    a_des。这样补偿在**力/期望姿态层**生效——matched 抬推力、unmatched 让 R_des
    主动倾转，控制器自己飞过去；避免在 baseline 算完后再往角速度上硬加 tilt 与
    姿态环对抗（悬停下 oracle 也补不动横向扰动的根因）。
    """
    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    e_p = np.asarray(p, dtype=float) - np.asarray(p_ref, dtype=float)
    e_v = np.asarray(v_world, dtype=float) - np.asarray(v_ref, dtype=float)
    a_des = np.asarray(a_ref, dtype=float) - kp_pos * e_p - kd_vel * e_v + gravity * e3
    if a_ff_world is not None:
        a_des = a_des + np.asarray(a_ff_world, dtype=float).reshape(3)

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
    solver_type: str = "SQP",
    integrator: Optional[str] = None,
    cost_analysis: bool = True,
    inertia_scaling: bool = False,
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
    # 扰动增广开关：扩成 6+n_arm 维模型扰动参数（base 6D 世界系 wrench + 臂关节力矩），
    # 供 in-model 注入喂入广义外力。in-model 注入**必须**增广（否则臂关节那块广义力无
    # 通道传给 MPC）；bolt-on 不需要增广模型，u_ad 直接补。
    _inject_early = str((l1 or {}).get("inject", "bolt_on")).strip().lower()
    obs_wholebody = control_mode == "direct" and (
        _inject_early == "in_model" or bool((l1 or {}).get("obs_wholebody", False))
    )
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
        solver_type=solver_type,
        integrator=integrator,
        obs_wholebody=obs_wholebody,
        inertia_scaling=inertia_scaling,
    )
    n_arm = nq - 7
    joint_weight_scales = (
        arm_joint_inertia_scales(pin_model, n_arm)
        if (inertia_scaling and n_arm > 0)
        else None
    )
    # 扰动参数维度：whole-body 时 6+n_arm（base 6 维世界系 wrench + 臂关节力矩），否则 6。
    n_dist_param = 6 + (n_arm if (obs_wholebody and n_arm > 0) else 0)
    n_robot = nq + nv
    if x_plan.shape[1] < n_robot:
        raise ValueError(
            f"x_plan has {x_plan.shape[1]} state columns but robot needs {n_robot} "
            f"(urdf={urdf_path})"
        )

    # ── Cost analysis：重建权重对角，用于每拍 NONLINEAR_LS 代价分项拆解 ──
    _Wst, _Rdiag, _ = croc_tracking_weights_to_W_R(
        n_arm=n_arm, w_pos=w_pos, w_att=w_att, w_joint=w_joint, w_vel=w_vel,
        w_omega=w_omega, w_joint_vel=w_joint_vel, w_control=w_control,
        w_u_thrust=w_u_thrust, w_u_joint_torque=w_u_joint_torque,
        w_state_track=w_state_track, w_terminal_track=w_terminal_track,
        joint_weight_scales=joint_weight_scales,
    )
    W_state_diag = np.diag(_Wst).astype(float)
    R_diag = np.diag(_Rdiag).astype(float)
    cost_term_keys = _cost_term_keys(n_arm)
    _s_track = float(w_state_track)
    mpc_cost_weights: Dict[str, float] = {
        "pos": w_pos * _s_track, "att": w_att * _s_track,
        "vel": w_vel * _s_track, "omega": w_omega * _s_track,
        "u_thrust": w_control * w_u_thrust, "terminal": float(w_terminal_track),
    }
    if n_arm > 0:
        mpc_cost_weights["joint"] = w_joint * _s_track
        mpc_cost_weights["joint_vel"] = w_joint_vel * _s_track
        mpc_cost_weights["u_torque"] = w_control * w_u_joint_torque * 10000.0
    mpc_costs: List[float] = []
    mpc_solve_t: List[float] = []
    mpc_cost_terms_hist: Dict[str, List[float]] = {k: [] for k in cost_term_keys}

    # 名义模型函数 f(x,u)：disturbance-aware 模型含自由参数 p_dist_w，这里代入 0 得到
    # 名义动力学（rollout/初值猜测/CTBR 有限差分 B 都应基于无扰名义模型，扰动只通过
    # solver 的 stage 参数进入预测）。无参数模型（actuator_first_order）直接构造。
    _p_sym = getattr(acados_model, "p", None)
    if _p_sym is not None and getattr(_p_sym, "shape", (0,))[0] > 0:
        _f_expl_nom = ca.substitute(
            acados_model.f_expl_expr, _p_sym, ca.DM.zeros(_p_sym.shape[0])
        )
        f_fun = ca.Function("f_nom", [acados_model.x, acados_model.u], [_f_expl_nom])
    else:
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
        ee_wrench_world,
        disturbance_base_equiv_world,
        _active as _dist_active,
        _quat_to_R_np,
    )

    dist_params = DisturbanceParams.from_dict(disturbance)
    l1_cfg = dict(l1) if l1 else {}
    l1_on = bool(l1_cfg.get("enabled", False)) and control_mode == "direct"
    # 扰动补偿开关：与扰动估计解耦。comp_on=False → 估计器照常运行并记录 σ̂（供
    # 真值/估计对照绘图），但不把 σ̂ 注入控制器（不做 bolt-on dFz/tilt、不做模型增广），
    # 即"只估计不补偿"。独立的位置误差反馈通道不受此开关影响。
    comp_on = bool(l1_cfg.get("comp_enabled", True))
    # 位置误差反馈通道：独立于 L1 估计，可在不开 L1 时单独用跟踪误差产生补偿。
    pos_fb_on = bool(l1_cfg.get("use_pos_feedback", False)) and control_mode == "direct"
    # 补偿来源（仅两种）：
    #   'adaptive' → 广义 L1：在广义动量上一阶自适应，在线估计**完整**广义外力
    #                τ̂_ext = [base力(3), base力矩(3), 臂关节(n)]（GeneralizedL1Estimator）；
    #   'oracle'   → 用扰动真值（假设可精确测量），同一注入管线，用于上界对照/调试。
    # 两者共用两条注入路径：bolt-on（u_b+u_ad）与 in-model（增广 MPC）。
    comp_mode = str(l1_cfg.get("mode", "adaptive")).strip().lower()
    if comp_mode not in ("adaptive", "oracle"):
        comp_mode = "adaptive"   # 旧 'momentum' 等估计来源已统一并入广义 L1
    oracle_on = l1_on and comp_mode == "oracle"
    # 广义 L1 估计开关（估计独立于补偿：comp_on=False 时仍估计并记录，仅不注入）。
    gen_l1_on = l1_on and comp_mode == "adaptive" and control_mode == "direct"
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

    # ── 补偿注入点：'bolt_on'（后嵌，默认：MPC 解完再叠 dFz/tilt）或
    #    'in_model'（模型增广 disturbance-aware MPC：把 σ̂·m 作为世界系扰动力参数
    #    喂进 acados 模型，MPC 预测时已知扰动、规划最优倾角/推力，不再事后叠加）。
    # in_model 仅 acados baseline + direct 生效；geometric 无 MPC 模型故不适用。
    inject_mode = str(l1_cfg.get("inject", "bolt_on")).strip().lower()
    dist_aware_on = (
        l1_on
        and comp_on
        and inject_mode == "in_model"
        and not geometric_on
        and control_mode == "direct"
    )
    # per-stage 扰动前馈：oracle（扰动解析已知）时，给 horizon 每个 stage 喂它对应
    # 预测时刻 t_k+i·dt_mpc 的真实世界系扰动力，而非全程常值。这样 MPC 能在扰动
    # 真正作用前就"看见"后段 stage 的扰动并提前预倾（preview/anticipation），
    # 显著削弱阶跃扰动瞬间的位置/姿态暂态。L1 估计模式无未来信息 → 仍常值保持。
    dist_aware_perstage = dist_aware_on and oracle_on
    # 模型增广的"估计更新时机"：
    #   'post'（默认，原行为）：求解后才更新 σ̂，下一拍才喂进模型 → 单拍滞后；
    #   'pre'：本拍先用当前量测做估计，立刻用于本拍 MPC 模型参数 → 消除单拍滞后。
    # 仅对 L1 估计的模型增广有意义（oracle 走 per-stage、用当前/未来解析真值，无滞后）。
    dist_aware_update = str(l1_cfg.get("dist_aware_update", "post")).strip().lower()
    dist_aware_pre = (
        dist_aware_on and (not dist_aware_perstage)
        and dist_aware_update == "pre"
    )

    n_rotors = 4
    nominal_mass = float(pin.computeTotalMass(pin_model))
    cm_cf_alloc = float(load_s500_config()["platform"]["cm"] / load_s500_config()["platform"]["cf"])
    rotors_cfg = load_s500_config()["platform"]["$rotors"]
    A_alloc = build_rotor_allocation(rotors_cfg, cm_cf_alloc)
    try:
        A_inv = np.linalg.inv(A_alloc)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A_alloc)

    # L1 估计 或 位置误差反馈 任一开启即创建增广器（二者共用同一补偿管线/注入）。
    l1_aug = None
    if l1_on or pos_fb_on:
        from l1_adaptive import L1AdaptiveAugmentation, L1Params

        l1_keys = {
            "as_gain", "wc_xy", "wc_z", "tilt_gain", "max_accel_xy", "max_accel_z",
            "max_sigma", "frame", "method", "use_pos_feedback", "k_pos_i_xy", "k_pos_i_z",
            "k_pos_p_xy", "k_pos_p_z", "max_pos_integral_xy", "max_pos_integral_z",
        }
        # enabled=L1 估计开关；位置反馈由 use_pos_feedback 单独控制，L1 关也能产生补偿。
        l1p = L1Params(enabled=bool(l1_on))
        for k in l1_keys:
            if k in l1_cfg and l1_cfg[k] is not None:
                setattr(l1p, k, l1_cfg[k])
        l1_aug = L1AdaptiveAugmentation(l1p)

    # ── 广义 L1 估计器：在广义动量上一阶自适应，估计完整广义外力 τ̂_ext (nv) ──
    # 内核为广义动量观测器（即广义 L1 的实现），增益 K 即各通道低通带宽。base 平动力
    # 的补偿映射（bolt-on 的 dFz/tilt + 位置积分）仍复用 l1_aug 的成熟管线。
    gen_est = None
    obs_grasp_reset_t = float(l1_cfg.get("obs_grasp_reset_t", -1.0))
    if gen_l1_on:
        from l1_adaptive import GeneralizedL1Estimator, GeneralizedL1Params

        gen_est = GeneralizedL1Estimator(
            pin_model, nq, nv,
            GeneralizedL1Params(
                enabled=True,
                k_force=float(l1_cfg.get("obs_k_force", 20.0)),
                k_torque=float(l1_cfg.get("obs_k_torque", 20.0)),
                k_arm=float(l1_cfg.get("obs_k_arm", 20.0)),
                max_force=float(l1_cfg.get("gen_max_force", 40.0)),
                max_torque=float(l1_cfg.get("gen_max_torque", 10.0)),
                max_arm=float(l1_cfg.get("gen_max_arm", 10.0)),
            ),
        )

    # EE 力旋量注入 / 负载真值折算所需：EE 帧 id 与一份独立 pin data。
    ee_id = _resolve_ee_frame_id(pin_model)
    dist_data = pin_model.createData()
    # 模拟负载仅 s500_uam(direct) 支持：plant 改用含负载惯量的增广模型积分。
    load_on = bool(dist_params.load_enable) and n_arm > 0 and control_mode == "direct"
    if dist_params.any_plant_disturbance():
        f_load = None
        load_model = None
        if load_on:
            from sim_disturbance import augment_pin_model_with_load

            f_load = build_plant_f_fun(urdf_path, load_params=dist_params, ee_frame_id=ee_id)
            load_model = augment_pin_model_with_load(pin_model, ee_id, dist_params)
        plant = DisturbedRK4Plant(
            f_fun, pin_model, sim_dt, nq, nv, nu,
            f_load=f_load, load_model=load_model,
            load_t0=float(dist_params.load_t0), load_t1=float(dist_params.load_t1),
        )
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
    # 同一拍同时记录机体系版本（= R(q_k)ᵀ·世界系），绘图按钮切换显示而非事后旋转。
    dist_force_body_log = np.zeros((n_steps, 3), dtype=float)
    dist_torque_body_log = np.zeros((n_steps, 3), dtype=float)
    dist_torque_world_log = np.zeros((n_steps, 3), dtype=float)
    l1_force_log = np.zeros((n_steps, 3), dtype=float)
    # 动量观测器：base 估计力矩（世界系），与真实 dist_torque 对比。
    l1_torque_log = np.zeros((n_steps, 3), dtype=float)
    # s500_uam 臂关节扰动力矩真值/估计（关节空间 [N·m]，仅 n_arm>0 时有效）。
    dist_joint_torque_log = (
        np.zeros((n_steps, n_arm), dtype=float) if n_arm > 0 else None
    )
    l1_joint_torque_log = (
        np.zeros((n_steps, n_arm), dtype=float) if n_arm > 0 else None
    )
    l1_force_body_log = np.zeros((n_steps, 3), dtype=float)
    l1_sigma_log = np.zeros((n_steps, 3), dtype=float)
    l1_aac_log = np.zeros((n_steps, 3), dtype=float)
    l1_al1_log = np.zeros((n_steps, 3), dtype=float)
    l1_al1_body_log = np.zeros((n_steps, 3), dtype=float)
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
    # disturbance-aware MPC：下发给 acados 模型参数的扰动 wrench（n_dist_param 维）。前 6
    #   维 = 世界系 [F_world(3), M_world(3)]：力 = m·σ̂（L1/oracle 估计），力矩仅 oracle/动量
    # 观测器提供；whole-body 时第 6: 维 = 关节空间扰动力矩（仅动量观测器提供）。
    sigma_for_mpc = np.zeros(n_dist_param, dtype=float)
    mom_grasp_done = False  # 估计器抓取事件软重置是否已触发
    # 广义外力估计供绘图（base 世界系力/力矩 + 臂关节力矩），每个 MPC 拍更新、跨 stride 保持。
    est_base_F_w = np.zeros(3, dtype=float)
    est_base_M_w = np.zeros(3, dtype=float)
    est_arm_tau = np.zeros(n_arm, dtype=float) if n_arm > 0 else None

    # ── 运行时间分解统计（找瓶颈用）：各段累计墙钟 [s] 与调用次数 ──────────
    prof_t: Dict[str, float] = {}
    prof_n: Dict[str, int] = {}

    def _prof(name: str, dt: float) -> None:
        prof_t[name] = prof_t.get(name, 0.0) + float(dt)
        prof_n[name] = prof_n.get(name, 0) + 1

    _loop_t0 = time.perf_counter()

    for k in range(n_steps):
        t_k = k * sim_dt
        plant.on_pre_step(t_k, k)
        do_mpc = k % mpc_stride == 0
        # 模型增广 'pre' 模式：本拍是否已在 MPC 求解前做过 L1 估计（避免后置块重复 step）。
        l1_pre_done = False

        # 控制器看到的状态估计（含量测噪声）。
        x_meas = (
            corrupt_state_estimate(dist_params, t_k, x, nq, nv, est_rng)
            if dist_on and dist_params.est_enable
            else x
        )

        _t_dompc = time.perf_counter()
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
                # L1 补偿走力层：把上一拍补偿加速度 a_ac（世界系，LPF 平滑，单拍滞后
                # 可忽略）前馈进 a_des，让 geometric 自己构造倾转姿态并飞过去；
                # 不再在 ctbr 块对 geometric 事后叠 tilt（见下方 geometric_on 门控）。
                a_ff_geo = (
                    np.asarray(l1_aug.a_ac, dtype=float).reshape(3)
                    if l1_aug is not None
                    else None
                )
                T_geo, omega_geo = _geometric_baseline_command(
                    p_now, v_world_now, R_meas_g, omega_now,
                    p_ref, v_ref, a_ref, yaw_ref, nominal_mass, GRAVITY,
                    kp_pos=geo_kp_pos, kd_vel=geo_kd_vel, kR=geo_kR,
                    kOmega=geo_kOmega, max_tilt_deg=geo_max_tilt,
                    a_ff_world=a_ff_geo,
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
                yref_running: List[np.ndarray] = []
                for i in range(N):
                    ti = t_k + i * dt_mpc
                    xr = interp_full_state_piecewise(ti, t_plan, x_plan, pin_model)
                    yref = _state_to_yref(xr, u_hover, nq, n_arm)
                    solver.cost_set(i, "yref", yref, api="new")
                    yref_running.append(yref)
                xrN = interp_full_state_piecewise(t_k + N * dt_mpc, t_plan, x_plan, pin_model)
                yref_terminal = _state_to_yref_e(xrN, nq, n_arm)
                solver.cost_set(N, "yref", yref_terminal, api="new")

                # 模型增广 'pre'：本拍先用当前量测做广义 L1 估计，立即用于本拍模型参数
                # （消除单拍滞后）。仅 adaptive(广义 L1) 的 in-model 走这里（oracle 走 per-stage）。
                # 补偿全在模型层，事后推力增量恒 0；广义 L1 产出 base 6D wrench + 臂关节力矩。
                if dist_aware_pre and gen_est is not None:
                    R_meas_pre = _quat_to_R_np(x_meas[3:7])
                    b3_pre = R_meas_pre[:, 2]
                    v_world_pre = R_meas_pre @ x_meas[nq : nq + 3]
                    pos_err_pre = x_meas[0:3] - np.asarray(xr_now[0:3], dtype=float)
                    # 位置积分通道仍由 l1_aug 维护（其平动 σ̂ 不再用于喂模型）。
                    if l1_aug is not None:
                        l1_aug.step(
                            control_dt, v_world_pre, l1_last_a_applied,
                            pos_err_world=pos_err_pre, R_bw=R_meas_pre,
                        )
                    if obs_grasp_reset_t >= 0.0 and not mom_grasp_done and (
                        t_k >= obs_grasp_reset_t
                    ):
                        gen_est.reset(
                            np.asarray(x_meas[:nq], dtype=float),
                            np.asarray(x_meas[nq : nq + nv], dtype=float),
                        )
                        mom_grasp_done = True
                    tau_app_pre = _tau_applied_generalized(
                        u_apply, A_alloc, nv, nu, n_rotors
                    )
                    gen_est.step(
                        np.asarray(x_meas[:nq], dtype=float),
                        np.asarray(x_meas[nq : nq + nv], dtype=float),
                        tau_app_pre, control_dt,
                    )
                    # 求解前尚无本拍 T_baseline，用上一拍 MPC 指令 u_apply 的总推力。
                    T_cmd_pre = float(np.sum(u_apply[:n_rotors]))
                    l1_last_a_applied = (
                        (T_cmd_pre / max(nominal_mass, 1e-6)) * b3_pre
                        - np.array([0.0, 0.0, GRAVITY])
                    )
                    l1_thrust_delta = np.zeros(nu, dtype=float)
                    l1_thrust_delta_l1 = np.zeros(nu, dtype=float)
                    l1_thrust_delta_pos = np.zeros(nu, dtype=float)
                    # 广义 L1：base 6D 世界系 wrench → p[:6]，臂关节力矩 → p[6:]。
                    sigma_for_mpc = np.zeros(n_dist_param, dtype=float)
                    _fill_sigma_mom_wb(
                        sigma_for_mpc, gen_est, R_meas_pre,
                        n_dist_param, n_arm, arm_only=False,
                    )
                    _bw_pre = gen_est.base_wrench_world(R_meas_pre)
                    est_base_F_w = np.asarray(_bw_pre[:3], dtype=float)
                    est_base_M_w = np.asarray(_bw_pre[3:6], dtype=float)
                    if n_arm > 0:
                        _ae = np.asarray(gen_est.arm_torque_ext(), dtype=float).reshape(-1)
                        est_arm_tau = np.zeros(n_arm, dtype=float)
                        est_arm_tau[: min(n_arm, _ae.size)] = _ae[:n_arm]
                    l1_pre_done = True

                # disturbance-aware MPC：下发世界系扰动力参数 p_dist_w 给 horizon。
                #  • oracle（per-stage）：逐 stage 按预测时刻的解析真值下发 → 提前预倾；
                #  • L1 估计：无未来信息，全程用上一拍 σ̂·m 常值保持；
                #  • 后嵌/非增广：恒 0（名义模型）。
                if dist_aware_perstage:
                    # 位置误差积分反馈（in_model 专用）：前馈只把"已知扰动"喂进模型，MPC 是
                    # 有限权重的比例式调节器，对常值扰动天然留有稳态偏差（提高 w_pos/w_att
                    # 只能缩小不能归零）。把积分+比例位置反馈折算成"附加等效扰动力"叠加进
                    # p[:3]，MPC 会多倾转一点去抵消它 → 真正的零稳态误差（offset-free）。
                    # 注：l1_aug.a_pos 用上一拍值（step_oracle 在本块之后才更新），单拍滞后。
                    pos_fb_force_w = np.zeros(3, dtype=float)
                    if pos_fb_on and l1_aug is not None:
                        pos_fb_force_w = -nominal_mass * np.asarray(
                            l1_aug.a_pos, dtype=float
                        ).reshape(3)
                    for i in range(N + 1):
                        ti = t_k + i * dt_mpc
                        xi = (
                            x_prev[i]
                            if (x_prev is not None and i < len(x_prev))
                            else x_meas
                        )
                        ui = (
                            u_prev[i]
                            if (u_prev is not None and i < len(u_prev))
                            else u_hover
                        )
                        Ri = _quat_to_R_np(np.asarray(xi[3:7], dtype=float))
                        vi_w = Ri @ np.asarray(xi[nq : nq + 3], dtype=float)
                        Ti = float(np.sum(np.asarray(ui)[:n_rotors]))
                        # base 等效世界系 6D wrench（base+EE+负载重力等效）；whole-body
                        # 时再取 EE/负载经 J^T 折算的臂关节力矩，喂给 p[6:]（否则 EE 侧向力
                        # 压弯机械臂的关节力矩没有任何通道补偿 → 稳态 EE 侧偏无法消除）。
                        _want_arm = n_dist_param > 6 and n_arm > 0
                        if _want_arm:
                            F_w_i, M_w_i, tau_arm_i = disturbance_base_equiv_world(
                                dist_params, ti, np.asarray(xi[:nq], dtype=float), Ri,
                                vi_w, Ti, n_rotors, pin_model, dist_data, ee_id, nv,
                                n_arm, return_arm=True,
                            )
                        else:
                            F_w_i, M_w_i = disturbance_base_equiv_world(
                                dist_params, ti, np.asarray(xi[:nq], dtype=float), Ri,
                                vi_w, Ti, n_rotors, pin_model, dist_data, ee_id, nv,
                                n_arm,
                            )
                            tau_arm_i = None
                        dT_i = 0.0
                        if dist_on and _dist_active(
                            ti, dist_params.thrust_enable,
                            dist_params.thrust_t0, dist_params.thrust_t1,
                        ):
                            dT_i = (
                                (dist_params.thrust_scale - 1.0) * Ti
                                + dist_params.thrust_bias
                            )
                        # 6D wrench：力 = 外力 + 推力偏差等效力；力矩 = 外力矩（世界系）。
                        p_i = np.zeros(n_dist_param, dtype=float)
                        p_i[:3] = F_w_i + dT_i * Ri[:, 2] + pos_fb_force_w
                        p_i[3:6] = np.asarray(M_w_i, dtype=float).reshape(3)
                        if tau_arm_i is not None:
                            n_take = min(n_dist_param - 6, int(n_arm), tau_arm_i.size)
                            p_i[6 : 6 + n_take] = np.asarray(
                                tau_arm_i, dtype=float
                            )[:n_take]
                        solver.set(i, "p", p_i)
                else:
                    p_dist_stage = (
                        sigma_for_mpc if dist_aware_on
                        else np.zeros(n_dist_param, dtype=float)
                    )
                    for i in range(N + 1):
                        solver.set(i, "p", p_dist_stage)

                t0 = time.perf_counter()
                status = int(solver.solve())
                wall_s = time.perf_counter() - t0
                n_iter = _acados_nlp_iterations(solver)

                mpc_solve_steps.append(k)
                mpc_iters.append(n_iter)
                mpc_wall_s.append(wall_s)
                _prof("mpc_solve", wall_s)

                # ── Cost analysis：总代价（get_cost）+ 分项拆解（与 Crocoddyl 对齐）──
                _t_cost = time.perf_counter()
                try:
                    _total_cost = float(solver.get_cost())
                except Exception:
                    _total_cost = float("nan")
                mpc_costs.append(_total_cost)
                mpc_solve_t.append(float(t_k))
                # 分项拆解开销不小（每控制步遍历 horizon 读 solver.get）。cost_analysis=False
                # 时跳过，仅保留总代价（get_cost 很便宜），分项填 NaN 保持数组对齐。
                if cost_analysis:
                    try:
                        _grp = _acados_cost_term_breakdown(
                            solver, N, nq, n_arm, W_state_diag, R_diag,
                            w_terminal_track, yref_running, yref_terminal,
                        )
                    except Exception:
                        _grp = {}
                    for _ck in cost_term_keys:
                        mpc_cost_terms_hist[_ck].append(float(_grp.get(_ck, float("nan"))))
                else:
                    for _ck in cost_term_keys:
                        mpc_cost_terms_hist[_ck].append(float("nan"))
                _prof("cost_breakdown", time.perf_counter() - _t_cost)

                if mpc_log_interval > 0 and len(mpc_solve_steps) % mpc_log_interval == 0:
                    print(
                        f"[acados track] t={t_k:.3f} status={status} iter={n_iter} wall={wall_s*1e3:.1f} ms"
                    )

                x_prev = [solver.get(i, "x") for i in range(N + 1)]
                u_prev = [solver.get(i, "u") for i in range(N)]
                u_apply = np.asarray(solver.get(0, "u"), dtype=float).flatten().copy()
                T_baseline = float(np.sum(u_apply[:n_rotors]))
                ctbr_omega_base = None

            # ── 扰动估计 + 注入：估计完整广义外力 → bolt-on(u_ad) 或 in-model(模型增广) ──
            # 'pre' 模式本拍已在求解前 step 过 → 跳过，避免估计器被二次步进。
            _t_est = time.perf_counter()
            if (gen_est is not None or oracle_on) and not l1_pre_done:
                R_meas = _quat_to_R_np(x_meas[3:7])
                b3 = R_meas[:, 2]
                v_world = R_meas @ x_meas[nq : nq + 3]
                T_cmd = float(T_baseline)
                pos_err = (x_meas[0:3] - np.asarray(xr_now[0:3], dtype=float))

                # ── 估计/真值：base 世界系 wrench [F_w(3), M_w(3)] + 臂关节力矩 τ_arm(n) ──
                base_F_w = np.zeros(3, dtype=float)
                base_M_w = np.zeros(3, dtype=float)
                arm_tau = np.zeros(n_arm, dtype=float) if n_arm > 0 else None
                if oracle_on:
                    # Oracle：用扰动真值（base 等效世界系 wrench + 臂关节力矩），与 plant
                    # 侧 dist_force_log 同口径，外加推力估计偏差等效力（沿机体 z）。
                    F_w_o, M_w_o, tau_arm_o = disturbance_base_equiv_world(
                        dist_params, t_k, x_meas[:nq], R_meas, v_world,
                        float(T_baseline), n_rotors, pin_model, dist_data, ee_id,
                        nv, n_arm, return_arm=True,
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
                    base_F_w = np.asarray(F_w_o, dtype=float).reshape(3) + dT_o * b3
                    base_M_w = np.asarray(M_w_o, dtype=float).reshape(3)
                    if n_arm > 0 and tau_arm_o is not None:
                        ta = np.asarray(tau_arm_o, dtype=float).reshape(-1)
                        arm_tau = np.zeros(n_arm, dtype=float)
                        arm_tau[: min(n_arm, ta.size)] = ta[:n_arm]
                else:
                    # adaptive：广义 L1 在广义动量上一阶自适应，估计完整广义外力 τ̂_ext。
                    if obs_grasp_reset_t >= 0.0 and not mom_grasp_done and (
                        t_k >= obs_grasp_reset_t
                    ):
                        gen_est.reset(
                            np.asarray(x_meas[:nq], dtype=float),
                            np.asarray(x_meas[nq : nq + nv], dtype=float),
                        )
                        mom_grasp_done = True
                    tau_app = _tau_applied_generalized(u_apply, A_alloc, nv, nu, n_rotors)
                    gen_est.step(
                        np.asarray(x_meas[:nq], dtype=float),
                        np.asarray(x_meas[nq : nq + nv], dtype=float),
                        tau_app, control_dt,
                    )
                    bw = gen_est.base_wrench_world(R_meas)
                    base_F_w = np.asarray(bw[:3], dtype=float)
                    base_M_w = np.asarray(bw[3:6], dtype=float)
                    if n_arm > 0:
                        ae = np.asarray(gen_est.arm_torque_ext(), dtype=float).reshape(-1)
                        arm_tau = np.zeros(n_arm, dtype=float)
                        arm_tau[: min(n_arm, ae.size)] = ae[:n_arm]

                # base 平动补偿加速度（世界系）σ=F_ext/m，补偿=-σ。复用 l1_aug 的 dFz/tilt
                # + 位置积分管线（step_oracle = 给定 σ̂ 算补偿，不经预测器）。
                sigma_world = base_F_w / max(nominal_mass, 1e-6)
                if l1_aug is not None:
                    a_ac = l1_aug.step_oracle(
                        control_dt, sigma_world, pos_err_world=pos_err, R_bw=R_meas
                    )
                else:
                    a_ac = np.zeros(3, dtype=float)

                if not comp_on:
                    # 仅估计不补偿：丢弃 L1 补偿分量，仅保留位置误差反馈 a_pos。
                    if l1_aug is not None:
                        l1_aug.a_l1 = np.zeros(3, dtype=float)
                        l1_aug._a_l1_body = np.zeros(3, dtype=float)
                        l1_aug.a_ac = np.asarray(l1_aug.a_pos, dtype=float).copy()
                        a_ac = l1_aug.a_ac.copy()
                    else:
                        a_ac = np.zeros(3, dtype=float)

                if geometric_on or dist_aware_on:
                    # in-model / geometric：补偿走模型层 / baseline 力层，不做事后推力映射。
                    l1_thrust_delta = np.zeros(nu, dtype=float)
                    l1_thrust_delta_l1 = np.zeros(nu, dtype=float)
                    l1_thrust_delta_pos = np.zeros(nu, dtype=float)
                    l1_last_a_applied = (
                        (T_cmd / max(nominal_mass, 1e-6)) * b3
                        - np.array([0.0, 0.0, GRAVITY])
                    )
                else:
                    # ── bolt-on：u = u_b + u_ad ─────────────────────────────────
                    # base 竖直/水平力 → dFz + 倾转（复用 a_ac 管线）；base 转动力矩 →
                    # 推力差动 -M_b；臂关节（全驱动）→ 直接补 -τ̂_arm。
                    dFz = nominal_mass * float(np.dot(a_ac, b3))
                    a_nom = (T_cmd / max(nominal_mass, 1e-6)) * b3 - np.array([0.0, 0.0, GRAVITY])
                    f_des = a_nom + a_ac + np.array([0.0, 0.0, GRAVITY])
                    nfd = float(np.linalg.norm(f_des))
                    M_body = np.zeros(3)
                    if nfd > 1e-6 and l1_aug is not None and l1_aug.params.tilt_gain > 0.0:
                        b3_des = f_des / nfd
                        e_tilt = np.cross(b3, b3_des)
                        M_body = R_meas.T @ (float(l1_aug.params.tilt_gain) * e_tilt)
                    # base 转动力矩补偿（机体系）：抵消外部 base 力矩 → -M_b（仅 comp_on）。
                    M_base_comp = R_meas.T @ (-base_M_w) if comp_on else np.zeros(3)
                    dthr = A_inv @ np.array([
                        dFz, M_body[0] + M_base_comp[0],
                        M_body[1] + M_base_comp[1], M_base_comp[2],
                    ])
                    l1_thrust_delta = np.zeros(nu, dtype=float)
                    l1_thrust_delta[:n_rotors] = dthr
                    # 臂关节直接补偿（全驱动）：u_ad_arm = -τ̂_arm（仅 comp_on）。
                    if comp_on and n_arm > 0 and nu > n_rotors and arm_tau is not None:
                        n_take = min(nu - n_rotors, n_arm)
                        l1_thrust_delta[n_rotors : n_rotors + n_take] = -arm_tau[:n_take]

                    # ── 推力增量分解：L1 纯补偿(a_l1+base力矩+臂) 与 位置通道(a_pos) ──
                    a_l1_w = np.asarray(l1_aug.a_l1, dtype=float) if l1_aug is not None else np.zeros(3)
                    a_pos_w = np.asarray(l1_aug.a_pos, dtype=float) if l1_aug is not None else np.zeros(3)
                    dFz_l1 = nominal_mass * float(np.dot(a_l1_w, b3))
                    dFz_pos = nominal_mass * float(np.dot(a_pos_w, b3))
                    M_l1 = np.zeros(3)
                    if nfd > 1e-6 and l1_aug is not None and l1_aug.params.tilt_gain > 0.0:
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
                    # base 力矩补偿归入 L1 补偿路（M_base_comp、Mz）。
                    l1_thrust_delta_l1[:n_rotors] = A_inv @ np.array(
                        [dFz_l1, M_l1[0] + M_base_comp[0], M_l1[1] + M_base_comp[1], M_base_comp[2]]
                    )
                    l1_thrust_delta_pos[:n_rotors] = A_inv @ np.array(
                        [dFz_pos, M_pos[0], M_pos[1], 0.0]
                    )
                    if comp_on and n_arm > 0 and nu > n_rotors and arm_tau is not None:
                        n_take = min(nu - n_rotors, n_arm)
                        l1_thrust_delta_l1[n_rotors : n_rotors + n_take] = -arm_tau[:n_take]
                    # 下一拍预测器输入：实际总推力 = baseline + L1 竖直注入 dFz。
                    T_actual = T_cmd + dFz
                    l1_last_a_applied = (
                        (T_actual / max(nominal_mass, 1e-6)) * b3
                        - np.array([0.0, 0.0, GRAVITY])
                    )

                # ── in-model：填 sigma_for_mpc（base 世界系 wrench + 臂关节）下拍喂模型 ──
                sigma_for_mpc = np.zeros(n_dist_param, dtype=float)
                if dist_aware_on and comp_on:
                    sigma_for_mpc[:3] = base_F_w
                    sigma_for_mpc[3:6] = base_M_w
                    if n_dist_param > 6 and n_arm > 0 and arm_tau is not None:
                        n_take = min(n_dist_param - 6, n_arm)
                        sigma_for_mpc[6 : 6 + n_take] = arm_tau[:n_take]
                # 估计供绘图：本拍 base wrench / 臂关节力矩（与 l1_aug.sigma_hat 一致）。
                est_base_F_w = base_F_w
                est_base_M_w = base_M_w
                est_arm_tau = arm_tau
            _prof("estimation", time.perf_counter() - _t_est)

            # ── CTBR 内环：总推力 + 前瞻角速度设定点 → 角速度 PID → 分配 ──────
            _t_ctbr = time.perf_counter()
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
                # geometric 路跳过：补偿已在 a_des 力层前馈进 T_geo/omega_geo，再叠 tilt
                # 会与几何姿态环对抗（悬停 oracle 横向补不动的根因）。
                # dist_aware 路跳过：补偿已作为模型参数喂进 MPC，u_apply/horizon 已含倾转，
                # 再叠 tilt 会与 MPC 预测姿态对抗。仅 acados 后嵌 CTBR 走这里。
                if l1_aug is not None and not geometric_on and not dist_aware_on:
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
                # 角加速度对旋翼力为仿射：α(r)=α0+B·r，α0=ω̇(r=0)=xdot0−B·u_apply 为
                # 零旋翼力时的开环角加速度，含重力力矩 **与机械臂反作用力矩**（aba 对广义力
                # 线性，故 α0 精确、非线性化近似）。要让实际角加速度=w_cmd，须解 B·r=w_cmd−α0。
                # 旧实现解 B·r=w_cmd 漏掉 α0：纯四旋翼水平悬停 α0≈0 看不出，但带臂/偏置负载
                # 时 α0≠0，稳态(PID→0)旋翼不平衡 arm 反作用力矩 → base 姿态漂移 → 经耦合
                # 引起 arm joint 静差。等价于以 MPC 旋翼力为前馈、PID 仅作修正。
                alpha0 = (
                    xdot0[nq + 3 : nq + 6]
                    - B @ np.asarray(u_apply[:n_rotors], dtype=float)
                )
                A_mix = np.vstack([np.ones((1, n_rotors)), B])
                b_mix = np.concatenate([[T_cmd_c], w_cmd - alpha0])
                try:
                    rotors = np.linalg.solve(A_mix, b_mix)
                except np.linalg.LinAlgError:
                    rotors = np.linalg.lstsq(A_mix, b_mix, rcond=None)[0]
                ctbr_u_target = u_apply.copy()
                ctbr_u_target[:n_rotors] = np.clip(rotors, min_thrust, max_thrust)
            _prof("ctbr", time.perf_counter() - _t_ctbr)
            _prof("control_total", time.perf_counter() - _t_dompc)

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
                # bolt-on 臂关节直接补偿：u_ad_arm 叠加到关节力矩通道（全驱动，无需分配）。
                if nu > n_rotors:
                    u_cmd[n_rotors:] = u_cmd[n_rotors:] + l1_thrust_delta[n_rotors:]
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

            _t_inject = time.perf_counter()
            if isinstance(plant, DisturbedRK4Plant):
                R_now = _quat_to_R_np(x[3:7])
                b3_now = R_now[:, 2]
                v_world_now = R_now @ x[nq : nq + 3]
                T_real = float(np.sum(u_real[:n_rotors]))
                q_now = np.asarray(x[:nq], dtype=float)
                # base wrench（const+var+drag）→ 机体系广义力 [:6]。
                F_ext, M_body_ext, M_world_ext = external_force_torque(
                    dist_params, t_k, R_now, v_world_now, T_real, n_rotors
                )
                tau_ext = np.zeros(nv, dtype=float)
                tau_ext[:3] = R_now.T @ F_ext
                tau_ext[3:6] = M_body_ext
                # EE/负载都需 EE 帧位姿：合并一次 framesForwardKinematics（省一次 FK）。
                _ee_on = dist_params.any_ee_disturbance() and n_arm > 0
                _load_now = load_on and _dist_active(
                    t_k, True, dist_params.load_t0, dist_params.load_t1
                )
                if _ee_on or _load_now:
                    pin.framesForwardKinematics(pin_model, dist_data, q_now)
                # EE-link 力旋量 → 经 LOCAL_WORLD_ALIGNED 雅可比折算到完整广义力。
                if _ee_on:
                    R_ee = np.asarray(dist_data.oMf[ee_id].rotation, dtype=float)
                    F_ee_w, M_ee_w = ee_wrench_world(dist_params, t_k, R_ee)
                    if np.any(np.abs(F_ee_w) > 1e-12) or np.any(np.abs(M_ee_w) > 1e-12):
                        J_ee = pin.computeFrameJacobian(
                            pin_model, dist_data, q_now, ee_id, pin.LOCAL_WORLD_ALIGNED
                        )
                        tau_ext += J_ee.T @ np.concatenate([F_ee_w, M_ee_w])
                plant.set_tau_ext_full(tau_ext)
                # 推力估计偏差等效力（沿机体 z，相对 MPC 基线 u_apply）。
                dT_base = 0.0
                if _dist_active(
                    t_k, dist_params.thrust_enable, dist_params.thrust_t0, dist_params.thrust_t1
                ):
                    T_base = float(np.sum(u_apply[:n_rotors]))
                    dT_base = (dist_params.thrust_scale - 1.0) * T_base + dist_params.thrust_bias
                # 负载真值（准静态重力等效，仅用于日志/oracle；plant 已用增广模型精确积分）。
                F_load_w = np.zeros(3, dtype=float)
                M_load_w = np.zeros(3, dtype=float)
                if _load_now:
                    oMf = dist_data.oMf[ee_id]
                    com_off = np.array(
                        [dist_params.load_com_x, dist_params.load_com_y, dist_params.load_com_z],
                        dtype=float,
                    )
                    p_com_w = np.asarray(oMf.translation, dtype=float) + (
                        np.asarray(oMf.rotation, dtype=float) @ com_off
                    )
                    F_load_w = float(dist_params.load_mass) * np.array(
                        [0.0, 0.0, -GRAVITY], dtype=float
                    )
                    M_load_w = np.cross(p_com_w - np.asarray(x[:3], dtype=float), F_load_w)
                # 真值扰动（世界系，base 等效）：base+EE 经 [:6] 折算 + 推力偏差 + 负载重力等效。
                dist_force_log[k] = (
                    R_now @ tau_ext[:3] + dT_base * b3_now + F_load_w
                )
                dist_force_body_log[k] = R_now.T @ dist_force_log[k]
                dist_torque_world_log[k] = R_now @ tau_ext[3:6] + M_load_w
                dist_torque_body_log[k] = R_now.T @ dist_torque_world_log[k]
                if dist_joint_torque_log is not None and n_arm > 0:
                    dist_joint_torque_log[k, :] = tau_ext[6 : 6 + n_arm]

            if l1_aug is not None or gen_est is not None:
                R_b = _quat_to_R_np(x[3:7])
                # 广义 L1 / oracle 估计：base 世界系力/力矩 + 臂关节力矩（与 plant 真值对比）。
                l1_force_log[k] = est_base_F_w
                l1_torque_log[k] = est_base_M_w
                l1_sigma_log[k] = est_base_F_w / max(nominal_mass, 1e-6)
                if l1_aug is not None:
                    l1_aac_log[k] = l1_aug.a_ac
                    l1_al1_log[k] = l1_aug.a_l1
                else:
                    l1_al1_log[k] = l1_sigma_log[k]
                if l1_joint_torque_log is not None and n_arm > 0 and est_arm_tau is not None:
                    _ae = np.asarray(est_arm_tau, dtype=float).reshape(-1)
                    l1_joint_torque_log[k, : min(n_arm, _ae.size)] = _ae[:n_arm]
                # 机体系版本：用本拍真实姿态把世界系估计/补偿旋到机体系一并记录。
                l1_force_body_log[k] = R_b.T @ l1_force_log[k]
                l1_al1_body_log[k] = R_b.T @ l1_al1_log[k]
            _prof("dist_inject_log", time.perf_counter() - _t_inject)

            _t_plant = time.perf_counter()
            x = plant.step(x, u_real)
            _prof("plant_step", time.perf_counter() - _t_plant)
            t_log[k + 1] = (k + 1) * sim_dt
            x_log[k + 1] = x

        if progress_cb is not None:
            progress_cb(k + 1, n_steps)

    # ── 运行时间分解汇总：按耗时排序打印，并随结果返回（GUI 可展示）──────────
    loop_wall = time.perf_counter() - _loop_t0
    n_ctrl = max(1, len(mpc_solve_steps))
    timing_summary = {
        "loop_wall_s": float(loop_wall),
        "n_steps": int(n_steps),
        "n_control_steps": int(n_ctrl),
        "sim_dt": float(sim_dt),
        "control_dt": float(control_dt),
        "rtf": float((n_steps * sim_dt) / loop_wall) if loop_wall > 0 else float("nan"),
        "sections": {
            name: {
                "total_s": float(prof_t[name]),
                "calls": int(prof_n.get(name, 0)),
                "avg_ms": float(prof_t[name] / max(1, prof_n.get(name, 1)) * 1e3),
                "pct": float(100.0 * prof_t[name] / loop_wall) if loop_wall > 0 else 0.0,
            }
            for name in prof_t
        },
    }
    try:
        print(
            f"[acados track] 仿真耗时分解  total={loop_wall:.2f}s  "
            f"sim_steps={n_steps} (dt={sim_dt*1e3:.1f}ms)  "
            f"control_steps={n_ctrl} (dt={control_dt*1e3:.1f}ms)  "
            f"RTF={timing_summary['rtf']:.2f}x"
        )
        for name, s in sorted(
            timing_summary["sections"].items(), key=lambda kv: -kv[1]["total_s"]
        ):
            print(
                f"    {name:18s} {s['total_s']:7.3f}s  {s['pct']:5.1f}%  "
                f"calls={s['calls']:6d}  avg={s['avg_ms']:.3f}ms"
            )
    except Exception:
        pass

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
        "mpc_costs": np.asarray(mpc_costs, dtype=float),
        "mpc_solve_t": np.asarray(mpc_solve_t, dtype=float),
        "mpc_cost_terms": {
            k: np.asarray(v, dtype=float) for k, v in mpc_cost_terms_hist.items()
        },
        "mpc_cost_groups": {},
        "mpc_cost_weights": {k: float(v) for k, v in mpc_cost_weights.items()},
        "timing": timing_summary,
        "mpc": shim,
        "control_mode": control_mode,
        "dt_mpc": dt_mpc,
        "horizon": N,
        "state_limits": limits,
        "disturbance_active": bool(dist_on),
        "l1_active": bool(l1_on),
        "pos_fb_active": bool(pos_fb_on),
        "comp_mode": ("oracle" if oracle_on else "adaptive"),
        "oracle_active": bool(oracle_on),
        "gen_l1_active": bool(gen_l1_on),
        "nominal_mass": nominal_mass,
        "dist_force_world": dist_force_log,
        "dist_force_body": dist_force_body_log,
        "dist_torque_body": dist_torque_body_log,
        "dist_torque_world": dist_torque_world_log,
        "l1_force_world": l1_force_log,
        "l1_torque_world": l1_torque_log,
        "dist_joint_torque": dist_joint_torque_log,
        "l1_joint_torque": l1_joint_torque_log,
        "n_arm": int(n_arm),
        "l1_force_body": l1_force_body_log,
        "l1_sigma": l1_sigma_log,
        "l1_a_ac": l1_aac_log,
        "l1_a_l1": l1_al1_log,
        "l1_a_l1_body": l1_al1_body_log,
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
        "inject_mode": ("in_model" if dist_aware_on else "bolt_on"),
        "dist_aware_active": bool(dist_aware_on),
        "dist_aware_perstage": bool(dist_aware_perstage),
        "dist_aware_update": ("pre" if dist_aware_pre else "post"),
        "obs_wholebody": bool(obs_wholebody),
        "n_dist_param": int(n_dist_param),
    }


def acados_closed_loop_to_ee_tracking_res(out: Dict[str, Any]) -> Dict[str, Any]:
    from s500_uam_crocoddyl_state_tracking_mpc import crocoddyl_closed_loop_to_ee_tracking_res

    return crocoddyl_closed_loop_to_ee_tracking_res(out)
