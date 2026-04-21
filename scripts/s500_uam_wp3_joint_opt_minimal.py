#!/usr/bin/env python3
"""
最小示例：从 URDF 经 s500_uam_acados_model 建立动力学，一条 OCP 联合优化 3 个航点。

- 动力学：build_acados_model()（Pinocchio + CasADi，与主工程一致）
- 航点：中间/终点用路径/终端约束 h(x)=q−p（9D 位形），p 为参数；非航点段放松
- 代价：NONLINEAR_LS，见 build_ocp() 内长注释（便于你改目标/加权/新项）

运行（在 scripts 目录或设置 PYTHONPATH）:
  python s500_uam_wp3_joint_opt_minimal.py
画图调用 s500_uam_acados_trajectory_plot.plot_acados_into_figure（与 trajectory GUI 主图 4×4 状态/控制一致）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
import pinocchio as pin
from pinocchio import casadi as cpin


def _preload_acados_shared_libs():
    """在加载 libacados 之前用绝对路径预载 qpOASES/hpipm/blasfeo（与 s500_uam_acados_trajectory 相同）。"""
    if os.name == "nt":
        return
    try:
        from ctypes import CDLL
    except ImportError:
        return
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if not root:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "acados"))
    libdir = os.path.join(root, "lib")
    if not os.path.isdir(libdir):
        return
    for name in ("libblasfeo.so.0", "libqpOASES_e.so", "libhpipm.so"):
        path = os.path.join(libdir, name)
        if os.path.isfile(path):
            CDLL(path)


def _ensure_acados_source_dir_env():
    """如果未设置 ACADOS_SOURCE_DIR，则按本仓库常见布局自动补齐，避免 acados_template 重复告警。"""
    if os.environ.get("ACADOS_SOURCE_DIR"):
        return
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "acados")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "acados")),
    ]
    for root in candidates:
        if os.path.isdir(os.path.join(root, "lib")) and os.path.isdir(os.path.join(root, "interfaces")):
            os.environ["ACADOS_SOURCE_DIR"] = root
            return


_preload_acados_shared_libs()
_ensure_acados_source_dir_env()

import matplotlib.pyplot as plt

from s500_uam_acados_model import build_acados_model, load_s500_config
from s500_uam_acados_trajectory_plot import CONTROL_INPUT_DIRECT, plot_acados_into_figure

# ---------------------------------------------------------------------------
# 物理/限位（与 trajectory GUI 一致的量级；改进算法时可单独调）
# ---------------------------------------------------------------------------
LIMITS = {
    "v_max": 2.0,
    "omega_max": 2.0,
    "j_angle_max": 2.0,
    "j_vel_max": np.deg2rad(20.0),  # 20 deg/s
}


# ========== 网格：两段时长 -> 射击步数 N 与航点节点索引 ==========
def shooting_nodes(durations: list[float], dt: float) -> tuple[float, int, list[int]]:
    """返回 (tf, N, nodes)；nodes[m] 为第 m 个航点所在节点 (m=0..2)，nodes[-1]==N。"""
    tf = float(sum(durations))
    parts = [max(1, int(round(float(d) / float(dt)))) for d in durations]
    N = int(sum(parts))
    nodes = [0]
    for p in parts:
        nodes.append(nodes[-1] + p)
    assert nodes[-1] == N
    return tf, N, nodes


# ========== 小工具：四元数、插值、代价参考向量 ==========
def _norm_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    return np.array([0.0, 0.0, 0.0, 1.0]) if n < 1e-12 else q / n


def _interp_x17(a: float, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    a = float(np.clip(a, 0.0, 1.0))
    o = (1.0 - a) * xa.reshape(17) + a * xb.reshape(17)
    o[3:7] = _norm_q(o[3:7])
    return o


def state17_at_time(t: float, wps: list[np.ndarray], durs: list[float]) -> np.ndarray:
    """分段线性 17D 状态（四元数 nlerp），用于初值猜测。"""
    t0 = [0.0]
    for d in durs:
        t0.append(t0[-1] + float(d))
    t = float(t)
    if t <= t0[0]:
        return np.asarray(wps[0], dtype=float).reshape(17).copy()
    if t >= t0[-1]:
        return np.asarray(wps[-1], dtype=float).reshape(17).copy()
    for m in range(len(durs)):
        ta, tb = t0[m], t0[m + 1]
        if t <= tb + 1e-12:
            alpha = (t - ta) / max(tb - ta, 1e-9)
            return _interp_x17(alpha, wps[m], wps[m + 1])
    return np.asarray(wps[-1], dtype=float).reshape(17).copy()


def q9_from_state17(s: np.ndarray) -> np.ndarray:
    q = np.asarray(s[:9], dtype=float).copy()
    q[3:7] = _norm_q(q[3:7])
    return q


def quat_to_R_expr(quat):
    """CasADi: quat [qx,qy,qz,qw] -> R."""
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        ca.horzcat(2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)),
        ca.horzcat(2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)),
    )


def so3_residual_expr(R_cur, R_ref):
    """SO(3) 残差（李代数近似）：0.5*vee(R_ref^T R - R^T R_ref)。"""
    E = ca.mtimes(R_ref.T, R_cur) - ca.mtimes(R_cur.T, R_ref)
    return 0.5 * ca.vertcat(E[2, 1], E[0, 2], E[1, 0])


def hover_direct_control_ref(pin_model, min_thrust: float, max_thrust: float) -> np.ndarray:
    """direct 模型下的控制参考：四旋翼各电机悬停推力 clip(mg/4)，机械臂 τ1=τ2=0。"""
    m = float(sum(inertia.mass for inertia in pin_model.inertias))
    t_each = float(np.clip(m * 9.81 / 4.0, min_thrust, max_thrust))
    return np.concatenate([np.full(4, t_each), np.zeros(2)])


def ee_kinematics_expr(acados_model, pin_model, frame_id: int):
    """返回 EE 的 (pos_world, R_world, v_lin_world) CasADi 表达式。"""
    nq = pin_model.nq
    q = acados_model.x[:nq]
    v = acados_model.x[nq : nq + pin_model.nv]
    quat = q[3:7]
    quat_u = quat / ca.fmax(ca.norm_2(quat), 1e-9)
    q_fk = ca.vertcat(q[:3], quat_u, q[7:nq])
    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()
    cpin.forwardKinematics(cmodel, cdata, q_fk, v)
    cpin.updateFramePlacements(cmodel, cdata)
    pose = cdata.oMf[frame_id]
    vel = cpin.getFrameVelocity(cmodel, cdata, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return pose.translation, pose.rotation, vel.linear


def quat_to_R_np(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = _norm_q(quat)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=float,
    )


def so3_residual_np(R_cur: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    E = R_ref.T @ R_cur - R_cur.T @ R_ref
    return 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]], dtype=float)


def euler_zyx_to_quat_np(rpy_rad: np.ndarray) -> np.ndarray:
    """ZYX 欧拉角 [roll,pitch,yaw] -> 四元数 [qx,qy,qz,qw]。"""
    r, p, y = np.asarray(rpy_rad, dtype=float).reshape(3)
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return _norm_q(np.array([qx, qy, qz, qw], dtype=float))


def ee_obs9_from_state17(pin_model, frame_id: int, x17: np.ndarray, q_ref_ee: np.ndarray) -> np.ndarray:
    """NumPy: [pos(3), so3_res(3), v_lin_world(3)] from a 17D waypoint state."""
    x = np.asarray(x17, dtype=float).reshape(17)
    q = x[:9].copy()
    q[3:7] = _norm_q(q[3:7])
    v = x[9:17].copy()
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q, v)
    pin.updateFramePlacements(pin_model, data)
    pose = data.oMf[frame_id]
    R_ref = quat_to_R_np(q_ref_ee)
    r_so3 = so3_residual_np(np.asarray(pose.rotation), R_ref)
    vel = pin.getFrameVelocity(pin_model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return np.concatenate([np.asarray(pose.translation).flatten(), r_so3, np.asarray(vel.linear).flatten()])


def _codegen_flags(export_dir: Path, model_name: str) -> tuple[bool, bool]:
    try:
        from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix
    except ImportError:
        return True, True
    lib = f"{get_shared_lib_prefix()}acados_ocp_solver_{model_name}{get_shared_lib_ext()}"
    d = Path(export_dir)
    if (d / lib).is_file():
        return False, False
    if (d / "Makefile").is_file():
        return False, True
    return True, True


# ========== OCP 构造：代价与约束分块写清 ==========
def build_ocp_with_ctrl_error(
    waypoints: list[np.ndarray],
    durations: list[float],
    dt: float,
    *,
    state_weight: float = 1.0,
    control_weight: float = 1,
    terminal_scale: float = 1.0,
    max_iter: int = 10,
    pos_err_gain: np.ndarray | list[float] | float = (0.08, 0.08, 0.08),
    enable_ctrl_error: bool = True,
):
    """
    单条 OCP：3 航点、联合优化。waypoints: 3×17D；durations: 2 段时长。

    Returns
    -------
    solver, N, pin_model
    """
    if len(waypoints) != 3 or len(durations) != 2:
        raise ValueError("需要 3 个航点、2 段时长")

    tf, N, nodes = shooting_nodes(durations, dt)
    acados_model, pin_model, nq, nv, nu = build_acados_model()
    cfg = load_s500_config()
    plat = cfg["platform"]
    mn, mx = float(plat["min_thrust"]), float(plat["max_thrust"])

    frame_id = pin_model.getFrameId("gripper_link")
    if frame_id < 0 or frame_id >= pin_model.nframes:
        raise ValueError("URDF 中未找到 gripper_link，用于中间航点 EE 约束。")
    ee_pos, ee_R, ee_v = ee_kinematics_expr(acados_model, pin_model, frame_id)

    # --- 模型参数 p ∈ R^27：前17维=全状态目标，后10维=EE [pos(3),quat(4),v(3)] 目标 ---
    p = ca.SX.sym("p_wp", 27)
    acados_model.p = p
    acados_model.name = "s500_uam_wp3min_ctrlerr" if enable_ctrl_error else "s500_uam_wp3min_baseline"
    x = acados_model.x
    q_ee_ref = p[20:24] / ca.fmax(ca.norm_2(p[20:24]), 1e-9)
    R_ee_ref = quat_to_R_expr(q_ee_ref)
    ee_so3 = so3_residual_expr(ee_R, R_ee_ref)
    k = np.asarray(pos_err_gain, dtype=float).reshape(-1)
    if k.size == 1:
        k = np.full(3, float(k[0]))
    if k.size != 3:
        raise ValueError("pos_err_gain 必须是标量或3维向量")
    # 位置控制误差模型：e_p = K * v_base_world（baselink速度越大，位置控制误差越大）
    quat_b = x[3:7] / ca.fmax(ca.norm_2(x[3:7]), 1e-9)
    R_bw = quat_to_R_expr(quat_b)  # body->world
    v_base_world = ca.mtimes(R_bw, x[9:12])
    p_ctrl_err = ca.diag(ca.DM(k)) @ v_base_world
    if enable_ctrl_error:
        acados_model.con_h_expr = ca.vertcat(x - p[:17], ee_pos - p[17:20], ee_so3, ee_v - p[24:27], p_ctrl_err)
        acados_model.con_h_expr_e = ca.vertcat(x - p[:17], ee_pos - p[17:20], ee_so3, ee_v - p[24:27], p_ctrl_err)
    else:
        acados_model.con_h_expr = ca.vertcat(x - p[:17], ee_pos - p[17:20], ee_so3, ee_v - p[24:27])
        acados_model.con_h_expr_e = ca.vertcat(x - p[:17], ee_pos - p[17:20], ee_so3, ee_v - p[24:27])

    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.dims.N = N
    ocp.solver_options.tf = tf
    ocp.solver_options.nlp_solver_max_iter = max_iter
    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = N

    # =========================================================================
    # 代价函数（NONLINEAR_LS）— 改进算法时主要改这里
    # -------------------------------------------------------------------------
    # acados 形式：阶段 k=0..N-1  l_k = 1/2 * || y(x,u) - yref ||_W^2
    #            终端      l_N = 1/2 * || y_N(x) - yref_e ||_{W_e}^2
    #
    # y(x,u) 的设计（姿态误差统一改为 SO(3) 残差）：
    #   - 位置 x,y,z
    #   - 姿态用 base 的 so3 残差 3 维（相对终点姿态）
    #   - 关节角 j1,j2
    #   - 速度 v(3), ω(3), jdot(2)  — 全部来自状态 x[9:17]
    #   - 控制 u（推力×4 + 关节力矩×2）叠在同一向量里，与状态一起进 W 的块对角加权
    #
    # 参考 yref / yref_e：状态拉向「终点航点」特征，其中姿态残差参考恒为0。
    #   running 的 u 参考为悬停四推力 + 机械臂零力矩（hover_direct_control_ref），不是 u≡0。
    #   终端 cost_y_e 不含 u，故无控制参考项。
    #   中间航点主要靠**硬约束** h=0 锁 q；若改为软航点，可把航点项加进 y 或加 CONVEX_OVER_NONLINEAR。
    #
    # 权重含义（可按需改相对比例 = 改轨迹“更贴参考 / 更省推力”）：
    #   W_state：各状态通道对角权重；terminal_scale 放大 W_e 使终点更贴 yref_e
    #   R：对 u 的惩罚；推力与力矩可不同量级（多旋翼常推力更便宜、力矩更贵）
    # =========================================================================
    q_base_ref = _norm_q(np.asarray(waypoints[-1], dtype=float).reshape(17)[3:7])
    R_base_ref = quat_to_R_expr(ca.DM(q_base_ref))
    R_base = quat_to_R_expr(x[3:7] / ca.fmax(ca.norm_2(x[3:7]), 1e-9))
    base_so3 = so3_residual_expr(R_base, R_base_ref)
    cost_y = ca.vertcat(x[0:3], base_so3, x[7:9], x[9:17], ocp.model.u)
    cost_y_e = ca.vertcat(x[0:3], base_so3, x[7:9], x[9:17])

    tgt = np.asarray(waypoints[-1], dtype=float).reshape(17)
    yref_s = np.concatenate([tgt[0:3], np.zeros(3), tgt[7:9], tgt[9:17]])
    u_ref = hover_direct_control_ref(pin_model, mn, mx)
    yref = np.concatenate([yref_s, u_ref])
    yref_e = yref_s

    w_pos = 1.0 * state_weight
    w_so3 = 30.0 * state_weight
    w_jq = 1.0 * state_weight
    w_v = 1.0 * state_weight
    w_omega = 1.0 * state_weight
    w_jdot = 1.0 * state_weight
    W_state = np.diag(
        [w_pos, w_pos, w_pos, w_so3, w_so3, w_so3, w_jq, w_jq, w_v, w_v, w_v, w_omega, w_omega, w_omega, w_jdot, w_jdot]
    )
    r_thrust = control_weight
    r_torque = control_weight * 1.0
    R = np.diag([r_thrust] * 4 + [r_torque] * 2)

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = cost_y
    ocp.cost.yref = yref
    ocp.cost.W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = cost_y_e
    ocp.cost.W_e = W_state * float(terminal_scale)

    # --- 控制/状态盒约束 ---（mn, mx 已用于 u_ref）
    ocp.constraints.lbu = np.array([mn] * 4 + [-2.0] * 2)
    ocp.constraints.ubu = np.array([mx] * 4 + [2.0] * 2)
    ocp.constraints.idxbu = np.arange(nu)

    lm = LIMITS
    ocp.constraints.idxbx = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    ocp.constraints.lbx = np.array(
        [-lm["j_angle_max"]] * 2 + [-lm["v_max"]] * 3 + [-lm["omega_max"]] * 3 + [-lm["j_vel_max"]] * 2
    )
    ocp.constraints.ubx = np.array(
        [lm["j_angle_max"]] * 2 + [lm["v_max"]] * 3 + [lm["omega_max"]] * 3 + [lm["j_vel_max"]] * 2
    )

    ocp.constraints.x0 = np.asarray(waypoints[0], dtype=float).reshape(17)

    huge = 1e6
    nh = 29 if enable_ctrl_error else 26
    ocp.constraints.lh = -huge * np.ones(nh)
    ocp.constraints.uh = huge * np.ones(nh)
    ocp.constraints.lh_e = -huge * np.ones(nh)
    ocp.constraints.uh_e = huge * np.ones(nh)
    ocp.parameter_values = np.zeros(27)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0

    script_dir = Path(__file__).resolve().parent
    if enable_ctrl_error:
        code_dir = script_dir.parent / "c_generated_code" / "s500_uam_wp3_minimal_ctrlerr"
        json_path = code_dir / "s500_uam_wp3min_ctrlerr_ocp.json"
    else:
        code_dir = script_dir.parent / "c_generated_code" / "s500_uam_wp3_minimal_baseline"
        json_path = code_dir / "s500_uam_wp3min_baseline_ocp.json"
    ocp.code_gen_opts.code_export_directory = str(code_dir)
    ocp.code_gen_opts.json_file = str(json_path)

    gen, bld = _codegen_flags(code_dir, acados_model.name)
    solver = AcadosOcpSolver(
        ocp, json_file=str(json_path), generate=gen, build=bld, verbose=False, check_reuse_possible=True
    )
    return solver, N, pin_model, k, enable_ctrl_error


def apply_hard_waypoints(
    solver,
    pin_model,
    waypoints: list[np.ndarray],
    durations: list[float],
    dt: float,
    grasp_ee_pos: np.ndarray,
    grasp_ee_quat: np.ndarray,
    grasp_ee_vel: np.ndarray | None = None,
    grasp_pos_err_max: np.ndarray | list[float] | float = (0.08, 0.08, 0.08),
    enable_ctrl_error: bool = True,
    loose: float = 1e6,
):
    """约束策略：起点全状态(x0)，中间点约束 EE [pos,SO3,v]，终点约束全状态。"""
    _, N, nodes = shooting_nodes(durations, dt)
    frame_id = pin_model.getFrameId("gripper_link")
    nh = 29 if enable_ctrl_error else 26
    L = -loose * np.ones(nh)
    U = loose * np.ones(nh)
    z27 = np.zeros(27)

    def cset(stage: int, lh: np.ndarray, uh: np.ndarray):
        try:
            solver.constraints_set(stage, "lh", lh, api="new")
            solver.constraints_set(stage, "uh", uh, api="new")
        except TypeError:
            solver.constraints_set(stage, "lh", lh)
            solver.constraints_set(stage, "uh", uh)

    for i in range(1, N):
        cset(i, L, U)
        solver.set(i, "p", z27)

    # 中间航点: EE [pos,SO3,v] 等式约束 (+ 可选位置控制误差界约束)
    k_mid = int(nodes[1])
    if 1 <= k_mid < N:
        p_mid = np.zeros(27)
        gpos = np.asarray(grasp_ee_pos, dtype=float).reshape(3)
        gquat = _norm_q(np.asarray(grasp_ee_quat, dtype=float).reshape(4))
        if grasp_ee_vel is None:
            gvel = np.zeros(3)
        else:
            gvel = np.asarray(grasp_ee_vel, dtype=float).reshape(3)
        p_mid[17:20] = gpos
        p_mid[20:24] = gquat
        p_mid[24:27] = gvel
        lh_mid, uh_mid = L.copy(), U.copy()
        lh_mid[17:], uh_mid[17:] = 0.0, 0.0
        if enable_ctrl_error:
            emax = np.asarray(grasp_pos_err_max, dtype=float).reshape(-1)
            if emax.size == 1:
                emax = np.full(3, float(emax[0]))
            if emax.size != 3:
                raise ValueError("grasp_pos_err_max 必须是标量或3维向量")
            lh_mid[26:29], uh_mid[26:29] = -emax, emax
        solver.set(k_mid, "p", p_mid)
        cset(k_mid, lh_mid, uh_mid)

    # 终点: 全状态等式约束
    p_end = np.zeros(27)
    p_end[:17] = np.asarray(waypoints[-1], dtype=float).reshape(17)
    lh_end, uh_end = L.copy(), U.copy()
    lh_end[:17], uh_end[:17] = 0.0, 0.0
    solver.set(N, "p", p_end)
    cset(N, lh_end, uh_end)


def warm_start(solver, wps: list[np.ndarray], durs: list[float], dt: float, pin_model):
    _, N, _ = shooting_nodes(durs, dt)
    cfg = load_s500_config()
    plat = cfg["platform"]
    u0 = hover_direct_control_ref(pin_model, float(plat["min_thrust"]), float(plat["max_thrust"]))
    tf = float(sum(durs))
    for i in range(N):
        tk = min(i * float(dt), tf - 1e-9)
        solver.set(i, "x", state17_at_time(tk, wps, durs))
        solver.set(i, "u", u0.copy())
    solver.set(N, "x", np.asarray(wps[-1], dtype=float).reshape(17))


def extract_open_loop(solver, N: int, nu: int) -> tuple[np.ndarray, np.ndarray]:
    nx = int(solver.get(0, "x").shape[0])
    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i] = solver.get(i, "x")
        simU[i] = solver.get(i, "u")
    simX[N] = solver.get(N, "x")
    return simX, simU


def ee_linear_velocity_world_series(simX: np.ndarray, pin_model, frame_id: int) -> np.ndarray:
    out = np.zeros((simX.shape[0], 3), dtype=float)
    for i, x in enumerate(simX):
        q = np.asarray(x[:9], dtype=float).copy()
        q[3:7] = _norm_q(q[3:7])
        v = np.asarray(x[9:17], dtype=float).copy()
        data = pin_model.createData()
        pin.forwardKinematics(pin_model, data, q, v)
        pin.updateFramePlacements(pin_model, data)
        vel = pin.getFrameVelocity(pin_model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        out[i, :] = np.asarray(vel.linear).flatten()
    return out


def base_linear_velocity_world_series(simX: np.ndarray) -> np.ndarray:
    """由状态中的 baselink 体坐标线速度 x[9:12] 转到世界系。"""
    out = np.zeros((simX.shape[0], 3), dtype=float)
    for i, x in enumerate(simX):
        R = quat_to_R_np(np.asarray(x[3:7], dtype=float))
        v_body = np.asarray(x[9:12], dtype=float).reshape(3)
        out[i, :] = R @ v_body
    return out


def main():
    # 统一配置区：抓取点（EE）位置/姿态/时刻 + 起终状态
    cfg = {
        # OCP 模式: "baseline" 保留原方法; "ctrl_error" 启用线性控制误差建模与抓取误差约束
        "ocp_mode": "baseline",
        "dt": 0.1,
        "total_time": 3.0,
        "grasp_time": 1.5,  # 从 t=0 开始计时，位于中间航点节点
        "grasp_ee_pos": np.array([0.0, 0.0, 1.0], dtype=float),
        # 配置时使用欧拉角（度）：[roll, pitch, yaw]，内部自动转四元数用于 SO(3) 误差计算
        "grasp_ee_euler_deg": np.array([0.0, 0.0, 0.0], dtype=float),
        "grasp_ee_vel": np.array([0.0, 0.0, 0.0], dtype=float),
        # 线性位置控制误差模型: e_p = K * v_base_world (逐轴)
        "pos_err_gain": np.array([0.1, 0.1, 0.1], dtype=float),
        # 抓取点处允许的位置控制误差上界 |e_p| <= emax
        "grasp_pos_err_max": np.array([0.02, 0.02, 0.02], dtype=float),
    }
    run_wp3_joint_opt(cfg, show_plots=True)


def run_wp3_joint_opt(cfg: dict | None = None, *, show_plots: bool = True) -> dict:
    """Run wp3 joint optimization and optionally render figures. Returns planning bundle dict."""
    if cfg is None:
        cfg = {}
    cfg = {
        "ocp_mode": "ctrl_error",
        "dt": 0.1,
        "total_time": 3.0,
        "grasp_time": 1.5,
        "grasp_ee_pos": np.array([0.0, 0.0, 1.0], dtype=float),
        "grasp_ee_euler_deg": np.array([0.0, 0.0, 0.0], dtype=float),
        "grasp_ee_vel": np.array([0.0, 0.0, 0.0], dtype=float),
        "pos_err_gain": np.array([0.08, 0.08, 0.08], dtype=float),
        "grasp_pos_err_max": np.array([0.06, 0.06, 0.06], dtype=float),
        "state_weight": 1.0,
        "control_weight": 1.0,
        "terminal_scale": 1.0,
        "max_iter": 10,
        "wp0": np.array([-1.5, 0, 1.5, 0.0, 0.0, 0.0], dtype=float),  # x,y,z,j1_deg,j2_deg,yaw_deg
        "wp2": np.array([1.5, 0, 1.5, 0.0, 0.0, 0.0], dtype=float),
        **cfg,
    }
    dt = float(cfg["dt"])
    tg = float(cfg["grasp_time"])
    tf = float(cfg["total_time"])
    if not (0.0 < tg < tf):
        raise ValueError("要求 0 < grasp_time < total_time")
    ocp_mode = str(cfg.get("ocp_mode", "ctrl_error")).strip().lower()
    if ocp_mode not in ("baseline", "ctrl_error"):
        raise ValueError("ocp_mode 必须是 'baseline' 或 'ctrl_error'")
    enable_ctrl_error = (ocp_mode == "ctrl_error")
    durs = [tg, tf - tg]
    grasp_euler_rad = np.radians(np.asarray(cfg["grasp_ee_euler_deg"], dtype=float).reshape(3))
    grasp_ee_quat = euler_zyx_to_quat_np(grasp_euler_rad)

    # 三个 17D 航点：[x,y,z,qx,qy,qz,qw,j1,j2,v,ω,jdot]
    w0 = np.asarray(cfg["wp0"], dtype=float).reshape(6)
    w2 = np.asarray(cfg["wp2"], dtype=float).reshape(6)
    wp0 = np.array(
        [w0[0], w0[1], w0[2], 0, 0, 0, 1, np.deg2rad(w0[3]), np.deg2rad(w0[4])] + [0] * 8, dtype=float
    )
    wp2 = np.array(
        [w2[0], w2[1], w2[2], 0, 0, 0, 1, np.deg2rad(w2[3]), np.deg2rad(w2[4])] + [0] * 8, dtype=float
    )
    # yaw from cfg wp yaw_deg
    qz0 = np.sin(np.deg2rad(w0[5]) * 0.5)
    qw0 = np.cos(np.deg2rad(w0[5]) * 0.5)
    qz2 = np.sin(np.deg2rad(w2[5]) * 0.5)
    qw2 = np.cos(np.deg2rad(w2[5]) * 0.5)
    wp0[5], wp0[6] = qz0, qw0
    wp2[5], wp2[6] = qz2, qw2
    # 中间状态仅用于 warm-start 插值；真正中间约束使用 grasp_ee_*（EE 位置/姿态/速度）
    wp1 = 0.5 * (wp0 + wp2)
    wp1[3:7] = _norm_q(wp1[3:7])
    wps = [wp0, wp1, wp2]

    print("Building OCP (first run compiles acados code)...")
    t0 = time.perf_counter()
    solver, N, pin_model, k_err, _use_err = build_ocp_with_ctrl_error(
        wps,
        durs,
        dt,
        pos_err_gain=cfg["pos_err_gain"],
        enable_ctrl_error=enable_ctrl_error,
        state_weight=float(cfg["state_weight"]),
        control_weight=float(cfg["control_weight"]),
        terminal_scale=float(cfg["terminal_scale"]),
        max_iter=int(cfg["max_iter"]),
    )
    apply_hard_waypoints(
        solver,
        pin_model,
        wps,
        durs,
        dt,
        grasp_ee_pos=cfg["grasp_ee_pos"],
        grasp_ee_quat=grasp_ee_quat,
        grasp_ee_vel=cfg["grasp_ee_vel"],
        grasp_pos_err_max=cfg["grasp_pos_err_max"],
        enable_ctrl_error=enable_ctrl_error,
    )
    warm_start(solver, wps, durs, dt, pin_model)

    st = solver.solve()
    print(f"solve status={st}  wall={time.perf_counter() - t0:.3f}s  cost={solver.get_cost():.6e}")

    nu = 6
    simX, simU = extract_open_loop(solver, N, nu)
    t_arr = np.linspace(0, sum(durs), N + 1)

    # 与 s500_uam_trajectory_gui 主画布相同：4×4 Base / EE / Arm / Control
    wp_times = np.cumsum([0.0] + [float(d) for d in durs])
    wp_pos_base = np.stack([np.asarray(w[:3], dtype=float) for w in wps], axis=0)
    # 仅中间点有效：标注 EE 抓取约束位置（位置标记）
    wp_pos_ee = np.array(
        [
            [np.nan, np.nan, np.nan],
            np.asarray(cfg["grasp_ee_pos"], dtype=float).reshape(3),
            [np.nan, np.nan, np.nan],
        ],
        dtype=float,
    )
    fig = None
    if show_plots:
        fig = plt.figure(figsize=(14, 10))
        plot_acados_into_figure(
            simX,
            simU,
            t_arr,
            fig,
            title="S500 UAM — 3-waypoint joint opt (minimal)",
            waypoint_times=wp_times,
            waypoint_positions_base=wp_pos_base,
            waypoint_positions_ee=wp_pos_ee,
            timing_info=None,
            control_layout=CONTROL_INPUT_DIRECT,
        )
    # 姿态文本标注：用设置时的欧拉角显示，误差计算内部使用四元数/SO(3)
    r_deg, p_deg, y_deg = np.asarray(cfg["grasp_ee_euler_deg"], dtype=float).reshape(3)
    if show_plots and fig is not None:
        fig.text(
            0.02,
            0.965,
            (
                f"Grasp EE constraint @ t={tg:.2f}s: pos={np.asarray(cfg['grasp_ee_pos']).round(3).tolist()} m, "
                f"rpy=[{r_deg:.1f}, {p_deg:.1f}, {y_deg:.1f}] deg"
            ),
            fontsize=9,
            color="darkorange",
        )

    if enable_ctrl_error and show_plots:
        # 位置控制误差显示：e_p = K * v_base_world（线性模型）
        v_base_w = base_linear_velocity_world_series(simX)
        e_p = v_base_w * np.asarray(k_err, dtype=float).reshape(1, 3)
        fig_err, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(t_arr, e_p[:, 0], "r-", label="e_px")
        ax.plot(t_arr, e_p[:, 1], "g-", label="e_py")
        ax.plot(t_arr, e_p[:, 2], "b-", label="e_pz")
        emax = np.asarray(cfg["grasp_pos_err_max"], dtype=float).reshape(3)
        for i, c in enumerate(("r", "g", "b")):
            ax.axhline(+emax[i], color=c, linestyle="--", alpha=0.4)
            ax.axhline(-emax[i], color=c, linestyle="--", alpha=0.4)
        ax.axvline(tg, color="darkorange", linestyle="--", alpha=0.7, label="grasp time")
        ax.set_title("Linear position-control error (e_p = K * v_base_world)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position-control error (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    if show_plots:
        plt.show()
    return {
        "method": "acados_wp3_joint_opt",
        "simX": simX,
        "simU": simU,
        "time_arr": t_arr,
        "waypoint_positions": wp_pos_base.tolist(),
        "waypoint_times": wp_times.tolist(),
        "timing": {"total_s": float(time.perf_counter() - t0), "n_iter": 0, "avg_ms_per_iter": 0.0},
        "control_layout": "direct",
    }


if __name__ == "__main__":
    main()
