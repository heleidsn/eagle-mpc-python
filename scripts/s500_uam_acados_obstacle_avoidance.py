#!/usr/bin/env python3
"""
S500 UAM 带避障的轨迹规划 (acados, 全身碰撞球)

任务场景:
  base_link 从 (-2, 0, 1.5) 出发  ->  末端执行器 (gripper) 抵达 (0, 0, 1) 抓取(EE 速度=0)
  ->  base_link 到 (2, 0, 1.5) 悬停。
  空间中布置 6 个障碍物(球 / 长方体 / 圆柱),分布在原始直线路径上,
  规划器必须绕开它们。

避障思路 (全身近似):
  - 机器人整机近似为 N 个 collision sphere(机体 1、四旋翼 4、机械臂连杆 2、末端 1)。
    每个时间点用 Pinocchio(CasADi)前向运动学得到各球心 p_i(q)。
  - 障碍物用其"到表面的解析距离"表达(球/长方体/圆柱各一种),
    因此每个 (机器人球 i, 障碍 j) 只需 1 条平滑不等式:
        dist_surface(p_i(q), obstacle_j)^2 >= (r_i + margin)^2          (球面)
        ||max(|p-c|-half, 0)||^2          >= (r_i + margin)^2          (长方体)
        d_radial^2 + d_axial^2            >= (r_i + margin)^2          (圆柱)
    这是工程上最稳、约束数最少的形式(N_sphere * N_obs 条/步)。
  - 碰撞约束作为软约束(slack 大惩罚),保证求解器即使初值穿模也能收敛并被推出障碍;
    抓取点的 EE 位置/速度作为该节点的硬等式约束。

用法:
    python s500_uam_acados_obstacle_avoidance.py                          # acados (默认)
    python s500_uam_acados_obstacle_avoidance.py --mode minsnap         # minimum snap
    python s500_uam_acados_obstacle_avoidance.py --mode compare         # 两者对比
    python s500_uam_acados_obstacle_avoidance.py --mode compare --reuse-build --no-show

依赖: acados_template, pinocchio(含 casadi 绑定), casadi, numpy, matplotlib
后续可在 GUI 中加按钮调用 plan_obstacle_avoidance_trajectory(...)。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from acados_runtime import preload_acados_shared_libs

    preload_acados_shared_libs()
    from acados_template import AcadosOcp, AcadosOcpSolver

    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin

from s500_uam_acados_model import build_acados_model, load_s500_config


# ----------------------------------------------------------------------------
# 状态/参考工具 (与 s500_uam_acados_trajectory.py 保持一致)
# ----------------------------------------------------------------------------
def make_uam_state(x, y, z, j1=0.0, j2=0.0, yaw=0.0) -> np.ndarray:
    """17 维状态 [x,y,z, qx,qy,qz,qw, j1,j2, vx,vy,vz, wx,wy,wz, j1dot,j2dot]。"""
    s = np.zeros(17)
    s[0], s[1], s[2] = x, y, z
    half = yaw / 2.0
    s[5] = np.sin(half)
    s[6] = np.cos(half)
    s[7], s[8] = j1, j2
    return s


def _state_to_cost_ref(state: np.ndarray) -> np.ndarray:
    """17 维状态 -> [pos(3), yaw, roll, pitch, jq(2), v(8)] 共 16 维代价参考。"""
    qx, qy, qz, qw = state[3], state[4], state[5], state[6]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return np.concatenate([state[0:3], [yaw], [roll], [pitch], state[7:9], state[9:17]])


def _quat_to_euler_zyx_ca(quat):
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    roll = ca.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = ca.fmin(1, ca.fmax(-1, 2 * (qw * qy - qz * qx)))
    pitch = ca.asin(sinp)
    yaw = ca.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float).reshape(4)
    n = float(np.linalg.norm(q))
    return np.array([0, 0, 0, 1.0]) if n < 1e-12 else q / n


def _interp_state17(alpha: float, xs: np.ndarray, xe: np.ndarray) -> np.ndarray:
    a = float(np.clip(alpha, 0, 1))
    out = (1 - a) * np.asarray(xs, float) + a * np.asarray(xe, float)
    out[3:7] = _normalize_quat(out[3:7])
    return out


# ----------------------------------------------------------------------------
# 机器人碰撞球 与 障碍物定义
# ----------------------------------------------------------------------------
# (frame_name, 该 frame 坐标系下的局部偏移, 半径)
ROBOT_SPHERES = [
    ("base_link", (0.0, 0.0, 0.02), 0.20),   # 机体中心
    ("rotor_1", (0.0, 0.0, 0.0), 0.13),      # 四旋翼
    ("rotor_2", (0.0, 0.0, 0.0), 0.13),
    ("rotor_3", (0.0, 0.0, 0.0), 0.13),
    ("rotor_4", (0.0, 0.0, 0.0), 0.13),
    ("link_1", (0.0, 0.0, -0.05), 0.07),     # 机械臂连杆 1
    ("link_2", (0.0, 0.0, -0.06), 0.07),     # 机械臂连杆 2
    ("gripper_link", (0.0, 0.0, 0.0), 0.05),  # 末端执行器
]

EE_FRAME = "gripper_link"


@dataclass
class Obstacle:
    kind: str                      # "sphere" | "box" | "cylinder"
    center: tuple                  # (x, y, z)
    radius: float = 0.0            # sphere/cylinder 半径
    size: tuple = (0.0, 0.0, 0.0)  # box 的全尺寸 (sx, sy, sz)
    height: float = 0.0            # cylinder 沿 z 的全高
    color: str = "tab:red"
    label: str = ""


def default_obstacles() -> list[Obstacle]:
    """6 个障碍,布置在 (-2,0,1.5)->(0,0,1 EE)->(2,0,1.5) 的原始走廊上。"""
    return [
        Obstacle("sphere", (-1.15, 0.05, 1.45), radius=0.28, color="#e74c3c", label="sphere-A"),
        Obstacle("cylinder", (-0.55, 0.0, 1.20), radius=0.16, height=1.0, color="#3498db", label="cyl-B"),
        Obstacle("box", (0.0, 0.0, 1.82), size=(0.9, 0.9, 0.28), color="#9b59b6", label="box-C(ceil)"),
        Obstacle("sphere", (0.65, -0.05, 1.50), radius=0.26, color="#e67e22", label="sphere-D"),
        Obstacle("cylinder", (1.25, 0.0, 1.20), radius=0.16, height=1.0, color="#1abc9c", label="cyl-E"),
        Obstacle("box", (0.45, 0.0, 1.02), size=(0.5, 0.5, 0.4), color="#34495e", label="box-F(low)"),
    ]


# ----------------------------------------------------------------------------
# 碰撞残差 (g >= 0 表示无碰撞), 适用于 CasADi(符号) 与 numpy(数值)
# ----------------------------------------------------------------------------
def _collision_residual(p, obs: Obstacle, ri: float, margin: float, backend):
    """返回 g = (到障碍表面的距离)^2 - (ri + margin)^2;  g >= 0 即安全。
    backend: ca 或 np,使其同时支持符号与数值计算。
    """
    cx, cy, cz = obs.center
    safe = (ri + margin)

    if obs.kind == "sphere":
        d2 = (p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2
        thr = (obs.radius + safe) ** 2
        return d2 - thr

    if obs.kind == "box":
        hx, hy, hz = obs.size[0] / 2.0, obs.size[1] / 2.0, obs.size[2] / 2.0
        dx = backend.fmax(backend.fabs(p[0] - cx) - hx, 0.0)
        dy = backend.fmax(backend.fabs(p[1] - cy) - hy, 0.0)
        dz = backend.fmax(backend.fabs(p[2] - cz) - hz, 0.0)
        d2 = dx * dx + dy * dy + dz * dz
        return d2 - safe ** 2

    if obs.kind == "cylinder":  # 轴沿 z 的有限圆柱
        rxy = backend.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + 1e-9)
        d_r = backend.fmax(rxy - obs.radius, 0.0)
        d_z = backend.fmax(backend.fabs(p[2] - cz) - obs.height / 2.0, 0.0)
        d2 = d_r * d_r + d_z * d_z
        return d2 - safe ** 2

    raise ValueError(f"未知障碍类型: {obs.kind}")


class _NpBackend:
    fmax = staticmethod(np.maximum)
    fabs = staticmethod(np.abs)
    sqrt = staticmethod(np.sqrt)


# ----------------------------------------------------------------------------
# OCP 构建
# ----------------------------------------------------------------------------
BIG = 1e6


def _phase_nodes(d1: float, d2: float, dt: float):
    N1 = max(1, int(round(d1 / dt)))
    N2 = max(1, int(round(d2 / dt)))
    return N1, N2, N1 + N2, float(d1 + d2)


def build_obstacle_ocp(
    start_state: np.ndarray,
    ee_grasp_pos: np.ndarray,
    end_state: np.ndarray,
    obstacles: list[Obstacle],
    d1: float,
    d2: float,
    dt: float,
    margin: float = 0.05,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 200.0,
    coll_slack_lin: float = 1e3,
    coll_slack_quad: float = 1e4,
    max_iter: int = 150,
    reuse_build: bool = False,
):
    if not ACADOS_AVAILABLE:
        raise ImportError("acados_template 不可用,请检查 acados 安装。")

    N1, N2, N, tf = _phase_nodes(d1, d2, dt)
    k_grasp = N1  # 抓取节点(相位 1 末端,内部节点)

    acados_model, pin_model, nq, nv, nu = build_acados_model()
    acados_model.name = "s500_uam_obstacle"

    x = acados_model.x
    q = x[:nq]
    v = x[nq:nq + nv]

    # ---- CasADi 前向运动学 (归一化四元数) ----
    quat = q[3:7]
    quat_u = quat / ca.fmax(ca.norm_2(quat), 1e-9)
    q_fk = ca.vertcat(q[:3], quat_u, q[7:nq])
    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()
    cpin.forwardKinematics(cmodel, cdata, q_fk, v)
    cpin.updateFramePlacements(cmodel, cdata)

    # 机器人碰撞球的世界球心
    sphere_centers = []
    sphere_radii = []
    for fname, off, rr in ROBOT_SPHERES:
        fid = int(pin_model.getFrameId(fname))
        oMf = cdata.oMf[fid]
        c = oMf.translation + oMf.rotation @ ca.DM(list(off))
        sphere_centers.append(c)
        sphere_radii.append(rr)

    # 碰撞约束: 每个(机器人球, 障碍)一条
    coll_g = []
    for c, ri in zip(sphere_centers, sphere_radii):
        for obs in obstacles:
            coll_g.append(_collision_residual(c, obs, ri, margin, ca))
    coll_expr = ca.vertcat(*coll_g)
    n_coll = coll_expr.size1()

    # 末端位置/速度 (抓取约束)
    ee_fid = int(pin_model.getFrameId(EE_FRAME))
    ee_pos = cdata.oMf[ee_fid].translation
    ee_vel = cpin.getFrameVelocity(
        cmodel, cdata, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    ).linear

    # 参数: 抓取点目标位置
    p_sym = ca.SX.sym("p_ee_target", 3)
    acados_model.p = p_sym
    ee_pos_err = ee_pos - p_sym

    # 路径约束: [碰撞(n_coll); EE 位置误差(3); EE 速度(3)]
    acados_model.con_h_expr = ca.vertcat(coll_expr, ee_pos_err, ee_vel)
    acados_model.con_h_expr_e = coll_expr  # 末端只保留碰撞

    # ---- OCP ----
    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.dims.N = N
    ocp.solver_options.tf = tf
    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = N
    ocp.solver_options.nlp_solver_max_iter = max_iter
    ocp.parameter_values = np.zeros(3)

    # ---- 代价: NONLINEAR_LS ----
    roll, pitch, yaw = _quat_to_euler_zyx_ca(quat)
    cost_y = ca.vertcat(x[0:3], yaw, roll, pitch, x[7:9], x[9:17], acados_model.u)
    cost_y_e = ca.vertcat(x[0:3], yaw, roll, pitch, x[7:9], x[9:17])
    ocp.model.cost_y_expr = cost_y
    ocp.model.cost_y_expr_e = cost_y_e

    w_pos, w_yaw, w_rp = 50.0 * state_weight, 20.0 * state_weight, 10.0 * state_weight
    w_jq, w_v, w_om, w_jd = 20.0 * state_weight, 1.0 * state_weight, 1.0 * state_weight, 5.0 * state_weight
    W_state = np.diag([w_pos, w_pos, w_pos, w_yaw, w_rp, w_rp, w_jq, w_jq,
                       w_v, w_v, w_v, w_om, w_om, w_om, w_jd, w_jd])
    R = np.diag([control_weight] * 4 + [control_weight * 1e4] * 2)
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))
    ocp.cost.W_e = W_state * float(waypoint_multiplier)
    # 初始 yref(每个 stage 之后会用 cost_set 覆盖为插值参考)
    ocp.cost.yref = np.concatenate([_state_to_cost_ref(end_state), np.zeros(nu)])
    ocp.cost.yref_e = _state_to_cost_ref(end_state)

    # ---- 控制 / 状态 box 约束 ----
    cfg = load_s500_config()
    platform = cfg["platform"]
    min_thrust, max_thrust = platform["min_thrust"], platform["max_thrust"]
    ocp.constraints.lbu = np.array([min_thrust] * 4 + [-2.0] * 2)
    ocp.constraints.ubu = np.array([max_thrust] * 4 + [2.0] * 2)
    ocp.constraints.idxbu = np.arange(nu)

    v_max, om_max, j_max, jv_max = 1.5, 2.5, 1.5, 3.0
    ocp.constraints.idxbx = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    ocp.constraints.lbx = np.array([-j_max, -j_max, -v_max, -v_max, -v_max,
                                    -om_max, -om_max, -om_max, -jv_max, -jv_max])
    ocp.constraints.ubx = np.array([j_max, j_max, v_max, v_max, v_max,
                                    om_max, om_max, om_max, jv_max, jv_max])
    ocp.constraints.x0 = np.asarray(start_state, float).flatten()

    # ---- 非线性路径约束边界 ----
    # 碰撞: g >= 0 (软); EE: 默认松弛(仅抓取节点收紧)
    lh = np.concatenate([np.zeros(n_coll), -BIG * np.ones(3), -BIG * np.ones(3)])
    uh = np.concatenate([BIG * np.ones(n_coll), BIG * np.ones(3), BIG * np.ones(3)])
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.lh_e = np.zeros(n_coll)
    ocp.constraints.uh_e = BIG * np.ones(n_coll)

    # 碰撞约束设为软约束(slack), EE 约束保持硬约束
    ocp.constraints.idxsh = np.arange(n_coll)
    ocp.constraints.idxsh_e = np.arange(n_coll)
    ocp.cost.zl = coll_slack_lin * np.ones(n_coll)
    ocp.cost.zu = coll_slack_lin * np.ones(n_coll)
    ocp.cost.Zl = coll_slack_quad * np.ones(n_coll)
    ocp.cost.Zu = coll_slack_quad * np.ones(n_coll)
    ocp.cost.zl_e = coll_slack_lin * np.ones(n_coll)
    ocp.cost.zu_e = coll_slack_lin * np.ones(n_coll)
    ocp.cost.Zl_e = coll_slack_quad * np.ones(n_coll)
    ocp.cost.Zu_e = coll_slack_quad * np.ones(n_coll)

    # ---- 求解器选项 ----
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0
    if hasattr(ocp.solver_options, "levenberg_marquardt"):
        ocp.solver_options.levenberg_marquardt = 1e-3
    if hasattr(ocp.solver_options, "qp_solver_iter_max"):
        ocp.solver_options.qp_solver_iter_max = 200

    script_dir = Path(__file__).parent
    code_export_dir = script_dir.parent / "c_generated_code" / "s500_uam_obstacle"
    json_path = code_export_dir / "s500_uam_obstacle_ocp.json"
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(json_path)

    # 代码生成 + 编译用时 (onboard 评估: 这部分是离线一次性开销)。
    # reuse_build=True 时若已存在编译好的 .so 则跳过 codegen/compile, 只加载。
    gen, bld = _acados_generate_build_flags(code_export_dir, acados_model.name, reuse_build)
    t_codegen0 = time.perf_counter()
    solver = AcadosOcpSolver(
        ocp, json_file=str(json_path), verbose=False, generate=gen, build=bld,
    )
    t_build = time.perf_counter() - t_codegen0

    meta = {
        "N": N, "N1": N1, "N2": N2, "tf": tf, "dt": dt,
        "k_grasp": k_grasp, "n_coll": n_coll,
        "n_h": int(acados_model.con_h_expr.size1()),
        "nx": nq + nv, "nu": nu,
        "min_thrust": min_thrust, "max_thrust": max_thrust,
        "pin_model": pin_model, "ee_fid": ee_fid,
        "t_build": t_build, "did_codegen": bool(gen), "did_compile": bool(bld),
        "code_export_dir": str(code_export_dir),
    }
    return solver, meta


def _acados_generate_build_flags(code_export_dir, model_name: str, reuse_build: bool):
    """返回 (generate, build)。reuse_build 且已编译出 .so 时跳过 codegen/compile。"""
    if not reuse_build:
        return True, True
    try:
        from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix
        lib = f"{get_shared_lib_prefix()}acados_ocp_solver_{model_name}{get_shared_lib_ext()}"
        so_path = Path(code_export_dir) / lib
        if so_path.is_file():
            return False, False
        if (Path(code_export_dir) / "Makefile").is_file():
            return False, True
    except Exception:
        pass
    return True, True


def _set_warmstart_and_grasp(solver, meta, start_state, ee_grasp_pos, end_state,
                             grasp_base_state):
    """初值: start -> grasp_base -> end 分段插值; 抓取节点收紧 EE 约束;
    每个 stage 设置代价参考(随时间从 start 走到 end)。"""
    N, N1, N2, k_grasp = meta["N"], meta["N1"], meta["N2"], meta["k_grasp"]
    min_thrust, max_thrust = meta["min_thrust"], meta["max_thrust"]
    nu = 6
    m_th = float(np.clip(0.5 * (min_thrust + max_thrust), min_thrust, max_thrust))
    u_hover = np.array([m_th] * 4 + [0.0, 0.0])

    n_coll = meta["n_coll"]
    tight = np.zeros(6)  # EE 位置误差(3) + EE 速度(3) = 0

    # 状态/控制初值 + 每步代价参考
    for k in range(N + 1):
        if k <= N1:
            a = k / max(N1, 1)
            xk = _interp_state17(a, start_state, grasp_base_state)
        else:
            a = (k - N1) / max(N2, 1)
            xk = _interp_state17(a, grasp_base_state, end_state)
        solver.set(k, "x", xk)
        # 代价参考(状态部分)
        yref_state = _state_to_cost_ref(xk if k != N else end_state)
        if k < N:
            solver.cost_set(k, "yref", np.concatenate([yref_state, np.zeros(nu)]))
        else:
            solver.cost_set(k, "yref", _state_to_cost_ref(end_state))

    for k in range(N):
        solver.set(k, "u", u_hover)

    # 默认所有内部节点 p=0(EE 行松弛); 抓取节点 p=目标 且收紧 EE 行
    for k in range(1, N):
        solver.set(k, "p", np.zeros(3))

    solver.set(k_grasp, "p", np.asarray(ee_grasp_pos, float).flatten())
    lh = np.concatenate([np.zeros(n_coll), tight])
    uh = np.concatenate([BIG * np.ones(n_coll), tight])
    try:
        solver.constraints_set(k_grasp, "lh", lh, api="new")
        solver.constraints_set(k_grasp, "uh", uh, api="new")
    except TypeError:
        solver.constraints_set(k_grasp, "lh", lh)
        solver.constraints_set(k_grasp, "uh", uh)


# ----------------------------------------------------------------------------
# 数值侧: FK 球心、间隙检查
# ----------------------------------------------------------------------------
def robot_sphere_centers_np(pin_model, data, x17: np.ndarray):
    q = np.asarray(x17[:9], float).copy()
    q[3:7] = _normalize_quat(q[3:7])
    pin.framesForwardKinematics(pin_model, data, q)
    centers, radii = [], []
    for fname, off, rr in ROBOT_SPHERES:
        fid = int(pin_model.getFrameId(fname))
        oMf = data.oMf[fid]
        c = np.asarray(oMf.translation, float) + np.asarray(oMf.rotation, float) @ np.array(off)
        centers.append(c)
        radii.append(rr)
    return np.array(centers), np.array(radii)


def min_clearance(pin_model, data, simX, obstacles, margin):
    """返回整条轨迹的最小间隙(>=0 安全) 以及发生位置信息。"""
    worst = np.inf
    worst_info = None
    for i in range(simX.shape[0]):
        centers, radii = robot_sphere_centers_np(pin_model, data, simX[i])
        for (c, ri) in zip(centers, radii):
            for obs in obstacles:
                g = _collision_residual(c, obs, ri, margin, _NpBackend)
                # g = dist_surface^2 - (ri+margin)^2  => 实际间隙(到表面 - 安全裕量)
                dist_surf = np.sqrt(max(g + (ri + margin) ** 2, 0.0))
                clearance = dist_surf - (ri + margin)
                if clearance < worst:
                    worst = clearance
                    worst_info = (i, obs.label)
    return worst, worst_info


# ----------------------------------------------------------------------------
# 求解器计时统计
# ----------------------------------------------------------------------------
def _collect_solve_stats(solver) -> dict:
    """收集 acados 求解器的迭代次数与各阶段耗时 (单位: s)。"""
    out = {}

    def _get(key):
        try:
            return solver.get_stats(key)
        except Exception:
            return None

    def _scalar(v):
        if v is None:
            return None
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        return float(arr.flatten()[0]) if arr.size == 1 else float(np.nansum(arr))

    sqp = _get("sqp_iter")
    if sqp is None:
        sqp = _get("nlp_iter")
    out["sqp_iter"] = int(_scalar(sqp)) if sqp is not None else -1
    qp = _get("qp_iter")
    out["qp_iter"] = int(_scalar(qp)) if qp is not None else -1
    for k in ("time_tot", "time_lin", "time_sim", "time_qp", "time_reg", "time_glob"):
        val = _scalar(_get(k))
        if val is not None:
            out[k] = val
    return out


# ----------------------------------------------------------------------------
# 顶层规划函数 (供 GUI 调用)
# ----------------------------------------------------------------------------
def plan_obstacle_avoidance_trajectory(
    start_xyz=(-2.0, 0.0, 1.5),
    ee_grasp_pos=(0.0, 0.0, 1.0),
    end_xyz=(2.0, 0.0, 1.5),
    obstacles: list[Obstacle] | None = None,
    d1: float = 4.0,
    d2: float = 4.0,
    dt: float = 0.04,
    margin: float = 0.05,
    max_iter: int = 150,
    reuse_build: bool = False,
    hot_resolves: int = 0,
    verbose: bool = True,
):
    """求解带避障的抓取轨迹。返回 (simX, simU, time_arr, info)。"""
    if obstacles is None:
        obstacles = default_obstacles()

    start_state = make_uam_state(*start_xyz, j1=0.0, j2=0.0)
    end_state = make_uam_state(*end_xyz, j1=0.0, j2=0.0)
    # 抓取时基座位置: EE 在 j=0 下比基座低 0.312m
    grasp_base_state = make_uam_state(
        ee_grasp_pos[0], ee_grasp_pos[1], ee_grasp_pos[2] + 0.312, j1=0.0, j2=0.0
    )

    solver, meta = build_obstacle_ocp(
        start_state, np.array(ee_grasp_pos, float), end_state, obstacles,
        d1=d1, d2=d2, dt=dt, margin=margin, max_iter=max_iter, reuse_build=reuse_build,
    )
    _set_warmstart_and_grasp(solver, meta, start_state, ee_grasp_pos, end_state, grasp_base_state)

    t0 = time.perf_counter()
    status = solver.solve()
    t_wall = time.perf_counter() - t0
    solve_stats = _collect_solve_stats(solver)
    n_iter = solve_stats.get("sqp_iter", -1)
    cost = None
    try:
        cost = float(solver.get_cost())
    except Exception:
        pass

    # 热启动重解计时: 模拟 onboard 上从上一帧解附近再优化的速度
    hot = []
    for _ in range(max(0, int(hot_resolves))):
        th0 = time.perf_counter()
        solver.solve()
        hot.append((time.perf_counter() - th0, _collect_solve_stats(solver)))

    N = meta["N"]
    nx = 17
    nu = 6
    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i] = solver.get(i, "x")
        simU[i] = solver.get(i, "u")
    simX[N] = solver.get(N, "x")
    time_arr = np.linspace(0, meta["tf"], N + 1)

    pin_model = meta["pin_model"]
    data = pin_model.createData()
    worst, worst_info = min_clearance(pin_model, data, simX, obstacles, margin)

    # 抓取节点实际 EE 位置 / 速度
    xg = simX[meta["k_grasp"]]
    qg = np.concatenate([xg[:3], _normalize_quat(xg[3:7]), xg[7:9]])
    vg = xg[9:17]
    pin.forwardKinematics(pin_model, data, qg, vg)
    pin.updateFramePlacements(pin_model, data)
    ee_at_grasp = np.asarray(data.oMf[meta["ee_fid"]].translation, float)
    ee_err = float(np.linalg.norm(ee_at_grasp - np.array(ee_grasp_pos)))
    ee_vel_grasp = np.asarray(pin.getFrameVelocity(
        pin_model, data, meta["ee_fid"], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear, float)
    ee_speed = float(np.linalg.norm(ee_vel_grasp))

    t_solve_cpu = solve_stats.get("time_tot", t_wall)
    ms_per_iter = (t_solve_cpu / n_iter * 1e3) if n_iter and n_iter > 0 else 0.0
    hot_times = [h[0] for h in hot]
    hot_cpu = [h[1].get("time_tot", h[0]) for h in hot]

    info = {
        "status": int(status), "n_iter": int(n_iter) if n_iter is not None else -1,
        "cost": cost, "t_wall": t_wall, "tf": meta["tf"], "dt": meta["dt"],
        "k_grasp": meta["k_grasp"], "n_coll": meta["n_coll"],
        "min_clearance": float(worst), "min_clearance_info": worst_info,
        "ee_grasp_err": ee_err, "ee_at_grasp": ee_at_grasp, "ee_speed_grasp": ee_speed,
        "obstacles": obstacles, "margin": margin, "pin_model": pin_model,
        "ee_fid": meta["ee_fid"],
        # ---- 计时统计 (onboard 评估) ----
        "timing": {
            "build_s": meta["t_build"],
            "total_s": meta["t_build"] + t_wall,
            "did_codegen": meta["did_codegen"],
            "did_compile": meta["did_compile"],
            "solve_wall_s": t_wall,
            "solve_cpu_s": t_solve_cpu,
            "ms_per_iter": ms_per_iter,
            "sqp_iter": n_iter,
            "qp_iter": solve_stats.get("qp_iter", -1),
            "time_lin_s": solve_stats.get("time_lin"),
            "time_sim_s": solve_stats.get("time_sim"),
            "time_qp_s": solve_stats.get("time_qp"),
            "time_reg_s": solve_stats.get("time_reg"),
            "hot_resolve_wall_s": hot_times,
            "hot_resolve_cpu_s": hot_cpu,
            "hot_resolve_mean_cpu_s": (float(np.mean(hot_cpu)) if hot_cpu else None),
        },
        "dims": {
            "N": meta["N"], "nx": meta["nx"], "nu": meta["nu"],
            "n_h_per_node": meta["n_h"], "n_coll_per_node": meta["n_coll"],
            "total_path_constraints": meta["N"] * meta["n_h"],
        },
    }
    if verbose:
        _print_report(info, ee_grasp_pos)
    return simX, simU, time_arr, info


def _fmt_ms(x):
    return "n/a" if x is None else f"{x * 1e3:.2f} ms"


def _print_report(info: dict, ee_grasp_pos):
    tm = info["timing"]
    dm = info["dims"]
    print(f"求解状态 status={info['status']}  cost={info['cost']}")
    print(f"  抓取节点 k={info['k_grasp']}  EE 误差={info['ee_grasp_err']*1000:.1f} mm  "
          f"EE 速度={info['ee_speed_grasp']*1000:.1f} mm/s  (目标 {ee_grasp_pos})")
    print(f"  全轨迹最小间隙={info['min_clearance']*1000:.1f} mm  @ step={info['min_clearance_info']}")
    if info["min_clearance"] < -1e-3:
        print("  [警告] 存在穿模(间隙<0),可增大 margin / slack 惩罚或延长时间。")

    print("-" * 64)
    print("[问题规模]")
    print(f"  N(打靶节点)={dm['N']}  nx={dm['nx']}  nu={dm['nu']}")
    print(f"  每节点非线性约束 n_h={dm['n_h_per_node']} (其中碰撞 {dm['n_coll_per_node']} 条 = 8球x6障碍)")
    print(f"  全程碰撞/路径约束总数 ≈ {dm['total_path_constraints']}")
    print("[编译 / 代码生成]  (onboard 离线一次性开销)")
    stage = []
    if tm["did_codegen"]:
        stage.append("codegen")
    if tm["did_compile"]:
        stage.append("compile")
    stage_s = "+".join(stage) if stage else "复用已编译 .so(仅加载)"
    print(f"  build 用时={tm['build_s']:.3f}s  ({stage_s})")
    print("[优化求解]  (onboard 在线开销)")
    print(f"  SQP 迭代次数={tm['sqp_iter']}  QP 迭代(累计)={tm['qp_iter']}")
    print(f"  求解 CPU 用时={tm['solve_cpu_s']*1e3:.1f} ms  (wall={tm['solve_wall_s']*1e3:.1f} ms)")
    print(f"  平均每次 SQP 迭代={tm['ms_per_iter']:.2f} ms")
    print(f"  耗时拆解: 线性化 lin={_fmt_ms(tm['time_lin_s'])}  积分 sim={_fmt_ms(tm['time_sim_s'])}  "
          f"QP={_fmt_ms(tm['time_qp_s'])}  正则 reg={_fmt_ms(tm['time_reg_s'])}")
    if tm["hot_resolve_cpu_s"]:
        hc = np.asarray(tm["hot_resolve_cpu_s"], float) * 1e3
        print(f"  热启动重解(已收敛后再解 {len(hc)} 次) CPU: "
              f"mean={hc.mean():.1f}ms  min={hc.min():.1f}ms  max={hc.max():.1f}ms")
    print("-" * 64)


# ----------------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------------
def _draw_obstacle(ax, obs: Obstacle):
    import numpy as np
    c = np.array(obs.center, float)
    if obs.kind == "sphere":
        u, vv = np.mgrid[0:2 * np.pi:18j, 0:np.pi:10j]
        xs = c[0] + obs.radius * np.cos(u) * np.sin(vv)
        ys = c[1] + obs.radius * np.sin(u) * np.sin(vv)
        zs = c[2] + obs.radius * np.cos(vv)
        ax.plot_surface(xs, ys, zs, color=obs.color, alpha=0.30, linewidth=0)
    elif obs.kind == "cylinder":
        z = np.linspace(c[2] - obs.height / 2, c[2] + obs.height / 2, 12)
        th = np.linspace(0, 2 * np.pi, 20)
        Th, Z = np.meshgrid(th, z)
        xs = c[0] + obs.radius * np.cos(Th)
        ys = c[1] + obs.radius * np.sin(Th)
        ax.plot_surface(xs, ys, Z, color=obs.color, alpha=0.30, linewidth=0)
    elif obs.kind == "box":
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        hx, hy, hz = np.array(obs.size) / 2
        corners = np.array([[sx, sy, sz] for sx in (-hx, hx)
                            for sy in (-hy, hy) for sz in (-hz, hz)]) + c
        idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
               [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
        faces = [corners[i] for i in idx]
        ax.add_collection3d(Poly3DCollection(faces, facecolor=obs.color,
                            alpha=0.30, edgecolor="k", linewidths=0.3))


def visualize(simX, info, save_path=None, show=True, n_snapshots=7):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pin_model = info["pin_model"]
    data = pin_model.createData()
    obstacles = info["obstacles"]

    base_xyz = simX[:, 0:3]
    ee_xyz = np.zeros_like(base_xyz)
    for i in range(simX.shape[0]):
        q = np.concatenate([simX[i][:3], _normalize_quat(simX[i][3:7]), simX[i][7:9]])
        pin.framesForwardKinematics(pin_model, data, q)
        ee_xyz[i] = np.asarray(data.oMf[info["ee_fid"]].translation, float)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    for obs in obstacles:
        _draw_obstacle(ax, obs)

    ax.plot(base_xyz[:, 0], base_xyz[:, 1], base_xyz[:, 2], "-", color="navy", lw=2, label="base path")
    ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], "-", color="green", lw=2, label="EE path")

    # 起点 / 抓取 / 终点
    ax.scatter(*base_xyz[0], c="black", s=60, marker="o", label="start (base)")
    ax.scatter(*base_xyz[-1], c="black", s=60, marker="s", label="end (base)")
    kg = info["k_grasp"]
    ax.scatter(*ee_xyz[kg], c="red", s=90, marker="*", label="grasp (EE)")

    # 机器人碰撞球快照
    snap_idx = np.linspace(0, simX.shape[0] - 1, n_snapshots).astype(int)
    for si in snap_idx:
        centers, radii = robot_sphere_centers_np(pin_model, data, simX[si])
        for (cc, rr) in zip(centers, radii):
            u, vv = np.mgrid[0:2 * np.pi:8j, 0:np.pi:5j]
            xs = cc[0] + rr * np.cos(u) * np.sin(vv)
            ys = cc[1] + rr * np.sin(u) * np.sin(vv)
            zs = cc[2] + rr * np.cos(vv)
            ax.plot_surface(xs, ys, zs, color="gold", alpha=0.12, linewidth=0)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    cl = info["min_clearance"] * 1000
    ax.set_title(f"S500 UAM obstacle-avoidance trajectory (acados)  status={info['status']}  "
                 f"min clearance={cl:.0f}mm  EE err={info['ee_grasp_err']*1000:.0f}mm")
    ax.legend(loc="upper left", fontsize=8)
    try:
        ax.set_box_aspect((4, 2, 2))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140)
        print(f"图已保存: {save_path}")
    if show:
        plt.show()
    return fig


def _quat_to_euler_np(quat: np.ndarray):
    """[qx,qy,qz,qw] -> (roll, pitch, yaw) rad，ZYX。"""
    qx, qy, qz, qw = quat
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


def plot_states(simX, simU, time_arr, info, save_path=None, show=True):
    """状态时间序列图: 位置/姿态/速度/角速度/关节/控制。"""
    import matplotlib.pyplot as plt

    t = np.asarray(time_arr, float)
    tc = t[:-1]  # 控制比状态少一个
    pin_model = info["pin_model"]
    data = pin_model.createData()

    # 姿态(欧拉角, deg) 与 EE 位置/速度
    rpy = np.zeros((simX.shape[0], 3))
    ee_xyz = np.zeros((simX.shape[0], 3))
    ee_spd = np.zeros(simX.shape[0])
    for i in range(simX.shape[0]):
        rpy[i] = _quat_to_euler_np(_normalize_quat(simX[i][3:7]))
        q = np.concatenate([simX[i][:3], _normalize_quat(simX[i][3:7]), simX[i][7:9]])
        v = simX[i][9:17]
        pin.forwardKinematics(pin_model, data, q, v)
        pin.updateFramePlacements(pin_model, data)
        ee_xyz[i] = np.asarray(data.oMf[info["ee_fid"]].translation, float)
        ee_spd[i] = float(np.linalg.norm(np.asarray(pin.getFrameVelocity(
            pin_model, data, info["ee_fid"], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear, float)))
    rpy_deg = np.degrees(rpy)
    t_grasp = info["k_grasp"] * info["dt"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))

    def _vline(ax):
        ax.axvline(t_grasp, color="red", ls="--", lw=1, alpha=0.7)

    # 1) 基座位置 + EE 位置
    ax = axes[0, 0]
    for k, lab in enumerate(["x", "y", "z"]):
        ax.plot(t, simX[:, k], label=f"base {lab}")
    ax.plot(t, ee_xyz[:, 2], "g--", lw=1, label="EE z")
    _vline(ax); ax.set_title("base position [m]"); ax.set_ylabel("m"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 2) EE 位置(三轴)
    ax = axes[0, 1]
    for k, lab in enumerate(["x", "y", "z"]):
        ax.plot(t, ee_xyz[:, k], label=f"EE {lab}")
    ax.scatter([t_grasp], [1.0], c="red", marker="*", s=80, zorder=5)
    _vline(ax); ax.set_title("end-effector position [m]"); ax.set_ylabel("m"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 3) 姿态欧拉角
    ax = axes[0, 2]
    for k, lab in enumerate(["roll", "pitch", "yaw"]):
        ax.plot(t, rpy_deg[:, k], label=lab)
    _vline(ax); ax.set_title("attitude (Euler ZYX) [deg]"); ax.set_ylabel("deg"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 4) 线速度(机体系)
    ax = axes[1, 0]
    for k, lab in enumerate(["vx", "vy", "vz"]):
        ax.plot(t, simX[:, 9 + k], label=lab)
    _vline(ax); ax.set_title("body linear velocity [m/s]"); ax.set_ylabel("m/s"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 5) 角速度(机体系)
    ax = axes[1, 1]
    for k, lab in enumerate(["wx", "wy", "wz"]):
        ax.plot(t, simX[:, 12 + k], label=lab)
    _vline(ax); ax.set_title("body angular velocity [rad/s]"); ax.set_ylabel("rad/s"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 6) EE 速度幅值
    ax = axes[1, 2]
    ax.plot(t, ee_spd, "g-", label="|v_EE|")
    _vline(ax); ax.set_title("EE speed magnitude [m/s]"); ax.set_ylabel("m/s"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 7) 关节角
    ax = axes[2, 0]
    ax.plot(t, np.degrees(simX[:, 7]), label="j1")
    ax.plot(t, np.degrees(simX[:, 8]), label="j2")
    _vline(ax); ax.set_title("arm joint angles [deg]"); ax.set_xlabel("t [s]"); ax.set_ylabel("deg"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 8) 关节角速度
    ax = axes[2, 1]
    ax.plot(t, simX[:, 15], label="j1_dot")
    ax.plot(t, simX[:, 16], label="j2_dot")
    _vline(ax); ax.set_title("arm joint velocity [rad/s]"); ax.set_xlabel("t [s]"); ax.set_ylabel("rad/s"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 9) 控制量: 4 个旋翼推力 + 机械臂力矩(右轴)
    ax = axes[2, 2]
    for k in range(4):
        ax.plot(tc, simU[:, k], label=f"T{k+1}")
    ax.set_title("controls"); ax.set_xlabel("t [s]"); ax.set_ylabel("thrust [N]")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(tc, simU[:, 4], "k--", lw=1, label="tau1")
    ax2.plot(tc, simU[:, 5], "m--", lw=1, label="tau2")
    ax2.set_ylabel("arm torque [N·m]")
    ax2.legend(fontsize=7, loc="upper right")
    _vline(ax)

    fig.suptitle(f"S500 UAM obstacle-avoidance states (acados)  "
                 f"grasp@t={t_grasp:.2f}s (red dashed)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        fig.savefig(save_path, dpi=140)
        print(f"状态图已保存: {save_path}")
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="S500 UAM 带避障轨迹规划 (acados / min-snap / compare)")
    parser.add_argument("--mode", type=str, default="acados",
                        choices=("acados", "minsnap", "compare"),
                        help="acados: 全非线性OCP; minsnap: 微分平坦+min-snap; compare: 两者对比")
    parser.add_argument("--d1", type=float, default=4.0, help="相位1(起点->抓取)时长 s")
    parser.add_argument("--d2", type=float, default=4.0, help="相位2(抓取->终点)时长 s")
    parser.add_argument("--dwell", type=float, default=0.5, help="[minsnap] 抓取点停留时长 s")
    parser.add_argument("--dt", type=float, default=0.04, help="时间步 s")
    parser.add_argument("--margin", type=float, default=0.05, help="安全裕量 m")
    parser.add_argument("--max-iter", type=int, default=150, help="SQP 最大迭代")
    parser.add_argument("--max-via-iters", type=int, default=10, help="[minsnap] 绕障 via-point 最大迭代")
    parser.add_argument("--reuse-build", action="store_true",
                        help="若已编译过则复用 .so(跳过 codegen/compile,用于评估纯加载+求解)")
    parser.add_argument("--hot-resolves", type=int, default=3,
                        help="收敛后再热启动重解次数(估计 onboard 在线重解速度)")
    parser.add_argument("--save", type=str, default=None, help="保存图片路径(.png)")
    parser.add_argument("--no-show", action="store_true", help="不弹窗显示")
    args = parser.parse_args()

    if args.mode in ("acados", "compare") and not ACADOS_AVAILABLE:
        print("ERROR: acados 未安装/不可用。")
        return 1

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    save = args.save or str(out_dir / "s500_uam_obstacle_avoidance.png")
    show = not args.no_show

    common_kw = dict(d1=args.d1, d2=args.d2, dt=args.dt, margin=args.margin)

    if args.mode == "minsnap":
        from s500_uam_minsnap_planner import plan_minsnap_obstacle_avoidance
        print("=" * 64)
        print("S500 UAM minimum-snap 避障轨迹 (微分平坦 base + dwell@抓取)")
        print("=" * 64)
        simX, simU, time_arr, info = plan_minsnap_obstacle_avoidance(
            dwell_time=args.dwell, max_via_iters=args.max_via_iters, **common_kw,
        )
        visualize(simX, info, save_path=save.replace(".png", "_minsnap.png") if save.endswith(".png") else save + "_minsnap.png", show=False)
        states_save = save.replace(".png", "_minsnap_states.png") if save.endswith(".png") else save + "_minsnap_states.png"
        plot_states(simX, simU, time_arr, info, save_path=states_save, show=show)
        return 0

    if args.mode == "compare":
        from s500_uam_minsnap_planner import plan_minsnap_obstacle_avoidance, compare_and_report
        print("=" * 64)
        print("S500 UAM 避障轨迹对比: acados OCP  vs  minimum snap")
        print("=" * 64)
        print(f"任务: base (-2,0,1.5) → EE抓取(0,0,1) → base (2,0,1.5)  障碍×6\n")

        print(">>> [1/2] acados OCP ...")
        acados_res = plan_obstacle_avoidance_trajectory(
            max_iter=args.max_iter, reuse_build=args.reuse_build,
            hot_resolves=0, **common_kw,
        )
        if acados_res[3]["status"] not in (0, 2):
            print("acados 求解失败"); return 1

        print("\n>>> [2/2] minimum snap ...")
        minsnap_res = plan_minsnap_obstacle_avoidance(
            dwell_time=args.dwell, max_via_iters=args.max_via_iters, **common_kw,
        )

        cmp_save = str(out_dir / "s500_uam_obstacle_compare.png")
        compare_and_report(acados_res, minsnap_res, save_path=cmp_save, show=show)
        return 0

    # default: acados only
    print("=" * 64)
    print("S500 UAM 带避障轨迹规划 (acados, 全身碰撞球)")
    print("=" * 64)
    print(f"起点 base=(-2,0,1.5)  抓取 EE=(0,0,1, v=0)  终点 base=(2,0,1.5)")
    print(f"障碍: 6 个 (球x2, 圆柱x2, 长方体x2)  dt={args.dt}  d1={args.d1}  d2={args.d2}")
    print()

    simX, simU, time_arr, info = plan_obstacle_avoidance_trajectory(
        max_iter=args.max_iter, reuse_build=args.reuse_build,
        hot_resolves=args.hot_resolves, **common_kw,
    )

    if info["status"] not in (0, 2):
        print(f"求解失败 status={info['status']}")
        return 1

    visualize(simX, info, save_path=save, show=False)
    states_save = save.replace(".png", "_states.png") if save.endswith(".png") else save + "_states.png"
    plot_states(simX, simU, time_arr, info, save_path=states_save, show=show)

    npz = save.replace(".png", ".npz") if save.endswith(".png") else save + ".npz"
    np.savez(npz, states=simX, controls=simU, time=time_arr)
    print(f"轨迹已保存: {npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
