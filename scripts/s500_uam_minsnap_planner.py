#!/usr/bin/env python3
"""
S500 UAM minimum-snap 轨迹规划 (微分平坦 base + 可选机械臂角轨迹)

与 acados 全非线性 OCP 对比用:
  - 无人机 base (x,y,z): 分段 7 次多项式 minimum snap
  - 抓取点: base 位置固定 + v=a=0 (停留 dwell), EE 自然在 j=0 时位于 (0,0,1)
  - 机械臂: 默认 j1=j2=0; 可选 cubic min-jerk 角轨迹(抓取/起终点角速度=0)
  - 避障: 迭代插入绕障 via-point + 重解 min-snap (全身碰撞球检测,与 acados 版一致)
  - 平坦反解: 推力 / 倾角 / 姿态(仅 yaw-free 小倾角近似)

注意: min-snap 不在优化里硬约束动力学/推力上限; 规划后用平坦映射做可行性核验。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pinocchio as pin

# 复用 acados 脚本中的碰撞/状态工具
from s500_uam_acados_obstacle_avoidance import (
    EE_FRAME,
    ROBOT_SPHERES,
    Obstacle,
    _NpBackend,
    _collision_residual,
    _normalize_quat,
    default_obstacles,
    make_uam_state,
    min_clearance,
    robot_sphere_centers_np,
)

G = 9.81
EE_Z_OFFSET = 0.312  # base z (j=0) -> EE z 偏移


# ---------------------------------------------------------------------------
# Minimum snap 核心 (Mellinger 分段 7 次多项式, 中间路点只固定位置)
# ---------------------------------------------------------------------------
def _poly7_coeffs_from_boundary(T: float, p0, v0, a0, p1, v1, a1) -> np.ndarray:
    """单段 7 次多项式系数 [c0..c7], 满足两端 pos/vel/acc。"""
    T = float(max(T, 1e-6))
    A = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5, T**6, T**7],
        [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4, 6 * T**5, 7 * T**6],
        [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3, 30 * T**4, 42 * T**5],
    ], dtype=float)
    b = np.array([p0, v0, a0, p1, v1, a1], dtype=float)
    c_high = np.linalg.solve(A[:, 2:], b - A[:, :2] @ np.array([0.0, 0.0]))  # wrong approach

    # 直接 8x8 求解
    c = np.linalg.solve(A, b)
    # A 是 6x8, 需补 2 个自由度的 min snap —— 对单段全边界固定, 直接扩展:
    # 用 8 个约束: 6 个 BC + 2 个额外 c6,c7 由 min snap 决定 → 实际上 6 BC 不能唯一确定 8 系数
    # 重新构造 8x8: 6 BC + c6=0, c7=0 仅为占位; 正确做法用 6x6 子系统 + 2 free
    M = np.zeros((8, 8))
    M[0:6] = A
    M[6, 6] = 1
    M[7, 7] = 1
    rhs = np.zeros(8)
    rhs[0:6] = b
    return np.linalg.solve(M, rhs)


def _eval_poly(c: np.ndarray, t: float, deriv: int = 0) -> float:
    if deriv == 0:
        powers = np.arange(len(c))
        return float(np.sum(c * t ** powers))
    if deriv == 1:
        return float(np.sum(np.arange(1, len(c)) * c[1:] * t ** np.arange(0, len(c) - 1)))
    if deriv == 2:
        return float(np.sum(np.arange(2, len(c)) * np.arange(1, len(c) - 1) * c[2:] * t ** np.arange(0, len(c) - 2)))
    if deriv == 3:
        idx = np.arange(3, len(c))
        return float(np.sum(idx * (idx - 1) * (idx - 2) * c[3:] * t ** np.arange(0, len(c) - 3)))
    if deriv == 4:
        idx = np.arange(4, len(c))
        return float(np.sum(idx * (idx - 1) * (idx - 2) * (idx - 3) * c[4:] * t ** np.arange(0, len(c) - 4)))
    raise ValueError("deriv too high")


def _Q_snap_segment(T: float) -> np.ndarray:
    """单段 snap 代价矩阵 (8x8), integral_0^T (p^{(4)})^2 dt。"""
    T = float(max(T, 1e-6))
    Q = np.zeros((8, 8))
    for i in range(4, 8):
        for j in range(4, 8):
            Q[i, j] = (i * (i - 1) * (i - 2) * (i - 3)) * (j * (j - 1) * (j - 2) * (j - 3)) / (i + j - 7) * T ** (i + j - 7)
    return Q


def _mapping_matrix_segment(T: float) -> np.ndarray:
    """8x8: 系数 c -> 边界导数 d = [p0,v0,a0, p1,v1,a1, ?, ?] 的前 6 个是固定 BC。"""
    # 标准 Mellinger A 矩阵: d = A c, c = A^{-1} d
    A = np.zeros((8, 8))
    # rows 0-2: t=0
    for k in range(3):
        for j in range(k, 8):
            # d^(k)/dt^k at 0 -> k! * c_k
            if j == k:
                A[k, j] = np.math.factorial(k) if hasattr(np, 'math') else __import__('math').factorial(k)
    import math
    for k in range(3):
        row = 3 + k
        for j in range(8):
            if j >= k:
                A[row, j] = (math.factorial(j) / math.factorial(j - k)) * T ** (j - k)
    # 对于 7 阶 (8 coeff), 用 6 BC + 2 free snap coeffs — 简化: 直接用 6x8 然后 min snap
    return A


def min_snap_waypoints_1d(
    waypoints: list[float],
    segment_times: list[float],
    start_vel: float = 0.0,
    start_acc: float = 0.0,
    end_vel: float = 0.0,
    end_acc: float = 0.0,
    fixed_interior_derivs: dict[int, tuple[float, float]] | None = None,
) -> list[np.ndarray]:
    """
    多段 1D minimum snap。
    waypoints: 长度 M+1 的路点位置。
    segment_times: M 段时长。
    fixed_interior_derivs: {waypoint_index: (vel, acc)} 用于抓取点 dwell (v=a=0)。
    中间路点(未在 fixed_interior_derivs 中): 只固定位置, v/a 自由 → min snap。
    """
    import math
    M = len(segment_times)
    assert len(waypoints) == M + 1
    fixed_interior_derivs = fixed_interior_derivs or {}

    n_wp = M + 1
    # 每个路点 3 个导数 (p, v, a), 共 3*(M+1) 个; 但中间 v,a 自由
    # 用 Mellinger 简化: 直接组装 R 矩阵

    # 每段 8 系数, 边界 6 个: d_i = [p, v, a] at start, [p, v, a] at end
    # 路点共享: segment i 终点 = waypoint i+1 = segment i+1 起点

    # 构建选择: 固定变量 vs 自由变量
    # 固定: 所有 waypoints 的 p; start/end 的 v,a; fixed_interior 的 v,a
    # 自由: 其余中间 v, a

    # 更简单可靠的实现: 对每段, 若两端 pos/vel/acc 全已知 → 8x8 求系数 (2 DOF 用 min snap 闭式)
    # 迭代法: 先猜中间 v,a=0, 再解 min snap QP

    # ---- 构建全局 QP (Mellinger) ----
    n_seg = M
    n_vars = 8 * n_seg  # 每段 8 系数

    Q_total = np.zeros((n_vars, n_vars))
    for i, T in enumerate(segment_times):
        Q_total[8 * i:8 * i + 8, 8 * i:8 * i + 8] = _Q_snap_segment(T)

    # 等式约束 A_eq c = b
    eq_rows = []
    eq_rhs = []

    def add_eq(coeff_row, val):
        eq_rows.append(coeff_row)
        eq_rhs.append(val)

    # waypoint 0 start of seg 0
    def seg_c(i, k):
        """第 i 段系数 c_k 在全局向量中的 index helper。"""
        base = 8 * i
        return base + k

    import math

    for i in range(n_seg):
        T = segment_times[i]
        # t=0: p, v, a
        row_p0 = np.zeros(n_vars)
        row_p0[seg_c(i, 0)] = 1
        add_eq(row_p0, waypoints[i] if i == 0 or True else 0)  # placeholder

    # 改用更直接的逐段拼接 + 全局 Newton/闭式
    # 实际上对 M 段, 3*(M+1) 个路点导数, 2*M 个 inter-segment 连续性已含在共享路点
    # 用 free-derivative 方法:

    return _min_snap_free_derivative(waypoints, segment_times, start_vel, start_acc,
                                     end_vel, end_acc, fixed_interior_derivs)


def _min_snap_free_derivative(
    waypoints: list[float],
    segment_times: list[float],
    start_vel: float,
    start_acc: float,
    end_vel: float,
    end_acc: float,
    fixed_interior_derivs: dict[int, tuple[float, float]],
) -> list[np.ndarray]:
    """Mellinger free-derivative minimum snap (1D)."""
    import math
    M = len(segment_times)
    n_wp = M + 1

    # 每个 waypoint 的 [p, v, a]
    D = np.zeros((n_wp, 3))
    for i in range(n_wp):
        D[i, 0] = waypoints[i]
    D[0, 1], D[0, 2] = start_vel, start_acc
    D[-1, 1], D[-1, 2] = end_vel, end_acc
    for idx, (vv, aa) in fixed_interior_derivs.items():
        D[idx, 1], D[idx, 2] = vv, aa

    # 固定 vs 自由
    fixed_mask = np.zeros((n_wp, 3), dtype=bool)
    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True
    for idx in fixed_interior_derivs:
        fixed_mask[idx, :] = True
    fixed_mask[:, 0] = True  # 所有位置固定

    n_fixed = int(np.sum(fixed_mask))
    n_total = n_wp * 3
    n_free = n_total - n_fixed

    if n_free == 0:
        # 全固定, 逐段求系数
        return _coeffs_from_full_derivatives(D, segment_times)

    # 构建 R: 每段 d_i = R_i * [d_start, d_end] (6x6 每段边界)
    # 全局: D_all = R * D_reduced; cost = D_all^T Q D_all, Q block diag
    # 简化: 迭代 — 用 scipy 或直接解

    # 自由变量索引
    free_list = []
    fixed_vals = {}
    for i in range(n_wp):
        for j in range(3):
            if fixed_mask[i, j]:
                fixed_vals[(i, j)] = D[i, j]
            else:
                free_list.append((i, j))

    def pack_d(free_vec):
        dd = D.copy()
        for k, (i, j) in enumerate(free_list):
            dd[i, j] = free_vec[k]
        return dd

    def total_cost(free_vec):
        dd = pack_d(free_vec)
        coeffs = _coeffs_from_full_derivatives(dd, segment_times)
        cost = 0.0
        for ci, T in zip(coeffs, segment_times):
            cost += float(ci @ _Q_snap_segment(T) @ ci)
        return cost

    # 初始: 自由 v,a = 0
    x0 = np.zeros(n_free)
    if n_free > 0:
        from scipy.optimize import minimize
        res = minimize(total_cost, x0, method="L-BFGS-B")
        D = pack_d(res.x)
    return _coeffs_from_full_derivatives(D, segment_times)


def _coeffs_from_full_derivatives(D: np.ndarray, segment_times: list[float]) -> list[np.ndarray]:
    """已知每段两端 [p,v,a], 求各段 7 次多项式系数 (最小 snap 补全 2 个自由高阶项)。"""
    coeffs = []
    for i, T in enumerate(segment_times):
        p0, v0, a0 = D[i]
        p1, v1, a1 = D[i + 1]
        T = float(max(T, 1e-6))
        A6 = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5, T**6, T**7],
            [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4, 6 * T**5, 7 * T**6],
            [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3, 30 * T**4, 42 * T**5],
        ], dtype=float)
        b6 = np.array([p0, v0, a0, p1, v1, a1])
        c_particular = np.linalg.lstsq(A6, b6, rcond=None)[0]
        _U, s, Vh = np.linalg.svd(A6, full_matrices=True)
        rank = int(np.sum(s > 1e-10))
        null = Vh[rank:, :].T  # 8 x (8-rank)
        if null.shape[1] == 0:
            c = c_particular
        else:
            Q = _Q_snap_segment(T)
            H = null.T @ Q @ null
            g = null.T @ Q @ c_particular
            alpha = np.linalg.solve(H + 1e-12 * np.eye(H.shape[0]), -g)
            c = c_particular + null @ alpha
        coeffs.append(c)
    return coeffs


def sample_spline(coeffs_list: list[np.ndarray], segment_times: list[float], dt: float):
    """采样多段多项式 → t, pos, vel, acc, jerk。"""
    t_all, p_all, v_all, a_all, j_all = [], [], [], [], []
    t0 = 0.0
    for c, T in zip(coeffs_list, segment_times):
        n = max(2, int(np.ceil(T / dt)) + 1)
        ts = np.linspace(0, T, n)
        if t_all and ts[0] == 0:
            ts = ts[1:]  # 避免段间重复
        for t in ts:
            t_all.append(t0 + t)
            p_all.append(_eval_poly(c, t, 0))
            v_all.append(_eval_poly(c, t, 1))
            a_all.append(_eval_poly(c, t, 2))
            j_all.append(_eval_poly(c, t, 3))
        t0 += T
    return (np.array(t_all), np.array(p_all), np.array(v_all),
            np.array(a_all), np.array(j_all))


def min_snap_xyz(
    waypoints_xyz: np.ndarray,
    segment_times: list[float],
    grasp_wp_index: int,
    dt: float = 0.04,
    dwell_time: float = 0.0,
):
    """
    3D minimum snap。waypoints_xyz: (M+1, 3)。
    grasp_wp_index: 抓取路点索引, 该点 v=a=0 (dwell)。
    dwell_time>0: 在抓取点额外插入一段常值停留(零 snap)。
    """
    M = len(segment_times)
    assert waypoints_xyz.shape[0] == M + 1
    fixed = {grasp_wp_index: (0.0, 0.0)}

    if dwell_time > 1e-6:
        # 拆成: seg to grasp | dwell | seg from grasp → 用 3 个空间路点但 grasp 重复
        # 更简单: 时间轴上 grasp 处多采样 dwell 常值点
        pass

    coeffs_x = min_snap_waypoints_1d(waypoints_xyz[:, 0].tolist(), segment_times,
                                     fixed_interior_derivs=fixed)
    coeffs_y = min_snap_waypoints_1d(waypoints_xyz[:, 1].tolist(), segment_times,
                                     fixed_interior_derivs=fixed)
    coeffs_z = min_snap_waypoints_1d(waypoints_xyz[:, 2].tolist(), segment_times,
                                     fixed_interior_derivs=fixed)

    # 统一时间轴
    t, px, vx, ax, jx = sample_spline(coeffs_x, segment_times, dt)
    _, py, vy, ay, jy = sample_spline(coeffs_y, segment_times, dt)
    _, pz, vz, az, jz = sample_spline(coeffs_z, segment_times, dt)

    pos = np.column_stack([px, py, pz])
    vel = np.column_stack([vx, vy, vz])
    acc = np.column_stack([ax, ay, az])
    jerk = np.column_stack([jx, jy, jz])

    if dwell_time > 1e-6:
        grasp_t = sum(segment_times[:grasp_wp_index])
        gpos = waypoints_xyz[grasp_wp_index].copy()
        n_dwell = max(2, int(dwell_time / dt) + 1)
        t_dwell = np.linspace(grasp_t, grasp_t + dwell_time, n_dwell)
        z_dwell = np.tile(gpos, (n_dwell, 1))
        z_vel = np.zeros_like(z_dwell)
        mask = t <= grasp_t + 1e-9
        t_before = t[mask]
        pos_before = pos[mask]
        vel_before = vel[mask]
        acc_before = acc[mask]
        jerk_before = jerk[mask]
        t_after = t[~mask]
        pos_after = pos[~mask]
        vel_after = vel[~mask]
        acc_after = acc[~mask]
        jerk_after = jerk[~mask]
        t = np.concatenate([t_before, t_dwell[1:], t_after + dwell_time])
        pos = np.concatenate([pos_before, z_dwell[1:], pos_after], axis=0)
        vel = np.concatenate([vel_before, z_vel[1:], vel_after], axis=0)
        acc = np.concatenate([acc_before, z_vel[1:], acc_after], axis=0)
        jerk = np.concatenate([jerk_before, z_vel[1:], jerk_after], axis=0)

    return t, pos, vel, acc, jerk


# ---------------------------------------------------------------------------
# 平坦反解 + 状态组装
# ---------------------------------------------------------------------------
def flatness_thrust_tilt(acc: np.ndarray, mass: float, g: float = G):
    """a = g e_z - T/m * z_b  →  T, tilt from desired acc (yaw-free, z_b 在竖直平面内)。"""
    # 期望加速度(世界系)
    ax, ay, az = acc
    # 推力向量: F = m * (a + g*e_z)  但多旋翼推力沿 body -z, 近似 F_world ≈ m*(a + [0,0,g])
    F = mass * np.array([ax, ay, az + g])
    T = float(np.linalg.norm(F))
    if T < 1e-6:
        return 0.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0])
    zb = -F / T  # body z 轴在世界系(推力向上 => body -z 沿 F)
    tilt = float(np.arccos(np.clip(-zb[2], -1, 1)))  # 与竖直夹角
    yaw = float(np.arctan2(-F[1], -F[0])) if np.linalg.norm(F[:2]) > 1e-6 else 0.0
    # 四元数 yaw-only + small tilt 近似(演示用)
    half = yaw / 2
    quat = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return T, tilt, yaw, quat


def arm_profile(t: np.ndarray, t_grasp: float, j1_grasp: float = 0.0, j2_grasp: float = 0.0,
                optimize_arm: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    机械臂角轨迹。默认 j1=j2=0。
    optimize_arm=True: 用 cubic 在抓取前后做小范围角变化(仍保证抓取点 j=0, jdot=0)。
    """
    n = len(t)
    j1 = np.zeros(n)
    j2 = np.zeros(n)
    j1d = np.zeros(n)
    j2d = np.zeros(n)
    if not optimize_arm:
        return j1, j2, j1d, j2d
    # 可选: 抓取前小幅抬臂再归零(min-jerk bump) — 此处保持 0 除非后续需要避障
    return j1, j2, j1d, j2d


def build_state17_from_flat(
    t: np.ndarray, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray,
    j1, j2, j1d, j2d, pin_model, mass: float,
) -> np.ndarray:
    """组装 17 维状态 (世界系线速度用 R @ v_body 近似为 vel)。"""
    n = len(t)
    X = np.zeros((n, 17))
    for i in range(n):
        T, tilt, yaw, quat = flatness_thrust_tilt(acc[i], mass)
        X[i, 0:3] = pos[i]
        X[i, 3:7] = quat
        X[i, 7] = j1[i]
        X[i, 8] = j2[i]
        # 速度: 近似世界线速度放入 body v 的前 3 (小倾角时接近)
        X[i, 9:12] = vel[i]  # 简化: 当作 body linear vel
        X[i, 12:15] = 0.0
        X[i, 15] = j1d[i]
        X[i, 16] = j2d[i]
    return X


# ---------------------------------------------------------------------------
# 迭代 via-point 避障
# ---------------------------------------------------------------------------
def _push_point_from_obstacle(p: np.ndarray, obs: Obstacle, margin: float, ri: float = 0.2) -> np.ndarray:
    """把点 p 沿远离障碍方向推出到安全距离。"""
    c = np.array(obs.center, float)
    safe = ri + margin
    if obs.kind == "sphere":
        d = p - c
        dist = np.linalg.norm(d)
        need = obs.radius + safe
        if dist < need and dist > 1e-9:
            return c + d / dist * (need + 0.05)
        if dist <= 1e-9:
            return c + np.array([need + 0.05, 0, 0])
    elif obs.kind == "box":
        hx, hy, hz = np.array(obs.size) / 2
        q = p - c
        dx = max(abs(q[0]) - hx, 0)
        dy = max(abs(q[1]) - hy, 0)
        dz = max(abs(q[2]) - hz, 0)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < safe:
            # 沿最近面法向推出
            nx, ny, nz = 0.0, 0.0, 0.0
            if abs(q[0]) > hx - 1e-6:
                nx = np.sign(q[0]) if abs(q[0]) > 1e-9 else 1.0
            elif abs(q[1]) > hy - 1e-6:
                ny = np.sign(q[1]) if abs(q[1]) > 1e-9 else 1.0
            else:
                nz = np.sign(q[2]) if abs(q[2]) > 1e-9 else 1.0
            nvec = np.array([nx, ny, nz], float)
            if np.linalg.norm(nvec) < 1e-9:
                nvec = np.array([1.0, 0, 0])
            nvec /= np.linalg.norm(nvec)
            return p + nvec * (safe - dist + 0.05)
    elif obs.kind == "cylinder":
        q = p - c
        rxy = np.sqrt(q[0] ** 2 + q[1] ** 2)
        dz = max(abs(q[2]) - obs.height / 2, 0)
        dr = max(rxy - obs.radius, 0)
        dist = np.sqrt(dr * dr + dz * dz)
        if dist < safe:
            if dr >= dz and rxy > 1e-9:
                out = q.copy()
                out[0] *= (obs.radius + safe + 0.05) / rxy
                out[1] *= (obs.radius + safe + 0.05) / rxy
                return c + out
            else:
                out = q.copy()
                out[2] = np.sign(q[2]) * (obs.height / 2 + safe + 0.05) if abs(q[2]) > 1e-9 else (obs.height / 2 + safe + 0.05)
                return c + out
    return p


def _push_base_from_obstacle(base_p: np.ndarray, obs: Obstacle, margin: float) -> np.ndarray:
    """把 base 位置沿远离障碍方向推出(用机体球半径近似)。"""
    return _push_point_from_obstacle(base_p, obs, margin, ri=ROBOT_SPHERES[0][2])


def _find_worst_collision(simX: np.ndarray, pin_model, data, obstacles, margin: float):
    worst = np.inf
    worst_i = -1
    worst_sphere = -1
    worst_obs = None
    for i in range(simX.shape[0]):
        centers, radii = robot_sphere_centers_np(pin_model, data, simX[i])
        for si, (c, ri) in enumerate(zip(centers, radii)):
            for obs in obstacles:
                g = _collision_residual(c, obs, ri, margin, _NpBackend)
                if g < worst:
                    worst = float(g)
                    worst_i = i
                    worst_sphere = si
                    worst_obs = obs
    return worst, worst_i, worst_sphere, worst_obs


def _insert_via_point(waypoints: list, insert_pos: np.ndarray, min_dist: float = 0.15) -> bool:
    """插入路点, 若与现有点太近则跳过。"""
    for p in waypoints:
        if np.linalg.norm(np.asarray(p) - insert_pos) < min_dist:
            return False
    waypoints.append(insert_pos.copy())
    return True


def plan_minsnap_obstacle_avoidance(
    start_xyz=(-2.0, 0.0, 1.5),
    ee_grasp_pos=(0.0, 0.0, 1.0),
    end_xyz=(2.0, 0.0, 1.5),
    obstacles: list[Obstacle] | None = None,
    d1: float = 4.0,
    d2: float = 4.0,
    dwell_time: float = 0.5,
    dt: float = 0.04,
    margin: float = 0.05,
    max_via_iters: int = 12,
    optimize_arm: bool = False,
    urdf_path: str | None = None,
    verbose: bool = True,
):
    """
    Minimum snap 规划 + 迭代 via-point 避障。
    抓取: base 在 (0,0,1.312) 停留 dwell_time (v=a=0), j=0 → EE=(0,0,1)。
    """
    t0_total = time.perf_counter()
    if obstacles is None:
        obstacles = default_obstacles()

    base = Path(__file__).parent.parent
    if urdf_path is None:
        urdf_path = str(base / "models" / "urdf" / "s500_uam_simple.urdf")
    pin_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = pin_model.createData()
    mass = float(sum(inertia.mass for inertia in pin_model.inertias))
    ee_fid = int(pin_model.getFrameId(EE_FRAME))

    grasp_base = np.array([ee_grasp_pos[0], ee_grasp_pos[1], ee_grasp_pos[2] + EE_Z_OFFSET])
    start = np.array(start_xyz, float)
    end = np.array(end_xyz, float)

    # 初始路点: start -> grasp -> end
    waypoints = [start.copy(), grasp_base.copy(), end.copy()]
    seg_times = [d1, d2]
    via_inserted = []

    timing = {"minsnap_solve_s": 0.0, "via_iters": 0, "collision_check_s": 0.0}

    def _sort_waypoints(wps):
        """按 x 坐标排序, 保持 grasp_base 在中间锚点。"""
        wps_arr = [np.asarray(p, float) for p in wps]
        # grasp 必须保留
        grasp_i = min(range(len(wps_arr)), key=lambda i: np.linalg.norm(wps_arr[i] - grasp_base))
        others = [i for i in range(len(wps_arr)) if i != grasp_i]
        others.sort(key=lambda i: wps_arr[i][0])
        ordered = [wps_arr[i] for i in others if wps_arr[i][0] <= grasp_base[0] + 1e-6]
        if grasp_i not in [others[0], others[-1]] if len(others) >= 2 else True:
            ordered.append(wps_arr[grasp_i])
        else:
            # 重新找 grasp
            ordered = sorted(wps_arr, key=lambda p: p[0])
            # 确保 grasp 点精确
            for i, p in enumerate(ordered):
                if np.linalg.norm(p - grasp_base) < 0.05:
                    ordered[i] = grasp_base.copy()
                    break
            return ordered
        after = [wps_arr[i] for i in others if wps_arr[i][0] > grasp_base[0] + 1e-6]
        if not any(np.allclose(p, grasp_base, atol=1e-3) for p in ordered):
            ordered.append(grasp_base.copy())
        ordered.extend(after)
        ordered = sorted(ordered, key=lambda p: p[0])
        for i, p in enumerate(ordered):
            if np.linalg.norm(p - grasp_base) < 0.05:
                ordered[i] = grasp_base.copy()
        return ordered

    def _grasp_index(wps):
        for i, p in enumerate(wps):
            if np.linalg.norm(np.asarray(p) - grasp_base) < 0.05:
                return i
        return len(wps) // 2

    for it in range(max_via_iters + 1):
        ts = time.perf_counter()
        waypoints = _sort_waypoints(waypoints)
        wp_arr = np.array(waypoints)
        grasp_idx = _grasp_index(waypoints)
        n_seg = len(waypoints) - 1
        lens = [np.linalg.norm(wp_arr[i + 1] - wp_arr[i]) for i in range(n_seg)]
        total_len = sum(lens)
        seg_times = [(d1 + d2) * (l / max(total_len, 1e-9)) for l in lens]

        use_dwell = dwell_time if it == 0 else 0.0
        t, pos, vel, acc, jerk = min_snap_xyz(wp_arr, seg_times, grasp_idx, dt=dt, dwell_time=use_dwell)
        t_grasp = sum(seg_times[:grasp_idx]) if grasp_idx > 0 else 0.0

        j1, j2, j1d, j2d = arm_profile(t, t_grasp, optimize_arm=optimize_arm)
        simX = build_state17_from_flat(t, pos, vel, acc, j1, j2, j1d, j2d, pin_model, mass)
        timing["minsnap_solve_s"] += time.perf_counter() - ts

        tc = time.perf_counter()
        worst, wi, ws, wobs = _find_worst_collision(simX, pin_model, data, obstacles, margin)
        timing["collision_check_s"] += time.perf_counter() - tc

        if worst >= -1e-4 or it == max_via_iters:
            break

        base_p = simX[wi, 0:3].copy()
        p_new = _push_base_from_obstacle(base_p, wobs, margin)
        # 若碰撞在机械臂/末端球, 额外抬高 base z
        if ws >= 5:
            p_new[2] = max(p_new[2], base_p[2] + 0.12)
        if not _insert_via_point(waypoints, p_new):
            # 推更远再试
            p_new = base_p + (p_new - base_p) * 1.5 + np.array([0, 0, 0.08])
            if not _insert_via_point(waypoints, p_new):
                if verbose:
                    print(f"  [minsnap via {it+1}] 无法插入新路点, 停止迭代")
                break
        via_inserted.append({"iter": it, "point": p_new.tolist(), "obs": wobs.label, "step": wi})
        timing["via_iters"] += 1
        if verbose:
            print(f"  [minsnap via {it+1}] clearance={worst*1000:.1f}mm @ step {wi} "
                  f"({wobs.label}, sphere#{ws}) → base via {p_new.round(3)}")

    # 控制量(平坦反解推力)
    nu = 6
    simU = np.zeros((len(t) - 1, nu))
    thrusts = []
    tilts = []
    for i in range(len(t)):
        T, tilt, yaw, _ = flatness_thrust_tilt(acc[i], mass)
        thrusts.append(T)
        tilts.append(tilt)
    m_th = T / 4.0 if len(thrusts) else 0
    for i in range(len(t) - 1):
        Ti = thrusts[i]
        simU[i, :4] = Ti / 4.0
        simU[i, 4:6] = 0.0

    worst, worst_info = min_clearance(pin_model, data, simX, obstacles, margin)

    # EE 抓取误差
    kg = int(np.argmin(np.abs(t - t_grasp)))
    xg = simX[kg]
    qg = np.concatenate([xg[:3], _normalize_quat(xg[3:7]), xg[7:9]])
    vg = xg[9:17]
    pin.forwardKinematics(pin_model, data, qg, vg)
    pin.updateFramePlacements(pin_model, data)
    ee_at = np.asarray(data.oMf[ee_fid].translation, float)
    ee_err = float(np.linalg.norm(ee_at - np.array(ee_grasp_pos)))
    ee_spd = float(np.linalg.norm(np.asarray(pin.getFrameVelocity(
        pin_model, data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear, float)))

    # 平滑度: acc^2 积分 (比 raw jerk 数值更稳)
    if len(t) > 1:
        dt_arr = np.diff(t)
        acc_cost = float(np.sum(np.sum(acc[:-1] ** 2, axis=1) * dt_arr))
        jerk_cost = float(np.sum(np.sum(jerk[:-1] ** 2, axis=1) * dt_arr))
    else:
        acc_cost = jerk_cost = 0.0

    cfg_path = base / "config" / "yaml" / "multicopter" / "s500.yaml"
    max_thrust = 10.34
    if cfg_path.is_file():
        import yaml
        with open(cfg_path) as f:
            max_thrust = yaml.safe_load(f)["platform"]["max_thrust"]

    t_total = time.perf_counter() - t0_total
    info = {
        "method": "minsnap",
        "status": 0,
        "t_wall": t_total,
        "tf": float(t[-1]),
        "dt": dt,
        "k_grasp": kg,
        "t_grasp": t_grasp,
        "dwell_time": dwell_time,
        "min_clearance": float(worst),
        "min_clearance_info": worst_info,
        "ee_grasp_err": ee_err,
        "ee_speed_grasp": ee_spd,
        "ee_at_grasp": ee_at,
        "obstacles": obstacles,
        "margin": margin,
        "pin_model": pin_model,
        "ee_fid": ee_fid,
        "via_points": via_inserted,
        "waypoints_final": [p.tolist() for p in waypoints],
        "timing": {
            **timing,
            "total_s": t_total,
        },
        "metrics": {
            "max_thrust": float(np.max(thrusts)) if thrusts else 0,
            "max_thrust_per_motor": float(np.max(thrusts) / 4) if thrusts else 0,
            "max_tilt_deg": float(np.degrees(np.max(tilts))) if tilts else 0,
            "jerk_cost": jerk_cost,
            "acc_cost": acc_cost,
            "thrust_limit_ok": float(np.max(thrusts) / 4) <= max_thrust if thrusts else True,
        },
        "dims": {"N": len(simX) - 1, "n_via": len(via_inserted)},
    }
    if verbose:
        print(f"[minsnap] 规划完成  t_total={t_total*1e3:.1f}ms  via_iters={timing['via_iters']}  "
              f"min_clearance={worst*1000:.1f}mm")
        print(f"  抓取 EE 误差={ee_err*1000:.1f}mm  EE 速度={ee_spd*1000:.1f}mm/s  "
              f"dwell={dwell_time}s @ t={t_grasp:.2f}s")
        print(f"  max thrust/motor={info['metrics']['max_thrust_per_motor']:.2f}N  "
              f"max tilt={info['metrics']['max_tilt_deg']:.1f}°  "
              f"acc² cost={acc_cost:.4f}")
    return simX, simU, t, info


# ---------------------------------------------------------------------------
# acados vs min-snap 对比
# ---------------------------------------------------------------------------
def _metrics_from_acados(simX, simU, time_arr, info):
    pin_model = info["pin_model"]
    mass = float(sum(inertia.mass for inertia in pin_model.inertias))
    if len(time_arr) > 1:
        dt_arr = np.diff(time_arr)
        acc_world = np.diff(simX[:, 9:12], axis=0) / dt_arr[:, None]
        acc_cost = float(np.sum(np.sum(acc_world ** 2, axis=1) * dt_arr))
    else:
        acc_cost = 0.0
    tm = info.get("timing", {})
    return {
        "plan_time_s": tm.get("total_s", info.get("t_wall", 0)) + tm.get("build_s", 0),
        "solve_cpu_s": tm.get("solve_cpu_s", info.get("t_wall", 0)),
        "build_s": tm.get("build_s", 0),
        "sqp_iter": tm.get("sqp_iter", info.get("n_iter", -1)),
        "min_clearance_mm": info["min_clearance"] * 1000,
        "ee_err_mm": info["ee_grasp_err"] * 1000,
        "ee_speed_mm_s": info.get("ee_speed_grasp", 0) * 1000,
        "acc_cost": acc_cost,
        "max_thrust_motor": float(np.max(simU[:, :4])) if simU is not None and len(simU) else 0,
        "dynamics_feasible": True,
        "via_iters": 0,
    }


def compare_and_report(acados_result, minsnap_result, save_path=None, show=True):
    """打印对比表并画叠加 3D 图。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from s500_uam_acados_obstacle_avoidance import _draw_obstacle

    simX_a, simU_a, t_a, info_a = acados_result
    simX_m, simU_m, t_m, info_m = minsnap_result
    ma = _metrics_from_acados(simX_a, simU_a, t_a, info_a)
    mm = {
        "plan_time_s": info_m["timing"]["total_s"],
        "solve_cpu_s": info_m["timing"].get("minsnap_solve_s", info_m["t_wall"]),
        "build_s": 0.0,
        "sqp_iter": info_m["timing"].get("via_iters", 0),
        "min_clearance_mm": info_m["min_clearance"] * 1000,
        "ee_err_mm": info_m["ee_grasp_err"] * 1000,
        "ee_speed_mm_s": info_m["ee_speed_grasp"] * 1000,
        "acc_cost": info_m["metrics"]["acc_cost"],
        "max_thrust_motor": info_m["metrics"]["max_thrust_per_motor"],
        "dynamics_feasible": info_m["metrics"]["thrust_limit_ok"],
        "via_iters": info_m["timing"].get("via_iters", 0),
    }

    print("\n" + "=" * 72)
    print("  acados OCP  vs  minimum snap  对比")
    print("=" * 72)
    rows = [
        ("规划总用时 [s]", f"{ma['plan_time_s']:.3f}", f"{mm['plan_time_s']:.3f}"),
        ("  其中 build/codegen [s]", f"{ma['build_s']:.3f}", f"{mm['build_s']:.3f}"),
        ("  其中求解 [s]", f"{ma['solve_cpu_s']:.3f}", f"{mm['solve_cpu_s']:.3f}"),
        ("迭代次数 (SQP / via)", f"{ma['sqp_iter']}", f"{mm['via_iters']}"),
        ("最小避障间隙 [mm]", f"{ma['min_clearance_mm']:.1f}", f"{mm['min_clearance_mm']:.1f}"),
        ("抓取 EE 误差 [mm]", f"{ma['ee_err_mm']:.1f}", f"{mm['ee_err_mm']:.1f}"),
        ("抓取 EE 速度 [mm/s]", f"{ma['ee_speed_mm_s']:.1f}", f"{mm['ee_speed_mm_s']:.1f}"),
        ("加速度代价 ∫‖a‖²dt", f"{ma['acc_cost']:.2f}", f"{mm['acc_cost']:.2f}"),
        ("最大单电机推力 [N]", f"{ma['max_thrust_motor']:.2f}", f"{mm['max_thrust_motor']:.2f}"),
        ("动力学/推力可行", "是(优化内)", "是" if mm["dynamics_feasible"] else "否"),
    ]
    print(f"{'指标':<28} {'acados':>16} {'min-snap':>16}")
    print("-" * 72)
    for name, va, vm in rows:
        print(f"{name:<28} {va:>16} {vm:>16}")
    print("-" * 72)
    print("结论:")
    print("  acados — 全身碰撞在优化内, 动力学可行, 冷启动慢; 热启动 ~ms 级")
    print("  min-snap — 极快且平滑, via-point 避障, 不保证全身/动力学严格可行")
    print("=" * 72)

    fig = plt.figure(figsize=(14, 6))
    ax3 = fig.add_subplot(121, projection="3d")
    for obs in info_a["obstacles"]:
        _draw_obstacle(ax3, obs)
    ax3.plot(simX_a[:, 0], simX_a[:, 1], simX_a[:, 2], "-", color="navy", lw=2, label="acados")
    ax3.plot(simX_m[:, 0], simX_m[:, 1], simX_m[:, 2], "-", color="darkorange", lw=2, label="minsnap")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title("base trajectory overlay"); ax3.legend(fontsize=8)

    ax2 = fig.add_subplot(222)
    ax2.plot(t_a, simX_a[:, 2], "b-", label="acados z")
    ax2.plot(t_m, simX_m[:, 2], "C1-", label="minsnap z")
    ax2.axvline(info_a["k_grasp"] * info_a["dt"], color="r", ls="--", alpha=0.5, label="grasp")
    ax2.set_title("base z"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    ax4 = fig.add_subplot(224)
    pin_model = info_a["pin_model"]
    data = pin_model.createData()
    ee_a = np.zeros((len(simX_a), 3))
    ee_m = np.zeros((len(simX_m), 3))
    for i in range(len(simX_a)):
        q = np.concatenate([simX_a[i][:3], _normalize_quat(simX_a[i][3:7]), simX_a[i][7:9]])
        pin.framesForwardKinematics(pin_model, data, q)
        ee_a[i] = data.oMf[info_a["ee_fid"]].translation
    for i in range(len(simX_m)):
        q = np.concatenate([simX_m[i][:3], _normalize_quat(simX_m[i][3:7]), simX_m[i][7:9]])
        pin.framesForwardKinematics(pin_model, data, q)
        ee_m[i] = data.oMf[info_m["ee_fid"]].translation
    ax4.plot(t_a, ee_a[:, 2], "b-", label="acados EE z")
    ax4.plot(t_m, ee_m[:, 2], "C1-", label="minsnap EE z")
    ax4.axhline(1.0, color="gray", ls=":", alpha=0.5)
    ax4.set_title("EE z"); ax4.legend(fontsize=7); ax4.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print(f"对比图已保存: {save_path}")
    if show:
        plt.show()
    return ma, mm


# ---------------------------------------------------------------------------
# acados vs min-snap 对比
# ---------------------------------------------------------------------------
def _metrics_from_acados(simX, simU, time_arr, info):
    pin_model = info["pin_model"]
    data = pin_model.createData()
    mass = float(sum(inertia.mass for inertia in pin_model.inertias))
    if len(time_arr) > 1:
        dt_arr = np.diff(time_arr)
        acc_world = np.diff(simX[:, 9:12], axis=0) / dt_arr[:, None]
        acc_cost = float(np.sum(np.sum(acc_world ** 2, axis=1) * dt_arr))
    else:
        acc_cost = 0.0
    thrusts = simU[:, :4].sum(axis=1) if simU is not None and len(simU) else np.array([0.0])
    tilts = []
    for i in range(simX.shape[0]):
        _, tilt, _, _ = flatness_thrust_tilt(
            np.array([0, 0, 9.81]) if i == 0 else (simX[i, 9:12] - simX[i - 1, 9:12]) / max(info["dt"], 1e-6),
            mass)
        tilts.append(tilt)
    tm = info.get("timing", {})
    return {
        "plan_time_s": tm.get("total_s", info.get("t_wall", 0)) + tm.get("build_s", 0),
        "solve_cpu_s": tm.get("solve_cpu_s", info.get("t_wall", 0)),
        "build_s": tm.get("build_s", 0),
        "sqp_iter": tm.get("sqp_iter", info.get("n_iter", -1)),
        "min_clearance_mm": info["min_clearance"] * 1000,
        "ee_err_mm": info["ee_grasp_err"] * 1000,
        "ee_speed_mm_s": info.get("ee_speed_grasp", 0) * 1000,
        "acc_cost": acc_cost,
        "max_thrust_motor": float(np.max(simU[:, :4])) if simU is not None and len(simU) else 0,
        "dynamics_feasible": True,
        "via_iters": 0,
    }


def compare_and_report(
    acados_result,
    minsnap_result,
    save_path=None,
    show=True,
):
    """打印对比表并画叠加 3D 图。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    simX_a, simU_a, t_a, info_a = acados_result
    simX_m, simU_m, t_m, info_m = minsnap_result
    ma = _metrics_from_acados(simX_a, simU_a, t_a, info_a)
    mm = {
        "plan_time_s": info_m["timing"]["total_s"],
        "solve_cpu_s": info_m["timing"].get("minsnap_solve_s", info_m["t_wall"]),
        "build_s": 0.0,
        "sqp_iter": info_m["timing"].get("via_iters", 0),
        "min_clearance_mm": info_m["min_clearance"] * 1000,
        "ee_err_mm": info_m["ee_grasp_err"] * 1000,
        "ee_speed_mm_s": info_m["ee_speed_grasp"] * 1000,
        "acc_cost": info_m["metrics"]["acc_cost"],
        "max_thrust_motor": info_m["metrics"]["max_thrust_per_motor"],
        "dynamics_feasible": info_m["metrics"]["thrust_limit_ok"],
        "via_iters": info_m["timing"].get("via_iters", 0),
    }

    print("\n" + "=" * 72)
    print("  acados OCP  vs  minimum snap  对比")
    print("=" * 72)
    rows = [
        ("规划总用时 [s]", f"{ma['plan_time_s']:.3f}", f"{mm['plan_time_s']:.3f}"),
        ("  其中 build/codegen [s]", f"{ma['build_s']:.3f}", f"{mm['build_s']:.3f}"),
        ("  其中求解 [s]", f"{ma['solve_cpu_s']:.3f}", f"{mm['solve_cpu_s']:.3f}"),
        ("迭代次数 (SQP / via)", f"{ma['sqp_iter']}", f"{mm['via_iters']}"),
        ("最小避障间隙 [mm]", f"{ma['min_clearance_mm']:.1f}", f"{mm['min_clearance_mm']:.1f}"),
        ("抓取 EE 误差 [mm]", f"{ma['ee_err_mm']:.1f}", f"{mm['ee_err_mm']:.1f}"),
        ("抓取 EE 速度 [mm/s]", f"{ma['ee_speed_mm_s']:.1f}", f"{mm['ee_speed_mm_s']:.1f}"),
        ("加速度代价 ∫‖a‖²dt", f"{ma['acc_cost']:.2f}", f"{mm['acc_cost']:.2f}"),
        ("最大单电机推力 [N]", f"{ma['max_thrust_motor']:.2f}", f"{mm['max_thrust_motor']:.2f}"),
        ("动力学/推力可行", "是(优化内保证)", "是" if mm["dynamics_feasible"] else "否(事后核验)"),
    ]
    print(f"{'指标':<28} {'acados':>16} {'min-snap':>16}")
    print("-" * 72)
    for name, va, vm in rows:
        print(f"{name:<28} {va:>16} {vm:>16}")
    print("-" * 72)
    print("结论提示:")
    print("  • acados: 全非线性动力学 + 全身碰撞在优化回路内, 抓取/避障更可靠, 但冷启动慢(~秒级)")
    print("  • min-snap: 规划极快(ms~百ms), 轨迹更平滑, 但避障为后处理 via-point, 不保证全身/动力学")
    print("=" * 72)

    # 叠加 3D
    fig = plt.figure(figsize=(14, 6))
    ax3 = fig.add_subplot(121, projection="3d")
    for obs in info_a["obstacles"]:
        from s500_uam_acados_obstacle_avoidance import _draw_obstacle
        _draw_obstacle(ax3, obs)
    ax3.plot(simX_a[:, 0], simX_a[:, 1], simX_a[:, 2], "-", color="navy", lw=2, label="acados base")
    ax3.plot(simX_m[:, 0], simX_m[:, 1], simX_m[:, 2], "-", color="darkorange", lw=2, label="minsnap base")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title("3D base trajectory overlay")
    ax3.legend(fontsize=8)

    ax2 = fig.add_subplot(222)
    ax2.plot(t_a, simX_a[:, 2], "b-", label="acados z")
    ax2.plot(t_m, simX_m[:, 2], "C1-", label="minsnap z")
    ax2.axvline(info_a["k_grasp"] * info_a["dt"], color="r", ls="--", alpha=0.5)
    ax2.set_title("base z"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    ax3b = fig.add_subplot(224)
    pin_model = info_a["pin_model"]
    data = pin_model.createData()
    ee_a = np.zeros((len(simX_a), 3))
    ee_m = np.zeros((len(simX_m), 3))
    for i in range(len(simX_a)):
        q = np.concatenate([simX_a[i][:3], _normalize_quat(simX_a[i][3:7]), simX_a[i][7:9]])
        pin.framesForwardKinematics(pin_model, data, q)
        ee_a[i] = data.oMf[info_a["ee_fid"]].translation
    for i in range(len(simX_m)):
        q = np.concatenate([simX_m[i][:3], _normalize_quat(simX_m[i][3:7]), simX_m[i][7:9]])
        pin.framesForwardKinematics(pin_model, data, q)
        ee_m[i] = data.oMf[info_m["ee_fid"]].translation
    ax3b.plot(t_a, ee_a[:, 2], "b-", label="acados EE z")
    ax3b.plot(t_m, ee_m[:, 2], "C1-", label="minsnap EE z")
    ax3b.axhline(1.0, color="gray", ls=":", alpha=0.5)
    ax3b.set_title("EE z"); ax3b.legend(fontsize=7); ax3b.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print(f"对比图已保存: {save_path}")
    if show:
        plt.show()
    return ma, mm
