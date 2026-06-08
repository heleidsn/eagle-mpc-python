#!/usr/bin/env python3
"""
PX4-style inner loop for Crocoddyl closed-loop simulation (thruster quad + arm).

MPC outputs ideal ``u_mpc = [T1..T4, τ_arm]``. This module:

1. Maps ``u_mpc`` to generalized torque ``τ`` via ``mpc.actuation.calc``.
2. Integrates base angular torque into a **body-rate setpoint** using the current
   rotational inertia block ``M[3:6, 3:6]`` from ``pin.crba``:
   ``ω_sp ← ω_sp + dt * M_ang^{-1} τ_{base,ang}``.
3. Applies a simple **rate PD** to obtain desired body moments for the mixer.
4. Solves the 4-rotor allocation (same linear model as in actuation) for ``T1..T4``
   with constrained ``ΣT = Σ u_mpc[:4]``.

Arm torques are passed through from ``u_mpc[4:6]`` (before optional plant lag).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np
import pinocchio as pin


def thrust_bounds_from_mpc(mpc: Any) -> Tuple[float, float]:
    plat = mpc._planner.s500_config["platform"]
    return float(plat["min_thrust"]), float(plat["max_thrust"])


def thruster_base_moment_jacobian_fd(
    mpc: Any,
    act_data: Any,
    x: np.ndarray,
    u0: np.ndarray,
    eps: float = 1.0,
) -> np.ndarray:
    """Return (3, 4) with columns ∂τ_base[3:6] / ∂T_j (finite differences on thrusters)."""
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    # 注意：act_data.tau 经 np.asarray 可能返回 C++ 缓冲的视图，跨 calc 调用会被覆盖
    # （tau0 与 tau 别名 → 差分恒为 0）。必须 np.array 强制拷贝。
    mpc.actuation.calc(act_data, x, u0)
    tau0 = np.array(act_data.tau, dtype=float).ravel()
    G = np.zeros((3, 4), dtype=float)
    for j in range(4):
        u = u0.copy()
        u[j] += float(eps)
        mpc.actuation.calc(act_data, x, u)
        tau = np.array(act_data.tau, dtype=float).ravel()
        G[:, j] = (tau[3:6] - tau0[3:6]) / float(eps)
    return G


def mix_total_thrust_and_moments(
    T_sum: float,
    M_des: np.ndarray,
    G: np.ndarray,
    T_min: float,
    T_max: float,
) -> np.ndarray:
    """Solve [1^T; G] T = [T_sum; M_des] then clip to box bounds."""
    G = np.asarray(G, dtype=float).reshape(3, 4)
    M_des = np.asarray(M_des, dtype=float).reshape(3)
    A = np.vstack([np.ones((1, 4)), G])
    b = np.concatenate([[float(T_sum)], M_des])
    try:
        T = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        T = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.clip(T, float(T_min), float(T_max))


def _broadcast_gain(g: float | np.ndarray, dim: int = 3) -> np.ndarray:
    g = np.asarray(g, dtype=float).reshape(-1)
    if g.size == 1:
        return np.full(dim, float(g[0]), dtype=float)
    if g.size != dim:
        raise ValueError(f"expected gain length 1 or {dim}, got {g.size}")
    return g


def bodyrate_from_horizon(
    xs_horizon: Optional[List[np.ndarray]],
    lookahead_s: float,
    dt_mpc: float,
    nq: int,
) -> Optional[np.ndarray]:
    """从 MPC horizon 状态在 lookahead 处线性插值出机体角速度设定点 (3,)。

    horizon 状态位于 0, dt_mpc, 2·dt_mpc, ...；取 t=lookahead_s 处的体角速度
    x[nq+3:nq+6] 按相邻格点线性插值。lookahead=dt_mpc 退化为 xs[1]。
    """
    if xs_horizon is None or len(xs_horizon) < 2 or dt_mpc <= 1e-9:
        return None

    def _rates(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).flatten()
        return np.array([x[nq + 3], x[nq + 4], x[nq + 5]], dtype=float)

    n_seg = len(xs_horizon) - 1
    s = min(max(float(lookahead_s) / float(dt_mpc), 0.0), float(n_seg))
    i0 = int(math.floor(s))
    if i0 >= n_seg:
        i0 = n_seg - 1
    frac = s - float(i0)
    if xs_horizon[i0].size < nq + 6 or xs_horizon[i0 + 1].size < nq + 6:
        return None
    return (1.0 - frac) * _rates(xs_horizon[i0]) + frac * _rates(xs_horizon[i0 + 1])


def ctbr_rate_pid_mix(
    mpc: Any,
    act_data: Any,
    x: np.ndarray,
    u_mpc: np.ndarray,
    omega_sp: np.ndarray,
    *,
    kp: float | np.ndarray,
    ki: float | np.ndarray,
    kd: float | np.ndarray,
    integ: np.ndarray,
    prev_err: Optional[np.ndarray],
    dt: float,
    int_limit: float = 0.0,
    max_torque: float = 0.0,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    fd_eps: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """规划角速度设定点 + 角速度 PID + 四旋翼混控 → u_plant。

    与 ROS / acados CTBR 一致：直接用规划 ω_sp 做设定点（不积分力矩），
    M_des = Kp·(ω_sp−ω) + Ki·∫ + Kd·dω/dt，再以总推力 Σu_mpc[:4] 混控分配。
    返回 (u_plant, integ_new, err, M_des)。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    u_mpc = np.asarray(u_mpc, dtype=float).reshape(-1)
    nq = mpc.robot_model.nq
    if thrust_min is None or thrust_max is None:
        tmn, tmx = thrust_bounds_from_mpc(mpc)
        thrust_min = tmn if thrust_min is None else float(thrust_min)
        thrust_max = tmx if thrust_max is None else float(thrust_max)

    omega_meas = x[nq + 3 : nq + 6].copy()
    err = np.asarray(omega_sp, dtype=float).reshape(3) - omega_meas
    integ = np.asarray(integ, dtype=float).reshape(3) + float(dt) * err
    if int_limit > 0.0:
        integ = np.clip(integ, -int_limit, int_limit)
    derr = (err - prev_err) / max(float(dt), 1e-9) if prev_err is not None else np.zeros(3)

    Kp = _broadcast_gain(kp, 3)
    Ki = _broadcast_gain(ki, 3)
    Kd = _broadcast_gain(kd, 3)
    # 惯量归一化：增益解释为闭环带宽 [rad/s]，τ = I_ang·(Kp·e+Ki·∫+Kd·ė)，
    # 与轴转动惯量无关，且在电机一阶滞后下稳定（避免纯高 P 增益失稳）。
    pin.crba(mpc.robot_model, mpc.robot_data, x[:nq])
    I_ang = np.asarray(mpc.robot_data.M[3:6, 3:6], dtype=float)
    M_des = I_ang @ (Kp * err + Ki * integ + Kd * derr)
    if max_torque > 0.0:
        M_des = np.clip(M_des, -max_torque, max_torque)

    G = thruster_base_moment_jacobian_fd(mpc, act_data, x, u_mpc, eps=fd_eps)
    T_sum = float(np.sum(u_mpc[:4]))
    T_rot = mix_total_thrust_and_moments(T_sum, M_des, G, thrust_min, thrust_max)
    u_plant = np.concatenate([T_rot, u_mpc[4:6]]) if u_mpc.size >= 6 else T_rot
    return u_plant, integ, err, M_des


def px4_rate_compute_plant_u(
    mpc: Any,
    act_data: Any,
    x: np.ndarray,
    u_mpc: np.ndarray,
    omega_sp: np.ndarray,
    *,
    sim_dt: float,
    rate_Kp: float | np.ndarray,
    rate_Kd: float | np.ndarray,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    fd_eps: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    u_mpc : (6,) ideal MPC command at this integrator step (ZOH within control_dt).
    omega_sp : (3,) integrated body-rate setpoint (same frame as ``v[3:6]``).

    Returns
    -------
    u_plant : (6,) thrusts + arm torques sent to the plant (before optional 1st-order lag).
    omega_sp_next : (3,) updated setpoint.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    u_mpc = np.asarray(u_mpc, dtype=float).reshape(-1)
    if u_mpc.size < 6:
        raise ValueError(f"px4_rate expects nu>=6, got {u_mpc.size}")
    nq = mpc.robot_model.nq
    nv = mpc.robot_model.nv
    if x.size < nq + nv:
        raise ValueError(f"state size {x.size} < nq+nv={nq + nv}")

    if thrust_min is None or thrust_max is None:
        tmn, tmx = thrust_bounds_from_mpc(mpc)
        thrust_min = tmn if thrust_min is None else float(thrust_min)
        thrust_max = tmx if thrust_max is None else float(thrust_max)

    omega_meas = x[nq + 3 : nq + 6].copy()
    omega_sp = np.asarray(omega_sp, dtype=float).reshape(3).copy()

    mpc.actuation.calc(act_data, x, u_mpc)
    tau = np.asarray(act_data.tau, dtype=float).ravel()
    if tau.size < 6:
        raise ValueError(f"actuation tau too short: {tau.size}")
    tau_ang = tau[3:6].copy()

    q = x[:nq]
    pin.crba(mpc.robot_model, mpc.robot_data, q)
    M = np.asarray(mpc.robot_data.M, dtype=float)
    I_ang = M[3:6, 3:6]
    domega = np.linalg.solve(I_ang + 1e-9 * np.eye(3), tau_ang)
    omega_sp = omega_sp + float(sim_dt) * domega

    Kp = _broadcast_gain(rate_Kp, 3)
    Kd = _broadcast_gain(rate_Kd, 3)
    M_des = Kp * (omega_sp - omega_meas) - Kd * omega_meas

    G = thruster_base_moment_jacobian_fd(mpc, act_data, x, u_mpc, eps=fd_eps)
    T_sum = float(np.sum(u_mpc[:4]))
    T_rot = mix_total_thrust_and_moments(T_sum, M_des, G, thrust_min, thrust_max)
    u_plant = np.concatenate([T_rot, u_mpc[4:6]])
    return u_plant, omega_sp
