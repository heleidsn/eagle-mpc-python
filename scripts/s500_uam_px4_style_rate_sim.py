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

from typing import Any, Tuple

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
    mpc.actuation.calc(act_data, x, u0)
    tau0 = np.asarray(act_data.tau, dtype=float).ravel()
    G = np.zeros((3, 4), dtype=float)
    for j in range(4):
        u = u0.copy()
        u[j] += float(eps)
        mpc.actuation.calc(act_data, x, u)
        tau = np.asarray(act_data.tau, dtype=float).ravel()
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
