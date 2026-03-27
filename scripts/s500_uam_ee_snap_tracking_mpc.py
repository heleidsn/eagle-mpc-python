#!/usr/bin/env python3
"""
EE-centric trajectory tracking with Acados MPC + minimum-snap EE reference.

Pipeline:
  1) Plan smooth EE position; reference yaw is fixed to 0 during planning (the yaw column in waypoints is no longer used for the reference trajectory).
  2) NMPC tracks EE position + heading using Pinocchio CasADi FK: the heading residual is the difference between [cos ψ, sin ψ] and the reference (avoids ±π jumps) + control regularization.
  3) Closed-loop: RK4 integrates with sim_dt; MPC can be recomputed at control_dt (e.g., 100 Hz), with ZOH holding u between MPC solves.
     4x4 base/EE/arm/control plots consistent with trajectory_gui, 3D base trajectory, tracking errors, and MPC solve statistics overview.

Depends on: acados_template, pinocchio, casadi, numpy, matplotlib, pyyaml
(same stack as s500_uam_acados_trajectory.py)

Usage:
  cd scripts && python s500_uam_ee_snap_tracking_mpc.py
  python s500_uam_ee_snap_tracking_mpc.py --T_sim 6 --N_mpc 40 --dt 0.05
  python s500_uam_ee_snap_tracking_mpc.py --track eight --T_sim 12 --eight_period 8
  python s500_uam_ee_tracking_gui.py   # PyQt5 GUI (trajectory / tracking mode / MPC parameters)

Control input mode (pick one at startup; see -c / --control-mode, or env var S500_UAM_CONTROL_MODE):
  direct / thrusters     → MPC directly optimizes [T1..T4, τ1, τ2]
  actuator / high_level  → MPC optimizes [ω_cmd, T_total, θ_cmd] + a first-order actuator mapping to T/τ

Examples:
  python s500_uam_ee_snap_tracking_mpc.py -c direct
  python s500_uam_ee_snap_tracking_mpc.py --control-mode actuator_first_order
  S500_UAM_CONTROL_MODE=high_level python s500_uam_ee_snap_tracking_mpc.py
"""

from __future__ import annotations

import argparse
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Control-mode aliases -> internal canonical names (consistent with create_ee_tracking_mpc_solver)
CONTROL_MODE_ALIASES = {
    "direct": "direct",
    "thrusters": "direct",
    "actuator_first_order": "actuator_first_order",
    "actuator": "actuator_first_order",
    "high_level": "actuator_first_order",
}

# EE reference trajectory aliases: --track on CLI and env var S500_UAM_EE_TRACK -> internal name
TRACK_TRAJECTORY_ALIASES = {
    "snap": "snap",
    "minimum_snap": "snap",
    "waypoints": "snap",
    "eight": "eight",
    "figure8": "eight",
    "lemniscate": "eight",
}

# -----------------------------------------------------------------------------
# Optional imports (fail with clear message)
# -----------------------------------------------------------------------------
try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False
    AcadosModel = None  # type: ignore

try:
    import casadi as ca
    import pinocchio as pin
    from pinocchio import casadi as cpin
    PINOCCHIO_AVAILABLE = True
except ImportError as e:
    PINOCCHIO_AVAILABLE = False
    _pin_err = e

try:
    from s500_uam_acados_model import build_acados_model, load_s500_config
    from s500_uam_trajectory_planner import make_uam_state
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    _deps_err = e

try:
    from s500_uam_acados_trajectory import plot_acados_into_figure, plot_acados_3d_into_figure

    PLOT_ACADOS_GUI_STYLE = True
except ImportError:
    plot_acados_into_figure = None  # type: ignore
    plot_acados_3d_into_figure = None  # type: ignore
    PLOT_ACADOS_GUI_STYLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EE_FRAME_NAME = "gripper_link"

# Align with s500_uam_acados_trajectory defaults
STATE_LIMITS = {
    "v_max": 1.0,
    "omega_max": 2.0,
    "j_angle_max": 2.0,
    "j_vel_max": 10.0,
}


def _pin_total_mass(robot_model: "pin.Model") -> float:
    """Total mass (sum of Pinocchio inertias, consistent with the magnitude used in the Crocoddyl scripts)."""
    return float(sum(inertia.mass for inertia in robot_model.inertias))


def _normalize_quat_in_state(x: np.ndarray, nq: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).flatten().copy()
    q = x[3:7]
    n = np.linalg.norm(q)
    if n > 1e-9:
        x[3:7] = q / n
    return x


def hover_thrust_controls(
    robot_model: "pin.Model",
    nu: int,
    min_thrust_each: float,
    max_thrust_each: float,
) -> np.ndarray:
    """Approximate hover: distribute mg equally across four thrusts, then clamp to [min_thrust, max_thrust]."""
    g = 9.81
    m = _pin_total_mass(robot_model)
    fz = m * g
    t_each = fz / 4.0
    t_each = float(np.clip(t_each, min_thrust_each, max_thrust_each))
    u = np.zeros(nu, dtype=float)
    u[:4] = t_each
    return u


# =============================================================================
# Minimum snap (1D): piecewise degree-7 polynomials, C3 at interior waypoints
# =============================================================================

def _poly_Q_snap(T: float) -> np.ndarray:
    """Quadratic cost matrix for int_0^T (d^4 p / dt^4)^2 dt, p(t)=sum c_i t^i."""
    Q = np.zeros((8, 8))
    for i in range(4, 8):
        bi = float(math.factorial(i) / math.factorial(i - 4))
        for j in range(4, 8):
            bj = float(math.factorial(j) / math.factorial(j - 4))
            exp = i + j - 7
            if exp <= 0:
                continue
            Q[i, j] = bi * bj * (T**exp) / exp
    Q = (Q + Q.T) * 0.5
    return Q


def _row_deriv_at_tau(k: int, tau: float) -> np.ndarray:
    """Row r such that r @ c = (d^k/dt^k p)(tau), p(t)=sum_{i=0}^7 c_i t^i."""
    r = np.zeros(8)
    if k == 0:
        for i in range(8):
            r[i] = tau**i
        return r
    for i in range(k, 8):
        # k-th derivative of t^i
        coeff = 1.0
        for m in range(k):
            coeff *= (i - m)
        r[i] = coeff * (tau ** (i - k)) if (i - k) >= 0 else 0.0
    return r


def minimum_snap_position_1d(waypoints_y: np.ndarray, times: np.ndarray) -> list[np.ndarray]:
    """
    Solve minimum-snap for one spatial axis.
    waypoints_y[k] at times[k]; rest-to-rest (v=a=j=0) at first and last time.
    Returns list of coefficient vectors c (8,) per segment.
    """
    times = np.asarray(times, dtype=float).flatten()
    w = np.asarray(waypoints_y, dtype=float).flatten()
    assert len(times) == len(w) >= 2
    n_seg = len(times) - 1
    nvar = 8 * n_seg

    Q = np.zeros((nvar, nvar))
    for s in range(n_seg):
        Ts = times[s + 1] - times[s]
        if Ts <= 0:
            raise ValueError("times must be strictly increasing")
        Qblk = _poly_Q_snap(Ts)
        Q[8 * s : 8 * (s + 1), 8 * s : 8 * (s + 1)] = Qblk

    rows = []
    rhs = []

    def add_block_row(left_seg: int | None, A0, right_seg: int | None, A1, bvec):
        """Add constraint [0..A0 on seg left] [A1 on seg right] = b."""
        row = np.zeros(nvar)
        if left_seg is not None:
            row[8 * left_seg : 8 * (left_seg + 1)] = A0
        if right_seg is not None:
            row[8 * right_seg : 8 * (right_seg + 1)] = A1
        rows.append(row)
        rhs.append(bvec)

    # Start (segment 0, tau=0)
    for k in range(4):
        add_block_row(0, _row_deriv_at_tau(k, 0.0), None, None, 0.0 if k > 0 else w[0])

    # End (last segment, tau=T)
    Ts_last = times[-1] - times[-2]
    for k in range(4):
        rk = _row_deriv_at_tau(k, Ts_last)
        add_block_row(n_seg - 1, rk, None, None, 0.0 if k > 0 else w[-1])

    # Interior junctions:
    # - hard waypoint pass-through on segment end position
    # - C1/C2/C3 continuity across segments
    # NOTE: Do not also add k=0 continuity here, it is redundant with
    # the hard position constraint and can make KKT ill-conditioned.
    for s in range(n_seg - 1):
        Ts = times[s + 1] - times[s]
        for k in range(1, 4):
            r_end = _row_deriv_at_tau(k, Ts)
            r_st = _row_deriv_at_tau(k, 0.0)
            add_block_row(s, r_end, s + 1, -r_st, 0.0)
        r_pos_end = _row_deriv_at_tau(0, Ts)
        add_block_row(s, r_pos_end, None, None, w[s + 1])
        # Also pin the next segment start to the same waypoint to avoid positional jumps.
        r_pos_start_next = _row_deriv_at_tau(0, 0.0)
        add_block_row(None, None, s + 1, r_pos_start_next, w[s + 1])

    A = np.vstack(rows)
    b = np.array(rhs, dtype=float)
    m, n = A.shape
    # KKT: min 0.5 c^T Q c  s.t. A c = b  =>  [2Q A^T; A 0] [c; λ] = [0; b]
    kkt = np.block([[2 * Q, A.T], [A, np.zeros((m, m))]])
    rhs_kkt = np.concatenate([np.zeros(nvar), b])
    try:
        sol = np.linalg.solve(kkt, rhs_kkt)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs_kkt, rcond=None)[0]
    c_all = sol[:nvar]

    coeffs = [c_all[8 * s : 8 * (s + 1)].copy() for s in range(n_seg)]
    return coeffs


def eval_poly_segment(c: np.ndarray, tau: float, k_deriv: int = 0) -> float:
    """Evaluate k-th derivative of polynomial with coeffs c at local time tau >= 0."""
    r = _row_deriv_at_tau(k_deriv, tau)
    return float(r @ c)


def sample_ee_minimum_snap_trajectory(
    waypoints_xyz_yaw: np.ndarray,
    times: np.ndarray,
    dt_sample: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Waypoints are (x,y,z) or (x,y,z,yaw). Position and yaw both use minimum snap.
    Yaw waypoints are first unwrapped to avoid 2π discontinuities.

    Returns t_grid, p_ref (N,3), yaw_ref (N,), dp_ref (N,3).
    """
    times = np.asarray(times, dtype=float).flatten()
    W = np.asarray(waypoints_xyz_yaw, dtype=float)
    if W.ndim != 2 or W.shape[0] != len(times):
        raise ValueError("waypoints rows must match len(times)")
    if W.shape[1] == 3:
        W = np.hstack([W, np.zeros((W.shape[0], 1), dtype=float)])
    elif W.shape[1] != 4:
        raise ValueError("waypoints must have shape (K,3) or (K,4)")

    cx = minimum_snap_position_1d(W[:, 0], times)
    cy = minimum_snap_position_1d(W[:, 1], times)
    cz = minimum_snap_position_1d(W[:, 2], times)
    yaw_wp = np.unwrap(W[:, 3].astype(float))
    cyaw = minimum_snap_position_1d(yaw_wp, times)

    t0, tf = times[0], times[-1]
    t_grid = np.arange(t0, tf + 0.5 * dt_sample, dt_sample)
    if t_grid[-1] > tf + 1e-9:
        t_grid = t_grid[:-1]
    if len(t_grid) == 0 or abs(t_grid[-1] - tf) > 1e-6:
        t_grid = np.append(t_grid, tf)

    def seg_index(t):
        for s in range(len(times) - 1):
            if t <= times[s + 1] + 1e-9:
                return s, t - times[s]
        return len(times) - 2, t - times[-2]

    N = len(t_grid)
    p = np.zeros((N, 3))
    dp = np.zeros((N, 3))
    yaw_ref = np.zeros(N)
    for i, t in enumerate(t_grid):
        s, tau = seg_index(t)
        p[i, 0] = eval_poly_segment(cx[s], tau, 0)
        p[i, 1] = eval_poly_segment(cy[s], tau, 0)
        p[i, 2] = eval_poly_segment(cz[s], tau, 0)
        dp[i, 0] = eval_poly_segment(cx[s], tau, 1)
        dp[i, 1] = eval_poly_segment(cy[s], tau, 1)
        dp[i, 2] = eval_poly_segment(cz[s], tau, 1)
        yaw_ref[i] = eval_poly_segment(cyaw[s], tau, 0)
    return t_grid, p, yaw_ref, dp


def sample_ee_figure_eight_trajectory(
    t_duration: float,
    dt_sample: float,
    center: np.ndarray,
    semi_axis: float,
    period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gerono figure-eight (vertical ∞): in the local xy plane
      x = cx + a*sin(θ), y = cy + a*sin(θ)*cos(θ), z = cz
    where θ = 2π * t / period; one closed figure-eight every ``period`` seconds.

    Reference yaw is fixed to 0 (no longer aligned to the velocity tangent).

    Returns t_grid, p_ref (N,3), yaw_ref (N,) zeros, dp_ref (N,3).
    """
    if period <= 0:
        raise ValueError("period must be positive")
    c = np.asarray(center, dtype=float).reshape(3)
    a = float(semi_axis)
    if a <= 0:
        raise ValueError("semi_axis must be positive")
    P = float(period)
    omega = 2.0 * math.pi / P

    t0, tf = 0.0, float(t_duration)
    t_grid = np.arange(t0, tf + 0.5 * dt_sample, dt_sample)
    if len(t_grid) > 0 and t_grid[-1] > tf + 1e-9:
        t_grid = t_grid[:-1]
    if len(t_grid) == 0 or abs(t_grid[-1] - tf) > 1e-6:
        t_grid = np.append(t_grid, tf)

    theta = omega * t_grid
    st = np.sin(theta)
    ct = np.cos(theta)
    # Use sin(θ)cos(θ) = (1/2)sin(2θ) to simplify writing the velocity
    s2t = np.sin(2.0 * theta)

    N = len(t_grid)
    p = np.zeros((N, 3))
    p[:, 0] = c[0] + a * st
    p[:, 1] = c[1] + 0.5 * a * s2t
    p[:, 2] = c[2]

    dp = np.zeros((N, 3))
    dp[:, 0] = a * ct * omega
    dp[:, 1] = a * np.cos(2.0 * theta) * omega
    dp[:, 2] = 0.0

    yaw_ref = np.zeros(N)
    return t_grid, p, yaw_ref, dp


# =============================================================================
# Acados MPC: EE position + heading tracking
# =============================================================================

def _casadi_matrix_to_yaw_world_z(R) -> ca.SX:
    """Closed-form ZYX yaw identical to Pinocchio ``rpy.matrixToRpy(R)[2]``: ``atan2(R[1,0], R[0,0])`` [rad]."""
    return ca.atan2(R[1, 0], R[0, 0])


def _casadi_ee_heading_cs_expr(acados_model, pin_model, frame_id: int) -> tuple[ca.SX, ca.SX]:
    """EE yaw in the world frame (ZYX); returns (cos(yaw), sin(yaw)) for an LS residual without periodic jumps."""
    nq = pin_model.nq
    q = acados_model.x[:nq]
    quat = q[3:7]
    quat_u = quat / ca.fmax(ca.norm_2(quat), 1e-9)
    q_fk = ca.vertcat(q[0:3], quat_u, q[7:nq])
    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()
    cpin.forwardKinematics(cmodel, cdata, q_fk)
    cpin.updateFramePlacements(cmodel, cdata)
    R = cdata.oMf[frame_id].rotation
    yaw = _casadi_matrix_to_yaw_world_z(R)
    return ca.cos(yaw), ca.sin(yaw)


def _casadi_ee_translation_expr(acados_model, pin_model, frame_id: int):
    """CasADi expression for EE world position from state x = [q; v] (safe quaternion normalization to avoid IRK internal-point degeneracy)."""
    nq = pin_model.nq
    q = acados_model.x[:nq]
    quat = q[3:7]
    quat_u = quat / ca.fmax(ca.norm_2(quat), 1e-9)
    q_fk = ca.vertcat(q[0:3], quat_u, q[7:nq])
    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()
    cpin.forwardKinematics(cmodel, cdata, q_fk)
    cpin.updateFramePlacements(cmodel, cdata)
    return cdata.oMf[frame_id].translation


def create_ee_tracking_mpc_solver(
    N: int,
    dt: float,
    w_ee: float = 500.0,
    w_ee_yaw: float = 200.0,
    w_u: float = 1e-4,
    w_ue: np.ndarray | None = None,
    max_iter: int = 40,
    warm_start: bool = True,
    control_mode: str = "direct",
) -> tuple:
    """
    NMPC horizon N, sampling time dt.

    Cost (NONLINEAR_LS): w_ee·||p_ee - p_ref||² + w_ee_yaw·(||c - c_ref||² + ||s - s_ref||²)
    + ||diag(w_ue)(u - u_ref)||², where (c,s) = (cos ψ, sin ψ) is the EE world-frame ZYX yaw consistent with the plan.

    control_mode:
      - ``direct``: u = [T1..T4, τ1, τ2] (consistent with the existing model)
      - ``actuator_first_order``: u = [ω_cmd(3), T_total_cmd, θ_cmd(2)]; the state is augmented with u_act,
        which first-order lags to the actual thrust/torque (see s500_uam_acados_actuator_layer.py)
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("acados_template not installed")
    if not PINOCCHIO_AVAILABLE:
        raise ImportError(f"pinocchio/casadi required: {_pin_err}")
    if not DEPS_OK:
        raise ImportError(f"project deps: {_deps_err}")

    ocp = AcadosOcp()
    if control_mode == "direct":
        acados_model, pin_model, nq, nv, nu = build_acados_model()
    elif control_mode == "actuator_first_order":
        from s500_uam_acados_actuator_layer import build_acados_model_actuator_first_order

        acados_model, pin_model, nq, nv, nu, _meta = build_acados_model_actuator_first_order()
    else:
        raise ValueError("control_mode must be 'direct' or 'actuator_first_order'")

    try:
        frame_id = pin_model.getFrameId(EE_FRAME_NAME)
    except Exception as e:
        raise ValueError(f"Frame '{EE_FRAME_NAME}' not found in URDF model") from e
    if frame_id < 0 or frame_id >= pin_model.nframes:
        raise ValueError(f"Frame '{EE_FRAME_NAME}' not found in URDF model")

    ee_p = _casadi_ee_translation_expr(acados_model, pin_model, frame_id)
    c_psi, s_psi = _casadi_ee_heading_cs_expr(acados_model, pin_model, frame_id)
    if w_ue is None:
        if control_mode == "direct":
            w_ue = np.array([1.0, 1.0, 1.0, 1.0, 50.0, 50.0], dtype=float)
        else:
            # ω(3), T_total, θ(2)
            w_ue = np.array([0.5, 0.5, 0.5, 5.0e-4, 50.0, 50.0], dtype=float)
    cost_y = ca.vertcat(ee_p, c_psi, s_psi, acados_model.u)
    ny = int(cost_y.shape[0])
    W = np.zeros((ny, ny))
    W[:3, :3] = np.eye(3) * w_ee
    W[3, 3] = w_ee_yaw
    W[4, 4] = w_ee_yaw
    for i in range(nu):
        W[5 + i, 5 + i] = w_ue[i]

    ocp.model = acados_model
    ocp.model.cost_y_expr = cost_y
    ocp.model.cost_y_expr_e = ca.vertcat(ee_p, c_psi, s_psi)
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = W
    W_e = np.zeros((5, 5))
    W_e[:3, :3] = np.eye(3) * (w_ee * 2.0)
    W_e[3, 3] = w_ee_yaw * 2.0
    W_e[4, 4] = w_ee_yaw * 2.0
    ocp.cost.W_e = W_e

    ocp.dims.N = N
    ocp.solver_options.tf = N * dt
    ocp.solver_options.nlp_solver_max_iter = max_iter
    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = N

    yref = np.zeros(ny)
    ocp.cost.yref = yref
    ocp.cost.yref_e = np.zeros(5)

    cfg = load_s500_config()
    platform = cfg["platform"]
    min_thrust = platform["min_thrust"]
    max_thrust = platform["max_thrust"]

    v_max = STATE_LIMITS["v_max"]
    om_max = STATE_LIMITS["omega_max"]
    j_max = STATE_LIMITS["j_angle_max"]
    jv_max = STATE_LIMITS["j_vel_max"]

    if control_mode == "direct":
        ocp.constraints.lbu = np.array([min_thrust] * 4 + [-2.0] * 2)
        ocp.constraints.ubu = np.array([max_thrust] * 4 + [2.0] * 2)
    else:
        # High-level commands: body angular rates, total thrust, and joint angle commands
        ocp.constraints.lbu = np.array(
            [-2.0 * om_max, -2.0 * om_max, -2.0 * om_max, 4.0 * min_thrust, -j_max, -j_max]
        )
        ocp.constraints.ubu = np.array(
            [2.0 * om_max, 2.0 * om_max, 2.0 * om_max, 4.0 * max_thrust, j_max, j_max]
        )
    ocp.constraints.idxbu = np.arange(nu)

    nx = nq + nv + (6 if control_mode == "actuator_first_order" else 0)
    robot_lbx = np.concatenate(
        [
            np.array([-8.0, -8.0, 0.05, -1.0, -1.0, -1.0, -1.0]),
            np.array([-j_max, -j_max]),
            np.array([-v_max, -v_max, -v_max, -om_max, -om_max, -om_max, -jv_max, -jv_max]),
        ]
    )
    robot_ubx = np.concatenate(
        [
            np.array([8.0, 8.0, 6.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([j_max, j_max]),
            np.array([v_max, v_max, v_max, om_max, om_max, om_max, jv_max, jv_max]),
        ]
    )
    if control_mode == "direct":
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx = robot_lbx
        ocp.constraints.ubx = robot_ubx
    else:
        uact_lbx = np.concatenate([np.full(4, min_thrust), np.full(2, -2.0)])
        uact_ubx = np.concatenate([np.full(4, max_thrust), np.full(2, 2.0)])
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx = np.concatenate([robot_lbx, uact_lbx])
        ocp.constraints.ubx = np.concatenate([robot_ubx, uact_ubx])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # With the actuator-augmented model, IRK internal points can make the quaternion non-unit and cause NaNs in ABA/FK (status=1); ERK is usually more stable.
    if control_mode == "actuator_first_order":
        ocp.solver_options.integrator_type = "ERK"
    else:
        ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0
    if hasattr(ocp.solver_options, "qp_solver_iter_max"):
        ocp.solver_options.qp_solver_iter_max = max(50, max_iter * 2)
    if hasattr(ocp.solver_options, "nlp_solver_tol_stat"):
        ocp.solver_options.nlp_solver_tol_stat = 1e-3
    if hasattr(ocp.solver_options, "nlp_solver_tol_eq"):
        ocp.solver_options.nlp_solver_tol_eq = 1e-3

    if control_mode == "direct":
        code_export_dir = REPO_ROOT / "c_generated_code" / "s500_uam_ee_track_mpc_pose"
    else:
        code_export_dir = REPO_ROOT / "c_generated_code" / "s500_uam_ee_track_mpc_act1_pose"
    code_export_dir.mkdir(parents=True, exist_ok=True)
    json_path = code_export_dir / "ocp.json"
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(json_path)

    ocp.constraints.x0 = np.zeros(nx)

    solver = AcadosOcpSolver(
        ocp,
        json_file=str(json_path),
        build=False,
        generate=False,
        verbose=False,
        check_reuse_possible=True,
    )
    return solver, acados_model, pin_model, nq, nv, nu, control_mode


def _make_f_expl_fun(acados_model):
    """Numpy-callable explicit dynamics xdot = f(x,u)."""
    return ca.Function("f", [acados_model.x, acados_model.u], [acados_model.f_expl_expr])


def rk4_step(f_fun, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).flatten()
    u = np.asarray(u, dtype=float).flatten()
    k1 = np.array(f_fun(x, u)).flatten()
    k2 = np.array(f_fun(x + 0.5 * dt * k1, u)).flatten()
    k3 = np.array(f_fun(x + 0.5 * dt * k2, u)).flatten()
    k4 = np.array(f_fun(x + dt * k3, u)).flatten()
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout_nominal_trajectory(
    f_fun,
    x0: np.ndarray,
    u_nom: np.ndarray,
    dt: float,
    N: int,
    nq: int,
) -> list[np.ndarray]:
    """Roll out with constant control along the horizon to obtain dynamics-consistent initial values x(0..N) (normalize the quaternion at each step)."""
    xs = []
    x = _normalize_quat_in_state(x0, nq)
    xs.append(x.copy())
    u_nom = np.asarray(u_nom, dtype=float).flatten()
    for _ in range(N):
        x = rk4_step(f_fun, x, u_nom, dt)
        x = _normalize_quat_in_state(x, nq)
        xs.append(x.copy())
    return xs


def set_solver_initial_guess(
    solver: AcadosOcpSolver,
    x_current: np.ndarray,
    x_rollout: list[np.ndarray],
    u_hover: np.ndarray,
    N: int,
):
    """
    Set SQP initial guesses:
    - stage0 = current ground truth
    - other stages from rollout
    - u at each stage = hover thrust

    Avoid default all-zero (invalid quaternion; thrust below min_thrust), which can cause the first-step QP to fail.
    """
    x0 = np.asarray(x_current, dtype=float).flatten()
    assert len(x_rollout) == N + 1
    for i in range(N + 1):
        solver.set(i, "x", x_rollout[i])
    solver.set(0, "x", x0)
    uh = np.asarray(u_hover, dtype=float).flatten()
    for i in range(N):
        solver.set(i, "u", uh)


def shift_solver_initial_guess(
    solver: AcadosOcpSolver,
    x_current: np.ndarray,
    x_prev: list[np.ndarray],
    u_prev: list[np.ndarray],
    N: int,
):
    """Shift the previous solution:
    x̃_k ← x*_{k+1}, then set x̃_0 to the current measurement (classic MPC warm start)."""
    for k in range(N):
        solver.set(k, "x", x_prev[k + 1])
    solver.set(N, "x", x_prev[N])
    solver.set(0, "x", np.asarray(x_current, dtype=float).flatten())
    for i in range(N - 1):
        solver.set(i, "u", u_prev[i + 1])
    solver.set(N - 1, "u", u_prev[N - 1])


def _acados_nlp_iterations(solver: AcadosOcpSolver) -> int:
    """Number of SQP iterations (compatible with different acados statistics field names)."""
    n_iter = solver.get_stats("nlp_iter")
    if n_iter is None:
        n_iter = solver.get_stats("sqp_iter")
    if n_iter is None:
        n_iter = solver.get_stats("ddp_iter")
    try:
        return int(n_iter) if n_iter is not None else -1
    except (TypeError, ValueError):
        return -1


def _acados_cpu_time_s(solver: AcadosOcpSolver) -> float:
    """CPU time of this solve() [s] (acados internal statistics)."""
    t_cpu = solver.get_stats("time_tot")
    if t_cpu is None:
        return float("nan")
    try:
        return float(t_cpu)
    except (TypeError, ValueError):
        return float("nan")


def align_uam_state_ee_to_world_position(
    x_robot: np.ndarray,
    pin_model: pin.Model,
    ee_position_world_des: np.ndarray,
    nq: int,
    nv: int,
) -> np.ndarray:
    """
    Translate the floating-base origin ``q[0:3]`` (world frame) so that the world position of ``EE_FRAME_NAME`` matches the desired target point.
    Quaternions, joints, and velocities remain unchanged; this is equivalent to a rigid-body translation of the whole coupled system.
    """
    x = np.asarray(x_robot, dtype=float).flatten().copy()
    p_des = np.asarray(ee_position_world_des, dtype=float).reshape(3)
    data = pin_model.createData()
    q = x[:nq].copy()
    v = x[nq : nq + nv].copy()
    pin.forwardKinematics(pin_model, data, q, v)
    pin.updateFramePlacements(pin_model, data)
    fid = pin_model.getFrameId(EE_FRAME_NAME)
    ee = np.array(data.oMf[fid].translation, dtype=float).flatten()
    delta = p_des - ee
    if float(np.linalg.norm(delta)) > 1e-9:
        print(
            f"[Initial alignment] Translate base Δxyz=({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}) m "
            f"so the EE start coincides with p_ref(0)"
        )
    x[0] += float(delta[0])
    x[1] += float(delta[1])
    x[2] += float(delta[2])
    return x


def interp_ref(t: float, t_grid: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """Linear interpolation of reference position."""
    if t <= t_grid[0]:
        return p_ref[0].copy()
    if t >= t_grid[-1]:
        return p_ref[-1].copy()
    idx = np.searchsorted(t_grid, t) - 1
    t0, t1 = t_grid[idx], t_grid[idx + 1]
    a = (t - t0) / (t1 - t0)
    return (1 - a) * p_ref[idx] + a * p_ref[idx + 1]


def interp_ref_yaw(t: float, t_grid: np.ndarray, yaw_ref: np.ndarray) -> float:
    """Linearly interpolate the continuous planned yaw(t) (after unwrap) [rad]."""
    y = np.asarray(yaw_ref, dtype=float).flatten()
    if t <= t_grid[0]:
        return float(y[0])
    if t >= t_grid[-1]:
        return float(y[-1])
    idx = np.searchsorted(t_grid, t) - 1
    t0, t1 = t_grid[idx], t_grid[idx + 1]
    a = (t - t0) / (t1 - t0)
    return float((1.0 - a) * y[idx] + a * y[idx + 1])


def _yaw_error_wrapped(ee_yaw: float, yaw_ref: float) -> float:
    """Yaw error: EE yaw minus reference, wrapped to (-π, π]."""
    return float(np.arctan2(np.sin(ee_yaw - yaw_ref), np.cos(ee_yaw - yaw_ref)))


def run_closed_loop(
    x0: np.ndarray,
    t_grid_ref: np.ndarray,
    p_ref: np.ndarray,
    yaw_ref: np.ndarray,
    solver: AcadosOcpSolver,
    acados_model: AcadosModel,
    pin_model: pin.Model,
    nq: int,
    nv: int,
    nu: int,
    dt_mpc: float,
    N: int,
    T_sim: float,
    sim_dt: float,
    control_dt: float | None = None,
    mpc_log_interval: int = 1,
    control_mode: str = "direct",
) -> dict:
    """Closed-loop: integrate dynamics with RK4 using ``sim_dt``; recompute MPC every ``control_dt`` (ZOH holds ``u``).

    When ``control_dt`` is ``None``, it matches ``sim_dt``, i.e., solve MPC at every simulation step.
    ``mpc_log_interval``: print statistics every k **MPC solves** (not sim sub-steps); 0 = only print the final summary.
    """
    from s500_uam_acados_actuator_layer import nominal_command_hover

    f_fun = _make_f_expl_fun(acados_model)

    n_robot = nq + nv
    nx = n_robot + (6 if control_mode == "actuator_first_order" else 0)
    sim_dt = float(sim_dt)
    if control_dt is None:
        control_dt = sim_dt
    else:
        control_dt = float(control_dt)
    if control_dt < sim_dt - 1e-15:
        raise ValueError("control_dt must be >= sim_dt (use ZOH at sim rate)")
    ratio = control_dt / sim_dt
    mpc_stride = 1 if ratio <= 1.0 + 1e-12 else max(1, int(round(ratio)))
    eff_ctl = mpc_stride * sim_dt
    if mpc_stride > 1 and abs(eff_ctl - control_dt) > 1e-5 * max(control_dt, sim_dt):
        print(
            f"Warning: control_dt={control_dt:g} s and sim_dt={sim_dt:g} s are not an integer multiple; "
            f"using stride={mpc_stride} (effective control period {eff_ctl:g} s)",
            file=sys.stderr,
        )

    n_steps = int(round(T_sim / sim_dt))
    t_log = np.zeros(n_steps + 1)
    x_log = np.zeros((n_steps + 1, nx))
    u_log = np.zeros((n_steps, nu))
    ee_log = np.zeros((n_steps + 1, 3))
    p_ref_log = np.zeros((n_steps + 1, 3))
    err_log = np.zeros(n_steps + 1)
    ee_yaw_log = np.zeros(n_steps + 1)
    yaw_ref_log = np.zeros(n_steps + 1)
    err_yaw_log = np.zeros(n_steps + 1)

    x = np.asarray(x0, dtype=float).flatten().copy()
    data = pin_model.createData()
    ee_id = pin_model.getFrameId(EE_FRAME_NAME)

    t_log[0] = 0.0
    x_log[0] = x
    q0 = x[:nq]
    pin.forwardKinematics(pin_model, data, q0, x[nq:n_robot])
    pin.updateFramePlacements(pin_model, data)
    ee_log[0] = data.oMf[ee_id].translation
    ee_yaw_log[0] = float(pin.rpy.matrixToRpy(data.oMf[ee_id].rotation)[2])
    p_ref_log[0] = interp_ref(0.0, t_grid_ref, p_ref)
    yaw_ref_log[0] = interp_ref_yaw(0.0, t_grid_ref, yaw_ref)
    err_log[0] = np.linalg.norm(ee_log[0] - p_ref_log[0])
    err_yaw_log[0] = _yaw_error_wrapped(ee_yaw_log[0], yaw_ref_log[0])

    cfg = load_s500_config()
    plat = cfg["platform"]
    if control_mode == "actuator_first_order":
        u_hover = nominal_command_hover(
            pin_model, x[:n_robot], plat["min_thrust"], plat["max_thrust"]
        )
    else:
        u_hover = hover_thrust_controls(pin_model, nu, plat["min_thrust"], plat["max_thrust"])
    if n_steps > 0:
        if control_mode == "actuator_first_order":
            print(
                f"MPC warm start (high-level command): T_total≈{u_hover[3]:.2f} N, θ_cmd≈({u_hover[4]:.3f},{u_hover[5]:.3f}) rad, "
                f"mass≈{_pin_total_mass(pin_model):.3f} kg, ||EE - p_ref(0)|| = {err_log[0]:.4f} m, "
                f"|yaw err| = {abs(err_yaw_log[0]):.4f} rad"
            )
        else:
            print(
                f"MPC warm start: hover thrust/rotor ≈ {u_hover[0]:.3f} N (mass≈{_pin_total_mass(pin_model):.3f} kg), "
                f"initial ||EE - p_ref(0)|| = {err_log[0]:.4f} m, |yaw err| = {abs(err_yaw_log[0]):.4f} rad"
            )

    x_prev: list[np.ndarray] | None = None
    u_prev: list[np.ndarray] | None = None

    mpc_n_iter = np.zeros(n_steps, dtype=int)
    mpc_cpu_s = np.full(n_steps, np.nan, dtype=float)
    mpc_wall_s = np.zeros(n_steps, dtype=float)
    mpc_status = np.zeros(n_steps, dtype=int)

    u_apply = np.zeros(nu, dtype=float)
    solve_idx = 0

    for k in range(n_steps):
        t_k = k * sim_dt
        do_mpc = k % mpc_stride == 0

        if do_mpc:
            # SQP initial guess: first solve uses hover rollout; afterwards shift the previous optimal solution
            if k == 0 or x_prev is None:
                x_roll = rollout_nominal_trajectory(f_fun, x, u_hover, dt_mpc, N, nq)
                set_solver_initial_guess(solver, x, x_roll, u_hover, N)
            else:
                shift_solver_initial_guess(solver, x, x_prev, u_prev, N)

            solver.constraints_set(0, "lbx", x, api="new")
            solver.constraints_set(0, "ubx", x, api="new")

            u_ref = (
                nominal_command_hover(pin_model, x[:n_robot], plat["min_thrust"], plat["max_thrust"])
                if control_mode == "actuator_first_order"
                else u_hover
            )
            for i in range(N):
                ti = t_k + i * dt_mpc
                pref = interp_ref(ti, t_grid_ref, p_ref)
                yawi = interp_ref_yaw(ti, t_grid_ref, yaw_ref)
                cy, sy = np.cos(yawi), np.sin(yawi)
                yref = np.concatenate([pref, [cy, sy], u_ref])
                solver.cost_set(i, "yref", yref, api="new")
            pref_e = interp_ref(t_k + N * dt_mpc, t_grid_ref, p_ref)
            yaw_e = interp_ref_yaw(t_k + N * dt_mpc, t_grid_ref, yaw_ref)
            yref_e = np.concatenate([pref_e, [np.cos(yaw_e), np.sin(yaw_e)]])
            solver.cost_set(N, "yref", yref_e, api="new")

            t_wall0 = time.perf_counter()
            status = solver.solve()
            wall_s = time.perf_counter() - t_wall0
            n_iter = _acados_nlp_iterations(solver)
            cpu_s = _acados_cpu_time_s(solver)
            ms_per_iter = (cpu_s / n_iter * 1000.0) if n_iter > 0 else float("nan")

            mpc_n_iter[k] = n_iter
            mpc_cpu_s[k] = cpu_s
            mpc_wall_s[k] = wall_s
            mpc_status[k] = int(status)

            is_last_mpc_before_end = k + mpc_stride >= n_steps
            do_print = mpc_log_interval > 0 and (
                solve_idx % mpc_log_interval == 0 or is_last_mpc_before_end
            )
            if do_print:
                st_str = "" if status == 0 else f" | status={status}"
                print(
                    f"[MPC solve={solve_idx:5d}] t_sim={t_k:.4f}s | n_iter={n_iter} | "
                    f"CPU={cpu_s:.4f}s | wall={wall_s:.4f}s | {ms_per_iter:.2f} ms/iter (avg){st_str}"
                )
            elif status != 0:
                print(f"Warning: acados status {status} at sim k={k}, t={t_k:.4f}")

            x_prev = [solver.get(i, "x").flatten().copy() for i in range(N + 1)]
            u_prev = [solver.get(i, "u").flatten().copy() for i in range(N)]

            u_apply = np.asarray(solver.get(0, "u"), dtype=float).flatten().copy()
            solve_idx += 1

        u_log[k] = u_apply
        x = rk4_step(f_fun, x, u_apply, sim_dt)

        t_log[k + 1] = t_k + sim_dt
        x_log[k + 1] = x
        q = x[:nq]
        pin.forwardKinematics(pin_model, data, q, x[nq:n_robot])
        pin.updateFramePlacements(pin_model, data)
        ee_log[k + 1] = data.oMf[ee_id].translation
        ee_yaw_log[k + 1] = float(pin.rpy.matrixToRpy(data.oMf[ee_id].rotation)[2])
        p_ref_log[k + 1] = interp_ref(t_k + sim_dt, t_grid_ref, p_ref)
        yaw_ref_log[k + 1] = interp_ref_yaw(t_k + sim_dt, t_grid_ref, yaw_ref)
        err_log[k + 1] = np.linalg.norm(ee_log[k + 1] - p_ref_log[k + 1])
        err_yaw_log[k + 1] = _yaw_error_wrapped(ee_yaw_log[k + 1], yaw_ref_log[k + 1])

    if n_steps > 0:
        solved = mpc_n_iter > 0
        n_solve = int(np.sum(solved))
        if n_solve > 0:
            print(
                f"MPC solve summary: {n_solve} MPC solves / {n_steps} simulation steps "
                f"(sim_dt={sim_dt:g}s, stride={mpc_stride} → equivalent control {eff_ctl:g}s) | "
                f"avg wall={np.mean(mpc_wall_s[solved]):.4f}s | "
                f"avg CPU={np.nanmean(mpc_cpu_s[solved]):.4f}s | "
                f"avg n_iter={np.mean(mpc_n_iter[solved]):.2f} | "
                f"total wall={np.sum(mpc_wall_s):.3f}s"
            )
        else:
            print("MPC solve summary: no MPC steps (check n_steps / stride)")

    out = {
        "t": t_log,
        "x": x_log,
        "u": u_log,
        "ee": ee_log,
        "p_ref": p_ref_log,
        "err": err_log,
        "ee_yaw": ee_yaw_log,
        "yaw_ref": yaw_ref_log,
        "err_yaw": err_yaw_log,
        "control_mode": control_mode,
        "sim_dt": sim_dt,
        "control_dt": eff_ctl,
        "mpc_stride": mpc_stride,
        "mpc_solve": {
            "nlp_iter": mpc_n_iter,
            "cpu_s": mpc_cpu_s,
            "wall_s": mpc_wall_s,
            "status": mpc_status,
        },
    }
    if control_mode == "actuator_first_order":
        out["u_act"] = x_log[:, n_robot : n_robot + 6]
    return out


def _suffix_figure_path(base: Path, suffix: str) -> Path:
    return base.with_name(base.stem + suffix + base.suffix)


def _extract_simX_closed_loop(res: dict) -> np.ndarray:
    """Extract robot [q;v] (17-dim) from the closed-loop state log; consistent with trajectory_gui / plot_acados."""
    x = np.asarray(res["x"], dtype=float)
    n_robot = min(17, x.shape[1])
    return x[:, :n_robot].copy()


def _mpc_timing_info_for_acados_plot(res: dict) -> dict:
    """Rough statistics used by the bottom-right placeholder of plot_acados_into_figure (avg SQP time per step)."""
    ms = res.get("mpc_solve") or {}
    nit = np.asarray(ms.get("nlp_iter", []), dtype=float)
    cpu = np.asarray(ms.get("cpu_s", []), dtype=float)
    if nit.size == 0:
        return {"n_iter": 0, "avg_ms_per_iter": 0.0, "total_s": 0.0}
    valid = nit > 0
    per_ms = np.where(valid, cpu / nit * 1000.0, np.nan)
    return {
        "n_iter": int(np.sum(nit)) if np.any(nit > 0) else 0,
        "avg_ms_per_iter": float(np.nanmean(per_ms)) if np.any(valid) else 0.0,
        "total_s": float(np.nansum(cpu)),
    }


def plot_minimum_snap_reference(
    t_ref: np.ndarray,
    p_ref: np.ndarray,
    waypoints: np.ndarray | None,
    t_wp: np.ndarray | None,
    yaw_ref: np.ndarray | None = None,
    out_path: Path | None = None,
    title: str = "Minimum-snap EE reference trajectory",
):
    """Visualize the EE reference before MPC/simulation (3D + XY/XZ + pose time-domain + velocity/acceleration + yaw)."""
    has_wp = waypoints is not None and t_wp is not None
    Wp4 = None
    if has_wp:
        Wraw = np.asarray(waypoints, dtype=float)
        if Wraw.shape[1] >= 4:
            Wp4 = Wraw
            W = Wraw[:, :3]
        else:
            W = Wraw.reshape(-1, 3)
        tw = np.asarray(t_wp, dtype=float).flatten()
    t_ref = np.asarray(t_ref, dtype=float).flatten()
    p_ref = np.asarray(p_ref, dtype=float)
    dp = np.gradient(p_ref, t_ref, axis=0, edge_order=2)
    v_norm = np.linalg.norm(dp, axis=1)
    ddp = np.gradient(dp, t_ref, axis=0, edge_order=2)
    a_norm = np.linalg.norm(ddp, axis=1)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=12, y=0.98)
    gs = fig.add_gridspec(4, 3, hspace=0.38, wspace=0.3, left=0.06, right=0.98, top=0.93, bottom=0.05)
    tinfo = {"fontsize": 9, "labelpad": 2}

    ax3 = fig.add_subplot(gs[0:2, 0:2], projection="3d")
    ax3.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], "b-", lw=1.5, label="p_ref(t)")
    if has_wp:
        ax3.scatter(W[:, 0], W[:, 1], W[:, 2], c="crimson", s=80, marker="o", label="waypoints", zorder=5)
        for k in range(len(tw)):
            ax3.text(W[k, 0], W[k, 1], W[k, 2], f"  t={tw[k]:.1f}s", fontsize=8, color="dimgray")
    ax3.set_xlabel("x [m]", **tinfo)
    ax3.set_ylabel("y [m]", **tinfo)
    ax3.set_zlabel("z [m]", **tinfo)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_title("3D reference path", fontsize=10)

    ax_xy = fig.add_subplot(gs[0, 2])
    ax_xy.plot(p_ref[:, 0], p_ref[:, 1], "b-", lw=1.5, label="p_ref (xy)")
    if has_wp:
        ax_xy.scatter(W[:, 0], W[:, 1], c="crimson", s=70, zorder=5, label="waypoints")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [m]", **tinfo)
    ax_xy.set_ylabel("y [m]", **tinfo)
    ax_xy.set_title("Horizontal (XY)", fontsize=10)
    ax_xy.legend(fontsize=7, loc="best")
    ax_xy.grid(True, alpha=0.3)

    ax_xz = fig.add_subplot(gs[1, 2])
    ax_xz.plot(p_ref[:, 0], p_ref[:, 2], "b-", lw=1.5, label="p_ref (xz)")
    if has_wp:
        ax_xz.scatter(W[:, 0], W[:, 2], c="crimson", s=70, zorder=5)
    ax_xz.set_xlabel("x [m]", **tinfo)
    ax_xz.set_ylabel("z [m]", **tinfo)
    ax_xz.set_title("Vertical profile (XZ)", fontsize=10)
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend(fontsize=7, loc="best")

    ax_t = fig.add_subplot(gs[2, 0])
    for j, name in enumerate("xyz"):
        ax_t.plot(t_ref, p_ref[:, j], label=f"p_{name}")
    if has_wp:
        for k in range(len(tw)):
            ax_t.axvline(tw[k], color="gray", ls=":", lw=0.8, alpha=0.7)
    ax_t.set_xlabel("t [s]", **tinfo)
    ax_t.set_ylabel("position [m]", **tinfo)
    ax_t.legend(loc="best", fontsize=7)
    ax_t.set_title("Position vs time", fontsize=10)
    ax_t.grid(True, alpha=0.3)

    ax_v = fig.add_subplot(gs[2, 1])
    ax_v.plot(t_ref, v_norm, "g-", lw=1.2, label=r"$\|\dot p\|$ (finite diff.)")
    ax_v.set_xlabel("t [s]", **tinfo)
    ax_v.set_ylabel("[m/s]", **tinfo)
    ax_v.set_title("Reference speed", fontsize=10)
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.3)

    ax_a = fig.add_subplot(gs[2, 2])
    ax_a.plot(t_ref, a_norm, "m-", lw=1.2, label=r"$\|\ddot p\|$ (finite diff.)")
    ax_a.set_xlabel("t [s]", **tinfo)
    ax_a.set_ylabel(r"[m/s²]", **tinfo)
    ax_a.set_title("Reference accel. magnitude", fontsize=10)
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)

    ax_yaw = fig.add_subplot(gs[3, :])
    if yaw_ref is not None:
        yr = np.asarray(yaw_ref, dtype=float).flatten()
        ax_yaw.plot(t_ref, np.degrees(yr), "c-", lw=1.2, label=r"$\psi_{\mathrm{ref}}$ (ZYX yaw, deg)")
    if Wp4 is not None and has_wp:
        ax_yaw.scatter(tw, np.degrees(Wp4[:, 3]), c="crimson", s=70, zorder=5, label="waypoint ψ")
    if has_wp:
        for k in range(len(tw)):
            ax_yaw.axvline(tw[k], color="gray", ls=":", lw=0.8, alpha=0.7)
    ax_yaw.set_xlabel("t [s]", **tinfo)
    ax_yaw.set_ylabel("deg", **tinfo)
    ax_yaw.set_title("EE yaw reference (planning)", fontsize=10)
    ax_yaw.legend(loc="best", fontsize=8)
    ax_yaw.grid(True, alpha=0.3)

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved planned trajectory figure: {out_path}")
    plt.show()


def _plot_tracking_dashboard(
    fig: plt.Figure,
    res: dict,
    simX: np.ndarray,
    plan_waypoints_xyz: np.ndarray | None = None,
):
    """EE tracking quality + base/reference/error + MPC solve statistics (complements GUI information)."""
    t = res["t"]
    base = simX[:, :3]
    ee = res["ee"]
    pref = res["p_ref"]
    err_vec = ee - pref
    ms = res.get("mpc_solve") or {}
    wall = np.asarray(ms.get("wall_s", []), dtype=float)
    nit = np.asarray(ms.get("nlp_iter", []), dtype=int)
    stat = np.asarray(ms.get("status", []), dtype=int)
    t_u = t[:-1]

    fig.clear()
    fig.suptitle("MPC EE tracking — overview (ref vs actual, errors, solver)", fontsize=12, y=0.98)
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.32, left=0.06, right=0.98, top=0.92, bottom=0.05)
    tinfo = {"fontsize": 9, "labelpad": 2}

    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.plot(base[:, 0], base[:, 1], base[:, 2], "b-", lw=1.5, label="Base")
    ax3.plot(ee[:, 0], ee[:, 1], ee[:, 2], "m-", lw=1.2, label="EE")
    ax3.plot(pref[:, 0], pref[:, 1], pref[:, 2], "k--", lw=1.0, alpha=0.7, label="p_ref")
    ax3.scatter(base[0, 0], base[0, 1], base[0, 2], c="g", s=60, label="Start")
    ax3.scatter(base[-1, 0], base[-1, 1], base[-1, 2], c="r", s=60, label="End")
    if plan_waypoints_xyz is not None:
        W = np.asarray(plan_waypoints_xyz, dtype=float)
        W = W[:, :3] if W.shape[1] > 3 else W.reshape(-1, 3)
        ax3.scatter(W[:, 0], W[:, 1], W[:, 2], c="orange", s=120, marker="*", label="plan WPs")
    ax3.set_xlabel("X [m]", **tinfo)
    ax3.set_ylabel("Y [m]", **tinfo)
    ax3.set_zlabel("Z [m]", **tinfo)
    ax3.set_title("3D: base / EE / p_ref", fontsize=10)
    ax3.legend(loc="upper left", fontsize=6, framealpha=0.9)
    _pts = np.vstack([base, ee, pref])
    br = float(np.ptp(_pts, axis=0).max())
    mid = _pts.mean(axis=0)
    r = max(br * 0.55, 0.25)
    ax3.set_xlim(mid[0] - r, mid[0] + r)
    ax3.set_ylim(mid[1] - r, mid[1] + r)
    ax3.set_zlim(mid[2] - r, mid[2] + r)
    try:
        ax3.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.plot(pref[:, 0], pref[:, 1], "k--", lw=1.2, alpha=0.8, label="p_ref (xy)")
    ax_xy.plot(ee[:, 0], ee[:, 1], "m-", lw=1.2, label="EE (xy)")
    ax_xy.plot(base[:, 0], base[:, 1], "b-", lw=1.0, alpha=0.6, label="Base (xy)")
    ax_xy.plot(base[0, 0], base[0, 1], "go", ms=5, label="Start")
    ax_xy.plot(base[-1, 0], base[-1, 1], "rs", ms=5, label="End")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("X [m]", **tinfo)
    ax_xy.set_ylabel("Y [m]", **tinfo)
    ax_xy.set_title("Horizontal (XY)", fontsize=10)
    ax_xy.legend(loc="best", fontsize=6, framealpha=0.9)
    ax_xy.grid(True, alpha=0.3)

    ax_xz = fig.add_subplot(gs[0, 2])
    ax_xz.plot(pref[:, 0], pref[:, 2], "k--", lw=1.2, alpha=0.8, label="p_ref (xz)")
    ax_xz.plot(ee[:, 0], ee[:, 2], "m-", lw=1.2, label="EE (xz)")
    ax_xz.plot(base[:, 0], base[:, 2], "b-", lw=1.0, alpha=0.6, label="Base (xz)")
    ax_xz.set_xlabel("X [m]", **tinfo)
    ax_xz.set_ylabel("Z [m]", **tinfo)
    ax_xz.set_title("Vertical profile (XZ)", fontsize=10)
    ax_xz.legend(loc="best", fontsize=6, framealpha=0.9)
    ax_xz.grid(True, alpha=0.3)

    ax_e = fig.add_subplot(gs[1, 0])
    ax_e.fill_between(t, 0.0, res["err"], color="red", alpha=0.2)
    ax_e.plot(t, res["err"], "r-", lw=1.2, label=r"$\|e\|$")
    ax_e.set_xlabel("t [s]", **tinfo)
    ax_e.set_ylabel("m", **tinfo)
    ax_e.set_title("EE position error norm", fontsize=10)
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.3)

    ax_ec = fig.add_subplot(gs[1, 1])
    for j, c in enumerate("rgb"):
        ax_ec.plot(t, err_vec[:, j], color=c, lw=1.0, label=f"e_{'xyz'[j]}")
    ax_ec.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax_ec.set_xlabel("t [s]", **tinfo)
    ax_ec.set_ylabel("m", **tinfo)
    ax_ec.set_title("EE error components (EE − p_ref)", fontsize=10)
    ax_ec.legend(loc="best", fontsize=7)
    ax_ec.grid(True, alpha=0.3)

    ax_pos = fig.add_subplot(gs[1, 2])
    cols = ("b", "g", "r")
    for j, name in enumerate("xyz"):
        ax_pos.plot(t, ee[:, j], color=cols[j], ls="-", lw=1.0, label=f"ee {name}")
    for j, name in enumerate("xyz"):
        ax_pos.plot(t, pref[:, j], color=cols[j], ls="--", lw=1.0, alpha=0.75, label=f"ref {name}")
    ax_pos.set_xlabel("t [s]", **tinfo)
    ax_pos.set_ylabel("position [m]", **tinfo)
    ax_pos.set_title("EE vs p_ref (xyz)", fontsize=10)
    ax_pos.legend(loc="best", fontsize=6, ncol=2)
    ax_pos.grid(True, alpha=0.3)

    ax_w = fig.add_subplot(gs[2, 0])
    if wall.size:
        ax_w.plot(t_u, wall * 1000.0, "C0-", lw=0.8, label="wall time / step")
        ax_w.set_xlabel("t [s]", **tinfo)
        ax_w.set_ylabel("ms", **tinfo)
        ax_w.set_title("MPC wall time per solve", fontsize=10)
        ax_w.legend(fontsize=8)
        ax_w.grid(True, alpha=0.3)
    else:
        ax_w.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_w.transAxes)

    ax_n = fig.add_subplot(gs[2, 1])
    if nit.size:
        ax_n.step(t_u, nit, where="post", color="C2", lw=0.9, label="nlp_iter")
        ax_n.set_xlabel("t [s]", **tinfo)
        ax_n.set_ylabel("count", **tinfo)
        ax_n.set_title("SQP iterations per MPC step", fontsize=10)
        ax_n.legend(fontsize=8)
        ax_n.grid(True, alpha=0.3)
    else:
        ax_n.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_n.transAxes)

    ax_st = fig.add_subplot(gs[2, 2])
    if stat.size:
        n_fail = int(np.sum(stat != 0))
        ax_st.bar([0, 1], [int(stat.size) - n_fail, n_fail], color=["seagreen", "salmon"], width=0.5)
        ax_st.set_xticks([0, 1])
        ax_st.set_xticklabels(["status 0", "≠0"])
        ax_st.set_ylabel("steps", **tinfo)
        ax_st.set_title(f"MPC exit status (fail {n_fail}/{len(stat)})", fontsize=10)
        ax_st.grid(True, axis="y", alpha=0.3)
    else:
        ax_st.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_st.transAxes)

    ax_y = fig.add_subplot(gs[3, :])
    if res.get("ee_yaw") is not None and res.get("yaw_ref") is not None:
        ax_y.plot(t, np.degrees(res["ee_yaw"]), "b-", lw=1.0, label="EE ψ (meas.)")
        ax_y.plot(t, np.degrees(res["yaw_ref"]), "k--", lw=1.0, alpha=0.8, label="ψ ref")
        ax_y2 = ax_y.twinx()
        ax_y2.plot(t, np.degrees(res["err_yaw"]), "r-", lw=0.9, alpha=0.85, label="ψ err (wrapped)")
        ax_y2.set_ylabel("deg", **tinfo, color="red")
        ax_y2.tick_params(axis="y", labelcolor="red")
        ax_y2.legend(loc="upper right", fontsize=7)
    ax_y.set_xlabel("t [s]", **tinfo)
    ax_y.set_ylabel("deg", **tinfo)
    ax_y.set_title("EE yaw tracking (ZYX, world)", fontsize=10)
    ax_y.legend(loc="upper left", fontsize=7)
    ax_y.grid(True, alpha=0.3)

    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=8)


def render_ee_tracking_results_to_figures(
    res: dict,
    fig_states,
    fig_3d,
    fig_dashboard,
    control_mode: str = "direct",
    plan_waypoints_xyz: np.ndarray | None = None,
    states_title: str | None = None,
) -> None:
    """
    Render closed-loop results into an existing ``matplotlib.figure.Figure`` (embedded in Qt, etc.); do not call ``plt.show()``.

    ``fig_states`` / ``fig_3d`` may be ``None`` (skip the corresponding subplots); ``fig_dashboard`` is required.
    """
    import matplotlib.figure

    t = res["t"]
    simX = _extract_simX_closed_loop(res)
    clayout = "high_level" if control_mode == "actuator_first_order" else "direct"
    ti = _mpc_timing_info_for_acados_plot(res)
    title = states_title or "MPC closed-loop (states & controls — same layout as trajectory GUI / acados)"

    if fig_states is not None and PLOT_ACADOS_GUI_STYLE and plot_acados_into_figure is not None:
        assert isinstance(fig_states, matplotlib.figure.Figure)
        plot_acados_into_figure(
            simX,
            res["u"],
            t,
            fig_states,
            title=title,
            waypoint_times=None,
            timing_info=ti,
            control_layout=clayout,
        )

    if fig_3d is not None and PLOT_ACADOS_GUI_STYLE and plot_acados_3d_into_figure is not None:
        wp_list = None
        if plan_waypoints_xyz is not None:
            W = np.asarray(plan_waypoints_xyz, dtype=float)
            W = W[:, :3] if W.shape[1] > 3 else W.reshape(-1, 3)
            wp_list = [W[i].copy() for i in range(len(W))]
        plot_acados_3d_into_figure(simX, fig_3d, waypoint_positions=wp_list)
        fig_3d.suptitle("Base 3D path (acados style) + plan waypoints", fontsize=11, y=0.98)

    assert isinstance(fig_dashboard, matplotlib.figure.Figure)
    _plot_tracking_dashboard(fig_dashboard, res, simX, plan_waypoints_xyz=plan_waypoints_xyz)


def plot_results(
    res: dict,
    out_path: Path | None = None,
    control_mode: str = "direct",
    plan_waypoints_xyz: np.ndarray | None = None,
):
    """
    Visualize the closed-loop results. If ``s500_uam_acados_trajectory.plot_acados_into_figure`` can be imported,
    first draw the 4x4 base/EE/arm/control panels consistent with trajectory_gui, then draw tracking and an MPC statistics overview.
    """
    fig_states = None
    fig_3d = None
    if PLOT_ACADOS_GUI_STYLE and plot_acados_into_figure is not None:
        fig_states = plt.figure(figsize=(18, 14))
    if PLOT_ACADOS_GUI_STYLE and plot_acados_3d_into_figure is not None:
        fig_3d = plt.figure(figsize=(10, 8))
    fig_dash = plt.figure(figsize=(15, 12))

    render_ee_tracking_results_to_figures(
        res,
        fig_states,
        fig_3d,
        fig_dash,
        control_mode=control_mode,
        plan_waypoints_xyz=plan_waypoints_xyz,
    )

    if out_path:
        if fig_states is not None:
            fig_states.savefig(out_path, dpi=150)
            print(f"Saved states/controls figure (primary): {out_path}")
        if fig_3d is not None:
            p3 = _suffix_figure_path(out_path, "_3d")
            fig_3d.savefig(p3, dpi=150)
            print(f"Saved 3D figure: {p3}")
        if fig_states is not None:
            p2 = _suffix_figure_path(out_path, "_tracking")
            fig_dash.savefig(p2, dpi=150)
            print(f"Saved tracking overview figure: {p2}")
        else:
            fig_dash.savefig(out_path, dpi=150)
            print(f"Saved figure: {out_path}")

    plt.show()


def run_ee_tracking_pipeline(
    track_canonical: str,
    *,
    waypoints_xyz_yaw: np.ndarray | None = None,
    times_wp: np.ndarray | None = None,
    eight_center: np.ndarray | None = None,
    eight_a: float = 0.22,
    eight_period: float = 6.0,
    T_sim: float = 8.0,
    sim_dt: float = 0.001,
    control_dt: float | None = None,
    dt_mpc: float = 0.05,
    N_mpc: int = 35,
    w_ee: float = 400.0,
    w_ee_yaw: float = 200.0,
    w_u_reg: float = 1e-4,
    max_iter: int = 20,
    mpc_log_interval: int = 1,
    control_mode_canonical: str = "direct",
    show_plan_figure: bool = False,
    plan_figure_path: Path | None = None,
    log_print: bool = True,
) -> dict:
    """
    Build the reference trajectory + Acados EE tracking MPC + closed-loop simulation. Shared by the CLI and ``s500_uam_ee_tracking_gui``.

    ``track_canonical``: ``"snap"`` or ``"eight"`` (internal name after resolving aliases).

    ``control_dt``: when ``control_dt`` is ``None`` and ``sim_dt < 0.01``, default to ``0.01`` (100 Hz); otherwise it equals ``sim_dt``.
    Explicitly passing it sets the ZOH period consistent with ``run_closed_loop``.

    Returns:
        dict containing ``res``, ``t_ref``, ``p_ref``, ``yaw_ref``, ``waypoints``, ``t_wp``,
        ``control_mode``, ``plan_title``, ``solver_meta``, etc.
    """
    if not ACADOS_AVAILABLE:
        raise RuntimeError("acados_template not installed")
    if not PINOCCHIO_AVAILABLE or not DEPS_OK:
        raise RuntimeError(
            f"Missing deps: pinocchio/casadi or project modules: "
            f"{_pin_err if not PINOCCHIO_AVAILABLE else _deps_err}"
        )

    track_canonical = TRACK_TRAJECTORY_ALIASES.get(track_canonical, track_canonical)
    if track_canonical not in ("snap", "eight"):
        raise ValueError(f"unknown track: {track_canonical}")

    dt_ref = min(0.02, float(sim_dt) * 0.5)
    waypoints: np.ndarray | None = None
    t_wp_out: np.ndarray | None = None
    plan_title = "EE reference"

    if track_canonical == "snap":
        if waypoints_xyz_yaw is None or times_wp is None:
            raise ValueError("track snap requires waypoints_xyz_yaw and times_wp")
        waypoints = np.asarray(waypoints_xyz_yaw, dtype=float)
        t_wp_out = np.asarray(times_wp, dtype=float).flatten()
        t_ref, p_ref, yaw_ref, _ = sample_ee_minimum_snap_trajectory(waypoints, t_wp_out, dt_ref)
        plan_title = "Minimum-snap EE reference (yaw interpolated from waypoints)"
    else:
        t_ref_dur = max(float(T_sim), float(eight_period) * 0.5)
        center = np.asarray(eight_center, dtype=float).reshape(3) if eight_center is not None else np.array(
            [0.55, 0.05, 0.92], dtype=float
        )
        t_ref, p_ref, yaw_ref, _ = sample_ee_figure_eight_trajectory(
            t_ref_dur,
            dt_ref,
            center=center,
            semi_axis=float(eight_a),
            period=float(eight_period),
        )
        plan_title = (
            f"Figure-eight EE reference (a={eight_a:.3f} m, T={eight_period:.2f} s, center={tuple(center)})"
        )

    if show_plan_figure:
        plot_minimum_snap_reference(
            t_ref,
            p_ref,
            waypoints,
            t_wp_out,
            yaw_ref=yaw_ref,
            out_path=plan_figure_path,
            title=plan_title,
        )

    x0 = make_uam_state(0.0, 0.0, 1.0, j1=0.0, j2=0.0, yaw=0.0)
    if log_print:
        print("Building EE-tracking MPC (first run may compile acados code)...")
    t0 = time.perf_counter()
    solver, acados_model, pin_model, nq, nv, nu, _ = create_ee_tracking_mpc_solver(
        N=N_mpc,
        dt=dt_mpc,
        w_ee=w_ee,
        w_ee_yaw=w_ee_yaw,
        w_u=w_u_reg,
        max_iter=max_iter,
        control_mode=control_mode_canonical,
    )
    if log_print:
        print(f"Solver ready in {time.perf_counter() - t0:.2f} s (control_mode={control_mode_canonical})")

    p0_ref = np.asarray(p_ref[0], dtype=float).reshape(3)
    x0 = align_uam_state_ee_to_world_position(x0, pin_model, p0_ref, nq, nv)

    if control_mode_canonical == "actuator_first_order":
        from s500_uam_acados_actuator_layer import pack_initial_state_with_actuators

        x0 = pack_initial_state_with_actuators(x0, pin_model=pin_model)

    sim_dt_f = float(sim_dt)
    if control_dt is None:
        ctl_dt = 0.01 if sim_dt_f < 0.01 - 1e-15 else sim_dt_f
    else:
        ctl_dt = float(control_dt)
    if log_print:
        print(
            f"Running closed-loop simulation (sim_dt={sim_dt_f:g} s, "
            f"control_dt={ctl_dt:g} s, MPC every {max(1, int(round(ctl_dt / sim_dt_f)))} sim steps)..."
        )
    res = run_closed_loop(
        x0=x0,
        t_grid_ref=t_ref,
        p_ref=p_ref,
        yaw_ref=yaw_ref,
        solver=solver,
        acados_model=acados_model,
        pin_model=pin_model,
        nq=nq,
        nv=nv,
        nu=nu,
        dt_mpc=dt_mpc,
        N=N_mpc,
        T_sim=T_sim,
        sim_dt=sim_dt_f,
        control_dt=ctl_dt,
        mpc_log_interval=mpc_log_interval,
        control_mode=control_mode_canonical,
    )
    if log_print:
        print(
            f"Final EE pos error norm: {res['err'][-1]:.4f} m (max {np.max(res['err']):.4f} m) | "
            f"yaw err: {res['err_yaw'][-1]:.4f} rad (max |err| {np.max(np.abs(res['err_yaw'])):.4f} rad)"
        )

    return {
        "res": res,
        "t_ref": t_ref,
        "p_ref": p_ref,
        "yaw_ref": yaw_ref,
        "waypoints": waypoints,
        "t_wp": t_wp_out,
        "control_mode": control_mode_canonical,
        "plan_title": plan_title,
        "track": track_canonical,
    }


def run_ee_tracking_from_reference_arrays(
    t_ref: np.ndarray,
    p_ref: np.ndarray,
    yaw_ref: np.ndarray,
    *,
    x0_init: np.ndarray | None = None,
    T_sim: float,
    sim_dt: float = 0.001,
    control_dt: float | None = None,
    dt_mpc: float = 0.05,
    N_mpc: int = 35,
    w_ee: float = 400.0,
    w_ee_yaw: float = 200.0,
    w_u_reg: float = 1e-4,
    max_iter: int = 20,
    mpc_log_interval: int = 1,
    control_mode_canonical: str = "direct",
    show_plan_figure: bool = False,
    plan_figure_path: Path | None = None,
    log_print: bool = True,
    plan_title: str = "EE reference",
    waypoints: np.ndarray | None = None,
    t_wp: np.ndarray | None = None,
    track_label: str = "custom_ref",
) -> dict:
    """
    Same closed-loop flow as ``run_ee_tracking_pipeline``, but the EE reference ``(t_ref, p_ref, yaw_ref)`` is provided by the caller
    (e.g., obtained from a full-state planned trajectory via FK). The time axis is shifted so that ``t_ref[0]=0``, consistent with ``run_closed_loop``.
    """
    if not ACADOS_AVAILABLE:
        raise RuntimeError("acados_template not installed")
    if not PINOCCHIO_AVAILABLE or not DEPS_OK:
        raise RuntimeError(
            f"Missing deps: pinocchio/casadi or project modules: "
            f"{_pin_err if not PINOCCHIO_AVAILABLE else _deps_err}"
        )

    t_ref = np.asarray(t_ref, dtype=float).flatten()
    p_ref = np.asarray(p_ref, dtype=float)
    if p_ref.ndim != 2 or p_ref.shape[1] != 3:
        raise ValueError("p_ref must have shape (N, 3)")
    yaw_ref = np.asarray(yaw_ref, dtype=float).flatten()
    if len(t_ref) != len(p_ref) or len(t_ref) != len(yaw_ref):
        raise ValueError("t_ref, p_ref, yaw_ref length mismatch")

    t0 = float(t_ref[0])
    t_ref = t_ref - t0

    if show_plan_figure:
        plot_minimum_snap_reference(
            t_ref,
            p_ref,
            waypoints,
            t_wp,
            yaw_ref=yaw_ref,
            out_path=plan_figure_path,
            title=plan_title,
        )

    if x0_init is None:
        x0 = make_uam_state(0.0, 0.0, 1.0, j1=0.0, j2=0.0, yaw=0.0)
    else:
        x0 = np.asarray(x0_init, dtype=float).flatten()
        if x0.size != 17:
            raise ValueError(f"x0_init must be 17-dim robot state, got shape {x0.shape}")
    if log_print:
        print("Building EE-tracking MPC (first run may compile acados code)...")
    t0b = time.perf_counter()
    solver, acados_model, pin_model, nq, nv, nu, _ = create_ee_tracking_mpc_solver(
        N=N_mpc,
        dt=dt_mpc,
        w_ee=w_ee,
        w_ee_yaw=w_ee_yaw,
        w_u=w_u_reg,
        max_iter=max_iter,
        control_mode=control_mode_canonical,
    )
    if log_print:
        print(f"Solver ready in {time.perf_counter() - t0b:.2f} s (control_mode={control_mode_canonical})")

    p0_ref = np.asarray(p_ref[0], dtype=float).reshape(3)
    x0 = align_uam_state_ee_to_world_position(x0, pin_model, p0_ref, nq, nv)

    if control_mode_canonical == "actuator_first_order":
        from s500_uam_acados_actuator_layer import pack_initial_state_with_actuators

        x0 = pack_initial_state_with_actuators(x0, pin_model=pin_model)

    sim_dt_f = float(sim_dt)
    if control_dt is None:
        ctl_dt = 0.01 if sim_dt_f < 0.01 - 1e-15 else sim_dt_f
    else:
        ctl_dt = float(control_dt)
    if log_print:
        print(
            f"Running closed-loop simulation (sim_dt={sim_dt_f:g} s, "
            f"control_dt={ctl_dt:g} s, MPC every {max(1, int(round(ctl_dt / sim_dt_f)))} sim steps)..."
        )
    res = run_closed_loop(
        x0=x0,
        t_grid_ref=t_ref,
        p_ref=p_ref,
        yaw_ref=yaw_ref,
        solver=solver,
        acados_model=acados_model,
        pin_model=pin_model,
        nq=nq,
        nv=nv,
        nu=nu,
        dt_mpc=dt_mpc,
        N=N_mpc,
        T_sim=T_sim,
        sim_dt=sim_dt_f,
        control_dt=ctl_dt,
        mpc_log_interval=mpc_log_interval,
        control_mode=control_mode_canonical,
    )
    if log_print:
        print(
            f"Final EE pos error norm: {res['err'][-1]:.4f} m (max {np.max(res['err']):.4f} m) | "
            f"yaw err: {res['err_yaw'][-1]:.4f} rad (max |err| {np.max(np.abs(res['err_yaw'])):.4f} rad)"
        )

    return {
        "res": res,
        "t_ref": t_ref,
        "p_ref": p_ref,
        "yaw_ref": yaw_ref,
        "waypoints": waypoints,
        "t_wp": t_wp,
        "control_mode": control_mode_canonical,
        "plan_title": plan_title,
        "track": track_label,
    }


def _default_control_mode_str() -> str:
    """If -c is not specified on the command line: use env var S500_UAM_CONTROL_MODE first; otherwise use direct."""
    v = os.environ.get("S500_UAM_CONTROL_MODE", "").strip().lower()
    if v and v in CONTROL_MODE_ALIASES:
        return v
    if v and v not in CONTROL_MODE_ALIASES:
        print(
            f"Warning: env var S500_UAM_CONTROL_MODE={v!r} is invalid; switched to direct."
            f" Allowed: {', '.join(sorted(CONTROL_MODE_ALIASES.keys()))}",
            file=sys.stderr,
        )
    return "direct"


def _default_track_trajectory_str() -> str:
    """If --track is not specified: read S500_UAM_EE_TRACK; otherwise use snap."""
    v = os.environ.get("S500_UAM_EE_TRACK", "").strip().lower()
    if v and v in TRACK_TRAJECTORY_ALIASES:
        return v
    if v and v not in TRACK_TRAJECTORY_ALIASES:
        print(
            f"Warning: env var S500_UAM_EE_TRACK={v!r} is invalid; switched to snap."
            f" Allowed: {', '.join(sorted(TRACK_TRAJECTORY_ALIASES.keys()))}",
            file=sys.stderr,
        )
    return "snap"


def main():
    parser = argparse.ArgumentParser(
        description="EE minimum-snap + Acados tracking MPC demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Control mode (-c / --control / --control-mode; uses S500_UAM_CONTROL_MODE when not set):\n"
            "  direct, thrusters          directly optimize quad thrust + joint torques [T1..T4, τ1, τ2]\n"
            "  actuator_first_order       same as above (full name)\n"
            "  actuator, high_level     body ω_cmd + total thrust + joint angle commands + first-order actuator\n"
            "\n"
            "EE reference trajectory (--track; uses S500_UAM_EE_TRACK when not set):\n"
            "  snap, minimum_snap         original demo: waypoints + minimum snap\n"
            "  eight, figure8             planar figure-eight (Gerono), see --eight_period / --eight_a / --eight_center\n"
        ),
    )
    parser.add_argument("--dt_mpc", type=float, default=0.1, help="MPC discretization [s]")
    parser.add_argument("--N_mpc", type=int, default=10, help="MPC horizon length")
    parser.add_argument("--sim_dt", type=float, default=0.001, help="Simulation integration step [s] (RK4)")
    parser.add_argument(
        "--control_dt",
        type=float,
        default=0.01,
        help="MPC recompute period [s] (ZOH; default 0.01=100 Hz; if it matches sim_dt, solve every step)",
    )
    parser.add_argument("--T_sim", type=float, default=8.0, help="Total simulation time [s]")
    parser.add_argument("--w_ee", type=float, default=400.0, help="LS weight on EE position")
    parser.add_argument(
        "--w_ee_yaw",
        type=float,
        default=200.0,
        help="LS weight on EE heading [cos ψ, sin ψ] (two terms; adjustable on the same scale as w_ee)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help=(
            "Save closed-loop result figures: main file is a states/controls 4x4 layout consistent with the GUI "
            "(if the acados trajectory module can be imported); also save stem_tracking.png (EE/error/MPC) "
            "and stem_3d.png (3D base + waypoints)"
        ),
    )
    parser.add_argument(
        "--save_plan",
        type=str,
        default="",
        help="Optional path to save minimum-snap planning figure (png); empty = only show window",
    )
    parser.add_argument(
        "--no_plan_fig",
        action="store_true",
        help="Skip plotting the planned minimum-snap trajectory before simulation",
    )
    parser.add_argument(
        "--mpc_log_interval",
        type=int,
        default=1,
        help="Print MPC solver statistics every k solves (not sim sub-steps); 0 = only print the summary",
    )
    parser.add_argument(
        "--mpc_max_iter",
        type=int,
        default=50,
        help="Maximum SQP iterations per MPC step (nlp_solver_max_iter)",
    )
    parser.add_argument(
        "-c",
        "--control",
        "--control-mode",
        dest="control_mode",
        type=str,
        choices=tuple(sorted(CONTROL_MODE_ALIASES.keys())),
        default=_default_control_mode_str(),
        metavar="MODE",
        help=(
            "MPC control-input form (default: env var S500_UAM_CONTROL_MODE, otherwise direct)."
            " direct/thrusters = rotor thrust + joint torques; actuator/high_level/actuator_first_order = high-level + first-order actuator"
        ),
    )
    parser.add_argument(
        "--track",
        dest="track",
        type=str,
        choices=tuple(sorted(TRACK_TRAJECTORY_ALIASES.keys())),
        default=_default_track_trajectory_str(),
        metavar="REF",
        help=(
            "Tracked EE reference trajectory (default: env var S500_UAM_EE_TRACK, otherwise snap)."
            " snap = waypoints minimum snap; eight/figure8 = figure-eight"
        ),
    )
    parser.add_argument(
        "--eight_a",
        type=float,
        default=0.22,
        metavar="M",
        help="Figure-eight half-width a [m]: approximate peak value along x is ±a (only for --track eight)",
    )
    parser.add_argument(
        "--eight_period",
        type=float,
        default=6.0,
        metavar="S",
        help="Figure-eight period per loop [s]; one closed figure-eight per period (only for --track eight)",
    )
    parser.add_argument(
        "--eight_center",
        type=float,
        nargs=3,
        default=[0.55, 0.05, 0.92],
        metavar=("CX", "CY", "CZ"),
        help="Figure-eight center (world frame) [m] (only for --track eight)",
    )
    args = parser.parse_args()
    args.control_mode = CONTROL_MODE_ALIASES[args.control_mode]
    args.track = TRACK_TRAJECTORY_ALIASES[args.track]

    print(
        f"[Startup] control mode: {args.control_mode} "
        f"({'direct: [T1..4, τ1, τ2]' if args.control_mode == 'direct' else 'high-level [ω, T_tot, θ] + first-order actuator'})"
    )
    print(f"[Startup] EE reference trajectory: {args.track}")

    if not ACADOS_AVAILABLE:
        print("acados_template not found. Install acados + pip install acados_template.", file=sys.stderr)
        sys.exit(1)
    if not PINOCCHIO_AVAILABLE or not DEPS_OK:
        print(f"Missing deps: pinocchio/casadi or project modules: {_pin_err if not PINOCCHIO_AVAILABLE else _deps_err}", file=sys.stderr)
        sys.exit(1)

    waypoints: np.ndarray | None = None
    t_wp: np.ndarray | None = None
    if args.track == "snap":
        t_wp = np.array([0.0, 2.5, 5.0, 8.0])
        waypoints = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.55, 0.35, 0.95, 0.25],
                [0.85, -0.15, 1.05, -0.4],
                [1.0, 0.2, 0.9, 0.15],
            ]
        )
    center = np.array(args.eight_center, dtype=float)

    out = run_ee_tracking_pipeline(
        args.track,
        waypoints_xyz_yaw=waypoints,
        times_wp=t_wp,
        eight_center=center,
        eight_a=float(args.eight_a),
        eight_period=float(args.eight_period),
        T_sim=float(args.T_sim),
        sim_dt=float(args.sim_dt),
        control_dt=float(args.control_dt),
        dt_mpc=float(args.dt_mpc),
        N_mpc=int(args.N_mpc),
        w_ee=float(args.w_ee),
        w_ee_yaw=float(args.w_ee_yaw),
        max_iter=int(args.mpc_max_iter),
        mpc_log_interval=int(args.mpc_log_interval),
        control_mode_canonical=args.control_mode,
        show_plan_figure=not args.no_plan_fig,
        plan_figure_path=Path(args.save_plan) if args.save_plan else None,
        log_print=True,
    )
    res = out["res"]
    save_out = Path(args.save) if args.save else None
    wp_plot = out["waypoints"] if args.track == "snap" and out["waypoints"] is not None else None
    plot_results(res, out_path=save_out, control_mode=args.control_mode, plan_waypoints_xyz=wp_plot)


if __name__ == "__main__":
    main()
