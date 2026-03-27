#!/usr/bin/env python3
"""
S500 UAM Trajectory Planning using acados

Uses acados for trajectory optimization with better support for state constraints.
Dynamics are obtained from URDF via Pinocchio + CasADi (no hand-written dynamics).

Features:
- Dynamics from URDF (Pinocchio buildModelFromUrdf)
- State constraints: joint limits, velocity bounds
- Path constraints: end-effector position (grasp point)
- Control constraints: thrust and torque limits

Usage:
    python s500_uam_acados_trajectory.py --simple
    python s500_uam_acados_trajectory.py --simple --duration 5 --dt 0.02

Requirements:
    - acados (pip install acados_template, build acados lib)
    - pinocchio, casadi, numpy, matplotlib, pyyaml
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

try:
    from acados_template import AcadosOcp, AcadosOcpSolver
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

try:
    from s500_uam_acados_model import build_acados_model
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    _model_err = e

try:
    from s500_uam_acados_cascade_actuator_model import (
        build_acados_model_cascade_actuator,
        pack_initial_state_cascade,
    )

    CASCADE_TRAJ_AVAILABLE = True
except ImportError:
    CASCADE_TRAJ_AVAILABLE = False
    build_acados_model_cascade_actuator = None
    pack_initial_state_cascade = None

from s500_uam_trajectory_planner import make_uam_state, compute_ee_kinematics_along_trajectory


def _quat_to_euler_zyx(quat):
    """Quaternion [qx,qy,qz,qw] -> (roll, pitch, yaw) in ZYX convention (CasADi)."""
    import casadi as ca
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    roll = ca.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = 2 * (qw * qy - qz * qx)
    sinp = ca.fmin(1, ca.fmax(-1, sinp))
    pitch = ca.asin(sinp)
    yaw = ca.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


def _state_to_cost_ref(state):
    """Convert 17-dim state to [pos(3), yaw(1), roll(1), pitch(1), jq(2), v(8)] for cost reference."""
    qx, qy, qz, qw = state[3], state[4], state[5], state[6]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return np.concatenate([
        state[0:3], [yaw], [roll], [pitch], state[7:9], state[9:17]
    ])


def load_s500_config():
    path = Path(__file__).parent.parent / 'config' / 'yaml' / 'multicopter' / 's500.yaml'
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# State limits (used in OCP and plot)
STATE_LIMITS = {
    "v_max": 1.0,           # m/s, max linear velocity
    "omega_max": 2.0,       # rad/s, max base angular velocity
    "j_angle_max": 2.0,     # rad, max joint angle (from URDF)
    "j_vel_max": 10.0,      # rad/s, max joint angular velocity (from URDF)
}

# Control input semantics (consistent with the 6 columns of simU; plotting and CLI switch using the same naming)
# direct: simU = [T1..T4, τ_arm1, τ_arm2]; cascade: [ωx, ωy, ωz, T_tot, θ1_cmd, θ2_cmd]
CONTROL_INPUT_DIRECT = "direct"
CONTROL_INPUT_CASCADE = "cascade"


def create_simple_ocp(
    start_state: np.ndarray,
    target_state: np.ndarray,
    duration: float = 5.0,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    is_waypoint: bool = False,
    max_iter: int = 200,
) -> "AcadosOcpSolver":
    """Create acados OCP for start -> target trajectory.
    N is computed from duration and dt: N = max(1, int(duration / dt)).
    state_weight, control_weight, waypoint_multiplier: from GUI, same semantics as Crocoddyl.
    is_waypoint: if True, scale terminal cost by waypoint_multiplier (for segment ends).
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("acados_template not installed. See https://docs.acados.org/installation/")
    if not MODEL_AVAILABLE:
        raise ImportError(f"Could not import s500_uam_acados_model: {_model_err}")

    import casadi as ca

    N = max(1, int(round(duration / dt)))

    ocp = AcadosOcp()
    acados_model, pin_model, nq, nv, nu = build_acados_model()
    ocp.model = acados_model

    nx = nq + nv

    ocp.dims.N = N
    ocp.solver_options.tf = duration
    ocp.solver_options.nlp_solver_max_iter = max_iter
    if hasattr(ocp.solver_options, 'N_horizon'):
        ocp.solver_options.N_horizon = N

    # Cost: track target with pos, yaw, roll, pitch, jq, v; yaw has separate weight
    # cost_y = [pos(3), yaw(1), roll(1), pitch(1), jq(2), v(8), u(6)]
    x = ocp.model.x
    quat = x[3:7]
    roll, pitch, yaw = _quat_to_euler_zyx(quat)
    cost_y = ca.vertcat(
        x[0:3], yaw, roll, pitch, x[7:9], x[9:17], ocp.model.u
    )
    cost_y_e = ca.vertcat(
        x[0:3], yaw, roll, pitch, x[7:9], x[9:17]
    )

    yref = np.concatenate([_state_to_cost_ref(target_state), np.zeros(nu)])
    yref_e = _state_to_cost_ref(target_state)

    # Base weight ratios (same relative importance); scaled by state_weight from GUI
    w_pos = 100.0 * state_weight
    w_yaw = 50.0 * state_weight
    w_rp = 10.0 * state_weight
    w_jq = 50.0 * state_weight
    w_v = 1.0 * state_weight
    w_omega = 1.0 * state_weight
    w_jdot = 10.0 * state_weight
    W_state = np.diag([
        w_pos, w_pos, w_pos, w_yaw, w_rp, w_rp,
        w_jq, w_jq,
        w_v, w_v, w_v, w_omega, w_omega, w_omega, w_jdot, w_jdot
    ])
    # Control: R = control_weight * [1,1,1,1, 100,100] to keep thrust:torque ratio
    r_thrust = control_weight
    r_torque = control_weight * 100.0
    R = np.diag([r_thrust] * 4 + [r_torque] * 2)
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = cost_y
    ocp.cost.yref = yref
    ocp.cost.W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))

    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = cost_y_e
    # Terminal weight: scale by waypoint_multiplier when segment end is a waypoint
    terminal_scale = waypoint_multiplier if is_waypoint else 1.0
    ocp.cost.W_e = W_state * terminal_scale

    # Control constraints
    cfg = load_s500_config()
    platform = cfg['platform']
    min_thrust = platform['min_thrust']
    max_thrust = platform['max_thrust']
    ocp.constraints.lbu = np.array([min_thrust] * 4 + [-2.0] * 2)
    ocp.constraints.ubu = np.array([max_thrust] * 4 + [2.0] * 2)
    ocp.constraints.idxbu = np.arange(nu)

    # State constraints: joint angles, velocities, angular velocities
    # x[7,8]=j1,j2; x[9:12]=vx,vy,vz; x[12:15]=wx,wy,wz; x[15:17]=j1_dot,j2_dot
    v_max = STATE_LIMITS["v_max"]
    om_max = STATE_LIMITS["omega_max"]
    j_max = STATE_LIMITS["j_angle_max"]
    jv_max = STATE_LIMITS["j_vel_max"]
    ocp.constraints.idxbx = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    ocp.constraints.lbx = np.array([
        -j_max, -j_max,
        -v_max, -v_max, -v_max,
        -om_max, -om_max, -om_max,
        -jv_max, -jv_max
    ])
    ocp.constraints.ubx = np.array([
        j_max, j_max,
        v_max, v_max, v_max,
        om_max, om_max, om_max,
        jv_max, jv_max
    ])

    # Initial state
    ocp.constraints.x0 = start_state

    # Solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.print_level = 0

    # Fixed code export path: reuse cached solver when OCP unchanged (same N, duration, etc.)
    script_dir = Path(__file__).parent
    code_export_dir = script_dir.parent / 'c_generated_code' / 's500_uam'
    json_path = code_export_dir / 's500_uam_ocp.json'
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(json_path)

    # generate=False, build=False + check_reuse_possible: if OCP hash matches cached json,
    # skip recompile; else regenerate. First run: no cache -> generates and builds.
    solver = AcadosOcpSolver(ocp, json_file=str(json_path), build=False, generate=False,
                             verbose=False, check_reuse_possible=True)
    return solver


def _x0_cascade(start_state: np.ndarray, pin_model) -> np.ndarray:
    """Convert a 17-dim robot initial state to 29-dim; if already 29-dim, return as-is (for multi-segment stitching)."""
    x = np.asarray(start_state, dtype=float).flatten()
    if x.size == 29:
        return x
    if x.size == 17:
        if pack_initial_state_cascade is None:
            raise RuntimeError("pack_initial_state_cascade not available")
        return pack_initial_state_cascade(x, pin_model)
    raise ValueError(f"cascade start_state must be 17 or 29, got {x.size}")


def _normalize_quaternion_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _interp_robot_state_17(alpha: float, x_start: np.ndarray, x_end: np.ndarray) -> np.ndarray:
    """Linear interpolation of pose/joints/velocity; quaternion via normalized nlerp. Used to keep the initial guess within box constraints (constant rollouts can overspeed)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    xs = np.asarray(x_start, dtype=float).reshape(17)
    xe = np.asarray(x_end, dtype=float).reshape(17)
    out = (1.0 - a) * xs + a * xe
    out[3:7] = _normalize_quaternion_np(out[3:7])
    return out


def _robot_state_17_from_waypoint(wp: np.ndarray) -> np.ndarray:
    w = np.asarray(wp, dtype=float).flatten()
    if w.size == 17:
        return w.copy()
    if w.size == 29:
        return w[:17].copy()
    raise ValueError(f"waypoint state must be 17 or 29, got {w.size}")


def _warm_start_cascade_trajectory_solver(
    solver,
    pin_model,
    start_state: np.ndarray,
    target_state: np.ndarray,
    N: int,
    min_thrust: float,
    max_thrust: float,
):
    """Interpolate 17-dim onboard states from start to target and pack them into 29-dim; u matches the filtered state z.

    Avoid the default all-zero initial guess; relative to a constant nominal dynamics rollout, propagate forward so that interpolation keeps velocities etc. near STATE_LIMITS, which helps the first-step QP.
    """
    if pack_initial_state_cascade is None:
        return
    x0 = _x0_cascade(start_state, pin_model)
    s17 = x0[:17]
    e17 = _robot_state_17_from_waypoint(target_state)
    z_sl = slice(17, 23)
    for i in range(N + 1):
        if i == 0:
            xi = x0
        else:
            s = float(i) / float(N)
            x17 = _interp_robot_state_17(s, s17, e17)
            xi = pack_initial_state_cascade(
                x17, pin_model, min_thrust=min_thrust, max_thrust=max_thrust
            )
        solver.set(i, "x", xi)
    for i in range(N):
        xgi = np.asarray(solver.get(i, "x"), dtype=float).flatten()
        solver.set(i, "u", xgi[z_sl])


def create_simple_ocp_cascade_actuator(
    start_state: np.ndarray,
    target_state: np.ndarray,
    duration: float = 5.0,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    is_waypoint: bool = False,
    max_iter: int = 200,
    tau_cmd: np.ndarray | None = None,
    tau_act: np.ndarray | None = None,
) -> "AcadosOcpSolver":
    """
    Same cost and constraint structure as create_simple_ocp; the dynamics are "first-order filtering per high-level ω, T, θ channel + first-order lag for the thrust/torque layer".
    State dimension is 29; control remains 6 (optimization variable is u_cmd).
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("acados_template not installed. See https://docs.acados.org/installation/")
    if not CASCADE_TRAJ_AVAILABLE:
        raise ImportError("Cascade actuator model not available (pinocchio/casadi/acados).")

    import casadi as ca

    N = max(1, int(round(duration / dt)))

    ocp = AcadosOcp()
    acados_model, pin_model, nq, nv, nu, _meta = build_acados_model_cascade_actuator(
        tau_cmd=tau_cmd, tau_act=tau_act
    )
    ocp.model = acados_model

    nx = int(acados_model.x.rows())
    ocp.dims.N = N
    ocp.solver_options.tf = duration
    ocp.solver_options.nlp_solver_max_iter = max_iter
    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = N

    x = ocp.model.x
    quat = x[3:7]
    roll, pitch, yaw = _quat_to_euler_zyx(quat)
    cost_y = ca.vertcat(
        x[0:3], yaw, roll, pitch, x[7:9], x[9:17], ocp.model.u
    )
    cost_y_e = ca.vertcat(
        x[0:3], yaw, roll, pitch, x[7:9], x[9:17]
    )

    yref = np.concatenate([_state_to_cost_ref(target_state), np.zeros(nu)])
    yref_e = _state_to_cost_ref(target_state)

    w_pos = 100.0 * state_weight
    w_yaw = 50.0 * state_weight
    w_rp = 10.0 * state_weight
    w_jq = 50.0 * state_weight
    w_v = 1.0 * state_weight
    w_omega = 1.0 * state_weight
    w_jdot = 10.0 * state_weight
    W_state = np.diag([
        w_pos, w_pos, w_pos, w_yaw, w_rp, w_rp,
        w_jq, w_jq,
        w_v, w_v, w_v, w_omega, w_omega, w_omega, w_jdot, w_jdot
    ])
    r_omega = control_weight
    r_T = control_weight
    r_theta = control_weight * 50.0
    R = np.diag([r_omega, r_omega, r_omega, r_T, r_theta, r_theta])
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = cost_y
    ocp.cost.yref = yref
    ocp.cost.W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))

    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = cost_y_e
    terminal_scale = waypoint_multiplier if is_waypoint else 1.0
    ocp.cost.W_e = W_state * terminal_scale

    cfg = load_s500_config()
    platform = cfg["platform"]
    min_thrust = platform["min_thrust"]
    max_thrust = platform["max_thrust"]
    v_max = STATE_LIMITS["v_max"]
    om_max = STATE_LIMITS["omega_max"]
    j_max = STATE_LIMITS["j_angle_max"]
    jv_max = STATE_LIMITS["j_vel_max"]

    ocp.constraints.lbu = np.array([-om_max, -om_max, -om_max, 4 * min_thrust, -j_max, -j_max])
    ocp.constraints.ubu = np.array([om_max, om_max, om_max, 4 * max_thrust, j_max, j_max])
    ocp.constraints.idxbu = np.arange(nu)

    idx_robot = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=int)
    idx_z = np.array([17, 18, 19, 20, 21, 22], dtype=int)
    idx_uact = np.array([23, 24, 25, 26, 27, 28], dtype=int)
    ocp.constraints.idxbx = np.concatenate([idx_robot, idx_z, idx_uact])
    ocp.constraints.lbx = np.concatenate([
        np.array([-j_max, -j_max, -v_max, -v_max, -v_max, -om_max, -om_max, -om_max, -jv_max, -jv_max]),
        np.array([-om_max, -om_max, -om_max, 4 * min_thrust, -j_max, -j_max]),
        np.array([min_thrust, min_thrust, min_thrust, min_thrust, -2.0, -2.0]),
    ])
    ocp.constraints.ubx = np.concatenate([
        np.array([j_max, j_max, v_max, v_max, v_max, om_max, om_max, om_max, jv_max, jv_max]),
        np.array([om_max, om_max, om_max, 4 * max_thrust, j_max, j_max]),
        np.array([max_thrust, max_thrust, max_thrust, max_thrust, 2.0, 2.0]),
    ])

    ocp.constraints.x0 = _x0_cascade(start_state, pin_model)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0
    if hasattr(ocp.solver_options, "qp_solver_iter_max"):
        ocp.solver_options.qp_solver_iter_max = max(80, max_iter * 2)
    if hasattr(ocp.solver_options, "nlp_solver_tol_stat"):
        ocp.solver_options.nlp_solver_tol_stat = 1e-2
    if hasattr(ocp.solver_options, "nlp_solver_tol_eq"):
        ocp.solver_options.nlp_solver_tol_eq = 1e-2
    dt_shoot = duration / N
    if hasattr(ocp.solver_options, "sim_method_num_steps"):
        # With large-step shooting, use enough sub-steps; otherwise the mismatch between the discretized dynamics and a rough initial guess can make the first-step QP ill-conditioned.
        ocp.solver_options.sim_method_num_steps = int(max(5, min(40, np.ceil(dt_shoot / 0.01))))

    script_dir = Path(__file__).parent
    code_export_dir = script_dir.parent / "c_generated_code" / "s500_uam_cascade_traj"
    json_path = code_export_dir / "s500_uam_cascade_traj_ocp.json"
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(json_path)

    solver = AcadosOcpSolver(
        ocp, json_file=str(json_path), build=False, generate=False,
        verbose=False, check_reuse_possible=True,
    )
    _warm_start_cascade_trajectory_solver(
        solver, pin_model, start_state, target_state, N, min_thrust, max_thrust
    )
    return solver


def _solve_ocp_with_live_log(solver, max_iter: int, label: str = "cascade"):
    """Run SQP step-by-step and print per-step residuals/step lengths in the terminal for debugging; on failure, fall back to a single solve + print_level.

    Returns:
        (status, n_sqp_steps): n_sqp_steps is the actual number of SQP steps executed (step-by-step mode); otherwise None.
    """
    use_step_loop = False
    try:
        solver.options_set("nlp_solver_max_iter", 1)
        solver.options_set("print_level", 0)
        try:
            solver.options_set("qp_print_level", 0)
        except Exception:
            pass
        use_step_loop = True
    except Exception:
        use_step_loop = False

    if not use_step_loop:
        try:
            solver.options_set("print_level", 1)
        except Exception:
            pass
        print(f"[{label}] single solve (step loop unavailable), using print_level if set.", flush=True)
        st = solver.solve()
        return st, None

    last = 2
    for it in range(max_iter):
        last = solver.solve()
        stats = None
        try:
            stats = solver.get_stats("statistics")
        except Exception:
            stats = None
        extra = ""
        if stats is not None:
            st = np.asarray(stats, dtype=float)
            if st.size > 0:
                row = st if st.ndim == 1 else st[-1]
                ncol = min(12, int(row.shape[0]))
                extra = " " + " ".join(f"{float(row[j]):.2e}" for j in range(ncol))
        print(f"[{label}] SQP {it + 1}/{max_iter}  status={last}{extra}", flush=True)
        if last == 0:
            return 0, it + 1
        if last not in (0, 2):
            print(f"[{label}] solver exit status {last}", flush=True)
            return last, it + 1
    print(f"[{label}] reached max_iter={max_iter} (last status={last})", flush=True)
    return last, max_iter


def run_simple_trajectory(
    start_state: np.ndarray = None,
    target_state: np.ndarray = None,
    duration: float = 5.0,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    max_iter: int = 200,
):
    """Run simple start->target trajectory optimization.
    N = duration / dt (auto-computed).
    state_weight, control_weight, waypoint_multiplier: from GUI.
    """
    if start_state is None:
        start_state = make_uam_state(0, 0, 1.0, j1=-1.2, j2=-0.6, yaw=0)
    if target_state is None:
        target_state = make_uam_state(1.0, 0.5, 1.2, j1=-0.8, j2=-0.3, yaw=np.pi / 4)

    solver = create_simple_ocp(
        start_state, target_state, duration, dt,
        state_weight=state_weight,
        control_weight=control_weight,
        waypoint_multiplier=waypoint_multiplier,
        is_waypoint=True,
        max_iter=max_iter,
    )
    t0 = time.perf_counter()
    status = solver.solve()
    t_wall = time.perf_counter() - t0

    # Get solver statistics (nlp_iter for SQP/DDP, time_tot for CPU time)
    t_cpu = solver.get_stats("time_tot")
    n_iter = solver.get_stats("nlp_iter")
    if n_iter is None:
        n_iter = solver.get_stats("sqp_iter")
    if n_iter is None:
        n_iter = solver.get_stats("ddp_iter")
    n_iter = int(n_iter) if n_iter is not None else -1
    t_per_iter = (t_cpu / n_iter * 1000) if n_iter > 0 else 0
    print(f"Optimization: {n_iter} iterations, {t_cpu:.4f}s CPU, {t_wall:.4f}s wall, {t_per_iter:.2f} ms/iter avg")

    if status != 0:
        print(f"acados solver returned status {status}")
        return None, None, None, None, None

    N = max(1, int(round(duration / dt)))
    nx = len(start_state)
    nu = 6
    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i, :] = solver.get(i, "x")
        simU[i, :] = solver.get(i, "u")
    simX[N, :] = solver.get(N, "x")

    dt_actual = duration / N
    time_arr = np.linspace(0, duration, N + 1)
    stats = {
        "n_iter": max(n_iter, 0),
        "total_s": float(t_wall),
        "avg_ms_per_iter": float(t_per_iter),
    }
    return simX, simU, time_arr, dt_actual, stats


def run_multiwaypoint_trajectory(
    waypoints: list,
    durations: list,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    max_iter: int = 200,
):
    """Run trajectory optimization through multiple waypoints (segment by segment).
    waypoints: list of 17-dim states
    durations: list of segment durations (len = len(waypoints) - 1)
    Returns: (simX, simU, time_arr, dt_actual) or (None, None, None, None) on failure.
    """
    if len(waypoints) != len(durations) + 1:
        raise ValueError("len(waypoints) must equal len(durations) + 1")
    all_X = []
    all_U = []
    all_stats = []
    x0 = waypoints[0]
    for i in range(len(durations)):
        x_target = waypoints[i + 1]
        result = run_simple_trajectory(
            x0, x_target, duration=durations[i], dt=dt,
            state_weight=state_weight,
            control_weight=control_weight,
            waypoint_multiplier=waypoint_multiplier,
            max_iter=max_iter,
        )
        if result[0] is None:
            return None, None, None, None, None
        simX, simU, time_arr, dt_actual, seg_stats = result
        if seg_stats:
            all_stats.append(seg_stats)
        if i == 0:
            all_X.append(simX)
            all_U.append(simU)
        else:
            all_X.append(simX[1:])
            all_U.append(simU)
        x0 = simX[-1]
    simX = np.vstack(all_X)
    simU = np.vstack(all_U)
    t_total = sum(durations)
    time_arr = np.linspace(0, t_total, len(simX))
    n_tot = sum(s["n_iter"] for s in all_stats) if all_stats else 0
    t_tot = sum(s.get("total_s", 0) for s in all_stats) if all_stats else 0
    merged_stats = {
        "n_iter": n_tot,
        "total_s": t_tot,
        "avg_ms_per_iter": (t_tot / n_tot * 1000) if n_tot > 0 else 0.0,
    }
    return simX, simU, time_arr, dt_actual, merged_stats


def run_simple_trajectory_cascade(
    start_state: np.ndarray = None,
    target_state: np.ndarray = None,
    duration: float = 5.0,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    max_iter: int = 200,
    tau_cmd: np.ndarray | None = None,
    tau_act: np.ndarray | None = None,
    debug_opt: bool = False,
    debug_label: str | None = None,
):
    """start/target are 17-dim robot states; returns simX as (N+1)x29 and simU as N×6 (high-level u_cmd).

    debug_opt: when True, print SQP residuals and other statistics step-by-step in the terminal (requires acados to support options_set for step-by-step iteration).
    debug_label: log prefix; for multi-waypoint runs, it can be set by the caller to be the segment index.
    """
    if start_state is None:
        start_state = make_uam_state(0, 0, 1.0, j1=-1.2, j2=-0.6, yaw=0)
    if target_state is None:
        target_state = make_uam_state(1.0, 0.5, 1.2, j1=-0.8, j2=-0.3, yaw=np.pi / 4)

    solver = create_simple_ocp_cascade_actuator(
        start_state, target_state, duration, dt,
        state_weight=state_weight,
        control_weight=control_weight,
        waypoint_multiplier=waypoint_multiplier,
        is_waypoint=True,
        max_iter=max_iter,
        tau_cmd=tau_cmd,
        tau_act=tau_act,
    )
    try:
        solver.options_set("levenberg_marquardt", 1e-2)
    except Exception:
        pass
    label = debug_label if debug_label else "cascade"
    t0 = time.perf_counter()
    n_sqp_logged = None
    if debug_opt:
        print(f"[{label}] start optimization  N={max(1, int(round(duration / dt)))}  max_iter={max_iter}", flush=True)
        status, n_sqp_logged = _solve_ocp_with_live_log(solver, max_iter, label=label)
    else:
        status = solver.solve()
    t_wall = time.perf_counter() - t0

    t_cpu = solver.get_stats("time_tot")
    if n_sqp_logged is not None:
        n_iter = int(n_sqp_logged)
    else:
        n_iter = solver.get_stats("nlp_iter")
        if n_iter is None:
            n_iter = solver.get_stats("sqp_iter")
        if n_iter is None:
            n_iter = solver.get_stats("ddp_iter")
        n_iter = int(n_iter) if n_iter is not None else -1
    t_per_iter = (t_cpu / n_iter * 1000) if n_iter > 0 else 0
    print(f"Optimization (cascade): {n_iter} iterations, {t_cpu:.4f}s CPU, {t_wall:.4f}s wall, {t_per_iter:.2f} ms/iter avg")

    if status not in [0, 2]:
        print(f"acados solver returned status {status}")
        return None, None, None, None, None

    N = max(1, int(round(duration / dt)))
    nx = int(solver.get(0, "x").shape[0])
    nu = int(solver.get(0, "u").shape[0])
    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i, :] = solver.get(i, "x")
        simU[i, :] = solver.get(i, "u")
    simX[N, :] = solver.get(N, "x")

    dt_actual = duration / N
    time_arr = np.linspace(0, duration, N + 1)
    stats = {
        "n_iter": max(n_iter, 0),
        "total_s": float(t_wall),
        "avg_ms_per_iter": float(t_per_iter),
    }
    return simX, simU, time_arr, dt_actual, stats


def run_multiwaypoint_trajectory_cascade(
    waypoints: list,
    durations: list,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    max_iter: int = 200,
    tau_cmd: np.ndarray | None = None,
    tau_act: np.ndarray | None = None,
    debug_opt: bool = False,
):
    """Multi-waypoint cascade optimization; segment-to-segment stitching uses the full 29-dim terminal state."""
    if len(waypoints) != len(durations) + 1:
        raise ValueError("len(waypoints) must equal len(durations) + 1")
    all_X = []
    all_U = []
    all_stats = []
    x0 = waypoints[0]
    for i in range(len(durations)):
        x_target = waypoints[i + 1]
        seg_label = f"cascade seg {i + 1}/{len(durations)}"
        result = run_simple_trajectory_cascade(
            x0, x_target, duration=durations[i], dt=dt,
            state_weight=state_weight,
            control_weight=control_weight,
            waypoint_multiplier=waypoint_multiplier,
            max_iter=max_iter,
            tau_cmd=tau_cmd,
            tau_act=tau_act,
            debug_opt=debug_opt,
            debug_label=seg_label,
        )
        if result[0] is None:
            return None, None, None, None, None
        simX, simU, time_arr, dt_actual, seg_stats = result
        if seg_stats:
            all_stats.append(seg_stats)
        if i == 0:
            all_X.append(simX)
            all_U.append(simU)
        else:
            all_X.append(simX[1:])
            all_U.append(simU)
        x0 = simX[-1]
    simX = np.vstack(all_X)
    simU = np.vstack(all_U)
    t_total = sum(durations)
    time_arr = np.linspace(0, t_total, len(simX))
    n_tot = sum(s["n_iter"] for s in all_stats) if all_stats else 0
    t_tot = sum(s.get("total_s", 0) for s in all_stats) if all_stats else 0
    merged_stats = {
        "n_iter": n_tot,
        "total_s": t_tot,
        "avg_ms_per_iter": (t_tot / n_tot * 1000) if n_tot > 0 else 0.0,
    }
    return simX, simU, time_arr, dt_actual, merged_stats


def _plot_pin_model_for_acados_fig():
    """Lazy-load Pinocchio model for EE kinematics in plots."""
    import pinocchio as pin
    urdf = Path(__file__).parent.parent / "models" / "urdf" / "s500_uam_simple.urdf"
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    data = model.createData()
    fid = model.getFrameId("gripper_link")
    return model, data, fid


def plot_acados_into_figure(simX, simU, time_arr, fig, title: str = "S500 UAM Trajectory (acados)", waypoint_times=None,
                            timing_info=None, control_layout: str = "direct"):
    """Plot acados trajectory into existing figure (4x4 layout, aligned with Crocoddyl).

    control_layout:
      - ``direct``: simU = [T1..T4, τ1, τ2] (default)
      - ``high_level``: simU = [ωx, ωy, ωz, T_tot, θ1, θ2] (cascade / high-level command)
    """
    if simX is None or fig is None:
        return None
    import matplotlib.pyplot as plt
    dt = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 0.02
    time_states = time_arr
    time_controls = np.linspace(0, time_arr[-1] - dt, len(simU)) if len(simU) == len(time_arr) - 1 else time_arr[:-1]
    if len(time_controls) != len(simU):
        time_controls = np.linspace(time_arr[0], time_arr[-1] - dt, len(simU))
    waypoint_indices = [int(t / dt) for t in (waypoint_times or [])] if waypoint_times else []

    def _quat_to_euler_row(quat):
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return roll, pitch, yaw
    euler = np.array([_quat_to_euler_row(simX[i, 3:7]) for i in range(len(simX))])

    ee_pos = ee_v = ee_rpy = ee_w = None
    try:
        pm, pdata, pfid = _plot_pin_model_for_acados_fig()
        ee_pos, ee_v, ee_rpy, ee_w = compute_ee_kinematics_along_trajectory(simX, pm, pdata, pfid)
    except Exception:
        ee_pos = np.zeros((len(simX), 3))
        ee_v = ee_rpy = ee_w = ee_pos

    def add_wp_lines(ax):
        for idx in waypoint_indices:
            if idx < len(time_states):
                ax.axvline(x=time_states[idx], color='orange', linestyle='--', alpha=0.5)

    positions = simX[:, :3]
    fig.clear()
    gs = fig.add_gridspec(4, 4, hspace=0.42, wspace=0.32, left=0.05, right=0.98, top=0.93, bottom=0.05)
    tinfo = {'fontsize': 9, 'labelpad': 2}

    # Row 0: Base
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(time_states, simX[:, 0], 'r-', label='x')
    ax00.plot(time_states, simX[:, 1], 'g-', label='y')
    ax00.plot(time_states, simX[:, 2], 'b-', label='z')
    add_wp_lines(ax00)
    ax00.set_xlabel('Time (s)', **tinfo)
    ax00.set_ylabel('Position (m)', **tinfo)
    ax00.set_title('Base Position', fontsize=9)
    ax00.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(time_states, simX[:, 9], 'r-', label='vx')
    ax01.plot(time_states, simX[:, 10], 'g-', label='vy')
    ax01.plot(time_states, simX[:, 11], 'b-', label='vz')
    add_wp_lines(ax01)
    ax01.set_xlabel('Time (s)', **tinfo)
    ax01.set_ylabel('Velocity (m/s)', **tinfo)
    ax01.set_title('Base Linear Velocity', fontsize=9)
    ax01.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax02 = fig.add_subplot(gs[0, 2])
    ax02.plot(time_states, np.degrees(euler[:, 0]), 'r-', label='roll')
    ax02.plot(time_states, np.degrees(euler[:, 1]), 'g-', label='pitch')
    ax02.plot(time_states, np.degrees(euler[:, 2]), 'b-', label='yaw')
    add_wp_lines(ax02)
    ax02.set_xlabel('Time (s)', **tinfo)
    ax02.set_ylabel('Angle (°)', **tinfo)
    ax02.set_title('Base Orientation (Euler)', fontsize=9)
    ax02.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax03 = fig.add_subplot(gs[0, 3])
    ax03.plot(time_states, simX[:, 12], 'r-', label='ωx')
    ax03.plot(time_states, simX[:, 13], 'g-', label='ωy')
    ax03.plot(time_states, simX[:, 14], 'b-', label='ωz')
    add_wp_lines(ax03)
    ax03.set_xlabel('Time (s)', **tinfo)
    ax03.set_ylabel('Angular vel (rad/s)', **tinfo)
    ax03.set_title('Base Angular Velocity', fontsize=9)
    ax03.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # Row 1: EE
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(time_states, ee_pos[:, 0], 'r-', label='x')
    ax10.plot(time_states, ee_pos[:, 1], 'g-', label='y')
    ax10.plot(time_states, ee_pos[:, 2], 'b-', label='z')
    add_wp_lines(ax10)
    ax10.set_xlabel('Time (s)', **tinfo)
    ax10.set_ylabel('Position (m)', **tinfo)
    ax10.set_title('EE Position', fontsize=9)
    ax10.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(time_states, ee_v[:, 0], 'r-', label='vx')
    ax11.plot(time_states, ee_v[:, 1], 'g-', label='vy')
    ax11.plot(time_states, ee_v[:, 2], 'b-', label='vz')
    add_wp_lines(ax11)
    ax11.set_xlabel('Time (s)', **tinfo)
    ax11.set_ylabel('Velocity (m/s)', **tinfo)
    ax11.set_title('EE Linear Velocity', fontsize=9)
    ax11.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax12 = fig.add_subplot(gs[1, 2])
    ax12.plot(time_states, np.degrees(ee_rpy[:, 0]), 'r-', label='roll')
    ax12.plot(time_states, np.degrees(ee_rpy[:, 1]), 'g-', label='pitch')
    ax12.plot(time_states, np.degrees(ee_rpy[:, 2]), 'b-', label='yaw')
    add_wp_lines(ax12)
    ax12.set_xlabel('Time (s)', **tinfo)
    ax12.set_ylabel('Angle (°)', **tinfo)
    ax12.set_title('EE Orientation (RPY)', fontsize=9)
    ax12.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax13 = fig.add_subplot(gs[1, 3])
    ax13.plot(time_states, ee_w[:, 0], 'r-', label='ωx')
    ax13.plot(time_states, ee_w[:, 1], 'g-', label='ωy')
    ax13.plot(time_states, ee_w[:, 2], 'b-', label='ωz')
    add_wp_lines(ax13)
    ax13.set_xlabel('Time (s)', **tinfo)
    ax13.set_ylabel('Angular vel (rad/s)', **tinfo)
    ax13.set_title('EE Angular Velocity', fontsize=9)
    ax13.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # Row 2: Arm & controls
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(time_states, np.degrees(simX[:, 7]), 'r-', label='j1')
    ax20.plot(time_states, np.degrees(simX[:, 8]), 'g-', label='j2')
    add_wp_lines(ax20)
    ax20.set_xlabel('Time (s)', **tinfo)
    ax20.set_ylabel('Angle (°)', **tinfo)
    ax20.set_title('Arm Joint Angles', fontsize=9)
    ax20.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax21 = fig.add_subplot(gs[2, 1])
    ax21.plot(time_states, simX[:, 15], 'r-', label='j1_dot')
    ax21.plot(time_states, simX[:, 16], 'g-', label='j2_dot')
    add_wp_lines(ax21)
    ax21.set_xlabel('Time (s)', **tinfo)
    ax21.set_ylabel('Angular vel (rad/s)', **tinfo)
    ax21.set_title('Arm Joint Angular Velocity', fontsize=9)
    ax21.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax22 = fig.add_subplot(gs[2, 2])
    ax23 = fig.add_subplot(gs[2, 3])
    if control_layout == "high_level" and simU.shape[1] >= 6:
        ax22.plot(time_controls, simU[:, 0], 'r-', label='ωx')
        ax22.plot(time_controls, simU[:, 1], 'g-', label='ωy')
        ax22.plot(time_controls, simU[:, 2], 'b-', label='ωz')
        ax22.set_ylabel('Ang. rate cmd (rad/s)', **tinfo)
        ax22.set_title('High-level ω cmd', fontsize=9)
        ax22.legend(loc='upper right', fontsize=7, framealpha=0.9)
        ax22.set_xlabel('Time (s)', **tinfo)
        ax23.plot(time_controls, simU[:, 3], 'k-', label='T_tot')
        ax23.plot(time_controls, simU[:, 4], 'r--', label='θ1 cmd')
        ax23.plot(time_controls, simU[:, 5], 'g--', label='θ2 cmd')
        ax23.set_xlabel('Time (s)', **tinfo)
        ax23.set_ylabel('T (N) / θ (rad)', **tinfo)
        ax23.set_title('High-level T & θ cmd', fontsize=9)
        ax23.legend(loc='upper right', fontsize=7, framealpha=0.9)
    else:
        colors = ['r', 'g', 'b', 'orange']
        for i in range(min(4, simU.shape[1])):
            ax22.plot(time_controls, simU[:, i], color=colors[i], label=f'T{i+1}')
        ax22.set_xlabel('Time (s)', **tinfo)
        ax22.set_ylabel('Thrust (N)', **tinfo)
        ax22.set_title('Base Control (Thrusters)', fontsize=9)
        ax22.legend(loc='upper right', fontsize=7, framealpha=0.9)
        if simU.shape[1] >= 6:
            ax23.plot(time_controls, simU[:, 4], 'r-', label='τ1')
            ax23.plot(time_controls, simU[:, 5], 'g-', label='τ2')
        ax23.set_xlabel('Time (s)', **tinfo)
        ax23.set_ylabel('Torque (N·m)', **tinfo)
        ax23.set_title('Arm Control (Joint Torques)', fontsize=9)
        ax23.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # Row 3
    ax30 = fig.add_subplot(gs[3, 0])
    ax30.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, label='Base')
    ax30.plot(ee_pos[:, 0], ee_pos[:, 1], 'm--', linewidth=1.2, label='EE')
    ax30.plot(positions[0, 0], positions[0, 1], 'go', markersize=6, label='Start')
    ax30.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=6, label='End')
    ax30.set_xlabel('X (m)', **tinfo)
    ax30.set_ylabel('Y (m)', **tinfo)
    ax30.set_title('Horizontal trajectory (XY)', fontsize=9)
    ax30.axis('equal')
    ax30.grid(True, alpha=0.3)
    ax30.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax31 = fig.add_subplot(gs[3, 1])
    ax31.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=1.5, label='Base')
    ax31.plot(ee_pos[:, 0], ee_pos[:, 2], 'm--', linewidth=1.2, label='EE')
    ax31.plot(positions[0, 0], positions[0, 2], 'go', markersize=6, label='Start')
    ax31.plot(positions[-1, 0], positions[-1, 2], 'rs', markersize=6, label='End')
    ax31.set_xlabel('X (m)', **tinfo)
    ax31.set_ylabel('Z (m)', **tinfo)
    ax31.set_title('Vertical profile (XZ)', fontsize=9)
    ax31.grid(True, alpha=0.3)
    ax31.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax32 = fig.add_subplot(gs[3, 2])
    ax32.text(0.5, 0.5, 'Acados: cost in solver log', ha='center', va='center', transform=ax32.transAxes)
    ax32.set_xlabel('Iteration', **tinfo)
    ax32.set_ylabel('Cost', **tinfo)
    ax32.set_title('Cost convergence', fontsize=9)
    ax32.grid(True, alpha=0.3)

    ax33 = fig.add_subplot(gs[3, 3])
    ti = timing_info or {}
    if ti.get('n_iter', 0) and ti.get('n_iter', 0) > 0:
        n_it = int(ti['n_iter'])
        avg_ms = float(ti.get('avg_ms_per_iter', 0))
        iters = np.arange(1, n_it + 1)
        ax33.plot(iters, np.full(n_it, avg_ms), 'g-', linewidth=2, label=f'Avg {avg_ms:.2f} ms/iter')
        ax33.fill_between(iters, 0, np.full(n_it, avg_ms), alpha=0.15, color='g')
        ax33.set_xlabel('Iteration', **tinfo)
        ax33.set_ylabel('Time per iter (ms)', **tinfo)
        ax33.set_title('Solver time / iteration', fontsize=9)
        ax33.legend(loc='upper right', fontsize=7)
        ax33.grid(True, alpha=0.3)
    else:
        ax33.text(0.5, 0.5, 'Timing N/A', ha='center', va='center', transform=ax33.transAxes)
        ax33.set_title('Solver time / iteration', fontsize=9)

    fig.suptitle(title, fontsize=12, y=0.98)
    all_axes = fig.get_axes()
    for ax in all_axes:
        ax.tick_params(axis='both', labelsize=8)
    return fig


def plot_acados_3d_into_figure(simX, fig, waypoint_positions=None):
    """Plot acados 3D trajectory into existing figure."""
    if simX is None or fig is None:
        return None
    positions = simX[:, :3]
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Base')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='r', s=100, label='End')
    if waypoint_positions:
        for wp in waypoint_positions:
            ax.scatter(wp[0], wp[1], wp[2], color='orange', s=150, marker='*')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory (acados)')
    ax.legend(loc='upper right', fontsize=8)
    all_pts = positions.copy()
    if waypoint_positions:
        all_pts = np.vstack([all_pts, np.array(waypoint_positions)])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5
    if max_range < 0.1:
        max_range = 0.5
    x_mid, y_mid, z_mid = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    ax.set_box_aspect([1, 1, 1])
    return fig


def plot_results(
    simX,
    simU,
    time_arr,
    save_path: str = None,
    control_input: str = CONTROL_INPUT_DIRECT,
):
    """Plot trajectory results with state limits shown.

    control_input:
      - ``CONTROL_INPUT_DIRECT`` / ``"direct"``: simU is quadrotor thrust + arm joint torques.
      - ``CONTROL_INPUT_CASCADE`` / ``"cascade"``: simU is body angular rates, total thrust, and joint angle commands (rad).
    """
    if simX is None:
        return
    lim = STATE_LIMITS
    use_high_level = control_input == CONTROL_INPUT_CASCADE
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    _ctrl_tag = "cascade cmd" if use_high_level else "direct thrust/torque"
    fig.suptitle(f"S500 UAM Trajectory (acados) — {_ctrl_tag}")

    # Row 0: position, orientation (euler), joint angles
    ax = axes[0, 0]
    ax.plot(time_arr, simX[:, 0], 'r-', label='x')
    ax.plot(time_arr, simX[:, 1], 'g-', label='y')
    ax.plot(time_arr, simX[:, 2], 'b-', label='z')
    ax.set_ylabel('Position (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    quat = simX[:, 3:7]
    euler = np.array([_quat_to_euler(quat[i:i+1])[0] for i in range(len(quat))])
    ax.plot(time_arr, np.degrees(euler[:, 0]), 'r-', label='roll')
    ax.plot(time_arr, np.degrees(euler[:, 1]), 'g-', label='pitch')
    ax.plot(time_arr, np.degrees(euler[:, 2]), 'b-', label='yaw')
    ax.set_ylabel('Euler (°)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(time_arr, np.degrees(simX[:, 7]), 'r-', label='j1')
    ax.plot(time_arr, np.degrees(simX[:, 8]), 'g-', label='j2')
    j_deg = np.degrees(lim["j_angle_max"])
    ax.axhline(j_deg, color='gray', linestyle='--', alpha=0.7, label=f'±{j_deg:.0f}°')
    ax.axhline(-j_deg, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Joint (°)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: velocities
    ax = axes[1, 0]
    ax.plot(time_arr, simX[:, 9], 'r-', label='vx')
    ax.plot(time_arr, simX[:, 10], 'g-', label='vy')
    ax.plot(time_arr, simX[:, 11], 'b-', label='vz')
    ax.axhline(lim["v_max"], color='gray', linestyle='--', alpha=0.7, label=f'±{lim["v_max"]} m/s')
    ax.axhline(-lim["v_max"], color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Velocity (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(time_arr, simX[:, 12], 'r-', label='ωx')
    ax.plot(time_arr, simX[:, 13], 'g-', label='ωy')
    ax.plot(time_arr, simX[:, 14], 'b-', label='ωz')
    ax.axhline(lim["omega_max"], color='gray', linestyle='--', alpha=0.7, label=f'±{lim["omega_max"]} rad/s')
    ax.axhline(-lim["omega_max"], color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Angular vel (rad/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(time_arr, simX[:, 15], 'r-', label='j1_dot')
    ax.plot(time_arr, simX[:, 16], 'g-', label='j2_dot')
    ax.axhline(lim["j_vel_max"], color='gray', linestyle='--', alpha=0.7, label=f'±{lim["j_vel_max"]} rad/s')
    ax.axhline(-lim["j_vel_max"], color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Joint vel (rad/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: controls (branch by control-input semantics)
    t_u = time_arr[:-1]
    ax = axes[2, 0]
    if use_high_level and simU.shape[1] >= 6:
        ax.plot(t_u, simU[:, 0], 'r-', label='ωx cmd')
        ax.plot(t_u, simU[:, 1], 'g-', label='ωy cmd')
        ax.plot(t_u, simU[:, 2], 'b-', label='ωz cmd')
        ax.axhline(lim["omega_max"], color='gray', linestyle='--', alpha=0.7, label=f'±{lim["omega_max"]} rad/s')
        ax.axhline(-lim["omega_max"], color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel('Body rate cmd (rad/s)')
    else:
        for i in range(min(4, simU.shape[1])):
            ax.plot(t_u, simU[:, i], label=f'T{i+1}')
        ax.set_ylabel('Thrust (N)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if use_high_level and simU.shape[1] >= 6:
        ax.plot(t_u, simU[:, 3], 'k-', label='T_tot cmd')
        ax.plot(t_u, np.degrees(simU[:, 4]), 'r--', label='θ1 cmd')
        ax.plot(t_u, np.degrees(simU[:, 5]), 'g--', label='θ2 cmd')
        j_deg = np.degrees(lim["j_angle_max"])
        ax.axhline(j_deg, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(-j_deg, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel('T_tot (N) / joint cmd (°)')
    else:
        if simU.shape[1] >= 6:
            ax.plot(t_u, simU[:, 4], 'r-', label='τ1')
            ax.plot(t_u, simU[:, 5], 'g-', label='τ2')
        ax.set_ylabel('Torque (N·m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.plot(simX[:, 0], simX[:, 1], 'b-')
    ax.plot(simX[0, 0], simX[0, 1], 'go', markersize=10)
    ax.plot(simX[-1, 0], simX[-1, 1], 'rs', markersize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY trajectory')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    plt.show()


def _quat_to_euler(quat):
    """Quat [qx,qy,qz,qw] to euler (roll,pitch,yaw) rad."""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.column_stack([roll, pitch, yaw])


def main():
    parser = argparse.ArgumentParser(description='S500 UAM trajectory planning with acados')
    parser.add_argument('--simple', action='store_true', help='Simple mode: start -> target')
    parser.add_argument('--duration', type=float, default=5.0, help='Trajectory duration (s)')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step (s), N = duration/dt')
    parser.add_argument('--save', type=str, help='Save plot path')
    parser.add_argument(
        '--control',
        type=str,
        choices=(CONTROL_INPUT_DIRECT, CONTROL_INPUT_CASCADE),
        default=CONTROL_INPUT_CASCADE,
        help=(
            f'Control parameterization: "{CONTROL_INPUT_DIRECT}" = [T1..T4,τ1,τ2] (default); '
            f'"{CONTROL_INPUT_CASCADE}" = [ωx,ωy,ωz,T_tot,θ1,θ2] (needs cascade model).'
        ),
    )
    parser.add_argument(
        '--debug-opt',
        action='store_true',
        help='Print per-iteration SQP stats while solving cascade trajectory (for debugging)',
    )
    args = parser.parse_args()

    if not ACADOS_AVAILABLE:
        print("ERROR: acados not installed.")
        print("  pip install acados_template")
        print("  Build acados: https://docs.acados.org/installation/")
        return 1
    if not MODEL_AVAILABLE:
        print(f"ERROR: Could not load model: {_model_err}")
        return 1

    print("S500 UAM acados trajectory optimization")
    print("=" * 50)

    start = make_uam_state(0, 0, 0.0, j1=-1.2, j2=-0.6, yaw=0)
    target = make_uam_state(1.0, 0, 0.5, j1=-0.8, j2=-0.3, yaw=np.deg2rad(0))
    N = max(1, int(round(args.duration / args.dt)))
    print(f"Start:  pos={start[:3]}, arm=[{np.degrees(start[7]):.0f}, {np.degrees(start[8]):.0f}]°")
    print(f"Target: pos={target[:3]}, arm=[{np.degrees(target[7]):.0f}, {np.degrees(target[8]):.0f}]°")
    print(f"Duration: {args.duration}s, dt: {args.dt}s, N: {N}")
    print(f"Control input mode: {args.control}")
    print()

    if args.control == CONTROL_INPUT_CASCADE:
        if not CASCADE_TRAJ_AVAILABLE:
            print("ERROR: cascade control requires pinocchio/casadi and s500_uam_acados_cascade_actuator_model.")
            return 1
        simX, simU, time_arr, dt, _stats = run_simple_trajectory_cascade(
            start,
            target,
            args.duration,
            args.dt,
            debug_opt=args.debug_opt,
        )
        plot_ctrl = CONTROL_INPUT_CASCADE
    else:
        simX, simU, time_arr, dt, _stats = run_simple_trajectory(
            start, target, args.duration, args.dt
        )
        plot_ctrl = CONTROL_INPUT_DIRECT

    if simX is not None:
        print("Optimization converged.")
        plot_results(simX, simU, time_arr, args.save, control_input=plot_ctrl)
        if args.save:
            np.savez(
                args.save.replace('.png', '.npz') if args.save.endswith('.png') else args.save + '.npz',
                states=simX,
                controls=simU,
                time=time_arr,
                control_input=np.array(args.control),
            )
    else:
        print("Optimization failed.")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
