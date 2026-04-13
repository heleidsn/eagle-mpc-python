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

from s500_uam_trajectory_planner import (
    make_uam_state,
    compute_ee_kinematics_along_trajectory,
    base_lin_ang_world_from_robot_state,
)


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
    # Control: R = control_weight * [1,1,1,1, 10000,10000] (arm τ weights 100× previous 100× thrust)
    r_thrust = control_weight
    r_torque = control_weight * 10000.0
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
    r_theta = control_weight * 5000.0
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


def plot_acados_into_figure(
    simX,
    simU,
    time_arr,
    fig,
    title: str = "S500 UAM Trajectory (acados)",
    waypoint_times=None,
    timing_info=None,
    control_layout: str = "direct",
    waypoint_positions_base=None,
    waypoint_positions_ee=None,
    traj_solver_meta=None,
    ref_time_states=None,
    ref_states=None,
    ref_time_controls=None,
    ref_controls=None,
):
    """Plot acados trajectory into existing figure (4x4 layout, aligned with Crocoddyl).

    control_layout:
      - ``direct``: simU = [T1..T4, τ1, τ2] (default)
      - ``high_level``: simU = [ωx, ωy, ωz, T_tot, θ1, θ2] (cascade / high-level command)

    traj_solver_meta:
      When set, overrides the bottom-row (3,2)/(3,3) panels that default to closed-loop Acados MPC text.
      Expected keys: ``backend`` (``"crocoddyl"`` | ``"acados_traj"``), optional ``costs`` (iter costs, croc),
      ``timing`` dict with ``n_iter``, ``avg_ms_per_iter``, ``total_s`` (same as MPC timing_info).
    """
    if simX is None or fig is None:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dt = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 0.02
    time_states = time_arr
    time_controls = np.linspace(0, time_arr[-1] - dt, len(simU)) if len(simU) == len(time_arr) - 1 else time_arr[:-1]
    if len(time_controls) != len(simU):
        time_controls = np.linspace(time_arr[0], time_arr[-1] - dt, len(simU))

    def _interp_series_to_time(src_t, src_y, tgt_t):
        if src_t is None or src_y is None or tgt_t is None:
            return None
        ts = np.asarray(src_t, dtype=float).flatten()
        yt = np.asarray(src_y, dtype=float)
        tt = np.asarray(tgt_t, dtype=float).flatten()
        if ts.size < 2 or tt.size == 0:
            return None
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)
        if yt.ndim != 2 or yt.shape[0] != ts.size:
            return None
        order = np.argsort(ts)
        ts = ts[order]
        yt = yt[order]
        out = np.full((tt.size, yt.shape[1]), np.nan, dtype=float)
        in_range = (tt >= ts[0]) & (tt <= ts[-1])
        if np.any(in_range):
            for j in range(yt.shape[1]):
                out[in_range, j] = np.interp(tt[in_range], ts, yt[:, j])
        return out
    def _quat_to_euler_row(quat):
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return roll, pitch, yaw
    euler = np.array([_quat_to_euler_row(simX[i, 3:7]) for i in range(len(simX))])
    refX_interp = _interp_series_to_time(ref_time_states, ref_states, time_states)
    refU_interp = _interp_series_to_time(ref_time_controls, ref_controls, time_controls)
    euler_ref = None
    if refX_interp is not None and refX_interp.shape[1] >= 7:
        euler_ref = np.array([_quat_to_euler_row(refX_interp[i, 3:7]) for i in range(len(refX_interp))])

    v_lin_w, w_base_w = base_lin_ang_world_from_robot_state(simX)
    v_lin_w_ref = None
    w_base_w_ref = None
    if refX_interp is not None and refX_interp.shape[1] >= 17:
        try:
            v_lin_w_ref, w_base_w_ref = base_lin_ang_world_from_robot_state(refX_interp)
        except Exception:
            v_lin_w_ref = None
            w_base_w_ref = None

    ee_pos = ee_v = ee_rpy = ee_w = None
    try:
        pm, pdata, pfid = _plot_pin_model_for_acados_fig()
        ee_pos, ee_v, ee_rpy, ee_w = compute_ee_kinematics_along_trajectory(simX, pm, pdata, pfid)
    except Exception:
        ee_pos = np.zeros((len(simX), 3))
        ee_v = ee_rpy = ee_w = ee_pos

    ee_pos_ref = None
    ee_v_ref = None
    ee_rpy_ref = None
    ee_w_ref = None
    if refX_interp is not None and refX_interp.shape[1] >= 7:
        ee_pos_ref = np.full((len(refX_interp), 3), np.nan, dtype=float)
        ee_v_ref = np.full((len(refX_interp), 3), np.nan, dtype=float)
        ee_rpy_ref = np.full((len(refX_interp), 3), np.nan, dtype=float)
        ee_w_ref = np.full((len(refX_interp), 3), np.nan, dtype=float)
        valid = np.isfinite(refX_interp).all(axis=1)
        if np.any(valid):
            try:
                pm_r, pdata_r, pfid_r = _plot_pin_model_for_acados_fig()
                ee_pos_r, ee_v_r, ee_rpy_r, ee_w_r = compute_ee_kinematics_along_trajectory(
                    refX_interp[valid], pm_r, pdata_r, pfid_r
                )
                ee_pos_ref[valid] = np.asarray(ee_pos_r, dtype=float)
                ee_v_ref[valid] = np.asarray(ee_v_r, dtype=float)
                ee_rpy_ref[valid] = np.asarray(ee_rpy_r, dtype=float)
                ee_w_ref[valid] = np.asarray(ee_w_r, dtype=float)
            except Exception:
                ee_pos_ref = None
                ee_v_ref = None
                ee_rpy_ref = None
                ee_w_ref = None

    def add_wp_lines(ax):
        if waypoint_times is None:
            witer = []
        else:
            witer = np.asarray(waypoint_times, dtype=float).flatten()
        for tv in witer:
            ax.axvline(x=float(tv), color="orange", linestyle="--", alpha=0.45, zorder=1)

    def scatter_wp_xyz_vs_time(ax, tw, pos_k_xyz, marker="o", size=44):
        """
        Mark constraint positions on a time-domain plot with three curves (x,y,z).
        pos_k_xyz: (K,3) with NaN rows skipped; tw: (K,) same K as planning table order.
        """
        if tw is None or pos_k_xyz is None:
            return False
        tw = np.asarray(tw, dtype=float).flatten()
        P = np.asarray(pos_k_xyz, dtype=float)
        if P.ndim != 2 or P.shape[1] < 3 or tw.size == 0:
            return False
        K = int(min(tw.size, P.shape[0]))
        colors = ("red", "green", "blue")
        any_p = False
        for k in range(K):
            row = P[k, :3]
            if not np.all(np.isfinite(row)):
                continue
            any_p = True
            tk = float(tw[k])
            for j in range(3):
                ax.scatter(
                    tk,
                    float(row[j]),
                    c=colors[j],
                    s=size,
                    marker=marker,
                    zorder=6,
                    edgecolors="black",
                    linewidths=0.4,
                    label="_nolegend_",
                )
        return any_p

    def _valid_wp_xyz(M):
        if M is None:
            return np.zeros((0, 3))
        A = np.asarray(M, dtype=float)
        if A.size == 0 or A.ndim != 2 or A.shape[1] < 3:
            return np.zeros((0, 3))
        v = np.isfinite(A[:, 0]) & np.isfinite(A[:, 1]) & np.isfinite(A[:, 2])
        return A[v, :3]

    positions = simX[:, :3]
    fig.clear()
    gs = fig.add_gridspec(4, 4, hspace=0.42, wspace=0.32, left=0.05, right=0.98, top=0.93, bottom=0.05)
    tinfo = {'fontsize': 9, 'labelpad': 2}

    # Row 0: Base
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(time_states, simX[:, 0], 'r-', label='x')
    ax00.plot(time_states, simX[:, 1], 'g-', label='y')
    ax00.plot(time_states, simX[:, 2], 'b-', label='z')
    if refX_interp is not None and refX_interp.shape[1] >= 3:
        ax00.plot(time_states, refX_interp[:, 0], 'r--', alpha=0.9, lw=1.1, label='ref x')
        ax00.plot(time_states, refX_interp[:, 1], 'g--', alpha=0.9, lw=1.1, label='ref y')
        ax00.plot(time_states, refX_interp[:, 2], 'b--', alpha=0.9, lw=1.1, label='ref z')
    add_wp_lines(ax00)
    h00, l00 = ax00.get_legend_handles_labels()
    if scatter_wp_xyz_vs_time(ax00, waypoint_times, waypoint_positions_base, marker="o", size=46):
        h00.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gold",
                markeredgecolor="k",
                markersize=6,
                lw=0,
                label="Base WP target",
            )
        )
        l00.append("Base WP target")
    if waypoint_times is not None and np.asarray(waypoint_times).size > 0:
        h00.append(Line2D([0], [0], color="orange", linestyle="--", alpha=0.55, lw=1.0, label="WP time"))
        l00.append("WP time")
    ax00.legend(h00, l00, loc='upper right', fontsize=7, framealpha=0.9)
    ax00.set_xlabel('Time (s)', **tinfo)
    ax00.set_ylabel('Position (m)', **tinfo)
    ax00.set_title('Base Position', fontsize=9)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(time_states, v_lin_w[:, 0], 'r-', label='vx')
    ax01.plot(time_states, v_lin_w[:, 1], 'g-', label='vy')
    ax01.plot(time_states, v_lin_w[:, 2], 'b-', label='vz')
    if v_lin_w_ref is not None:
        ax01.plot(time_states, v_lin_w_ref[:, 0], 'r--', alpha=0.9, lw=1.1, label='ref vx')
        ax01.plot(time_states, v_lin_w_ref[:, 1], 'g--', alpha=0.9, lw=1.1, label='ref vy')
        ax01.plot(time_states, v_lin_w_ref[:, 2], 'b--', alpha=0.9, lw=1.1, label='ref vz')
    add_wp_lines(ax01)
    ax01.set_xlabel('Time (s)', **tinfo)
    ax01.set_ylabel('Velocity (m/s)', **tinfo)
    ax01.set_title('Base linear vel. (world)', fontsize=9)
    ax01.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax02 = fig.add_subplot(gs[0, 2])
    ax02.plot(time_states, np.degrees(euler[:, 0]), 'r-', label='roll')
    ax02.plot(time_states, np.degrees(euler[:, 1]), 'g-', label='pitch')
    ax02.plot(time_states, np.degrees(euler[:, 2]), 'b-', label='yaw')
    if euler_ref is not None:
        ax02.plot(time_states, np.degrees(euler_ref[:, 0]), 'r--', alpha=0.9, lw=1.1, label='ref roll')
        ax02.plot(time_states, np.degrees(euler_ref[:, 1]), 'g--', alpha=0.9, lw=1.1, label='ref pitch')
        ax02.plot(time_states, np.degrees(euler_ref[:, 2]), 'b--', alpha=0.9, lw=1.1, label='ref yaw')
    add_wp_lines(ax02)
    ax02.set_xlabel('Time (s)', **tinfo)
    ax02.set_ylabel('Angle (°)', **tinfo)
    ax02.set_title('Base Orientation (Euler)', fontsize=9)
    ax02.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax03 = fig.add_subplot(gs[0, 3])
    ax03.plot(time_states, np.degrees(w_base_w[:, 0]), 'r-', label='ωx')
    ax03.plot(time_states, np.degrees(w_base_w[:, 1]), 'g-', label='ωy')
    ax03.plot(time_states, np.degrees(w_base_w[:, 2]), 'b-', label='ωz')
    if w_base_w_ref is not None:
        ax03.plot(time_states, np.degrees(w_base_w_ref[:, 0]), 'r--', alpha=0.9, lw=1.1, label='ref ωx')
        ax03.plot(time_states, np.degrees(w_base_w_ref[:, 1]), 'g--', alpha=0.9, lw=1.1, label='ref ωy')
        ax03.plot(time_states, np.degrees(w_base_w_ref[:, 2]), 'b--', alpha=0.9, lw=1.1, label='ref ωz')
    add_wp_lines(ax03)
    ax03.set_xlabel('Time (s)', **tinfo)
    ax03.set_ylabel('Angular vel (deg/s)', **tinfo)
    ax03.set_title('Base angular vel. (world)', fontsize=9)
    ax03.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # Row 1: EE
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(time_states, ee_pos[:, 0], 'r-', label='x')
    ax10.plot(time_states, ee_pos[:, 1], 'g-', label='y')
    ax10.plot(time_states, ee_pos[:, 2], 'b-', label='z')
    if ee_pos_ref is not None:
        ax10.plot(time_states, ee_pos_ref[:, 0], 'r--', alpha=0.9, lw=1.1, label='ref x')
        ax10.plot(time_states, ee_pos_ref[:, 1], 'g--', alpha=0.9, lw=1.1, label='ref y')
        ax10.plot(time_states, ee_pos_ref[:, 2], 'b--', alpha=0.9, lw=1.1, label='ref z')
    add_wp_lines(ax10)
    h10, l10 = ax10.get_legend_handles_labels()
    if scatter_wp_xyz_vs_time(ax10, waypoint_times, waypoint_positions_ee, marker="*", size=72):
        h10.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="darkorange",
                markeredgecolor="k",
                markersize=9,
                lw=0,
                label="EE WP target",
            )
        )
        l10.append("EE WP target")
    if waypoint_times is not None and np.asarray(waypoint_times).size > 0:
        h10.append(Line2D([0], [0], color="orange", linestyle="--", alpha=0.55, lw=1.0, label="WP time"))
        l10.append("WP time")
    ax10.legend(h10, l10, loc='upper right', fontsize=7, framealpha=0.9)
    ax10.set_xlabel('Time (s)', **tinfo)
    ax10.set_ylabel('Position (m)', **tinfo)
    ax10.set_title('EE Position', fontsize=9)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(time_states, ee_v[:, 0], 'r-', label='vx')
    ax11.plot(time_states, ee_v[:, 1], 'g-', label='vy')
    ax11.plot(time_states, ee_v[:, 2], 'b-', label='vz')
    if ee_v_ref is not None:
        ax11.plot(time_states, ee_v_ref[:, 0], 'r--', alpha=0.9, lw=1.1, label='ref vx')
        ax11.plot(time_states, ee_v_ref[:, 1], 'g--', alpha=0.9, lw=1.1, label='ref vy')
        ax11.plot(time_states, ee_v_ref[:, 2], 'b--', alpha=0.9, lw=1.1, label='ref vz')
    add_wp_lines(ax11)
    ax11.set_xlabel('Time (s)', **tinfo)
    ax11.set_ylabel('Velocity (m/s)', **tinfo)
    ax11.set_title('EE linear vel. (world)', fontsize=9)
    ax11.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax12 = fig.add_subplot(gs[1, 2])
    ax12.plot(time_states, np.degrees(ee_rpy[:, 0]), 'r-', label='roll')
    ax12.plot(time_states, np.degrees(ee_rpy[:, 1]), 'g-', label='pitch')
    ax12.plot(time_states, np.degrees(ee_rpy[:, 2]), 'b-', label='yaw')
    if ee_rpy_ref is not None:
        ax12.plot(time_states, np.degrees(ee_rpy_ref[:, 0]), 'r--', alpha=0.9, lw=1.1, label='ref roll')
        ax12.plot(time_states, np.degrees(ee_rpy_ref[:, 1]), 'g--', alpha=0.9, lw=1.1, label='ref pitch')
        ax12.plot(time_states, np.degrees(ee_rpy_ref[:, 2]), 'b--', alpha=0.9, lw=1.1, label='ref yaw')
    add_wp_lines(ax12)
    ax12.set_xlabel('Time (s)', **tinfo)
    ax12.set_ylabel('Angle (°)', **tinfo)
    ax12.set_title('EE Orientation (RPY)', fontsize=9)
    ax12.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax13 = fig.add_subplot(gs[1, 3])
    ax13.plot(time_states, np.degrees(ee_w[:, 0]), 'r-', label='ωx')
    ax13.plot(time_states, np.degrees(ee_w[:, 1]), 'g-', label='ωy')
    ax13.plot(time_states, np.degrees(ee_w[:, 2]), 'b-', label='ωz')
    if ee_w_ref is not None:
        ax13.plot(time_states, np.degrees(ee_w_ref[:, 0]), 'r--', alpha=0.9, lw=1.1, label='ref ωx')
        ax13.plot(time_states, np.degrees(ee_w_ref[:, 1]), 'g--', alpha=0.9, lw=1.1, label='ref ωy')
        ax13.plot(time_states, np.degrees(ee_w_ref[:, 2]), 'b--', alpha=0.9, lw=1.1, label='ref ωz')
    add_wp_lines(ax13)
    ax13.set_xlabel('Time (s)', **tinfo)
    ax13.set_ylabel('Angular vel (deg/s)', **tinfo)
    ax13.set_title('EE angular vel. (world)', fontsize=9)
    ax13.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # Row 2: Arm & controls
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(time_states, np.degrees(simX[:, 7]), 'r-', label='j1')
    ax20.plot(time_states, np.degrees(simX[:, 8]), 'g-', label='j2')
    if refX_interp is not None and refX_interp.shape[1] >= 9:
        ax20.plot(time_states, np.degrees(refX_interp[:, 7]), 'r--', alpha=0.9, lw=1.1, label='ref j1')
        ax20.plot(time_states, np.degrees(refX_interp[:, 8]), 'g--', alpha=0.9, lw=1.1, label='ref j2')
    add_wp_lines(ax20)
    ax20.set_xlabel('Time (s)', **tinfo)
    ax20.set_ylabel('Angle (°)', **tinfo)
    ax20.set_title('Arm Joint Angles', fontsize=9)
    ax20.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax21 = fig.add_subplot(gs[2, 1])
    ax21.plot(time_states, np.degrees(simX[:, 15]), 'r-', label='j1_dot')
    ax21.plot(time_states, np.degrees(simX[:, 16]), 'g-', label='j2_dot')
    if refX_interp is not None and refX_interp.shape[1] >= 17:
        ax21.plot(time_states, np.degrees(refX_interp[:, 15]), 'r--', alpha=0.9, lw=1.1, label='ref j1_dot')
        ax21.plot(time_states, np.degrees(refX_interp[:, 16]), 'g--', alpha=0.9, lw=1.1, label='ref j2_dot')
    add_wp_lines(ax21)
    ax21.set_xlabel('Time (s)', **tinfo)
    ax21.set_ylabel('Joint rate (deg/s)', **tinfo)
    ax21.set_title('Arm joint angular velocity', fontsize=9)
    ax21.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax22 = fig.add_subplot(gs[2, 2])
    ax23 = fig.add_subplot(gs[2, 3])
    if control_layout == "high_level" and simU.shape[1] >= 6:
        ax22.plot(time_controls, np.degrees(simU[:, 0]), 'r-', label='ωx')
        ax22.plot(time_controls, np.degrees(simU[:, 1]), 'g-', label='ωy')
        ax22.plot(time_controls, np.degrees(simU[:, 2]), 'b-', label='ωz')
        if refU_interp is not None and refU_interp.shape[1] >= 3:
            ax22.plot(time_controls, np.degrees(refU_interp[:, 0]), 'r--', alpha=0.9, lw=1.1, label='ref ωx')
            ax22.plot(time_controls, np.degrees(refU_interp[:, 1]), 'g--', alpha=0.9, lw=1.1, label='ref ωy')
            ax22.plot(time_controls, np.degrees(refU_interp[:, 2]), 'b--', alpha=0.9, lw=1.1, label='ref ωz')
        ax22.set_ylabel('Ang. rate cmd (deg/s)', **tinfo)
        ax22.set_title('High-level ω cmd', fontsize=9)
        ax22.legend(loc='upper right', fontsize=7, framealpha=0.9)
        ax22.set_xlabel('Time (s)', **tinfo)
        ax23.plot(time_controls, simU[:, 3], 'k-', label='T_tot')
        ax23.plot(time_controls, simU[:, 4], 'r--', label='θ1 cmd')
        ax23.plot(time_controls, simU[:, 5], 'g--', label='θ2 cmd')
        if refU_interp is not None and refU_interp.shape[1] >= 6:
            ax23.plot(time_controls, refU_interp[:, 3], 'k--', alpha=0.9, lw=1.1, label='ref T_tot')
            ax23.plot(time_controls, refU_interp[:, 4], 'r:', alpha=0.9, lw=1.1, label='ref θ1')
            ax23.plot(time_controls, refU_interp[:, 5], 'g:', alpha=0.9, lw=1.1, label='ref θ2')
        ax23.set_xlabel('Time (s)', **tinfo)
        ax23.set_ylabel('T (N) / θ (rad)', **tinfo)
        ax23.set_title('High-level T & θ cmd', fontsize=9)
        ax23.legend(loc='upper right', fontsize=7, framealpha=0.9)
    else:
        colors = ['r', 'g', 'b', 'orange']
        for i in range(min(4, simU.shape[1])):
            ax22.plot(time_controls, simU[:, i], color=colors[i], label=f'T{i+1}')
        if refU_interp is not None and refU_interp.shape[1] >= 4:
            for i in range(4):
                ax22.plot(
                    time_controls,
                    refU_interp[:, i],
                    linestyle='--',
                    color=colors[i],
                    alpha=0.9,
                    lw=1.1,
                    label=f'ref T{i+1}',
                )
        ax22.set_xlabel('Time (s)', **tinfo)
        ax22.set_ylabel('Thrust (N)', **tinfo)
        ax22.set_title('Base Control (Thrusters)', fontsize=9)
        ax22.legend(loc='upper right', fontsize=7, framealpha=0.9)
        if simU.shape[1] >= 6:
            ax23.plot(time_controls, simU[:, 4], 'r-', label='τ1')
            ax23.plot(time_controls, simU[:, 5], 'g-', label='τ2')
        if refU_interp is not None and refU_interp.shape[1] >= 6:
            ax23.plot(time_controls, refU_interp[:, 4], 'r--', alpha=0.9, lw=1.1, label='ref τ1')
            ax23.plot(time_controls, refU_interp[:, 5], 'g--', alpha=0.9, lw=1.1, label='ref τ2')
        ax23.set_xlabel('Time (s)', **tinfo)
        ax23.set_ylabel('Torque (N·m)', **tinfo)
        ax23.set_title('Arm Control (Joint Torques)', fontsize=9)
        ax23.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # Row 3
    ax30 = fig.add_subplot(gs[3, 0])
    ax30.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, label='Base')
    ax30.plot(ee_pos[:, 0], ee_pos[:, 1], 'm--', linewidth=1.2, label='EE')
    if refX_interp is not None and refX_interp.shape[1] >= 3:
        ax30.plot(refX_interp[:, 0], refX_interp[:, 1], color='tab:blue', linestyle='--', linewidth=1.2, label='Base ref')
    if ee_pos_ref is not None:
        ax30.plot(ee_pos_ref[:, 0], ee_pos_ref[:, 1], color='purple', linestyle=':', linewidth=1.1, label='EE ref')
    ax30.plot(positions[0, 0], positions[0, 1], 'go', markersize=6, label='Start')
    ax30.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=6, label='End')
    Bwp = _valid_wp_xyz(waypoint_positions_base)
    Ewp = _valid_wp_xyz(waypoint_positions_ee)
    if Bwp.shape[0]:
        ax30.scatter(Bwp[:, 0], Bwp[:, 1], c="tab:blue", s=50, marker="s", zorder=5, label="plan Base WP")
    if Ewp.shape[0]:
        ax30.scatter(Ewp[:, 0], Ewp[:, 1], c="darkorange", s=65, marker="*", zorder=6, label="plan EE WP")
    ax30.set_xlabel('X (m)', **tinfo)
    ax30.set_ylabel('Y (m)', **tinfo)
    ax30.set_title('Horizontal trajectory (XY)', fontsize=9)
    ax30.axis('equal')
    ax30.grid(True, alpha=0.3)
    ax30.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax31 = fig.add_subplot(gs[3, 1])
    ax31.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=1.5, label='Base')
    ax31.plot(ee_pos[:, 0], ee_pos[:, 2], 'm--', linewidth=1.2, label='EE')
    if refX_interp is not None and refX_interp.shape[1] >= 3:
        ax31.plot(refX_interp[:, 0], refX_interp[:, 2], color='tab:blue', linestyle='--', linewidth=1.2, label='Base ref')
    if ee_pos_ref is not None:
        ax31.plot(ee_pos_ref[:, 0], ee_pos_ref[:, 2], color='purple', linestyle=':', linewidth=1.1, label='EE ref')
    ax31.plot(positions[0, 0], positions[0, 2], 'go', markersize=6, label='Start')
    ax31.plot(positions[-1, 0], positions[-1, 2], 'rs', markersize=6, label='End')
    if Bwp.shape[0]:
        ax31.scatter(Bwp[:, 0], Bwp[:, 2], c="tab:blue", s=50, marker="s", zorder=5, label="plan Base WP")
    if Ewp.shape[0]:
        ax31.scatter(Ewp[:, 0], Ewp[:, 2], c="darkorange", s=65, marker="*", zorder=6, label="plan EE WP")
    ax31.set_xlabel('X (m)', **tinfo)
    ax31.set_ylabel('Z (m)', **tinfo)
    ax31.set_title('Vertical profile (XZ)', fontsize=9)
    ax31.grid(True, alpha=0.3)
    ax31.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax32 = fig.add_subplot(gs[3, 2])
    ax33 = fig.add_subplot(gs[3, 3])

    def _fill_footer_traj_opt(ax_cost, ax_time, meta: dict):
        backend = str(meta.get("backend") or "")
        tim = meta.get("timing") or {}
        n_it = int(tim.get("n_iter", 0) or 0)
        avg_ms = float(tim.get("avg_ms_per_iter", 0) or 0)
        tot_s = float(tim.get("total_s", 0) or 0)
        if backend == "crocoddyl":
            costs = meta.get("costs")
            if costs is not None and len(costs) > 0:
                ax_cost.semilogy(np.asarray(costs, dtype=float), "b-", linewidth=2)
                ax_cost.set_title("Crocoddyl: cost vs iter", fontsize=9)
            else:
                ax_cost.text(
                    0.5,
                    0.5,
                    "Crocoddyl BoxFDDP\n(no cost log)",
                    ha="center",
                    va="center",
                    transform=ax_cost.transAxes,
                    fontsize=9,
                )
                ax_cost.set_title("Crocoddyl: cost", fontsize=9)
            ax_cost.set_xlabel("Iteration", **tinfo)
            ax_cost.set_ylabel("Cost", **tinfo)
            ax_cost.grid(True, alpha=0.3)
        elif backend == "acados_traj":
            ax_cost.text(
                0.5,
                0.5,
                "Acados trajectory OCP\n(per-iter cost not plotted)",
                ha="center",
                va="center",
                transform=ax_cost.transAxes,
                fontsize=9,
            )
            ax_cost.set_title("Trajectory optimization", fontsize=9)
            ax_cost.set_xlabel("—", **tinfo)
            ax_cost.set_ylabel("—", **tinfo)
            ax_cost.grid(True, alpha=0.3)
        else:
            ax_cost.text(0.5, 0.5, "Unknown solver meta", ha="center", va="center", transform=ax_cost.transAxes)
            ax_cost.set_title("Solver", fontsize=9)
            ax_cost.grid(True, alpha=0.3)

        if n_it > 0 and avg_ms > 0:
            iters = np.arange(1, n_it + 1)
            ax_time.plot(iters, np.full(n_it, avg_ms), "g-", linewidth=2, label=f"Avg {avg_ms:.2f} ms/iter")
            ax_time.fill_between(iters, 0, np.full(n_it, avg_ms), alpha=0.15, color="g")
            ax_time.set_xlabel("Iteration", **tinfo)
            ax_time.set_ylabel("Time per iter (ms)", **tinfo)
            ttl = "Crocoddyl: time / iter" if backend == "crocoddyl" else "Acados traj.: time / iter"
            ax_time.set_title(ttl, fontsize=9)
            ax_time.legend(loc="upper right", fontsize=7)
            ax_time.grid(True, alpha=0.3)
        else:
            msg = f"Total {tot_s:.3f} s" if tot_s > 0 else "Timing N/A"
            if n_it > 0:
                msg = f"{n_it} iter, {msg}"
            ax_time.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_time.transAxes, fontsize=9)
            ax_time.set_title("Solver time", fontsize=9)
            ax_time.grid(True, alpha=0.3)

    if traj_solver_meta is not None:
        _fill_footer_traj_opt(ax32, ax33, traj_solver_meta)
    else:
        ax32.text(0.5, 0.5, 'Acados: cost in solver log', ha='center', va='center', transform=ax32.transAxes)
        ax32.set_xlabel('Iteration', **tinfo)
        ax32.set_ylabel('Cost', **tinfo)
        ax32.set_title('Cost convergence', fontsize=9)
        ax32.grid(True, alpha=0.3)

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


def plot_acados_3d_into_figure(
    simX,
    fig,
    waypoint_positions=None,
    waypoint_positions_ee=None,
    ref_states=None,
):
    """Plot acados 3D trajectory into existing figure."""
    if simX is None or fig is None:
        return None
    positions = np.asarray(simX[:, :3], dtype=float)
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    finite_row = np.isfinite(positions).all(axis=1)
    if not np.any(finite_row):
        ax.text2D(
            0.1,
            0.5,
            "3D: no finite base positions (NaN/Inf in sim states)",
            transform=ax.transAxes,
        )
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory (acados)')
        return fig
    X_plot = np.asarray(simX, dtype=float)[finite_row]
    positions = positions[finite_row]
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Base')
    ee_meas = None
    try:
        pm, pdata, pfid = _plot_pin_model_for_acados_fig()
        ee_meas, _, _, _ = compute_ee_kinematics_along_trajectory(X_plot, pm, pdata, pfid)
        ee_meas = np.asarray(ee_meas, dtype=float)
    except Exception:
        ee_meas = None
    if ee_meas is not None and ee_meas.shape[0] == positions.shape[0]:
        ax.plot(
            ee_meas[:, 0],
            ee_meas[:, 1],
            ee_meas[:, 2],
            color='m',
            linestyle='-',
            linewidth=1.4,
            label='EE',
        )
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='r', s=100, label='End')
    if waypoint_positions:
        for wp in waypoint_positions:
            ax.scatter(wp[0], wp[1], wp[2], color="tab:blue", s=130, marker="s", label="_nolegend_")
    if waypoint_positions_ee:
        for wp in waypoint_positions_ee:
            ax.scatter(wp[0], wp[1], wp[2], color="darkorange", s=150, marker="*", label="_nolegend_")
    if ref_states is not None:
        Xr = np.asarray(ref_states, dtype=float)
        if Xr.ndim == 2 and Xr.shape[1] >= 3 and Xr.shape[0] >= 2:
            vr = np.isfinite(Xr[:, :3]).all(axis=1)
            if np.any(vr):
                Br = Xr[vr, :3]
                ax.plot(
                    Br[:, 0],
                    Br[:, 1],
                    Br[:, 2],
                    color="tab:orange",
                    linestyle="--",
                    linewidth=1.5,
                    label="Base ref",
                )
                all_pts = np.vstack([positions.copy(), Br])
                try:
                    pm_r, pdata_r, pfid_r = _plot_pin_model_for_acados_fig()
                    EEr, _, _, _ = compute_ee_kinematics_along_trajectory(Xr[vr], pm_r, pdata_r, pfid_r)
                    EEr = np.asarray(EEr, dtype=float)
                    if EEr.ndim == 2 and EEr.shape[1] >= 3 and EEr.shape[0] > 1:
                        ax.plot(
                            EEr[:, 0],
                            EEr[:, 1],
                            EEr[:, 2],
                            color="purple",
                            linestyle="--",
                            linewidth=1.2,
                            label="EE ref",
                        )
                        all_pts = np.vstack([all_pts, EEr[:, :3]])
                except Exception:
                    pass
            else:
                all_pts = positions.copy()
        else:
            all_pts = positions.copy()
    else:
        all_pts = positions.copy()
    h_base, h_ee = bool(waypoint_positions), bool(waypoint_positions_ee)
    if h_base:
        ax.scatter([], [], color="tab:blue", s=130, marker="s", label="plan Base WP")
    if h_ee:
        ax.scatter([], [], color="darkorange", s=150, marker="*", label="plan EE WP")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory (acados)')
    ax.legend(loc='upper right', fontsize=8)
    if waypoint_positions:
        wp_arr = np.asarray(waypoint_positions, dtype=float).reshape(-1, 3)
        wp_ok = np.isfinite(wp_arr).all(axis=1)
        if np.any(wp_ok):
            all_pts = np.vstack([all_pts, wp_arr[wp_ok]])
    if waypoint_positions_ee:
        wpe = np.asarray(waypoint_positions_ee, dtype=float).reshape(-1, 3)
        wpe_ok = np.isfinite(wpe).all(axis=1)
        if np.any(wpe_ok):
            all_pts = np.vstack([all_pts, wpe[wpe_ok]])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5
    if not np.isfinite(max_range) or max_range <= 0:
        max_range = 0.5
    if max_range < 0.1:
        max_range = 0.5
    x_mid, y_mid, z_mid = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    if not all(np.isfinite([x_mid, y_mid, z_mid, max_range])):
        x_mid, y_mid, z_mid, max_range = 0.0, 0.0, 1.0, 0.5
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

    v_lin_w, w_base_w = base_lin_ang_world_from_robot_state(simX)
    om_deg = float(np.degrees(lim["omega_max"]))
    jvd_deg = float(np.degrees(lim["j_vel_max"]))

    # Row 1: velocities
    ax = axes[1, 0]
    ax.plot(time_arr, v_lin_w[:, 0], 'r-', label='vx')
    ax.plot(time_arr, v_lin_w[:, 1], 'g-', label='vy')
    ax.plot(time_arr, v_lin_w[:, 2], 'b-', label='vz')
    ax.axhline(lim["v_max"], color='gray', linestyle='--', alpha=0.7, label=f'±{lim["v_max"]} m/s')
    ax.axhline(-lim["v_max"], color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Base lin. vel. world (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(time_arr, np.degrees(w_base_w[:, 0]), 'r-', label='ωx')
    ax.plot(time_arr, np.degrees(w_base_w[:, 1]), 'g-', label='ωy')
    ax.plot(time_arr, np.degrees(w_base_w[:, 2]), 'b-', label='ωz')
    ax.axhline(om_deg, color='gray', linestyle='--', alpha=0.7, label=f'±{om_deg:.0f} deg/s')
    ax.axhline(-om_deg, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Base ang. vel. world (deg/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(time_arr, np.degrees(simX[:, 15]), 'r-', label='j1_dot')
    ax.plot(time_arr, np.degrees(simX[:, 16]), 'g-', label='j2_dot')
    ax.axhline(jvd_deg, color='gray', linestyle='--', alpha=0.7, label=f'±{jvd_deg:.0f} deg/s')
    ax.axhline(-jvd_deg, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Joint rate (deg/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: controls (branch by control-input semantics)
    t_u = time_arr[:-1]
    ax = axes[2, 0]
    if use_high_level and simU.shape[1] >= 6:
        ax.plot(t_u, np.degrees(simU[:, 0]), 'r-', label='ωx cmd')
        ax.plot(t_u, np.degrees(simU[:, 1]), 'g-', label='ωy cmd')
        ax.plot(t_u, np.degrees(simU[:, 2]), 'b-', label='ωz cmd')
        ax.axhline(om_deg, color='gray', linestyle='--', alpha=0.7, label=f'±{om_deg:.0f} deg/s')
        ax.axhline(-om_deg, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel('Body rate cmd (deg/s)')
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
