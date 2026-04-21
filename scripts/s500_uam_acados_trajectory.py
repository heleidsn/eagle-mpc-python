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
    python s500_uam_acados_trajectory.py --problem single --duration 5 --dt 0.02
    python s500_uam_acados_trajectory.py --problem multi --segments 2 --duration 6
    python s500_uam_acados_trajectory.py --problem unified --segments 2 --duration 6
    python s500_uam_acados_trajectory.py --problem multi --multi-preset planner
    python s500_uam_acados_trajectory.py --control cascade --problem multi --debug-opt

    With default logging, SQP runs one iteration per ``solve()`` so each iteration is printed as it happens;
    after completion you still get NLP cost and timings (the full ``print_statistics`` table is skipped
    when a live stream was used). Pass --quiet-opt for a single batched solve and minimal output.

Requirements:
    - acados (pip install acados_template, build acados lib)
    - pinocchio, casadi, numpy, pyyaml
    - matplotlib for plotting (imported by s500_uam_acados_trajectory_plot.py)
"""

import argparse
import os
import subprocess
import time
import numpy as np
import yaml
from pathlib import Path


def _preload_acados_shared_libs():
    """Load qpOASES/hpipm/blasfeo before libacados. Runtime changes to LD_LIBRARY_PATH do not
    affect dlopen from an already-started Python process; absolute-path preload fixes OSError
    on libqpOASES_e.so when ACADOS_SOURCE_DIR is not set in the shell.
    """
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
    # Order follows NEEDED chain (blasfeo <- hpipm; qpOASES standalone; then libacados).
    for name in ("libblasfeo.so.0", "libqpOASES_e.so", "libhpipm.so"):
        path = os.path.join(libdir, name)
        if os.path.isfile(path):
            CDLL(path)


try:
    from acados_template import AcadosOcp, AcadosOcpSolver

    ACADOS_AVAILABLE = True
    _preload_acados_shared_libs()
except ImportError:
    ACADOS_AVAILABLE = False

try:
    from s500_uam_acados_model import build_acados_model
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    _model_err = e

try:
    from s500_uam_acados_model import (
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
    create_uam_simple_waypoints,
)
from s500_uam_acados_trajectory_plot import (
    plot_acados_into_figure,
    plot_acados_3d_into_figure,
    plot_results,
    plot_sqp_cost_vs_iteration,
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


def _ensure_tera_renderer_compat():
    """Use t_renderer 0.0.34 when the default binary fails with old glibc (e.g. GLIBC_2.32+)."""
    try:
        from acados_template import get_tera
        from acados_template.utils import get_tera_exec_path
    except ImportError:
        return
    tera_path = get_tera_exec_path()
    needs_compat = False
    if not (os.path.isfile(tera_path) and os.access(tera_path, os.X_OK)):
        needs_compat = True
    else:
        try:
            proc = subprocess.run([tera_path], capture_output=True, timeout=5)
            out = (proc.stdout or b"") + (proc.stderr or b"")
            if b"GLIBC_" in out or b"version `" in out:
                needs_compat = True
        except OSError:
            needs_compat = True
    if not needs_compat:
        return
    try:
        if os.path.isfile(tera_path):
            os.remove(tera_path)
    except OSError:
        pass
    get_tera(tera_version="0.0.34", force_download=True)


def _acados_ocp_generate_build_flags(code_export_dir: Path | str, model_name: str):
    """Return (generate, build) for AcadosOcpSolver.

    ``is_code_reuse_possible`` only compares the OCP hash to the cached JSON; it does not
    check that ``libacados_ocp_solver_<name>.so`` was built. After a failed or partial
    codegen (e.g. t_renderer error), the directory can contain JSON and CasADi exports but
    no shared library — we must compile, or fully regenerate when no Makefile exists.
    """
    try:
        from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix
    except ImportError:
        return True, True
    lib_fname = f"{get_shared_lib_prefix()}acados_ocp_solver_{model_name}{get_shared_lib_ext()}"
    d = Path(code_export_dir)
    so_path = d / lib_fname
    if so_path.is_file():
        return False, False
    if (d / "Makefile").is_file():
        return False, True
    return True, True


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

    gen, bld = _acados_ocp_generate_build_flags(code_export_dir, acados_model.name)
    solver = AcadosOcpSolver(
        ocp,
        json_file=str(json_path),
        build=bld,
        generate=gen,
        verbose=False,
        check_reuse_possible=True,
    )
    return solver


def _unified_shooting_parts(durations: list, dt: float) -> tuple[float, int, list[int], list[int]]:
    """Return (tf, N, N_parts, nodes) with nodes[m] = shooting node index of waypoint m (m=0..M-1)."""
    tf = float(sum(float(d) for d in durations))
    N_parts = [max(1, int(round(float(d) / float(dt)))) for d in durations]
    N = int(sum(N_parts))
    nodes = [0]
    for m in range(1, len(N_parts) + 1):
        nodes.append(int(sum(N_parts[:m])))
    assert nodes[-1] == N
    return tf, N, N_parts, nodes


def _q_configuration_from_state17(s: np.ndarray) -> np.ndarray:
    q = np.asarray(s[:9], dtype=float).copy().reshape(9)
    q[3:7] = _normalize_quaternion_np(q[3:7])
    return q


def create_unified_multiwaypoint_ocp(
    waypoints: list,
    durations: list,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    max_iter: int = 200,
) -> "AcadosOcpSolver":
    """Single OCP over the full horizon with hard equality on q (9D) at interior + terminal waypoints.

    Waypoint m is enforced at shooting node ``nodes[m]`` (see :func:`_unified_shooting_parts`).
    Initial state x0 is fixed to ``waypoints[0]`` (full 17D). Intermediate waypoints m=1..M-2 use
    path constraint h(x)=q-q_ref; final waypoint uses terminal h_e. Non-waypoint stages use loose h bounds.
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("acados_template not installed.")
    if not MODEL_AVAILABLE:
        raise ImportError(f"Could not import s500_uam_acados_model: {_model_err}")
    if len(waypoints) != len(durations) + 1:
        raise ValueError("len(waypoints) must equal len(durations) + 1")

    import casadi as ca

    tf, N, _N_parts, _nodes = _unified_shooting_parts(durations, dt)
    if N < 1:
        raise ValueError("invalid horizon N")

    acados_model, _pin, nq, nv, nu = build_acados_model()
    p = ca.SX.sym("p_wp", 9)
    acados_model.p = p
    acados_model.name = "s500_uam_unified"
    x = acados_model.x
    acados_model.con_h_expr = x[:9] - p
    acados_model.con_h_expr_e = x[:9] - p

    ocp = AcadosOcp()
    ocp.model = acados_model
    nx = nq + nv

    ocp.dims.N = N
    ocp.solver_options.tf = tf
    ocp.solver_options.nlp_solver_max_iter = max_iter
    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = N

    target_state = np.asarray(waypoints[-1], dtype=float).flatten()
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
    r_thrust = control_weight
    r_torque = control_weight * 10000.0
    R = np.diag([r_thrust] * 4 + [r_torque] * 2)
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = cost_y
    ocp.cost.yref = yref
    ocp.cost.W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = cost_y_e
    ocp.cost.W_e = W_state * float(waypoint_multiplier)

    cfg = load_s500_config()
    platform = cfg["platform"]
    min_thrust = platform["min_thrust"]
    max_thrust = platform["max_thrust"]
    ocp.constraints.lbu = np.array([min_thrust] * 4 + [-2.0] * 2)
    ocp.constraints.ubu = np.array([max_thrust] * 4 + [2.0] * 2)
    ocp.constraints.idxbu = np.arange(nu)

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

    x0 = np.asarray(waypoints[0], dtype=float).flatten()
    if x0.size != 17:
        raise ValueError(f"unified waypoints must be 17D robot states, first has size {x0.size}")
    ocp.constraints.x0 = x0

    nh = 9
    huge = 1e6
    ocp.constraints.lh = -huge * np.ones(nh)
    ocp.constraints.uh = huge * np.ones(nh)
    ocp.constraints.lh_e = -huge * np.ones(nh)
    ocp.constraints.uh_e = huge * np.ones(nh)

    ocp.parameter_values = np.zeros(9)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0

    script_dir = Path(__file__).parent
    code_export_dir = script_dir.parent / "c_generated_code" / "s500_uam_unified"
    json_path = code_export_dir / "s500_uam_unified_ocp.json"
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(json_path)

    gen, bld = _acados_ocp_generate_build_flags(code_export_dir, acados_model.name)
    solver = AcadosOcpSolver(
        ocp,
        json_file=str(json_path),
        build=bld,
        generate=gen,
        verbose=False,
        check_reuse_possible=True,
    )
    return solver


def _apply_unified_waypoint_hard_constraints(
    solver,
    waypoints: list,
    durations: list,
    dt: float,
    loose_abs: float = 1e6,
):
    """Tighten h bounds to 0 at waypoint nodes; loose elsewhere. Set parameters p = q_ref at each stage.

    acados uses nh=0 at shooting stage 0 (no path h on x0); path ``con_h_expr`` applies at stages 1..N-1.
    Terminal ``con_h_expr_e`` is set at stage N via the same ``lh``/``uh`` field in the Python API.
    """
    _, N, _parts, nodes = _unified_shooting_parts(durations, dt)
    M = len(waypoints)
    nh = 9
    loose_lh = -loose_abs * np.ones(nh)
    loose_uh = loose_abs * np.ones(nh)
    tight = np.zeros(nh)
    zparam = np.zeros(nh)

    def _cset(stage: int, lh: np.ndarray, uh: np.ndarray) -> None:
        try:
            solver.constraints_set(stage, "lh", lh, api="new")
            solver.constraints_set(stage, "uh", uh, api="new")
        except TypeError:
            solver.constraints_set(stage, "lh", lh)
            solver.constraints_set(stage, "uh", uh)

    # Path constraints: stages 1 .. N-1 only (stage 0 has nh=0 in acados_template).
    for i in range(1, N):
        _cset(i, loose_lh, loose_uh)
        solver.set(i, "p", zparam)

    for m in range(1, M - 1):
        k = int(nodes[m])
        q_ref = _q_configuration_from_state17(waypoints[m])
        if 1 <= k < N:
            solver.set(k, "p", q_ref)
            _cset(k, tight, tight)

    qN = _q_configuration_from_state17(waypoints[-1])
    solver.set(N, "p", qN)
    _cset(N, tight, tight)


def _robot_state17_at_time_on_waypoints(t: float, waypoints: list, durations: list) -> np.ndarray:
    """Piecewise-linear (in time) interpolation between consecutive waypoints in 17D."""
    taus = [0.0]
    for d in durations:
        taus.append(taus[-1] + float(d))
    t = float(t)
    if t <= taus[0] + 1e-12:
        return np.asarray(waypoints[0], dtype=float).reshape(17).copy()
    if t >= taus[-1] - 1e-12:
        return np.asarray(waypoints[-1], dtype=float).reshape(17).copy()
    for m in range(len(durations)):
        t0, t1 = taus[m], taus[m + 1]
        if t <= t1 + 1e-12:
            denom = max(t1 - t0, 1e-9)
            alpha = float(np.clip((t - t0) / denom, 0.0, 1.0))
            return _interp_robot_state_17(
                alpha,
                np.asarray(waypoints[m], dtype=float).reshape(17),
                np.asarray(waypoints[m + 1], dtype=float).reshape(17),
            )
    return np.asarray(waypoints[-1], dtype=float).reshape(17).copy()


def _warm_start_unified_direct_guess(
    solver,
    waypoints: list,
    durations: list,
    dt: float,
):
    """Piecewise-linear guess in 17D state along waypoint times; hover-like thrust for u."""
    tf, N, _parts, _nodes = _unified_shooting_parts(durations, dt)

    cfg = load_s500_config()
    m_th = (cfg["platform"]["min_thrust"] + cfg["platform"]["max_thrust"]) / 2.0
    u_hover = np.array([m_th] * 4 + [0.0, 0.0], dtype=float)

    for k in range(N + 1):
        tk = k * tf / N
        xk = _robot_state17_at_time_on_waypoints(tk, waypoints, durations)
        solver.set(k, "x", xk)

    for i in range(N):
        solver.set(i, "u", u_hover)


def run_unified_multiwaypoint_trajectory(
    waypoints: list,
    durations: list,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    max_iter: int = 200,
    verbose_opt: bool = True,
):
    """Single-OCP multi-waypoint trajectory with hard q constraints at each waypoint node (direct dynamics only)."""
    solver = create_unified_multiwaypoint_ocp(
        waypoints,
        durations,
        dt=dt,
        state_weight=state_weight,
        control_weight=control_weight,
        waypoint_multiplier=waypoint_multiplier,
        max_iter=max_iter,
    )
    _warm_start_unified_direct_guess(solver, waypoints, durations, dt)
    _apply_unified_waypoint_hard_constraints(solver, waypoints, durations, dt)

    t0 = time.perf_counter()
    status, n_live_steps, cost_trace = _acados_solve_with_optional_live_log(
        solver, max_iter, verbose_opt, "unified"
    )
    t_wall = time.perf_counter() - t0
    t_cpu = solver.get_stats("time_tot")
    if n_live_steps is not None:
        n_iter = int(n_live_steps)
    else:
        n_iter = solver.get_stats("nlp_iter")
        if n_iter is None:
            n_iter = solver.get_stats("sqp_iter")
        n_iter = int(n_iter) if n_iter is not None else -1
    t_per_iter = (t_cpu / n_iter * 1000) if n_iter > 0 else 0
    print(
        f"Optimization (unified): {n_iter} iters, {t_cpu:.4f}s CPU, {t_wall:.4f}s wall, "
        f"{t_per_iter:.2f} ms/iter  status={status}"
    )
    _report_acados_optimization_log(
        solver,
        status,
        header="[unified]",
        verbose=verbose_opt,
        skip_statistics_table=(n_live_steps is not None),
    )

    if status not in (0, 2):
        print(f"acados solver returned status {status}")
        return None, None, None, None, None

    tf, N, _parts, _nodes = _unified_shooting_parts(durations, dt)
    nx = len(np.asarray(waypoints[0]).flatten())
    nu = 6
    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i, :] = solver.get(i, "x")
        simU[i, :] = solver.get(i, "u")
    simX[N, :] = solver.get(N, "x")
    dt_actual = tf / N
    time_arr = np.linspace(0, tf, N + 1)
    stats = {
        "n_iter": max(n_iter, 0),
        "total_s": float(t_wall),
        "avg_ms_per_iter": float(t_per_iter),
    }
    if cost_trace:
        stats["cost_trace"] = cost_trace
        stats["cost_trace_iters"] = list(range(1, len(cost_trace) + 1))
    return simX, simU, time_arr, dt_actual, stats


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

    gen, bld = _acados_ocp_generate_build_flags(code_export_dir, acados_model.name)
    solver = AcadosOcpSolver(
        ocp,
        json_file=str(json_path),
        build=bld,
        generate=gen,
        verbose=False,
        check_reuse_possible=True,
    )
    _warm_start_cascade_trajectory_solver(
        solver, pin_model, start_state, target_state, N, min_thrust, max_thrust
    )
    return solver


def _report_acados_optimization_log(
    solver,
    status: int,
    *,
    header: str = "",
    verbose: bool = True,
    skip_statistics_table: bool = False,
) -> None:
    """After ``solve()``: print SQP iteration table, NLP cost, and timing (acados_template API)."""
    if not verbose:
        return
    prefix = f"{header} " if header else ""
    print(f"{prefix}--- acados optimization log (exit status {status}) ---", flush=True)
    if not skip_statistics_table:
        try:
            solver.print_statistics()
        except Exception as exc:
            print(f"  print_statistics failed: {exc}", flush=True)
    try:
        cost = float(solver.get_cost())
        print(f"{prefix}NLP cost at solution: {cost:.8e}", flush=True)
    except Exception as exc:
        print(f"{prefix}get_cost: {exc}", flush=True)
    try:
        bits = []
        for key in ("nlp_iter", "sqp_iter", "qp_iter"):
            v = solver.get_stats(key)
            if v is not None:
                bits.append(f"{key}={int(v)}")
        for key in ("time_tot", "time_lin", "time_sim", "time_qp", "time_reg"):
            v = solver.get_stats(key)
            if v is not None and float(v) > 0:
                bits.append(f"{key}={float(v) * 1000:.3f}ms")
        if bits:
            print(f"{prefix}" + " | ".join(bits), flush=True)
    except Exception:
        pass
    print(f"{prefix}--- end optimization log ---", flush=True)


def _acados_return_status_meaning(status: int) -> str:
    """Short decode of acados ``ocp_nlp_solver`` return values (see ``acados/utils/types.h``)."""
    s = int(status)
    names = {
        -1: "UNKNOWN",
        0: "SUCCESS",
        1: "NAN_DETECTED",
        2: "MAXITER",
        3: "MINSTEP",
        4: "QP_FAILURE",
        5: "READY",
        6: "UNBOUNDED",
        7: "TIMEOUT",
        8: "QPSCALING_BOUNDS",
        9: "INFEASIBLE",
    }
    return names.get(s, f"code_{s}")


def _solve_ocp_with_live_log(solver, max_iter: int, label: str = "cascade"):
    """Run SQP step-by-step and print per-step residuals/step lengths in the terminal for debugging; on failure, fall back to a single solve + print_level.

    When ``nlp_solver_max_iter=1``, each inner ``solve()`` often returns status **2 (MAXITER)** because one SQP
    iteration hits the per-call limit; the outer loop continues until **0 (SUCCESS)** or another terminal status.

    Statistics columns (when available) follow acados ``ocp_nlp_sqp``: ``res_stat``, ``res_eq``, ``res_ineq``,
    ``res_comp``, then ``qp_stat``, ``qp_iter``, ``alpha``, ...

    Returns:
        (status, n_sqp_steps, cost_trace): ``cost_trace`` is a list of NLP costs after each inner ``solve()`` in
        step-by-step mode; otherwise ``None``.
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
        return st, None, None

    print(
        f"[{label}] live SQP: inner solve uses max_iter=1 → status 2 (MAXITER) each step is normal; "
        f"wait for status 0 (SUCCESS). Stats: res_stat res_eq res_ineq res_comp qp_stat qp_iter alpha …",
        flush=True,
    )
    last = 2
    cost_trace: list[float] = []
    for it in range(max_iter):
        last = solver.solve()
        try:
            cost_trace.append(float(solver.get_cost()))
        except Exception:
            cost_trace.append(float("nan"))
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
        c_iter = cost_trace[-1]
        cost_s = f"  cost={c_iter:.8e}"
        print(
            f"[{label}] SQP {it + 1}/{max_iter}  status={last} ({_acados_return_status_meaning(last)})"
            f"{cost_s}{extra}",
            flush=True,
        )
        if last == 0:
            return 0, it + 1, cost_trace
        if last not in (0, 2):
            print(f"[{label}] solver exit status {last}", flush=True)
            return last, it + 1, cost_trace
    print(f"[{label}] reached max_iter={max_iter} (last status={last})", flush=True)
    return last, max_iter, cost_trace


def _merge_sqp_cost_traces(all_stats: list) -> tuple[list[float], list[int]]:
    """Concatenate per-segment ``cost_trace`` with a global 1-based iteration index."""
    costs: list[float] = []
    iters: list[int] = []
    off = 0
    for s in all_stats:
        tr = s.get("cost_trace")
        if not tr:
            continue
        for k, c in enumerate(tr):
            costs.append(float(c))
            iters.append(off + k + 1)
        off += len(tr)
    return costs, iters


def _acados_solve_with_optional_live_log(
    solver, max_iter: int, verbose_opt: bool, label: str
) -> tuple[int, int | None, list[float] | None]:
    """Run NLP solve; if ``verbose_opt``, use one SQP step per ``solve()`` call so each iteration prints immediately.

    Returns ``(status, n_sqp_executed, cost_trace)`` where ``n_sqp_executed`` is set when the live loop was used
    (including early convergence); ``None`` means a single full ``solve()`` was used (quiet or step-loop unavailable).
    ``cost_trace`` is only filled in the live step loop.
    """
    if not verbose_opt:
        return int(solver.solve()), None, None
    status, n_live, cost_trace = _solve_ocp_with_live_log(solver, max_iter, label=label)
    return int(status), n_live, cost_trace


def run_simple_trajectory(
    start_state: np.ndarray = None,
    target_state: np.ndarray = None,
    duration: float = 5.0,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    max_iter: int = 200,
    verbose_opt: bool = True,
    _segment_label: str | None = None,
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
    hdr = _segment_label if _segment_label else "direct"
    t0 = time.perf_counter()
    status, n_live_steps, cost_trace = _acados_solve_with_optional_live_log(
        solver, max_iter, verbose_opt, hdr
    )
    t_wall = time.perf_counter() - t0

    # Get solver statistics (nlp_iter for SQP/DDP, time_tot for CPU time)
    t_cpu = solver.get_stats("time_tot")
    if n_live_steps is not None:
        n_iter = int(n_live_steps)
    else:
        n_iter = solver.get_stats("nlp_iter")
        if n_iter is None:
            n_iter = solver.get_stats("sqp_iter")
        if n_iter is None:
            n_iter = solver.get_stats("ddp_iter")
        n_iter = int(n_iter) if n_iter is not None else -1
    t_per_iter = (t_cpu / n_iter * 1000) if n_iter > 0 else 0
    seg = f"{_segment_label} " if _segment_label else ""
    print(
        f"{seg}Optimization: {n_iter} iterations, {t_cpu:.4f}s CPU, {t_wall:.4f}s wall, "
        f"{t_per_iter:.2f} ms/iter avg"
    )
    _report_acados_optimization_log(
        solver,
        status,
        header=f"[{hdr}]",
        verbose=verbose_opt,
        skip_statistics_table=(n_live_steps is not None),
    )

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
    if cost_trace:
        stats["cost_trace"] = cost_trace
        stats["cost_trace_iters"] = list(range(1, len(cost_trace) + 1))
    return simX, simU, time_arr, dt_actual, stats


def run_multiwaypoint_trajectory(
    waypoints: list,
    durations: list,
    dt: float = 0.02,
    state_weight: float = 1.0,
    control_weight: float = 1e-5,
    waypoint_multiplier: float = 1000.0,
    max_iter: int = 200,
    verbose_opt: bool = True,
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
            verbose_opt=verbose_opt,
            _segment_label=f"segment {i + 1}/{len(durations)}",
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
    mc, mi = _merge_sqp_cost_traces(all_stats)
    if mc:
        merged_stats["cost_trace"] = mc
        merged_stats["cost_trace_iters"] = mi
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
    verbose_opt: bool = True,
    _segment_label: str | None = None,
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
    hdr = _segment_label if _segment_label else (label or "cascade")
    t0 = time.perf_counter()
    n_sqp_logged = None
    cost_trace = None
    if debug_opt:
        print(f"[{label}] start optimization  N={max(1, int(round(duration / dt)))}  max_iter={max_iter}", flush=True)
        status, n_sqp_logged, cost_trace = _solve_ocp_with_live_log(solver, max_iter, label=label)
    elif verbose_opt:
        status, n_sqp_logged, cost_trace = _solve_ocp_with_live_log(solver, max_iter, label=hdr)
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
    seg = f"{_segment_label} " if _segment_label else ""
    print(
        f"{seg}Optimization (cascade): {n_iter} iterations, {t_cpu:.4f}s CPU, {t_wall:.4f}s wall, "
        f"{t_per_iter:.2f} ms/iter avg"
    )
    _report_acados_optimization_log(
        solver,
        status,
        header=f"[{hdr}]",
        verbose=verbose_opt,
        skip_statistics_table=(n_sqp_logged is not None),
    )

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
    if cost_trace:
        stats["cost_trace"] = cost_trace
        stats["cost_trace_iters"] = list(range(1, len(cost_trace) + 1))
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
    verbose_opt: bool = True,
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
            verbose_opt=verbose_opt,
            _segment_label=f"segment {i + 1}/{len(durations)}",
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
    mc, mi = _merge_sqp_cost_traces(all_stats)
    if mc:
        merged_stats["cost_trace"] = mc
        merged_stats["cost_trace_iters"] = mi
    return simX, simU, time_arr, dt_actual, merged_stats


def _yaw_from_uam_state(state: np.ndarray) -> float:
    qx, qy, qz, qw = state[3], state[4], state[5], state[6]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))


def _waypoints_linear_start_target(
    start: np.ndarray, target: np.ndarray, n_segments: int
) -> list:
    """n_segments >= 1: return n_segments+1 waypoint states along start..target (pos, j1/j2, yaw)."""
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    x0, y0, z0, j10, j20 = start[0], start[1], start[2], start[7], start[8]
    x1, y1, z1, j11, j21 = target[0], target[1], target[2], target[7], target[8]
    yaw0, yaw1 = _yaw_from_uam_state(start), _yaw_from_uam_state(target)
    wps = []
    for i in range(n_segments + 1):
        a = i / n_segments
        wps.append(
            make_uam_state(
                (1 - a) * x0 + a * x1,
                (1 - a) * y0 + a * y1,
                (1 - a) * z0 + a * z1,
                j1=(1 - a) * j10 + a * j11,
                j2=(1 - a) * j20 + a * j21,
                yaw=(1 - a) * yaw0 + a * yaw1,
            )
        )
    return wps


def _parse_segment_durations_csv(
    csv: str | None, n_segments: int, total_duration: float
) -> list[float]:
    """If csv is None, split total_duration equally across n_segments."""
    if csv is None:
        if n_segments < 1:
            raise ValueError("n_segments must be >= 1")
        d = float(total_duration) / n_segments
        return [d] * n_segments
    parts = [float(x.strip()) for x in csv.split(",") if x.strip()]
    if len(parts) != n_segments:
        raise ValueError(
            f"--segment-durations: expected {n_segments} values, got {len(parts)} ({parts!r})"
        )
    return parts


def _resolve_main_multi_waypoints(args, start: np.ndarray, target: np.ndarray):
    """Build (waypoints, durations, summary) for --problem multi / unified demos."""
    if args.multi_preset == "planner":
        waypoints, durations = create_uam_simple_waypoints()
        return (
            waypoints,
            durations,
            f"planner preset: {len(waypoints)} WPs, total {float(sum(durations)):.2f}s",
        )
    if args.segments < 1:
        raise ValueError("--segments must be >= 1")
    durations = _parse_segment_durations_csv(
        args.segment_durations, args.segments, args.duration
    )
    waypoints = _waypoints_linear_start_target(start, target, args.segments)
    return (
        waypoints,
        durations,
        f"linear: {len(waypoints)} WPs, {args.segments} segs, total {float(sum(durations)):.2f}s",
    )


def main():
    parser = argparse.ArgumentParser(description='S500 UAM trajectory planning with acados')
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Deprecated no-op (kept for old scripts); use --problem single.',
    )
    parser.add_argument('--duration', type=float, default=3.0, help='Trajectory duration (s)')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step (s), N = duration/dt')
    parser.add_argument('--save', type=str, help='Save plot path')
    parser.add_argument(
        '--problem',
        type=str,
        choices=('single', 'multi', 'unified'),
        default='unified',
        help=(
            'single: one OCP start→target; multi: independent OCP per segment; '
            'unified: one OCP with hard q equality at each waypoint node (direct dynamics only).'
        ),
    )
    parser.add_argument(
        '--segments',
        type=int,
        default=2,
        help='(multi/unified + linear preset) number of segments; waypoints = segments + 1.',
    )
    parser.add_argument(
        '--segment-durations',
        type=str,
        default=None,
        help=(
            '(multi/unified + linear preset) comma-separated segment lengths in seconds, '
            'e.g. 1.5,1.5. Default: split --duration equally across --segments.'
        ),
    )
    parser.add_argument(
        '--multi-preset',
        type=str,
        choices=('linear', 'planner'),
        default='linear',
        help=(
            '(problem=multi or unified) linear: interpolate between demo start/target; '
            'planner: built-in create_uam_simple_waypoints() path (fixed durations).'
        ),
    )
    parser.add_argument('--state-weight', type=float, default=1.0, help='Stage cost scale (pos/orientation/vel).')
    parser.add_argument('--control-weight', type=float, default=1e-5, help='Control regularization weight.')
    parser.add_argument(
        '--waypoint-multiplier',
        type=float,
        default=1000.0,
        help='Terminal state weight scale at segment ends (same as GUI / Crocoddyl).',
    )
    parser.add_argument('--max-iter', type=int, default=200, help='SQP max iterations per segment.')
    parser.add_argument(
        '--control',
        type=str,
        choices=(CONTROL_INPUT_DIRECT, CONTROL_INPUT_CASCADE),
        default=CONTROL_INPUT_DIRECT,
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
    parser.add_argument(
        '--quiet-opt',
        action='store_true',
        help='Suppress detailed acados log (iter table, NLP cost, timings) on stdout',
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

    _ensure_tera_renderer_compat()

    print("S500 UAM acados trajectory optimization")
    print("=" * 50)

    start = make_uam_state(0, 0, 0.0, j1=-1.2, j2=-0.6, yaw=0)
    target = make_uam_state(1.0, 0, 0.5, j1=-0.8, j2=-0.3, yaw=np.deg2rad(0))
    N = max(1, int(round(args.duration / args.dt)))
    print(
        f"Problem: {args.problem}  |  control: {args.control}  |  multi-preset: {args.multi_preset}"
    )
    print(f"Start:  pos={start[:3]}, arm=[{np.degrees(start[7]):.0f}, {np.degrees(start[8]):.0f}]°")
    print(f"Target: pos={target[:3]}, arm=[{np.degrees(target[7]):.0f}, {np.degrees(target[8]):.0f}]°")
    print(f"dt: {args.dt}s, N (single-segment horizon): {N}")
    print(
        f"Costs: state_weight={args.state_weight}, control_weight={args.control_weight}, "
        f"waypoint_multiplier={args.waypoint_multiplier}, max_iter={args.max_iter}"
    )
    print()

    run_kw = dict(
        dt=args.dt,
        state_weight=args.state_weight,
        control_weight=args.control_weight,
        waypoint_multiplier=args.waypoint_multiplier,
        max_iter=args.max_iter,
        verbose_opt=not args.quiet_opt,
    )

    if args.problem == 'single':
        print(f"Single segment duration: {args.duration}s")
        print()
        if args.control == CONTROL_INPUT_CASCADE:
            if not CASCADE_TRAJ_AVAILABLE:
                print("ERROR: cascade control requires pinocchio/casadi and s500_uam_acados_model (cascade helpers).")
                return 1
            simX, simU, time_arr, dt, _stats = run_simple_trajectory_cascade(
                start,
                target,
                args.duration,
                debug_opt=args.debug_opt,
                **run_kw,
            )
            plot_ctrl = CONTROL_INPUT_CASCADE
        else:
            simX, simU, time_arr, dt, _stats = run_simple_trajectory(
                start, target, args.duration, **run_kw
            )
            plot_ctrl = CONTROL_INPUT_DIRECT
    elif args.problem == "unified":
        if args.control != CONTROL_INPUT_DIRECT:
            print("ERROR: unified mode requires --control direct (single OCP, 17D state).")
            return 1
        try:
            waypoints, durations, descr = _resolve_main_multi_waypoints(args, start, target)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
        print(f"Unified single OCP — {descr}")
        for i, wp in enumerate(waypoints):
            print(f"  wp{i}: pos={wp[:3]}, arm=[{np.degrees(wp[7]):.0f}, {np.degrees(wp[8]):.0f}]°")
        print()
        simX, simU, time_arr, dt, _stats = run_unified_multiwaypoint_trajectory(
            waypoints,
            durations,
            **run_kw,
        )
        plot_ctrl = CONTROL_INPUT_DIRECT
    else:
        try:
            waypoints, durations, descr = _resolve_main_multi_waypoints(args, start, target)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
        print(f"Multi (segmented) — {descr}")
        for i, wp in enumerate(waypoints):
            print(f"  wp{i}: pos={wp[:3]}, arm=[{np.degrees(wp[7]):.0f}, {np.degrees(wp[8]):.0f}]°")
        print()

        if args.control == CONTROL_INPUT_CASCADE:
            if not CASCADE_TRAJ_AVAILABLE:
                print("ERROR: cascade control requires pinocchio/casadi and s500_uam_acados_model (cascade helpers).")
                return 1
            simX, simU, time_arr, dt, _stats = run_multiwaypoint_trajectory_cascade(
                waypoints,
                durations,
                debug_opt=args.debug_opt,
                **run_kw,
            )
            plot_ctrl = CONTROL_INPUT_CASCADE
        else:
            simX, simU, time_arr, dt, _stats = run_multiwaypoint_trajectory(
                waypoints,
                durations,
                **run_kw,
            )
            plot_ctrl = CONTROL_INPUT_DIRECT

    if simX is not None:
        print("Optimization converged.")
        plot_results(simX, simU, time_arr, args.save, control_input=plot_ctrl)
        ct = _stats.get("cost_trace") if _stats else None
        if ct:
            cost_save = None
            if args.save:
                p = Path(args.save)
                stem = p.stem if p.suffix else p.name
                cost_save = str(p.with_name(f"{stem}_sqp_cost.png"))
            plot_sqp_cost_vs_iteration(
                ct,
                iteration_indices=_stats.get("cost_trace_iters"),
                title=f"S500 UAM acados — NLP cost ({args.problem})",
                save_path=cost_save,
                show=True,
            )
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
