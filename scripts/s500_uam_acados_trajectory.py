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

from s500_uam_trajectory_planner import make_uam_state


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


def create_simple_ocp(
    start_state: np.ndarray,
    target_state: np.ndarray,
    duration: float = 5.0,
    dt: float = 0.02,
) -> "AcadosOcpSolver":
    """Create acados OCP for start -> target trajectory.
    N is computed from duration and dt: N = max(1, int(duration / dt)).
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
    ocp.solver_options.nlp_solver_max_iter = 200
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

    # Weights: pos(100), yaw(50), roll/pitch(10), jq(50), v_lin(1), omega(1), j_dot(10), u
    w_pos = 100.0
    w_yaw = 50.0
    w_rp = 10.0
    w_jq = 50.0
    w_v = 1.0
    w_omega = 1.0
    w_jdot = 10.0
    W_state = np.diag([
        w_pos, w_pos, w_pos, w_yaw, w_rp, w_rp,
        w_jq, w_jq,
        w_v, w_v, w_v, w_omega, w_omega, w_omega, w_jdot, w_jdot
    ])
    R = np.diag([1e-5] * 4 + [1e-3] * 2)
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = cost_y
    ocp.cost.yref = yref
    ocp.cost.W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))

    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = cost_y_e
    ocp.cost.W_e = W_state

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


def run_simple_trajectory(
    start_state: np.ndarray = None,
    target_state: np.ndarray = None,
    duration: float = 5.0,
    dt: float = 0.02,
):
    """Run simple start->target trajectory optimization.
    N = duration / dt (auto-computed).
    """
    if start_state is None:
        start_state = make_uam_state(0, 0, 1.0, j1=-1.2, j2=-0.6, yaw=0)
    if target_state is None:
        target_state = make_uam_state(1.0, 0.5, 1.2, j1=-0.8, j2=-0.3, yaw=np.pi / 4)

    solver = create_simple_ocp(start_state, target_state, duration, dt)
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
        return None, None, None, None

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
    return simX, simU, time_arr, dt_actual


def plot_results(simX, simU, time_arr, save_path: str = None):
    """Plot trajectory results with state limits shown."""
    if simX is None:
        return
    lim = STATE_LIMITS
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("S500 UAM Trajectory (acados)")

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

    # Row 2: controls
    t_u = time_arr[:-1]
    ax = axes[2, 0]
    for i in range(4):
        ax.plot(t_u, simU[:, i], label=f'T{i+1}')
    ax.set_ylabel('Thrust (N)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t_u, simU[:, 4], 'r-', label='τ1')
    ax.plot(t_u, simU[:, 5], 'g-', label='τ2')
    ax.set_ylabel('Torque (N·m)')
    ax.legend()
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
    target = make_uam_state(1.0, 0, 0.5, j1=-0.8, j2=-0.3, yaw=np.deg2rad(45))
    N = max(1, int(round(args.duration / args.dt)))
    print(f"Start:  pos={start[:3]}, arm=[{np.degrees(start[7]):.0f}, {np.degrees(start[8]):.0f}]°")
    print(f"Target: pos={target[:3]}, arm=[{np.degrees(target[7]):.0f}, {np.degrees(target[8]):.0f}]°")
    print(f"Duration: {args.duration}s, dt: {args.dt}s, N: {N}")
    print()

    simX, simU, time_arr, dt = run_simple_trajectory(
        start, target, args.duration, args.dt
    )

    if simX is not None:
        print("Optimization converged.")
        plot_results(simX, simU, time_arr, args.save)
        if args.save:
            np.savez(args.save.replace('.png', '.npz') if args.save.endswith('.png') else args.save + '.npz',
                     states=simX, controls=simU, time=time_arr)
    else:
        print("Optimization failed.")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
