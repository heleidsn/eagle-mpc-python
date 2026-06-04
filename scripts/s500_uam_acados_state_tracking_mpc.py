#!/usr/bin/env python3
"""
Acados NMPC closed-loop tracking along a full-state plan (t_plan, x_plan).

Cost weights are driven by the same GUI parameters as Crocoddyl full-state tracking
(w_pos, w_att, w_joint, w_vel, w_omega, w_joint_vel, w_control, w_u_thrust,
w_u_joint_torque, w_state_track, w_terminal_track).

Plant integration reuses the CasADi RK4 helper from s500_uam_closed_loop_plant /
s500_uam_ee_snap_tracking_mpc (same as Acados EE tracking).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from acados_template import AcadosOcp, AcadosOcpSolver

    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

try:
    import casadi as ca
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError as e:
    PINOCCHIO_AVAILABLE = False
    _pin_err = e

try:
    from s500_uam_acados_model import build_acados_model
    from s500_uam_acados_trajectory import (
        STATE_LIMITS,
        _state_to_cost_ref,
        load_s500_config,
    )

    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    _deps_err = e

from s500_uam_crocoddyl_state_tracking_mpc import interp_full_state_piecewise
from s500_uam_closed_loop_plant import CasadiRK4Plant
from s500_uam_ee_snap_tracking_mpc import (
    EE_FRAME_NAME,
    REPO_ROOT,
    _acados_cpu_time_s,
    _acados_nlp_iterations,
    _make_f_expl_fun,
    hover_thrust_controls,
    rollout_nominal_trajectory,
    set_solver_initial_guess,
    shift_solver_initial_guess,
)

SCRIPT_DIR = Path(__file__).resolve().parent


class _AcadosTrackMpcShim:
    """Minimal holder so ``crocoddyl_closed_loop_to_ee_tracking_res`` can reuse EE plots."""

    def __init__(self, robot_model: pin.Model, ee_frame_id: int):
        self.robot_model = robot_model
        self._planner = type("_PlannerShim", (), {"ee_frame_id": int(ee_frame_id)})()


def croc_tracking_weights_to_W_R(
    *,
    w_pos: float = 1.0,
    w_att: float = 1.0,
    w_joint: float = 1.0,
    w_vel: float = 1.0,
    w_omega: float = 1.0,
    w_joint_vel: float = 1.0,
    w_control: float = 1e-3,
    w_u_thrust: float = 1.0,
    w_u_joint_torque: float = 1.0,
    w_state_track: float = 10.0,
    w_terminal_track: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map Crocoddyl full-state tracking GUI weights to Acados NONLINEAR_LS W / W_e / R.

    State cost layout matches ``s500_uam_acados_trajectory._state_to_cost_ref``:
    [pos(3), yaw, roll, pitch, j1, j2, v_lin(3), ω(3), j̇(2)].
    """
    s = float(w_state_track)
    W_state = np.diag(
        [
            float(w_pos) * s,
            float(w_pos) * s,
            float(w_pos) * s,
            float(w_att) * s,
            float(w_att) * s,
            float(w_att) * s,
            float(w_joint) * s,
            float(w_joint) * s,
            float(w_vel) * s,
            float(w_vel) * s,
            float(w_vel) * s,
            float(w_omega) * s,
            float(w_omega) * s,
            float(w_omega) * s,
            float(w_joint_vel) * s,
            float(w_joint_vel) * s,
        ]
    )
    r_thrust = float(w_control) * float(w_u_thrust)
    r_torque = float(w_control) * float(w_u_joint_torque) * 10000.0
    R = np.diag([r_thrust] * 4 + [r_torque] * 2)
    W_e = W_state * float(w_terminal_track)
    return W_state, R, W_e


def create_full_state_tracking_mpc_solver(
    N: int,
    dt: float,
    *,
    w_pos: float = 1.0,
    w_att: float = 1.0,
    w_joint: float = 1.0,
    w_vel: float = 1.0,
    w_omega: float = 1.0,
    w_joint_vel: float = 1.0,
    w_control: float = 1e-3,
    w_u_thrust: float = 1.0,
    w_u_joint_torque: float = 1.0,
    w_state_track: float = 10.0,
    w_terminal_track: float = 100.0,
    max_iter: int = 40,
    control_mode: str = "direct",
) -> tuple:
    if not ACADOS_AVAILABLE:
        raise ImportError("acados_template not installed")
    if not PINOCCHIO_AVAILABLE:
        raise ImportError(f"pinocchio/casadi required: {_pin_err}")
    if not DEPS_OK:
        raise ImportError(f"project deps: {_deps_err}")

    if control_mode == "direct":
        acados_model, pin_model, nq, nv, nu = build_acados_model()
        code_subdir = "s500_uam_full_state_track_mpc"
    elif control_mode == "actuator_first_order":
        from s500_uam_acados_model import build_acados_model_actuator_first_order

        acados_model, pin_model, nq, nv, nu, _meta = build_acados_model_actuator_first_order()
        code_subdir = "s500_uam_full_state_track_mpc_act1"
    else:
        raise ValueError("control_mode must be 'direct' or 'actuator_first_order'")

    W_state, R, W_e = croc_tracking_weights_to_W_R(
        w_pos=w_pos,
        w_att=w_att,
        w_joint=w_joint,
        w_vel=w_vel,
        w_omega=w_omega,
        w_joint_vel=w_joint_vel,
        w_control=w_control,
        w_u_thrust=w_u_thrust,
        w_u_joint_torque=w_u_joint_torque,
        w_state_track=w_state_track,
        w_terminal_track=w_terminal_track,
    )

    ocp = AcadosOcp()
    ocp.model = acados_model
    nx = nq + nv + (6 if control_mode == "actuator_first_order" else 0)

    x = ocp.model.x
    quat = x[3:7]
    roll, pitch, yaw = _quat_to_euler_zyx_ca(quat)
    cost_y = ca.vertcat(
        x[0:3], yaw, roll, pitch, x[7:9], x[9:17], ocp.model.u
    )
    cost_y_e = ca.vertcat(x[0:3], yaw, roll, pitch, x[7:9], x[9:17])

    ocp.model.cost_y_expr = cost_y
    ocp.model.cost_y_expr_e = cost_y_e
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))
    ocp.cost.W_e = W_e
    ny = int(cost_y.shape[0])
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(cost_y_e.shape[0])

    ocp.dims.N = int(N)
    ocp.solver_options.tf = float(N) * float(dt)
    ocp.solver_options.nlp_solver_max_iter = int(max_iter)
    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = int(N)

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
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx = robot_lbx
        ocp.constraints.ubx = robot_ubx
    else:
        ocp.constraints.lbu = np.array(
            [-2.0 * om_max, -2.0 * om_max, -2.0 * om_max, 4.0 * min_thrust, -j_max, -j_max]
        )
        ocp.constraints.ubu = np.array(
            [2.0 * om_max, 2.0 * om_max, 2.0 * om_max, 4.0 * max_thrust, j_max, j_max]
        )
        uact_lbx = np.concatenate([np.full(4, min_thrust), np.full(2, -2.0)])
        uact_ubx = np.concatenate([np.full(4, max_thrust), np.full(2, 2.0)])
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
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx = np.concatenate([robot_lbx, uact_lbx])
        ocp.constraints.ubx = np.concatenate([robot_ubx, uact_ubx])

    ocp.constraints.idxbu = np.arange(nu)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = (
        "ERK" if control_mode == "actuator_first_order" else "IRK"
    )
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0
    if hasattr(ocp.solver_options, "qp_solver_iter_max"):
        ocp.solver_options.qp_solver_iter_max = max(50, int(max_iter) * 2)

    code_export_dir = REPO_ROOT / "c_generated_code" / code_subdir
    code_export_dir.mkdir(parents=True, exist_ok=True)
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(code_export_dir / "ocp.json")
    ocp.constraints.x0 = np.zeros(nx)

    solver = AcadosOcpSolver(ocp, json_file=str(code_export_dir / "ocp.json"))
    return solver, acados_model, pin_model, nq, nv, nu


def _quat_to_euler_zyx_ca(quat):
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    roll = ca.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = 2 * (qw * qy - qz * qx)
    sinp = ca.fmin(1, ca.fmax(-1, sinp))
    pitch = ca.asin(sinp)
    yaw = ca.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


def _state_to_yref(x_ref: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    return np.concatenate([_state_to_cost_ref(x_ref), np.asarray(u_ref, dtype=float).flatten()])


def _state_to_yref_e(x_ref: np.ndarray) -> np.ndarray:
    return _state_to_cost_ref(x_ref)


def run_closed_loop_track_full_state_plan_acados(
    x0: np.ndarray,
    t_plan: np.ndarray,
    x_plan: np.ndarray,
    T_sim: float,
    sim_dt: float,
    control_dt: float,
    dt_mpc: float,
    N: int,
    *,
    w_pos: float = 1.0,
    w_att: float = 1.0,
    w_joint: float = 1.0,
    w_vel: float = 1.0,
    w_omega: float = 1.0,
    w_joint_vel: float = 1.0,
    w_control: float = 1e-3,
    w_u_thrust: float = 1.0,
    w_u_joint_torque: float = 1.0,
    w_state_track: float = 10.0,
    w_terminal_track: float = 100.0,
    mpc_max_iter: int = 40,
    mpc_log_interval: int = 0,
    control_mode: str = "direct",
    urdf_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Receding-horizon Acados tracking of ``(t_plan, x_plan)`` with RK4 plant."""
    t_plan = np.asarray(t_plan, dtype=float).flatten()
    x_plan = np.asarray(x_plan, dtype=float)
    if x_plan.ndim != 2 or len(t_plan) != len(x_plan):
        raise ValueError("x_plan must be 2D with same length as t_plan")

    solver, acados_model, pin_model, nq, nv, nu = create_full_state_tracking_mpc_solver(
        N,
        dt_mpc,
        w_pos=w_pos,
        w_att=w_att,
        w_joint=w_joint,
        w_vel=w_vel,
        w_omega=w_omega,
        w_joint_vel=w_joint_vel,
        w_control=w_control,
        w_u_thrust=w_u_thrust,
        w_u_joint_torque=w_u_joint_torque,
        w_state_track=w_state_track,
        w_terminal_track=w_terminal_track,
        max_iter=mpc_max_iter,
        control_mode=control_mode,
    )

    f_fun = _make_f_expl_fun(acados_model)
    plant = CasadiRK4Plant(f_fun, sim_dt, nu)
    n_robot = nq + nv
    nx = n_robot + (6 if control_mode == "actuator_first_order" else 0)

    sim_dt = float(sim_dt)
    control_dt = float(control_dt)
    if control_dt < sim_dt - 1e-15:
        raise ValueError("control_dt must be >= sim_dt")
    mpc_stride = max(1, int(round(control_dt / sim_dt)))

    cfg = load_s500_config()
    plat = cfg["platform"]
    if control_mode == "actuator_first_order":
        from s500_uam_acados_model import nominal_command_hover

        u_hover = nominal_command_hover(
            pin_model, np.asarray(x0, dtype=float).flatten()[:n_robot],
            plat["min_thrust"], plat["max_thrust"],
        )
    else:
        u_hover = hover_thrust_controls(
            pin_model, nu, plat["min_thrust"], plat["max_thrust"]
        )

    n_steps = max(1, int(round(float(T_sim) / sim_dt)))
    t_log = np.zeros(n_steps + 1, dtype=float)
    x_log = np.zeros((n_steps + 1, nx), dtype=float)
    u_log = np.zeros((n_steps, nu), dtype=float)

    x = np.asarray(x0, dtype=float).flatten().copy()
    if x.size < nx:
        x_pad = np.zeros(nx, dtype=float)
        x_pad[: x.size] = x
        x = x_pad
    t_log[0] = 0.0
    x_log[0] = x

    x_prev: Optional[List[np.ndarray]] = None
    u_prev: Optional[List[np.ndarray]] = None
    mpc_solve_steps: List[int] = []
    mpc_iters: List[int] = []
    mpc_wall_s: List[float] = []
    u_apply = u_hover.copy()

    for k in range(n_steps):
        t_k = k * sim_dt
        plant.on_pre_step(t_k, k)
        do_mpc = k % mpc_stride == 0

        if do_mpc:
            if k == 0 or x_prev is None:
                x_roll = rollout_nominal_trajectory(f_fun, x, u_hover, dt_mpc, N, nq)
                set_solver_initial_guess(solver, x, x_roll, u_hover, N)
            else:
                shift_solver_initial_guess(solver, x, x_prev, u_prev, N)

            solver.constraints_set(0, "lbx", x, api="new")
            solver.constraints_set(0, "ubx", x, api="new")

            for i in range(N):
                ti = t_k + i * dt_mpc
                xr = interp_full_state_piecewise(ti, t_plan, x_plan, pin_model)
                yref = _state_to_yref(xr, u_hover)
                solver.cost_set(i, "yref", yref, api="new")
            xrN = interp_full_state_piecewise(t_k + N * dt_mpc, t_plan, x_plan, pin_model)
            solver.cost_set(N, "yref", _state_to_yref_e(xrN), api="new")

            t0 = time.perf_counter()
            status = int(solver.solve())
            wall_s = time.perf_counter() - t0
            n_iter = _acados_nlp_iterations(solver)

            mpc_solve_steps.append(k)
            mpc_iters.append(n_iter)
            mpc_wall_s.append(wall_s)

            if mpc_log_interval > 0 and len(mpc_solve_steps) % mpc_log_interval == 0:
                print(
                    f"[acados track] t={t_k:.3f} status={status} iter={n_iter} wall={wall_s*1e3:.1f} ms"
                )

            x_prev = [solver.get(i, "x") for i in range(N + 1)]
            u_prev = [solver.get(i, "u") for i in range(N)]
            u_apply = np.asarray(solver.get(0, "u"), dtype=float).flatten().copy()

        if k < n_steps:
            u_log[k] = u_apply
            x = plant.step(x, u_apply)
            t_log[k + 1] = (k + 1) * sim_dt
            x_log[k + 1] = x

    ee_id = int(pin_model.getFrameId(EE_FRAME_NAME))
    shim = _AcadosTrackMpcShim(pin_model, ee_id)

    return {
        "track_mode": "full_state_trajectory",
        "t_plan": t_plan,
        "x_plan": x_plan,
        "time": t_log,
        "states": x_log[:, :n_robot],
        "controls": u_log,
        "n_inner": mpc_stride,
        "mpc_solve_steps": mpc_solve_steps,
        "mpc_iters": mpc_iters,
        "mpc_wall_s": mpc_wall_s,
        "mpc": shim,
        "control_mode": control_mode,
        "dt_mpc": dt_mpc,
        "horizon": N,
    }


def acados_closed_loop_to_ee_tracking_res(out: Dict[str, Any]) -> Dict[str, Any]:
    from s500_uam_crocoddyl_state_tracking_mpc import crocoddyl_closed_loop_to_ee_tracking_res

    return crocoddyl_closed_loop_to_ee_tracking_res(out)
