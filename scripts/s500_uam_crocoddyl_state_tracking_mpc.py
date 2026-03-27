#!/usr/bin/env python3
"""
S500 UAM — Crocoddyl full-state tracking MPC (constant reference)

Running cost: state tracking (x_ref) + state regularization (x_nom) + control regularization (u_ref ≈ hover thrust)
Terminal cost: state tracking only to x_ref

Simulation: similar to run_numeric_sim; sim_dt integration, control_dt applies ZOH to the MPC outputs; forward uses
IntegratedActionModelEuler (same FreeFwdDynamics as MPC).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pinocchio as pin
import crocoddyl

from s500_uam_trajectory_planner import (
    S500UAMTrajectoryPlanner,
    compute_ee_kinematics_along_trajectory,
    make_uam_state,
)


def _apply_first_order_actuator(
    u_act: np.ndarray,
    u_cmd: np.ndarray,
    *,
    tau_thrust: float,
    tau_theta: float,
    dt: float,
) -> np.ndarray:
    """
    First-order actuator discrete update (Euler):
      du/dt = (u_cmd - u_act) / tau
    Here thrust (T1..T4) and joint torques (tau1..tau2) use different tau values.
    """
    u_act = np.asarray(u_act, dtype=float).reshape(-1)
    u_cmd = np.asarray(u_cmd, dtype=float).reshape(-1)
    if u_act.shape != u_cmd.shape:
        raise ValueError(f"u_act/u_cmd shape mismatch: {u_act.shape} vs {u_cmd.shape}")
    if u_act.size < 6:
        raise ValueError(f"expect at least 6 control dims (T1..T4, tau1..tau2), got {u_act.size}")

    out = u_act.copy()
    if tau_thrust is not None and tau_thrust > 0:
        out[:4] = out[:4] + dt * (u_cmd[:4] - out[:4]) / float(tau_thrust)
    else:
        out[:4] = u_cmd[:4]

    if tau_theta is not None and tau_theta > 0:
        out[4:6] = out[4:6] + dt * (u_cmd[4:6] - out[4:6]) / float(tau_theta)
    else:
        out[4:6] = u_cmd[4:6]

    return out


def interp_full_state_piecewise(
    t_query: float,
    t_nodes: np.ndarray,
    x_nodes: np.ndarray,
    robot_model: pin.Model,
) -> np.ndarray:
    """Piecewise geodesic interp in q, linear in v, clamped to [t_nodes[0], t_nodes[-1]]."""
    t_nodes = np.asarray(t_nodes, dtype=float).flatten()
    x_nodes = np.asarray(x_nodes, dtype=float)
    if len(t_nodes) < 2 or len(x_nodes) < 2:
        return np.asarray(x_nodes[0], dtype=float).flatten()
    tq = float(np.clip(t_query, t_nodes[0], t_nodes[-1]))
    idx = int(np.searchsorted(t_nodes, tq, side="right"))
    i = max(1, min(idx, len(t_nodes) - 1))
    t0, t1 = float(t_nodes[i - 1]), float(t_nodes[i])
    alpha = (tq - t0) / (t1 - t0) if t1 > t0 + 1e-15 else 0.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    nq = robot_model.nq
    nv = robot_model.nv
    qa = x_nodes[i - 1][:nq]
    qb = x_nodes[i][:nq]
    va = x_nodes[i - 1][nq : nq + nv]
    vb = x_nodes[i][nq : nq + nv]
    q = pin.interpolate(robot_model, qa, qb, alpha)
    v = (1.0 - alpha) * va + alpha * vb
    return np.concatenate([q, v])


class UAMCrocoddylStateTrackingMPC:
    """Build a receding-horizon Box-FDDP MPC under a constant x_ref."""

    def __init__(
        self,
        s500_yaml_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        dt_mpc: float = 0.05,
        horizon: int = 25,
        w_state_track: float = 10.0,
        w_state_reg: float = 0.1,
        w_control: float = 1e-3,
        w_terminal_track: float = 100.0,
        use_thrust_constraints: bool = True,
    ):
        self.dt_mpc = float(dt_mpc)
        self.horizon = int(horizon)
        self.w_state_track = float(w_state_track)
        self.w_state_reg = float(w_state_reg)
        self.w_control = float(w_control)
        self.w_terminal_track = float(w_terminal_track)
        self.use_thrust_constraints = bool(use_thrust_constraints)

        self._planner = S500UAMTrajectoryPlanner(
            s500_yaml_path=s500_yaml_path, urdf_path=urdf_path
        )
        self.state = self._planner.state
        self.actuation = self._planner.actuation
        self.robot_model = self._planner.robot_model
        self.s500_config = self._planner.s500_config
        self.nu = self.actuation.nu

        mass = self.robot_model.inertias[1].mass
        self._hover_thrust = mass * 9.81 / 4.0
        self._u_ref = np.array([self._hover_thrust] * 4 + [0.0] * (self.nu - 4))

        if use_thrust_constraints:
            p = self.s500_config["platform"]
            self._u_lb = np.array([p["min_thrust"]] * 4 + [-2.0] * 2)
            self._u_ub = np.array([p["max_thrust"]] * 4 + [2.0] * 2)
        else:
            self._u_lb = -1e6 * np.ones(self.nu)
            self._u_ub = 1e6 * np.ones(self.nu)

    def _make_running_cost(self, x_ref: np.ndarray, x_nom: np.ndarray) -> crocoddyl.CostModelSum:
        nu = self.nu
        c = crocoddyl.CostModelSum(self.state, nu)
        act_x = crocoddyl.ActivationModelQuad(self.state.ndx)
        c.addCost(
            "x_track",
            crocoddyl.CostModelResidual(
                self.state,
                act_x,
                crocoddyl.ResidualModelState(self.state, np.asarray(x_ref, dtype=float), nu),
            ),
            self.w_state_track,
        )
        c.addCost(
            "x_reg",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelQuad(self.state.ndx),
                crocoddyl.ResidualModelState(self.state, np.asarray(x_nom, dtype=float), nu),
            ),
            self.w_state_reg,
        )
        act_u = crocoddyl.ActivationModelQuad(nu)
        c.addCost(
            "u_reg",
            crocoddyl.CostModelResidual(
                self.state,
                act_u,
                crocoddyl.ResidualModelControl(self.state, self._u_ref.copy()),
            ),
            self.w_control,
        )
        return c

    def _make_terminal_cost(self, x_ref: np.ndarray) -> crocoddyl.CostModelSum:
        nu = self.nu
        c = crocoddyl.CostModelSum(self.state, nu)
        c.addCost(
            "x_track_term",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelQuad(self.state.ndx),
                crocoddyl.ResidualModelState(self.state, np.asarray(x_ref, dtype=float), nu),
            ),
            self.w_terminal_track,
        )
        return c

    def _make_integrated_running(self, x_ref: np.ndarray, x_nom: np.ndarray):
        cost = self._make_running_cost(x_ref, x_nom)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost
        )
        inte = crocoddyl.IntegratedActionModelEuler(diff, self.dt_mpc)
        inte.u_lb = self._u_lb.copy()
        inte.u_ub = self._u_ub.copy()
        return inte

    def _make_integrated_terminal(self, x_ref: np.ndarray):
        cost = self._make_terminal_cost(x_ref)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost
        )
        return crocoddyl.IntegratedActionModelEuler(diff, 0.0)

    def build_shooting_problem(
        self, x0: np.ndarray, x_ref: np.ndarray, x_nom: np.ndarray
    ) -> crocoddyl.ShootingProblem:
        x0 = np.asarray(x0, dtype=float).flatten()
        running = [
            self._make_integrated_running(x_ref, x_nom) for _ in range(self.horizon)
        ]
        term = self._make_integrated_terminal(x_ref)
        return crocoddyl.ShootingProblem(x0, running, term)

    def build_shooting_problem_along_plan(
        self,
        x0: np.ndarray,
        x_nom: np.ndarray,
        t_start: float,
        t_plan: np.ndarray,
        x_plan: np.ndarray,
    ) -> crocoddyl.ShootingProblem:
        """Horizon nodes track states sampled from a time-parameterized full-state plan."""
        x0 = np.asarray(x0, dtype=float).flatten()
        running = []
        for k in range(self.horizon):
            tk = t_start + k * self.dt_mpc
            xrk = interp_full_state_piecewise(tk, t_plan, x_plan, self.robot_model)
            running.append(self._make_integrated_running(xrk, x_nom))
        tkN = t_start + self.horizon * self.dt_mpc
        xN = interp_full_state_piecewise(tkN, t_plan, x_plan, self.robot_model)
        term = self._make_integrated_terminal(xN)
        return crocoddyl.ShootingProblem(x0, running, term)

    def make_sim_integrator(self, sim_dt: float, x_ref: np.ndarray, x_nom: np.ndarray):
        """Same dynamics as MPC; integration step is sim_dt (the cost is only a placeholder, and forward does not depend on its value)."""
        cost = self._make_running_cost(x_ref, x_nom)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost
        )
        inte = crocoddyl.IntegratedActionModelEuler(diff, float(sim_dt))
        return inte, inte.createData()

    def integrate_one(
        self, sim_int: crocoddyl.IntegratedActionModelEuler, sim_data, x: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float).flatten()
        u = np.asarray(u, dtype=float).flatten()
        sim_int.calc(sim_data, x, u)
        return np.array(sim_data.xnext, dtype=float).copy()


def run_closed_loop_state_tracking(
    x0: np.ndarray,
    x_ref: np.ndarray,
    x_nom: np.ndarray,
    T_sim: float,
    sim_dt: float,
    control_dt: float,
    dt_mpc: float,
    horizon: int,
    w_state_track: float = 10.0,
    w_state_reg: float = 0.1,
    w_control: float = 1e-3,
    w_terminal_track: float = 100.0,
    mpc_max_iter: int = 60,
    use_thrust_constraints: bool = True,
    use_actuator_first_order: bool = False,
    tau_thrust: float = 0.06,
    tau_theta: float = 0.05,
    s500_yaml_path: Optional[str] = None,
    urdf_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Closed-loop simulation: solve the MPC once every control_dt with ZOH in between; propagate the state using sim_dt forward integration.
    """
    mpc = UAMCrocoddylStateTrackingMPC(
        s500_yaml_path=s500_yaml_path,
        urdf_path=urdf_path,
        dt_mpc=dt_mpc,
        horizon=horizon,
        w_state_track=w_state_track,
        w_state_reg=w_state_reg,
        w_control=w_control,
        w_terminal_track=w_terminal_track,
        use_thrust_constraints=use_thrust_constraints,
    )

    sim_dt = float(sim_dt)
    control_dt = float(control_dt)
    if sim_dt <= 0 or control_dt <= 0:
        raise ValueError("sim_dt and control_dt must be positive")
    n_inner = max(1, int(round(control_dt / sim_dt)))
    err_inner = abs(n_inner * sim_dt - control_dt)
    if err_inner > 0.1 * sim_dt:
        n_inner = max(1, int(np.ceil(control_dt / sim_dt)))

    sim_int, sim_data = mpc.make_sim_integrator(sim_dt, x_ref, x_nom)

    n_total = max(1, int(np.ceil(T_sim / sim_dt)))
    t_arr = np.arange(n_total, dtype=float) * sim_dt

    x = np.asarray(x0, dtype=float).copy().flatten()
    time_data: List[float] = []
    state_data: List[np.ndarray] = []
    ctrl_data: List[np.ndarray] = []
    mpc_costs: List[float] = []
    mpc_iters: List[int] = []
    mpc_solve_t: List[float] = []
    mpc_solve_steps: List[int] = []
    mpc_wall_s: List[float] = []
    track_norm: List[float] = []

    xs_guess: Optional[List[np.ndarray]] = None
    us_guess: Optional[List[np.ndarray]] = None

    u_cmd_hold = mpc._u_ref.copy()
    u_act = u_cmd_hold.copy()
    nq = mpc.robot_model.nq

    for step in range(n_total):
        t = step * sim_dt
        time_data.append(t)
        state_data.append(x.copy())
        # ctrl_data records the actuator's "actual applied input" u_act (after first-order response)

        dq = pin.difference(mpc.robot_model, x_ref[:nq], x[:nq])
        dv = x[nq:] - x_ref[nq:]
        track_norm.append(float(np.linalg.norm(np.concatenate([dq, dv]))))

        if step % n_inner == 0:
            prob = mpc.build_shooting_problem(x, x_ref, x_nom)
            solver = crocoddyl.SolverBoxFDDP(prob)
            solver.convergence_init = 1e-9
            solver.convergence_stop = 1e-7
            try:
                solver.setCallbacks([])
            except Exception:
                pass

            if xs_guess is None:
                xs_init = [x.copy() for _ in range(horizon + 1)]
                us_init = [mpc._u_ref.copy() for _ in range(horizon)]
            else:
                xs_init = xs_guess
                us_init = us_guess

            t_solve0 = time.perf_counter()
            converged = solver.solve(xs_init, us_init, mpc_max_iter)
            wall_s = time.perf_counter() - t_solve0
            if verbose and step % (5 * n_inner) == 0:
                print(
                    f"t={t:.3f} converged={converged} cost={solver.cost:.4f} iters={solver.iter}"
                )

            u_cmd_hold = np.array(solver.us[0], dtype=float).copy()
            mpc_costs.append(float(solver.cost))
            mpc_iters.append(int(solver.iter))
            mpc_solve_t.append(t)
            mpc_solve_steps.append(step)
            mpc_wall_s.append(float(wall_s))

            xs_guess = [solver.xs[i + 1].copy() for i in range(horizon)] + [
                solver.xs[-1].copy()
            ]
            xs_guess[0] = x.copy()
            us_guess = [solver.us[i + 1].copy() for i in range(horizon - 1)] + [
                solver.us[-1].copy()
            ]

        # Update the first-order actuator response (ZOH: u_cmd_hold stays constant within n_inner)
        if use_actuator_first_order:
            u_act = _apply_first_order_actuator(
                u_act,
                u_cmd_hold,
                tau_thrust=tau_thrust,
                tau_theta=tau_theta,
                dt=sim_dt,
            )
        else:
            u_act = u_cmd_hold.copy()

        ctrl_data.append(u_act.copy())

        if step < n_total - 1:
            x = mpc.integrate_one(sim_int, sim_data, x, u_act)

    return {
        "time": np.array(time_data),
        "states": np.array(state_data),
        "controls": np.array(ctrl_data),
        "x_ref": np.asarray(x_ref, dtype=float).copy(),
        "x_nom": np.asarray(x_nom, dtype=float).copy(),
        "track_norm": np.array(track_norm),
        "mpc_costs": np.array(mpc_costs),
        "mpc_iters": np.array(mpc_iters, dtype=int),
        "mpc_solve_t": np.array(mpc_solve_t),
        "mpc_solve_steps": np.array(mpc_solve_steps, dtype=int),
        "mpc_wall_s": np.array(mpc_wall_s, dtype=float),
        "sim_dt": sim_dt,
        "control_dt": control_dt,
        "n_inner": n_inner,
        "mpc": mpc,
    }


def run_closed_loop_track_full_state_plan(
    x0: np.ndarray,
    t_plan: np.ndarray,
    x_plan: np.ndarray,
    x_nom: np.ndarray,
    T_sim: float,
    sim_dt: float,
    control_dt: float,
    dt_mpc: float,
    horizon: int,
    w_state_track: float = 10.0,
    w_state_reg: float = 0.1,
    w_control: float = 1e-3,
    w_terminal_track: float = 100.0,
    mpc_max_iter: int = 60,
    use_thrust_constraints: bool = True,
    use_actuator_first_order: bool = False,
    tau_thrust: float = 0.06,
    tau_theta: float = 0.05,
    s500_yaml_path: Optional[str] = None,
    urdf_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Track a planned full-state trajectory (t_plan, x_plan) with Crocoddyl receding-horizon tracking; the reference within the horizon is sampled with a time shift.
    """
    t_plan = np.asarray(t_plan, dtype=float).flatten()
    x_plan = np.asarray(x_plan, dtype=float)
    if x_plan.ndim != 2 or len(t_plan) != len(x_plan):
        raise ValueError("x_plan must be 2D with same length as t_plan")

    mpc = UAMCrocoddylStateTrackingMPC(
        s500_yaml_path=s500_yaml_path,
        urdf_path=urdf_path,
        dt_mpc=dt_mpc,
        horizon=horizon,
        w_state_track=w_state_track,
        w_state_reg=w_state_reg,
        w_control=w_control,
        w_terminal_track=w_terminal_track,
        use_thrust_constraints=use_thrust_constraints,
    )

    sim_dt = float(sim_dt)
    control_dt = float(control_dt)
    n_inner = max(1, int(round(control_dt / sim_dt)))
    if abs(n_inner * sim_dt - control_dt) > 0.1 * sim_dt:
        n_inner = max(1, int(np.ceil(control_dt / sim_dt)))

    t_mid = 0.5 * (float(t_plan[0]) + float(t_plan[-1]))
    x_mid = interp_full_state_piecewise(t_mid, t_plan, x_plan, mpc.robot_model)
    sim_int, sim_data = mpc.make_sim_integrator(sim_dt, x_mid, x_nom)

    n_total = max(1, int(np.ceil(T_sim / sim_dt)))
    x = np.asarray(x0, dtype=float).copy().flatten()
    time_data: List[float] = []
    state_data: List[np.ndarray] = []
    ctrl_data: List[np.ndarray] = []
    mpc_costs: List[float] = []
    mpc_iters: List[int] = []
    mpc_solve_t: List[float] = []
    mpc_solve_steps: List[int] = []
    mpc_wall_s: List[float] = []
    track_norm: List[float] = []

    xs_guess: Optional[List[np.ndarray]] = None
    us_guess: Optional[List[np.ndarray]] = None
    u_cmd_hold = mpc._u_ref.copy()
    u_act = u_cmd_hold.copy()
    nq = mpc.robot_model.nq

    for step in range(n_total):
        t = step * sim_dt
        time_data.append(t)
        state_data.append(x.copy())

        xr = interp_full_state_piecewise(t, t_plan, x_plan, mpc.robot_model)
        dq = pin.difference(mpc.robot_model, xr[:nq], x[:nq])
        dv = x[nq:] - xr[nq:]
        track_norm.append(float(np.linalg.norm(np.concatenate([dq, dv]))))

        if step % n_inner == 0:
            prob = mpc.build_shooting_problem_along_plan(
                x, x_nom, t, t_plan, x_plan
            )
            solver = crocoddyl.SolverBoxFDDP(prob)
            solver.convergence_init = 1e-9
            solver.convergence_stop = 1e-7
            try:
                solver.setCallbacks([])
            except Exception:
                pass

            if xs_guess is None:
                xs_init = [x.copy() for _ in range(horizon + 1)]
                us_init = [mpc._u_ref.copy() for _ in range(horizon)]
            else:
                xs_init = xs_guess
                us_init = us_guess

            t_solve0 = time.perf_counter()
            converged = solver.solve(xs_init, us_init, mpc_max_iter)
            wall_s = time.perf_counter() - t_solve0
            if verbose and step % (5 * n_inner) == 0:
                print(
                    f"t={t:.3f} converged={converged} cost={solver.cost:.4f} iters={solver.iter}"
                )

            u_cmd_hold = np.array(solver.us[0], dtype=float).copy()
            mpc_costs.append(float(solver.cost))
            mpc_iters.append(int(solver.iter))
            mpc_solve_t.append(t)
            mpc_solve_steps.append(step)
            mpc_wall_s.append(float(wall_s))

            xs_guess = [solver.xs[i + 1].copy() for i in range(horizon)] + [
                solver.xs[-1].copy()
            ]
            xs_guess[0] = x.copy()
            us_guess = [solver.us[i + 1].copy() for i in range(horizon - 1)] + [
                solver.us[-1].copy()
            ]

        # Update the first-order actuator response (ZOH: u_cmd_hold stays constant within n_inner)
        if use_actuator_first_order:
            u_act = _apply_first_order_actuator(
                u_act,
                u_cmd_hold,
                tau_thrust=tau_thrust,
                tau_theta=tau_theta,
                dt=sim_dt,
            )
        else:
            u_act = u_cmd_hold.copy()

        ctrl_data.append(u_act.copy())

        if step < n_total - 1:
            x = mpc.integrate_one(sim_int, sim_data, x, u_act)

    return {
        "time": np.array(time_data),
        "states": np.array(state_data),
        "controls": np.array(ctrl_data),
        "t_plan": t_plan.copy(),
        "x_plan": x_plan.copy(),
        "x_nom": np.asarray(x_nom, dtype=float).copy(),
        "track_mode": "full_state_trajectory",
        "track_norm": np.array(track_norm),
        "mpc_costs": np.array(mpc_costs),
        "mpc_iters": np.array(mpc_iters, dtype=int),
        "mpc_solve_t": np.array(mpc_solve_t),
        "mpc_solve_steps": np.array(mpc_solve_steps, dtype=int),
        "mpc_wall_s": np.array(mpc_wall_s, dtype=float),
        "sim_dt": sim_dt,
        "control_dt": control_dt,
        "n_inner": n_inner,
        "mpc": mpc,
    }


def crocoddyl_closed_loop_to_ee_tracking_res(out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the output of ``run_closed_loop_state_tracking`` or ``run_closed_loop_track_full_state_plan`` into
    the required ``res`` structure for ``s500_uam_ee_snap_tracking_mpc.render_ee_tracking_results_to_figures``.

    If ``out["track_mode"] == "full_state_trajectory"``, then ``p_ref`` / ``yaw_ref`` are interpolated along ``t_plan``/``x_plan``
    to match the planned reference.
    """
    mpc = out["mpc"]
    t = np.asarray(out["time"], dtype=float).flatten()
    X = np.asarray(out["states"], dtype=float)
    U = np.asarray(out["controls"], dtype=float)
    n_t = len(t)
    n_inner = int(out["n_inner"])

    data = mpc.robot_model.createData()
    fid = mpc._planner.ee_frame_id
    ee_pos, _, ee_rpy, _ = compute_ee_kinematics_along_trajectory(
        X, mpc.robot_model, data, fid
    )

    if out.get("track_mode") == "full_state_trajectory":
        t_plan = np.asarray(out["t_plan"], dtype=float).flatten()
        x_plan = np.asarray(out["x_plan"], dtype=float)
        Xr = np.array(
            [
                interp_full_state_piecewise(float(ti), t_plan, x_plan, mpc.robot_model)
                for ti in t
            ]
        )
        p_ref, _, prpy_ref, _ = compute_ee_kinematics_along_trajectory(
            Xr, mpc.robot_model, data, fid
        )
        yaw_ref = prpy_ref[:, 2].astype(float)
    else:
        x_ref = np.asarray(out["x_ref"], dtype=float).flatten()
        xr = x_ref.reshape(1, -1)
        pref0, _, prpy0, _ = compute_ee_kinematics_along_trajectory(
            xr, mpc.robot_model, data, fid
        )
        p_ref = np.repeat(pref0, n_t, axis=0)
        yaw_ref_val = float(prpy0[0, 2])
        yaw_ref = np.full(n_t, yaw_ref_val, dtype=float)

    err = np.linalg.norm(ee_pos - p_ref, axis=1)

    ee_yaw = ee_rpy[:, 2].astype(float)
    err_yaw = ee_yaw - yaw_ref
    err_yaw = (err_yaw + np.pi) % (2.0 * np.pi) - np.pi

    n_mpc = max(0, n_t - 1)
    nit = np.zeros(n_mpc, dtype=int)
    wall = np.zeros(n_mpc, dtype=float)
    stat = np.zeros(n_mpc, dtype=int)
    steps = out.get("mpc_solve_steps")
    iters = out.get("mpc_iters")
    walls = out.get("mpc_wall_s")
    if steps is not None and iters is not None and len(steps) == len(iters):
        walls_arr = walls if walls is not None and len(walls) == len(iters) else None
        for j in range(len(steps)):
            st = int(steps[j])
            if n_mpc <= 0:
                break
            si = min(max(st, 0), n_mpc - 1)
            nit[si] = int(iters[j])
            if walls_arr is not None:
                wall[si] = float(walls_arr[j])
            stat[si] = 0

    return {
        "t": t,
        "x": X,
        "u": U,
        "ee": ee_pos,
        "p_ref": p_ref,
        "err": err,
        "ee_yaw": ee_yaw,
        "yaw_ref": yaw_ref,
        "err_yaw": err_yaw,
        "control_mode": "direct",
        "sim_dt": float(out["sim_dt"]),
        "control_dt": float(out["control_dt"]),
        "mpc_stride": n_inner,
        "mpc_solve": {
            "nlp_iter": nit,
            "cpu_s": wall.copy(),
            "wall_s": wall,
            "status": stat,
        },
    }


def default_hover_nominal() -> np.ndarray:
    """Nominal state used for weak regularization: body origin height 1 m, level attitude, joints 0, velocity 0."""
    x = make_uam_state(0.0, 0.0, 1.0, j1=0.0, j2=0.0, yaw=0.0)
    return x
