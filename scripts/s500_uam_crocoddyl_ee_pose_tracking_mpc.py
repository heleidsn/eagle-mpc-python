#!/usr/bin/env python3
"""
S500 UAM — Crocoddyl EE pose tracking (MPC)

EE-centric tracking uses a pose residual on the EE frame:
  - translation + rotation (SE3) via ``ResidualModelFramePlacement``
  - rolling-horizon MPC solved by ``crocoddyl.SolverBoxFDDP``

Control (direct mode):
  u = [T1, T2, T3, T4, tau1, tau2]

Reference provided as:
  - t_ref: (N,) seconds
  - p_ref: (N,3) EE position in world frame
  - yaw_ref: (N,) ZYX yaw angle in rad

For rotation target we assume roll/pitch = 0 and yaw = yaw_ref.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pinocchio as pin
import crocoddyl
import matplotlib.pyplot as plt

from s500_uam_trajectory_planner import (
    S500UAMTrajectoryPlanner,
    compute_ee_kinematics_along_trajectory,
    make_uam_state,
)


def _interp_ref_pose(
    tq: float,
    t_ref: np.ndarray,
    p_ref: np.ndarray,
    yaw_ref: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Linear interpolation for position and yaw (yaw is unwrapped first)."""
    t_ref = np.asarray(t_ref, dtype=float).flatten()
    p_ref = np.asarray(p_ref, dtype=float)
    yaw_ref = np.asarray(yaw_ref, dtype=float).flatten()

    if len(t_ref) < 2:
        raise ValueError("t_ref must have at least 2 points")
    if p_ref.shape != (len(t_ref), 3):
        raise ValueError(f"p_ref must have shape (len(t_ref),3); got {p_ref.shape}")
    if len(yaw_ref) != len(t_ref):
        raise ValueError("yaw_ref length mismatch with t_ref")

    tq = float(np.clip(tq, t_ref[0], t_ref[-1]))
    px = float(np.interp(tq, t_ref, p_ref[:, 0]))
    py = float(np.interp(tq, t_ref, p_ref[:, 1]))
    pz = float(np.interp(tq, t_ref, p_ref[:, 2]))

    yaw_u = np.unwrap(yaw_ref)
    yaw = float(np.interp(tq, t_ref, yaw_u))

    return np.array([px, py, pz], dtype=float), yaw


def _yaw_to_rotation_matrix(yaw: float, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """Construct a target rotation from roll/pitch/yaw (world ZYX yaw)."""
    return pin.rpy.rpyToMatrix(roll, pitch, yaw)


@dataclass
class EETrackingWeights:
    w_pos: float = 10.0
    w_rot_rp: float = 1.0   # roll/pitch rotation weights (small vs yaw)
    w_rot_yaw: float = 1.0
    w_vel_lin: float = 1.0
    w_vel_ang_rp: float = 1.0  # angular velocity x/y
    w_vel_ang_yaw: float = 1.0  # angular velocity z (yaw rate)
    w_u: float = 0
    w_terminal_scale: float = 3.0
    w_state_reg: float = 0.0  # optional; default off


class UAMEEPoseTrackingCrocoddylMPC:
    def __init__(
        self,
        *,
        s500_yaml_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        dt_mpc: float = 0.05,
        horizon: int = 25,
        u_weights: EETrackingWeights = EETrackingWeights(),
        use_thrust_constraints: bool = True,
    ):
        self.dt_mpc = float(dt_mpc)
        self.horizon = int(horizon)
        self.w = u_weights
        self.use_thrust_constraints = bool(use_thrust_constraints)

        self._planner = S500UAMTrajectoryPlanner(
            s500_yaml_path=s500_yaml_path,
            urdf_path=urdf_path,
        )
        self.state = self._planner.state
        self.actuation = self._planner.actuation
        self.robot_model = self._planner.robot_model
        self.robot_data = self._planner.robot_data
        self.ee_frame_id = self._planner.ee_frame_id

        self.nu = self.actuation.nu
        self.nq = self.robot_model.nq
        self.nv = self.robot_model.nv

        mass = float(self.robot_model.inertias[1].mass)
        hover_thrust = mass * 9.81 / 4.0
        self._u_ref = np.array([hover_thrust] * 4 + [0.0] * (self.nu - 4), dtype=float)

        if use_thrust_constraints:
            p = self._planner.s500_config["platform"]
            self._u_lb = np.array([p["min_thrust"]] * 4 + [-2.0] * 2, dtype=float)
            self._u_ub = np.array([p["max_thrust"]] * 4 + [2.0] * 2, dtype=float)
        else:
            self._u_lb = -1e6 * np.ones(self.nu, dtype=float)
            self._u_ub = 1e6 * np.ones(self.nu, dtype=float)

    def _make_running_cost(
        self,
        p_des: np.ndarray,
        yaw_des: float,
        v_lin_des: np.ndarray,
        w_ang_des: np.ndarray,
    ) -> crocoddyl.CostModelSum:
        nu = self.nu
        c = crocoddyl.CostModelSum(self.state, nu)

        R_des = _yaw_to_rotation_matrix(yaw_des)
        T_des = pin.SE3(R_des, np.asarray(p_des, dtype=float).reshape(3))

        # ResidualModelFramePlacement residual dimension is 6 (t + rot).
        pose_act = crocoddyl.ActivationModelWeightedQuad(
            np.array(
                [
                    self.w.w_pos,
                    self.w.w_pos,
                    self.w.w_pos,
                    self.w.w_rot_rp,
                    self.w.w_rot_rp,
                    self.w.w_rot_yaw,
                ],
                dtype=float,
            )
        )
        ee_res = crocoddyl.ResidualModelFramePlacement(self.state, self.ee_frame_id, T_des, nu)
        c.addCost(
            "ee_pose",
            crocoddyl.CostModelResidual(self.state, pose_act, ee_res),
            1.0,
        )

        # EE velocity tracking
        # Desired motion is (linear velocity; angular velocity) in world-aligned coordinates.
        if (
            self.w.w_vel_lin > 0
            or self.w.w_vel_ang_rp > 0
            or self.w.w_vel_ang_yaw > 0
        ):
            rf = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            vel_act = crocoddyl.ActivationModelWeightedQuad(
                np.array(
                    [
                        self.w.w_vel_lin,
                        self.w.w_vel_lin,
                        self.w.w_vel_lin,
                        self.w.w_vel_ang_rp,
                        self.w.w_vel_ang_rp,
                        self.w.w_vel_ang_yaw,
                    ],
                    dtype=float,
                )
            )
            v_lin_des = np.asarray(v_lin_des, dtype=float).reshape(3)
            w_ang_des = np.asarray(w_ang_des, dtype=float).reshape(3)
            vel_motion_ref = pin.Motion(v_lin_des, w_ang_des)

            vel_res = None
            # Some crocoddyl builds may provide a Tpl variant; fall back if not.
            if hasattr(crocoddyl, "ResidualModelFrameVelocityTpl"):
                try:
                    vel_res = crocoddyl.ResidualModelFrameVelocityTpl(
                        self.state, self.ee_frame_id, vel_motion_ref, rf, nu
                    )
                except Exception:
                    vel_res = None
            if vel_res is None:
                vel_res = crocoddyl.ResidualModelFrameVelocity(
                    self.state, self.ee_frame_id, vel_motion_ref, rf, nu
                )

            c.addCost(
                "ee_vel",
                crocoddyl.CostModelResidual(self.state, vel_act, vel_res),
                1.0,
            )

        # Control regularization around nominal hover command.
        if self.w.w_u > 0:
            u_act = crocoddyl.ActivationModelQuad(nu)
            u_res = crocoddyl.ResidualModelControl(self.state, self._u_ref.copy())
            c.addCost(
                "u_reg",
                crocoddyl.CostModelResidual(self.state, u_act, u_res),
                self.w.w_u,
            )

        # Optional state regularization (off by default).
        if self.w.w_state_reg > 0:
            x_nom = np.zeros(self.nq + self.nv, dtype=float)
            x_nom[2] = 1.0
            x_nom[6] = 1.0
            x_act_weights = np.ones(int(self.state.ndx), dtype=float)
            # Ignore velocity part: residual ordered as [q_diff, v_diff] in tangent.
            if self.nv > 0 and self.state.ndx >= self.nv:
                x_act_weights[-self.nv:] = 0.0
            x_act = crocoddyl.ActivationModelWeightedQuad(x_act_weights)
            x_res = crocoddyl.ResidualModelState(self.state, x_nom, nu)
            c.addCost(
                "x_reg",
                crocoddyl.CostModelResidual(self.state, x_act, x_res),
                self.w.w_state_reg,
            )

        return c

    def _make_terminal_cost(
        self,
        p_des: np.ndarray,
        yaw_des: float,
        v_lin_des: np.ndarray,
        w_ang_des: np.ndarray,
    ) -> crocoddyl.CostModelSum:
        # Reuse the same residuals; terminal scaling is handled by global weight multipliers
        # in practice (kept simple here).
        return self._make_running_cost(p_des, yaw_des, v_lin_des, w_ang_des)

    def _make_integrated_running(
        self,
        p_des: np.ndarray,
        yaw_des: float,
        v_lin_des: np.ndarray,
        w_ang_des: np.ndarray,
    ) -> crocoddyl.IntegratedActionModelEuler:
        cost = self._make_running_cost(p_des, yaw_des, v_lin_des, w_ang_des)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, cost)
        inte = crocoddyl.IntegratedActionModelEuler(diff, self.dt_mpc)
        inte.u_lb = self._u_lb.copy()
        inte.u_ub = self._u_ub.copy()
        return inte

    def _make_integrated_terminal(
        self,
        p_des: np.ndarray,
        yaw_des: float,
        v_lin_des: np.ndarray,
        w_ang_des: np.ndarray,
    ) -> crocoddyl.IntegratedActionModelEuler:
        # Terminal: same residual, larger weight.
        cost = self._make_running_cost(p_des, yaw_des, v_lin_des, w_ang_des)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, cost)
        terminal = crocoddyl.IntegratedActionModelEuler(diff, 0.0)
        return terminal

    def build_shooting_problem_along_ref(
        self,
        x0: np.ndarray,
        t_start: float,
        t_ref: np.ndarray,
        p_ref: np.ndarray,
        yaw_ref: np.ndarray,
        dp_ref: np.ndarray,
        dyaw_ref: np.ndarray,
    ) -> crocoddyl.ShootingProblem:
        x0 = np.asarray(x0, dtype=float).flatten()

        running: List[crocoddyl.IntegratedActionModelEuler] = []
        for k in range(self.horizon):
            tk = float(t_start + k * self.dt_mpc)
            p_des_k, yaw_des_k = _interp_ref_pose(tk, t_ref, p_ref, yaw_ref)
            v_lin_k = np.array(
                [
                    np.interp(tk, t_ref, dp_ref[:, 0]),
                    np.interp(tk, t_ref, dp_ref[:, 1]),
                    np.interp(tk, t_ref, dp_ref[:, 2]),
                ],
                dtype=float,
            )
            yaw_rate_k = float(np.interp(tk, t_ref, dyaw_ref))
            w_ang_k = np.array([0.0, 0.0, yaw_rate_k], dtype=float)
            running.append(
                self._make_integrated_running(p_des_k, yaw_des_k, v_lin_k, w_ang_k)
            )

        tN = float(t_start + self.horizon * self.dt_mpc)
        p_des_N, yaw_des_N = _interp_ref_pose(tN, t_ref, p_ref, yaw_ref)
        v_lin_N = np.array(
            [
                np.interp(tN, t_ref, dp_ref[:, 0]),
                np.interp(tN, t_ref, dp_ref[:, 1]),
                np.interp(tN, t_ref, dp_ref[:, 2]),
            ],
            dtype=float,
        )
        yaw_rate_N = float(np.interp(tN, t_ref, dyaw_ref))
        w_ang_N = np.array([0.0, 0.0, yaw_rate_N], dtype=float)
        terminal = self._make_integrated_terminal(p_des_N, yaw_des_N, v_lin_N, w_ang_N)
        return crocoddyl.ShootingProblem(x0, running, terminal)


def run_closed_loop_ee_pose_tracking(
    *,
    x0: np.ndarray,
    t_ref: np.ndarray,
    p_ref: np.ndarray,
    yaw_ref: np.ndarray,
    dt_mpc: float,
    horizon: int,
    sim_dt: float,
    control_dt: float,
    max_iter: int,
    use_thrust_constraints: bool,
    weights: EETrackingWeights = EETrackingWeights(),
    verbose: bool = True,
) -> dict:
    mpc = UAMEEPoseTrackingCrocoddylMPC(
        dt_mpc=dt_mpc,
        horizon=horizon,
        u_weights=weights,
        use_thrust_constraints=use_thrust_constraints,
    )

    # Precompute reference velocities for EE tracking.
    t_ref = np.asarray(t_ref, dtype=float).flatten()
    p_ref = np.asarray(p_ref, dtype=float)
    yaw_ref = np.asarray(yaw_ref, dtype=float).flatten()
    yaw_ref_u = np.unwrap(yaw_ref)
    if p_ref.ndim != 2 or p_ref.shape[1] != 3:
        raise ValueError(f"p_ref must have shape (N,3), got {p_ref.shape}")
    if len(t_ref) != len(p_ref) or len(yaw_ref_u) != len(t_ref):
        raise ValueError("t_ref/p_ref/yaw_ref length mismatch")

    # dp/dt and dyaw/dt
    dp_ref = np.gradient(p_ref, t_ref, axis=0)
    dyaw_ref = np.gradient(yaw_ref_u, t_ref)

    # Zero-cost placeholder for forward integration (dynamics only).
    cost0 = crocoddyl.CostModelSum(mpc.state, mpc.nu)
    diff0 = crocoddyl.DifferentialActionModelFreeFwdDynamics(mpc.state, mpc.actuation, cost0)
    sim_int = crocoddyl.IntegratedActionModelEuler(diff0, float(sim_dt))
    sim_data = sim_int.createData()

    sim_dt = float(sim_dt)
    control_dt = float(control_dt)
    n_inner = max(1, int(round(control_dt / sim_dt)))

    T_sim = float(t_ref[-1] - t_ref[0])
    n_total = max(1, int(np.ceil(T_sim / sim_dt)))

    x = np.asarray(x0, dtype=float).flatten()
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    ts: List[float] = []

    u_cmd_hold = mpc._u_ref.copy()
    xs_guess: Optional[List[np.ndarray]] = None
    us_guess: Optional[List[np.ndarray]] = None

    for step in range(n_total):
        t = step * sim_dt
        ts.append(t)
        xs.append(x.copy())
        us.append(u_cmd_hold.copy())

        # Update MPC every control_dt.
        if step % n_inner == 0:
            t_start = float(t)
            prob = mpc.build_shooting_problem_along_ref(
                x,
                t_start,
                t_ref,
                p_ref,
                yaw_ref_u,
                dp_ref,
                dyaw_ref,
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
            converged = solver.solve(xs_init, us_init, max_iter)
            wall_s = time.perf_counter() - t_solve0

            if verbose and step % max(1, 5 * n_inner) == 0:
                print(f"[EE pose MPC] t={t:.3f} converged={converged} cost={solver.cost:.4f} iters={solver.iter} wall={wall_s:.2f}s")

            u_cmd_hold = np.array(solver.us[0], dtype=float).copy()

            xs_guess = [solver.xs[i + 1].copy() for i in range(horizon)] + [solver.xs[-1].copy()]
            xs_guess[0] = x.copy()
            us_guess = [solver.us[i + 1].copy() for i in range(horizon - 1)] + [solver.us[-1].copy()]

        # Forward integrate one sim step.
        if step < n_total - 1:
            sim_int.calc(sim_data, x, u_cmd_hold)
            x = np.array(sim_data.xnext, dtype=float).flatten().copy()

    xs_arr = np.asarray(xs, dtype=float)
    ee_pos, _, ee_rpy, _ = compute_ee_kinematics_along_trajectory(
        xs_arr, mpc.robot_model, mpc.robot_data, mpc.ee_frame_id
    )
    yaw_meas = ee_rpy[:, 2].astype(float)

    # Reference aligned with our ts (t_ref starts at 0 in CLI generation below).
    ref_p = np.stack(
        [_interp_ref_pose(t, t_ref, p_ref, yaw_ref_u)[0] for t in ts], axis=0
    )
    ref_yaw = np.array(
        [_interp_ref_pose(t, t_ref, p_ref, yaw_ref_u)[1] for t in ts], dtype=float
    )

    return {
        "t": np.asarray(ts, dtype=float),
        "states": xs_arr,
        "u": np.asarray(us, dtype=float),
        "ee": ee_pos,
        "yaw_meas": yaw_meas,
        "p_ref": ref_p,
        "yaw_ref": ref_yaw,
        "mpc": mpc,
    }


def main():
    parser = argparse.ArgumentParser(description="Crocoddyl EE pose tracking MPC (ResidualModelFramePlacement)")
    parser.add_argument("--T_sim", type=float, default=8.0)
    parser.add_argument("--sim_dt", type=float, default=0.002)
    parser.add_argument("--control_dt", type=float, default=0.01)
    parser.add_argument("--dt_mpc", type=float, default=0.05)
    parser.add_argument("--N_mpc", type=int, default=25)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--use_thrust_constraints", action=argparse.BooleanOptionalAction, default=True)

    # Initial full state
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--y0", type=float, default=0.0)
    parser.add_argument("--z0", type=float, default=1.0)
    parser.add_argument("--j1", type=float, default=0.0)
    parser.add_argument("--j2", type=float, default=0.0)
    parser.add_argument("--yaw0", type=float, default=0.0)

    # Optional initial state bias (affects the *actual* initial EE pose via FK)
    parser.add_argument("--bias_x", type=float, default=0.0)
    parser.add_argument("--bias_y", type=float, default=0.0)
    parser.add_argument("--bias_z", type=float, default=0.0)
    parser.add_argument("--bias_j1", type=float, default=0.0)
    parser.add_argument("--bias_j2", type=float, default=0.0)
    parser.add_argument("--bias_yaw", type=float, default=0.0)

    # Reference offset: make reference initial EE pose different from actual EE pose.
    # (trajectory start / planning initial point)
    parser.add_argument("--ref_offset_x", type=float, default=0.5)
    parser.add_argument("--ref_offset_y", type=float, default=0.5)
    parser.add_argument("--ref_offset_z", type=float, default=0.5)
    parser.add_argument("--ref_offset_yaw", type=float, default=0.0)

    # Target EE pose reference (only yaw changes rotation)
    parser.add_argument("--target_x", type=float, default=1.5)
    parser.add_argument("--target_y", type=float, default=0.4)
    parser.add_argument("--target_z", type=float, default=1.5)
    parser.add_argument("--target_yaw", type=float, default=1.0)

    # Weights
    parser.add_argument("--w_pos", type=float, default=100.0)
    parser.add_argument("--w_rot_rp", type=float, default=1.0)
    parser.add_argument("--w_rot_yaw", type=float, default=1.0)
    parser.add_argument("--w_vel_lin", type=float, default=1.0)
    parser.add_argument("--w_vel_ang_rp", type=float, default=1.0)
    parser.add_argument("--w_vel_ang_yaw", type=float, default=1.0)
    parser.add_argument("--w_u", type=float, default=0.1)
    parser.add_argument("--w_state_reg", type=float, default=0.0)

    parser.add_argument("--no_verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    planner = S500UAMTrajectoryPlanner()
    x_init = make_uam_state(
        args.x0 + args.bias_x,
        args.y0 + args.bias_y,
        args.z0 + args.bias_z,
        j1=args.j1 + args.bias_j1,
        j2=args.j2 + args.bias_j2,
        yaw=args.yaw0 + args.bias_yaw,
    )
    x_init = np.asarray(x_init, dtype=float).copy()

    # Derive start EE pose from FK.
    q = x_init[: planner.robot_model.nq]
    v = x_init[planner.robot_model.nq :]
    pin.forwardKinematics(planner.robot_model, planner.robot_data, q, v)
    pin.updateFramePlacements(planner.robot_model, planner.robot_data)
    ee_frame = planner.robot_data.oMf[planner.ee_frame_id]
    p_start = np.array(ee_frame.translation, dtype=float)
    p_start_ref = p_start + np.array(
        [args.ref_offset_x, args.ref_offset_y, args.ref_offset_z], dtype=float
    )
    # pin.rpy.matrixToRpy returns a (roll, pitch, yaw) ndarray in pinocchio.
    yaw_start = float(pin.rpy.matrixToRpy(ee_frame.rotation)[2])  # ZYX yaw component
    yaw_start_ref = yaw_start + float(args.ref_offset_yaw)

    # Build reference arrays.
    dt_ref = min(0.02, max(0.001, args.control_dt))
    t_ref = np.arange(0.0, float(args.T_sim) + 1e-12, dt_ref, dtype=float)
    alpha = np.clip(t_ref / float(args.T_sim), 0.0, 1.0)
    p_ref = (1.0 - alpha)[:, None] * p_start_ref[None, :] + alpha[:, None] * np.array(
        [args.target_x, args.target_y, args.target_z], dtype=float
    )[None, :]
    yaw_ref = (1.0 - alpha) * yaw_start_ref + alpha * float(args.target_yaw)

    weights = EETrackingWeights(
        w_pos=float(args.w_pos),
        w_rot_rp=float(args.w_rot_rp),
        w_rot_yaw=float(args.w_rot_yaw),
        w_vel_lin=float(args.w_vel_lin),
        w_vel_ang_rp=float(args.w_vel_ang_rp),
        w_vel_ang_yaw=float(args.w_vel_ang_yaw),
        w_u=float(args.w_u),
        w_terminal_scale=3.0,
        w_state_reg=float(args.w_state_reg),
    )

    out = run_closed_loop_ee_pose_tracking(
        x0=x_init,
        t_ref=t_ref,
        p_ref=p_ref,
        yaw_ref=yaw_ref,
        dt_mpc=float(args.dt_mpc),
        horizon=int(args.N_mpc),
        sim_dt=float(args.sim_dt),
        control_dt=float(args.control_dt),
        max_iter=int(args.max_iter),
        use_thrust_constraints=bool(args.use_thrust_constraints),
        weights=weights,
        verbose=not bool(args.no_verbose),
    )

    t = out["t"]
    ee = out["ee"]
    pref = out["p_ref"]
    yaw_m = out["yaw_meas"]
    yaw_r = out["yaw_ref"]

    # Plot position + yaw + controls.
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(t, ee[:, 0], "r-", lw=1.2, label="ee x")
    axes[0].plot(t, ee[:, 1], "g-", lw=1.2, label="ee y")
    axes[0].plot(t, ee[:, 2], "b-", lw=1.2, label="ee z")
    axes[0].plot(t, pref[:, 0], "r--", lw=1.0, alpha=0.8, label="ref x")
    axes[0].plot(t, pref[:, 1], "g--", lw=1.0, alpha=0.8, label="ref y")
    axes[0].plot(t, pref[:, 2], "b--", lw=1.0, alpha=0.8, label="ref z")
    axes[0].set_ylabel("position [m]")
    axes[0].set_title("EE position tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(t, np.degrees(yaw_m), "k-", lw=1.2, label="yaw meas")
    axes[1].plot(t, np.degrees(yaw_r), "c--", lw=1.0, alpha=0.9, label="yaw ref")
    axes[1].set_ylabel("yaw [deg]")
    axes[1].set_title("EE yaw tracking (ZYX)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    U = out["u"]
    if U.ndim == 2 and U.shape[1] >= 6:
        # Controls: [T1..T4, tau1, tau2]
        for i in range(4):
            axes[2].plot(t, U[:, i], lw=1.0, label=f"T{i+1}")
        axes[2].plot(t, U[:, 4], "r-", lw=1.2, label="tau1")
        axes[2].plot(t, U[:, 5], "g-", lw=1.2, label="tau2")
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("control")
    axes[2].set_title("Controls (thrusts + joint torques)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=7, ncol=3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

