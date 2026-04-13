#!/usr/bin/env python3
"""
S500 UAM — Crocoddyl EE pose tracking (MPC)

EE-centric tracking uses separate costs on the EE frame:
  - position: ``ResidualModelFrameTranslation``
  - orientation: ``ResidualModelFramePlacement`` with translation block zeroed in activation
  - spatial velocity: ``ResidualModelFrameVelocity`` (LOCAL_WORLD_ALIGNED)
  - optional full-state residual vs a plan reference ``x_ref(t)``
  - rolling-horizon MPC solved by ``crocoddyl.SolverBoxFDDP``

Control (direct mode):
  u = [T1, T2, T3, T4, tau1, tau2]

Reference provided as:
  - t_ref: (N,) seconds
  - p_ref: (N,3) EE position in world frame
  - yaw_ref: (N,) ZYX yaw angle in rad

For rotation target we assume roll/pitch = 0 and yaw = yaw_ref.

Optional **sim-only payload**: when enabled, closed-loop integration uses a **copy** of the
Pinocchio model and adds a **solid-sphere** payload (CoM at gripper joint origin,
``Ixx=Iyy=Izz = 2/5·m·r²``) on the EE link at ``t_grasp``; MPC keeps the original model.

MPC problem construction (costs, shooting along EE reference) lives in
``s500_uam_crocoddyl_state_tracking_mpc`` (``UAMEEPoseTrackingCrocoddylMPC``); this module
keeps closed-loop simulation, CLI, and plotting.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

import numpy as np
import pinocchio as pin
import crocoddyl
import matplotlib.pyplot as plt

from s500_uam_trajectory_planner import (
    S500UAMTrajectoryPlanner,
    compute_ee_kinematics_along_trajectory,
    make_uam_state,
)
from s500_uam_closed_loop_plant import (
    CrocoddylEulerPlant,
    PayloadSchedulePlant,
    mpc_inner_stride,
)
from s500_uam_crocoddyl_state_tracking_mpc import (
    EETrackingWeights,
    UAMEEPoseTrackingCrocoddylMPC,
    _apply_first_order_actuator,
    interp_ref_pose,
    solid_sphere_principal_inertias,
    _apply_payload_inertia_on_plant_model,
)


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
    use_actuator_first_order: bool = False,
    tau_thrust: float = 0.06,
    tau_theta: float = 0.05,
    t_plan: Optional[np.ndarray] = None,
    x_plan: Optional[np.ndarray] = None,
    sim_payload_enable: bool = False,
    sim_payload_t_grasp: float = 1.0,
    sim_payload_mass: float = 0.2,
    sim_payload_sphere_radius: float = 0.02,
) -> dict:
    mpc = UAMEEPoseTrackingCrocoddylMPC(
        dt_mpc=dt_mpc,
        horizon=horizon,
        u_weights=weights,
        use_thrust_constraints=use_thrust_constraints,
    )
    # MPC uses ideal u; optional first-order lag only on the simulation plant (same as full-state tracker).

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

    sim_dt = float(sim_dt)
    t_grasp = max(0.0, float(sim_payload_t_grasp))
    m_pay = float(sim_payload_mass)
    r_sph = max(1e-6, float(sim_payload_sphere_radius))
    use_sim_plant_payload = bool(sim_payload_enable) and m_pay > 1e-9
    ixx_p, iyy_p, izz_p = solid_sphere_principal_inertias(m_pay, r_sph)
    com_pl = np.zeros(3, dtype=float)

    # Forward dynamics for closed-loop plant: separate Pinocchio copy if sim-only payload is enabled
    # (MPC shooting problem keeps using mpc.robot_model).
    if use_sim_plant_payload:
        sim_model = pin.Model(mpc.robot_model)
        sim_state, sim_actuation = mpc._planner.thruster_actuation_for_model(sim_model)
        cost0 = crocoddyl.CostModelSum(sim_state, mpc.nu)
        diff0 = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            sim_state, sim_actuation, cost0
        )
    else:
        sim_model = None
        cost0 = crocoddyl.CostModelSum(mpc.state, mpc.nu)
        diff0 = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            mpc.state, mpc.actuation, cost0
        )
    sim_int = crocoddyl.IntegratedActionModelEuler(diff0, float(sim_dt))
    sim_data = sim_int.createData()
    base_plant = CrocoddylEulerPlant(sim_int, sim_data)
    if use_sim_plant_payload:

        def _apply_payload_once() -> None:
            assert sim_model is not None
            _apply_payload_inertia_on_plant_model(
                sim_model,
                mpc.ee_frame_id,
                m_pay,
                com_pl,
                ixx_p,
                iyy_p,
                izz_p,
            )

        plant: CrocoddylEulerPlant | PayloadSchedulePlant = PayloadSchedulePlant(
            base_plant, t_grasp, _apply_payload_once
        )
    else:
        plant = base_plant

    control_dt = float(control_dt)
    n_inner = mpc_inner_stride(control_dt, sim_dt)

    T_sim = float(t_ref[-1] - t_ref[0])
    n_total = max(1, int(np.ceil(T_sim / sim_dt)))

    x = np.asarray(x0, dtype=float).flatten()
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    ts: List[float] = []

    u_cmd_hold = mpc._u_ref.copy()
    u_act = u_cmd_hold.copy()
    xs_guess: Optional[List[np.ndarray]] = None
    us_guess: Optional[List[np.ndarray]] = None

    for step in range(n_total):
        t = step * sim_dt
        ts.append(t)
        xs.append(x.copy())
        plant.on_pre_step(t, step)

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
                t_plan=t_plan,
                x_plan=x_plan,
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

        us.append(u_act.copy())

        if step < n_total - 1:
            x = plant.step(x, u_act)

    xs_arr = np.asarray(xs, dtype=float)
    ee_pos, _, ee_rpy, _ = compute_ee_kinematics_along_trajectory(
        xs_arr, mpc.robot_model, mpc.robot_data, mpc.ee_frame_id
    )
    yaw_meas = ee_rpy[:, 2].astype(float)

    # Reference aligned with our ts (t_ref starts at 0 in CLI generation below).
    ref_p = np.stack(
        [interp_ref_pose(t, t_ref, p_ref, yaw_ref_u)[0] for t in ts], axis=0
    )
    ref_yaw = np.array(
        [interp_ref_pose(t, t_ref, p_ref, yaw_ref_u)[1] for t in ts], dtype=float
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
        "sim_plant_payload_applied": bool(
            isinstance(plant, PayloadSchedulePlant) and plant.schedule_applied
        ),
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
    parser.add_argument("--w_state_track", type=float, default=0.0)
    parser.add_argument("--w_terminal_scale", type=float, default=3.0)

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
        w_terminal_scale=float(args.w_terminal_scale),
        w_state_reg=float(args.w_state_reg),
        w_state_track=float(args.w_state_track),
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

