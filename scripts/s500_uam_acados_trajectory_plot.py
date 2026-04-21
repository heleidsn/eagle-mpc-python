#!/usr/bin/env python3
"""Matplotlib visualization for S500 UAM acados trajectories (4x4 GUI dashboard, 3D, CLI figure)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from s500_uam_trajectory_planner import (
    compute_ee_kinematics_along_trajectory,
    base_lin_ang_world_from_robot_state,
)

# Keep in sync with s500_uam_acados_trajectory.CONTROL_INPUT_*
CONTROL_INPUT_DIRECT = "direct"
CONTROL_INPUT_CASCADE = "cascade"

STATE_LIMITS = {
    "v_max": 1.0,
    "omega_max": 2.0,
    "j_angle_max": 2.0,
    "j_vel_max": 10.0,
}


def _quat_to_euler(quat):
    """Quat Nx4 [qx,qy,qz,qw] to euler (roll,pitch,yaw) rad."""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.column_stack([roll, pitch, yaw])


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


def plot_sqp_cost_vs_iteration(
    cost_trace,
    iteration_indices=None,
    *,
    title: str = "SQP NLP cost vs iteration",
    save_path: str | None = None,
    show: bool = True,
):
    """Plot NLP objective after each SQP inner solve (from ``get_cost()``), e.g. from ``stats['cost_trace']``."""
    c = np.asarray(cost_trace, dtype=float)
    if c.size == 0:
        return
    if iteration_indices is not None and len(iteration_indices) == len(c):
        x = np.asarray(iteration_indices, dtype=float)
    else:
        x = np.arange(1, len(c) + 1, dtype=float)
    mask = np.isfinite(c)
    if not np.any(mask):
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x[mask], c[mask], "b.-", linewidth=1.2, markersize=5)
    ax.set_xlabel("SQP iteration")
    ax.set_ylabel("NLP cost")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SQP cost plot saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


__all__ = [
    "plot_acados_into_figure",
    "plot_acados_3d_into_figure",
    "plot_results",
    "plot_sqp_cost_vs_iteration",
    "CONTROL_INPUT_DIRECT",
    "CONTROL_INPUT_CASCADE",
]
