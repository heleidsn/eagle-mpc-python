from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def load_result_csv(csv_path: Path):
    t = []
    p = []
    v = []
    q = []
    w = []
    u = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            p.append([float(row["p_x"]), float(row["p_y"]), float(row["p_z"])])
            v.append([float(row["v_x"]), float(row["v_y"]), float(row["v_z"])])
            q.append(
                [
                    float(row.get("q_w", "nan")),
                    float(row.get("q_x", "nan")),
                    float(row.get("q_y", "nan")),
                    float(row.get("q_z", "nan")),
                ]
            )
            w.append(
                [
                    float(row.get("omega_x", "nan")),
                    float(row.get("omega_y", "nan")),
                    float(row.get("omega_z", "nan")),
                ]
            )
            u.append(
                [
                    float(row.get("u_1", "nan")),
                    float(row.get("u_2", "nan")),
                    float(row.get("u_3", "nan")),
                    float(row.get("u_4", "nan")),
                ]
            )
    return np.asarray(t), np.asarray(p), np.asarray(v), np.asarray(q), np.asarray(w), np.asarray(u)


def load_tracking_csv(csv_path: Path):
    t = []
    p = []
    p_ref = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["time"]))
            p.append([float(row["px"]), float(row["py"]), float(row["pz"])])
            p_ref.append([float(row["ref_px"]), float(row["ref_py"]), float(row["ref_pz"])])
    t_arr = np.asarray(t)
    p_arr = np.asarray(p)
    p_ref_arr = np.asarray(p_ref)
    e_p = p_ref_arr - p_arr
    return t_arr, p_arr, p_ref_arr, e_p


def load_track(track_path: Path):
    with track_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gates = np.asarray(cfg.get("gates", []), dtype=float)
    init_pos = np.asarray(cfg.get("initial", {}).get("position", [0.0, 0.0, 0.0]), dtype=float)
    end_pos = np.asarray(cfg.get("end", {}).get("position", [np.nan, np.nan, np.nan]), dtype=float)
    return gates, init_pos, end_pos


def estimate_gate_pass_times(t: np.ndarray, p: np.ndarray, gates: np.ndarray):
    if gates.size == 0:
        return []
    pass_info = []
    for i, gate in enumerate(gates):
        d = np.linalg.norm(p - gate[None, :], axis=1)
        k = int(np.argmin(d))
        pass_info.append(
            {
                "gate_idx": i,
                "time": float(t[k]),
                "traj_idx": k,
                "distance": float(d[k]),
                "gate_pos": gate,
            }
        )
    return pass_info


def quaternion_to_euler_deg(q: np.ndarray):
    if q.shape[0] == 0 or np.isnan(q).all():
        return np.full((q.shape[0], 3), np.nan)
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg(np.column_stack((roll, pitch, yaw)))


def draw_time_markers(ax, gate_pass_info):
    for info in gate_pass_info:
        gt = info["time"]
        ax.axvline(gt, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)


def plot_optimization_result(
    t: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    gates: np.ndarray,
    init_pos: np.ndarray,
    end_pos: np.ndarray,
    gate_pass_info,
    meta: dict,
):
    speed = np.linalg.norm(v, axis=1)
    euler_deg = quaternion_to_euler_deg(q)
    omega_deg = np.rad2deg(w)
    u_total = np.nansum(u, axis=1)
    total_time = float(t[-1] - t[0]) if t.size > 1 else 0.0
    max_speed = float(np.nanmax(speed)) if speed.size > 0 else 0.0
    mean_speed = float(np.nanmean(speed)) if speed.size > 0 else 0.0
    traj_length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if p.shape[0] > 1 else 0.0

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.0, 1.0], hspace=0.42, wspace=0.30)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[0, 2])
    ax_ang = fig.add_subplot(gs[1, 1])
    ax_omg = fig.add_subplot(gs[1, 2])
    ax_thr = fig.add_subplot(gs[2, 1])
    ax_info = fig.add_subplot(gs[2, 2])

    points = p.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_speed = 0.5 * (speed[:-1] + speed[1:]) if speed.size > 1 else np.array([0.0])
    lc = Line3DCollection(segments, cmap="viridis", linewidth=2.5)
    lc.set_array(seg_speed)
    if speed.size > 0:
        lc.set_clim(float(np.nanmin(speed)), float(np.nanmax(speed)))
    ax3d.add_collection3d(lc)
    cbar = fig.colorbar(lc, ax=ax3d, fraction=0.03, pad=0.08)
    cbar.set_label("speed [m/s]")
    ax3d.scatter(p[0, 0], p[0, 1], p[0, 2], marker="o", label="opt start")
    ax3d.scatter(p[-1, 0], p[-1, 1], p[-1, 2], marker="x", label="opt end")
    ax3d.scatter(init_pos[0], init_pos[1], init_pos[2], marker="^", label="track init")
    if np.all(np.isfinite(end_pos)):
        ax3d.scatter(end_pos[0], end_pos[1], end_pos[2], marker="*", label="track end")
    if gates.size > 0:
        ax3d.scatter(gates[:, 0], gates[:, 1], gates[:, 2], marker="s", label="gates")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("Optimization trajectory (3D)")
    ax3d.legend(fontsize=8, loc="upper left")

    ax_pos.plot(t, p[:, 0], label="p_x")
    ax_pos.plot(t, p[:, 1], label="p_y")
    ax_pos.plot(t, p[:, 2], label="p_z")
    ax_pos.set_ylabel("position [m]")
    ax_pos.set_title("Position")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(fontsize=8, ncol=3)
    draw_time_markers(ax_pos, gate_pass_info)

    ax_vel.plot(t, v[:, 0], label="v_x")
    ax_vel.plot(t, v[:, 1], label="v_y")
    ax_vel.plot(t, v[:, 2], label="v_z")
    ax_vel.plot(t, speed, "--", label="|v|")
    ax_vel.set_ylabel("velocity [m/s]")
    ax_vel.set_title("Velocity")
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(fontsize=8, ncol=4)
    draw_time_markers(ax_vel, gate_pass_info)

    if np.isnan(euler_deg).all():
        ax_ang.text(0.03, 0.5, "angle unavailable for simple model", transform=ax_ang.transAxes, va="center")
    else:
        ax_ang.plot(t, euler_deg[:, 0], label="roll")
        ax_ang.plot(t, euler_deg[:, 1], label="pitch")
        ax_ang.plot(t, euler_deg[:, 2], label="yaw")
        ax_ang.legend(fontsize=8, ncol=3)
    ax_ang.set_ylabel("angle [deg]")
    ax_ang.set_title("Attitude")
    ax_ang.grid(True, alpha=0.3)
    draw_time_markers(ax_ang, gate_pass_info)

    if np.isnan(omega_deg).all():
        ax_omg.text(0.03, 0.5, "angular rate unavailable for simple model", transform=ax_omg.transAxes, va="center")
    else:
        ax_omg.plot(t, omega_deg[:, 0], label="omega_x")
        ax_omg.plot(t, omega_deg[:, 1], label="omega_y")
        ax_omg.plot(t, omega_deg[:, 2], label="omega_z")
        ax_omg.legend(fontsize=8, ncol=3)
    ax_omg.set_ylabel("ang vel [deg/s]")
    ax_omg.set_title("Angular velocity")
    ax_omg.grid(True, alpha=0.3)
    draw_time_markers(ax_omg, gate_pass_info)

    if np.isnan(u).all():
        ax_thr.text(0.03, 0.5, "thrust unavailable", transform=ax_thr.transAxes, va="center")
    else:
        if not np.isnan(u[:, 3]).all():
            ax_thr.plot(t, u[:, 0], label="u1")
            ax_thr.plot(t, u[:, 1], label="u2")
            ax_thr.plot(t, u[:, 2], label="u3")
            ax_thr.plot(t, u[:, 3], label="u4")
            thrust_min = float(meta.get("thrust_min", 0.0))
            thrust_max = float(meta.get("thrust_max", 0.0))
            ax_thr.axhline(thrust_min, linestyle="--", linewidth=1.0, color="gray", label="T_min")
            ax_thr.axhline(thrust_max, linestyle="--", linewidth=1.0, color="black", label="T_max")
            ax_thr.set_ylabel("thrust [N]")
        else:
            ax_thr.plot(t, u[:, 0], label="a_x")
            ax_thr.plot(t, u[:, 1], label="a_y")
            ax_thr.plot(t, u[:, 2], label="a_z")
            ax_thr.set_ylabel("acc cmd")
        ax_thr.legend(fontsize=8, ncol=5)
    ax_thr.set_title("Thrust / control input")
    ax_thr.grid(True, alpha=0.3)
    ax_thr.set_xlabel("t [s]")
    draw_time_markers(ax_thr, gate_pass_info)

    ax_info.axis("off")
    info_text = (
        f"timestamp: {meta.get('timestamp', 'unknown')}\n"
        f"planning: {meta.get('planning', 'unknown')}\n"
        f"model: {meta['model']}\n"
        f"source: {meta['source']}\n"
        f"solver: {meta['solver']}\n"
        f"total time: {total_time:.3f} s\n"
        f"max speed: {max_speed:.3f} m/s\n"
        f"mean speed: {mean_speed:.3f} m/s\n"
        f"path length: {traj_length:.3f} m\n"
        f"samples: {int(t.size)}\n"
        f"gates: {int(gates.shape[0]) if gates.ndim > 1 else 0}"
    )
    ax_info.text(
        0.02,
        0.98,
        info_text,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "whitesmoke", "alpha": 0.95},
    )
    fig.tight_layout()
    return fig


def plot_tracking_result(t_real: np.ndarray, p_real: np.ndarray, p_ref_real: np.ndarray):
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot(p_ref_real[:, 0], p_ref_real[:, 1], p_ref_real[:, 2], label="ref")
    ax3d.plot(p_real[:, 0], p_real[:, 1], p_real[:, 2], label="real")
    ax3d.set_title("Tracking result (3D)")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend()
    fig.tight_layout()
    return fig


def main():
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    parser = argparse.ArgumentParser(description="Plot segment optimization/tracking results")
    parser.add_argument(
        "--result-csv",
        type=str,
        default=str(this_dir / "result_segment.csv"),
        help="Path to result CSV",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=str(this_dir / "track.yaml"),
        help="Path to track yaml",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(this_dir / "result_segment.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--tracking-csv",
        type=str,
        default=str(repo_root / "tracking_results" / "result_segment_latest_vmax5_yaw_fixed_0deg__px4.csv"),
        help="Path to tracking CSV",
    )
    parser.add_argument("--plot-mode", type=str, default="optimization", choices=["optimization", "tracking"])
    parser.add_argument("--disable-tracking", action="store_true", help="Disable loading tracking overlays")
    parser.add_argument("--meta-planning-mode", type=str, default="unknown", help="Metadata: planning mode")
    parser.add_argument("--meta-timestamp", type=str, default="unknown", help="Metadata: result timestamp")
    parser.add_argument("--meta-dynamics-model", type=str, default="unknown", help="Metadata: planner dynamics model")
    parser.add_argument("--meta-dynamics-source", type=str, default="unknown", help="Metadata: dynamics parameter source")
    parser.add_argument("--meta-solver-status", type=str, default="unknown", help="Metadata: optimizer return status")
    parser.add_argument("--show", action="store_true", help="Show plot window")
    parser.add_argument("--meta-thrust-min", type=float, default=0.0, help="Metadata: per-rotor minimum thrust [N]")
    parser.add_argument("--meta-thrust-max", type=float, default=0.0, help="Metadata: per-rotor maximum thrust [N]")
    args = parser.parse_args()

    result_csv = Path(args.result_csv)
    if not result_csv.exists():
        raise FileNotFoundError(f"Result CSV does not exist: {result_csv}")

    t, p, v, q, w, u = load_result_csv(result_csv)
    gates, init_pos, end_pos = load_track(Path(args.track))
    gate_pass_info = estimate_gate_pass_times(t, p, gates)
    if args.plot_mode == "optimization":
        fig = plot_optimization_result(
            t=t,
            p=p,
            v=v,
            q=q,
            w=w,
            u=u,
            gates=gates,
            init_pos=init_pos,
            end_pos=end_pos,
            gate_pass_info=gate_pass_info,
            meta={
                "timestamp": args.meta_timestamp,
                "planning": args.meta_planning_mode,
                "model": args.meta_dynamics_model,
                "source": args.meta_dynamics_source,
                "solver": args.meta_solver_status,
                "thrust_min": args.meta_thrust_min,
                "thrust_max": args.meta_thrust_max,
            },
        )
    else:
        tracking_path = Path(args.tracking_csv)
        if not tracking_path.exists():
            raise FileNotFoundError(f"Tracking CSV does not exist: {tracking_path}")
        t_real, p_real, p_ref_real, _ = load_tracking_csv(tracking_path)
        fig = plot_tracking_result(t_real, p_real, p_ref_real)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

