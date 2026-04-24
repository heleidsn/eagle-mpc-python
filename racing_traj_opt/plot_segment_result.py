from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import joblib


def load_result_csv(csv_path: Path):
    t = []
    p = []
    v = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            p.append([float(row["p_x"]), float(row["p_y"]), float(row["p_z"])])
            v.append([float(row["v_x"]), float(row["v_y"]), float(row["v_z"])])
    return np.asarray(t), np.asarray(p), np.asarray(v)


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


def predict_position_error_with_nn(
    model_path: Path,
    t: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    if not model_path.exists():
        raise FileNotFoundError(f"NN model not found: {model_path}")
    model_obj = joblib.load(model_path)
    scaler = model_obj["scaler"]
    mlp = model_obj["mlp"]
    window_size = int(model_obj.get("window_size", 1))

    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    a = np.gradient(v, axis=0) / dt[:, None]
    j = np.gradient(a, axis=0) / dt[:, None]

    yaw = np.arctan2(v[:, 1], np.maximum(np.abs(v[:, 0]) > 1e-9, True) * v[:, 0])
    yaw_rate = np.gradient(yaw) / dt
    omega = np.zeros_like(v)
    g_vec = np.array([0.0, 0.0, 9.81])
    acc_total = a + g_vec[None, :]
    acc_norm = np.linalg.norm(acc_total, axis=1, keepdims=True) + 1e-9
    body_z_world = acc_total / acc_norm
    # Approximate normalized thrust command from required total acceleration.
    u_thrust = np.clip((np.linalg.norm(acc_total, axis=1) / 9.81 - 1.0) * 0.2 + 0.5, 0.0, 1.0)
    thrust_margin = 1.0 - u_thrust

    ep_prev = np.zeros(3)
    ev_prev = np.zeros(3)
    x_frames = []
    ep_pred = []
    for i in range(t.shape[0]):
        x_i = np.hstack(
            (
                ep_prev,
                ev_prev,
                a[i],
                j[i],
                yaw[i],
                yaw_rate[i],
                v[i],
                omega[i],
                body_z_world[i],
                u_thrust[i],
                thrust_margin[i],
            )
        )
        x_frames.append(x_i)
        x_window = np.array(x_frames[-window_size:])
        if x_window.shape[0] < window_size:
            pad = np.tile(x_window[0], (window_size - x_window.shape[0], 1))
            x_window = np.vstack((pad, x_window))
        x_in = x_window.reshape(1, -1)
        x_in_s = scaler.transform(x_in)
        y_hat = mlp.predict(x_in_s)[0]
        ep_hat = y_hat[0:3]
        ev_hat = y_hat[3:6]
        ep_pred.append(ep_hat)
        ep_prev = ep_hat
        ev_prev = ev_hat
    return np.asarray(ep_pred)


def main():
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent

    parser = argparse.ArgumentParser(description="Plot segment optimization result CSV")
    parser.add_argument(
        "--result-csv",
        type=str,
        default=str(repo_root / "trajectory" / "result_segment_latest.csv"),
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
        default=str(repo_root / "trajectory" / "result_segment_latest.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--nn-model",
        type=str,
        default=str(repo_root / "tracking_results" / "tracking_error_nn_model.joblib"),
        help="Path to trained NN model for error prediction",
    )
    parser.add_argument(
        "--tracking-csv",
        type=str,
        default=str(repo_root / "tracking_results" / "result_segment_latest_vmax5_yaw_fixed_0deg__px4.csv"),
        help="Path to real tracking CSV for actual position error",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window")
    args = parser.parse_args()

    result_csv = Path(args.result_csv)
    if not result_csv.exists():
        raise FileNotFoundError(f"Result CSV does not exist: {result_csv}")

    t, p, v = load_result_csv(result_csv)
    gates, init_pos, end_pos = load_track(Path(args.track))
    speed = np.linalg.norm(v, axis=1)
    gate_pass_info = estimate_gate_pass_times(t, p, gates)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    axv = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1])
    axp = fig.add_subplot(gs[2, 1])

    # Color 3D trajectory by speed magnitude
    points = p.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_speed = 0.5 * (speed[:-1] + speed[1:])
    lc = Line3DCollection(segments, cmap="viridis", linewidth=2.5)
    lc.set_array(seg_speed)
    lc.set_clim(float(np.min(speed)), float(np.max(speed)))
    ax3d.add_collection3d(lc)
    cbar = fig.colorbar(lc, ax=ax3d, fraction=0.035, pad=0.08)
    cbar.set_label("speed [m/s]")
    ax3d.scatter(p[0, 0], p[0, 1], p[0, 2], marker="o", label="opt start")
    ax3d.scatter(p[-1, 0], p[-1, 1], p[-1, 2], marker="x", label="opt end")
    ax3d.scatter(init_pos[0], init_pos[1], init_pos[2], marker="^", label="track init")
    if np.all(np.isfinite(end_pos)):
        ax3d.scatter(end_pos[0], end_pos[1], end_pos[2], marker="*", label="track end")
    if gates.size > 0:
        ax3d.scatter(gates[:, 0], gates[:, 1], gates[:, 2], marker="s", label="gates")
        for info in gate_pass_info:
            gp = info["gate_pos"]
            gt = info["time"]
            ax3d.text(
                gp[0],
                gp[1],
                gp[2],
                f"g{info['gate_idx']}: {gt:.2f}s",
                fontsize=8,
                color="black",
            )
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("3D trajectory (colored by speed)")
    ax3d.legend(fontsize=8, loc="upper left")

    axv.plot(t, v[:, 0], linewidth=1.8, label="v_x")
    axv.plot(t, v[:, 1], linewidth=1.8, label="v_y")
    axv.plot(t, v[:, 2], linewidth=1.8, label="v_z")
    axv.plot(t, speed, linewidth=1.2, linestyle="--", label="|v|")
    axv.set_xlabel("t [s]")
    axv.set_ylabel("velocity [m/s]")
    axv.set_title("Velocity profile (xyz + norm)")
    axv.grid(True, alpha=0.3)
    axv.legend()
    for info in gate_pass_info:
        gt = info["time"]
        axv.axvline(gt, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
        axv.text(gt, axv.get_ylim()[1] * 0.95, f"g{info['gate_idx']}", rotation=90, va="top", ha="right", fontsize=8)

    tracking_path = Path(args.tracking_csv)
    tracking_loaded = False
    t_real = None
    p_real = None
    p_ref_real = None
    if tracking_path.exists():
        t_real, p_real, p_ref_real, e_p_real = load_tracking_csv(tracking_path)
        # Align actual tracking data to planned trajectory time horizon.
        plan_t_min = float(np.min(t))
        plan_t_max = float(np.max(t))
        mask = (t_real >= plan_t_min) & (t_real <= plan_t_max)
        if np.any(mask):
            t_real = t_real[mask]
            p_real = p_real[mask]
            p_ref_real = p_ref_real[mask]
            e_p_real = e_p_real[mask]
        tracking_loaded = True
        ax3d.plot(p_real[:, 0], p_real[:, 1], p_real[:, 2], color="tab:red", linewidth=1.5, alpha=0.9, label="tracking(real)")
        ax3d.plot(p_ref_real[:, 0], p_ref_real[:, 1], p_ref_real[:, 2], color="tab:orange", linewidth=1.2, alpha=0.8, label="tracking ref(real)")
        ax3d.legend(fontsize=8, loc="upper left")

        # New subplot: real position tracking error per axis.
        e_p_real_norm = np.linalg.norm(e_p_real, axis=1)
        axp.plot(t_real, e_p_real[:, 0], linewidth=1.2, label="e_px_actual")
        axp.plot(t_real, e_p_real[:, 1], linewidth=1.2, label="e_py_actual")
        axp.plot(t_real, e_p_real[:, 2], linewidth=1.2, label="e_pz_actual")
        axp.plot(t_real, e_p_real_norm, linewidth=1.6, linestyle="--", label="|e_p|_actual")
        axp.set_title("Real position tracking error")
        axp.set_xlabel("t [s]")
        axp.set_ylabel("error [m]")
        axp.grid(True, alpha=0.3)
        axp.legend(fontsize=8, ncol=2)
        for info in gate_pass_info:
            gt = info["time"]
            axp.axvline(gt, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
            axp.text(gt, axp.get_ylim()[1] * 0.95, f"g{info['gate_idx']}", rotation=90, va="top", ha="right", fontsize=8)
    else:
        axp.text(
            0.05,
            0.5,
            f"tracking csv not found:\n{tracking_path}",
            transform=axp.transAxes,
            va="center",
            ha="left",
        )
        axp.set_axis_off()

    # Predicted position control error from trained NN model.
    try:
        ep_pred = predict_position_error_with_nn(Path(args.nn_model), t, p, v)
        ep_norm = np.linalg.norm(ep_pred, axis=1)
        axe.plot(t, ep_pred[:, 0], linewidth=1.4, label="e_px_pred")
        axe.plot(t, ep_pred[:, 1], linewidth=1.4, label="e_py_pred")
        axe.plot(t, ep_pred[:, 2], linewidth=1.4, label="e_pz_pred")
        axe.plot(t, ep_norm, linewidth=1.8, linestyle="--", label="|e_p|_pred")

        axe.set_title("Predicted position control error (NN)")
        axe.set_xlabel("t [s]")
        axe.set_ylabel("error [m]")
        axe.grid(True, alpha=0.3)
        axe.legend(fontsize=8, ncol=2)
        for info in gate_pass_info:
            gt = info["time"]
            axe.axvline(gt, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
            axe.text(gt, axe.get_ylim()[1] * 0.95, f"g{info['gate_idx']}", rotation=90, va="top", ha="right", fontsize=8)
    except Exception as exc:
        axe.text(
            0.05,
            0.5,
            f"NN error subplot unavailable:\n{exc}",
            transform=axe.transAxes,
            va="center",
            ha="left",
        )
        axe.set_axis_off()

    fig.tight_layout()
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

