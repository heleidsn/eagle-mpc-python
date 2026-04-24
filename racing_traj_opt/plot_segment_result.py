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
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            p.append([float(row["p_x"]), float(row["p_y"]), float(row["p_z"])])
            v.append([float(row["v_x"]), float(row["v_y"]), float(row["v_z"])])
    return np.asarray(t), np.asarray(p), np.asarray(v)


def load_track(track_path: Path):
    with track_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gates = np.asarray(cfg.get("gates", []), dtype=float)
    init_pos = np.asarray(cfg.get("initial", {}).get("position", [0.0, 0.0, 0.0]), dtype=float)
    end_pos = np.asarray(cfg.get("end", {}).get("position", [np.nan, np.nan, np.nan]), dtype=float)
    return gates, init_pos, end_pos


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
    parser.add_argument("--show", action="store_true", help="Show plot window")
    args = parser.parse_args()

    result_csv = Path(args.result_csv)
    if not result_csv.exists():
        raise FileNotFoundError(f"Result CSV does not exist: {result_csv}")

    t, p, v = load_result_csv(result_csv)
    gates, init_pos, end_pos = load_track(Path(args.track))
    speed = np.linalg.norm(v, axis=1)

    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axv = fig.add_subplot(1, 2, 2)

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
    ax3d.scatter(p[0, 0], p[0, 1], p[0, 2], marker="o", label="traj start")
    ax3d.scatter(p[-1, 0], p[-1, 1], p[-1, 2], marker="x", label="traj end")
    ax3d.scatter(init_pos[0], init_pos[1], init_pos[2], marker="^", label="track init")
    if np.all(np.isfinite(end_pos)):
        ax3d.scatter(end_pos[0], end_pos[1], end_pos[2], marker="*", label="track end")
    if gates.size > 0:
        ax3d.scatter(gates[:, 0], gates[:, 1], gates[:, 2], marker="s", label="gates")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("3D trajectory (colored by speed)")
    ax3d.legend(fontsize=8)

    axv.plot(t, v[:, 0], linewidth=1.8, label="v_x")
    axv.plot(t, v[:, 1], linewidth=1.8, label="v_y")
    axv.plot(t, v[:, 2], linewidth=1.8, label="v_z")
    axv.plot(t, speed, linewidth=1.2, linestyle="--", label="|v|")
    axv.set_xlabel("t [s]")
    axv.set_ylabel("velocity [m/s]")
    axv.set_title("Velocity profile (xyz + norm)")
    axv.grid(True, alpha=0.3)
    axv.legend()

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

