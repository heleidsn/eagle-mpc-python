#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty csv: {path}")
    out: Dict[str, np.ndarray] = {}
    for k in rows[0].keys():
        out[k] = np.asarray([float(r[k]) for r in rows], dtype=float)
    return out


def load_gate_events(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing gate event csv: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        events = []
        for row in reader:
            events.append(
                {
                    "kind": str(row.get("kind", "gate")),
                    "index": int(float(row.get("index", 0))),
                    "t": float(row["t"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                }
            )
    return events


def apply_gate_z_offset(events: list[dict], z_offset_m: float) -> list[dict]:
    """Shift gate/end z positions to match real-tracking world-frame offset."""
    out = []
    for ev in events:
        shifted = dict(ev)
        shifted["z"] = float(shifted["z"]) + float(z_offset_m)
        out.append(shifted)
    return out


def position_from_real_csv(data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack((data["px"], data["py"], data["pz"]))


def reference_position_from_real_csv(data: Dict[str, np.ndarray]) -> np.ndarray:
    if all(k in data for k in ("ref_px", "ref_py", "ref_pz")):
        return np.column_stack((data["ref_px"], data["ref_py"], data["ref_pz"]))
    return np.full((len(data["time"]), 3), np.nan, dtype=float)


def interp_position_at_times(t: np.ndarray, p: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    out = np.zeros((len(query_t), 3), dtype=float)
    for j in range(3):
        out[:, j] = np.interp(query_t, t, p[:, j])
    return out


def compute_gate_errors(real_data: Dict[str, np.ndarray], gate_events: list[dict], label: str) -> list[dict]:
    t = np.asarray(real_data["time"], dtype=float)
    p = position_from_real_csv(real_data)
    query_t = np.asarray([ev["t"] for ev in gate_events], dtype=float)
    p_at = interp_position_at_times(t, p, query_t)

    rows = []
    for i, ev in enumerate(gate_events):
        gate_p = np.array([ev["x"], ev["y"], ev["z"]], dtype=float)
        err = p_at[i] - gate_p
        rows.append(
            {
                "trajectory": label,
                "kind": ev["kind"],
                "index": int(ev["index"]),
                "t_plan": float(ev["t"]),
                "gate_x": float(gate_p[0]),
                "gate_y": float(gate_p[1]),
                "gate_z": float(gate_p[2]),
                "real_x": float(p_at[i, 0]),
                "real_y": float(p_at[i, 1]),
                "real_z": float(p_at[i, 2]),
                "err_x": float(err[0]),
                "err_y": float(err[1]),
                "err_z": float(err[2]),
                "err_norm": float(np.linalg.norm(err)),
            }
        )
    return rows


def save_gate_error_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "trajectory",
        "kind",
        "index",
        "t_plan",
        "gate_z_offset_m",
        "gate_x",
        "gate_y",
        "gate_z",
        "real_x",
        "real_y",
        "real_z",
        "err_x",
        "err_y",
        "err_z",
        "err_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def gate_label(row: dict) -> str:
    return f"G{int(row['index'])}" if row["kind"] == "gate" else "End"


def set_3d_equal(ax, *xyz_arrays: np.ndarray, margin: float = 0.06) -> None:
    pts = []
    for arr in xyz_arrays:
        A = np.asarray(arr, dtype=float)
        if A.ndim == 2 and A.shape[1] >= 3 and A.shape[0] > 0:
            pts.append(A[:, :3])
    if not pts:
        return
    P = np.vstack(pts)
    ok = np.isfinite(P).all(axis=1)
    P = P[ok]
    if P.size == 0:
        return
    lo = np.min(P, axis=0)
    hi = np.max(P, axis=0)
    center = 0.5 * (lo + hi)
    span = max(float(np.max(hi - lo)), 1e-9)
    radius = 0.5 * span * (1.0 + margin)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def main() -> None:
    base = Path(__file__).resolve().parent
    cfg = {
        "trajectory_name": "racing_s500_new",
        "gate_z_offset_m": 3.0,
    }

    traj_dir = base / "results" / cfg["trajectory_name"]
    real_path = traj_dir / "traj_real.csv"
    real_nn_path = traj_dir / "traj_real_nn.csv"
    gates_path = traj_dir / "traj_gates.csv"
    gates_nn_path = traj_dir / "traj_gates_nn.csv"

    for p in (real_path, real_nn_path, gates_path, gates_nn_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    real = load_csv(real_path)
    real_nn = load_csv(real_nn_path)
    gate_z_offset_m = float(cfg.get("gate_z_offset_m", 0.0))
    gates = apply_gate_z_offset(load_gate_events(gates_path), gate_z_offset_m)
    gates_nn = apply_gate_z_offset(load_gate_events(gates_nn_path), gate_z_offset_m)

    rows_real = compute_gate_errors(real, gates, "traj_real")
    rows_nn = compute_gate_errors(real_nn, gates_nn, "traj_real_nn")
    for row in rows_real + rows_nn:
        row["gate_z_offset_m"] = gate_z_offset_m
    all_rows = rows_real + rows_nn

    error_csv = traj_dir / "traj_real_gate_errors.csv"
    save_gate_error_csv(error_csv, all_rows)

    t_real = real["time"]
    t_nn = real_nn["time"]
    p_real = position_from_real_csv(real)
    p_nn = position_from_real_csv(real_nn)
    p_ref = reference_position_from_real_csv(real)
    p_ref_nn = reference_position_from_real_csv(real_nn)

    labels_real = [gate_label(r) for r in rows_real]
    labels_nn = [gate_label(r) for r in rows_nn]
    x_real = np.arange(len(rows_real), dtype=float)
    x_nn = np.arange(len(rows_nn), dtype=float)

    err_norm_real = np.asarray([r["err_norm"] for r in rows_real], dtype=float)
    err_norm_nn = np.asarray([r["err_norm"] for r in rows_nn], dtype=float)
    err_comp_real = np.asarray([[r["err_x"], r["err_y"], r["err_z"]] for r in rows_real], dtype=float)
    err_comp_nn = np.asarray([[r["err_x"], r["err_y"], r["err_z"]] for r in rows_nn], dtype=float)

    fig = plt.figure(figsize=(22, 13))
    fig.suptitle(
        f"Real tracking gate error: {cfg['trajectory_name']} | gate z offset = {gate_z_offset_m:.2f} m",
        fontsize=18,
        y=0.985,
    )
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.2, 1.0],
        height_ratios=[1.0, 1.0, 0.8],
        hspace=0.34,
        wspace=0.24,
        left=0.045,
        right=0.985,
        top=0.93,
        bottom=0.07,
    )

    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax3.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], color="C0", ls="--", lw=1.6, alpha=0.75, label="planned ref")
    ax3.plot(p_ref_nn[:, 0], p_ref_nn[:, 1], p_ref_nn[:, 2], color="C1", ls="--", lw=1.6, alpha=0.75, label="planned_nn ref")
    ax3.plot(p_real[:, 0], p_real[:, 1], p_real[:, 2], color="C0", lw=2.4, label="traj_real")
    ax3.plot(p_nn[:, 0], p_nn[:, 1], p_nn[:, 2], color="C1", lw=2.4, label="traj_real_nn")

    gate_pts = np.asarray([[ev["x"], ev["y"], ev["z"]] for ev in gates if ev["kind"] == "gate"], dtype=float)
    if gate_pts.size:
        ax3.scatter(gate_pts[:, 0], gate_pts[:, 1], gate_pts[:, 2], color="k", marker="o", s=70, label="gates")
        for row in rows_real:
            if row["kind"] == "gate":
                ax3.text(row["gate_x"], row["gate_y"], row["gate_z"], f" {gate_label(row)}", fontsize=9, color="k")

    real_gate_pts = np.asarray([[r["real_x"], r["real_y"], r["real_z"]] for r in rows_real], dtype=float)
    nn_gate_pts = np.asarray([[r["real_x"], r["real_y"], r["real_z"]] for r in rows_nn], dtype=float)
    ax3.scatter(real_gate_pts[:, 0], real_gate_pts[:, 1], real_gate_pts[:, 2], color="C0", marker="x", s=85, label="real @ gate t")
    ax3.scatter(nn_gate_pts[:, 0], nn_gate_pts[:, 1], nn_gate_pts[:, 2], color="C1", marker="x", s=85, label="real_nn @ gate t")

    ax3.set_xlabel("X [m]", fontsize=12, labelpad=9)
    ax3.set_ylabel("Y [m]", fontsize=12, labelpad=9)
    ax3.set_zlabel("Z [m]", fontsize=12, labelpad=9)
    ax3.set_title("3D real tracking trajectory and gate-time samples", fontsize=15)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="both", labelsize=10)
    set_3d_equal(ax3, p_ref, p_ref_nn, p_real, p_nn, gate_pts, real_gate_pts, nn_gate_pts)

    ax_bar = fig.add_subplot(gs[0, 1])
    width = 0.38
    ax_bar.bar(x_real - width / 2, err_norm_real, width=width, color="C0", alpha=0.78, label="traj_real")
    ax_bar.bar(x_nn + width / 2, err_norm_nn, width=width, color="C1", alpha=0.78, label="traj_real_nn")
    ax_bar.set_xticks(x_real)
    ax_bar.set_xticklabels(labels_real, rotation=0)
    ax_bar.set_ylabel("distance to gate [m]", fontsize=12)
    ax_bar.set_title("Gate-time distance error (key metric)", fontsize=15)
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(fontsize=10)
    ax_bar.tick_params(axis="both", labelsize=11)
    for i, val in enumerate(err_norm_real):
        ax_bar.text(i - width / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="C0")
    for i, val in enumerate(err_norm_nn):
        ax_bar.text(i + width / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="C1")

    ax_comp = fig.add_subplot(gs[1, 1])
    colors = ["tab:red", "tab:green", "tab:blue"]
    for j, name in enumerate("xyz"):
        ax_comp.plot(x_real, err_comp_real[:, j], color=colors[j], ls="--", lw=1.7, marker="o", label=f"real e_{name}")
        ax_comp.plot(x_nn, err_comp_nn[:, j], color=colors[j], ls="-", lw=2.0, marker="x", label=f"real_nn e_{name}")
    ax_comp.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_comp.set_xticks(x_real)
    ax_comp.set_xticklabels(labels_real)
    ax_comp.set_ylabel("component error [m]", fontsize=12)
    ax_comp.set_title("Gate-time error components", fontsize=15)
    ax_comp.grid(True, alpha=0.3)
    ax_comp.legend(fontsize=8, ncol=3, loc="upper right")
    ax_comp.tick_params(axis="both", labelsize=11)

    ax_ts = fig.add_subplot(gs[2, 1])
    ref_err = np.linalg.norm(p_real - p_ref, axis=1)
    ref_err_nn = np.linalg.norm(p_nn - p_ref_nn, axis=1)
    ax_ts.plot(t_real, ref_err, color="C0", lw=2.0, label="traj_real vs ref")
    ax_ts.plot(t_nn, ref_err_nn, color="C1", lw=2.0, label="traj_real_nn vs ref")
    for row in rows_real:
        ax_ts.axvline(row["t_plan"], color="C0", ls=":", lw=0.9, alpha=0.45)
        ax_ts.text(row["t_plan"], 0.96, gate_label(row), rotation=90, va="top", ha="right", fontsize=7, color="C0", transform=ax_ts.get_xaxis_transform())
    for row in rows_nn:
        ax_ts.axvline(row["t_plan"], color="C1", ls=":", lw=0.9, alpha=0.45)
    ax_ts.set_xlabel("t [s]", fontsize=12)
    ax_ts.set_ylabel("|p_real - p_ref| [m]", fontsize=12)
    ax_ts.set_title("Position tracking error over time", fontsize=15)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(fontsize=10)
    ax_ts.tick_params(axis="both", labelsize=11)

    fig_path = traj_dir / "traj_real_compare.png"
    plt.show()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"Saved gate error csv: {error_csv}")
    print(f"Saved figure:         {fig_path}")


if __name__ == "__main__":
    main()
