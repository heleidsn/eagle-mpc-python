#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np


def load_error_model_joblib(path: Path):
    """Load sklearn/joblib error models across NumPy 1.x/2.x pickle differences."""
    try:
        return joblib.load(path)
    except (ModuleNotFoundError, ValueError) as first_err:
        import sys as _sys

        try:
            import numpy.core as _np_core

            _sys.modules.setdefault("numpy._core", _np_core)
            for name in (
                "multiarray",
                "numeric",
                "fromnumeric",
                "umath",
                "_multiarray_umath",
                "records",
                "numerictypes",
            ):
                try:
                    mod = __import__(f"numpy.core.{name}", fromlist=["*"])
                    _sys.modules.setdefault(f"numpy._core.{name}", mod)
                except Exception:
                    pass

            class _DummyRng:
                def __init__(self, *args, **kwargs):
                    pass

                def __setstate__(self, state):
                    pass

                def __getstate__(self):
                    return None

            import numpy.random._pickle as _nrp

            _nrp.__bit_generator_ctor = lambda *args, **kwargs: _DummyRng()
            _nrp.__generator_ctor = lambda *args, **kwargs: _DummyRng()
            _nrp.__randomstate_ctor = lambda *args, **kwargs: _DummyRng()
            return joblib.load(path)
        except Exception as second_err:
            raise RuntimeError(
                f"Failed to load error model {path}.\n"
                f"Initial error: {first_err}\n"
                f"Compatibility fallback error: {second_err}"
            ) from second_err


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
        return []
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


def mark_gate_times(ax, events: list[dict], color: str, label_prefix: str, annotate: bool = True) -> None:
    labeled = False
    y_text = 0.96 if str(label_prefix).startswith("no") else 0.84
    for ev in events:
        ls = ":" if ev["kind"] == "gate" else "--"
        label = f"{label_prefix} gates" if not labeled else None
        ax.axvline(float(ev["t"]), color=color, linestyle=ls, linewidth=0.8, alpha=0.55, label=label)
        if annotate:
            gate_label = f"G{int(ev['index'])}" if ev["kind"] == "gate" else "End"
            ax.text(
                float(ev["t"]),
                y_text,
                gate_label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color=color,
                alpha=0.9,
                transform=ax.get_xaxis_transform(),
            )
        labeled = True


def mark_position_tolerance(ax, error_bound: float, *, norm: bool = False) -> None:
    if error_bound <= 0:
        return
    if norm:
        ax.axhline(error_bound, color="crimson", ls="--", lw=1.2, alpha=0.65, label=f"pos tol {error_bound:.2f} m")
    else:
        ax.axhline(error_bound, color="crimson", ls="--", lw=1.1, alpha=0.6, label=f"+tol {error_bound:.2f} m")
        ax.axhline(-error_bound, color="crimson", ls="--", lw=1.1, alpha=0.6, label=f"-tol {error_bound:.2f} m")


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


def safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if x.size else float("nan")


def summarize(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    t = data["t"]
    p = np.column_stack([data["p_x"], data["p_y"], data["p_z"]])
    v = np.column_stack([data["v_x"], data["v_y"], data["v_z"]])
    u = np.column_stack([data["u_1"], data["u_2"], data["u_3"], data["u_4"]])

    speed = np.linalg.norm(v, axis=1)
    seg = np.diff(p, axis=0)
    path_length = float(np.sum(np.linalg.norm(seg, axis=1)))
    duration = float(t[-1] - t[0]) if t.size >= 2 else 0.0

    return {
        "samples": float(len(t)),
        "duration_s": duration,
        "path_length_m": path_length,
        "speed_mean_mps": safe_mean(speed),
        "speed_max_mps": float(np.max(speed)),
        "speed_p95_mps": float(np.percentile(speed, 95)),
        "u_mean": safe_mean(u),
        "u_rms": float(np.sqrt(np.mean(u ** 2))),
        "u_max": float(np.max(u)),
        "z_min_m": float(np.min(p[:, 2])),
        "z_max_m": float(np.max(p[:, 2])),
    }


def _quat_to_body_z_world(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
    r02 = 2.0 * (qx * qz + qw * qy)
    r12 = 2.0 * (qy * qz - qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.column_stack((r02, r12, r22))


def _finite_diff(vec: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vec)
    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    for i in range(vec.shape[1]):
        out[:, i] = np.gradient(vec[:, i]) / dt
    return out


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.unwrap(np.arctan2(siny, cosy))


def _default_feature_names() -> list[str]:
    return [
        "ep_x", "ep_y", "ep_z",
        "ev_x", "ev_y", "ev_z",
        "a_ref_x", "a_ref_y", "a_ref_z",
        "j_ref_x", "j_ref_y", "j_ref_z",
        "yaw_ref", "yaw_rate_ref",
        "v_world_x", "v_world_y", "v_world_z",
        "omega_x", "omega_y", "omega_z",
        "body_z_world_x", "body_z_world_y", "body_z_world_z",
        "u_thrust", "thrust_margin",
    ]


def predict_tracking_error_with_model(
    data: Dict[str, np.ndarray], model_obj: Dict[str, object]
) -> Dict[str, np.ndarray]:
    # Build the exact feature layout requested by the loaded model. Legacy models
    # use the original 25-D layout; model_3 uses plan-only 11-D features.
    t = data["t"]
    p = np.column_stack((data["p_x"], data["p_y"], data["p_z"]))
    v = np.column_stack((data["v_x"], data["v_y"], data["v_z"]))
    q = np.column_stack((data["q_w"], data["q_x"], data["q_y"], data["q_z"]))
    omega = np.column_stack((data["omega_x"], data["omega_y"], data["omega_z"]))
    u = np.column_stack((data["u_1"], data["u_2"], data["u_3"], data["u_4"]))

    # For optimization result csv the planned trajectory itself is the reference.
    ep = np.zeros_like(p)
    ev = np.zeros_like(v)
    a_ref = _finite_diff(v, t)
    j_ref = _finite_diff(a_ref, t)
    yaw_ref = _yaw_from_quat_wxyz(q).reshape((-1, 1))
    yaw_rate_ref = _finite_diff(yaw_ref, t)
    body_z_world = _quat_to_body_z_world(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    u_thrust = np.clip(np.sum(u, axis=1, keepdims=True) / 48.0, 0.0, 1.0)
    thrust_margin = 1.0 - u_thrust

    feature_map = {
        "ep_x": ep[:, 0], "ep_y": ep[:, 1], "ep_z": ep[:, 2],
        "ev_x": ev[:, 0], "ev_y": ev[:, 1], "ev_z": ev[:, 2],
        "v_ref_x": v[:, 0], "v_ref_y": v[:, 1], "v_ref_z": v[:, 2],
        "a_ref_x": a_ref[:, 0], "a_ref_y": a_ref[:, 1], "a_ref_z": a_ref[:, 2],
        "j_ref_x": j_ref[:, 0], "j_ref_y": j_ref[:, 1], "j_ref_z": j_ref[:, 2],
        "yaw_ref": yaw_ref[:, 0],
        "yaw_rate_ref": yaw_rate_ref[:, 0],
        "v_world_x": v[:, 0], "v_world_y": v[:, 1], "v_world_z": v[:, 2],
        "omega_x": omega[:, 0], "omega_y": omega[:, 1], "omega_z": omega[:, 2],
        "body_z_world_x": body_z_world[:, 0],
        "body_z_world_y": body_z_world[:, 1],
        "body_z_world_z": body_z_world[:, 2],
        "u_thrust": u_thrust[:, 0],
        "thrust_margin": thrust_margin[:, 0],
    }
    feature_names = list(model_obj.get("feature_names_per_frame") or _default_feature_names())
    missing = [name for name in feature_names if name not in feature_map]
    if missing:
        raise ValueError(f"Unsupported error model feature(s): {missing}")
    x_frame = np.column_stack([feature_map[name] for name in feature_names])
    scaler = model_obj["scaler"]
    mlp = model_obj["mlp"]
    x_scaled = scaler.transform(x_frame)
    y_pred = mlp.predict(x_scaled)
    target_names = list(
        model_obj.get("target_names")
        or ["ep_x", "ep_y", "ep_z", "ev_x", "ev_y", "ev_z"]
    )

    def _cols(names: tuple[str, str, str], fallback_start: int) -> np.ndarray:
        idx = [target_names.index(n) if n in target_names else fallback_start + i for i, n in enumerate(names)]
        return y_pred[:, idx]

    ep_pred = _cols(("ep_x", "ep_y", "ep_z"), 0)
    ev_pred = _cols(("ev_x", "ev_y", "ev_z"), 3)
    return {
        "ep_pred": ep_pred,
        "ev_pred": ev_pred,
        "ep_pred_norm": np.linalg.norm(ep_pred, axis=1),
        "ev_pred_norm": np.linalg.norm(ev_pred, axis=1),
    }


def format_delta(a: float, b: float) -> str:
    d = b - a
    if abs(a) > 1e-9:
        r = d / a * 100.0
        return f"{d:+.6f} ({r:+.2f}%)"
    return f"{d:+.6f}"


def main() -> None:
    base = Path(__file__).resolve().parent
    cfg = {
        "trajectory_name": "racing_s500_new",
        "error_model_name": "model_3",
        "error_bound": 0.10,
    }
    traj_dir = base / "results" / cfg["trajectory_name"]
    no_err_path = traj_dir / "traj_planned.csv"
    with_err_path = traj_dir / "traj_planned_nn.csv"
    no_gate_path = traj_dir / "traj_gates.csv"
    with_gate_path = traj_dir / "traj_gates_nn.csv"

    candidates = [
        traj_dir / f"{cfg['error_model_name']}.joblib",
        traj_dir / "error_model.joblib",
    ]
    candidates.extend(sorted(traj_dir.glob("*.joblib")))
    candidates.append(base / "error_model" / f"{cfg['error_model_name']}.joblib")
    error_model_path = next((p for p in candidates if p.exists()), None)

    if not no_err_path.exists() or not with_err_path.exists():
        raise FileNotFoundError(
            "Missing input csv. Expected:\n"
            f"- {no_err_path}\n"
            f"- {with_err_path}"
        )
    if error_model_path is None:
        raise FileNotFoundError(
            "Missing error model. Expected one of:\n"
            + "\n".join(f"- {p}" for p in candidates)
        )

    no_err = load_csv(no_err_path)
    with_err = load_csv(with_err_path)
    no_gates = load_gate_events(no_gate_path)
    we_gates = load_gate_events(with_gate_path)
    s_no = summarize(no_err)
    s_we = summarize(with_err)
    model_obj = load_error_model_joblib(error_model_path)
    no_pred = predict_tracking_error_with_model(no_err, model_obj)
    we_pred = predict_tracking_error_with_model(with_err, model_obj)
    error_bound = float(cfg.get("error_bound", 0.0))

    report_path = traj_dir / "traj_compare_summary.txt"
    keys = [
        "samples",
        "duration_s",
        "path_length_m",
        "speed_mean_mps",
        "speed_max_mps",
        "speed_p95_mps",
        "u_mean",
        "u_rms",
        "u_max",
        "z_min_m",
        "z_max_m",
    ]

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Compare no_err_model vs with_err_model: {cfg['trajectory_name']}\n")
        f.write(f"no_err:      {no_err_path}\n")
        f.write(f"with_err:    {with_err_path}\n")
        f.write(f"error_model: {error_model_path}\n\n")
        f.write(f"no_err_gates:   {no_gate_path if no_gates else 'N/A'}\n")
        f.write(f"with_err_gates: {with_gate_path if we_gates else 'N/A'}\n\n")
        f.write(f"position_error_bound: {error_bound:.6f} m\n\n")
        for k in keys:
            a = s_no[k]
            b = s_we[k]
            f.write(
                f"{k:16s} | no_err={a:12.6f} | with_err={b:12.6f} | delta={format_delta(a, b)}\n"
            )
        f.write("\nPredicted tracking error (error model):\n")
        ep_no = float(np.mean(no_pred["ep_pred_norm"]))
        ep_we = float(np.mean(we_pred["ep_pred_norm"]))
        ev_no = float(np.mean(no_pred["ev_pred_norm"]))
        ev_we = float(np.mean(we_pred["ev_pred_norm"]))
        f.write(
            f"{'ep_pred_mean':16s} | no_err={ep_no:12.6f} | with_err={ep_we:12.6f} | delta={format_delta(ep_no, ep_we)}\n"
        )
        f.write(
            f"{'ev_pred_mean':16s} | no_err={ev_no:12.6f} | with_err={ev_we:12.6f} | delta={format_delta(ev_no, ev_we)}\n"
        )

    no_speed = np.linalg.norm(
        np.column_stack([no_err["v_x"], no_err["v_y"], no_err["v_z"]]), axis=1
    )
    we_speed = np.linalg.norm(
        np.column_stack([with_err["v_x"], with_err["v_y"], with_err["v_z"]]), axis=1
    )
    no_u = np.sum(np.column_stack([no_err["u_1"], no_err["u_2"], no_err["u_3"], no_err["u_4"]]), axis=1)
    we_u = np.sum(np.column_stack([with_err["u_1"], with_err["u_2"], with_err["u_3"], with_err["u_4"]]), axis=1)

    # Main figure: error-centric layout for 4K screens.
    line_w = 2.2
    thin_w = 1.5
    title_fs = 15
    label_fs = 13
    tick_fs = 11
    legend_fs = 10

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Predicted tracking error comparison: {cfg['trajectory_name']} | error model: {error_model_path.name}",
        fontsize=18,
        y=0.985,
    )
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.42,
        wspace=0.22,
        left=0.04,
        right=0.985,
        top=0.94,
        bottom=0.055,
    )

    p_no = np.column_stack([no_err["p_x"], no_err["p_y"], no_err["p_z"]])
    p_we = np.column_stack([with_err["p_x"], with_err["p_y"], with_err["p_z"]])

    ax_norm = fig.add_subplot(gs[0, 0])
    ax_norm.plot(no_err["t"], no_pred["ep_pred_norm"], color="C0", lw=line_w, label=r"planned $\|e_p\|$")
    ax_norm.plot(with_err["t"], we_pred["ep_pred_norm"], color="C1", lw=line_w, label=r"planned_nn $\|e_p\|$")
    mark_position_tolerance(ax_norm, error_bound, norm=True)
    mark_gate_times(ax_norm, no_gates, "0.55", "no_err")
    mark_gate_times(ax_norm, we_gates, "0.20", "with_err")
    ax_norm.set_title("Predicted position error norm", fontsize=title_fs)
    ax_norm.set_ylabel(r"$\|e_p\|$ [m]", fontsize=label_fs)
    ax_norm.tick_params(axis="both", labelsize=tick_fs)
    ax_norm.grid(True, alpha=0.3)
    ax_norm.legend(fontsize=legend_fs, loc="upper right")

    ax_ev_norm = fig.add_subplot(gs[0, 1], sharex=ax_norm)
    ax_ev_norm.plot(no_err["t"], no_pred["ev_pred_norm"], color="C0", lw=line_w, label=r"planned $\|e_v\|$")
    ax_ev_norm.plot(with_err["t"], we_pred["ev_pred_norm"], color="C1", lw=line_w, label=r"planned_nn $\|e_v\|$")
    mark_gate_times(ax_ev_norm, no_gates, "0.55", "no_err")
    mark_gate_times(ax_ev_norm, we_gates, "0.20", "with_err")
    ax_ev_norm.set_title("Predicted velocity error norm", fontsize=title_fs)
    ax_ev_norm.set_ylabel(r"$\|e_v\|$ [m/s]", fontsize=label_fs)
    ax_ev_norm.tick_params(axis="both", labelsize=tick_fs)
    ax_ev_norm.grid(True, alpha=0.3)
    ax_ev_norm.legend(fontsize=legend_fs, loc="upper right")

    err_axes = []
    for j, name in enumerate("xyz"):
        ax = fig.add_subplot(gs[j + 1, 0], sharex=ax_norm)
        ax.plot(no_err["t"], no_pred["ep_pred"][:, j], color="C0", ls="--", lw=thin_w, alpha=0.85, label="planned")
        ax.plot(with_err["t"], we_pred["ep_pred"][:, j], color="C1", ls="-", lw=line_w, label="planned_nn")
        mark_gate_times(ax, no_gates, "0.55", "no_err")
        mark_gate_times(ax, we_gates, "0.20", "with_err")
        mark_position_tolerance(ax, error_bound, norm=False)
        ax.axhline(0.0, color="k", lw=0.9, alpha=0.4)
        ax.set_title(f"Predicted position error e_p{name}")
        ax.set_ylabel("[m]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=legend_fs, loc="upper right")
        ax.set_title(f"Position error e_p{name}", fontsize=title_fs)
        ax.set_ylabel("[m]", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        err_axes.append(ax)

    vel_err_axes = []
    for j, name in enumerate("xyz"):
        ax = fig.add_subplot(gs[j + 1, 1], sharex=ax_norm)
        ax.plot(no_err["t"], no_pred["ev_pred"][:, j], color="C0", ls="--", lw=thin_w, alpha=0.85, label="planned")
        ax.plot(with_err["t"], we_pred["ev_pred"][:, j], color="C1", ls="-", lw=line_w, label="planned_nn")
        mark_gate_times(ax, no_gates, "0.55", "no_err")
        mark_gate_times(ax, we_gates, "0.20", "with_err")
        ax.axhline(0.0, color="k", lw=0.9, alpha=0.4)
        ax.set_title(f"Velocity error e_v{name}", fontsize=title_fs)
        ax.set_ylabel("[m/s]", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.3)
        vel_err_axes.append(ax)

    lines = [
        "Summary (no -> nn)",
        f"duration [s]: {s_no['duration_s']:.3f} -> {s_we['duration_s']:.3f} | {format_delta(s_no['duration_s'], s_we['duration_s'])}",
        f"path length [m]: {s_no['path_length_m']:.3f} -> {s_we['path_length_m']:.3f} | {format_delta(s_no['path_length_m'], s_we['path_length_m'])}",
        f"speed max [m/s]: {s_no['speed_max_mps']:.3f} -> {s_we['speed_max_mps']:.3f} | {format_delta(s_no['speed_max_mps'], s_we['speed_max_mps'])}",
        f"u rms: {s_no['u_rms']:.3f} -> {s_we['u_rms']:.3f} | {format_delta(s_no['u_rms'], s_we['u_rms'])}",
        f"mean |e_p| [m]: {ep_no:.4f} -> {ep_we:.4f} | {format_delta(ep_no, ep_we)}",
        f"mean |e_v| [m/s]: {ev_no:.4f} -> {ev_we:.4f} | {format_delta(ev_no, ev_we)}",
        f"gate csv: {len(no_gates)} no / {len(we_gates)} nn events",
    ]

    for ax in (ax_norm, ax_ev_norm, *err_axes, *vel_err_axes):
        ax.set_xlabel("t [s]", fontsize=label_fs)

    fig_path = traj_dir / "traj_compare.png"

    # Overview figure: trajectory geometry and secondary metrics.
    fig_overview = plt.figure(figsize=(18, 10))
    fig_overview.suptitle(
        f"Trajectory overview: {cfg['trajectory_name']} | error model: {error_model_path.name}",
        fontsize=17,
        y=0.98,
    )
    gs_over = fig_overview.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1.0, 0.9],
        hspace=0.32,
        wspace=0.28,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.08,
    )
    ax3 = fig_overview.add_subplot(gs_over[:, 0], projection="3d")
    ax3.plot(p_no[:, 0], p_no[:, 1], p_no[:, 2], color="C0", lw=2.5, label="traj_planned")
    ax3.plot(p_we[:, 0], p_we[:, 1], p_we[:, 2], color="C1", lw=2.5, label="traj_planned_nn")
    ax3.scatter(p_no[0, 0], p_no[0, 1], p_no[0, 2], color="green", s=100, label="start")
    ax3.scatter(p_no[-1, 0], p_no[-1, 1], p_no[-1, 2], color="crimson", s=100, label="end")
    gate_pts = []
    if no_gates:
        G = np.asarray([[ev["x"], ev["y"], ev["z"]] for ev in no_gates if ev["kind"] == "gate"], dtype=float)
        if G.size:
            ax3.scatter(G[:, 0], G[:, 1], G[:, 2], marker="o", s=75, color="C0", alpha=0.75, label="planned gates")
            gate_pts.append(G)
    if we_gates:
        G = np.asarray([[ev["x"], ev["y"], ev["z"]] for ev in we_gates if ev["kind"] == "gate"], dtype=float)
        if G.size:
            ax3.scatter(G[:, 0], G[:, 1], G[:, 2], marker="x", s=105, color="C1", alpha=0.85, label="nn gates")
            gate_pts.append(G)
    ax3.set_xlabel("X [m]", fontsize=12, labelpad=9)
    ax3.set_ylabel("Y [m]", fontsize=12, labelpad=9)
    ax3.set_zlabel("Z [m]", fontsize=12, labelpad=9)
    ax3.set_title("3D trajectory and gate locations", fontsize=14)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.tick_params(axis="both", labelsize=10)
    ax3.grid(True, alpha=0.3)
    set_3d_equal(ax3, p_no, p_we, *gate_pts)

    ax_u = fig_overview.add_subplot(gs_over[0, 1])
    ax_u.plot(no_err["t"], no_u, color="C0", lw=2.0, label="planned sum(u)")
    ax_u.plot(with_err["t"], we_u, color="C1", lw=2.0, label="planned_nn sum(u)")
    mark_gate_times(ax_u, no_gates, "0.55", "no_err")
    mark_gate_times(ax_u, we_gates, "0.20", "with_err")
    ax_u.set_title("Total control", fontsize=13)
    ax_u.set_xlabel("t [s]", fontsize=11)
    ax_u.set_ylabel("sum(u)", fontsize=11)
    ax_u.tick_params(axis="both", labelsize=10)
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(fontsize=9, loc="upper right")

    ax_info = fig_overview.add_subplot(gs_over[1, 1])
    ax_info.axis("off")
    ax_info.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax_info.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.9, edgecolor="0.8"),
    )
    ax_info.set_title("Metrics", loc="left", fontsize=13)

    overview_path = traj_dir / "traj_compare_overview.png"
    plt.show()
    fig.savefig(fig_path, dpi=160)
    fig_overview.savefig(overview_path, dpi=160)
    plt.close(fig)
    plt.close(fig_overview)

    print(f"Saved summary: {report_path}")
    print(f"Saved figure:  {fig_path}")
    print(f"Saved overview: {overview_path}")


if __name__ == "__main__":
    main()

