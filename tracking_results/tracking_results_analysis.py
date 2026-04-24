from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import csv


@dataclass
class TrajectoryPair:
    name: str
    plan_path: Path
    tracking_path: Path


def read_csv_with_header(path: Path) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.ndim == 0:
        # A single-row csv from genfromtxt may become scalar structured array.
        data = np.array([data], dtype=data.dtype)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def build_pairs(base_dir: Path) -> List[TrajectoryPair]:
    plan_files = sorted(base_dir.glob("*_plan.csv"))
    pairs: List[TrajectoryPair] = []
    for plan_path in plan_files:
        base_name = plan_path.name[: -len("_plan.csv")]
        tracking_path = base_dir / f"{base_name}__px4.csv"
        if tracking_path.exists():
            pairs.append(TrajectoryPair(base_name, plan_path, tracking_path))
    return pairs


def interpolate_plan_to_tracking(plan: Dict[str, np.ndarray], tracking_time: np.ndarray) -> Dict[str, np.ndarray]:
    plan_t = plan["t"]
    t_clip = np.clip(tracking_time, plan_t[0], plan_t[-1])
    return {
        "px": np.interp(t_clip, plan_t, plan["px"]),
        "py": np.interp(t_clip, plan_t, plan["py"]),
        "pz": np.interp(t_clip, plan_t, plan["pz"]),
        "vx": np.interp(t_clip, plan_t, plan["vx"]),
        "vy": np.interp(t_clip, plan_t, plan["vy"]),
        "vz": np.interp(t_clip, plan_t, plan["vz"]),
        "ax": np.interp(t_clip, plan_t, plan["ax"]),
        "ay": np.interp(t_clip, plan_t, plan["ay"]),
        "az": np.interp(t_clip, plan_t, plan["az"]),
        "speed": np.interp(
            t_clip,
            plan_t,
            np.linalg.norm(np.column_stack((plan["vx"], plan["vy"], plan["vz"])), axis=1),
        ),
    }


def rotate_body_velocity_to_world(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    qw: np.ndarray,
    vx_b: np.ndarray,
    vy_b: np.ndarray,
    vz_b: np.ndarray,
) -> np.ndarray:
    # Assume quaternion (x,y,z,w) describes body->world orientation.
    # Convert body-frame velocity to world-frame: v_w = R(q) * v_b.
    x = qx
    y = qy
    z = qz
    w = qw

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    vx_w = r00 * vx_b + r01 * vy_b + r02 * vz_b
    vy_w = r10 * vx_b + r11 * vy_b + r12 * vz_b
    vz_w = r20 * vx_b + r21 * vy_b + r22 * vz_b
    return np.column_stack((vx_w, vy_w, vz_w))


def compute_tracking_speed_world(tracking: Dict[str, np.ndarray]) -> np.ndarray:
    required = {"qx", "qy", "qz", "qw", "vx_b", "vy_b", "vz_b"}
    if not required.issubset(tracking.keys()):
        missing = sorted(required - set(tracking.keys()))
        raise KeyError(f"Tracking CSV missing required columns for body->world velocity conversion: {missing}")
    v_world = rotate_body_velocity_to_world(
        tracking["qx"],
        tracking["qy"],
        tracking["qz"],
        tracking["qw"],
        tracking["vx_b"],
        tracking["vy_b"],
        tracking["vz_b"],
    )
    return np.linalg.norm(v_world, axis=1)


def fit_error_speed_models(error: np.ndarray, tracking_speed: np.ndarray, plan_speed: np.ndarray) -> Dict[str, float]:
    x1 = tracking_speed
    x2 = np.abs(tracking_speed - plan_speed)
    y = error

    valid = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    x1 = x1[valid]
    x2 = x2[valid]
    y = y[valid]
    if y.size < 10:
        return {"samples": float(y.size)}

    # Linear model: error = a0 + a1*|v_track| + a2*|v_track-v_plan|
    a_linear = np.column_stack((np.ones_like(x1), x1, x2))
    coef_linear, *_ = np.linalg.lstsq(a_linear, y, rcond=None)
    pred_linear = a_linear @ coef_linear

    # Quadratic model: add second-order terms for a better nonlinear fit.
    a_quad = np.column_stack((np.ones_like(x1), x1, x2, x1 * x1, x2 * x2, x1 * x2))
    coef_quad, *_ = np.linalg.lstsq(a_quad, y, rcond=None)
    pred_quad = a_quad @ coef_quad

    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot < 1e-12:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    return {
        "samples": float(y.size),
        "linear_coef_0": float(coef_linear[0]),
        "linear_coef_vtrack": float(coef_linear[1]),
        "linear_coef_vdiff": float(coef_linear[2]),
        "linear_r2": r2(y, pred_linear),
        "quad_coef_0": float(coef_quad[0]),
        "quad_coef_vtrack": float(coef_quad[1]),
        "quad_coef_vdiff": float(coef_quad[2]),
        "quad_coef_vtrack2": float(coef_quad[3]),
        "quad_coef_vdiff2": float(coef_quad[4]),
        "quad_coef_cross": float(coef_quad[5]),
        "quad_r2": r2(y, pred_quad),
        "mean_error": float(np.mean(y)),
        "max_error": float(np.max(y)),
    }


def predict_error_from_metrics(
    metrics: Dict[str, float], tracking_speed: np.ndarray, plan_speed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    x1 = tracking_speed
    x2 = np.abs(tracking_speed - plan_speed)

    y_linear = (
        metrics["linear_coef_0"]
        + metrics["linear_coef_vtrack"] * x1
        + metrics["linear_coef_vdiff"] * x2
    )
    y_quad = (
        metrics["quad_coef_0"]
        + metrics["quad_coef_vtrack"] * x1
        + metrics["quad_coef_vdiff"] * x2
        + metrics["quad_coef_vtrack2"] * x1 * x1
        + metrics["quad_coef_vdiff2"] * x2 * x2
        + metrics["quad_coef_cross"] * x1 * x2
    )
    return y_linear, y_quad


def save_fit_plot_for_pair(
    pair_name: str,
    pos_error: np.ndarray,
    tracking_speed_world: np.ndarray,
    plan_speed: np.ndarray,
    metrics: Dict[str, float],
    output_dir: Path,
) -> None:
    required_keys = {
        "linear_coef_0",
        "linear_coef_vtrack",
        "linear_coef_vdiff",
        "quad_coef_0",
        "quad_coef_vtrack",
        "quad_coef_vdiff",
        "quad_coef_vtrack2",
        "quad_coef_vdiff2",
        "quad_coef_cross",
    }
    if not required_keys.issubset(metrics.keys()):
        return

    valid = np.isfinite(pos_error) & np.isfinite(tracking_speed_world) & np.isfinite(plan_speed)
    y = pos_error[valid]
    x1 = tracking_speed_world[valid]
    x2 = np.abs(tracking_speed_world[valid] - plan_speed[valid])
    if y.size < 10:
        return

    y_linear, y_quad = predict_error_from_metrics(metrics, x1, plan_speed[valid])

    fig = plt.figure(figsize=(14, 4.8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # 1) Error vs world speed
    ax1.scatter(x1, y, s=8, alpha=0.35, label="samples")
    order = np.argsort(x1)
    ax1.plot(x1[order], y_linear[order], linewidth=2, label="linear pred")
    ax1.plot(x1[order], y_quad[order], linewidth=2, label="quadratic pred")
    ax1.set_title("Error vs |v_track_world|")
    ax1.set_xlabel("|v_track_world| [m/s]")
    ax1.set_ylabel("error [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Error vs speed mismatch
    ax2.scatter(x2, y, s=8, alpha=0.35, label="samples")
    order2 = np.argsort(x2)
    ax2.plot(x2[order2], y_linear[order2], linewidth=2, label="linear pred")
    ax2.plot(x2[order2], y_quad[order2], linewidth=2, label="quadratic pred")
    ax2.set_title("Error vs |v_track-v_plan|")
    ax2.set_xlabel("|v_track-v_plan| [m/s]")
    ax2.set_ylabel("error [m]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) Predicted vs measured error
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    ax3.scatter(y, y_linear, s=8, alpha=0.35, label=f"linear (R2={metrics['linear_r2']:.3f})")
    ax3.scatter(y, y_quad, s=8, alpha=0.35, label=f"quad (R2={metrics['quad_r2']:.3f})")
    ax3.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1.2, label="ideal y=x")
    ax3.set_title("Predicted vs Measured Error")
    ax3.set_xlabel("measured error [m]")
    ax3.set_ylabel("predicted error [m]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.suptitle(f"{pair_name} - Fit Visualization", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"{pair_name}_fit.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_plot_for_pair(
    pair_name: str,
    plan_interp: Dict[str, np.ndarray],
    tracking: Dict[str, np.ndarray],
    tracking_speed_world: np.ndarray,
    pos_error: np.ndarray,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(16, 10))

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 3, 2)
    ax_xz = fig.add_subplot(2, 3, 3)
    ax_v = fig.add_subplot(2, 3, 4)
    ax_vcomp = fig.add_subplot(2, 3, 5)
    ax_err = fig.add_subplot(2, 3, 6)

    # 3D trajectory
    ax3d.plot(plan_interp["px"], plan_interp["py"], plan_interp["pz"], label="plan(interp)", linewidth=2)
    ax3d.plot(tracking["px"], tracking["py"], tracking["pz"], label="tracking", linewidth=1.5)
    xyz_all = np.column_stack(
        (
            np.concatenate((plan_interp["px"], tracking["px"])),
            np.concatenate((plan_interp["py"], tracking["py"])),
            np.concatenate((plan_interp["pz"], tracking["pz"])),
        )
    )
    mins = np.min(xyz_all, axis=0)
    maxs = np.max(xyz_all, axis=0)
    centers = 0.5 * (mins + maxs)
    half_range = 0.5 * np.max(maxs - mins)
    if half_range < 1e-9:
        half_range = 1.0
    ax3d.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax3d.set_ylim(centers[1] - half_range, centers[1] + half_range)
    ax3d.set_zlim(centers[2] - half_range, centers[2] + half_range)
    ax3d.set_title(f"{pair_name} - 3D Trajectory")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend()

    # XY plane
    ax_xy.plot(plan_interp["px"], plan_interp["py"], label="plan(interp)", linewidth=2)
    ax_xy.plot(tracking["px"], tracking["py"], label="tracking", linewidth=1.5)
    xy_all = np.column_stack(
        (
            np.concatenate((plan_interp["px"], tracking["px"])),
            np.concatenate((plan_interp["py"], tracking["py"])),
        )
    )
    xy_mins = np.min(xy_all, axis=0)
    xy_maxs = np.max(xy_all, axis=0)
    xy_center = 0.5 * (xy_mins + xy_maxs)
    xy_half_range = 0.5 * np.max(xy_maxs - xy_mins)
    if xy_half_range < 1e-9:
        xy_half_range = 1.0
    ax_xy.set_xlim(xy_center[0] - xy_half_range, xy_center[0] + xy_half_range)
    ax_xy.set_ylim(xy_center[1] - xy_half_range, xy_center[1] + xy_half_range)
    ax_xy.set_title("XY Plane")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend()

    # XZ plane
    ax_xz.plot(plan_interp["px"], plan_interp["pz"], label="plan(interp)", linewidth=2)
    ax_xz.plot(tracking["px"], tracking["pz"], label="tracking", linewidth=1.5)
    xz_all = np.column_stack(
        (
            np.concatenate((plan_interp["px"], tracking["px"])),
            np.concatenate((plan_interp["pz"], tracking["pz"])),
        )
    )
    xz_mins = np.min(xz_all, axis=0)
    xz_maxs = np.max(xz_all, axis=0)
    xz_center = 0.5 * (xz_mins + xz_maxs)
    xz_half_range = 0.5 * np.max(xz_maxs - xz_mins)
    if xz_half_range < 1e-9:
        xz_half_range = 1.0
    ax_xz.set_xlim(xz_center[0] - xz_half_range, xz_center[0] + xz_half_range)
    ax_xz.set_ylim(xz_center[1] - xz_half_range, xz_center[1] + xz_half_range)
    ax_xz.set_title("XZ Plane")
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend()

    # Speed tracking
    time_s = tracking["time"]
    ax_v.plot(time_s, plan_interp["speed"], label="|v_plan|", linewidth=2)
    ax_v.plot(time_s, tracking_speed_world, label="|v_track_world| (from v_body + q)", linewidth=1.5)
    if {"vx_b", "vy_b", "vz_b"}.issubset(tracking.keys()):
        speed_body = np.linalg.norm(
            np.column_stack((tracking["vx_b"], tracking["vy_b"], tracking["vz_b"])),
            axis=1,
        )
        ax_v.plot(time_s, speed_body, label="|v_body|", linewidth=1.0, alpha=0.8)
    ax_v.set_title("Speed Tracking")
    ax_v.set_xlabel("time [s]")
    ax_v.set_ylabel("speed [m/s]")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

    # Velocity component tracking in world frame
    v_world = rotate_body_velocity_to_world(
        tracking["qx"],
        tracking["qy"],
        tracking["qz"],
        tracking["qw"],
        tracking["vx_b"],
        tracking["vy_b"],
        tracking["vz_b"],
    )
    ax_vcomp.plot(time_s, plan_interp["vx"], label="plan vx", linewidth=1.8)
    ax_vcomp.plot(time_s, v_world[:, 0], label="track vx(world)", linewidth=1.2)
    ax_vcomp.plot(time_s, plan_interp["vy"], label="plan vy", linewidth=1.8)
    ax_vcomp.plot(time_s, v_world[:, 1], label="track vy(world)", linewidth=1.2)
    ax_vcomp.plot(time_s, plan_interp["vz"], label="plan vz", linewidth=1.8)
    ax_vcomp.plot(time_s, v_world[:, 2], label="track vz(world)", linewidth=1.2)
    ax_vcomp.set_title("Velocity Components (World)")
    ax_vcomp.set_xlabel("time [s]")
    ax_vcomp.set_ylabel("velocity [m/s]")
    ax_vcomp.grid(True, alpha=0.3)
    ax_vcomp.legend(ncol=2, fontsize=8)

    # Tracking position error curve
    ax_err.plot(time_s, pos_error, label="position error", linewidth=1.5, color="tab:red")
    ax_err.axhline(np.mean(pos_error), color="tab:gray", linestyle="--", linewidth=1.2, label="mean error")
    ax_err.set_title("Tracking Error")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("error [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()

    fig.tight_layout()
    out_path = output_dir / f"{pair_name}_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_model_report(results: List[Tuple[str, Dict[str, float]]], out_txt: Path) -> None:
    lines: List[str] = []
    lines.append("Tracking error vs speed fitting report")
    lines.append("=" * 42)
    for name, metrics in results:
        lines.append(f"\n[{name}]")
        for key, value in metrics.items():
            if np.isnan(value):
                lines.append(f"{key}: nan")
            else:
                lines.append(f"{key}: {value:.6f}")
        if {"linear_coef_0", "linear_coef_vtrack", "linear_coef_vdiff"}.issubset(metrics):
            lines.append(
                "linear model: error = c0 + c1*|v_track| + c2*|v_track-v_plan|"
            )
        if {"quad_coef_0", "quad_coef_vtrack", "quad_coef_vdiff", "quad_coef_vtrack2", "quad_coef_vdiff2", "quad_coef_cross"}.issubset(metrics):
            lines.append(
                "quadratic model: error = c0 + c1*|v_track| + c2*|v_track-v_plan| + c3*|v_track|^2 + c4*|v_track-v_plan|^2 + c5*|v_track|*|v_track-v_plan|"
            )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metrics_csv(results: List[Tuple[str, Dict[str, float]]], out_csv: Path) -> None:
    rows: List[Dict[str, object]] = []
    for section, metrics in results:
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "section": section,
                    "metric": metric_name,
                    "value": float(metric_value),
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack((np.ones_like(x), x))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    return coef, pred, _r2_score(y, pred)


def _fit_quadratic(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack((np.ones_like(x), x, x * x))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    return coef, pred, _r2_score(y, pred)


def _fit_multi_linear(features: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack((np.ones(features.shape[0]), features))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    return coef, pred, _r2_score(y, pred)


def _fit_multi_quadratic(features: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    n = features.shape[1]
    columns = [np.ones(features.shape[0])]
    for i in range(n):
        columns.append(features[:, i])
    for i in range(n):
        columns.append(features[:, i] * features[:, i])
    for i in range(n):
        for j in range(i + 1, n):
            columns.append(features[:, i] * features[:, j])
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    return coef, pred, _r2_score(y, pred)


def save_global_fit_plot(
    all_error: np.ndarray,
    all_tracking_speed_world: np.ndarray,
    all_plan_speed: np.ndarray,
    all_plan_acc: np.ndarray,
    all_plan_jerk: np.ndarray,
    all_plan_snap: np.ndarray,
    output_dir: Path,
) -> Dict[str, float]:
    valid = (
        np.isfinite(all_error)
        & np.isfinite(all_tracking_speed_world)
        & np.isfinite(all_plan_speed)
        & np.isfinite(all_plan_acc)
        & np.isfinite(all_plan_jerk)
        & np.isfinite(all_plan_snap)
    )
    y = all_error[valid]
    v_track = all_tracking_speed_world[valid]
    v_plan = all_plan_speed[valid]
    v_diff = np.abs(v_track - v_plan)
    a_plan = all_plan_acc[valid]
    jerk_plan = all_plan_jerk[valid]
    snap_plan = all_plan_snap[valid]

    feature_map = {
        "v_track": v_track,
        "v_plan": v_plan,
        "v_diff": v_diff,
        "acc": a_plan,
        "jerk": jerk_plan,
        "snap": snap_plan,
    }

    single_linear_r2: Dict[str, float] = {}
    single_quad_r2: Dict[str, float] = {}
    single_linear_coef: Dict[str, np.ndarray] = {}
    single_quad_coef: Dict[str, np.ndarray] = {}
    for name, x in feature_map.items():
        coef_l, pred_l, r2_l = _fit_linear(x, y)
        coef_q, pred_q, r2_q = _fit_quadratic(x, y)
        single_linear_r2[name] = r2_l
        single_quad_r2[name] = r2_q
        single_linear_coef[name] = coef_l
        single_quad_coef[name] = coef_q

    features = np.column_stack((v_track, v_plan, v_diff, a_plan, jerk_plan, snap_plan))
    coef_multi_lin, pred_multi_lin, r2_multi_lin = _fit_multi_linear(features, y)
    coef_multi_quad, pred_multi_quad, r2_multi_quad = _fit_multi_quadratic(features, y)

    # Sensitivity: standardized multi-linear coefficients (absolute value).
    f_mean = np.mean(features, axis=0, keepdims=True)
    f_std = np.std(features, axis=0, keepdims=True) + 1e-12
    y_mean = np.mean(y)
    y_std = np.std(y) + 1e-12
    features_z = (features - f_mean) / f_std
    y_z = (y - y_mean) / y_std
    coef_z, _, _ = _fit_multi_linear(features_z, y_z)
    sensitivity = np.abs(coef_z[1:])

    fig = plt.figure(figsize=(20, 14))
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)
    ax4 = fig.add_subplot(3, 3, 4)
    ax5 = fig.add_subplot(3, 3, 5)
    ax6 = fig.add_subplot(3, 3, 6)
    ax7 = fig.add_subplot(3, 3, 7)
    ax8 = fig.add_subplot(3, 3, 8)
    ax9 = fig.add_subplot(3, 3, 9)

    feature_names = list(feature_map.keys())
    idx = np.arange(len(feature_names))

    ax1.bar(idx, sensitivity)
    ax1.set_xticks(idx)
    ax1.set_xticklabels(feature_names, rotation=25)
    ax1.set_title("Global Sensitivity (std linear coef abs)")
    ax1.set_ylabel("sensitivity")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(idx - 0.2, [single_linear_r2[k] for k in feature_names], width=0.4, label="linear")
    ax2.bar(idx + 0.2, [single_quad_r2[k] for k in feature_names], width=0.4, label="quadratic")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(feature_names, rotation=25)
    ax2.set_title("Single-Variable Fit R2")
    ax2.set_ylabel("R2")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    ax3.scatter(y, pred_multi_lin, s=6, alpha=0.25, label=f"multi-linear R2={r2_multi_lin:.3f}")
    ax3.scatter(y, pred_multi_quad, s=6, alpha=0.25, label=f"multi-quad R2={r2_multi_quad:.3f}")
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    ax3.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1.2, label="ideal")
    ax3.set_title("Predicted vs Measured (multivariate)")
    ax3.set_xlabel("measured error [m]")
    ax3.set_ylabel("predicted error [m]")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    for ax, name in zip((ax4, ax5, ax6, ax7), ("v_diff", "acc", "jerk", "snap")):
        x = feature_map[name]
        ax.scatter(x, y, s=6, alpha=0.2, label="samples")
        order = np.argsort(x)
        coef_l = single_linear_coef[name]
        coef_q = single_quad_coef[name]
        y_l = coef_l[0] + coef_l[1] * x
        y_q = coef_q[0] + coef_q[1] * x + coef_q[2] * x * x
        ax.plot(x[order], y_l[order], linewidth=2, label=f"L1 R2={single_linear_r2[name]:.3f}")
        ax.plot(x[order], y_q[order], linewidth=2, label=f"L2 R2={single_quad_r2[name]:.3f}")
        ax.set_title(f"error vs {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("error [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax8.scatter(v_diff, y, s=6, alpha=0.2, label="samples")
    order = np.argsort(v_diff)
    coef_l = single_linear_coef["v_diff"]
    coef_q = single_quad_coef["v_diff"]
    y_l = coef_l[0] + coef_l[1] * v_diff
    y_q = coef_q[0] + coef_q[1] * v_diff + coef_q[2] * v_diff * v_diff
    ax8.plot(v_diff[order], y_l[order], linewidth=2, label=f"linear R2={single_linear_r2['v_diff']:.3f}")
    ax8.plot(v_diff[order], y_q[order], linewidth=2, label=f"quadratic R2={single_quad_r2['v_diff']:.3f}")
    ax8.set_title("Dominant variable deep dive: v_diff")
    ax8.set_xlabel("|v_track-v_plan| [m/s]")
    ax8.set_ylabel("error [m]")
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    text_lines = [
        f"samples: {y.size}",
        f"multi-linear R2: {r2_multi_lin:.4f}",
        f"multi-quadratic R2: {r2_multi_quad:.4f}",
        "single-variable R2 (linear / quadratic):",
    ]
    for name in feature_names:
        text_lines.append(f"- {name}: {single_linear_r2[name]:.4f} / {single_quad_r2[name]:.4f}")
    text_lines.append("std sensitivity ranking:")
    rank_idx = np.argsort(-sensitivity)
    for i in rank_idx:
        text_lines.append(f"- {feature_names[i]}: {sensitivity[i]:.4f}")
    ax9.axis("off")
    ax9.text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("All Trajectories: Detailed Global Fit and Sensitivity Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "all_trajectories_fit.png", dpi=150)
    plt.close(fig)

    metrics: Dict[str, float] = {
        "samples": float(y.size),
        "r2_multi_linear": r2_multi_lin,
        "r2_multi_quadratic": r2_multi_quad,
    }
    for name in feature_names:
        metrics[f"r2_linear_{name}"] = single_linear_r2[name]
        metrics[f"r2_quadratic_{name}"] = single_quad_r2[name]
    for i, name in enumerate(feature_names):
        metrics[f"sensitivity_{name}"] = float(sensitivity[i])
        metrics[f"coef_multi_linear_{name}"] = float(coef_multi_lin[i + 1])
    return metrics


def compute_jerk_snap_magnitude(time_s: np.ndarray, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dt = np.gradient(time_s)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    jx = np.gradient(ax) / dt
    jy = np.gradient(ay) / dt
    jz = np.gradient(az) / dt
    sx = np.gradient(jx) / dt
    sy = np.gradient(jy) / dt
    sz = np.gradient(jz) / dt
    jerk = np.sqrt(jx * jx + jy * jy + jz * jz)
    snap = np.sqrt(sx * sx + sy * sy + sz * sz)
    return jerk, snap


def save_multivariate_analysis_plot(
    all_error: np.ndarray,
    all_v: np.ndarray,
    all_a: np.ndarray,
    all_j: np.ndarray,
    all_s: np.ndarray,
    output_dir: Path,
) -> Dict[str, float]:
    valid = (
        np.isfinite(all_error)
        & np.isfinite(all_v)
        & np.isfinite(all_a)
        & np.isfinite(all_j)
        & np.isfinite(all_s)
    )
    y = all_error[valid]
    v = all_v[valid]
    a = all_a[valid]
    j = all_j[valid]
    s = all_s[valid]

    def single_r2(x: np.ndarray) -> float:
        mat = np.column_stack((np.ones_like(x), x))
        coef, *_ = np.linalg.lstsq(mat, y, rcond=None)
        pred = mat @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot

    X = np.column_stack((np.ones_like(v), v, a, j, s))
    coef_multi, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred_multi = X @ coef_multi
    ss_res = float(np.sum((y - pred_multi) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    multi_r2 = 1.0 - ss_res / ss_tot

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)

    scatter_alpha = 0.25
    scatter_size = 7
    ax1.scatter(v, y, s=scatter_size, alpha=scatter_alpha)
    ax1.set_title("error vs |v_plan|")
    ax1.set_xlabel("|v_plan| [m/s]")
    ax1.set_ylabel("error [m]")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(a, y, s=scatter_size, alpha=scatter_alpha)
    ax2.set_title("error vs |a_plan|")
    ax2.set_xlabel("|a_plan| [m/s^2]")
    ax2.set_ylabel("error [m]")
    ax2.grid(True, alpha=0.3)

    ax3.scatter(j, y, s=scatter_size, alpha=scatter_alpha)
    ax3.set_title("error vs |jerk_plan|")
    ax3.set_xlabel("|jerk_plan| [m/s^3]")
    ax3.set_ylabel("error [m]")
    ax3.grid(True, alpha=0.3)

    ax4.scatter(s, y, s=scatter_size, alpha=scatter_alpha)
    ax4.set_title("error vs |snap_plan|")
    ax4.set_xlabel("|snap_plan| [m/s^4]")
    ax4.set_ylabel("error [m]")
    ax4.grid(True, alpha=0.3)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    ax5.scatter(y, pred_multi, s=scatter_size, alpha=scatter_alpha, label=f"multi fit R2={multi_r2:.3f}")
    ax5.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1.2, label="ideal y=x")
    ax5.set_title("predicted vs measured (v,a,j,s)")
    ax5.set_xlabel("measured error [m]")
    ax5.set_ylabel("predicted error [m]")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    fig.suptitle("Global Multivariate Error Analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "all_trajectories_multivariate.png", dpi=150)
    plt.close(fig)

    return {
        "samples": float(y.size),
        "r2_v_only": single_r2(v),
        "r2_a_only": single_r2(a),
        "r2_jerk_only": single_r2(j),
        "r2_snap_only": single_r2(s),
        "r2_multi_v_a_j_s": multi_r2,
        "coef_v": float(coef_multi[1]),
        "coef_a": float(coef_multi[2]),
        "coef_jerk": float(coef_multi[3]),
        "coef_snap": float(coef_multi[4]),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "analysis_plots"
    output_dir.mkdir(exist_ok=True)

    pairs = build_pairs(base_dir)
    if not pairs:
        raise FileNotFoundError(f"No valid *_plan.csv + *__px4.csv pairs found in {base_dir}")

    fit_results: List[Tuple[str, Dict[str, float]]] = []
    all_error_list: List[np.ndarray] = []
    all_tracking_speed_list: List[np.ndarray] = []
    all_plan_speed_list: List[np.ndarray] = []
    all_plan_acc_list: List[np.ndarray] = []
    all_plan_jerk_list: List[np.ndarray] = []
    all_plan_snap_list: List[np.ndarray] = []

    for pair in pairs:
        plan = read_csv_with_header(pair.plan_path)
        tracking = read_csv_with_header(pair.tracking_path)

        plan_interp = interpolate_plan_to_tracking(plan, tracking["time"])
        tracking_speed_world = compute_tracking_speed_world(tracking)

        pos_error = np.sqrt(
            (tracking["px"] - plan_interp["px"]) ** 2
            + (tracking["py"] - plan_interp["py"]) ** 2
            + (tracking["pz"] - plan_interp["pz"]) ** 2
        )
        all_error_list.append(pos_error)
        all_tracking_speed_list.append(tracking_speed_world)
        all_plan_speed_list.append(plan_interp["speed"])
        plan_acc = np.sqrt(plan_interp["ax"] ** 2 + plan_interp["ay"] ** 2 + plan_interp["az"] ** 2)
        plan_jerk, plan_snap = compute_jerk_snap_magnitude(
            tracking["time"], plan_interp["ax"], plan_interp["ay"], plan_interp["az"]
        )
        all_plan_acc_list.append(plan_acc)
        all_plan_jerk_list.append(plan_jerk)
        all_plan_snap_list.append(plan_snap)
        save_plot_for_pair(pair.name, plan_interp, tracking, tracking_speed_world, pos_error, output_dir)
        print(f"Saved plot: {output_dir / f'{pair.name}_analysis.png'}")

    all_error = np.concatenate(all_error_list)
    all_tracking_speed_world = np.concatenate(all_tracking_speed_list)
    all_plan_speed = np.concatenate(all_plan_speed_list)
    all_plan_acc = np.concatenate(all_plan_acc_list)
    all_plan_jerk = np.concatenate(all_plan_jerk_list)
    all_plan_snap = np.concatenate(all_plan_snap_list)
    global_metrics = fit_error_speed_models(all_error, all_tracking_speed_world, all_plan_speed)
    fit_results.append(("all_trajectories", global_metrics))
    detailed_metrics = save_global_fit_plot(
        all_error=all_error,
        all_tracking_speed_world=all_tracking_speed_world,
        all_plan_speed=all_plan_speed,
        all_plan_acc=all_plan_acc,
        all_plan_jerk=all_plan_jerk,
        all_plan_snap=all_plan_snap,
        output_dir=output_dir,
    )
    fit_results.append(("all_trajectories_detailed", detailed_metrics))

    report_path = base_dir / "tracking_error_speed_fit_report.txt"
    write_model_report(fit_results, report_path)
    print(f"Saved report: {report_path}")
    metrics_csv_path = base_dir / "tracking_error_speed_fit_metrics.csv"
    save_metrics_csv(fit_results, metrics_csv_path)
    print(f"Saved metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()
