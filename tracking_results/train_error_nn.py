from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import joblib


@dataclass
class TrajectoryPair:
    name: str
    plan_path: Path
    tracking_path: Path


FEATURE_NAMES: List[str] = [
    "v_track_world_norm",
    "v_plan_norm",
    "v_diff_norm",
    "a_plan_norm",
    "jerk_plan_norm",
    "snap_plan_norm",
    "vx_world_error",
    "vy_world_error",
    "vz_world_error",
]


def read_csv_with_header(path: Path) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def build_pairs(base_dir: Path) -> List[TrajectoryPair]:
    pairs: List[TrajectoryPair] = []
    for plan_path in sorted(base_dir.glob("*_plan.csv")):
        base_name = plan_path.name[: -len("_plan.csv")]
        tracking_path = base_dir / f"{base_name}__px4.csv"
        if tracking_path.exists():
            pairs.append(TrajectoryPair(base_name, plan_path, tracking_path))
    if not pairs:
        raise FileNotFoundError(f"No valid pairs found in {base_dir}")
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


def build_dataset_for_pair(pair: TrajectoryPair) -> Tuple[np.ndarray, np.ndarray]:
    plan = read_csv_with_header(pair.plan_path)
    tracking = read_csv_with_header(pair.tracking_path)
    plan_interp = interpolate_plan_to_tracking(plan, tracking["time"])

    pos_error = np.sqrt(
        (tracking["px"] - plan_interp["px"]) ** 2
        + (tracking["py"] - plan_interp["py"]) ** 2
        + (tracking["pz"] - plan_interp["pz"]) ** 2
    )

    v_world = rotate_body_velocity_to_world(
        tracking["qx"],
        tracking["qy"],
        tracking["qz"],
        tracking["qw"],
        tracking["vx_b"],
        tracking["vy_b"],
        tracking["vz_b"],
    )
    v_track = np.linalg.norm(v_world, axis=1)
    v_plan = np.sqrt(plan_interp["vx"] ** 2 + plan_interp["vy"] ** 2 + plan_interp["vz"] ** 2)
    v_diff = np.abs(v_track - v_plan)
    a_plan = np.sqrt(plan_interp["ax"] ** 2 + plan_interp["ay"] ** 2 + plan_interp["az"] ** 2)
    jerk_plan, snap_plan = compute_jerk_snap_magnitude(
        tracking["time"], plan_interp["ax"], plan_interp["ay"], plan_interp["az"]
    )

    # Baseline recommended features + velocity component mismatch.
    v_err = v_world - np.column_stack((plan_interp["vx"], plan_interp["vy"], plan_interp["vz"]))
    X = np.column_stack(
        (
            v_track,
            v_plan,
            v_diff,
            a_plan,
            jerk_plan,
            snap_plan,
            v_err[:, 0],
            v_err[:, 1],
            v_err[:, 2],
        )
    )
    y = pos_error
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    return X[valid], y[valid]


def split_pairs(pairs: List[TrajectoryPair]) -> Tuple[List[TrajectoryPair], List[TrajectoryPair], List[TrajectoryPair]]:
    # Trajectory-level split to avoid leakage.
    # 7 trajectories -> 5 train / 1 val / 1 test
    names_sorted = sorted(pairs, key=lambda p: p.name)
    train = names_sorted[:5]
    val = names_sorted[5:6]
    test = names_sorted[6:]
    return train, val, test


def concat_dataset(pairs: List[TrajectoryPair]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for pair in pairs:
        x, y = build_dataset_for_pair(pair)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def concat_dataset_with_groups(pairs: List[TrajectoryPair]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    for pair in pairs:
        x, y = build_dataset_for_pair(pair)
        xs.append(x)
        ys.append(y)
        groups.append(np.full(y.shape[0], pair.name, dtype=object))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(groups, axis=0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_metrics_csv(metrics: Dict[str, float], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def save_kfold_metrics_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    columns = ["fold", "val_trajectories", "train_samples", "val_samples", "val_r2", "val_mae", "val_rmse", "val_r2_log_domain"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_kfold_plots(
    fold_plot_data: List[Dict[str, object]],
    metric_rows: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    # Scatter panel per fold
    n = len(fold_plot_data)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    for i, item in enumerate(fold_plot_data, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        y_true = item["y_true"]
        y_pred = item["y_pred"]
        title = item["title"]
        r2 = item["r2"]
        ax.scatter(y_true, y_pred, s=8, alpha=0.25)
        y_min = float(min(np.min(y_true), np.min(y_pred)))
        y_max = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1.2)
        ax.set_title(f"{title}\nR2={r2:.3f}")
        ax.set_xlabel("true error [m]")
        ax.set_ylabel("pred error [m]")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Trajectory GroupKFold: true vs pred per fold", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "tracking_error_nn_kfold_scatter.png", dpi=150)
    plt.close(fig)

    # Metric bars per fold
    fold_rows = [row for row in metric_rows if isinstance(row.get("fold"), int)]
    if not fold_rows:
        return
    folds = [int(row["fold"]) for row in fold_rows]
    r2s = [float(row["val_r2"]) for row in fold_rows]
    maes = [float(row["val_mae"]) for row in fold_rows]
    rmses = [float(row["val_rmse"]) for row in fold_rows]
    x = np.arange(len(folds))

    fig2 = plt.figure(figsize=(12, 5))
    ax1 = fig2.add_subplot(1, 2, 1)
    ax2 = fig2.add_subplot(1, 2, 2)
    ax1.bar(x, r2s)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(f) for f in folds])
    ax1.set_title("KFold Validation R2")
    ax1.set_xlabel("fold")
    ax1.set_ylabel("R2")
    ax1.grid(True, axis="y", alpha=0.3)

    width = 0.38
    ax2.bar(x - width / 2, maes, width=width, label="MAE")
    ax2.bar(x + width / 2, rmses, width=width, label="RMSE")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(f) for f in folds])
    ax2.set_title("KFold Validation Error")
    ax2.set_xlabel("fold")
    ax2.set_ylabel("error [m]")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(output_dir / "tracking_error_nn_kfold_metrics.png", dpi=150)
    plt.close(fig2)


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Left: parity scatter
    ax1.scatter(y_true, y_pred, s=8, alpha=0.3, label="samples (true,pred)")
    y_min = float(min(np.min(y_true), np.min(y_pred)))
    y_max = float(max(np.max(y_true), np.max(y_pred)))
    ax1.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1.2, label="ideal y=x")
    ax1.set_xlabel("true error [m]")
    ax1.set_ylabel("pred error [m]")
    ax1.set_title("Parity Plot")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: direct true-vs-pred values over sample index
    idx = np.arange(y_true.shape[0])
    ax2.plot(idx, y_true, linewidth=1.4, label="true error")
    ax2.plot(idx, y_pred, linewidth=1.2, label="pred error")
    ax2.set_xlabel("test sample index")
    ax2.set_ylabel("error [m]")
    ax2.set_title("Prediction Curve")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("NN Error Prediction (test split)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_names_csv(output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "feature_name"])
        for i, name in enumerate(FEATURE_NAMES):
            writer.writerow([i, name])


def save_model_doc(
    output_path: Path,
    model_path: Path,
    feature_csv_path: Path,
    metrics_csv_path: Path,
    kfold_csv_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Tracking Error NN Model说明")
    lines.append("")
    lines.append("## 模型文件")
    lines.append(f"- 模型路径: `{model_path}`")
    lines.append(f"- 输入特征顺序: `{feature_csv_path}`")
    lines.append(f"- 单次切分评估: `{metrics_csv_path}`")
    lines.append(f"- GroupKFold评估: `{kfold_csv_path}`")
    lines.append("")
    lines.append("## 输入 (X)")
    lines.append("输入是9维浮点向量，顺序必须严格一致：")
    for i, name in enumerate(FEATURE_NAMES):
        lines.append(f"{i}. `{name}`")
    lines.append("")
    lines.append("各特征定义：")
    lines.append("- `v_track_world_norm`: tracking机体系速度经四元数旋转到世界系后的速度模长")
    lines.append("- `v_plan_norm`: planning速度模长")
    lines.append("- `v_diff_norm`: `|v_track_world_norm - v_plan_norm|`")
    lines.append("- `a_plan_norm`: planning加速度模长")
    lines.append("- `jerk_plan_norm`: planning jerk模长（由加速度对时间求导）")
    lines.append("- `snap_plan_norm`: planning snap模长（由jerk对时间求导）")
    lines.append("- `vx_world_error, vy_world_error, vz_world_error`: 世界系速度分量误差")
    lines.append("")
    lines.append("## 输出 (y)")
    lines.append("- 网络输出为 `log1p(position_error)` 域的预测值，脚本中会做 `expm1` 还原。")
    lines.append("- 最终物理输出是 `position_error`，定义为：")
    lines.append("  `sqrt((px_track-px_plan)^2 + (py_track-py_plan)^2 + (pz_track-pz_plan)^2)`，单位米。")
    lines.append("")
    lines.append("## 推理注意事项")
    lines.append("1. 推理时必须使用同样的输入定义和顺序。")
    lines.append("2. 模型文件是包含 `StandardScaler + MLPRegressor` 的 Pipeline，可直接 `model.predict(X)`。")
    lines.append("3. `predict` 后需执行：`y_pred = np.expm1(y_pred_log)`，并建议 `clip(y_pred, 0, +inf)`。")
    lines.append("")
    lines.append("## 训练结构")
    lines.append("- MLP: hidden layers `(128, 128)`, ReLU, Adam, alpha=1e-3")
    lines.append("- 目标变换: `log1p(error)`")
    lines.append("- 验证策略: 按轨迹 GroupKFold 交叉验证")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "analysis_plots"
    output_dir.mkdir(exist_ok=True)

    pairs = build_pairs(base_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    X_train, y_train = concat_dataset(train_pairs)
    X_val, y_val = concat_dataset(val_pairs)
    X_test, y_test = concat_dataset(test_pairs)

    # Shuffle training samples to reduce local time-correlation bias for MLP.
    rng = np.random.default_rng(42)
    perm = rng.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Use log-domain target to reduce heavy-tail effect of tracking error.
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 128),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train_log)

    # Predict in log-domain and invert back to meter-domain.
    train_pred = np.expm1(model.predict(X_train))
    val_pred = np.expm1(model.predict(X_val))
    test_pred = np.expm1(model.predict(X_test))
    train_pred = np.clip(train_pred, 0.0, None)
    val_pred = np.clip(val_pred, 0.0, None)
    test_pred = np.clip(test_pred, 0.0, None)

    metrics: Dict[str, float] = {}
    metrics["num_features"] = float(X_train.shape[1])
    metrics["train_samples"] = float(X_train.shape[0])
    metrics["val_samples"] = float(X_val.shape[0])
    metrics["test_samples"] = float(X_test.shape[0])
    metrics["target_transform_log1p"] = 1.0

    for prefix, y_true, y_pred in (
        ("train", y_train, train_pred),
        ("val", y_val, val_pred),
        ("test", y_test, test_pred),
    ):
        scores = evaluate(y_true, y_pred)
        for key, value in scores.items():
            metrics[f"{prefix}_{key}"] = value
    for prefix, y_true_log, y_pred_meter in (
        ("train", y_train_log, train_pred),
        ("val", y_val_log, val_pred),
        ("test", y_test_log, test_pred),
    ):
        y_pred_log = np.log1p(np.clip(y_pred_meter, 0.0, None))
        metrics[f"{prefix}_r2_log_domain"] = float(r2_score(y_true_log, y_pred_log))

    metrics_csv = base_dir / "tracking_error_nn_metrics.csv"
    save_metrics_csv(metrics, metrics_csv)
    save_prediction_plot(y_test, test_pred, output_dir / "tracking_error_nn_test_pred.png")
    model_path = base_dir / "tracking_error_nn_model.joblib"
    joblib.dump(model, model_path)
    feature_csv_path = base_dir / "tracking_error_nn_feature_names.csv"
    save_feature_names_csv(feature_csv_path)

    print("Train pairs:", [p.name for p in train_pairs])
    print("Val pairs:", [p.name for p in val_pairs])
    print("Test pairs:", [p.name for p in test_pairs])
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved plot: {output_dir / 'tracking_error_nn_test_pred.png'}")

    # Group K-fold CV on all trajectories (trajectory-wise split, no leakage).
    X_all, y_all, groups_all = concat_dataset_with_groups(pairs)
    unique_groups = np.unique(groups_all)
    n_splits = min(5, unique_groups.shape[0])
    gkf = GroupKFold(n_splits=n_splits)
    kfold_rows: List[Dict[str, object]] = []
    kfold_plot_data: List[Dict[str, object]] = []
    kfold_r2: List[float] = []
    kfold_mae: List[float] = []
    kfold_rmse: List[float] = []
    kfold_r2_log: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups_all), start=1):
        X_tr = X_all[train_idx]
        y_tr = y_all[train_idx]
        X_va = X_all[val_idx]
        y_va = y_all[val_idx]
        val_groups = sorted(set(groups_all[val_idx].tolist()))

        rng = np.random.default_rng(42 + fold_idx)
        perm = rng.permutation(X_tr.shape[0])
        X_tr = X_tr[perm]
        y_tr = y_tr[perm]

        y_tr_log = np.log1p(y_tr)
        y_va_log = np.log1p(y_va)

        fold_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 128),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=1000,
                        random_state=42 + fold_idx,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                    ),
                ),
            ]
        )
        fold_model.fit(X_tr, y_tr_log)
        y_va_pred = np.expm1(fold_model.predict(X_va))
        y_va_pred = np.clip(y_va_pred, 0.0, None)

        scores = evaluate(y_va, y_va_pred)
        y_va_pred_log = np.log1p(y_va_pred)
        r2_log = float(r2_score(y_va_log, y_va_pred_log))

        kfold_r2.append(scores["r2"])
        kfold_mae.append(scores["mae"])
        kfold_rmse.append(scores["rmse"])
        kfold_r2_log.append(r2_log)

        kfold_rows.append(
            {
                "fold": fold_idx,
                "val_trajectories": "|".join(val_groups),
                "train_samples": X_tr.shape[0],
                "val_samples": X_va.shape[0],
                "val_r2": scores["r2"],
                "val_mae": scores["mae"],
                "val_rmse": scores["rmse"],
                "val_r2_log_domain": r2_log,
            }
        )
        kfold_plot_data.append(
            {
                "title": "|".join(val_groups),
                "y_true": y_va,
                "y_pred": y_va_pred,
                "r2": scores["r2"],
            }
        )

    kfold_rows.append(
        {
            "fold": "mean",
            "val_trajectories": "all",
            "train_samples": "",
            "val_samples": "",
            "val_r2": float(np.mean(kfold_r2)),
            "val_mae": float(np.mean(kfold_mae)),
            "val_rmse": float(np.mean(kfold_rmse)),
            "val_r2_log_domain": float(np.mean(kfold_r2_log)),
        }
    )
    kfold_rows.append(
        {
            "fold": "std",
            "val_trajectories": "all",
            "train_samples": "",
            "val_samples": "",
            "val_r2": float(np.std(kfold_r2)),
            "val_mae": float(np.std(kfold_mae)),
            "val_rmse": float(np.std(kfold_rmse)),
            "val_r2_log_domain": float(np.std(kfold_r2_log)),
        }
    )

    kfold_csv = base_dir / "tracking_error_nn_kfold_metrics.csv"
    save_kfold_metrics_csv(kfold_rows, kfold_csv)
    save_kfold_plots(kfold_plot_data, kfold_rows, output_dir)
    model_doc_path = base_dir / "tracking_error_nn_model_README.md"
    save_model_doc(
        output_path=model_doc_path,
        model_path=model_path,
        feature_csv_path=feature_csv_path,
        metrics_csv_path=metrics_csv,
        kfold_csv_path=kfold_csv,
    )
    print(f"Saved K-fold metrics: {kfold_csv}")
    print(f"Saved model: {model_path}")
    print(f"Saved feature names: {feature_csv_path}")
    print(f"Saved model doc: {model_doc_path}")


if __name__ == "__main__":
    main()

