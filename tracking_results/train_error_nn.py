from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import joblib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class TrajectoryPair:
    name: str
    plan_path: Path
    tracking_path: Path


WINDOW_SIZE = 1
EPOCHS = 200

FEATURE_NAMES_PER_FRAME: List[str] = [
    "ep_x",
    "ep_y",
    "ep_z",
    "ev_x",
    "ev_y",
    "ev_z",
    "a_ref_x",
    "a_ref_y",
    "a_ref_z",
    "j_ref_x",
    "j_ref_y",
    "j_ref_z",
    "yaw_ref",
    "yaw_rate_ref",
    "v_world_x",
    "v_world_y",
    "v_world_z",
    "omega_x",
    "omega_y",
    "omega_z",
    "body_z_world_x",
    "body_z_world_y",
    "body_z_world_z",
    "u_thrust",
    "thrust_margin",
]

TARGET_NAMES: List[str] = [
    "ep_x",
    "ep_y",
    "ep_z",
    "ev_x",
    "ev_y",
    "ev_z",
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
        raise FileNotFoundError(f"No valid *_plan.csv + *__px4.csv pairs found in {base_dir}")
    return pairs


def split_pairs(pairs: List[TrajectoryPair]) -> Tuple[List[TrajectoryPair], List[TrajectoryPair]]:
    ordered = sorted(pairs, key=lambda p: p.name)
    # keep trajectory-level split; last 2 trajectories for test
    return ordered[:-2], ordered[-2:]


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
        "yaw": np.interp(t_clip, plan_t, plan["yaw"]),
        "dyaw": np.interp(t_clip, plan_t, plan["dyaw"]),
    }


def rotate_body_to_world(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    qw: np.ndarray,
    vx_b: np.ndarray,
    vy_b: np.ndarray,
    vz_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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
    v_world = np.column_stack((vx_w, vy_w, vz_w))
    body_z_world = np.column_stack((r02, r12, r22))
    return v_world, body_z_world


def compute_jerk(time_s: np.ndarray, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    dt = np.gradient(time_s)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    jx = np.gradient(ax) / dt
    jy = np.gradient(ay) / dt
    jz = np.gradient(az) / dt
    return np.column_stack((jx, jy, jz))


def make_windowed(x: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if window_size <= 1:
        return x, y
    n = x.shape[0]
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in range(window_size - 1, n):
        xs.append(x[i - window_size + 1 : i + 1].reshape(-1))
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


def build_xy_for_pair(pair: TrajectoryPair, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    plan = read_csv_with_header(pair.plan_path)
    tracking = read_csv_with_header(pair.tracking_path)
    t = tracking["time"]
    plan_interp = interpolate_plan_to_tracking(plan, t)
    j_ref = compute_jerk(t, plan_interp["ax"], plan_interp["ay"], plan_interp["az"])
    v_world, body_z_world = rotate_body_to_world(
        tracking["qx"],
        tracking["qy"],
        tracking["qz"],
        tracking["qw"],
        tracking["vx_b"],
        tracking["vy_b"],
        tracking["vz_b"],
    )

    p_track = np.column_stack((tracking["px"], tracking["py"], tracking["pz"]))
    p_ref = np.column_stack((plan_interp["px"], plan_interp["py"], plan_interp["pz"]))
    e_p = p_ref - p_track
    v_ref = np.column_stack((plan_interp["vx"], plan_interp["vy"], plan_interp["vz"]))
    e_v = v_ref - v_world

    u_thrust = tracking["cmd_thrust_norm"] if "cmd_thrust_norm" in tracking else np.zeros_like(t)
    u_thrust = np.where(np.isfinite(u_thrust), u_thrust, 0.0)
    thrust_margin = 1.0 - u_thrust
    omega = np.column_stack((tracking["wx_b"], tracking["wy_b"], tracking["wz_b"]))

    x_frame = np.column_stack(
        (
            e_p,
            e_v,
            plan_interp["ax"],
            plan_interp["ay"],
            plan_interp["az"],
            j_ref,
            plan_interp["yaw"],
            plan_interp["dyaw"],
            v_world,
            omega,
            body_z_world,
            u_thrust,
            thrust_margin,
        )
    )
    y = np.column_stack((e_p, e_v))

    valid = np.all(np.isfinite(x_frame), axis=1) & np.all(np.isfinite(y), axis=1)
    x_frame = x_frame[valid]
    y = y[valid]
    return make_windowed(x_frame, y, window_size)


def concat_pairs(pairs: List[TrajectoryPair], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for pair in pairs:
        x, y = build_xy_for_pair(pair, window_size)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def evaluate_multi(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    ep_true = y_true[:, 0:3]
    ev_true = y_true[:, 3:6]
    ep_pred = y_pred[:, 0:3]
    ev_pred = y_pred[:, 3:6]
    ep_norm_true = np.linalg.norm(ep_true, axis=1)
    ep_norm_pred = np.linalg.norm(ep_pred, axis=1)
    ev_norm_true = np.linalg.norm(ev_true, axis=1)
    ev_norm_pred = np.linalg.norm(ev_pred, axis=1)
    return {
        "r2_all_dims": float(r2_score(y_true, y_pred)),
        "mae_all_dims": float(mean_absolute_error(y_true, y_pred)),
        "rmse_all_dims": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_ep_norm": float(r2_score(ep_norm_true, ep_norm_pred)),
        "mae_ep_norm": float(mean_absolute_error(ep_norm_true, ep_norm_pred)),
        "rmse_ep_norm": float(np.sqrt(mean_squared_error(ep_norm_true, ep_norm_pred))),
        "r2_ev_norm": float(r2_score(ev_norm_true, ev_norm_pred)),
        "mae_ev_norm": float(mean_absolute_error(ev_norm_true, ev_norm_pred)),
        "rmse_ev_norm": float(np.sqrt(mean_squared_error(ev_norm_true, ev_norm_pred))),
    }


def save_metrics_csv(metrics: Dict[str, float], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def save_feature_names_csv(output_path: Path, window_size: int) -> None:
    names: List[str] = []
    for w in range(window_size):
        for fn in FEATURE_NAMES_PER_FRAME:
            names.append(f"t-{window_size-1-w}:{fn}")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "feature_name"])
        for i, name in enumerate(names):
            writer.writerow([i, name])


def save_loss_plot(train_loss: List[float], test_loss: List[float], output_path: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(1, len(train_loss) + 1), train_loss, label="train loss (MSE)")
    ax.plot(np.arange(1, len(test_loss) + 1), test_loss, label="test loss (MSE)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training/Test Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    ep_true = np.linalg.norm(y_true[:, 0:3], axis=1)
    ep_pred = np.linalg.norm(y_pred[:, 0:3], axis=1)
    ev_true = np.linalg.norm(y_true[:, 3:6], axis=1)
    ev_pred = np.linalg.norm(y_pred[:, 3:6], axis=1)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.scatter(ep_true, ep_pred, s=8, alpha=0.25, label="e_p norm")
    mn = float(min(np.min(ep_true), np.min(ep_pred)))
    mx = float(max(np.max(ep_true), np.max(ep_pred)))
    ax1.plot([mn, mx], [mn, mx], "k--", linewidth=1.2)
    ax1.set_title("Position Error Norm: true vs pred")
    ax1.set_xlabel("true")
    ax1.set_ylabel("pred")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.scatter(ev_true, ev_pred, s=8, alpha=0.25, label="e_v norm")
    mn2 = float(min(np.min(ev_true), np.min(ev_pred)))
    mx2 = float(max(np.max(ev_true), np.max(ev_pred)))
    ax2.plot([mn2, mx2], [mn2, mx2], "k--", linewidth=1.2)
    ax2.set_title("Velocity Error Norm: true vs pred")
    ax2.set_xlabel("true")
    ax2.set_ylabel("pred")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_per_trajectory_ep_comparison_plot(
    pairs: List[TrajectoryPair],
    scaler: StandardScaler,
    mlp: MLPRegressor,
    window_size: int,
    output_path: Path,
) -> None:
    n = len(pairs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(14, 4 * nrows))

    for idx, pair in enumerate(pairs, start=1):
        x_pair, y_pair = build_xy_for_pair(pair, window_size)
        x_pair_s = scaler.transform(x_pair)
        y_pred = mlp.predict(x_pair_s)

        ep_true = np.linalg.norm(y_pair[:, 0:3], axis=1)
        ep_pred = np.linalg.norm(y_pred[:, 0:3], axis=1)
        t_idx = np.arange(ep_true.shape[0])

        ax = fig.add_subplot(nrows, ncols, idx)
        ax.plot(t_idx, ep_true, linewidth=1.4, label="true |e_p|")
        ax.plot(t_idx, ep_pred, linewidth=1.2, label="pred |e_p|")
        ax.set_title(pair.name)
        ax.set_xlabel("sample index")
        ax.set_ylabel("|e_p| [m]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Position Control Error per Trajectory: True vs Predicted", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model_doc(
    output_path: Path,
    model_path: Path,
    feature_csv_path: Path,
    metrics_csv_path: Path,
    window_size: int,
) -> None:
    lines: List[str] = []
    lines.append("# Tracking Error NN 模型说明")
    lines.append("")
    lines.append("## 模型文件")
    lines.append(f"- 模型: `{model_path}`")
    lines.append(f"- 输入特征顺序: `{feature_csv_path}`")
    lines.append(f"- 训练指标: `{metrics_csv_path}`")
    lines.append("")
    lines.append("## 输入 X")
    lines.append(f"- 单帧特征维度: {len(FEATURE_NAMES_PER_FRAME)}")
    lines.append(f"- 窗口长度: {window_size} 帧")
    lines.append(f"- 实际输入维度: {len(FEATURE_NAMES_PER_FRAME) * window_size}")
    lines.append("- 单帧特征按以下顺序拼接：")
    for i, name in enumerate(FEATURE_NAMES_PER_FRAME):
        lines.append(f"  - {i}: `{name}`")
    lines.append("")
    lines.append("## 输出 y")
    lines.append("- 6维向量，拟合当前误差：")
    lines.append("  - 位置误差 `e_p = p_ref - p` 的3个分量")
    lines.append("  - 速度误差 `e_v = v_ref - v` 的3个分量")
    lines.append("")
    lines.append("## 训练设置")
    lines.append("- MLP 结构: `(128, 128)`")
    lines.append("- 激活: ReLU, 优化器: Adam")
    lines.append("- 训练轮数: 逐epoch warm-start, 共 `EPOCHS` 轮")
    lines.append("- 训练/测试loss图: `analysis_plots/tracking_error_nn_loss.png`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "analysis_plots"
    output_dir.mkdir(exist_ok=True)

    pairs = build_pairs(base_dir)
    train_pairs, test_pairs = split_pairs(pairs)
    X_train, y_train = concat_pairs(train_pairs, WINDOW_SIZE)
    X_test, y_test = concat_pairs(test_pairs, WINDOW_SIZE)

    rng = np.random.default_rng(42)
    perm = rng.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=42,
    )

    train_loss: List[float] = []
    test_loss: List[float] = []
    for _ in range(EPOCHS):
        mlp.fit(X_train_s, y_train)
        pred_train = mlp.predict(X_train_s)
        pred_test = mlp.predict(X_test_s)
        train_loss.append(float(mean_squared_error(y_train, pred_train)))
        test_loss.append(float(mean_squared_error(y_test, pred_test)))

    pred_train = mlp.predict(X_train_s)
    pred_test = mlp.predict(X_test_s)
    metrics: Dict[str, float] = {
        "window_size": float(WINDOW_SIZE),
        "num_features_per_frame": float(len(FEATURE_NAMES_PER_FRAME)),
        "input_dim_total": float(X_train.shape[1]),
        "output_dim": float(y_train.shape[1]),
        "train_samples": float(X_train.shape[0]),
        "test_samples": float(X_test.shape[0]),
    }
    for prefix, y_t, y_p in (("train", y_train, pred_train), ("test", y_test, pred_test)):
        scores = evaluate_multi(y_t, y_p)
        for k, v in scores.items():
            metrics[f"{prefix}_{k}"] = v

    model_obj = {
        "scaler": scaler,
        "mlp": mlp,
        "window_size": WINDOW_SIZE,
        "feature_names_per_frame": FEATURE_NAMES_PER_FRAME,
        "target_names": TARGET_NAMES,
    }
    model_path = base_dir / "tracking_error_nn_model.joblib"
    joblib.dump(model_obj, model_path)

    metrics_csv = base_dir / "tracking_error_nn_metrics.csv"
    feature_csv = base_dir / "tracking_error_nn_feature_names.csv"
    save_metrics_csv(metrics, metrics_csv)
    save_feature_names_csv(feature_csv, WINDOW_SIZE)
    save_loss_plot(train_loss, test_loss, output_dir / "tracking_error_nn_loss.png")
    save_prediction_plot(y_test, pred_test, output_dir / "tracking_error_nn_test_pred.png")
    save_per_trajectory_ep_comparison_plot(
        pairs=pairs,
        scaler=scaler,
        mlp=mlp,
        window_size=WINDOW_SIZE,
        output_path=output_dir / "tracking_error_nn_ep_all_trajectories.png",
    )
    save_model_doc(base_dir / "tracking_error_nn_model_README.md", model_path, feature_csv, metrics_csv, WINDOW_SIZE)

    print("Train pairs:", [p.name for p in train_pairs])
    print("Test pairs:", [p.name for p in test_pairs])
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved loss plot: {output_dir / 'tracking_error_nn_loss.png'}")
    print(f"Saved trajectory comparison plot: {output_dir / 'tracking_error_nn_ep_all_trajectories.png'}")


if __name__ == "__main__":
    main()

