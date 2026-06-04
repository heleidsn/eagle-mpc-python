#!/usr/bin/env python3
"""
S500 UAM integrated GUI: trajectory planning (full-state / EE-only) + closed-loop tracking (Crocoddyl along the plan / Acados EE-centric).

右侧绘图标签页：States（位置、姿态、速度、角速度、线加/角加、jerk、snap）、Controls（s500 执行器）、
Base 3D（等比例 XYZ）、Tracking / MPC（轨迹、位置/速度跟踪误差、运动学范数、MPC 或 snap）、Cost analysis。
UAM 模式下 Controls 页提示见 States 的 Acados 布局；EE 的 Tracking 页含速度误差等（见 s500_uam_ee_snap_tracking_mpc）。

Usage:
  python uam_suite_gui.py
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

APP_NAME = "UAM Flight Studio"
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "uam_suite_icon.png"

# Local GUI state/cache files are consolidated under a single hidden directory to
# keep the working tree tidy. Legacy files at the repo root are migrated on first run.
GUI_STATE_DIR = Path(__file__).resolve().parent / ".gui_state"


def _gui_state_path(filename: str) -> Path:
    """Return path under .gui_state/, migrating a legacy root-level file if present."""
    GUI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    new_path = GUI_STATE_DIR / filename
    legacy_path = Path(__file__).resolve().with_name(filename)
    if (not new_path.exists()) and legacy_path.exists():
        try:
            legacy_path.replace(new_path)
        except Exception:
            try:
                new_path.write_bytes(legacy_path.read_bytes())
            except Exception:
                return legacy_path
    return new_path


DEFAULT_PARAMS_PATH = _gui_state_path("uam_suite_gui_params.json")
LAST_SESSION_PATH = _gui_state_path("uam_suite_gui_last_session.json")


def _app_icon():
    """Return the application QIcon if the asset exists, else None."""
    try:
        from PyQt5.QtGui import QIcon

        if APP_ICON_PATH.exists():
            return QIcon(str(APP_ICON_PATH))
    except Exception:
        pass
    return None
SAVED_TRAJECTORIES_DIR = Path(__file__).resolve().parent / "saved_trajectories"
SAVED_TRAJECTORIES_INDEX = SAVED_TRAJECTORIES_DIR / "index.json"
AUTOSAVE_TRAJECTORY_ID = "__autosave__"  # legacy single-slot id (migration only)


def _plan_slot_autosave_id(robot: str, template_id: str) -> str:
    """每个 robot + template 各有一个自动保存槽位。"""
    return f"__autosave___{_safe_name_token(robot)}_{_safe_name_token(template_id)}"
TEMPLATE_DISPLAY_NAMES_PATH = _gui_state_path("template_display_names.json")
USER_TEMPLATES_PATH = _gui_state_path("user_templates.json")
TEMPLATE_CONTROL_POINTS_PATH = _gui_state_path("template_control_points.json")
# 每个 ROS tracking 控制算法各自维护一套 MPC/控制器参数（持久化，跨重启）。
RN_CONTROLLER_PROFILES_PATH = _gui_state_path("ros_tracking_controller_profiles.json")

# Matplotlib 字号倍率（4K 屏自动放大，可用 UAM_PLOT_SCALE 覆盖）
_MPL_FONT_SCALE = 1.0

# 左侧 Planning 面板：按 plan_mode 固定滚动区高度，避免切换 Trajectory template 时跳动。
_PLAN_PATH_SCROLL_H = {0: 400, 1: 340, 2: 420}  # full-state / EE / acc
_PLAN_OPT_SCROLL_H = {0: 380, 1: 0, 2: 180}  # 第 2 组滚动区高度（full-state / acc）

TAB_PLAN = "planning"
TAB_TRACK = "tracking"
TAB_ROS = "ros_tracking"


def _quat_to_euler_row(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return np.array([roll, pitch, yaw], dtype=float)


def _euler_deg_from_simX(simX: np.ndarray) -> np.ndarray:
    simX = np.asarray(simX, dtype=float)
    euler = np.zeros((len(simX), 3), dtype=float)
    for i in range(len(simX)):
        euler[i] = _quat_to_euler_row(simX[i, 3:7])
    return np.degrees(euler)


def _set_mplot3d_equal_xyz(ax, *xyz_arrays: np.ndarray | None, margin: float = 0.06) -> None:
    """Make X/Y/Z axes use the same scale (cube box) for mplot3d."""
    pts: list[np.ndarray] = []
    for arr in xyz_arrays:
        if arr is None:
            continue
        A = np.asarray(arr, dtype=float)
        if A.ndim == 2 and A.shape[1] >= 3 and A.shape[0] > 0:
            pts.append(A[:, :3])
    if not pts:
        return
    P = np.vstack(pts)
    ok = np.isfinite(P).all(axis=1)
    P = P[ok]
    if P.shape[0] == 0:
        return
    lo = np.min(P, axis=0)
    hi = np.max(P, axis=0)
    ctr = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    r = 0.5 * span * (1.0 + margin) if span > 1e-9 else 0.5
    ax.set_xlim(float(ctr[0] - r), float(ctr[0] + r))
    ax.set_ylim(float(ctr[1] - r), float(ctr[1] + r))
    ax.set_zlim(float(ctr[2] - r), float(ctr[2] + r))
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def _set_2d_path_equal_meters(ax, *xy2: np.ndarray, margin: float = 0.06) -> None:
    """XY 或 XZ 路径：横纵轴采用相同米制比例（取包络正方形）。"""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for A in xy2:
        if A is None:
            continue
        M = np.asarray(A, dtype=float)
        if M.size == 0 or M.ndim != 2 or M.shape[1] < 2:
            continue
        xs.append(M[:, 0].ravel())
        ys.append(M[:, 1].ravel())
    if not xs:
        ax.set_aspect("equal", adjustable="box")
        return
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size == 0:
        ax.set_aspect("equal", adjustable="box")
        return
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    half = 0.5 * span * (1.0 + margin)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


def _extract_x17(res: dict) -> np.ndarray:
    x = np.asarray(res["x"], dtype=float)
    n = min(17, x.shape[1])
    return x[:, :n].copy()


def _ensure_uam17_for_s500_base_plot(X: np.ndarray) -> np.ndarray:
    """将状态规范为 17 列 UAM 布局，供 s500 机体图按固定列取 twist（9:15）。"""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return X
    n, m = int(X.shape[0]), int(X.shape[1])
    if m >= 17:
        return X[:, :17].copy()
    if m == 13:
        out = np.zeros((n, 17), dtype=float)
        out[:, :7] = X[:, :7]
        out[:, 9:15] = X[:, 7:13]
        return out
    if m == 7:
        # 仅参考位姿 [xyz+quat]，速度/关节补零（与 plot 中 ref 仅有 p/q 一致）
        out = np.zeros((n, 17), dtype=float)
        out[:, :7] = X[:, :7]
        return out
    # 其它列数：左对齐写入再补零，避免 X[:,9:12] 变成空切片
    out = np.zeros((n, 17), dtype=float)
    out[:, : min(m, 17)] = X[:, : min(m, 17)]
    return out


def _first_order_accel_response_piecewise(
    t_nodes: np.ndarray,
    a_cmd: np.ndarray,
    tau_s: float,
) -> np.ndarray:
    """
    离散网格上实现 τ·da/dt + a = a_cmd，各段内 a_cmd 取左端点常值（与 acc_track 积分约定一致）。
    初值 a(0)=0（冷启动）。τ 极小时退回为指令轨迹。
    """
    t_nodes = np.asarray(t_nodes, dtype=float).reshape(-1)
    a_cmd = np.asarray(a_cmd, dtype=float).reshape(-1, 3)
    n = int(t_nodes.size)
    out = np.zeros_like(a_cmd, dtype=float)
    if n < 2:
        return out
    if tau_s <= 1e-12:
        return a_cmd.copy()
    for i in range(1, n):
        dt = float(t_nodes[i] - t_nodes[i - 1])
        if dt <= 0.0:
            out[i] = out[i - 1]
            continue
        dec = float(np.exp(-dt / tau_s))
        c = a_cmd[i - 1]
        out[i, :] = c + (out[i - 1] - c) * dec
    return out


def _snap_default_rows() -> list[list[float]]:
    return [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.35, 0.95, 14.3, 2.5],
        [0.85, -0.15, 1.05, -22.9, 5.0],
        [1.0, 0.2, 0.9, 8.6, 8.0],
    ]


def _full_wp_default_rows() -> list[list]:
    """[Type, x, y, z, j1/roll°, j2/pitch°, yaw°, t[s]]. Base/EEp: j1,j2,yaw. EE: roll,pitch,yaw."""
    return [
        ["Base", 0.0, 0.0, 1.0, -68.8, -34.4, 0.0, 0.0],
        ["Base", 1.0, 0.5, 1.2, -45.8, -17.2, 45.0, 5.0],
    ]


def _normalize_wp_type_for_combo(cell0: str) -> str:
    """Map saved/free-text type to combo label: Base | EE | EEp."""
    try:
        from s500_uam_trajectory_gui import mixed_wp_row_kind

        k = mixed_wp_row_kind(cell0)
    except Exception:
        s = str(cell0).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        if s.startswith("eep") or s in ("eepos", "eeposition"):
            k = "ee_pos"
        elif s.startswith("e"):
            k = "ee_pose"
        else:
            k = "base"
    if k == "base":
        return "Base"
    if k == "ee_pos":
        return "EEp"
    return "EE"


def _migrate_mixed_wp_rows_v1_to_v2(rows: list) -> list:
    """v1: Base/EEp columns are yaw,j1,j2; v2: j1,j2,yaw. EE rows remain roll,pitch,yaw.

    Trailing optional element (index 8) is the per-row ``zero_v`` flag and is preserved as-is.
    """
    out: list = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 8:
            out.append(list(row) if isinstance(row, (list, tuple)) else row)
            continue
        r = list(row)
        kind = _normalize_wp_type_for_combo(r[0])
        if kind in ("Base", "EEp"):
            yaw, j1, j2 = float(r[4]), float(r[5]), float(r[6])
            r[4], r[5], r[6] = j1, j2, yaw
        out.append(r)
    return out


def _predict_gazebo_spawn_sdf_path(launch_file: str, model_type: str) -> tuple[str | None, str]:
    """
    For eagle_mpc_python SITL launches, resolve the SDF file that spawn_model / PX4 will use.
    Paths follow the <arg name="sdf" / "sdf_file"> rules in those launch files.
    """
    lf = Path(launch_file).name.lower()
    mt = (model_type or "real").strip().lower()
    rel: str | None = None
    if lf == "s500_sitl.launch":
        rel = (
            "models/sdf/s500_uam/s500_ideal.sdf"
            if mt == "ideal"
            else "models/sdf/s500_uam/s500.sdf"
        )
    elif lf == "s500_uam_sitl.launch":
        rel = (
            "models/sdf/s500_uam/s500_uam_ideal.sdf"
            if mt == "ideal"
            else "models/sdf/s500_uam/s500_uam_real.sdf"
        )
    else:
        return None, (
            f"SDF path not auto-resolved for '{lf}' "
            "(supported: s500_sitl.launch, s500_uam_sitl.launch)."
        )
    try:
        proc = subprocess.run(
            ["rospack", "find", "eagle_mpc_python"],
            capture_output=True,
            text=True,
            timeout=6.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return None, f"Could not run rospack ({e!r}). Source ROS workspace, then retry."
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return None, f"rospack find eagle_mpc_python failed: {err or proc.returncode}"
    root = Path(proc.stdout.strip())
    full = (root / rel).resolve()
    hint = "ok" if full.is_file() else "path computed but file not found (check package install)"
    return str(full), hint


def _safe_name_token(text: str) -> str:
    s = str(text or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", "/", "\\", "."):
            out.append("_")
    tok = "".join(out).strip("_")
    return tok or "trajectory"


def _mpl_pt(pt: float) -> float:
    return float(pt) * _MPL_FONT_SCALE


def _detect_plot_font_scale(app) -> float:
    """4K 屏自动放大绘图字号；可用环境变量 UAM_PLOT_SCALE 覆盖。"""
    env = os.environ.get("UAM_PLOT_SCALE", "").strip()
    if env:
        try:
            return max(0.8, min(3.0, float(env)))
        except ValueError:
            pass
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return 1.0
    try:
        dpr = float(screen.devicePixelRatio())
        phys_w = screen.geometry().width() * dpr
        if phys_w >= 3200:
            return 1.5
        if phys_w >= 2560:
            return 1.35
    except Exception:
        pass
    return 1.0


def _load_saved_trajectory_index() -> dict:
    if not SAVED_TRAJECTORIES_INDEX.is_file():
        return {"version": 1, "last_loaded_id": None, "items": {}}
    try:
        data = json.loads(SAVED_TRAJECTORIES_INDEX.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("index root must be object")
        data.setdefault("version", 1)
        data.setdefault("last_loaded_id", None)
        items = data.get("items")
        if isinstance(items, list):
            data["items"] = {str(it.get("id")): it for it in items if isinstance(it, dict) and it.get("id")}
        elif not isinstance(items, dict):
            data["items"] = {}
        return data
    except Exception:
        return {"version": 1, "last_loaded_id": None, "items": {}}


def _load_template_display_names() -> dict[str, str]:
    if not TEMPLATE_DISPLAY_NAMES_PATH.is_file():
        return {}
    try:
        data = json.loads(TEMPLATE_DISPLAY_NAMES_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if v}
    except Exception:
        return {}


def _write_template_display_names(names: dict[str, str]) -> None:
    TEMPLATE_DISPLAY_NAMES_PATH.write_text(
        json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_user_templates() -> dict[str, dict]:
    if not USER_TEMPLATES_PATH.is_file():
        return {}
    try:
        data = json.loads(USER_TEMPLATES_PATH.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else None
        if isinstance(items, dict):
            return {str(k): v for k, v in items.items() if isinstance(v, dict)}
        return {}
    except Exception:
        return {}


def _write_user_templates(items: dict[str, dict]) -> None:
    USER_TEMPLATES_PATH.write_text(
        json.dumps({"version": 1, "items": items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_template_control_points() -> dict[str, dict]:
    if not TEMPLATE_CONTROL_POINTS_PATH.is_file():
        return {}
    try:
        data = json.loads(TEMPLATE_CONTROL_POINTS_PATH.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else None
        if isinstance(items, dict):
            return {str(k): v for k, v in items.items() if isinstance(v, dict)}
        return {}
    except Exception:
        return {}


def _write_template_control_points(items: dict[str, dict]) -> None:
    TEMPLATE_CONTROL_POINTS_PATH.write_text(
        json.dumps({"version": 1, "items": items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_saved_trajectory_index(index: dict) -> None:
    SAVED_TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_TRAJECTORIES_INDEX.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _saved_trajectory_paths(traj_id: str) -> tuple[Path, Path]:
    stem = SAVED_TRAJECTORIES_DIR / _safe_name_token(traj_id)
    return stem.with_suffix(".npz"), stem.with_suffix(".meta.json")


def _plan_bundle_meta_fields(pb: dict) -> dict:
    meta: dict = {"kind": str(pb.get("kind", ""))}
    for key in (
        "velocity_frame",
        "ee_track_kind",
        "plan_mixed_wp_rows",
        "waypoints",
        "t_wp",
    ):
        if key in pb and pb[key] is not None:
            meta[key] = pb[key]
    return meta


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write_plan_bundle_files(traj_id: str, pb: dict, entry_meta: dict) -> None:
    SAVED_TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _saved_trajectory_paths(traj_id)
    arrays: dict[str, np.ndarray] = {}
    for key, val in pb.items():
        if val is None or key in ("plan_mixed_wp_rows", "waypoints", "t_wp"):
            continue
        if isinstance(val, np.ndarray):
            arrays[key] = np.asarray(val)
    np.savez_compressed(npz_path, **arrays)
    meta = _json_safe(dict(entry_meta))
    meta.update(_json_safe(_plan_bundle_meta_fields(pb)))
    meta["id"] = traj_id
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_plan_bundle_files(traj_id: str) -> dict | None:
    npz_path, meta_path = _saved_trajectory_paths(traj_id)
    if not npz_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(npz_path, allow_pickle=False)
        pb: dict = {}
        for key in ("kind", "velocity_frame", "ee_track_kind"):
            if key in meta:
                pb[key] = meta[key]
        for key in ("plan_mixed_wp_rows", "waypoints", "t_wp"):
            if key in meta and meta[key] is not None:
                val = meta[key]
                if key == "waypoints" and isinstance(val, list):
                    pb[key] = np.asarray(val, dtype=float)
                elif key == "t_wp" and isinstance(val, list):
                    pb[key] = np.asarray(val, dtype=float)
                else:
                    pb[key] = val
        for key in data.files:
            pb[key] = np.asarray(data[key])
        if "kind" not in pb and meta.get("kind"):
            pb["kind"] = meta["kind"]
        return pb
    except Exception:
        return None


def build_ee_ref_from_full_state(
    t_plan: np.ndarray,
    x_plan: np.ndarray,
    robot_model,
    ee_frame_id: int,
    T_sim: float,
    sim_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an EE reference along a full-state plan over [0, T_sim]; time starts at 0 (consistent with run_closed_loop)."""
    from s500_uam_crocoddyl_state_tracking_mpc import (
        compute_ee_kinematics_along_trajectory,
        interp_full_state_piecewise,
    )

    dt_ref = min(0.02, float(sim_dt) * 0.5)
    t0 = float(t_plan[0])
    span = max(float(T_sim), float(t_plan[-1]) - t0)
    tau = np.arange(0.0, span + 1e-12, dt_ref)
    t_abs = np.minimum(tau + t0, float(t_plan[-1]))
    X = np.array(
        [interp_full_state_piecewise(float(tt), t_plan, x_plan, robot_model) for tt in t_abs]
    )
    data = robot_model.createData()
    p_ee, _, rpy, _ = compute_ee_kinematics_along_trajectory(
        X, robot_model, data, ee_frame_id
    )
    yaw = np.unwrap(rpy[:, 2].astype(float))
    return tau, p_ee, yaw


class EeRefPlanWorker(QThread):
    finished = pyqtSignal(bool, str, object)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from s500_uam_ee_snap_tracking_mpc import (
                sample_ee_figure_eight_trajectory,
                sample_ee_minimum_snap_trajectory,
            )

            p = self.params
            mode = p.get("mode", "snap")
            def _apply_zero_speed_buffer(
                t: np.ndarray,
                pos: np.ndarray,
                yaw: np.ndarray,
                vel: np.ndarray,
                acc: np.ndarray,
                dyaw: np.ndarray,
                buffer_s: float,
                dt_hint: float,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                t = np.asarray(t, dtype=float).flatten()
                pos = np.asarray(pos, dtype=float)
                yaw = np.asarray(yaw, dtype=float).flatten()
                vel = np.asarray(vel, dtype=float)
                acc = np.asarray(acc, dtype=float)
                dyaw = np.asarray(dyaw, dtype=float).flatten()
                b = max(0.0, float(buffer_s))
                if b <= 1e-9 or t.size == 0:
                    return t, pos, yaw, vel, acc, dyaw
                dt = float(dt_hint) if float(dt_hint) > 1e-9 else (
                    float(np.median(np.diff(t))) if t.size >= 2 else 0.02
                )
                nbuf = max(1, int(np.round(b / max(dt, 1e-9))))
                t_pre = np.arange(0, nbuf, dtype=float) * dt
                t_mid = t + float(nbuf) * dt
                t_post = t_mid[-1] + (np.arange(1, nbuf + 1, dtype=float) * dt)
                p_pre = np.repeat(pos[0:1, :], nbuf, axis=0)
                p_post = np.repeat(pos[-1:, :], nbuf, axis=0)
                y_pre = np.repeat(yaw[0:1], nbuf, axis=0)
                y_post = np.repeat(yaw[-1:], nbuf, axis=0)
                z3_pre = np.zeros((nbuf, 3), dtype=float)
                z3_post = np.zeros((nbuf, 3), dtype=float)
                z1_pre = np.zeros(nbuf, dtype=float)
                z1_post = np.zeros(nbuf, dtype=float)
                t_out = np.concatenate([t_pre, t_mid, t_post], axis=0)
                p_out = np.vstack([p_pre, pos, p_post])
                y_out = np.concatenate([y_pre, yaw, y_post], axis=0)
                v_out = np.vstack([z3_pre, vel, z3_post])
                a_out = np.vstack([z3_pre, acc, z3_post])
                dy_out = np.concatenate([z1_pre, dyaw, z1_post], axis=0)
                return t_out, p_out, y_out, v_out, a_out, dy_out
            def _numdiff_1d(y: np.ndarray, t: np.ndarray) -> np.ndarray:
                y = np.asarray(y, dtype=float).flatten()
                t = np.asarray(t, dtype=float).flatten()
                if y.size <= 1 or t.size <= 1:
                    return np.zeros_like(y, dtype=float)
                dt = np.diff(t)
                dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
                g = np.gradient(y, t)
                return np.asarray(g, dtype=float).flatten()
            if mode == "eight":
                center = np.asarray(p["eight_center"], dtype=float).reshape(3)
                t_grid, p_ref, yaw_ref, dp_ref = sample_ee_figure_eight_trajectory(
                    t_duration=float(p["t_duration"]),
                    dt_sample=float(p["dt_sample"]),
                    center=center,
                    semi_axis=float(p["eight_a"]),
                    period=float(p["eight_period"]),
                )
                if t_grid.size >= 2:
                    ddp_ref = np.gradient(dp_ref, t_grid, axis=0)
                else:
                    ddp_ref = np.zeros_like(dp_ref)
                dyaw_ref = _numdiff_1d(yaw_ref, t_grid)
                payload = {
                    "kind": "ee_ref",
                    "track_kind": "eight",
                    "t_ref": t_grid,
                    "p_ref": p_ref,
                    "yaw_ref": yaw_ref,
                    "dp_ref": dp_ref,
                    "ddp_ref": ddp_ref,
                    "dyaw_ref": dyaw_ref,
                    "waypoints_xyz_yaw": None,
                    "t_wp": None,
                }
            elif mode == "sun_ellipse":
                dt = float(p["dt_sample"])
                vmax = max(1e-6, float(p["vmax"]))
                amax = max(1e-6, float(p["amax"]))
                n_ell = max(1.0, float(p["ellipticity"]))
                loops = max(1, int(p.get("loops", 1)))
                center = np.asarray(p.get("center", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
                plane = str(p.get("plane", "horizontal")).strip().lower()
                yaw_const = float(np.deg2rad(float(p.get("yaw_const_deg", 0.0))))
                yaw_hold = bool(p.get("yaw_hold", False))

                # Sun et al. (2024), Sec. VI-A, Eq. (36)-(38)
                rmax = (vmax * vmax) / amax
                k = amax / vmax
                rmin = rmax / n_ell
                T = 2.0 * np.pi / max(k, 1e-9)
                t_end = float(loops) * T
                t_grid = np.arange(0.0, t_end + 1e-12, dt, dtype=float)
                if t_grid.size < 2:
                    t_grid = np.array([0.0, max(dt, 1e-2)], dtype=float)
                # Quintic time-scaling: motion starts/ends with zero speed.
                tau = np.clip(t_grid / max(t_end, 1e-9), 0.0, 1.0)
                sigma = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
                sigma_dot = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / max(t_end, 1e-9)
                sigma_ddot = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / max(t_end * t_end, 1e-9)
                phi_end = 2.0 * np.pi * float(loops)
                s = phi_end * sigma
                s_dot = phi_end * sigma_dot
                s_ddot = phi_end * sigma_ddot

                p_ref = np.zeros((t_grid.size, 3), dtype=float)
                dp_ref = np.zeros((t_grid.size, 3), dtype=float)
                ddp_ref = np.zeros((t_grid.size, 3), dtype=float)
                yaw_ref = np.zeros(t_grid.size, dtype=float)

                if plane.startswith("v"):
                    p_ref[:, 0] = center[0] + rmax * np.sin(s)
                    p_ref[:, 1] = center[1]
                    p_ref[:, 2] = center[2] + rmin * np.cos(s)
                    dp_ref[:, 0] = rmax * np.cos(s) * s_dot
                    dp_ref[:, 2] = -rmin * np.sin(s) * s_dot
                    ddp_ref[:, 0] = rmax * (-np.sin(s) * s_dot * s_dot + np.cos(s) * s_ddot)
                    ddp_ref[:, 2] = -rmin * (np.cos(s) * s_dot * s_dot + np.sin(s) * s_ddot)
                    yaw_ref[:] = yaw_const
                else:
                    p_ref[:, 0] = center[0] + rmax * np.sin(s)
                    p_ref[:, 1] = center[1] + rmin * np.cos(s)
                    p_ref[:, 2] = center[2]
                    dp_ref[:, 0] = rmax * np.cos(s) * s_dot
                    dp_ref[:, 1] = -rmin * np.sin(s) * s_dot
                    ddp_ref[:, 0] = rmax * (-np.sin(s) * s_dot * s_dot + np.cos(s) * s_ddot)
                    ddp_ref[:, 1] = -rmin * (np.cos(s) * s_dot * s_dot + np.sin(s) * s_ddot)
                    if yaw_hold:
                        yaw_ref[:] = yaw_const
                    else:
                        to_center = center[:2].reshape(1, 2) - p_ref[:, :2]
                        yaw_ref = np.unwrap(np.arctan2(to_center[:, 1], to_center[:, 0]))

                dyaw_ref = _numdiff_1d(yaw_ref, t_grid)
                t_grid, p_ref, yaw_ref, dp_ref, ddp_ref, dyaw_ref = _apply_zero_speed_buffer(
                    t_grid,
                    p_ref,
                    yaw_ref,
                    dp_ref,
                    ddp_ref,
                    dyaw_ref,
                    float(p.get("buffer_s", 1.0)),
                    dt,
                )
                payload = {
                    "kind": "ee_ref",
                    "track_kind": "sun_ellipse",
                    "t_ref": t_grid,
                    "p_ref": p_ref,
                    "yaw_ref": yaw_ref,
                    "dp_ref": dp_ref,
                    "ddp_ref": ddp_ref,
                    "dyaw_ref": dyaw_ref,
                    "waypoints_xyz_yaw": None,
                    "t_wp": None,
                }
            elif mode == "circle":
                dt = float(p["dt_sample"])
                center = np.asarray(p.get("center", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
                radius = max(1e-6, float(p.get("radius", 1.0)))
                period = max(1e-6, float(p.get("period", 6.0)))
                loops = max(1, int(p.get("loops", 3)))
                duration = max(dt, float(p.get("duration", loops * period)))
                yaw_const = float(np.deg2rad(float(p.get("yaw_const_deg", 0.0))))
                yaw_hold = bool(p.get("yaw_hold", False))

                t_grid = np.arange(0.0, duration + 1e-12, dt, dtype=float)
                if t_grid.size < 2:
                    t_grid = np.array([0.0, max(dt, 1e-2)], dtype=float)

                # Quintic time-scaling for zero start/end speed.
                tau = np.clip(t_grid / max(duration, 1e-9), 0.0, 1.0)
                sigma = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
                sigma_dot = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / max(duration, 1e-9)
                sigma_ddot = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / max(duration * duration, 1e-9)
                # Total phase covered during the motion window; quintic scaling enforces zero start/end speed.
                phi_end = 2.0 * np.pi * (duration / period)
                s = phi_end * sigma
                s_dot = phi_end * sigma_dot
                s_ddot = phi_end * sigma_ddot

                p_ref = np.zeros((t_grid.size, 3), dtype=float)
                dp_ref = np.zeros((t_grid.size, 3), dtype=float)
                ddp_ref = np.zeros((t_grid.size, 3), dtype=float)
                yaw_ref = np.zeros(t_grid.size, dtype=float)

                p_ref[:, 0] = center[0] + radius * np.cos(s)
                p_ref[:, 1] = center[1] + radius * np.sin(s)
                p_ref[:, 2] = center[2]
                dp_ref[:, 0] = -radius * np.sin(s) * s_dot
                dp_ref[:, 1] = radius * np.cos(s) * s_dot
                ddp_ref[:, 0] = -radius * (np.cos(s) * s_dot * s_dot + np.sin(s) * s_ddot)
                ddp_ref[:, 1] = radius * (-np.sin(s) * s_dot * s_dot + np.cos(s) * s_ddot)

                if yaw_hold:
                    yaw_ref[:] = yaw_const
                else:
                    # Heading points to circle center.
                    to_center = center[:2].reshape(1, 2) - p_ref[:, :2]
                    yaw_ref = np.unwrap(np.arctan2(to_center[:, 1], to_center[:, 0]))

                dyaw_ref = _numdiff_1d(yaw_ref, t_grid)
                t_grid, p_ref, yaw_ref, dp_ref, ddp_ref, dyaw_ref = _apply_zero_speed_buffer(
                    t_grid,
                    p_ref,
                    yaw_ref,
                    dp_ref,
                    ddp_ref,
                    dyaw_ref,
                    float(p.get("buffer_s", 1.0)),
                    dt,
                )

                payload = {
                    "kind": "ee_ref",
                    "track_kind": "circle",
                    "t_ref": t_grid,
                    "p_ref": p_ref,
                    "yaw_ref": yaw_ref,
                    "dp_ref": dp_ref,
                    "ddp_ref": ddp_ref,
                    "dyaw_ref": dyaw_ref,
                    "waypoints_xyz_yaw": None,
                    "t_wp": None,
                }
            elif mode == "csv_import":
                csv_path = str(p.get("csv_path", "")).strip()
                if not csv_path:
                    raise ValueError("CSV path is empty.")
                csv_file = Path(csv_path).expanduser()
                if not csv_file.is_absolute():
                    csv_file = (Path(__file__).resolve().parent / csv_file).resolve()
                if not csv_file.exists():
                    raise FileNotFoundError(f"CSV file not found: {csv_file}")
                arr = np.genfromtxt(str(csv_file), delimiter=",", names=True, dtype=float, encoding="utf-8")
                if arr.size == 0:
                    raise ValueError(f"CSV has no data rows: {csv_file}")
                required = ("t", "p_x", "p_y", "p_z", "v_x", "v_y", "v_z")
                names = tuple(arr.dtype.names or ())
                missing = [k for k in required if k not in names]
                if missing:
                    raise ValueError(f"CSV missing columns: {missing}; required={required}")
                t_raw = np.atleast_1d(np.asarray(arr["t"], dtype=float).flatten())
                pos_raw = np.column_stack(
                    [
                        np.asarray(arr["p_x"], dtype=float).flatten(),
                        np.asarray(arr["p_y"], dtype=float).flatten(),
                        np.asarray(arr["p_z"], dtype=float).flatten(),
                    ]
                )
                vel_raw = np.column_stack(
                    [
                        np.asarray(arr["v_x"], dtype=float).flatten(),
                        np.asarray(arr["v_y"], dtype=float).flatten(),
                        np.asarray(arr["v_z"], dtype=float).flatten(),
                    ]
                )
                valid = (
                    np.isfinite(t_raw)
                    & np.all(np.isfinite(pos_raw), axis=1)
                    & np.all(np.isfinite(vel_raw), axis=1)
                )
                if int(np.sum(valid)) < 2:
                    raise ValueError("CSV valid samples < 2 after filtering NaN/Inf.")
                t0 = t_raw[valid]
                p0 = pos_raw[valid]
                v0 = vel_raw[valid]
                order = np.argsort(t0)
                t1 = t0[order]
                p1 = p0[order]
                v1 = v0[order]
                keep = np.ones(t1.size, dtype=bool)
                keep[1:] = np.diff(t1) > 1e-12
                t_base = t1[keep]
                p_ref = p1[keep]
                vel_base = v1[keep]
                if t_base.size < 2:
                    raise ValueError("CSV time values are not strictly increasing.")
                z_offset_m = float(p.get("z_offset_m", 0.0))
                p_ref[:, 2] = p_ref[:, 2] + z_offset_m
                t_base = t_base - float(t_base[0])
                vmax_raw = float(np.max(np.linalg.norm(vel_base, axis=1)))
                vmax_limit = max(1e-6, float(p.get("vmax_limit", 5.0)))
                time_scale = max(1.0, vmax_raw / vmax_limit)
                t_ref = t_base * time_scale
                dp_ref = vel_base / time_scale
                ddp_ref = np.gradient(dp_ref, t_ref, axis=0)
                yaw_hold = bool(p.get("yaw_hold", False))
                yaw_const = float(np.deg2rad(float(p.get("yaw_const_deg", 0.0))))
                if yaw_hold:
                    yaw_ref = np.full(t_ref.size, yaw_const, dtype=float)
                    dyaw_ref = np.zeros(t_ref.size, dtype=float)
                else:
                    yaw_ref = np.zeros(t_ref.size, dtype=float)
                    v_xy = np.linalg.norm(dp_ref[:, :2], axis=1)
                    yaw_dyn = np.unwrap(np.arctan2(dp_ref[:, 1], dp_ref[:, 0]))
                    idx0 = int(np.argmax(v_xy > 1e-4)) if np.any(v_xy > 1e-4) else 0
                    yaw_ref[:] = float(yaw_dyn[idx0])
                    for i in range(t_ref.size):
                        if v_xy[i] > 1e-4:
                            yaw_ref[i] = yaw_dyn[i]
                        elif i > 0:
                            yaw_ref[i] = yaw_ref[i - 1]
                    dyaw_ref = _numdiff_1d(yaw_ref, t_ref)
                payload = {
                    "kind": "ee_ref",
                    "track_kind": "csv_import",
                    "t_ref": t_ref,
                    "p_ref": p_ref,
                    "yaw_ref": yaw_ref,
                    "dp_ref": dp_ref,
                    "ddp_ref": ddp_ref,
                    "dyaw_ref": dyaw_ref,
                    "waypoints_xyz_yaw": None,
                    "t_wp": None,
                    "meta": {
                        "csv_path": str(csv_file),
                        "vmax_raw": vmax_raw,
                        "vmax_limit": vmax_limit,
                        "time_scale": time_scale,
                        "z_offset_m": z_offset_m,
                        "yaw_hold": yaw_hold,
                        "yaw_const_deg": float(np.rad2deg(yaw_const)),
                    },
                }
            else:
                rows = p["rows"]
                deg = np.pi / 180.0
                wp = np.zeros((len(rows), 4), dtype=float)
                tw = np.zeros(len(rows), dtype=float)
                for i, r in enumerate(rows):
                    wp[i, 0] = r[0]
                    wp[i, 1] = r[1]
                    wp[i, 2] = r[2]
                    wp[i, 3] = r[3] * deg
                    tw[i] = r[4]
                t_grid, p_ref, yaw_ref, dp_ref = sample_ee_minimum_snap_trajectory(
                    wp, tw, float(p["dt_sample"])
                )
                if t_grid.size >= 2:
                    ddp_ref = np.gradient(dp_ref, t_grid, axis=0)
                else:
                    ddp_ref = np.zeros_like(dp_ref)
                dyaw_ref = _numdiff_1d(yaw_ref, t_grid)
                payload = {
                    "kind": "ee_ref",
                    "track_kind": "snap",
                    "t_ref": t_grid,
                    "p_ref": p_ref,
                    "yaw_ref": yaw_ref,
                    "dp_ref": dp_ref,
                    "ddp_ref": ddp_ref,
                    "dyaw_ref": dyaw_ref,
                    "waypoints_xyz_yaw": wp,
                    "t_wp": tw,
                }
            self.finished.emit(
                True,
                "",
                payload,
            )
        except Exception:
            self.finished.emit(False, traceback.format_exc(), None)


class TrackCrocAlongPlanWorker(QThread):
    finished = pyqtSignal(bool, str, object)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from s500_uam_crocoddyl_state_tracking_mpc import (
                crocoddyl_closed_loop_to_ee_tracking_res,
                default_hover_nominal,
                run_closed_loop_track_full_state_plan,
            )

            p = self.params
            x_nom = p.get("x_nom")
            if x_nom is None:
                x_nom = default_hover_nominal()
            out = run_closed_loop_track_full_state_plan(
                p["x0"],
                p["t_plan"],
                p["x_plan"],
                x_nom,
                p["T_sim"],
                p["sim_dt"],
                p["control_dt"],
                p["dt_mpc"],
                p["horizon"],
                w_state_track=p.get("w_state_track", 10.0),
                w_state_reg=p.get("w_state_reg", 0.1),
                w_control=p.get("w_control", 1e-3),
                w_terminal_track=p.get("w_terminal_track", 100.0),
                w_pos=p.get("w_pos", 1.0),
                w_att=p.get("w_att", 1.0),
                w_joint=p.get("w_joint", 1.0),
                w_vel=p.get("w_vel", 1.0),
                w_omega=p.get("w_omega", 1.0),
                w_joint_vel=p.get("w_joint_vel", 1.0),
                w_u_thrust=p.get("w_u_thrust", 1.0),
                w_u_joint_torque=p.get("w_u_joint_torque", 1.0),
                mpc_max_iter=p.get("mpc_max_iter", 60),
                use_thrust_constraints=p.get("use_thrust_constraints", True),
                use_actuator_first_order=p.get("use_actuator_first_order", False),
                tau_thrust=p.get("tau_thrust", 0.06),
                tau_theta=p.get("tau_theta", 0.05),
                sim_payload_enable=p.get("sim_payload_enable", False),
                sim_payload_t_grasp=p.get("sim_payload_t_grasp", 1.0),
                sim_payload_mass=p.get("sim_payload_mass", 0.2),
                sim_payload_sphere_radius=p.get("sim_payload_sphere_r", 0.02),
                sim_control_stack=p.get("sim_control_stack", "direct"),
                px4_rate_Kp=p.get("px4_rate_Kp", 12.0),
                px4_rate_Kd=p.get("px4_rate_Kd", 1.5),
                s500_yaml_path=p.get("s500_yaml_path"),
                urdf_path=p.get("urdf_path"),
                verbose=False,
            )
            res = crocoddyl_closed_loop_to_ee_tracking_res(out)
            self.finished.emit(True, "", {"out": out, "res": res})
        except Exception:
            self.finished.emit(False, traceback.format_exc(), None)


class TrackAcadosAlongPlanWorker(QThread):
    finished = pyqtSignal(bool, str, object)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from s500_uam_acados_state_tracking_mpc import (
                acados_closed_loop_to_ee_tracking_res,
                run_closed_loop_track_full_state_plan_acados,
            )

            p = self.params
            out = run_closed_loop_track_full_state_plan_acados(
                p["x0"],
                p["t_plan"],
                p["x_plan"],
                p["T_sim"],
                p["sim_dt"],
                p["control_dt"],
                p["dt_mpc"],
                p["N"],
                w_pos=p.get("w_pos", 1.0),
                w_att=p.get("w_att", 1.0),
                w_joint=p.get("w_joint", 1.0),
                w_vel=p.get("w_vel", 1.0),
                w_omega=p.get("w_omega", 1.0),
                w_joint_vel=p.get("w_joint_vel", 1.0),
                w_control=p.get("w_control", 1e-3),
                w_u_thrust=p.get("w_u_thrust", 1.0),
                w_u_joint_torque=p.get("w_u_joint_torque", 1.0),
                w_state_track=p.get("w_state_track", 10.0),
                w_terminal_track=p.get("w_terminal_track", 100.0),
                mpc_max_iter=p.get("mpc_max_iter", 40),
                mpc_log_interval=p.get("mpc_log_interval", 0),
                control_mode=p.get("control_mode", "direct"),
            )
            res = acados_closed_loop_to_ee_tracking_res(out)
            self.finished.emit(True, "", {"out": out, "res": res, "control_mode": p.get("control_mode", "direct")})
        except Exception:
            self.finished.emit(False, traceback.format_exc(), None)


class TrackEeAcadosWorker(QThread):
    finished = pyqtSignal(bool, str, object)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            import s500_uam_ee_snap_tracking_mpc as em

            p = self.params
            out = em.run_ee_tracking_from_reference_arrays(
                p["t_ref"],
                p["p_ref"],
                p["yaw_ref"],
                x0_init=p.get("x0_init"),
                T_sim=p["T_sim"],
                sim_dt=p["sim_dt"],
                control_dt=p["control_dt"],
                dt_mpc=p["dt_mpc"],
                N_mpc=p["N_mpc"],
                w_ee=p["w_ee"],
                w_ee_yaw=p["w_ee_yaw"],
                max_iter=p["mpc_max_iter"],
                mpc_log_interval=p["mpc_log_interval"],
                control_mode_canonical=p["control_mode"],
                show_plan_figure=False,
                log_print=False,
                plan_title=p.get("plan_title", "EE ref"),
                waypoints=p.get("waypoints"),
                t_wp=p.get("t_wp"),
                track_label=p.get("track_label", "suite"),
            )
            self.finished.emit(True, "", out)
        except Exception:
            self.finished.emit(False, traceback.format_exc(), None)


class TrackEeCrocWorker(QThread):
    finished = pyqtSignal(bool, str, object)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from s500_uam_crocoddyl_ee_pose_tracking_mpc import (
                EETrackingWeights,
                run_closed_loop_ee_pose_tracking,
            )

            p = self.params
            weights = EETrackingWeights(
                w_pos=float(
                    p.get("croc_ee_w_pos", p.get("w_ee", 400.0))
                ),
                w_rot_rp=float(p.get("croc_ee_w_rot_rp", 1.0)),
                w_rot_yaw=float(
                    p.get("croc_ee_w_rot_yaw", p.get("w_ee_yaw", 200.0))
                ),
                w_vel_lin=float(p.get("croc_ee_w_vel_lin", 1.0)),
                w_vel_ang_rp=float(p.get("croc_ee_w_vel_ang_rp", 1.0)),
                w_vel_ang_yaw=float(p.get("croc_ee_w_vel_ang_yaw", 1.0)),
                w_u=float(p.get("croc_ee_w_u", 0.0)),
                w_terminal_scale=float(p.get("croc_ee_w_terminal", 3.0)),
                w_state_reg=float(p.get("w_state_reg", 0.0)),
                w_state_track=float(p.get("w_state_track", 0.0)),
            )
            out = run_closed_loop_ee_pose_tracking(
                x0=p["x0"],
                t_ref=p["t_ref"],
                p_ref=p["p_ref"],
                yaw_ref=p["yaw_ref"],
                dt_mpc=p["dt_mpc"],
                horizon=p["N_mpc"],
                sim_dt=p["sim_dt"],
                control_dt=p["control_dt"],
                max_iter=p["mpc_max_iter"],
                use_thrust_constraints=bool(p.get("use_thrust_constraints", True)),
                weights=weights,
                verbose=False,
                use_actuator_first_order=bool(p.get("use_actuator_first_order", False)),
                tau_thrust=float(p.get("tau_thrust", 0.06)),
                tau_theta=float(p.get("tau_theta", 0.05)),
                t_plan=p.get("t_plan"),
                x_plan=p.get("x_plan"),
                sim_payload_enable=bool(p.get("sim_payload_enable", False)),
                sim_payload_t_grasp=float(p.get("sim_payload_t_grasp", 1.0)),
                sim_payload_mass=float(p.get("sim_payload_mass", 0.2)),
                sim_payload_sphere_radius=float(p.get("sim_payload_sphere_r", 0.02)),
            )

            t = np.asarray(out["t"], dtype=float).flatten()
            x = np.asarray(out["states"], dtype=float)
            u = np.asarray(out["u"], dtype=float)
            ee = np.asarray(out["ee"], dtype=float)
            p_ref = np.asarray(out["p_ref"], dtype=float)
            ee_yaw = np.asarray(out["yaw_meas"], dtype=float).flatten()
            yaw_ref = np.asarray(out["yaw_ref"], dtype=float).flatten()
            err = np.linalg.norm(ee - p_ref, axis=1)
            err_yaw = (ee_yaw - yaw_ref + np.pi) % (2.0 * np.pi) - np.pi
            n_inner = max(1, int(round(float(p["control_dt"]) / float(p["sim_dt"]))))
            n_mpc = max(0, len(t) - 1)
            mpc_wall = np.zeros(n_mpc, dtype=float)
            mpc_iter = np.zeros(n_mpc, dtype=int)
            mpc_stat = np.zeros(n_mpc, dtype=int)
            mpc_total_cost = np.full(n_mpc, np.nan, dtype=float)
            steps = np.asarray(out.get("mpc_solve_steps", []), dtype=int).flatten()
            iters = np.asarray(out.get("mpc_iters", []), dtype=int).flatten()
            walls = np.asarray(out.get("mpc_wall_s", []), dtype=float).flatten()
            costs = np.asarray(out.get("mpc_costs", []), dtype=float).flatten()
            n_solves = int(min(len(steps), len(iters), len(walls), len(costs)))
            for j in range(n_solves):
                si = min(max(int(steps[j]), 0), max(0, n_mpc - 1))
                if n_mpc <= 0:
                    break
                mpc_iter[si] = int(iters[j])
                mpc_wall[si] = float(walls[j])
                mpc_total_cost[si] = float(costs[j])
                mpc_stat[si] = 0

            res = {
                "t": t,
                "x": x,
                "u": u,
                "ee": ee,
                "p_ref": p_ref,
                "err": err,
                "ee_yaw": ee_yaw,
                "yaw_ref": yaw_ref,
                "err_yaw": err_yaw,
                "control_mode": "direct",
                "sim_dt": float(p["sim_dt"]),
                "control_dt": float(p["control_dt"]),
                "mpc_stride": n_inner,
                "mpc_solve": {
                    "nlp_iter": mpc_iter,
                    "cpu_s": mpc_wall.copy(),
                    "wall_s": mpc_wall,
                    "status": mpc_stat,
                    "total_cost": mpc_total_cost,
                },
                "mpc_cost_t": np.asarray(out.get("mpc_solve_t", []), dtype=float),
                "mpc_cost_total": costs,
                "mpc_cost_terms": {
                    k: np.asarray(v, dtype=float)
                    for k, v in (out.get("mpc_cost_terms", {}) or {}).items()
                },
                "mpc_cost_groups": {
                    k: np.asarray(v, dtype=float)
                    for k, v in (out.get("mpc_cost_groups", {}) or {}).items()
                },
                "mpc_cost_weights": {
                    k: float(v)
                    for k, v in (out.get("mpc_cost_weights", {}) or {}).items()
                },
            }
            self.finished.emit(True, "", {"out": out, "res": res})
        except Exception:
            self.finished.emit(False, traceback.format_exc(), None)


class MeshcatPlaybackWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        urdf_path: str,
        states: np.ndarray,
        dt: float,
        traj_points: dict[str, np.ndarray] | None = None,
    ):
        super().__init__()
        self.urdf_path = urdf_path
        self.states = np.asarray(states, dtype=float)
        self.dt = float(max(1e-4, dt))
        self.traj_points = traj_points or {}

    def run(self):
        try:
            import pinocchio as pin

            urdf = Path(self.urdf_path).resolve()
            urdf_to_load = urdf
            # If the URDF uses example-robot-data package URIs, remap to local models/s500_uam meshes.
            # This keeps meshcat visualization working without requiring system-wide ROS package setup.
            local_mesh_root = Path(__file__).resolve().parent / "models" / "s500_uam" / "meshes"
            uri_prefix = "package://example-robot-data/robots/s500_description/s500_uam/meshes/"
            if local_mesh_root.exists():
                txt = urdf.read_text(encoding="utf-8")
                if uri_prefix in txt:
                    file_prefix = local_mesh_root.resolve().as_uri() + "/"
                    patched = txt.replace(uri_prefix, file_prefix)
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".urdf", delete=False, encoding="utf-8"
                    ) as tf:
                        tf.write(patched)
                        urdf_to_load = Path(tf.name)
            package_dirs = [str(urdf.parent), str(urdf.parent.parent), str(urdf.parent.parent.parent)]
            warn_msg = ""
            try:
                model, collision_model, visual_model = pin.buildModelsFromUrdf(
                    str(urdf_to_load),
                    package_dirs=package_dirs,
                    root_joint=pin.JointModelFreeFlyer(),
                )
            except TypeError:
                # Backward-compatible fallback for older pinocchio Python bindings.
                model, collision_model, visual_model = pin.buildModelsFromUrdf(
                    str(urdf_to_load), package_dirs, pin.JointModelFreeFlyer()
                )
            except Exception:
                # Fallback: build kinematic model only when mesh resources are unavailable.
                model = pin.buildModelFromUrdf(str(urdf_to_load), pin.JointModelFreeFlyer())
                collision_model = pin.GeometryModel()
                visual_model = pin.GeometryModel()
                warn_msg = (
                    "Mesh resources not found; using model-only playback "
                    "(geometry may not be visible)."
                )
            viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
            viz.initViewer(open=True)
            viz.loadViewerModel("s500_uam")
            # Draw planned/generated trajectories (if meshcat geometry API is available).
            try:
                import meshcat.geometry as g

                def _draw_line(name: str, pts: np.ndarray, color_rgb: tuple[int, int, int]):
                    P = np.asarray(pts, dtype=float)
                    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
                        return
                    pos = P.T
                    color = (int(color_rgb[0]) << 16) | (int(color_rgb[1]) << 8) | int(color_rgb[2])
                    geom = g.Line(g.PointsGeometry(pos), g.LineBasicMaterial(color=color, linewidth=2.0))
                    viz.viewer[f"s500_uam_paths/{name}"].set_object(geom)

                _draw_line("base", self.traj_points.get("base"), (0, 114, 189))  # blue
                _draw_line("ee", self.traj_points.get("ee"), (213, 94, 0))       # orange
                _draw_line("ref", self.traj_points.get("ref"), (0, 158, 115))     # green
            except Exception:
                pass
            nq = int(model.nq)
            if self.states.ndim != 2 or self.states.shape[0] == 0:
                raise ValueError("No states to visualize.")
            n = int(self.states.shape[0])

            def _viewer_closed() -> bool:
                viewer = getattr(viz, "viewer", None)
                if viewer is None:
                    return False
                win = getattr(viewer, "window", None)
                if win is None:
                    return False
                for attr in ("closed", "is_closed"):
                    if hasattr(win, attr):
                        v = getattr(win, attr)
                        try:
                            vv = v() if callable(v) else v
                            if isinstance(vv, bool):
                                return vv
                        except Exception:
                            return True
                return False

            i = 0
            while True:
                if self.isInterruptionRequested():
                    break
                if _viewer_closed():
                    break
                q = np.asarray(self.states[i % n, :nq], dtype=float).flatten()
                try:
                    viz.display(q)
                except Exception:
                    # If browser/server connection is gone, stop playback.
                    break
                time.sleep(self.dt)
                i += 1
            self.finished.emit(True, warn_msg)
        except Exception:
            self.finished.emit(False, traceback.format_exc())


class UamSuiteGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self._plan_bundle: dict | None = None
        self._full_plan_result: dict | None = None
        self._lazy_pin_planner = None
        self._plan_worker = None
        self._track_worker = None
        self._meshcat_worker = None
        self._last_track_res: dict | None = None
        self._manual_ref_overlay: dict | None = None
        self._params_path: Path = DEFAULT_PARAMS_PATH
        self._last_plan_sorted_wp_rows: list | None = None
        self._gazebo_process = None
        self._viz_process = None
        self._task_trajectories = {
            "s500_uam": [
                "full_state_default",
                "wp3_joint_opt",
                "minimum_snap",
                "figure8",
                "acc_track",
            ],
            "s500": [
                "full_state_crocoddyl",
                "minimum_snap",
                "figure8",
                "sun_ellipse",
                "circle",
                "csv_import",
                "acc_track",
            ],
        }
        self._template_display_names: dict[str, str] = _load_template_display_names()
        self._user_templates: dict[str, dict] = _load_user_templates()
        self._template_control_points: dict[str, dict] = _load_template_control_points()

        try:
            from s500_uam_trajectory_gui import (
                ACADOS_AVAILABLE,
                CASCADE_TRAJ_AVAILABLE,
                CROCODDYL_AVAILABLE,
                OptimizationWorker,
                mixed_wp_row_kind,
                wp_to_state,
            )
            from s500_uam_trajectory_planner import make_uam_state

            self._ACADOS_AVAILABLE = ACADOS_AVAILABLE
            self._CASCADE_TRAJ_AVAILABLE = CASCADE_TRAJ_AVAILABLE
            self._CROCODDYL_AVAILABLE = CROCODDYL_AVAILABLE
            self.OptimizationWorker = OptimizationWorker
            self._wp_to_state = wp_to_state
            self._mixed_wp_row_kind = mixed_wp_row_kind
            self._make_uam_state = make_uam_state
        except Exception as e:
            self._ACADOS_AVAILABLE = False
            self._CASCADE_TRAJ_AVAILABLE = False
            self._CROCODDYL_AVAILABLE = False
            self.OptimizationWorker = None
            self._wp_to_state = None
            self._mixed_wp_row_kind = None
            self._make_uam_state = None
            self._import_err = e
        else:
            self._import_err = None

        try:
            import s500_uam_ee_snap_tracking_mpc as em

            self._EE_MPC_OK = bool(
                em.ACADOS_AVAILABLE and em.PINOCCHIO_AVAILABLE and em.DEPS_OK
            )
            self._ee_mpc = em
        except Exception:
            self._EE_MPC_OK = False
            self._ee_mpc = None
        try:
            import s500_uam_crocoddyl_ee_pose_tracking_mpc as _croc_ee

            self._CROC_EE_OK = True
            self._croc_ee_mpc = _croc_ee
        except Exception:
            self._CROC_EE_OK = False
            self._croc_ee_mpc = None

        self.planner = None
        self._build_ui()
        self._load_params_from_path(self._params_path, silent_if_missing=True)
        # Ensure UI sections match current default/loaded task selection on startup.
        self._refresh_task_selection_ui()
        self._refresh_saved_trajectory_combo()
        self._try_restore_last_saved_trajectory()
        self._restore_last_session_selection()
        self.log(
            "Tracking: 4 modes — Croc/Acados full-state plan (需 Full state 规划), "
            "Acados EE-centric, Croc EE pose."
        )
        if not self._EE_MPC_OK:
            self.log(
                "Acados tracking 不可用（缺 acados/pinocchio 或 DEPS）；"
                "「Acados — full-state plan」与「Acados — EE-centric」将灰显。"
            )

    def _init_croc_planner(self):
        if not self._CROCODDYL_AVAILABLE:
            return
        try:
            from s500_uam_trajectory_planner import S500UAMTrajectoryPlanner

            urdf_path = self._selected_robot_urdf_path()
            self.planner = S500UAMTrajectoryPlanner(urdf_path=urdf_path)
            robot_name = self.task_robot_combo.currentText() if hasattr(self, "task_robot_combo") else "s500_uam"
            self.log(
                f"[planner] robot={robot_name} -> urdf={urdf_path}"
            )
        except Exception:
            self.planner = None

    def _selected_robot_urdf_path(self) -> str:
        root = Path(__file__).resolve().parent
        if self._is_s500_mode():
            return str(root / "models" / "urdf" / "s500_simple.urdf")
        return str(root / "models" / "urdf" / "s500_uam_simple.urdf")

    def _robot_model_and_ee(self):
        if self._lazy_pin_planner is None:
            from s500_uam_trajectory_planner import S500UAMTrajectoryPlanner

            self._lazy_pin_planner = S500UAMTrajectoryPlanner(
                urdf_path=self._selected_robot_urdf_path()
            )
        pl = self._lazy_pin_planner
        return pl.robot_model, pl.ee_frame_id

    def _aligned_x0_from_ee_ref(
        self, p_ref: np.ndarray, yaw_ref: np.ndarray, x_seed: np.ndarray | None = None
    ) -> np.ndarray:
        from s500_uam_trajectory_planner import make_uam_state
        from s500_uam_ee_snap_tracking_mpc import align_uam_state_ee_to_world_position

        p_ref = np.asarray(p_ref, dtype=float)
        yaw_ref = np.asarray(yaw_ref, dtype=float).flatten()
        if p_ref.ndim != 2 or p_ref.shape[1] != 3 or len(p_ref) == 0:
            raise ValueError("p_ref must have shape (N,3), N>=1")
        yaw0 = float(yaw_ref[0]) if len(yaw_ref) > 0 else 0.0
        if x_seed is None:
            x0 = np.asarray(make_uam_state(0.0, 0.0, 1.0, j1=0.0, j2=0.0, yaw=yaw0), dtype=float)
        else:
            x0 = np.asarray(x_seed, dtype=float).flatten()[:17].copy()
        rm, _ = self._robot_model_and_ee()
        x0 = align_uam_state_ee_to_world_position(
            x0, rm, np.asarray(p_ref[0], dtype=float).reshape(3), nq=rm.nq, nv=rm.nv
        )
        return x0

    def _build_ui(self):
        self.setWindowTitle(f"{APP_NAME} — Plan · Track · Fly")
        _icon = _app_icon()
        if _icon is not None:
            self.setWindowIcon(_icon)
        # 初始大小不超过标准 1080p；最终会由 _fit_window_to_screen() 按屏幕再夹紧。
        self.resize(1600, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.left_tabs = QTabWidget()
        root.addWidget(self.left_tabs, stretch=0)

        # ----- Plan tab -----
        tab_plan = QWidget()
        plan_layout = QVBoxLayout(tab_plan)
        self.left_tabs.addTab(tab_plan, "Planning")
        plan_splitter = QSplitter(Qt.Vertical)
        plan_layout.addWidget(plan_splitter)
        plan_top = QWidget()
        plan_top_layout = QVBoxLayout(plan_top)
        plan_top_layout.setContentsMargins(0, 0, 0, 0)
        plan_splitter.addWidget(plan_top)

        # Group 1: robot + trajectory template + path editing (waypoints / curves).
        self.task_group = QGroupBox("1. Task settings")
        task_outer = QVBoxLayout()
        tg = QGridLayout()
        tg.addWidget(QLabel("Robot mode"), 0, 0)
        self.task_robot_combo = QComboBox()
        self.task_robot_combo.addItems(["s500_uam", "s500"])
        self.task_robot_combo.setCurrentIndex(1)
        tg.addWidget(self.task_robot_combo, 0, 1)
        tg.addWidget(QLabel("Trajectory template"), 1, 0)
        traj_pick_row = QHBoxLayout()
        self.task_traj_combo = QComboBox()
        traj_pick_row.addWidget(self.task_traj_combo, 1)
        self.new_template_btn = QPushButton("New")
        self.new_template_btn.setToolTip("基于当前所有配置新建一个自定义 Trajectory template（保存当前参数快照）。")
        self.new_template_btn.clicked.connect(self._create_template_from_current)
        traj_pick_row.addWidget(self.new_template_btn)
        self.rename_template_btn = QPushButton("Rename")
        self.rename_template_btn.setToolTip("重命名当前 Trajectory template 在下拉框中的显示名称（不改变模板类型）。")
        self.rename_template_btn.clicked.connect(self._rename_current_trajectory_template)
        traj_pick_row.addWidget(self.rename_template_btn)
        self.delete_template_btn = QPushButton("Delete")
        self.delete_template_btn.setToolTip("删除当前选中的自定义 template（内置 template 不可删除）。")
        self.delete_template_btn.clicked.connect(self._delete_current_user_template)
        traj_pick_row.addWidget(self.delete_template_btn)
        traj_pick_widget = QWidget()
        traj_pick_widget.setLayout(traj_pick_row)
        tg.addWidget(traj_pick_widget, 1, 1)
        self.task_hint_label = QLabel("")
        self.task_hint_label.setWordWrap(True)
        self.task_hint_label.setStyleSheet("color: palette(mid);")
        tg.addWidget(self.task_hint_label, 2, 0, 1, 2)
        self.task_robot_combo.currentTextChanged.connect(self._on_task_robot_changed)
        self.task_traj_combo.currentIndexChanged.connect(
            lambda _idx: self._on_task_traj_changed()
        )
        task_outer.addLayout(tg)
        self.path_stack = QStackedWidget()
        self.path_scroll = self._make_plan_panel_scroll(self.path_stack)
        task_outer.addWidget(self.path_scroll)
        cp_row = QHBoxLayout()
        cp_row.addStretch(1)
        self.save_control_points_btn = QPushButton("Save control points")
        self.save_control_points_btn.setToolTip(
            "保存当前 Trajectory template 的控制点参数（航点/曲线/加速度配置）。\n"
            "下次选择该 robot + template 时自动恢复这些控制点。"
        )
        self.save_control_points_btn.clicked.connect(self._save_template_control_points)
        cp_row.addWidget(self.save_control_points_btn)
        task_outer.addLayout(cp_row)
        self.task_group.setLayout(task_outer)
        self.task_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        plan_top_layout.addWidget(self.task_group)
        # Backward-compatible alias (path editing stack).
        self.plan_stack = self.path_stack

        self.plan_mode_combo = QComboBox()
        self.plan_mode_combo.addItems(
            ["Full state (default)", "Position trajectory", "Acc tracking test"]
        )
        self.plan_mode_combo.setCurrentIndex(0)
        self.plan_mode_combo.currentIndexChanged.connect(self._on_plan_mode)

        # Group 2: trajectory / solver settings (dt, method, optional optimization params).
        self.traj_group = QGroupBox("2. Trajectory setting")
        traj_layout = QVBoxLayout()

        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("Sampling dt [s]"))
        self.dt_plan = QDoubleSpinBox()
        self.dt_plan.setRange(0.001, 0.5)
        self.dt_plan.setValue(0.02)
        dt_row.addWidget(self.dt_plan)
        dt_row.addStretch(1)
        traj_layout.addLayout(dt_row)

        # Optimization method selector (pulled out of the full-state stack so it is
        # always an explicit step right after the trajectory choice).
        self.method_row_widget = QWidget()
        method_row = QHBoxLayout(self.method_row_widget)
        method_row.setContentsMargins(0, 0, 0, 0)
        self.method_combo = QComboBox()
        self._method_ids: list[str] = []
        if self._CROCODDYL_AVAILABLE:
            self.method_combo.addItem("Crocoddyl (BoxDDP)")
            self._method_ids.append("crocoddyl")
            self.method_combo.addItem("Crocoddyl (BoxDDP + actuator 1st-order OCP)")
            self._method_ids.append("crocoddyl_actuator_ocp")
        if self._ACADOS_AVAILABLE:
            self.method_combo.addItem("Acados (thrusters + τ)")
            self._method_ids.append("acados")
            if self._CASCADE_TRAJ_AVAILABLE:
                self.method_combo.addItem("Acados (ω,T,θ + 1st-order)")
                self._method_ids.append("acados_cascade")
            self.method_combo.addItem("Acados (wp3_joint_opt)")
            self._method_ids.append("acados_wp3_joint_opt")
        if not self._method_ids:
            self.method_combo.addItem("(No solver available)")
            self._method_ids.append("none")
        method_row.addWidget(QLabel("Method"))
        method_row.addWidget(self.method_combo, 1)
        traj_layout.addWidget(self.method_row_widget)
        self.method_combo.currentIndexChanged.connect(
            self._refresh_plan_actuator_taus_enabled
        )

        self.opt_stack = QStackedWidget()
        self.opt_scroll = self._make_plan_panel_scroll(self.opt_stack)
        traj_layout.addWidget(self.opt_scroll)
        self.traj_group.setLayout(traj_layout)
        plan_top_layout.addWidget(self.traj_group)

        # Path stack 0: full-state waypoints
        w_full_path = QWidget()
        g_full = QVBoxLayout(w_full_path)

        self._wp_type_help_label = QLabel(
            "Columns j1/roll, j2/pitch, yaw: for Base and EEp they mean j1 deg, j2 deg, base yaw deg; "
            "for EE they mean roll deg, pitch deg, yaw deg (ZYX). "
            "EEp constrains only end-effector position; the three angles are alignment seeds. "
            "Zero v: 勾选则在该航点上约束速度为 0 (Base：完整状态速度；EE 模式：EE 笛卡尔速度)，"
            "取消勾选则不约束该航点速度 (仅 full_state_crocoddyl 生效)."
        )
        self._wp_type_help_label.setWordWrap(True)
        # g_full.addWidget(self._wp_type_help_label)
        self.wp_table = QTableWidget(2, 9)
        self.wp_table.setHorizontalHeaderLabels(
            ["Type", "x", "y", "z", "j1/roll°", "j2/pitch°", "yaw°", "t [s]", "Zero v"]
        )
        wp_header = self.wp_table.horizontalHeader()
        wp_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in range(1, 9):
            wp_header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        wp_header.setStretchLastSection(False)
        wp_header.setMinimumSectionSize(56)
        for r, row in enumerate(_full_wp_default_rows()):
            self.wp_table.setCellWidget(r, 0, self._make_wp_type_combo(str(row[0])))
            for c, val in enumerate(row[1:], start=1):
                self.wp_table.setItem(r, c, QTableWidgetItem(f"{float(val):g}"))
            self.wp_table.setCellWidget(r, 8, self._make_wp_zero_v_widget(True))
        self._refresh_wp_rows_angle_enabled()
        g_full.addWidget(self.wp_table)
        wp_btn = QHBoxLayout()
        add_r = QPushButton("Add row")
        add_r.clicked.connect(self._add_wp_row)
        del_r = QPushButton("Delete last row")
        del_r.clicked.connect(self._del_wp_row)
        wp_btn.addWidget(add_r)
        wp_btn.addWidget(del_r)
        g_full.addLayout(wp_btn)
        self.path_stack.addWidget(w_full_path)

        # Opt stack 0: full-state solver / cost parameters
        w_full_opt = QWidget()
        g_full_opt = QVBoxLayout(w_full_opt)
        cost_g = QGridLayout()
        self.max_iter_plan = QSpinBox()
        self.max_iter_plan.setRange(10, 2000)
        self.max_iter_plan.setValue(200)
        self.state_w = QDoubleSpinBox()
        self.state_w.setRange(1e-4, 1e4)
        self.state_w.setValue(1.0)
        self.ctrl_w = QDoubleSpinBox()
        self.ctrl_w.setRange(1e-3, 100.0)
        self.ctrl_w.setValue(1e-5)
        self.wp_mult = QDoubleSpinBox()
        self.wp_mult.setRange(1, 1e6)
        self.wp_mult.setValue(1000.0)
        self.plan_croc_use_actuator_first_order = QCheckBox("Enable")
        self.plan_croc_use_actuator_first_order.setChecked(False)
        self.plan_croc_use_actuator_first_order.toggled.connect(
            self._refresh_plan_actuator_taus_enabled
        )
        self.plan_tau_motor = QDoubleSpinBox()
        self.plan_tau_motor.setRange(0.001, 2.0)
        self.plan_tau_motor.setDecimals(3)
        self.plan_tau_motor.setSingleStep(0.005)
        self.plan_tau_motor.setValue(0.06)
        self.plan_tau_joint = QDoubleSpinBox()
        self.plan_tau_joint.setRange(0.001, 2.0)
        self.plan_tau_joint.setDecimals(3)
        self.plan_tau_joint.setSingleStep(0.005)
        self.plan_tau_joint.setValue(0.05)
        cost_g.addWidget(QLabel("max_iter"), 0, 0)
        cost_g.addWidget(self.max_iter_plan, 0, 1)
        cost_g.addWidget(QLabel("state_w"), 0, 2)
        cost_g.addWidget(self.state_w, 0, 3)
        cost_g.addWidget(QLabel("ctrl_w"), 1, 0)
        cost_g.addWidget(self.ctrl_w, 1, 1)
        cost_g.addWidget(QLabel("wp_mult"), 1, 2)
        cost_g.addWidget(self.wp_mult, 1, 3)
        cost_g.addWidget(QLabel("Croc actuator 1st-order"), 2, 0)
        cost_g.addWidget(self.plan_croc_use_actuator_first_order, 2, 1)
        cost_g.addWidget(QLabel("tau motor thrust [s]"), 2, 2)
        cost_g.addWidget(self.plan_tau_motor, 2, 3)
        cost_g.addWidget(QLabel("tau joint torque [s]"), 3, 0)
        cost_g.addWidget(self.plan_tau_joint, 3, 1)
        self.ee_knot_w = QDoubleSpinBox()
        self.ee_knot_w.setRange(1.0, 1e6)
        self.ee_knot_w.setDecimals(1)
        self.ee_knot_w.setValue(5000.0)
        self.ee_knot_state_reg_w = QDoubleSpinBox()
        self.ee_knot_state_reg_w.setRange(0.0, 1e4)
        self.ee_knot_state_reg_w.setDecimals(4)
        self.ee_knot_state_reg_w.setValue(0.0)
        cost_g.addWidget(QLabel("EE knot w"), 4, 0)
        cost_g.addWidget(self.ee_knot_w, 4, 1)
        cost_g.addWidget(QLabel("EE knot state_reg w (0=off)"), 4, 2)
        cost_g.addWidget(self.ee_knot_state_reg_w, 4, 3)
        self.ee_knot_rot_w = QDoubleSpinBox()
        self.ee_knot_rot_w.setRange(0.0, 1e6)
        self.ee_knot_rot_w.setDecimals(1)
        self.ee_knot_rot_w.setValue(1000.0)
        cost_g.addWidget(QLabel("EE knot rot w (0=position only)"), 5, 0)
        cost_g.addWidget(self.ee_knot_rot_w, 5, 1)
        self.ee_knot_vel_w = QDoubleSpinBox()
        self.ee_knot_vel_w.setRange(0.0, 1e6)
        self.ee_knot_vel_w.setDecimals(1)
        self.ee_knot_vel_w.setValue(200.0)
        cost_g.addWidget(QLabel("EE knot vel w (ref=0)"), 5, 2)
        cost_g.addWidget(self.ee_knot_vel_w, 5, 3)
        self.ee_knot_vel_pitch_w = QDoubleSpinBox()
        self.ee_knot_vel_pitch_w.setRange(0.0, 1e6)
        self.ee_knot_vel_pitch_w.setDecimals(1)
        self.ee_knot_vel_pitch_w.setValue(0.0)
        cost_g.addWidget(QLabel("EE vel pitch ωy w (0=off)"), 6, 0)
        cost_g.addWidget(self.ee_knot_vel_pitch_w, 6, 1)
        wg = QGroupBox("Full-state optimization parameters")
        wg.setLayout(cost_g)
        g_full_opt.addWidget(wg)

        wp3g = QGridLayout()
        self.wp3_mode_combo = QComboBox()
        self.wp3_mode_combo.addItems(["baseline", "ctrl_error"])
        self.wp3_total_time = QDoubleSpinBox(); self.wp3_total_time.setRange(0.5, 30.0); self.wp3_total_time.setValue(3.0)
        self.wp3_grasp_time = QDoubleSpinBox(); self.wp3_grasp_time.setRange(0.1, 30.0); self.wp3_grasp_time.setValue(1.5)
        self.wp3_gx = QDoubleSpinBox(); self.wp3_gy = QDoubleSpinBox(); self.wp3_gz = QDoubleSpinBox()
        for w, v in ((self.wp3_gx, 0.0), (self.wp3_gy, 0.0), (self.wp3_gz, 1.0)):
            w.setRange(-20, 20); w.setDecimals(3); w.setValue(v)
        self.wp3_gr = QDoubleSpinBox(); self.wp3_gp = QDoubleSpinBox(); self.wp3_gyaw = QDoubleSpinBox()
        for w in (self.wp3_gr, self.wp3_gp, self.wp3_gyaw):
            w.setRange(-180, 180); w.setDecimals(2); w.setValue(0.0)
        self.wp3_kx = QDoubleSpinBox(); self.wp3_ky = QDoubleSpinBox(); self.wp3_kz = QDoubleSpinBox()
        for w in (self.wp3_kx, self.wp3_ky, self.wp3_kz):
            w.setRange(0.0, 10.0); w.setDecimals(3); w.setValue(0.08)
        self.wp3_ex = QDoubleSpinBox(); self.wp3_ey = QDoubleSpinBox(); self.wp3_ez = QDoubleSpinBox()
        for w in (self.wp3_ex, self.wp3_ey, self.wp3_ez):
            w.setRange(0.0, 10.0); w.setDecimals(3); w.setValue(0.06)
        self.wp3_w0x = QDoubleSpinBox(); self.wp3_w0y = QDoubleSpinBox(); self.wp3_w0z = QDoubleSpinBox()
        self.wp3_w0j1 = QDoubleSpinBox(); self.wp3_w0j2 = QDoubleSpinBox(); self.wp3_w0yaw = QDoubleSpinBox()
        self.wp3_w2x = QDoubleSpinBox(); self.wp3_w2y = QDoubleSpinBox(); self.wp3_w2z = QDoubleSpinBox()
        self.wp3_w2j1 = QDoubleSpinBox(); self.wp3_w2j2 = QDoubleSpinBox(); self.wp3_w2yaw = QDoubleSpinBox()
        for w, v in (
            (self.wp3_w0x, -1.5), (self.wp3_w0y, 0.0), (self.wp3_w0z, 1.5), (self.wp3_w0j1, 0.0), (self.wp3_w0j2, 0.0), (self.wp3_w0yaw, 0.0),
            (self.wp3_w2x, 1.5), (self.wp3_w2y, 0.0), (self.wp3_w2z, 1.5), (self.wp3_w2j1, 0.0), (self.wp3_w2j2, 0.0), (self.wp3_w2yaw, 0.0),
        ):
            w.setRange(-50, 50); w.setDecimals(3); w.setValue(v)
        wp3g.addWidget(QLabel("mode"), 0, 0); wp3g.addWidget(self.wp3_mode_combo, 0, 1)
        wp3g.addWidget(QLabel("total_time"), 0, 2); wp3g.addWidget(self.wp3_total_time, 0, 3)
        wp3g.addWidget(QLabel("grasp_time"), 0, 4); wp3g.addWidget(self.wp3_grasp_time, 0, 5)
        wp3g.addWidget(QLabel("grasp pos x/y/z"), 1, 0); wp3g.addWidget(self.wp3_gx, 1, 1); wp3g.addWidget(self.wp3_gy, 1, 2); wp3g.addWidget(self.wp3_gz, 1, 3)
        wp3g.addWidget(QLabel("grasp r/p/yaw (deg)"), 1, 4); wp3g.addWidget(self.wp3_gr, 1, 5); wp3g.addWidget(self.wp3_gp, 1, 6); wp3g.addWidget(self.wp3_gyaw, 1, 7)
        wp3g.addWidget(QLabel("pos_err_gain kx/ky/kz"), 2, 0); wp3g.addWidget(self.wp3_kx, 2, 1); wp3g.addWidget(self.wp3_ky, 2, 2); wp3g.addWidget(self.wp3_kz, 2, 3)
        wp3g.addWidget(QLabel("grasp_pos_err_max"), 2, 4); wp3g.addWidget(self.wp3_ex, 2, 5); wp3g.addWidget(self.wp3_ey, 2, 6); wp3g.addWidget(self.wp3_ez, 2, 7)
        wp3g.addWidget(QLabel("wp0 x y z j1 j2 yaw"), 3, 0); wp3g.addWidget(self.wp3_w0x, 3, 1); wp3g.addWidget(self.wp3_w0y, 3, 2); wp3g.addWidget(self.wp3_w0z, 3, 3); wp3g.addWidget(self.wp3_w0j1, 3, 4); wp3g.addWidget(self.wp3_w0j2, 3, 5); wp3g.addWidget(self.wp3_w0yaw, 3, 6)
        wp3g.addWidget(QLabel("wp2 x y z j1 j2 yaw"), 4, 0); wp3g.addWidget(self.wp3_w2x, 4, 1); wp3g.addWidget(self.wp3_w2y, 4, 2); wp3g.addWidget(self.wp3_w2z, 4, 3); wp3g.addWidget(self.wp3_w2j1, 4, 4); wp3g.addWidget(self.wp3_w2j2, 4, 5); wp3g.addWidget(self.wp3_w2yaw, 4, 6)
        self.wp3_group = QGroupBox("wp3_joint_opt settings")
        self.wp3_group.setLayout(wp3g)
        g_full_opt.addWidget(self.wp3_group)
        self.opt_stack.addWidget(w_full_opt)
        self._refresh_plan_actuator_taus_enabled()

        # Path stack 1: EE / position trajectory editing
        w_ee = QWidget()
        g_ee = QVBoxLayout(w_ee)
        self.ee_type_row_widget = QWidget()
        ee_type_row = QHBoxLayout(self.ee_type_row_widget)
        ee_type_row.setContentsMargins(0, 0, 0, 0)
        self.ee_plan_type_combo = QComboBox()
        self.ee_plan_type_combo.addItems(
            ["Minimum snap (waypoints)", "Figure-eight (figure-8)", "Sun2024 ellipse", "Circle", "Import CSV"]
        )
        self.ee_type_label = QLabel("EE trajectory type")
        ee_type_row.addWidget(self.ee_type_label)
        ee_type_row.addWidget(self.ee_plan_type_combo)
        g_ee.addWidget(self.ee_type_row_widget)
        self.ee_wp_label = QLabel("EE waypoints (x,y,z m, yaw°, time s) — consistent with the EE tracking GUI")
        g_ee.addWidget(self.ee_wp_label)
        self.ee_wp_table = QTableWidget(4, 5)
        self.ee_wp_table.setHorizontalHeaderLabels(["x", "y", "z", "yaw°", "t [s]"])
        ee_header = self.ee_wp_table.horizontalHeader()
        ee_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        ee_header.setStretchLastSection(False)
        ee_header.setMinimumSectionSize(56)
        for r, row in enumerate(_snap_default_rows()):
            for c, val in enumerate(row):
                self.ee_wp_table.setItem(r, c, QTableWidgetItem(f"{val:g}"))
        g_ee.addWidget(self.ee_wp_table)
        self.ee_eight_group = QGroupBox("Figure-eight parameters")
        ee8 = QGridLayout()
        self.ee_eight_cx = QDoubleSpinBox()
        self.ee_eight_cy = QDoubleSpinBox()
        self.ee_eight_cz = QDoubleSpinBox()
        for w, v in zip((self.ee_eight_cx, self.ee_eight_cy, self.ee_eight_cz), (0.55, 0.05, 0.92)):
            w.setRange(-20, 20)
            w.setDecimals(3)
            w.setSingleStep(0.05)
            w.setValue(v)
        self.ee_eight_a = QDoubleSpinBox()
        self.ee_eight_a.setRange(0.05, 2.0)
        self.ee_eight_a.setDecimals(3)
        self.ee_eight_a.setValue(0.22)
        self.ee_eight_period = QDoubleSpinBox()
        self.ee_eight_period.setRange(0.5, 120.0)
        self.ee_eight_period.setDecimals(2)
        self.ee_eight_period.setValue(6.0)
        self.ee_eight_tdur = QDoubleSpinBox()
        self.ee_eight_tdur.setRange(0.5, 240.0)
        self.ee_eight_tdur.setDecimals(2)
        self.ee_eight_tdur.setValue(8.0)
        ee8.addWidget(QLabel("Center cx"), 0, 0)
        ee8.addWidget(self.ee_eight_cx, 0, 1)
        ee8.addWidget(QLabel("cy"), 0, 2)
        ee8.addWidget(self.ee_eight_cy, 0, 3)
        ee8.addWidget(QLabel("cz"), 1, 0)
        ee8.addWidget(self.ee_eight_cz, 1, 1)
        ee8.addWidget(QLabel("Half-width a [m]"), 1, 2)
        ee8.addWidget(self.ee_eight_a, 1, 3)
        ee8.addWidget(QLabel("Period [s]"), 2, 0)
        ee8.addWidget(self.ee_eight_period, 2, 1)
        ee8.addWidget(QLabel("Duration [s]"), 2, 2)
        ee8.addWidget(self.ee_eight_tdur, 2, 3)
        self.ee_eight_group.setLayout(ee8)
        g_ee.addWidget(self.ee_eight_group)
        self.ee_sun_group = QGroupBox("Sun2024 ellipse parameters")
        sun = QGridLayout()
        self.ee_sun_plane_combo = QComboBox()
        self.ee_sun_plane_combo.addItems(["horizontal", "vertical"])
        self.ee_sun_vmax = QDoubleSpinBox()
        self.ee_sun_vmax.setRange(0.1, 50.0)
        self.ee_sun_vmax.setDecimals(2)
        self.ee_sun_vmax.setValue(10.0)
        self.ee_sun_amax = QDoubleSpinBox()
        self.ee_sun_amax.setRange(0.1, 100.0)
        self.ee_sun_amax.setDecimals(2)
        self.ee_sun_amax.setValue(20.0)
        self.ee_sun_n = QDoubleSpinBox()
        self.ee_sun_n.setRange(1.0, 20.0)
        self.ee_sun_n.setDecimals(2)
        self.ee_sun_n.setValue(2.0)
        self.ee_sun_loops = QSpinBox()
        self.ee_sun_loops.setRange(1, 20)
        self.ee_sun_loops.setValue(2)
        self.ee_sun_cx = QDoubleSpinBox()
        self.ee_sun_cy = QDoubleSpinBox()
        self.ee_sun_cz = QDoubleSpinBox()
        for w, v in zip((self.ee_sun_cx, self.ee_sun_cy, self.ee_sun_cz), (0.0, 0.0, 1.0)):
            w.setRange(-20, 20)
            w.setDecimals(3)
            w.setSingleStep(0.05)
            w.setValue(v)
        self.ee_sun_yaw_const = QDoubleSpinBox()
        self.ee_sun_yaw_const.setRange(-180.0, 180.0)
        self.ee_sun_yaw_const.setDecimals(2)
        self.ee_sun_yaw_const.setValue(0.0)
        self.ee_sun_yaw_hold = QCheckBox("Keep yaw constant")
        self.ee_sun_yaw_hold.setChecked(False)
        self.ee_sun_buffer = QDoubleSpinBox()
        self.ee_sun_buffer.setRange(0.0, 20.0)
        self.ee_sun_buffer.setDecimals(2)
        self.ee_sun_buffer.setValue(1.0)
        sun.addWidget(QLabel("Plane"), 0, 0)
        sun.addWidget(self.ee_sun_plane_combo, 0, 1)
        sun.addWidget(QLabel("Vmax [m/s]"), 0, 2)
        sun.addWidget(self.ee_sun_vmax, 0, 3)
        sun.addWidget(QLabel("amax [m/s²]"), 1, 0)
        sun.addWidget(self.ee_sun_amax, 1, 1)
        sun.addWidget(QLabel("ellipticity n"), 1, 2)
        sun.addWidget(self.ee_sun_n, 1, 3)
        sun.addWidget(QLabel("loops"), 2, 0)
        sun.addWidget(self.ee_sun_loops, 2, 1)
        sun.addWidget(QLabel("center cx"), 2, 2)
        sun.addWidget(self.ee_sun_cx, 2, 3)
        sun.addWidget(QLabel("cy"), 3, 0)
        sun.addWidget(self.ee_sun_cy, 3, 1)
        sun.addWidget(QLabel("cz"), 3, 2)
        sun.addWidget(self.ee_sun_cz, 3, 3)
        sun.addWidget(QLabel("yaw const [deg] (vertical)"), 4, 0)
        sun.addWidget(self.ee_sun_yaw_const, 4, 1)
        sun.addWidget(self.ee_sun_yaw_hold, 5, 0, 1, 2)
        sun.addWidget(QLabel("buffer each end [s]"), 4, 2)
        sun.addWidget(self.ee_sun_buffer, 4, 3)
        self.ee_sun_group.setLayout(sun)
        g_ee.addWidget(self.ee_sun_group)
        self.ee_circle_group = QGroupBox("Circle parameters")
        cir = QGridLayout()
        self.ee_circle_cx = QDoubleSpinBox()
        self.ee_circle_cy = QDoubleSpinBox()
        self.ee_circle_cz = QDoubleSpinBox()
        for w, v in zip((self.ee_circle_cx, self.ee_circle_cy, self.ee_circle_cz), (0.0, 0.0, 1.0)):
            w.setRange(-20, 20); w.setDecimals(3); w.setSingleStep(0.05); w.setValue(v)
        self.ee_circle_r = QDoubleSpinBox()
        self.ee_circle_r.setRange(0.05, 20.0); self.ee_circle_r.setDecimals(3); self.ee_circle_r.setValue(1.0)
        self.ee_circle_period = QDoubleSpinBox()
        self.ee_circle_period.setRange(0.2, 120.0); self.ee_circle_period.setDecimals(2); self.ee_circle_period.setValue(6.0)
        self.ee_circle_loops = QSpinBox()
        self.ee_circle_loops.setRange(1, 200)
        self.ee_circle_loops.setValue(3)
        self.ee_circle_tdur = QDoubleSpinBox()
        self.ee_circle_tdur.setRange(0.2, 240.0); self.ee_circle_tdur.setDecimals(2); self.ee_circle_tdur.setValue(18.0)
        self.ee_circle_tdur.setReadOnly(True)
        self.ee_circle_tdur.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.ee_circle_yaw_const = QDoubleSpinBox()
        self.ee_circle_yaw_const.setRange(-180.0, 180.0); self.ee_circle_yaw_const.setDecimals(2); self.ee_circle_yaw_const.setValue(0.0)
        self.ee_circle_yaw_hold = QCheckBox("Keep yaw constant"); self.ee_circle_yaw_hold.setChecked(False)
        self.ee_circle_buffer = QDoubleSpinBox()
        self.ee_circle_buffer.setRange(0.0, 20.0); self.ee_circle_buffer.setDecimals(2); self.ee_circle_buffer.setValue(1.0)
        cir.addWidget(QLabel("center cx"), 0, 0); cir.addWidget(self.ee_circle_cx, 0, 1)
        cir.addWidget(QLabel("cy"), 0, 2); cir.addWidget(self.ee_circle_cy, 0, 3)
        cir.addWidget(QLabel("cz"), 1, 0); cir.addWidget(self.ee_circle_cz, 1, 1)
        cir.addWidget(QLabel("radius [m]"), 1, 2); cir.addWidget(self.ee_circle_r, 1, 3)
        cir.addWidget(QLabel("period [s]"), 2, 0); cir.addWidget(self.ee_circle_period, 2, 1)
        cir.addWidget(QLabel("loops"), 2, 2); cir.addWidget(self.ee_circle_loops, 2, 3)
        cir.addWidget(QLabel("duration [s]"), 3, 0); cir.addWidget(self.ee_circle_tdur, 3, 1)
        cir.addWidget(QLabel("yaw const [deg]"), 3, 2); cir.addWidget(self.ee_circle_yaw_const, 3, 3)
        cir.addWidget(self.ee_circle_yaw_hold, 4, 0, 1, 2)
        cir.addWidget(QLabel("buffer each end [s]"), 4, 2); cir.addWidget(self.ee_circle_buffer, 4, 3)
        self.ee_circle_group.setLayout(cir)
        g_ee.addWidget(self.ee_circle_group)
        self.ee_circle_period.valueChanged.connect(
            lambda _v: self.ee_circle_tdur.setValue(self.ee_circle_period.value() * self.ee_circle_loops.value())
        )
        self.ee_circle_loops.valueChanged.connect(
            lambda _v: self.ee_circle_tdur.setValue(self.ee_circle_period.value() * self.ee_circle_loops.value())
        )
        self.ee_csv_group = QGroupBox("CSV trajectory import")
        csv_g = QGridLayout()
        self.ee_csv_path = QLineEdit("")
        self.ee_csv_path.setPlaceholderText("trajectory/result_segment_latest.csv")
        self.ee_csv_browse_btn = QPushButton("Browse CSV")
        self.ee_csv_vmax_limit = QDoubleSpinBox()
        self.ee_csv_vmax_limit.setRange(0.1, 50.0)
        self.ee_csv_vmax_limit.setDecimals(2)
        self.ee_csv_vmax_limit.setValue(5.0)
        self.ee_csv_z_offset = QDoubleSpinBox()
        self.ee_csv_z_offset.setRange(-20.0, 20.0)
        self.ee_csv_z_offset.setDecimals(3)
        self.ee_csv_z_offset.setValue(0.0)
        self.ee_csv_yaw_const = QDoubleSpinBox()
        self.ee_csv_yaw_const.setRange(-180.0, 180.0)
        self.ee_csv_yaw_const.setDecimals(2)
        self.ee_csv_yaw_const.setValue(0.0)
        self.ee_csv_yaw_hold = QCheckBox("Keep yaw constant")
        self.ee_csv_yaw_hold.setChecked(False)
        csv_g.addWidget(QLabel("CSV path"), 0, 0)
        csv_g.addWidget(self.ee_csv_path, 0, 1, 1, 3)
        csv_g.addWidget(self.ee_csv_browse_btn, 0, 4)
        csv_g.addWidget(QLabel("Max speed limit [m/s]"), 1, 0)
        csv_g.addWidget(self.ee_csv_vmax_limit, 1, 1)
        csv_g.addWidget(QLabel("Z offset [m]"), 2, 0)
        csv_g.addWidget(self.ee_csv_z_offset, 2, 1)
        csv_g.addWidget(self.ee_csv_yaw_hold, 1, 2, 1, 2)
        csv_g.addWidget(QLabel("yaw const [deg]"), 3, 0)
        csv_g.addWidget(self.ee_csv_yaw_const, 3, 1)
        self.ee_csv_group.setLayout(csv_g)
        g_ee.addWidget(self.ee_csv_group)
        self.ee_csv_browse_btn.clicked.connect(self._browse_ee_csv_file)
        # Reuse global sampling dt from Task configuration (avoid duplicated dt controls).
        self.dt_ee_sample = self.dt_plan
        self.ee_plan_type_combo.currentIndexChanged.connect(self._on_ee_plan_type_changed)
        self._on_ee_plan_type_changed()
        self.path_stack.addWidget(w_ee)
        self.opt_stack.addWidget(QWidget())  # EE snap: no extra optimization panel

        # Path stack 2: world acceleration reference (step / sine)
        w_acc = QWidget()
        g_acc = QVBoxLayout(w_acc)
        g_acc.addWidget(
            QLabel(
                "在世界系指定基座线加速度 a_W(t)（单轴）"
            )
        )
        acc_row0 = QHBoxLayout()
        self.acc_track_shape_combo = QComboBox()
        self.acc_track_shape_combo.addItems(["Step", "Sine"])
        acc_row0.addWidget(QLabel("Profile"))
        acc_row0.addWidget(self.acc_track_shape_combo)
        self.acc_track_axis_combo = QComboBox()
        self.acc_track_axis_combo.addItems(["World X", "World Y", "World Z"])
        acc_row0.addWidget(QLabel("Axis"))
        acc_row0.addWidget(self.acc_track_axis_combo)
        g_acc.addLayout(acc_row0)

        acc_p = QGridLayout()
        self.acc_track_px = QDoubleSpinBox()
        self.acc_track_py = QDoubleSpinBox()
        self.acc_track_pz = QDoubleSpinBox()
        for w, v in ((self.acc_track_px, 0.0), (self.acc_track_py, 0.0), (self.acc_track_pz, 1.0)):
            w.setRange(-50.0, 50.0)
            w.setDecimals(4)
            w.setSingleStep(0.05)
            w.setValue(v)
        self.acc_track_yaw_deg = QDoubleSpinBox()
        self.acc_track_yaw_deg.setRange(-180.0, 180.0)
        self.acc_track_yaw_deg.setDecimals(2)
        self.acc_track_yaw_deg.setValue(0.0)
        self.acc_track_duration = QDoubleSpinBox()
        self.acc_track_duration.setRange(0.1, 300.0)
        self.acc_track_duration.setDecimals(2)
        self.acc_track_duration.setValue(8.0)
        acc_p.addWidget(QLabel("Initial p x/y/z [m]"), 0, 0)
        acc_p.addWidget(self.acc_track_px, 0, 1)
        acc_p.addWidget(self.acc_track_py, 0, 2)
        acc_p.addWidget(self.acc_track_pz, 0, 3)
        acc_p.addWidget(QLabel("Yaw const [deg]"), 1, 0)
        acc_p.addWidget(self.acc_track_yaw_deg, 1, 1)
        acc_p.addWidget(QLabel("Duration [s]"), 1, 2)
        acc_p.addWidget(self.acc_track_duration, 1, 3)
        g_acc.addLayout(acc_p)

        self.acc_track_step_group = QGroupBox("Step profile")
        sg = QGridLayout()
        self.acc_track_step_time = QDoubleSpinBox()
        self.acc_track_step_time.setRange(0.0, 300.0)
        self.acc_track_step_time.setDecimals(3)
        self.acc_track_step_time.setValue(2.0)
        self.acc_track_a_before = QDoubleSpinBox()
        self.acc_track_a_after = QDoubleSpinBox()
        for w in (self.acc_track_a_before, self.acc_track_a_after):
            w.setRange(-50.0, 50.0)
            w.setDecimals(4)
            w.setSingleStep(0.1)
        self.acc_track_a_before.setValue(0.0)
        self.acc_track_a_after.setValue(1.0)
        self.acc_track_pulse_end = QDoubleSpinBox()
        self.acc_track_pulse_end.setRange(0.0, 300.0)
        self.acc_track_pulse_end.setDecimals(3)
        self.acc_track_pulse_end.setValue(4.0)
        self.acc_track_brake_to_rest = QCheckBox("脉冲结束后制动至终点速度为零")
        self.acc_track_brake_to_rest.setChecked(True)
        self.acc_track_brake_to_rest.setToolTip(
            "在 t₁ 之前保持阶跃后的加速度；t₁ 之后用常值制动加速度，使该轴在轨迹终点速度约为 0。"
        )
        sg.addWidget(QLabel("Step time t₀ [s] (a = before if t < t₀)"), 0, 0)
        sg.addWidget(self.acc_track_step_time, 0, 1)
        sg.addWidget(QLabel("a before [m/s²]"), 0, 2)
        sg.addWidget(self.acc_track_a_before, 0, 3)
        sg.addWidget(QLabel("a after [m/s²] (t₀ ≤ t < t₁)"), 1, 0)
        sg.addWidget(self.acc_track_a_after, 1, 1)
        sg.addWidget(QLabel("脉冲结束 t₁ [s]"), 1, 2)
        sg.addWidget(self.acc_track_pulse_end, 1, 3)
        sg.addWidget(self.acc_track_brake_to_rest, 2, 0, 1, 4)
        self.acc_track_step_group.setLayout(sg)
        g_acc.addWidget(self.acc_track_step_group)

        self.acc_track_sin_group = QGroupBox("Sine profile")
        sng = QGridLayout()
        self.acc_track_sin_amp = QDoubleSpinBox()
        self.acc_track_sin_amp.setRange(0.0, 50.0)
        self.acc_track_sin_amp.setDecimals(4)
        self.acc_track_sin_amp.setValue(0.5)
        self.acc_track_sin_freq = QDoubleSpinBox()
        self.acc_track_sin_freq.setRange(0.0, 10.0)
        self.acc_track_sin_freq.setDecimals(4)
        self.acc_track_sin_freq.setValue(0.2)
        self.acc_track_sin_phase_deg = QDoubleSpinBox()
        self.acc_track_sin_phase_deg.setRange(-360.0, 360.0)
        self.acc_track_sin_phase_deg.setDecimals(2)
        self.acc_track_sin_phase_deg.setValue(0.0)
        sng.addWidget(QLabel("Amplitude A [m/s²]"), 0, 0)
        sng.addWidget(self.acc_track_sin_amp, 0, 1)
        sng.addWidget(QLabel("Frequency f [Hz]"), 0, 2)
        sng.addWidget(self.acc_track_sin_freq, 0, 3)
        sng.addWidget(QLabel("Phase [deg]"), 1, 0)
        sng.addWidget(self.acc_track_sin_phase_deg, 1, 1)
        self.acc_track_sin_group.setLayout(sng)
        g_acc.addWidget(self.acc_track_sin_group)

        self.acc_track_shape_combo.currentIndexChanged.connect(
            self._on_acc_track_shape_changed
        )
        self._on_acc_track_shape_changed()
        self.path_stack.addWidget(w_acc)

        # Opt stack 2: acc-track actuator / solver options
        w_acc_opt = QWidget()
        g_acc_opt = QVBoxLayout(w_acc_opt)
        self.acc_track_actuator_group = QGroupBox("Actuator dynamics (rotor 1st-order, optional)")
        ag_dyn = QGridLayout()
        self.acc_track_rotor_dyn_chk = QCheckBox(
            "规划时考虑加速度一阶响应 (τ·da/dt + a = a_cmd，模拟旋翼/推力回路)"
        )
        self.acc_track_rotor_dyn_chk.setChecked(False)
        self.acc_track_rotor_dyn_chk.setToolTip(
            "勾选后：先将理想指令加速度 a_cmd 经一阶低通得到可实现加速度 a，再对 a 积分得到 v、p。"
            "初值 a(0)=0。τ 为等效时间常数 [s]。"
        )
        self.acc_track_rotor_tau = QDoubleSpinBox()
        self.acc_track_rotor_tau.setRange(0.001, 10.0)
        self.acc_track_rotor_tau.setDecimals(3)
        self.acc_track_rotor_tau.setSingleStep(0.01)
        self.acc_track_rotor_tau.setValue(0.1)
        self.acc_track_rotor_tau.setToolTip("一阶时间常数 τ [s]，典型旋翼推力回路约 0.05–0.2 s。")
        ag_dyn.addWidget(self.acc_track_rotor_dyn_chk, 0, 0, 1, 2)
        ag_dyn.addWidget(QLabel("时间常数 τ [s]"), 1, 0)
        ag_dyn.addWidget(self.acc_track_rotor_tau, 1, 1)
        self.acc_track_actuator_group.setLayout(ag_dyn)
        g_acc_opt.addWidget(self.acc_track_actuator_group)

        def _on_acc_rotor_dyn_toggled(checked: bool) -> None:
            self.acc_track_rotor_tau.setEnabled(bool(checked))

        self.acc_track_rotor_dyn_chk.toggled.connect(_on_acc_rotor_dyn_toggled)
        _on_acc_rotor_dyn_toggled(self.acc_track_rotor_dyn_chk.isChecked())
        self.opt_stack.addWidget(w_acc_opt)

        self._on_task_robot_changed(self.task_robot_combo.currentText())
        self._on_plan_mode()
        self._update_plan_panel_heights()
        self.plan_actions_group = QGroupBox("Actions")
        plan_actions = QVBoxLayout()
        self.task_generate_btn = QPushButton("Generate trajectory")
        self.task_generate_btn.setMinimumHeight(44)
        self.task_generate_btn.clicked.connect(self._generate_task_trajectory_now)
        plan_actions.addWidget(self.task_generate_btn)
        plan_actions_row2 = QHBoxLayout()
        self.meshcat_plan_btn = QPushButton("Visualize planned trajectory (Meshcat)")
        self.meshcat_plan_btn.clicked.connect(self._visualize_planned_meshcat)
        self.meshcat_plan_btn.setEnabled(False)
        plan_actions_row2.addWidget(self.meshcat_plan_btn)
        self.save_plan_params_btn = QPushButton("Save Planning parameters")
        self.save_plan_params_btn.clicked.connect(lambda: self._save_tab_params(TAB_PLAN))
        self.save_plan_params_as_btn = QPushButton("Save Planning parameters as")
        self.save_plan_params_as_btn.clicked.connect(lambda: self._save_tab_params_as(TAB_PLAN))
        plan_actions_row2.addWidget(self.save_plan_params_btn)
        plan_actions_row2.addWidget(self.save_plan_params_as_btn)
        plan_actions.addLayout(plan_actions_row2)
        saved_row = QHBoxLayout()
        saved_row.addWidget(QLabel("Saved trajectory"))
        self.saved_traj_combo = QComboBox()
        self.saved_traj_combo.setMinimumWidth(160)
        self.saved_traj_combo.setToolTip("已保存的优化轨迹；启动 GUI 时会自动加载上次使用的条目。")
        saved_row.addWidget(self.saved_traj_combo, 1)
        self.load_saved_traj_btn = QPushButton("Load")
        self.load_saved_traj_btn.clicked.connect(self._load_selected_saved_trajectory)
        self.save_saved_traj_btn = QPushButton("Save…")
        self.save_saved_traj_btn.clicked.connect(self._save_trajectory_as_named)
        saved_row.addWidget(self.load_saved_traj_btn)
        saved_row.addWidget(self.save_saved_traj_btn)
        plan_actions.addLayout(saved_row)
        self.plan_actions_group.setLayout(plan_actions)
        self.plan_actions_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        plan_top_layout.addWidget(self.plan_actions_group)
        # 让以上几个分组靠顶部排列，避免在高栏中被均匀拉开产生空隙。
        plan_top_layout.addStretch(1)

        self.plan_info_group = QGroupBox("Info")
        info_layout = QVBoxLayout()
        self.plan_info_text = QTextEdit()
        self.plan_info_text.setReadOnly(True)
        self.plan_info_text.setPlaceholderText("Planning messages and hints...")
        info_layout.addWidget(self.plan_info_text)
        self.plan_info_group.setLayout(info_layout)
        plan_splitter.addWidget(self.plan_info_group)
        plan_splitter.setStretchFactor(0, 3)
        plan_splitter.setStretchFactor(1, 1)
        plan_splitter.setSizes([720, 180])

        # ----- Track tab (scrollable; method selector at top) -----
        tab_track = QWidget()
        track_outer = QVBoxLayout(tab_track)
        track_outer.setContentsMargins(0, 0, 0, 0)
        track_scroll = QScrollArea()
        track_scroll.setWidgetResizable(True)
        track_scroll.setFrameShape(QScrollArea.NoFrame)
        track_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        track_inner = QWidget()
        tk = QVBoxLayout(track_inner)
        track_scroll.setWidget(track_inner)
        track_outer.addWidget(track_scroll)
        self.left_tabs.addTab(tab_track, "Sim Tracking")

        self.track_mode_combo = QComboBox()
        # Let the combo follow the left panel width instead of forcing it wider.
        self.track_mode_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.track_mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.track_mode_combo.setMinimumContentsLength(12)
        _track_mode_items = [
            (
                "Crocoddyl — full-state plan",
                "沿 Full state 规划轨迹闭环跟踪（Crocoddyl，与下方 w_pos 等权重一致）",
            ),
            (
                "Acados — full-state plan",
                "沿 Full state 规划轨迹闭环跟踪（Acados NMPC，代价权重与 Croc 全状态跟踪共用）",
            ),
            (
                "Acados — EE-centric",
                "末端 EE 位置/航向跟踪（需 EE 类规划或 FK 参考）",
            ),
            (
                "Crocoddyl — EE pose",
                "Crocoddyl 末端位姿跟踪",
            ),
        ]
        for label, tip in _track_mode_items:
            self.track_mode_combo.addItem(label)
            self.track_mode_combo.setItemData(
                self.track_mode_combo.count() - 1, tip, Qt.ToolTipRole
            )
        self.T_sim = QDoubleSpinBox()
        self.T_sim.setRange(0.5, 120.0)
        self.T_sim.setValue(8.0)
        self.sim_dt = QDoubleSpinBox()
        self.sim_dt.setRange(0.0005, 0.05)
        self.sim_dt.setDecimals(4)
        self.sim_dt.setValue(0.001)
        self.control_dt = QDoubleSpinBox()
        self.control_dt.setRange(0.001, 0.2)
        self.control_dt.setDecimals(3)
        self.control_dt.setValue(0.01)
        self.dt_mpc = QDoubleSpinBox()
        self.dt_mpc.setRange(0.01, 0.2)
        self.dt_mpc.setValue(0.05)
        self.N_mpc = QSpinBox()
        self.N_mpc.setRange(5, 80)
        self.N_mpc.setValue(35)
        self.w_ee = QDoubleSpinBox()
        self.w_ee.setRange(1.0, 5000.0)
        self.w_ee.setValue(400.0)
        self.w_ee_yaw = QDoubleSpinBox()
        self.w_ee_yaw.setRange(0.0, 2000.0)
        self.w_ee_yaw.setValue(200.0)

        # Crocoddyl EE pose tracking: per-term cost weights (Tracking mode index 3).
        self.croc_ee_w_pos = QDoubleSpinBox()
        self.croc_ee_w_pos.setRange(0.0, 5000.0)
        self.croc_ee_w_pos.setValue(400.0)
        self.croc_ee_w_rot_rp = QDoubleSpinBox()
        self.croc_ee_w_rot_rp.setRange(0.0, 2000.0)
        self.croc_ee_w_rot_rp.setValue(1.0)
        self.croc_ee_w_rot_yaw = QDoubleSpinBox()
        self.croc_ee_w_rot_yaw.setRange(0.0, 2000.0)
        self.croc_ee_w_rot_yaw.setValue(200.0)
        self.croc_ee_w_vel_lin = QDoubleSpinBox()
        self.croc_ee_w_vel_lin.setRange(0.0, 5000.0)
        self.croc_ee_w_vel_lin.setValue(1.0)
        self.croc_ee_w_vel_ang_rp = QDoubleSpinBox()
        self.croc_ee_w_vel_ang_rp.setRange(0.0, 5000.0)
        self.croc_ee_w_vel_ang_rp.setValue(1.0)
        self.croc_ee_w_vel_ang_yaw = QDoubleSpinBox()
        self.croc_ee_w_vel_ang_yaw.setRange(0.0, 5000.0)
        self.croc_ee_w_vel_ang_yaw.setValue(1.0)
        self.croc_ee_w_u = QDoubleSpinBox()
        self.croc_ee_w_u.setRange(0.0, 100.0)
        self.croc_ee_w_u.setDecimals(6)
        self.croc_ee_w_u.setValue(0.0)
        self.croc_ee_w_terminal = QDoubleSpinBox()
        self.croc_ee_w_terminal.setRange(0.0, 100.0)
        self.croc_ee_w_terminal.setDecimals(3)
        self.croc_ee_w_terminal.setValue(3.0)

        self.mpc_max_iter = QSpinBox()
        self.mpc_max_iter.setRange(1, 200)
        self.mpc_max_iter.setValue(20)
        self.mpc_log_iv = QSpinBox()
        self.mpc_log_iv.setRange(0, 1000)
        self.mpc_log_iv.setValue(0)
        self.control_mode_track = QComboBox()
        self.control_mode_track.addItems(["direct (thrust + τ)", "actuator_first_order (ω, T, θ)"])

        # τ for Crocoddyl closed-loop plant lag (full-state + EE-pose modes when "Plant u first-order lag" is on).
        self.tau_thrust_track = QDoubleSpinBox()
        self.tau_thrust_track.setRange(0.0, 2.0)
        self.tau_thrust_track.setDecimals(3)
        self.tau_thrust_track.setSingleStep(0.005)
        self.tau_thrust_track.setValue(0.06)

        self.tau_theta_track = QDoubleSpinBox()
        self.tau_theta_track.setRange(0.0, 2.0)
        self.tau_theta_track.setDecimals(3)
        self.tau_theta_track.setSingleStep(0.005)
        self.tau_theta_track.setValue(0.05)

        self.track_sim_control_stack = QComboBox()
        self.track_sim_control_stack.addItems(
            [
                "direct (MPC u → plant)",
                "px4_rate (ΣT + ω setpoint + mixer)",
            ]
        )
        self.track_sim_control_stack.currentIndexChanged.connect(
            self._refresh_sim_plant_controls_state
        )
        self.px4_rate_Kp_track = QDoubleSpinBox()
        self.px4_rate_Kp_track.setRange(0.0, 500.0)
        self.px4_rate_Kp_track.setDecimals(3)
        self.px4_rate_Kp_track.setSingleStep(0.5)
        self.px4_rate_Kp_track.setValue(12.0)
        self.px4_rate_Kd_track = QDoubleSpinBox()
        self.px4_rate_Kd_track.setRange(0.0, 100.0)
        self.px4_rate_Kd_track.setDecimals(3)
        self.px4_rate_Kd_track.setSingleStep(0.05)
        self.px4_rate_Kd_track.setValue(1.5)
        self._px4_gain_row = QWidget()
        _px4_h = QHBoxLayout(self._px4_gain_row)
        _px4_h.setContentsMargins(0, 0, 0, 0)
        _px4_h.addWidget(QLabel("Kp"))
        _px4_h.addWidget(self.px4_rate_Kp_track)
        _px4_h.addWidget(QLabel("Kd"))
        _px4_h.addWidget(self.px4_rate_Kd_track)
        _px4_h.addStretch(1)

        self.croc_horizon = QSpinBox()
        self.croc_horizon.setRange(5, 120)
        self.croc_horizon.setValue(40)
        self.croc_mpc_iter = QSpinBox()
        self.croc_mpc_iter.setRange(10, 300)
        self.croc_mpc_iter.setValue(60)
        self.w_state_track = QDoubleSpinBox()
        self.w_state_track.setRange(0.0, 1e5)
        self.w_state_track.setValue(10.0)
        self.w_state_reg = QDoubleSpinBox()
        self.w_state_reg.setRange(0.0, 1e5)
        self.w_state_reg.setValue(0.1)
        self.w_control = QDoubleSpinBox()
        self.w_control.setRange(0.0, 100.0)
        self.w_control.setDecimals(6)
        self.w_control.setValue(1e-3)
        self.w_terminal_track = QDoubleSpinBox()
        self.w_terminal_track.setRange(0.0, 1e6)
        self.w_terminal_track.setValue(100.0)
        self.w_pos = QDoubleSpinBox(); self.w_pos.setRange(0.0, 1e5); self.w_pos.setValue(1.0)
        self.w_att = QDoubleSpinBox(); self.w_att.setRange(0.0, 1e5); self.w_att.setValue(1.0)
        self.w_joint = QDoubleSpinBox(); self.w_joint.setRange(0.0, 1e5); self.w_joint.setValue(1.0)
        self.w_vel = QDoubleSpinBox(); self.w_vel.setRange(0.0, 1e5); self.w_vel.setValue(1.0)
        self.w_omega = QDoubleSpinBox(); self.w_omega.setRange(0.0, 1e5); self.w_omega.setValue(1.0)
        self.w_joint_vel = QDoubleSpinBox(); self.w_joint_vel.setRange(0.0, 1e5); self.w_joint_vel.setValue(1.0)
        self.w_u_thrust = QDoubleSpinBox(); self.w_u_thrust.setRange(0.0, 1e5); self.w_u_thrust.setValue(1.0)
        self.w_u_joint_torque = QDoubleSpinBox(); self.w_u_joint_torque.setRange(0.0, 1e5); self.w_u_joint_torque.setValue(1.0)
        self.croc_use_actuator_first_order = QCheckBox("Enable")
        self.croc_use_actuator_first_order.setChecked(False)
        self.croc_use_actuator_first_order.toggled.connect(self._refresh_track_sim_actuator_taus_enabled)
        self.croc_ee_use_thrust_constraints = QCheckBox("enable")
        self.croc_ee_use_thrust_constraints.setChecked(True)

        # Sim-only payload (Croc EE): checkbox + one row (t_grasp, mass, sphere r → I=⅖mr², CoM at origin).
        self.sim_payload_enable = QCheckBox("Enable (MPC nominal model unchanged)")
        self.sim_payload_enable.setChecked(False)
        self.sim_payload_t_grasp = QDoubleSpinBox()
        self.sim_payload_t_grasp.setRange(0.0, 500.0)
        self.sim_payload_t_grasp.setDecimals(3)
        self.sim_payload_t_grasp.setSingleStep(0.1)
        self.sim_payload_t_grasp.setValue(1.0)
        self.sim_payload_mass = QDoubleSpinBox()
        self.sim_payload_mass.setRange(0.0, 50.0)
        self.sim_payload_mass.setDecimals(4)
        self.sim_payload_mass.setSingleStep(0.01)
        self.sim_payload_mass.setValue(0.2)
        self.sim_payload_row = QWidget()
        _spr = QHBoxLayout(self.sim_payload_row)
        _spr.setContentsMargins(0, 0, 0, 0)
        _spr.addWidget(QLabel("t_grasp [s]"))
        _spr.addWidget(self.sim_payload_t_grasp)
        _spr.addWidget(QLabel("mass [kg]"))
        _spr.addWidget(self.sim_payload_mass)
        self.sim_payload_inertia_lbl = QLabel("")
        self.sim_payload_inertia_lbl.setStyleSheet("color: palette(mid);")
        _spr.addWidget(self.sim_payload_inertia_lbl)
        _spr.addStretch(1)

        self.sim_payload_mass.valueChanged.connect(self._refresh_sim_payload_inertia_hint)
        self.sim_payload_enable.toggled.connect(self._on_sim_payload_enable_toggled)
        self._refresh_sim_payload_inertia_hint()

        # Closed-loop simulator (time stepping, plant dynamics extras). Independent of MPC cost / horizon.
        self._track_sim_actuator_hint = QLabel(
            "Note: \"Croc actuator 1st-order\" in the planning panel only affects the trajectory optimization model. "
            "The option here only affects the closed-loop integration plant (Crocoddyl full-state / EE-pose modes): "
            "MPC still solves with ideal u, while simulation can apply a first-order lag to u. "
            "In Acados mode, actuator_first_order under \"Control mode\" belongs to the NMPC model and is independent "
            "from the plant lag option below."
        )
        self._track_sim_actuator_hint.setWordWrap(True)
        self._track_sim_actuator_hint.setStyleSheet("color: palette(mid);")

        sim_g = QGridLayout()
        sim_g.addWidget(QLabel("T_sim [s]"), 0, 0)
        sim_g.addWidget(self.T_sim, 0, 1)
        sim_g.addWidget(QLabel("sim_dt"), 1, 0)
        sim_g.addWidget(self.sim_dt, 1, 1)
        sim_g.addWidget(QLabel("control_dt"), 2, 0)
        sim_g.addWidget(self.control_dt, 2, 1)
        sim_g.addWidget(self._track_sim_actuator_hint, 3, 0, 1, 2)
        sim_g.addWidget(QLabel("Plant: u first-order lag"), 4, 0)
        sim_g.addWidget(self.croc_use_actuator_first_order, 4, 1)
        sim_g.addWidget(QLabel("Sim control stack"), 5, 0)
        sim_g.addWidget(self.track_sim_control_stack, 5, 1)
        sim_g.addWidget(QLabel("PX4-style rate PD"), 6, 0)
        sim_g.addWidget(self._px4_gain_row, 6, 1)
        sim_g.addWidget(QLabel("τ_thrust [s]"), 7, 0)
        sim_g.addWidget(self.tau_thrust_track, 7, 1)
        sim_g.addWidget(QLabel("τ_θ [s]"), 8, 0)
        sim_g.addWidget(self.tau_theta_track, 8, 1)
        self._sim_payload_label = QLabel("Plant: simulation-only payload")
        sim_g.addWidget(self._sim_payload_label, 9, 0)
        sim_g.addWidget(self.sim_payload_enable, 9, 1)
        sim_g.addWidget(self.sim_payload_row, 10, 0, 1, 2)
        sg = QGroupBox("Simulation parameters (closed-loop simulator)")
        sg.setLayout(sim_g)

        self._track_method_hint = QLabel(
            "先完成 Full state 规划后，可选 Crocoddyl 或 Acados 沿规划跟踪；"
            "Acados — full-state plan 与 Crocoddyl 共用 w_pos / w_att / w_vel 等权重。"
        )
        self._track_method_hint.setWordWrap(True)
        self._track_method_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        track_method_group = QGroupBox("Tracking method")
        tm_lay = QVBoxLayout(track_method_group)
        tm_lay.addWidget(self._track_method_hint)
        tm_lay.addWidget(self.track_mode_combo)
        tk.addWidget(track_method_group)

        # Algorithm-dependent parameters (two-column flow; dynamic visibility by algorithm)
        self._algo_grid = QGridLayout()
        self._algo_grid.setHorizontalSpacing(12)
        # Two label/widget column pairs: 0=label,1=widget | 2=label,3=widget
        self._algo_grid.setColumnStretch(1, 1)
        self._algo_grid.setColumnStretch(3, 1)
        self._algo_rows: list[tuple[QLabel, QWidget]] = []
        for lab, w in [
            ("dt_mpc", self.dt_mpc),
            ("N (horizon)", self.N_mpc),
            ("w_ee (Acados)", self.w_ee),
            ("w_ee_yaw (Acados)", self.w_ee_yaw),
            ("Croc EE w_pos", self.croc_ee_w_pos),
            ("Croc EE w_rot_rp", self.croc_ee_w_rot_rp),
            ("Croc EE w_rot_yaw", self.croc_ee_w_rot_yaw),
            ("Croc EE w_vel_lin", self.croc_ee_w_vel_lin),
            ("Croc EE w_vel_ang_rp", self.croc_ee_w_vel_ang_rp),
            ("Croc EE w_vel_ang_yaw", self.croc_ee_w_vel_ang_yaw),
            ("Croc EE w_u (ctrl reg)", self.croc_ee_w_u),
            ("Croc EE terminal scale", self.croc_ee_w_terminal),
            ("mpc max_iter", self.mpc_max_iter),
            ("mpc log ivl", self.mpc_log_iv),
            ("Control mode", self.control_mode_track),
            ("Croc horizon steps", self.croc_horizon),
            ("Croc MPC max_iter", self.croc_mpc_iter),
            ("w_state_track", self.w_state_track),
            ("w_state_reg", self.w_state_reg),
            ("w_control", self.w_control),
            ("w_terminal_track", self.w_terminal_track),
            ("w_pos", self.w_pos),
            ("w_att", self.w_att),
            ("w_joint", self.w_joint),
            ("w_vel", self.w_vel),
            ("w_omega", self.w_omega),
            ("w_joint_vel", self.w_joint_vel),
            ("w_u_thrust", self.w_u_thrust),
            ("w_u_joint_torque", self.w_u_joint_torque),
            ("use thrust constraints", self.croc_ee_use_thrust_constraints),
        ]:
            lb = QLabel(lab)
            self._algo_rows.append((lb, w))

        self.track_algo_group = QGroupBox("Algorithm parameters")
        algo_wrap = QVBoxLayout()
        algo_wrap.addLayout(self._algo_grid)
        self.track_algo_group.setLayout(algo_wrap)
        tk.addWidget(self.track_algo_group)
        self.track_mode_combo.currentIndexChanged.connect(self._on_track_mode_changed)
        self._on_track_mode_changed()

        self.run_track_btn = QPushButton("Run closed-loop tracking")
        self.run_track_btn.clicked.connect(self._run_track)
        self.run_track_btn.setEnabled(False)
        tk.addWidget(self.run_track_btn)

        self.meshcat_track_btn = QPushButton("Visualize closed-loop trajectory (Meshcat)")
        self.meshcat_track_btn.clicked.connect(self._visualize_tracked_meshcat)
        self.meshcat_track_btn.setEnabled(False)
        tk.addWidget(self.meshcat_track_btn)

        tk.addWidget(sg)
        self._refresh_sim_plant_controls_state()

        # ----- Regulation panel (embedded in Tracking tab) -----
        self.reg_group = QGroupBox("Regulation (same controllers/weights as Tracking)")
        reg = QVBoxLayout()

        reg_note = QLabel(
            "Regulation uses the same controllers and algorithm parameters as Tracking "
            "(for direct comparison and tuning)."
        )
        reg_note.setWordWrap(True)
        reg_note.setStyleSheet("color: palette(mid);")
        reg.addWidget(reg_note)

        self.reg_mode_combo = QComboBox()
        self.reg_mode_combo.addItems(
            [
                "Crocoddyl - full-state regulation",
                "Crocoddyl - EE pose regulation",
            ]
        )
        reg.addWidget(QLabel("Regulation method"))
        reg.addWidget(self.reg_mode_combo)

        # Full-state: compact table (rows x0 / x_ref), same columns as planning-style state rows
        self.reg_full_state_label = QLabel(
            "Full-state regulation — state [x,y,z m; j1,j2,yaw °] (row: x0, x_ref)"
        )
        self.reg_full_state_label.setWordWrap(True)
        reg.addWidget(self.reg_full_state_label)
        self.reg_full_state_table = QTableWidget(2, 6)
        self.reg_full_state_table.setHorizontalHeaderLabels(
            ["x [m]", "y [m]", "z [m]", "j1°", "j2°", "yaw°"]
        )
        self.reg_full_state_table.setVerticalHeaderLabels(["x0", "x_ref"])
        _rfh = self.reg_full_state_table.horizontalHeader()
        _rfh.setSectionResizeMode(QHeaderView.Stretch)
        _reg_full_defaults = [
            {"x": 0.0, "y": 0.0, "z": 1.0, "j1": -68.8, "j2": -34.4, "yaw": 0.0},
            {"x": 1.0, "y": 0.5, "z": 1.2, "j1": -45.8, "j2": -17.2, "yaw": 45.0},
        ]
        for r, rowd in enumerate(_reg_full_defaults):
            for c, key in enumerate(["x", "y", "z", "j1", "j2", "yaw"]):
                self.reg_full_state_table.setItem(
                    r, c, QTableWidgetItem(f"{float(rowd[key]):g}")
                )
        reg.addWidget(self.reg_full_state_table)

        # EE regulation: state table + single-row EE pose table
        self.reg_ee_state_label = QLabel(
            "EE regulation — state [x,y,z m; j1,j2,yaw °] (row: x0, x_ref)"
        )
        self.reg_ee_state_label.setWordWrap(True)
        reg.addWidget(self.reg_ee_state_label)
        self.reg_ee_state_table = QTableWidget(2, 6)
        self.reg_ee_state_table.setHorizontalHeaderLabels(
            ["x [m]", "y [m]", "z [m]", "j1°", "j2°", "yaw°"]
        )
        self.reg_ee_state_table.setVerticalHeaderLabels(["x0", "x_ref"])
        _reh = self.reg_ee_state_table.horizontalHeader()
        _reh.setSectionResizeMode(QHeaderView.Stretch)
        _reg_ee_state_defaults = [
            {"x": 0.0, "y": 0.0, "z": 1.0, "j1": -68.8, "j2": -34.4, "yaw": 0.0},
            {"x": 0.0, "y": 0.0, "z": 1.0, "j1": -68.8, "j2": -34.4, "yaw": 0.0},
        ]
        for r, rowd in enumerate(_reg_ee_state_defaults):
            for c, key in enumerate(["x", "y", "z", "j1", "j2", "yaw"]):
                self.reg_ee_state_table.setItem(
                    r, c, QTableWidgetItem(f"{float(rowd[key]):g}")
                )
        reg.addWidget(self.reg_ee_state_table)

        self.reg_ee_pose_label = QLabel("EE regulation — target EE pose (world)")
        reg.addWidget(self.reg_ee_pose_label)
        self.reg_ee_pose_table = QTableWidget(1, 4)
        self.reg_ee_pose_table.setHorizontalHeaderLabels(["x [m]", "y [m]", "z [m]", "yaw°"])
        _rph = self.reg_ee_pose_table.horizontalHeader()
        _rph.setSectionResizeMode(QHeaderView.Stretch)
        self.reg_ee_pose_table.setVerticalHeaderLabels(["target"])
        for c, val in enumerate([1.0, 0.2, 0.9, 0.0]):
            self.reg_ee_pose_table.setItem(0, c, QTableWidgetItem(f"{val:g}"))
        reg.addWidget(self.reg_ee_pose_table)

        self.reg_run_btn = QPushButton("Run closed-loop regulation")
        self.reg_run_btn.clicked.connect(self._run_regulation)
        reg.addWidget(self.reg_run_btn)
        self.reg_group.setLayout(reg)
        tk.addWidget(self.reg_group)
        self.reg_mode_combo.currentIndexChanged.connect(self._on_reg_mode_changed)
        self._on_reg_mode_changed()

        track_param_btns = QHBoxLayout()
        self.save_track_params_btn = QPushButton("Save Tracking parameters")
        self.save_track_params_btn.clicked.connect(lambda: self._save_tab_params(TAB_TRACK))
        self.save_track_params_as_btn = QPushButton("Save Tracking parameters as")
        self.save_track_params_as_btn.clicked.connect(lambda: self._save_tab_params_as(TAB_TRACK))
        track_param_btns.addWidget(self.save_track_params_btn)
        track_param_btns.addWidget(self.save_track_params_as_btn)
        tk.addLayout(track_param_btns)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(140)
        tk.addWidget(self.log_text)

        # ----- ROS Tracking tab -----
        tab_ros_track = QWidget()
        rtt_scroll_area = QScrollArea()
        rtt_scroll_area.setWidgetResizable(True)
        rtt_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        rtt_inner = QWidget()
        rtt = QVBoxLayout(rtt_inner)
        rtt_scroll_area.setWidget(rtt_inner)
        rtt_outer = QVBoxLayout(tab_ros_track)
        rtt_outer.setContentsMargins(0, 0, 0, 0)
        rtt_outer.addWidget(rtt_scroll_area)
        self.left_tabs.addTab(tab_ros_track, "ROS Tracking")

        # 内部跟踪：run_tracking_controller 子进程句柄
        self._rn_process = None

        gazebo_group = QGroupBox("Gazebo")
        gz = QGridLayout()
        gz.addWidget(QLabel("Package"), 0, 0)
        self.gz_pkg_combo = QComboBox()
        self.gz_pkg_combo.setEditable(True)
        self.gz_pkg_combo.addItems(["eagle_mpc_python"])
        self.gz_pkg_combo.setCurrentText("eagle_mpc_python")
        gz.addWidget(self.gz_pkg_combo, 0, 1)
        gz.addWidget(QLabel("Launch file"), 1, 0)
        self.gz_launch_combo = QComboBox()
        self.gz_launch_combo.setEditable(True)
        self.gz_launch_combo.addItems(["s500_uam_sitl.launch", "s500_sitl.launch"])
        self.gz_launch_combo.setCurrentText("s500_uam_sitl.launch")
        gz.addWidget(self.gz_launch_combo, 1, 1)
        gz.addWidget(QLabel("Model"), 2, 0)
        self.gz_model_combo = QComboBox()
        self.gz_model_combo.addItems(["s500_uam", "s500"])
        gz.addWidget(self.gz_model_combo, 2, 1)
        gz.addWidget(QLabel("Model type"), 3, 0)
        self.gz_model_type_combo = QComboBox()
        self.gz_model_type_combo.addItems(["real", "ideal"])
        self.gz_model_type_combo.setCurrentText("real")
        gz.addWidget(self.gz_model_type_combo, 3, 1)
        gz.addWidget(QLabel("Environment"), 4, 0)
        self.gz_world_combo = QComboBox()
        self.gz_world_combo.setEditable(True)
        self.gz_world_combo.addItems(["table_beer_with_stand", "empty", "warehouse"])
        self.gz_world_combo.setCurrentText("table_beer_with_stand")
        gz.addWidget(self.gz_world_combo, 4, 1)
        gz.addWidget(QLabel("Extra args"), 5, 0)
        self.gz_args_edit = QLineEdit("")
        gz.addWidget(self.gz_args_edit, 5, 1)
        gz_btn_row = QHBoxLayout()
        self.gz_start_btn = QPushButton("Start Gazebo")
        self.gz_start_btn.clicked.connect(self._start_ros_gazebo)
        self.gz_stop_btn = QPushButton("Stop Gazebo")
        self.gz_stop_btn.clicked.connect(self._stop_ros_gazebo)
        gz_btn_row.addWidget(self.gz_start_btn)
        gz_btn_row.addWidget(self.gz_stop_btn)
        gz.addLayout(gz_btn_row, 6, 0, 1, 2)
        gazebo_group.setLayout(gz)
        rtt.addWidget(gazebo_group)

        # ── Drone status (MAVROS) + flight-mode services ─────────────────────
        status_group = QGroupBox("Drone Status  (MAVROS)")
        status_v = QVBoxLayout()
        st_grid = QGridLayout()
        st_grid.setHorizontalSpacing(10)
        for c in (1, 3):
            st_grid.setColumnStretch(c, 1)

        def _mk_status_value(text="—"):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: bold;")
            return lbl

        st_grid.addWidget(QLabel("Connection"), 0, 0)
        self.rn_st_conn = _mk_status_value()
        st_grid.addWidget(self.rn_st_conn, 0, 1)
        st_grid.addWidget(QLabel("Arming"), 0, 2)
        self.rn_st_arm = _mk_status_value()
        st_grid.addWidget(self.rn_st_arm, 0, 3)

        st_grid.addWidget(QLabel("Flight mode"), 1, 0)
        self.rn_st_mode = _mk_status_value()
        st_grid.addWidget(self.rn_st_mode, 1, 1)
        st_grid.addWidget(QLabel("Tracking"), 1, 2)
        self.rn_st_track = _mk_status_value("node off")
        st_grid.addWidget(self.rn_st_track, 1, 3)

        self.rn_st_ekf_lbl = QLabel("EKF pos [m]")
        st_grid.addWidget(self.rn_st_ekf_lbl, 2, 0)
        self.rn_st_ekf = _mk_status_value()
        st_grid.addWidget(self.rn_st_ekf, 2, 1, 1, 3)

        st_grid.addWidget(QLabel("Vision pos [m]"), 3, 0)
        self.rn_st_vision = _mk_status_value()
        st_grid.addWidget(self.rn_st_vision, 3, 1, 1, 3)

        st_grid.addWidget(QLabel("Localization"), 4, 0)
        self.rn_st_loc = _mk_status_value()
        self.rn_st_loc.setToolTip(
            "EKF (/mavros/local_position/pose) 与 Vision (/mavros/vision_pose/pose) 位置之差：\n"
            "• <10cm：OK\n"
            "• 10~50cm：WARN（定位漂移）\n"
            "• >50cm：UNAVAILABLE（定位不可用，请勿起飞）"
        )
        st_grid.addWidget(self.rn_st_loc, 4, 1, 1, 3)
        status_v.addLayout(st_grid)

        mode_btn_row = QHBoxLayout()
        self.rn_offboard_btn = QPushButton("Set OFFBOARD")
        self.rn_offboard_btn.setToolTip("调用 /mavros/set_mode 切换到 OFFBOARD 模式。")
        self.rn_offboard_btn.clicked.connect(lambda: self._call_set_flight_mode("OFFBOARD"))
        mode_btn_row.addWidget(self.rn_offboard_btn)

        self.rn_posctl_btn = QPushButton("Set POSCTL")
        self.rn_posctl_btn.setToolTip("调用 /mavros/set_mode 切换到位置控制 (POSCTL) 模式。")
        self.rn_posctl_btn.clicked.connect(lambda: self._call_set_flight_mode("POSCTL"))
        mode_btn_row.addWidget(self.rn_posctl_btn)

        self.rn_arm_btn = QPushButton("Arm")
        self.rn_arm_btn.setToolTip("调用 /mavros/cmd/arming 解锁。")
        self.rn_arm_btn.clicked.connect(lambda: self._call_arm(True))
        mode_btn_row.addWidget(self.rn_arm_btn)

        self.rn_disarm_btn = QPushButton("Disarm")
        self.rn_disarm_btn.setToolTip("调用 /mavros/cmd/arming 上锁。")
        self.rn_disarm_btn.clicked.connect(lambda: self._call_arm(False))
        mode_btn_row.addWidget(self.rn_disarm_btn)
        status_v.addLayout(mode_btn_row)

        # 仅依赖 Gazebo + PX4 SITL + MAVROS 的一键起飞（无需 MPC 跟踪节点）
        gz_takeoff_row = QHBoxLayout()
        self.rn_gz_takeoff_btn = QPushButton("Takeoff 1m (AUTO)")
        self.rn_gz_takeoff_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #1565c0; color: white; }"
        )
        self.rn_gz_takeoff_btn.setToolTip(
            "仅需 Gazebo + PX4 SITL + MAVROS：\n"
            "使用 PX4 AUTO.TAKEOFF 模式一次性解锁并起飞到约 1 米后自动 LOITER 悬停。\n"
            "不占用 OFFBOARD、不持续发流，因此不会与后续 MPC 的 OFFBOARD 接管冲突。\n"
            "（可在仅启动 Gazebo / PX4 SITL 后直接使用。）"
        )
        self.rn_gz_takeoff_btn.clicked.connect(lambda: self._call_gazebo_takeoff(1.0))
        gz_takeoff_row.addWidget(self.rn_gz_takeoff_btn)
        status_v.addLayout(gz_takeoff_row)

        status_group.setLayout(status_v)
        rtt.addWidget(status_group)

        # ── MPC runtime stats (/suite_mpc/stats) ─────────────────────────────
        mpc_stat_group = QGroupBox("MPC Runtime  (/suite_mpc/stats)")
        ms_grid = QGridLayout()
        ms_grid.setHorizontalSpacing(10)
        for c in (1, 3):
            ms_grid.setColumnStretch(c, 1)

        ms_grid.addWidget(QLabel("Loop rate"), 0, 0)
        self.rn_ms_hz = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_hz, 0, 1)
        ms_grid.addWidget(QLabel("Phase / status"), 0, 2)
        self.rn_ms_phase = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_phase, 0, 3)

        ms_grid.addWidget(QLabel("Solve [ms]"), 1, 0)
        self.rn_ms_solve = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_solve, 1, 1)
        ms_grid.addWidget(QLabel("avg / max [ms]"), 1, 2)
        self.rn_ms_solve_stat = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_solve_stat, 1, 3)

        ms_grid.addWidget(QLabel("Iters (SQP/QP)"), 2, 0)
        self.rn_ms_iters = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_iters, 2, 1)
        ms_grid.addWidget(QLabel("Horizon / dt"), 2, 2)
        self.rn_ms_horizon = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_horizon, 2, 3)

        ms_grid.addWidget(QLabel("MPC cost"), 3, 0)
        self.rn_ms_cost = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_cost, 3, 1)
        ms_grid.addWidget(QLabel("Yaw err [deg]"), 3, 2)
        self.rn_ms_err_yaw = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_err_yaw, 3, 3)

        ms_grid.addWidget(QLabel("Pos err x/y/z [m]"), 4, 0)
        self.rn_ms_err_xyz = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_err_xyz, 4, 1)
        ms_grid.addWidget(QLabel("Pos err total [m]"), 4, 2)
        self.rn_ms_err_pos = _mk_status_value()
        ms_grid.addWidget(self.rn_ms_err_pos, 4, 3)
        mpc_stat_group.setLayout(ms_grid)
        rtt.addWidget(mpc_stat_group)

        # 状态监听延迟到有 ROS master 时再初始化（启动 Gazebo / 节点后），
        # 避免在没有 roscore 时 rospy.init_node 阻塞 GUI 启动。
        self._status_monitor_inited = False

        # ── ROS Tracking Node (run_tracking_controller.py) ───────────────────
        ros_node_group = QGroupBox("ROS Tracking Node  (run_tracking_controller.py)")
        ros_node_layout = QVBoxLayout()

        _rn_hint = QLabel(
            "使用 scripts/ 中的 Crocoddyl Python MPC（与 GUI 数值仿真相同的控制器）"
            "在 ROS 环境中进行在线闭环跟踪。\n"
            "支持全状态跟踪（croc_full_state）和 EE 位姿跟踪（croc_ee_pose）两种模式，"
            "MPC 参数在下方独立设置，与仿真参数互不影响。"
        )
        _rn_hint.setWordWrap(True)
        _rn_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        ros_node_layout.addWidget(_rn_hint)

        rn_grid = QGridLayout()

        rn_grid.addWidget(QLabel("Controller mode"), 0, 0)
        self.rn_controller_combo = QComboBox()
        self.rn_controller_combo.addItems(
            ["croc_full_state", "acados_full_state", "croc_ee_pose", "px4", "geometric"]
        )
        self.rn_controller_combo.setToolTip(
            "croc_full_state:   Crocoddyl 全状态跟踪 (build_shooting_problem_along_plan)\n"
            "acados_full_state: Acados NMPC 全状态跟踪（实时，s500 与 s500_uam 均支持）\n"
            "croc_ee_pose:      Crocoddyl EE 位姿跟踪 (build_shooting_problem_along_ee_ref)\n"
            "px4:             run_tracking_controller 内部发送 PositionTarget 给 PX4\n"
            "geometric:       run_tracking_controller 内置 geometric（body_rate + thrust）"
        )
        rn_grid.addWidget(self.rn_controller_combo, 0, 1)

        rn_grid.addWidget(QLabel("Odom source"), 1, 0)
        self.rn_odom_combo = QComboBox()
        self.rn_odom_combo.addItems(["gazebo", "mavros"])
        self.rn_odom_combo.setToolTip(
            "gazebo: 订阅 /gazebo/model_states\nmavros: 订阅 /mavros/local_position/odom"
        )
        rn_grid.addWidget(self.rn_odom_combo, 1, 1)

        rn_grid.addWidget(QLabel("Control rate [Hz]"), 2, 0)
        self.rn_control_rate = QDoubleSpinBox()
        self.rn_control_rate.setRange(10.0, 200.0)
        self.rn_control_rate.setSingleStep(10.0)
        self.rn_control_rate.setValue(50.0)
        rn_grid.addWidget(self.rn_control_rate, 2, 1)

        rn_grid.addWidget(QLabel("Arm control mode"), 3, 0)
        self.rn_arm_mode_combo = QComboBox()
        self.rn_arm_mode_combo.addItems(["position", "position_velocity", "velocity"])
        rn_grid.addWidget(self.rn_arm_mode_combo, 3, 1)

        rn_grid.addWidget(QLabel("Use simulation"), 4, 0)
        self.rn_use_sim_check = QCheckBox()
        self.rn_use_sim_check.setChecked(True)
        self.rn_use_sim_check.setToolTip(
            "勾选：从 /arm_controller/joint_states 读取关节状态（Gazebo 仿真）\n"
            "不勾选：从 /joint_states 读取（实机）"
        )
        rn_grid.addWidget(self.rn_use_sim_check, 4, 1)

        rn_grid.addWidget(QLabel("max_thrust_total [N]"), 5, 0)
        self.rn_max_thrust_total = QDoubleSpinBox()
        self.rn_max_thrust_total.setRange(1.0, 500.0)
        self.rn_max_thrust_total.setDecimals(2)
        self.rn_max_thrust_total.setSingleStep(1.0)
        self.rn_max_thrust_total.setValue(7.43 * 4)
        self.rn_max_thrust_total.setToolTip(
            "CTBR 总推力归一化的分母（全部旋翼最大推力之和，单位 N）。\n"
            "thrust_cmd = sum(rotor_thrust_N) / max_thrust_total，再 clip 到 [0,1]。\n"
            "需与 PX4 实际最大推力一致；启动 tracking node 前设置（在线更新不改此项）。"
        )
        rn_grid.addWidget(self.rn_max_thrust_total, 5, 1)

        ros_node_layout.addLayout(rn_grid)

        # ── ROS MPC Parameters ────────────────────────────────────────────────
        rn_mpc_group = QGroupBox("MPC Parameters")
        rn_mpc_vbox = QVBoxLayout()
        rn_mpc_vbox.setSpacing(4)

        # 公共参数（两种模式均显示）
        rn_common_grid = QGridLayout()
        rn_common_grid.setColumnStretch(1, 1)
        rn_common_grid.setColumnStretch(3, 1)

        rn_common_grid.addWidget(QLabel("dt_mpc [s]"), 0, 0)
        self.rn_dt_mpc = QDoubleSpinBox()
        self.rn_dt_mpc.setRange(0.01, 0.2)
        self.rn_dt_mpc.setDecimals(3)
        self.rn_dt_mpc.setValue(0.05)
        rn_common_grid.addWidget(self.rn_dt_mpc, 0, 1)

        rn_common_grid.addWidget(QLabel("Horizon N"), 0, 2)
        self.rn_horizon = QSpinBox()
        self.rn_horizon.setRange(5, 120)
        self.rn_horizon.setValue(40)
        rn_common_grid.addWidget(self.rn_horizon, 0, 3)

        rn_common_grid.addWidget(QLabel("max_iter"), 1, 0)
        self.rn_mpc_max_iter = QSpinBox()
        self.rn_mpc_max_iter.setRange(1, 300)
        self.rn_mpc_max_iter.setValue(60)
        rn_common_grid.addWidget(self.rn_mpc_max_iter, 1, 1)

        rn_mpc_vbox.addLayout(rn_common_grid)

        # ── Acados 求解器选项（仅 acados_full_state）──────────────────────────
        self._rn_acados_panel = QWidget()
        rn_acados_grid = QGridLayout(self._rn_acados_panel)
        rn_acados_grid.setContentsMargins(0, 0, 0, 0)
        rn_acados_grid.setColumnStretch(1, 1)
        rn_acados_grid.setColumnStretch(3, 1)
        rn_acados_grid.addWidget(QLabel("solver_mode"), 0, 0)
        self.rn_acados_solver_mode = QComboBox()
        self.rn_acados_solver_mode.addItems(["rti", "sqp"])
        self.rn_acados_solver_mode.setToolTip("rti: 实时迭代（推荐 100Hz）；sqp: 每步多轮 SQP（更慢更稳）")
        rn_acados_grid.addWidget(self.rn_acados_solver_mode, 0, 1)
        rn_acados_grid.addWidget(QLabel("integrator"), 0, 2)
        self.rn_acados_integrator = QComboBox()
        self.rn_acados_integrator.addItems(["ERK", "IRK"])
        rn_acados_grid.addWidget(self.rn_acados_integrator, 0, 3)
        rn_acados_grid.addWidget(QLabel("hpipm_mode"), 1, 0)
        self.rn_acados_hpipm = QComboBox()
        self.rn_acados_hpipm.addItems(["SPEED", "BALANCE", "ROBUST"])
        rn_acados_grid.addWidget(self.rn_acados_hpipm, 1, 1)
        rn_acados_grid.addWidget(QLabel("qp_iter_max"), 1, 2)
        self.rn_acados_qp_iter = QSpinBox()
        self.rn_acados_qp_iter.setRange(5, 500)
        self.rn_acados_qp_iter.setValue(20)
        rn_acados_grid.addWidget(self.rn_acados_qp_iter, 1, 3)
        rn_mpc_vbox.addWidget(self._rn_acados_panel)

        # ── full-state 代价权重（Crocoddyl / Acados 各一套 profile，切换算法时自动换）──
        self.rn_fs_weights_title = QLabel("Cost weights — Crocoddyl profile")
        self.rn_fs_weights_title.setStyleSheet("color: palette(mid); font-size: 11px;")
        rn_mpc_vbox.addWidget(self.rn_fs_weights_title)

        self._rn_fs_panel = QWidget()
        rn_fs_grid = QGridLayout(self._rn_fs_panel)
        rn_fs_grid.setContentsMargins(0, 0, 0, 0)
        rn_fs_grid.setColumnStretch(1, 1)
        rn_fs_grid.setColumnStretch(3, 1)

        rn_fs_grid.addWidget(QLabel("w_state_track"), 0, 0)
        self.rn_w_state_track = QDoubleSpinBox()
        self.rn_w_state_track.setRange(0.0, 1e5)
        self.rn_w_state_track.setValue(10.0)
        rn_fs_grid.addWidget(self.rn_w_state_track, 0, 1)

        rn_fs_grid.addWidget(QLabel("w_state_reg"), 0, 2)
        self.rn_w_state_reg = QDoubleSpinBox()
        self.rn_w_state_reg.setRange(0.0, 1e5)
        self.rn_w_state_reg.setValue(0.1)
        rn_fs_grid.addWidget(self.rn_w_state_reg, 0, 3)

        rn_fs_grid.addWidget(QLabel("w_control"), 1, 0)
        self.rn_w_control = QDoubleSpinBox()
        self.rn_w_control.setRange(0.0, 100.0)
        self.rn_w_control.setDecimals(6)
        self.rn_w_control.setValue(1e-3)
        rn_fs_grid.addWidget(self.rn_w_control, 1, 1)

        rn_fs_grid.addWidget(QLabel("w_terminal"), 1, 2)
        self.rn_w_terminal_track = QDoubleSpinBox()
        self.rn_w_terminal_track.setRange(0.0, 1e6)
        self.rn_w_terminal_track.setValue(3.0)
        rn_fs_grid.addWidget(self.rn_w_terminal_track, 1, 3)

        rn_fs_grid.addWidget(QLabel("w_pos / w_att"), 2, 0)
        self.rn_w_pos = QDoubleSpinBox(); self.rn_w_pos.setRange(0.0, 1e5); self.rn_w_pos.setValue(1.0)
        self.rn_w_att = QDoubleSpinBox(); self.rn_w_att.setRange(0.0, 1e5); self.rn_w_att.setValue(1.0)
        _rn_pa = QWidget(); _rn_pa_l = QHBoxLayout(_rn_pa); _rn_pa_l.setContentsMargins(0, 0, 0, 0)
        _rn_pa_l.addWidget(self.rn_w_pos); _rn_pa_l.addWidget(self.rn_w_att)
        rn_fs_grid.addWidget(_rn_pa, 2, 1)

        rn_fs_grid.addWidget(QLabel("w_vel / w_omega"), 2, 2)
        self.rn_w_vel = QDoubleSpinBox(); self.rn_w_vel.setRange(0.0, 1e5); self.rn_w_vel.setValue(1.0)
        self.rn_w_omega = QDoubleSpinBox(); self.rn_w_omega.setRange(0.0, 1e5); self.rn_w_omega.setValue(1.0)
        _rn_vo = QWidget(); _rn_vo_l = QHBoxLayout(_rn_vo); _rn_vo_l.setContentsMargins(0, 0, 0, 0)
        _rn_vo_l.addWidget(self.rn_w_vel); _rn_vo_l.addWidget(self.rn_w_omega)
        rn_fs_grid.addWidget(_rn_vo, 2, 3)

        rn_fs_grid.addWidget(QLabel("w_joint / w_joint_vel"), 3, 0)
        self.rn_w_joint = QDoubleSpinBox(); self.rn_w_joint.setRange(0.0, 1e5); self.rn_w_joint.setValue(1.0)
        self.rn_w_joint_vel = QDoubleSpinBox(); self.rn_w_joint_vel.setRange(0.0, 1e5); self.rn_w_joint_vel.setValue(1.0)
        _rn_jj = QWidget(); _rn_jj_l = QHBoxLayout(_rn_jj); _rn_jj_l.setContentsMargins(0, 0, 0, 0)
        _rn_jj_l.addWidget(self.rn_w_joint); _rn_jj_l.addWidget(self.rn_w_joint_vel)
        rn_fs_grid.addWidget(_rn_jj, 3, 1)

        rn_fs_grid.addWidget(QLabel("w_u_thrust / w_u_joint"), 3, 2)
        self.rn_w_u_thrust = QDoubleSpinBox(); self.rn_w_u_thrust.setRange(0.0, 1e5); self.rn_w_u_thrust.setValue(1.0)
        self.rn_w_u_joint_torque = QDoubleSpinBox(); self.rn_w_u_joint_torque.setRange(0.0, 1e5); self.rn_w_u_joint_torque.setValue(1.0)
        _rn_uu = QWidget(); _rn_uu_l = QHBoxLayout(_rn_uu); _rn_uu_l.setContentsMargins(0, 0, 0, 0)
        _rn_uu_l.addWidget(self.rn_w_u_thrust); _rn_uu_l.addWidget(self.rn_w_u_joint_torque)
        rn_fs_grid.addWidget(_rn_uu, 3, 3)

        rn_mpc_vbox.addWidget(self._rn_fs_panel)

        # ── croc_ee_pose 专用参数面板 ────────────────────────────────────────
        self._rn_ee_panel = QWidget()
        rn_ee_grid = QGridLayout(self._rn_ee_panel)
        rn_ee_grid.setContentsMargins(0, 0, 0, 0)
        rn_ee_grid.setColumnStretch(1, 1)
        rn_ee_grid.setColumnStretch(3, 1)

        rn_ee_grid.addWidget(QLabel("ee w_pos"), 0, 0)
        self.rn_ee_w_pos = QDoubleSpinBox()
        self.rn_ee_w_pos.setRange(0.0, 5000.0)
        self.rn_ee_w_pos.setValue(400.0)
        rn_ee_grid.addWidget(self.rn_ee_w_pos, 0, 1)

        rn_ee_grid.addWidget(QLabel("ee w_rot_rp"), 0, 2)
        self.rn_ee_w_rot_rp = QDoubleSpinBox()
        self.rn_ee_w_rot_rp.setRange(0.0, 2000.0)
        self.rn_ee_w_rot_rp.setValue(1.0)
        rn_ee_grid.addWidget(self.rn_ee_w_rot_rp, 0, 3)

        rn_ee_grid.addWidget(QLabel("ee w_rot_yaw"), 1, 0)
        self.rn_ee_w_rot_yaw = QDoubleSpinBox()
        self.rn_ee_w_rot_yaw.setRange(0.0, 2000.0)
        self.rn_ee_w_rot_yaw.setValue(200.0)
        rn_ee_grid.addWidget(self.rn_ee_w_rot_yaw, 1, 1)

        rn_ee_grid.addWidget(QLabel("ee w_vel_lin"), 1, 2)
        self.rn_ee_w_vel_lin = QDoubleSpinBox()
        self.rn_ee_w_vel_lin.setRange(0.0, 5000.0)
        self.rn_ee_w_vel_lin.setValue(1.0)
        rn_ee_grid.addWidget(self.rn_ee_w_vel_lin, 1, 3)

        rn_ee_grid.addWidget(QLabel("ee w_vel_ang_rp"), 2, 0)
        self.rn_ee_w_vel_ang_rp = QDoubleSpinBox()
        self.rn_ee_w_vel_ang_rp.setRange(0.0, 5000.0)
        self.rn_ee_w_vel_ang_rp.setValue(1.0)
        rn_ee_grid.addWidget(self.rn_ee_w_vel_ang_rp, 2, 1)

        rn_ee_grid.addWidget(QLabel("ee w_vel_ang_yaw"), 2, 2)
        self.rn_ee_w_vel_ang_yaw = QDoubleSpinBox()
        self.rn_ee_w_vel_ang_yaw.setRange(0.0, 5000.0)
        self.rn_ee_w_vel_ang_yaw.setValue(1.0)
        rn_ee_grid.addWidget(self.rn_ee_w_vel_ang_yaw, 2, 3)

        rn_ee_grid.addWidget(QLabel("ee w_u"), 3, 0)
        self.rn_ee_w_u = QDoubleSpinBox()
        self.rn_ee_w_u.setRange(0.0, 100.0)
        self.rn_ee_w_u.setDecimals(6)
        self.rn_ee_w_u.setValue(0.0)
        rn_ee_grid.addWidget(self.rn_ee_w_u, 3, 1)

        rn_ee_grid.addWidget(QLabel("ee w_terminal"), 3, 2)
        self.rn_ee_w_terminal = QDoubleSpinBox()
        self.rn_ee_w_terminal.setRange(0.0, 100.0)
        self.rn_ee_w_terminal.setDecimals(3)
        self.rn_ee_w_terminal.setValue(3.0)
        rn_ee_grid.addWidget(self.rn_ee_w_terminal, 3, 3)

        rn_mpc_vbox.addWidget(self._rn_ee_panel)

        # ── geometric 专用参数面板 ─────────────────────────────────────────────
        self._rn_geo_panel = QWidget()
        rn_geo_grid = QGridLayout(self._rn_geo_panel)
        rn_geo_grid.setContentsMargins(0, 0, 0, 0)
        rn_geo_grid.setColumnStretch(1, 1)
        rn_geo_grid.setColumnStretch(3, 1)

        rn_geo_grid.addWidget(QLabel("geo_kp_pos"), 0, 0)
        self.rn_geo_kp_pos = QDoubleSpinBox()
        self.rn_geo_kp_pos.setRange(0.0, 100.0)
        self.rn_geo_kp_pos.setDecimals(3)
        self.rn_geo_kp_pos.setValue(4.0)
        rn_geo_grid.addWidget(self.rn_geo_kp_pos, 0, 1)

        rn_geo_grid.addWidget(QLabel("geo_kd_vel"), 0, 2)
        self.rn_geo_kd_vel = QDoubleSpinBox()
        self.rn_geo_kd_vel.setRange(0.0, 100.0)
        self.rn_geo_kd_vel.setDecimals(3)
        self.rn_geo_kd_vel.setValue(2.5)
        rn_geo_grid.addWidget(self.rn_geo_kd_vel, 0, 3)

        rn_geo_grid.addWidget(QLabel("geo_kR"), 1, 0)
        self.rn_geo_kR = QDoubleSpinBox()
        self.rn_geo_kR.setRange(0.0, 100.0)
        self.rn_geo_kR.setDecimals(3)
        self.rn_geo_kR.setValue(4.0)
        rn_geo_grid.addWidget(self.rn_geo_kR, 1, 1)

        rn_geo_grid.addWidget(QLabel("geo_kOmega"), 1, 2)
        self.rn_geo_kOmega = QDoubleSpinBox()
        self.rn_geo_kOmega.setRange(0.0, 100.0)
        self.rn_geo_kOmega.setDecimals(3)
        self.rn_geo_kOmega.setValue(0.35)
        rn_geo_grid.addWidget(self.rn_geo_kOmega, 1, 3)

        rn_geo_grid.addWidget(QLabel("geo_max_tilt_deg"), 2, 0)
        self.rn_geo_max_tilt_deg = QDoubleSpinBox()
        self.rn_geo_max_tilt_deg.setRange(1.0, 89.0)
        self.rn_geo_max_tilt_deg.setDecimals(1)
        self.rn_geo_max_tilt_deg.setValue(35.0)
        rn_geo_grid.addWidget(self.rn_geo_max_tilt_deg, 2, 1)

        rn_mpc_vbox.addWidget(self._rn_geo_panel)

        # ── 保存控制器参数（每个算法各自一套，持久化到磁盘，跨重启复用）──────
        self.rn_save_ctrl_params_btn = QPushButton("Save controller parameters")
        self.rn_save_ctrl_params_btn.setToolTip(
            "将当前算法的 MPC/控制器参数保存到磁盘，并为每个 tracking 算法\n"
            "（croc_full_state / acados_full_state / croc_ee_pose / geometric ...）\n"
            "各自维护一套设置；下次启动 GUI 自动恢复，无需重新调参。"
        )
        self.rn_save_ctrl_params_btn.clicked.connect(self._rn_save_controller_profiles)
        rn_mpc_vbox.addWidget(self.rn_save_ctrl_params_btn)

        rn_mpc_group.setLayout(rn_mpc_vbox)
        ros_node_layout.addWidget(rn_mpc_group)

        # Crocoddyl / Acados 各维护一套代价权重（切换 controller mode 时自动保存/加载）
        self._RN_FS_WEIGHT_KEYS = (
            "w_state_track", "w_state_reg", "w_control", "w_terminal_track",
            "w_pos", "w_att", "w_joint", "w_vel", "w_omega", "w_joint_vel",
            "w_u_thrust", "w_u_joint_torque", "mpc_max_iter",
        )
        self._RN_ACADOS_SOLVER_KEYS = (
            "acados_solver_mode", "acados_integrator", "acados_hpipm_mode", "acados_qp_iter_max",
        )
        self._rn_mpc_weight_profiles: dict = {
            "croc_full_state": {}, "acados_full_state": {},
            "croc_ee_pose": {}, "px4": {}, "geometric": {},
        }
        self._rn_mpc_profile_mode: str | None = None
        self._rn_init_mpc_weight_profiles()
        # 从磁盘加载每个算法各自保存过的控制器参数（若有）。
        self._rn_load_controller_profiles()
        self.rn_controller_combo.currentIndexChanged.connect(self._rn_on_controller_mode_changed)
        self._rn_on_controller_mode_changed(0)


        self.rn_status_label = QLabel("节点状态：未启动")
        self.rn_status_label.setStyleSheet("color: gray;")
        ros_node_layout.addWidget(self.rn_status_label)

        # 按钮行 1：启动 / 停止节点进程
        rn_btn_row1 = QHBoxLayout()
        self.rn_launch_btn = QPushButton("▶  Launch ROS Tracking Node")
        self.rn_launch_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; }"
        )
        self.rn_launch_btn.setToolTip(
            "将当前规划导出为 npz，并启动 run_tracking_controller.py 子进程。\n"
            "需要 ROS Master 运行中，且 Python 环境与 run_controller 相同。"
        )
        self.rn_launch_btn.clicked.connect(self._launch_tracking_node)
        self.rn_launch_btn.setEnabled(False)
        rn_btn_row1.addWidget(self.rn_launch_btn)

        self.rn_kill_btn = QPushButton("■  Kill Node")
        self.rn_kill_btn.setStyleSheet(
            "QPushButton { background-color: #b71c1c; color: white; }"
        )
        self.rn_kill_btn.clicked.connect(self._kill_tracking_node)
        self.rn_kill_btn.setEnabled(False)
        rn_btn_row1.addWidget(self.rn_kill_btn)
        ros_node_layout.addLayout(rn_btn_row1)

        # 按钮行 2：ROS 服务调用（紧凑网格布局，节约横向空间）
        rn_btn_grid = QGridLayout()
        rn_btn_grid.setHorizontalSpacing(8)
        rn_btn_grid.setVerticalSpacing(6)
        self.rn_start_svc_btn = QPushButton("rosservice: /start_tracking")
        self.rn_start_svc_btn.setToolTip(
            "调用 /start_tracking 服务开始轨迹跟踪（需要 OFFBOARD 且已解锁）。"
        )
        self.rn_start_svc_btn.clicked.connect(self._call_start_tracking_service)
        self.rn_start_svc_btn.setEnabled(False)
        rn_btn_grid.addWidget(self.rn_start_svc_btn, 0, 0)

        self.rn_stop_svc_btn = QPushButton("rosservice: /stop_tracking")
        self.rn_stop_svc_btn.setToolTip("调用 /stop_tracking 服务暂停跟踪。")
        self.rn_stop_svc_btn.clicked.connect(self._call_stop_tracking_service)
        self.rn_stop_svc_btn.setEnabled(False)
        rn_btn_grid.addWidget(self.rn_stop_svc_btn, 0, 1)

        self.rn_save_svc_btn = QPushButton("rosservice: /save_data")
        self.rn_save_svc_btn.setToolTip("调用 /save_data 服务保存录制数据。")
        self.rn_save_svc_btn.clicked.connect(self._call_save_data_service)
        self.rn_save_svc_btn.setEnabled(False)
        rn_btn_grid.addWidget(self.rn_save_svc_btn, 1, 0)

        self.rn_update_ctrl_btn = QPushButton("rosservice: /update_controller_params")
        self.rn_update_ctrl_btn.setToolTip(
            "在线更新当前 controller mode 与参数（MPC / geometric），无需重启节点。"
        )
        self.rn_update_ctrl_btn.clicked.connect(self._call_update_controller_params)
        self.rn_update_ctrl_btn.setEnabled(False)
        rn_btn_grid.addWidget(self.rn_update_ctrl_btn, 1, 1)

        self.rn_update_traj_btn = QPushButton("rosservice: /update_trajectory")
        self.rn_update_traj_btn.setToolTip(
            "重新读取当前导出的轨迹并重建 MPC（不重启节点）。"
        )
        self.rn_update_traj_btn.clicked.connect(self._call_update_trajectory_service)
        self.rn_update_traj_btn.setEnabled(False)
        rn_btn_grid.addWidget(self.rn_update_traj_btn, 2, 0, 1, 2)
        ros_node_layout.addLayout(rn_btn_grid)

        # 按钮行 3：Take off / Reset
        rn_btn_row3 = QHBoxLayout()
        self.rn_takeoff_btn = QPushButton("Take off")
        self.rn_takeoff_btn.setStyleSheet(
            "QPushButton { background-color: #6a1b9a; color: white; font-weight: bold; }"
        )
        self.rn_takeoff_btn.setToolTip(
            "一键自动起飞（需 MAVROS 已连接 PX4）：\n"
            "1) 在当前位姿持续发 setpoint（满足 OFFBOARD 前置）\n"
            "2) 自动切换 OFFBOARD 并解锁\n"
            "3) 爬升到 1 m，再飞到轨迹起点 x_plan[0] 悬停\n"
            "调用前会导出并刷新节点中的最新规划。"
        )
        self.rn_takeoff_btn.clicked.connect(self._call_take_off)
        self.rn_takeoff_btn.setEnabled(False)
        rn_btn_row3.addWidget(self.rn_takeoff_btn)

        self.rn_reset_svc_btn = QPushButton("rosservice: /reset_to_initial")
        self.rn_reset_svc_btn.setStyleSheet(
            "QPushButton { background-color: #e65100; color: white; font-weight: bold; }"
        )
        self.rn_reset_svc_btn.setToolTip(
            "调用 /reset_to_initial 服务：\n"
            "• 停止轨迹跟踪，重置 warm-start 缓存\n"
            "• 启用 MPC 归位模式，驱动机器人回到 x_plan[0]\n"
            "• 到达目标后自动停止归位控制"
        )
        self.rn_reset_svc_btn.clicked.connect(self._call_reset_to_initial_service)
        self.rn_reset_svc_btn.setEnabled(False)
        rn_btn_row3.addWidget(self.rn_reset_svc_btn)
        ros_node_layout.addLayout(rn_btn_row3)

        # 按钮行 4：离线绘图（加载 npz 数据并在 GUI 中绘制跟踪结果）
        rn_btn_row4 = QHBoxLayout()
        self.rn_plot_data_btn = QPushButton("📊 Plot Saved Tracking Data")
        self.rn_plot_data_btn.setToolTip(
            "打开 npz 文件（由 /save_data 保存），\n"
            "在右侧图表区绘制实际轨迹与参考轨迹的对比及误差。"
        )
        self.rn_plot_data_btn.clicked.connect(self._plot_ros_tracking_data)
        rn_btn_row4.addWidget(self.rn_plot_data_btn)
        ros_node_layout.addLayout(rn_btn_row4)

        ros_node_group.setLayout(ros_node_layout)
        rtt.addWidget(ros_node_group)

        ros_param_btns = QHBoxLayout()
        self.save_ros_params_btn = QPushButton("Save ROS Tracking parameters")
        self.save_ros_params_btn.clicked.connect(lambda: self._save_tab_params(TAB_ROS))
        self.save_ros_params_as_btn = QPushButton("Save ROS Tracking parameters as")
        self.save_ros_params_as_btn.clicked.connect(lambda: self._save_tab_params_as(TAB_ROS))
        ros_param_btns.addWidget(self.save_ros_params_btn)
        ros_param_btns.addWidget(self.save_ros_params_as_btn)
        rtt.addLayout(ros_param_btns)

        # ── Regulation Target 设置组 ──────────────────────────────────────────
        reg_group = QGroupBox("Regulation Target  (MPC 镇定目标)")
        reg_layout = QVBoxLayout()

        _reg_hint = QLabel(
            "设置 MPC regulation 的目标状态（速度默认为 0）。\n"
            "节点启动时自动以 x_plan[0] 为目标进入 regulation 模式；\n"
            "/reset_to_initial 将目标重置为 x_plan[0]；\n"
            "/stop_tracking 将目标更新为当前实际位置（原地悬停）。"
        )
        _reg_hint.setWordWrap(True)
        _reg_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        reg_layout.addWidget(_reg_hint)

        # 两排布局：第 0 排 x / y / z；第 1 排 yaw / j1 / j2
        reg_grid = QGridLayout()
        for c in (1, 3, 5):
            reg_grid.setColumnStretch(c, 1)

        reg_grid.addWidget(QLabel("x [m]"), 0, 0)
        self.rn_reg_x = QDoubleSpinBox()
        self.rn_reg_x.setRange(-50.0, 50.0); self.rn_reg_x.setSingleStep(0.1); self.rn_reg_x.setValue(0.0)
        reg_grid.addWidget(self.rn_reg_x, 0, 1)

        reg_grid.addWidget(QLabel("y [m]"), 0, 2)
        self.rn_reg_y = QDoubleSpinBox()
        self.rn_reg_y.setRange(-50.0, 50.0); self.rn_reg_y.setSingleStep(0.1); self.rn_reg_y.setValue(0.0)
        reg_grid.addWidget(self.rn_reg_y, 0, 3)

        reg_grid.addWidget(QLabel("z [m]"), 0, 4)
        self.rn_reg_z = QDoubleSpinBox()
        self.rn_reg_z.setRange(0.0, 20.0); self.rn_reg_z.setSingleStep(0.05); self.rn_reg_z.setValue(1.0)
        reg_grid.addWidget(self.rn_reg_z, 0, 5)

        reg_grid.addWidget(QLabel("yaw [°]"), 1, 0)
        self.rn_reg_yaw = QDoubleSpinBox()
        self.rn_reg_yaw.setRange(-180.0, 180.0); self.rn_reg_yaw.setSingleStep(5.0); self.rn_reg_yaw.setValue(0.0)
        reg_grid.addWidget(self.rn_reg_yaw, 1, 1)

        reg_grid.addWidget(QLabel("j1 [°]"), 1, 2)
        self.rn_reg_j1 = QDoubleSpinBox()
        self.rn_reg_j1.setRange(-180.0, 180.0); self.rn_reg_j1.setSingleStep(5.0); self.rn_reg_j1.setValue(0.0)
        reg_grid.addWidget(self.rn_reg_j1, 1, 3)

        reg_grid.addWidget(QLabel("j2 [°]"), 1, 4)
        self.rn_reg_j2 = QDoubleSpinBox()
        self.rn_reg_j2.setRange(-180.0, 180.0); self.rn_reg_j2.setSingleStep(5.0); self.rn_reg_j2.setValue(0.0)
        reg_grid.addWidget(self.rn_reg_j2, 1, 5)

        reg_layout.addLayout(reg_grid)

        reg_btn_row = QHBoxLayout()
        self.rn_set_reg_btn = QPushButton("📍 Set Regulation Target")
        self.rn_set_reg_btn.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; font-weight: bold; }"
        )
        self.rn_set_reg_btn.setToolTip(
            "发布 regulation 目标到节点（话题 ~/regulation_target）。\n"
            "节点收到后立即切换 MPC 镇定目标，warm-start 重置。\n"
            "若当前正在 tracking，目标暂存，stop 后生效。"
        )
        self.rn_set_reg_btn.clicked.connect(self._call_set_regulation_target)
        self.rn_set_reg_btn.setEnabled(False)
        reg_btn_row.addWidget(self.rn_set_reg_btn)
        reg_layout.addLayout(reg_btn_row)

        reg_group.setLayout(reg_layout)
        rtt.addWidget(reg_group)
        rtt.addStretch(1)

        # ----- Right: plots -----
        right = QTabWidget()
        self._right_plot_tabs = right
        root.addWidget(right, stretch=1)

        def embed_fig(title: str, figsize=(14, 9)):
            w = QWidget()
            l = QVBoxLayout(w)
            fig = Figure(figsize=figsize)
            cv = FigureCanvas(fig)
            tb = NavigationToolbar(cv, w)
            l.addWidget(tb)
            l.addWidget(cv)
            right.addTab(w, title)
            return fig, cv

        self.fig_states, self.cv_states = embed_fig("States", (12, 12))
        self.fig_control, self.cv_control = embed_fig("Control", (12, 11))
        # CTBR/feedback 与控制输入合并到同一个 "Control" 标签页（别名复用）。
        self.fig_ctbr, self.cv_ctbr = self.fig_control, self.cv_control
        self.fig_3d_track, self.cv_3d_track = embed_fig("Base 3D", (10, 8))
        self.fig_traj_dash, self.cv_traj_dash = embed_fig("Tracking / MPC", (12, 11))
        self.fig_cost_analysis, self.cv_cost_analysis = embed_fig("Cost analysis", (12, 10))
        # Backward-compatible aliases for existing planning preview rendering.
        self.fig_combined, self.cv_combined = self.fig_states, self.cv_states

        if self._import_err:
            self.log(f"trajectory_gui import warning: {self._import_err!r}")
        if not self._EE_MPC_OK:
            self.log("EE MPC (Acados) is unavailable: EE-centric tracking will fail.")
        if not self._CROC_EE_OK:
            self.log("Crocoddyl EE pose tracking is unavailable.")

    from PyQt5.QtCore import pyqtSlot as _pyqtSlot

    @_pyqtSlot(str)
    def log(self, msg: str) -> None:
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        if hasattr(self, "plan_info_text"):
            self.plan_info_text.append(msg)
            self.plan_info_text.verticalScrollBar().setValue(
                self.plan_info_text.verticalScrollBar().maximum()
            )

    @staticmethod
    def _mixed_rows_to_plot_xyz(
        sorted_rows: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """From GUI mixed waypoint rows to (times, base_xyz with NaN for EE rows, ee_xyz with NaN for Base rows)."""
        if not sorted_rows:
            return None, None, None
        tw = np.array([float(r[7]) for r in sorted_rows], dtype=float)
        base: list[list[float]] = []
        ee: list[list[float]] = []
        for r in sorted_rows:
            is_ee = str(r[0]).strip().lower().startswith("e")
            x, y, z = float(r[1]), float(r[2]), float(r[3])
            if is_ee:
                base.append([float("nan"), float("nan"), float("nan")])
                ee.append([x, y, z])
            else:
                base.append([x, y, z])
                ee.append([float("nan"), float("nan"), float("nan")])
        return tw, np.array(base, dtype=float), np.array(ee, dtype=float)

    def _redraw_combined_views(self, res: dict | None = None) -> None:
        # Planning preview uses the same rendering framework as EE tracking GUI.
        # Tracking results are rendered by `_render_tracking_figures` directly.
        if res is None and self._plan_bundle is not None:
            pb = self._plan_bundle
            if self._ee_mpc is not None:
                kind = pb.get("kind")
                if kind in ("full_croc", "full_acados"):
                    self._render_planning_reference_full_state()
                    return
                if kind == "ee_snap":
                    self._render_planning_reference_ee_snap()
                    return
        # Fallback (for ee_snap planning or missing deps)
        self._draw_suite_states_3d_combined(res)
        self.cv_combined.draw()

    def _set_control_tab_uam_placeholder(self) -> None:
        """UAM / Acados：旋翼与关节指令仍在 States 页的 4×4 面板中。"""
        self.fig_control.clear()
        ax = self.fig_control.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "UAM / Acados：执行器指令见「States」标签页中的 Acados 布局。",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")

    def _render_s500_base_only_planning_figures(
        self,
        t_rel: np.ndarray,
        simX: np.ndarray,
        title_prefix: str,
        u_plan: np.ndarray | None = None,
        *,
        t_ref: np.ndarray | None = None,
        x_ref: np.ndarray | None = None,
        t_u_ref: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,
        mpc_solve: dict | None = None,
    ) -> None:
        """s500 机体状态/控制图。可选 ``t_ref``/``x_ref`` 在同一时间轴上叠加参考（虚线），用于对比跟踪效果。"""
        t = np.asarray(t_rel, dtype=float).flatten()
        X = _ensure_uam17_for_s500_base_plot(np.asarray(simX, dtype=float))
        if X.ndim != 2 or X.shape[0] != t.shape[0]:
            raise ValueError("Invalid base-only plotting arrays")

        Xr_on_t: np.ndarray | None = None
        if t_ref is not None and x_ref is not None:
            tr = np.asarray(t_ref, dtype=float).flatten()
            Xraw = _ensure_uam17_for_s500_base_plot(np.asarray(x_ref, dtype=float))
            if Xraw.ndim == 2 and Xraw.shape[0] == tr.size and tr.size >= 1 and t.size >= 1:
                if tr.shape == t.shape and np.allclose(tr, t, rtol=0.0, atol=1e-6):
                    Xr_on_t = Xraw.copy()
                elif tr.size >= 2 and t.size >= 2:
                    Xr_on_t = np.column_stack([np.interp(t, tr, Xraw[:, c]) for c in range(Xraw.shape[1])])
                else:
                    Xr_on_t = np.tile(Xraw[0:1, :], (t.size, 1))
        if Xr_on_t is not None:
            Xr_on_t = _ensure_uam17_for_s500_base_plot(Xr_on_t)

        self.fig_states.clear()

        pos = X[:, 0:3]
        eul = _euler_deg_from_simX(X)
        vel = X[:, 9:12] if X.shape[1] >= 12 else np.zeros((len(t), 3), dtype=float)
        omg = X[:, 12:15] if X.shape[1] >= 15 else np.zeros((len(t), 3), dtype=float)
        omg_deg = np.degrees(omg)
        if len(t) >= 2:
            acc = np.gradient(vel, t, axis=0)
            jerk = np.gradient(acc, t, axis=0)
            snap = np.gradient(jerk, t, axis=0)
            omg_rad = np.asarray(omg, dtype=float)
            alpha_rad = np.gradient(omg_rad, t, axis=0)
        else:
            acc = np.zeros_like(vel)
            jerk = np.zeros_like(vel)
            snap = np.zeros_like(vel)
            alpha_rad = np.zeros_like(omg)

        alpha_deg = np.degrees(alpha_rad)

        pos_r = eul_r = vel_r = omg_deg_r = acc_r = jerk_r = snap_r = alpha_deg_r = None
        if Xr_on_t is not None and Xr_on_t.shape[0] == t.shape[0]:
            pos_r = Xr_on_t[:, 0:3]
            eul_r = _euler_deg_from_simX(Xr_on_t)
            vel_r = Xr_on_t[:, 9:12] if Xr_on_t.shape[1] >= 12 else np.zeros((len(t), 3), dtype=float)
            omg_r = Xr_on_t[:, 12:15] if Xr_on_t.shape[1] >= 15 else np.zeros((len(t), 3), dtype=float)
            omg_deg_r = np.degrees(omg_r)
            if len(t) >= 2:
                acc_r = np.gradient(vel_r, t, axis=0)
                jerk_r = np.gradient(acc_r, t, axis=0)
                snap_r = np.gradient(jerk_r, t, axis=0)
                alpha_deg_r = np.degrees(np.gradient(np.asarray(omg_r, dtype=float), t, axis=0))
            else:
                acc_r = np.zeros_like(vel_r)
                jerk_r = np.zeros_like(vel_r)
                snap_r = np.zeros_like(vel_r)
                alpha_deg_r = np.zeros_like(omg_deg_r)

        gs = self.fig_states.add_gridspec(4, 2, hspace=0.38, wspace=0.32)
        axs = [self.fig_states.add_subplot(gs[i, j]) for i in range(4) for j in range(2)]
        titles = (
            ("Base position", "m"),
            ("Base orientation (Euler ZYX)", "deg"),
            ("Base linear velocity", "m/s"),
            ("Base angular velocity", "deg/s"),
            ("Base linear acceleration (d v/dt)", "m/s²"),
            ("Base angular acceleration (d ω/dt)", "deg/s²"),
            ("Base linear jerk (d a/dt)", "m/s³"),
            ("Base linear snap (d j/dt)", "m/s⁴"),
        )
        series_meas = (
            (pos, ("x", "y", "z")),
            (eul, ("roll", "pitch", "yaw")),
            (vel, ("vx", "vy", "vz")),
            (omg_deg, ("ωx", "ωy", "ωz")),
            (acc, ("ax", "ay", "az")),
            (alpha_deg, ("αx", "αy", "αz")),
            (jerk, ("jx", "jy", "jz")),
            (snap, ("sx", "sy", "sz")),
        )
        series_ref_rows = [None] * 8
        if pos_r is not None:
            series_ref_rows = [
                (pos_r, ("x", "y", "z")),
                (eul_r, ("roll", "pitch", "yaw")),
                (vel_r, ("vx", "vy", "vz")),
                (omg_deg_r, ("ωx", "ωy", "ωz")),
                (acc_r, ("ax", "ay", "az")),
                (alpha_deg_r, ("αx", "αy", "αz")),
                (jerk_r, ("jx", "jy", "jz")),
                (snap_r, ("sx", "sy", "sz")),
            ]
        colors = ("r", "g", "b")
        for ax, (arr, names), (ttl, yl), ref_row in zip(axs[:8], series_meas, titles[:8], series_ref_rows):
            for j in range(3):
                ax.plot(t, arr[:, j], colors[j] + "-", lw=1.05, label=names[j])
            if ref_row is not None:
                if (
                    not isinstance(ref_row, (tuple, list))
                    or len(ref_row) < 2
                    or ref_row[0] is None
                    or ref_row[1] is None
                ):
                    pass
                else:
                    arr_rf, names_rf = ref_row[0], ref_row[1]
                    if (
                        hasattr(arr_rf, "shape")
                        and arr_rf.ndim == 2
                        and arr_rf.shape[0] == t.size
                        and arr_rf.shape[1] >= 3
                        and len(names_rf) >= 3
                    ):
                        for j in range(3):
                            ax.plot(
                                t,
                                arr_rf[:, j],
                                colors[j] + "--",
                                lw=1.0,
                                alpha=0.88,
                                label=f"{names_rf[j]} ref",
                            )
            ax.set_title(ttl, fontsize=_mpl_pt(9))
            ax.set_ylabel(yl)
            ax.set_xlabel("t [s]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=5, framealpha=0.9, ncol=3)

        suf = f"{title_prefix} — base state (pos…snap)"
        if pos_r is not None:
            suf += " (meas solid / ref dashed)"
        self.fig_states.suptitle(suf, fontsize=11, y=0.995)

        # ── Controls tab (s500) ─────────────────────────────────────────────
        self.fig_control.clear()
        ax_u = self.fig_control.add_subplot(111)
        has_u_ref = u_ref is not None and t_u_ref is not None
        if u_plan is not None:
            U = np.asarray(u_plan, dtype=float)
            if U.ndim == 2 and U.shape[0] > 0:
                nu = int(U.shape[1])
                tu = t[: U.shape[0]]
                for j in range(nu):
                    ax_u.plot(tu, U[:, j], lw=1.0, label=f"u{j+1}")
                if has_u_ref:
                    Ur = np.asarray(u_ref, dtype=float)
                    tur = np.asarray(t_u_ref, dtype=float).flatten()
                    if Ur.ndim == 2 and Ur.shape[0] == tur.size and Ur.shape[1] == nu:
                        for j in range(nu):
                            ax_u.plot(
                                tu,
                                np.interp(tu, tur, Ur[:, j]),
                                "--",
                                lw=0.95,
                                alpha=0.88,
                                label=f"u{j+1} ref",
                            )
                ax_u.legend(loc="upper right", fontsize=_mpl_pt(7), framealpha=0.9, ncol=2)
            else:
                ax_u.text(0.5, 0.5, "No u_plan", ha="center", va="center", transform=ax_u.transAxes)
        else:
            ax_u.text(0.5, 0.5, "No u_plan", ha="center", va="center", transform=ax_u.transAxes)
        ctl_suf = "Control inputs (plan)"
        if pos_r is not None and has_u_ref:
            ctl_suf = "Control inputs (meas vs ref)"
        elif pos_r is not None:
            ctl_suf = "Control inputs (meas)"
        ax_u.set_title(ctl_suf, fontsize=_mpl_pt(10))
        ax_u.set_ylabel("u")
        ax_u.set_xlabel("t [s]")
        ax_u.grid(True, alpha=0.3)
        self.fig_control.suptitle(f"{title_prefix} — s500 actuation", fontsize=11, y=0.98)

        self.fig_3d_track.clear()
        ax3 = self.fig_3d_track.add_subplot(111, projection="3d")
        lbl3 = "tracked" if pos_r is not None else "base ref"
        ax3.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", lw=1.6, label=lbl3)
        if pos_r is not None:
            ax3.plot(
                pos_r[:, 0],
                pos_r[:, 1],
                pos_r[:, 2],
                color="tab:orange",
                ls="--",
                lw=1.35,
                alpha=0.92,
                label="reference",
            )
        ax3.set_xlabel("X [m]")
        ax3.set_ylabel("Y [m]")
        ax3.set_zlabel("Z [m]")
        ax3.set_title(f"{title_prefix} (base_link only, equal XYZ scale)", fontsize=_mpl_pt(10))
        ax3.legend(loc="upper left", fontsize=_mpl_pt(7), framealpha=0.9)
        _set_mplot3d_equal_xyz(ax3, pos, pos_r)

        self.fig_traj_dash.clear()
        self.fig_traj_dash.suptitle(
            f"{title_prefix} — tracking errors & kinematics (see Controls tab for u)",
            fontsize=11,
            y=0.98,
        )
        ad = [self.fig_traj_dash.add_subplot(4, 2, i + 1) for i in range(8)]
        ad[0].plot(pos[:, 0], pos[:, 1], "b-", lw=1.2, label="meas")
        if pos_r is not None:
            ad[0].plot(pos_r[:, 0], pos_r[:, 1], color="tab:orange", ls="--", lw=1.1, alpha=0.9, label="ref")
        ad[0].set_title("XY path", fontsize=_mpl_pt(9))
        ad[0].set_xlabel("x [m]")
        ad[0].set_ylabel("y [m]")
        _xy_stack = [np.column_stack((pos[:, 0], pos[:, 1]))]
        if pos_r is not None:
            _xy_stack.append(np.column_stack((pos_r[:, 0], pos_r[:, 1])))
        _set_2d_path_equal_meters(ad[0], *_xy_stack)
        if pos_r is not None:
            ad[0].legend(loc="best", fontsize=_mpl_pt(7))
        ad[1].plot(pos[:, 0], pos[:, 2], "b-", lw=1.2, label="meas")
        if pos_r is not None:
            ad[1].plot(pos_r[:, 0], pos_r[:, 2], color="tab:orange", ls="--", lw=1.1, alpha=0.9, label="ref")
        ad[1].set_title("XZ path", fontsize=_mpl_pt(9))
        ad[1].set_xlabel("x [m]")
        ad[1].set_ylabel("z [m]")
        _xz_stack = [np.column_stack((pos[:, 0], pos[:, 2]))]
        if pos_r is not None:
            _xz_stack.append(np.column_stack((pos_r[:, 0], pos_r[:, 2])))
        _set_2d_path_equal_meters(ad[1], *_xz_stack)
        if pos_r is not None:
            ad[1].legend(loc="best", fontsize=_mpl_pt(7))

        if pos_r is not None:
            e_p = pos - pos_r
            for j, c in enumerate("rgb"):
                ad[2].plot(t, e_p[:, j], color=c, lw=1.0, label=f"e_{'xyz'[j]}")
            ad[2].plot(t, np.linalg.norm(e_p, axis=1), "k--", lw=0.95, alpha=0.65, label=r"$\|e_p\|$")
            ad[2].axhline(0.0, color="gray", ls=":", lw=0.7)
            ad[2].set_title("Base position tracking error", fontsize=_mpl_pt(9))
        else:
            ad[2].text(0.5, 0.5, "No reference — position error N/A", ha="center", va="center", transform=ad[2].transAxes)
            ad[2].set_title("Base position tracking error", fontsize=_mpl_pt(9))
        ad[2].set_xlabel("t [s]")
        ad[2].set_ylabel("m")
        if pos_r is not None:
            ad[2].legend(loc="best", fontsize=6, ncol=2)
        ad[2].grid(True, alpha=0.3)

        if vel_r is not None:
            e_v = vel - vel_r
            for j, c in enumerate("rgb"):
                ad[3].plot(t, e_v[:, j], color=c, lw=1.0, label=f"e_v{'xyz'[j]}")
            ad[3].plot(t, np.linalg.norm(e_v, axis=1), "k--", lw=0.95, alpha=0.65, label=r"$\|e_v\|$")
            ad[3].axhline(0.0, color="gray", ls=":", lw=0.7)
            ad[3].set_title("Base velocity tracking error", fontsize=_mpl_pt(9))
        else:
            ad[3].text(0.5, 0.5, "No reference — velocity error N/A", ha="center", va="center", transform=ad[3].transAxes)
            ad[3].set_title("Base velocity tracking error", fontsize=_mpl_pt(9))
        ad[3].set_xlabel("t [s]")
        ad[3].set_ylabel("m/s")
        if vel_r is not None:
            ad[3].legend(loc="best", fontsize=6, ncol=2)
        ad[3].grid(True, alpha=0.3)

        speed = np.linalg.norm(vel, axis=1)
        ad[4].plot(t, speed, "k-", lw=1.2, label="meas")
        if vel_r is not None:
            ad[4].plot(t, np.linalg.norm(vel_r, axis=1), color="tab:orange", ls="--", lw=1.05, alpha=0.9, label="ref")
        ad[4].set_title("Speed norm", fontsize=_mpl_pt(9))
        ad[4].set_xlabel("t [s]")
        ad[4].set_ylabel("m/s")
        if vel_r is not None:
            ad[4].legend(loc="best", fontsize=_mpl_pt(7))
        ad[4].grid(True, alpha=0.3)
        acc_norm = np.linalg.norm(acc, axis=1)
        jerk_norm = np.linalg.norm(jerk, axis=1)
        ad[5].plot(t, acc_norm, "m-", lw=1.2, label="meas")
        if acc_r is not None:
            ad[5].plot(t, np.linalg.norm(acc_r, axis=1), color="tab:orange", ls="--", lw=1.05, alpha=0.9, label="ref")
        ad[5].set_title("Acceleration norm", fontsize=_mpl_pt(9))
        ad[5].set_xlabel("t [s]")
        ad[5].set_ylabel("m/s²")
        if acc_r is not None:
            ad[5].legend(loc="best", fontsize=_mpl_pt(7))
        ad[5].grid(True, alpha=0.3)
        ad[6].plot(t, jerk_norm, "c-", lw=1.2, label="meas")
        if jerk_r is not None:
            ad[6].plot(t, np.linalg.norm(jerk_r, axis=1), color="tab:orange", ls="--", lw=1.05, alpha=0.9, label="ref")
        ad[6].set_title("Jerk norm", fontsize=_mpl_pt(9))
        ad[6].set_xlabel("t [s]")
        ad[6].set_ylabel("m/s³")
        if jerk_r is not None:
            ad[6].legend(loc="best", fontsize=_mpl_pt(7))
        ad[6].grid(True, alpha=0.3)

        ms = mpc_solve if isinstance(mpc_solve, dict) else {}
        wall = np.asarray(ms.get("wall_s", []), dtype=float).flatten()
        nit = np.asarray(ms.get("nlp_iter", []), dtype=float).flatten().astype(int, copy=False)
        n_t = int(t.size)
        nw = int(wall.size)
        if nw == 0:
            t_u_mpc = t[:-1] if n_t > 1 else t
        elif nw == n_t:
            t_u_mpc = t.copy()
        elif nw == n_t - 1:
            t_u_mpc = t[:-1]
        else:
            m = min(nw, n_t)
            t_u_mpc = t[:m]
            wall = wall[:m]
        L = int(wall.size)
        if L and nit.size > L:
            nit = nit[:L]
        if wall.size:
            ax_m = ad[7]
            ax_t = ax_m.twinx()
            ax_m.plot(t_u_mpc, wall * 1000.0, "C0-", lw=0.9, label="wall time")
            ax_m.set_ylabel("MPC wall time [ms]", color="C0")
            ax_m.tick_params(axis="y", labelcolor="C0")
            if nit.size == wall.size:
                ax_t.step(t_u_mpc, nit, where="post", color="C2", lw=0.85, label="nlp_iter")
                ax_t.set_ylabel("SQP iterations", color="C2")
                ax_t.tick_params(axis="y", labelcolor="C2")
            ax_m.set_title("MPC solve (wall time + iterations)", fontsize=_mpl_pt(9))
            ax_m.set_xlabel("t [s]")
            ax_m.grid(True, alpha=0.3)
        else:
            snap_norm = np.linalg.norm(snap, axis=1)
            ad[7].plot(t, snap_norm, color="tab:purple", lw=1.05, label=r"$\|$snap$\|$ meas")
            if snap_r is not None:
                ad[7].plot(t, np.linalg.norm(snap_r, axis=1), "k--", lw=0.95, alpha=0.85, label=r"$\|$snap$\|$ ref")
            ad[7].set_title("Linear snap norm (no MPC log)", fontsize=_mpl_pt(9))
            ad[7].set_xlabel("t [s]")
            ad[7].set_ylabel("m/s⁴")
            ad[7].legend(loc="best", fontsize=_mpl_pt(7))
            ad[7].grid(True, alpha=0.3)

    def _render_planning_reference_full_state(self) -> None:
        """Render the full-state planning *reference* using the same dashboard framework as EE tracking GUI."""
        import matplotlib.figure

        pb = self._plan_bundle
        assert pb is not None
        em = self._ee_mpc
        assert em is not None

        t = np.asarray(pb["t_plan"], dtype=float).flatten()
        X = np.asarray(pb["x_plan"], dtype=float)
        if X.ndim != 2:
            raise ValueError(f"x_plan must be 2D, got shape {X.shape}")
        if X.shape[1] > 17:
            X = X[:, :17]
        if X.shape[0] != t.shape[0]:
            raise ValueError(f"t_plan and x_plan length mismatch: {t.shape[0]} vs {X.shape[0]}")

        t_rel = t - t[0]
        simX = X

        # EE reference from FK along the planned full-state trajectory.
        # In s500 mode (no arm), keep EE channels empty.
        if self._is_s500_mode():
            ee_pos = np.full((simX.shape[0], 3), np.nan, dtype=float)
            yaw_ref = np.full(simX.shape[0], np.nan, dtype=float)
            ee_yaw = yaw_ref.copy()
        else:
            rm, eid = self._robot_model_and_ee()
            data = rm.createData()
            from s500_uam_trajectory_planner import compute_ee_kinematics_along_trajectory

            ee_pos, _, ee_rpy, _ = compute_ee_kinematics_along_trajectory(simX, rm, data, eid)
            yaw_ref = np.unwrap(np.asarray(ee_rpy[:, 2], dtype=float).flatten())
            ee_yaw = yaw_ref.copy()

        n = len(t_rel)
        u_plan = np.asarray(pb.get("u_plan", np.zeros((0, 6), dtype=float)), dtype=float)
        if u_plan.ndim != 2:
            u_plan = np.zeros((0, 6), dtype=float)
        n_u = max(0, n - 1)
        if u_plan.shape[0] > n_u:
            u = u_plan[:n_u, :]
        elif u_plan.shape[0] < n_u:
            if u_plan.shape[0] == 0:
                u = np.zeros((n_u, 6), dtype=float)
            else:
                pad = np.repeat(u_plan[-1:, :], n_u - u_plan.shape[0], axis=0)
                u = np.vstack([u_plan, pad])
        else:
            u = u_plan
        err = np.zeros(n, dtype=float)
        err_yaw = np.zeros(n, dtype=float)

        res_ref = {
            "t": t_rel,
            "x": simX,
            "u": u,
            "ee": np.asarray(ee_pos, dtype=float),
            "p_ref": np.asarray(ee_pos, dtype=float),
            "err": err,
            "ee_yaw": ee_yaw,
            "yaw_ref": yaw_ref,
            "err_yaw": err_yaw,
            "control_mode": "direct",
            "sim_dt": float(self.sim_dt.value()),
            "control_dt": float(self.control_dt.value()),
            "mpc_stride": 1,
            "mpc_solve": {"nlp_iter": [], "cpu_s": [], "wall_s": [], "status": []},
        }

        tw_rel = None
        base_wp = None
        ee_wp = None
        rows = pb.get("plan_mixed_wp_rows")
        if rows:
            tw, bx, ex = self._mixed_rows_to_plot_xyz(rows)
            if tw is not None:
                t0 = float(t.flatten()[0])
                tw_rel = tw - t0
                base_wp, ee_wp = bx, ex

        fs = self.fig_states if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_into_figure else None
        f3 = self.fig_3d_track if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_3d_into_figure else None

        traj_meta = None
        fr = getattr(self, "_full_plan_result", None)
        if fr and pb.get("kind") == "full_croc":
            pl = fr.get("planner")
            costs: list[float] = []
            if pl is not None:
                cl = getattr(pl, "_cost_logger", None)
                if cl is not None and hasattr(cl, "costs") and cl.costs is not None:
                    costs = [float(c) for c in cl.costs]
            tim = fr.get("timing") or {}
            traj_meta = {
                "backend": "crocoddyl",
                "costs": costs,
                "timing": {
                    "n_iter": int(tim.get("n_iter", 0)),
                    "avg_ms_per_iter": float(tim.get("avg_ms_per_iter", 0)),
                    "total_s": float(tim.get("total_s", 0)),
                },
            }
        elif fr and pb.get("kind") == "full_acados":
            tim = fr.get("timing") or {}
            traj_meta = {
                "backend": "acados_traj",
                "costs": None,
                "timing": {
                    "n_iter": int(tim.get("n_iter", 0)),
                    "avg_ms_per_iter": float(tim.get("avg_ms_per_iter", 0)),
                    "total_s": float(tim.get("total_s", 0)),
                },
            }

        # s500 has no arm/EE: render base-only plots.
        if self._is_s500_mode():
            self._render_s500_base_only_planning_figures(
                t_rel, simX, title_prefix="Planned reference", u_plan=u
            )
        else:
            self._set_control_tab_uam_placeholder()
            # Render into the same 3 figures as the tracking GUI.
            em.render_ee_tracking_results_to_figures(
                res_ref,
                fs,
                f3,
                self.fig_traj_dash,
                control_mode="direct",
                plan_waypoints_xyz=None,
                plan_waypoint_times=tw_rel,
                plan_waypoints_base_xyz=base_wp,
                plan_waypoints_ee_xyz=ee_wp,
                states_title="Planned reference",
                traj_solver_meta=traj_meta,
            )
        self.cv_states.draw()
        self.cv_control.draw()
        self.cv_3d_track.draw()
        self.cv_traj_dash.draw()

    def _render_planning_reference_ee_snap(self) -> None:
        """Render the EE-only (minimum-snap) planning reference using the same dashboard framework."""
        import pinocchio as pin

        pb = self._plan_bundle
        assert pb is not None
        em = self._ee_mpc
        assert em is not None

        t_raw = np.asarray(pb["t_ref"], dtype=float).flatten()
        p_ref = np.asarray(pb["p_ref"], dtype=float)
        yaw_ref = np.asarray(pb["yaw_ref"], dtype=float).flatten()

        if p_ref.ndim != 2 or p_ref.shape[1] != 3:
            raise ValueError(f"p_ref must have shape (N,3); got {p_ref.shape}")
        n = len(t_raw)
        if len(p_ref) != n or len(yaw_ref) != n:
            raise ValueError("t_ref/p_ref/yaw_ref length mismatch")

        t_rel = t_raw - t_raw[0]
        yaw_ref_u = np.unwrap(yaw_ref)
        dp_ref = np.asarray(pb.get("dp_ref"), dtype=float) if pb.get("dp_ref") is not None else None
        dyaw_ref = np.asarray(pb.get("dyaw_ref"), dtype=float).flatten() if pb.get("dyaw_ref") is not None else None
        if dp_ref is None or dp_ref.ndim != 2 or dp_ref.shape[0] != n or dp_ref.shape[1] != 3:
            dp_ref = np.zeros((n, 3), dtype=float)
        if dyaw_ref is None or dyaw_ref.size != n:
            if n >= 2:
                dyaw_ref = np.gradient(yaw_ref_u, t_rel)
            else:
                dyaw_ref = np.zeros(n, dtype=float)

        # Build a plotting-friendly full-state sequence (17D convention used by dashboard):
        # [x,y,z,qx,qy,qz,qw,j1,j2,vx,vy,vz,wx,wy,wz,j1dot,j2dot]
        # For s500, j* channels stay zero. Attitude/omega are reconstructed from (ddp_ref, yaw_ref).
        ddp_explicit = pb.get("ddp_ref")
        if ddp_explicit is not None:
            ddp_ref = np.asarray(ddp_explicit, dtype=float)
            if ddp_ref.ndim != 2 or ddp_ref.shape[0] != n or ddp_ref.shape[1] != 3:
                ddp_ref = None
        else:
            ddp_ref = None
        if ddp_ref is None:
            if n >= 2:
                ddp_ref = np.gradient(dp_ref, t_rel, axis=0)
            else:
                ddp_ref = np.zeros_like(dp_ref)

        def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            nrm = float(np.linalg.norm(v))
            if nrm < 1e-9:
                return np.asarray(fallback, dtype=float).copy()
            return (np.asarray(v, dtype=float) / nrm).copy()

        R_seq: list[np.ndarray] = []
        quat_seq = np.zeros((n, 4), dtype=float)
        for i in range(n):
            a_des = np.asarray(ddp_ref[i], dtype=float) + np.array([0.0, 0.0, 9.81], dtype=float)
            b3 = _normalize(a_des, np.array([0.0, 0.0, 1.0], dtype=float))
            yaw_i = float(yaw_ref_u[i])
            b1_yaw = np.array([np.cos(yaw_i), np.sin(yaw_i), 0.0], dtype=float)
            b2 = np.cross(b3, b1_yaw)
            if np.linalg.norm(b2) < 1e-9:
                b2 = np.array([-np.sin(yaw_i), np.cos(yaw_i), 0.0], dtype=float)
            b2 = _normalize(b2, np.array([0.0, 1.0, 0.0], dtype=float))
            b1 = _normalize(np.cross(b2, b3), np.array([1.0, 0.0, 0.0], dtype=float))
            R = np.column_stack([b1, b2, b3])
            R_seq.append(R)
            q = pin.Quaternion(R)
            quat_seq[i, :] = np.array([q.x, q.y, q.z, q.w], dtype=float)

        omega_seq = np.zeros((n, 3), dtype=float)
        if n >= 2:
            for i in range(n):
                if i == 0:
                    dt = float(max(t_rel[1] - t_rel[0], 1e-9))
                    Rdot = (R_seq[1] - R_seq[0]) / dt
                elif i == n - 1:
                    dt = float(max(t_rel[-1] - t_rel[-2], 1e-9))
                    Rdot = (R_seq[-1] - R_seq[-2]) / dt
                else:
                    dt = float(max(t_rel[i + 1] - t_rel[i - 1], 1e-9))
                    Rdot = (R_seq[i + 1] - R_seq[i - 1]) / dt
                W = R_seq[i].T @ Rdot
                omega_seq[i, 0] = 0.5 * (W[2, 1] - W[1, 2])
                omega_seq[i, 1] = 0.5 * (W[0, 2] - W[2, 0])
                omega_seq[i, 2] = 0.5 * (W[1, 0] - W[0, 1])
        else:
            omega_seq[:, 2] = dyaw_ref

        simX = np.zeros((n, 17), dtype=float)
        simX[:, 0:3] = p_ref
        simX[:, 3:7] = quat_seq
        simX[:, 9:12] = dp_ref
        simX[:, 12:15] = omega_seq

        # Placeholder controls for acados-style control plots.
        u = np.zeros((max(0, n - 1), 6), dtype=float)
        err = np.zeros(n, dtype=float)
        err_yaw = np.zeros(n, dtype=float)

        res_ref = {
            "t": t_rel,
            "x": simX,
            "u": u,
            "ee": p_ref,
            "p_ref": p_ref,
            "err": err,
            "ee_yaw": yaw_ref_u.copy(),
            "yaw_ref": yaw_ref_u.copy(),
            "err_yaw": err_yaw,
            "control_mode": "direct",
            "sim_dt": float(self.sim_dt.value()),
            "control_dt": float(self.control_dt.value()),
            "mpc_stride": 1,
            "mpc_solve": {"nlp_iter": [], "cpu_s": [], "wall_s": [], "status": []},
            "waypoints": pb.get("waypoints"),
        }

        if self._is_s500_mode():
            self._render_s500_base_only_planning_figures(
                t_rel, simX, title_prefix="Planned position trajectory", u_plan=u
            )
        else:
            self._set_control_tab_uam_placeholder()
            em.render_ee_tracking_results_to_figures(
                res_ref,
                self.fig_states if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_into_figure else None,
                self.fig_3d_track if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_3d_into_figure else None,
                self.fig_traj_dash,
                control_mode="direct",
                plan_waypoints_xyz=pb.get("waypoints"),
                states_title="Planned reference (EE-only)",
            )
        self.cv_states.draw()
        self.cv_control.draw()
        self.cv_3d_track.draw()
        self.cv_traj_dash.draw()

    def _draw_suite_states_3d_combined(self, res: dict | None = None) -> None:
        """Single figure: left column time-domain states (ref dashed / real solid), right column 3D base + EE."""
        fig = self.fig_combined
        fig.clear()
        pb = self._plan_bundle
        if pb is None:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Please finish planning first",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=_mpl_pt(12),
            )
            ax.axis("off")
            return

        has_real = res is not None
        gs = fig.add_gridspec(
            4,
            2,
            width_ratios=[1.7, 1.0],
            hspace=0.36,
            wspace=0.26,
            left=0.06,
            right=0.98,
            top=0.91,
            bottom=0.07,
        )
        tinfo = {"fontsize": 9, "labelpad": 2}
        axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
        ax3d = fig.add_subplot(gs[:, 1], projection="3d")

        s500_mode = self._is_s500_mode()
        t_ref = base_ref = X_r = ee_ref = None
        if pb["kind"] in ("full_croc", "full_acados"):
            tp = np.asarray(pb["t_plan"], dtype=float).flatten()
            t_ref = tp - tp[0]
            X_r = np.asarray(pb["x_plan"], dtype=float)
            if X_r.shape[1] > 17:
                X_r = X_r[:, :17]
            base_ref = X_r[:, :3]
            if not s500_mode:
                try:
                    rm, eid = self._robot_model_and_ee()
                    data = rm.createData()
                    from s500_uam_trajectory_planner import (
                        compute_ee_kinematics_along_trajectory,
                    )

                    ee_ref, _, _, _ = compute_ee_kinematics_along_trajectory(
                        X_r, rm, data, eid
                    )
                except Exception:
                    ee_ref = None
        elif pb["kind"] == "ee_snap":
            tr = np.asarray(pb["t_ref"], dtype=float).flatten()
            t_ref = tr - tr[0]
            ee_ref = None if s500_mode else np.asarray(pb["p_ref"], dtype=float)

        t_m = X_m = base_m = ee_m = None
        if has_real:
            assert res is not None
            t_m = np.asarray(res["t"], dtype=float).flatten()
            X_m = _extract_x17(res)
            base_m = X_m[:, :3]
            ee_m = None if s500_mode else np.asarray(res["ee"], dtype=float)

        u_ref = None
        if pb.get("kind") in ("full_croc", "full_acados"):
            U = np.asarray(pb.get("u_plan"), dtype=float) if pb.get("u_plan") is not None else None
            if U is not None and U.ndim == 2 and U.shape[0] > 0:
                u_ref = U
        u_real = None
        if has_real and res is not None:
            Ur = np.asarray(res.get("u"), dtype=float) if res.get("u") is not None else None
            if Ur is not None and Ur.ndim == 2 and Ur.shape[0] > 0:
                u_real = Ur

        def _style_leg(ax):
            ax.legend(loc="upper right", fontsize=6, framealpha=0.88, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=_mpl_pt(8))

        # 0: base position
        ax = axes[0]
        if base_ref is not None:
            ax.plot(t_ref, base_ref[:, 0], "r--", lw=1.0, alpha=0.9, label="ref x")
            ax.plot(t_ref, base_ref[:, 1], "g--", lw=1.0, alpha=0.9, label="ref y")
            ax.plot(t_ref, base_ref[:, 2], "b--", lw=1.0, alpha=0.9, label="ref z")
        if base_m is not None:
            ax.plot(t_m, base_m[:, 0], "r-", lw=1.1, label="real x")
            ax.plot(t_m, base_m[:, 1], "g-", lw=1.1, label="real y")
            ax.plot(t_m, base_m[:, 2], "b-", lw=1.1, label="real z")
        ax.set_xlabel("t [s]", **tinfo)
        ax.set_ylabel("m", **tinfo)
        ax.set_title("Base position", fontsize=_mpl_pt(9))
        _style_leg(ax)

        # 1: base euler
        ax = axes[1]
        if X_r is not None:
            er = _euler_deg_from_simX(X_r)
            ax.plot(t_ref, er[:, 0], "r--", lw=1.0, alpha=0.9, label="ref roll")
            ax.plot(t_ref, er[:, 1], "g--", lw=1.0, alpha=0.9, label="ref pitch")
            ax.plot(t_ref, er[:, 2], "b--", lw=1.0, alpha=0.9, label="ref yaw")
        if X_m is not None:
            em = _euler_deg_from_simX(X_m)
            ax.plot(t_m, em[:, 0], "r-", lw=1.1, label="real roll")
            ax.plot(t_m, em[:, 1], "g-", lw=1.1, label="real pitch")
            ax.plot(t_m, em[:, 2], "b-", lw=1.1, label="real yaw")
        ax.set_xlabel("t [s]", **tinfo)
        ax.set_ylabel("deg", **tinfo)
        ax.set_title("Base orientation (Euler ZYX)", fontsize=_mpl_pt(9))
        _style_leg(ax)

        # 2: EE position (or base linear velocity in s500 mode)
        ax = axes[2]
        if s500_mode:
            if X_r is not None and X_r.shape[1] >= 12:
                ax.plot(t_ref, X_r[:, 9], "r--", lw=1.0, alpha=0.9, label="ref vx")
                ax.plot(t_ref, X_r[:, 10], "g--", lw=1.0, alpha=0.9, label="ref vy")
                ax.plot(t_ref, X_r[:, 11], "b--", lw=1.0, alpha=0.9, label="ref vz")
            if X_m is not None and X_m.shape[1] >= 12:
                ax.plot(t_m, X_m[:, 9], "r-", lw=1.1, label="real vx")
                ax.plot(t_m, X_m[:, 10], "g-", lw=1.1, label="real vy")
                ax.plot(t_m, X_m[:, 11], "b-", lw=1.1, label="real vz")
        else:
            if ee_ref is not None:
                ax.plot(t_ref, ee_ref[:, 0], "r--", lw=1.0, alpha=0.9, label="ref x")
                ax.plot(t_ref, ee_ref[:, 1], "g--", lw=1.0, alpha=0.9, label="ref y")
                ax.plot(t_ref, ee_ref[:, 2], "b--", lw=1.0, alpha=0.9, label="ref z")
            if ee_m is not None:
                ax.plot(t_m, ee_m[:, 0], "r-", lw=1.1, label="real x")
                ax.plot(t_m, ee_m[:, 1], "g-", lw=1.1, label="real y")
                ax.plot(t_m, ee_m[:, 2], "b-", lw=1.1, label="real z")
        ax.set_xlabel("t [s]", **tinfo)
        ax.set_ylabel("m/s" if s500_mode else "m", **tinfo)
        ax.set_title("Base linear velocity" if s500_mode else "EE position (FK ref / meas. real)", fontsize=_mpl_pt(9))
        _style_leg(ax)

        # 3: arm joints (or control inputs in s500 mode)
        ax = axes[3]
        if s500_mode:
            if u_ref is not None and t_ref is not None:
                tu_ref = t_ref[: u_ref.shape[0]]
                for j in range(int(u_ref.shape[1])):
                    ax.plot(tu_ref, u_ref[:, j], "--", lw=1.0, alpha=0.9, label=f"ref u{j+1}")
            if u_real is not None and t_m is not None:
                tu_real = t_m[: u_real.shape[0]]
                for j in range(int(u_real.shape[1])):
                    ax.plot(tu_real, u_real[:, j], "-", lw=1.1, label=f"real u{j+1}")
        else:
            if X_r is not None:
                ax.plot(
                    t_ref,
                    np.degrees(X_r[:, 7]),
                    "r--",
                    lw=1.0,
                    alpha=0.9,
                    label="ref j1",
                )
                ax.plot(
                    t_ref,
                    np.degrees(X_r[:, 8]),
                    "g--",
                    lw=1.0,
                    alpha=0.9,
                    label="ref j2",
                )
            if X_m is not None:
                ax.plot(t_m, np.degrees(X_m[:, 7]), "r-", lw=1.1, label="real j1")
                ax.plot(t_m, np.degrees(X_m[:, 8]), "g-", lw=1.1, label="real j2")
        ax.set_xlabel("t [s]", **tinfo)
        ax.set_ylabel("u" if s500_mode else "deg", **tinfo)
        ax.set_title("Control inputs" if s500_mode else "Arm joints", fontsize=_mpl_pt(9))
        _style_leg(ax)

        # 3D
        if base_ref is not None:
            ax3d.plot(
                base_ref[:, 0],
                base_ref[:, 1],
                base_ref[:, 2],
                color="tab:orange",
                ls="--",
                lw=1.4,
                label="base ref",
            )
        if ee_ref is not None:
            ax3d.plot(
                ee_ref[:, 0],
                ee_ref[:, 1],
                ee_ref[:, 2],
                color="brown",
                ls="--",
                lw=1.15,
                label="EE ref",
            )
        if base_m is not None:
            ax3d.plot(
                base_m[:, 0],
                base_m[:, 1],
                base_m[:, 2],
                "b-",
                lw=1.6,
                label="base real",
            )
        if ee_m is not None:
            ax3d.plot(
                ee_m[:, 0],
                ee_m[:, 1],
                ee_m[:, 2],
                "m-",
                lw=1.25,
                label="EE real",
            )
        if (not s500_mode) and pb.get("kind") == "ee_snap" and pb.get("waypoints") is not None:
            W = np.asarray(pb["waypoints"], dtype=float)
            W = W[:, :3] if W.shape[1] >= 3 else W.reshape(-1, 3)
            ax3d.scatter(
                W[:, 0],
                W[:, 1],
                W[:, 2],
                c="crimson",
                marker="*",
                s=90,
                label="EE waypoints",
            )
        ax3d.set_xlabel("X [m]", **tinfo)
        ax3d.set_ylabel("Y [m]", **tinfo)
        ax3d.set_zlabel("Z [m]", **tinfo)
        ax3d.set_title("3D: ref (dashed) · real (solid)", fontsize=_mpl_pt(10))
        ax3d.legend(loc="upper left", fontsize=6, framealpha=0.9)
        try:
            pts = []
            for arr in (base_ref, ee_ref, base_m, ee_m):
                if arr is not None and len(arr):
                    pts.append(arr)
            if pts:
                P = np.vstack(pts)
                br = float(np.ptp(P, axis=0).max())
                mid = P.mean(axis=0)
                r = max(br * 0.55, 0.25)
                ax3d.set_xlim(mid[0] - r, mid[0] + r)
                ax3d.set_ylim(mid[1] - r, mid[1] + r)
                ax3d.set_zlim(mid[2] - r, mid[2] + r)
            ax3d.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        for ax in axes:
            ax.tick_params(axis="both", labelsize=_mpl_pt(8))
        ax3d.tick_params(axis="both", labelsize=_mpl_pt(8))
        subt = "(only ref)" if not has_real else "(ref + real)"
        fig.suptitle(f"Plan ref (dashed) · closed-loop real (solid) {subt}", fontsize=_mpl_pt(12), y=0.98)

    def _make_plan_panel_scroll(self, stack: QStackedWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(stack)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return scroll

    def _schedule_plan_panel_height_refresh(self) -> None:
        """布局/显隐变更后延迟一帧再定高，避免 sizeHint 未更新导致第 2 组高度跳动。"""
        if not hasattr(self, "_plan_height_timer"):
            self._plan_height_timer = QTimer(self)
            self._plan_height_timer.setSingleShot(True)
            self._plan_height_timer.timeout.connect(self._update_plan_panel_heights)
        self._plan_height_timer.start(0)

    def _update_plan_panel_heights(self) -> None:
        idx = 0
        if hasattr(self, "plan_mode_combo"):
            idx = int(self.plan_mode_combo.currentIndex())
        path_h = int(_PLAN_PATH_SCROLL_H.get(idx, 360))
        opt_h = int(_PLAN_OPT_SCROLL_H.get(idx, 0))

        if hasattr(self, "path_scroll"):
            self.path_scroll.setFixedHeight(path_h)
        if hasattr(self, "opt_scroll"):
            show_opt = idx != 1
            self.opt_scroll.setVisible(show_opt)
            self.opt_scroll.setFixedHeight(opt_h if show_opt else 0)

        if hasattr(self, "task_group"):
            self.task_group.setMaximumHeight(16777215)
        if hasattr(self, "traj_group"):
            self.traj_group.setMaximumHeight(16777215)

    def _refresh_trajectory_setting_height(self) -> None:
        self._schedule_plan_panel_height_refresh()

    def _on_plan_mode(self):
        idx = self.plan_mode_combo.currentIndex()
        if hasattr(self, "path_stack"):
            self.path_stack.setCurrentIndex(idx)
        if hasattr(self, "opt_stack"):
            self.opt_stack.setCurrentIndex(idx)
        # Method 选择仅对 full-state 优化有意义（位置/EE 与 acc 测试使用各自的求解器）。
        if hasattr(self, "method_row_widget"):
            self.method_row_widget.setVisible(idx == 0)
        self._schedule_plan_panel_height_refresh()
        self._refresh_actions_height()

    def _refresh_task_selection_ui(self) -> None:
        """Apply current robot/trajectory selection to the planning panes."""
        if not hasattr(self, "task_robot_combo") or not hasattr(self, "task_traj_combo"):
            return
        # Startup/param-restore: keep the already-selected trajectory rather than
        # forcing the robot's first one (that reset only applies to manual robot switches).
        desired = self._current_trajectory_template_id()
        self._on_task_robot_changed(self.task_robot_combo.currentText())
        if desired and self.task_traj_combo.findData(desired) >= 0:
            i = self.task_traj_combo.findData(desired)
            if i != self.task_traj_combo.currentIndex():
                self.task_traj_combo.setCurrentIndex(i)
        self._apply_task_to_planning(emit_log=False)

    def _restore_last_session_selection(self) -> None:
        """Restore the last-used Robot mode and Trajectory template across launches.

        Matches by robot name and trajectory template id (robust to index/order
        changes); falls back silently if the file is missing or invalid.
        """
        try:
            if not LAST_SESSION_PATH.exists():
                return
            data = json.loads(LAST_SESSION_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        robot = data.get("robot_mode")
        traj_id = data.get("traj_template_id")
        if robot:
            i = self.task_robot_combo.findText(str(robot))
            if i >= 0 and i != self.task_robot_combo.currentIndex():
                self.task_robot_combo.setCurrentIndex(i)
                self._refresh_task_selection_ui()
        if traj_id:
            j = self.task_traj_combo.findData(str(traj_id))
            if j >= 0 and j != self.task_traj_combo.currentIndex():
                self.task_traj_combo.setCurrentIndex(j)

    def _save_last_session_selection(self) -> None:
        try:
            payload = {
                "robot_mode": self.task_robot_combo.currentText(),
                "traj_template_id": self._current_trajectory_template_id(),
            }
            LAST_SESSION_PATH.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_last_session_selection()
        super().closeEvent(event)

    def _is_s500_mode(self) -> bool:
        return hasattr(self, "task_robot_combo") and self.task_robot_combo.currentText() == "s500"

    def _s500_plot_sanitize_res(self, res: dict) -> dict:
        """For s500 plotting, hide EE-specific channels while keeping base/state channels."""
        out = dict(res)
        t = np.asarray(out.get("t", []), dtype=float).flatten()
        n = int(t.size)
        out["ee"] = np.full((n, 3), np.nan, dtype=float)
        out["p_ref"] = np.full((n, 3), np.nan, dtype=float)
        out["err"] = np.full(n, np.nan, dtype=float)
        out["ee_yaw"] = np.full(n, np.nan, dtype=float)
        out["yaw_ref"] = np.full(n, np.nan, dtype=float)
        out["err_yaw"] = np.full(n, np.nan, dtype=float)
        out["waypoints"] = None
        return out

    def _refresh_task_config_height(self) -> None:
        self._refresh_trajectory_setting_height()

    def _refresh_actions_height(self) -> None:
        if not hasattr(self, "plan_actions_group"):
            return
        h = int(max(110, min(280, self.plan_actions_group.sizeHint().height() + 10)))
        self.plan_actions_group.setMaximumHeight(h)

    def _template_label(self, template_id: str) -> str:
        tid = str(template_id)
        if tid in self._user_templates:
            return str(self._user_templates[tid].get("name", tid))
        return self._template_display_names.get(tid, tid)

    def _is_user_template(self, template_id: str) -> bool:
        return str(template_id) in self._user_templates

    def _template_base_kind(self, template_id: str) -> str:
        tid = str(template_id)
        ud = self._user_templates.get(tid)
        if isinstance(ud, dict):
            return str(ud.get("base_template", "minimum_snap"))
        return tid

    def _user_template_ids_for_robot(self, robot_mode: str) -> list[str]:
        out = []
        for tid, ud in self._user_templates.items():
            if not isinstance(ud, dict):
                continue
            if str(ud.get("robot", "")) == str(robot_mode):
                out.append(tid)
        out.sort(key=lambda t: str(self._user_templates[t].get("created_at", "")))
        return out

    def _current_trajectory_template_id(self) -> str:
        if not hasattr(self, "task_traj_combo"):
            return ""
        data = self.task_traj_combo.currentData()
        if data is not None and str(data).strip():
            return str(data)
        return self.task_traj_combo.currentText()

    def _populate_task_traj_combo(self, robot_mode: str, select_id: str | None = None) -> None:
        if not hasattr(self, "task_traj_combo"):
            return
        if select_id is None:
            select_id = self._current_trajectory_template_id()
        ids = list(self._task_trajectories.get(robot_mode, []))
        ids += self._user_template_ids_for_robot(robot_mode)
        self.task_traj_combo.blockSignals(True)
        self.task_traj_combo.clear()
        for tid in ids:
            self.task_traj_combo.addItem(self._template_label(tid), tid)
        if select_id and self.task_traj_combo.findData(select_id) >= 0:
            self.task_traj_combo.setCurrentIndex(self.task_traj_combo.findData(select_id))
        elif ids:
            self.task_traj_combo.setCurrentIndex(0)
        self.task_traj_combo.blockSignals(False)

    def _rename_current_trajectory_template(self) -> None:
        tid = self._current_trajectory_template_id()
        if not tid:
            QMessageBox.information(self, "Rename template", "请先选择一个 Trajectory template。")
            return
        old_label = self._template_label(tid)
        is_user = self._is_user_template(tid)
        hint = f"基础模板：{self._template_base_kind(tid)}" if is_user else f"模板类型：{tid}"
        new_label, ok = QInputDialog.getText(
            self,
            "Rename trajectory template",
            f"{hint}\n显示名称：",
            text=old_label,
        )
        if not ok:
            return
        new_label = str(new_label).strip()
        if not new_label:
            QMessageBox.warning(self, "Rename template", "显示名称不能为空。")
            return
        try:
            if is_user:
                self._user_templates[tid]["name"] = new_label
                _write_user_templates(self._user_templates)
            else:
                if new_label == tid:
                    self._template_display_names.pop(tid, None)
                else:
                    self._template_display_names[tid] = new_label
                _write_template_display_names(self._template_display_names)
        except Exception as e:
            QMessageBox.critical(self, "Rename template", f"保存名称失败：{e!r}")
            return
        robot = self.task_robot_combo.currentText() if hasattr(self, "task_robot_combo") else "s500"
        self._populate_task_traj_combo(robot, select_id=tid)
        self._refresh_saved_trajectory_combo()
        self.log(f"[template] Renamed “{tid}”: “{old_label}” → “{new_label}”")

    def _create_template_from_current(self) -> None:
        cur_id = self._current_trajectory_template_id()
        if not cur_id:
            QMessageBox.information(self, "New template", "请先选择一个基础 Trajectory template。")
            return
        base = self._template_base_kind(cur_id)
        robot = self.task_robot_combo.currentText() if hasattr(self, "task_robot_combo") else "s500"
        default_name = f"{self._template_label(cur_id)} (copy)"
        name, ok = QInputDialog.getText(
            self,
            "New trajectory template",
            f"基于当前配置新建模板\n机器人：{robot}    基础模板：{base}\n模板名称：",
            text=default_name,
        )
        if not ok:
            return
        name = str(name).strip()
        if not name:
            QMessageBox.warning(self, "New template", "模板名称不能为空。")
            return
        try:
            params = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "New template", f"读取当前配置失败：{e!r}")
            return
        tid = "user_" + uuid.uuid4().hex[:10]
        self._user_templates[tid] = {
            "id": tid,
            "name": name,
            "robot": robot,
            "base_template": base,
            "params": params,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
        try:
            _write_user_templates(self._user_templates)
        except Exception as e:
            self._user_templates.pop(tid, None)
            QMessageBox.critical(self, "New template", f"保存模板失败：{e!r}")
            return
        self._populate_task_traj_combo(robot, select_id=tid)
        self._on_task_traj_changed()
        self.log(f"[template] Created custom template “{name}” (base={base}, robot={robot})")

    def _delete_current_user_template(self) -> None:
        tid = self._current_trajectory_template_id()
        if not self._is_user_template(tid):
            QMessageBox.information(self, "Delete template", "内置 template 不可删除，仅能删除自定义 template。")
            return
        name = self._template_label(tid)
        if (
            QMessageBox.question(
                self,
                "Delete template",
                f"确定删除自定义模板 “{name}” 吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self._user_templates.pop(tid, None)
        try:
            _write_user_templates(self._user_templates)
        except Exception as e:
            QMessageBox.critical(self, "Delete template", f"保存失败：{e!r}")
            return
        robot = self.task_robot_combo.currentText() if hasattr(self, "task_robot_combo") else "s500"
        self._populate_task_traj_combo(robot)
        self._on_task_traj_changed()
        self.log(f"[template] Deleted custom template “{name}”")

    def _on_task_robot_changed(self, robot_mode: str) -> None:
        # Robot switch must invalidate cached Pinocchio/Crocoddyl model.
        self._lazy_pin_planner = None
        self.planner = None
        # 每个 robot 拥有独立的轨迹集合：切换 robot 后回到该 robot 的第一个默认轨迹。
        builtin = self._task_trajectories.get(robot_mode, [])
        first_id = builtin[0] if builtin else None
        self._populate_task_traj_combo(robot_mode, select_id=first_id)
        self._apply_robot_mode_ui_labels(robot_mode)
        self._on_task_traj_changed()
        self._sync_gazebo_launch_with_task()

    def _apply_robot_mode_ui_labels(self, robot_mode: str) -> None:
        # s500 has no arm EE-tracking semantics in UI wording.
        if robot_mode == "s500":
            self.plan_mode_combo.setItemText(1, "Position trajectory")
            if hasattr(self, "ee_type_label"):
                self.ee_type_label.setText("Trajectory type")
            if hasattr(self, "ee_wp_label"):
                self.ee_wp_label.setText("Waypoints (x,y,z m, yaw°, time s)")
            if hasattr(self, "ee_type_row_widget"):
                self.ee_type_row_widget.setVisible(False)
        else:
            self.plan_mode_combo.setItemText(1, "Manipulator/EE trajectory")
            if hasattr(self, "ee_type_label"):
                self.ee_type_label.setText("EE trajectory type")
            if hasattr(self, "ee_wp_label"):
                self.ee_wp_label.setText("EE waypoints (x,y,z m, yaw°, time s) — consistent with the EE tracking GUI")
            if hasattr(self, "ee_type_row_widget"):
                self.ee_type_row_widget.setVisible(True)

    def _on_task_traj_changed(self) -> None:
        traj_name = self._current_trajectory_template_id()
        hints = {
            "full_state_default": "全状态航点轨迹（使用 Planning 页 Full-state OCP）",
            "full_state_crocoddyl": "全状态航点轨迹（s500 + Crocoddyl）",
            "wp3_joint_opt": "三航点联合优化（acados wp3_joint_opt）",
            "minimum_snap": "最小 snap 位置轨迹",
            "figure8": "8 字位置轨迹",
            "sun_ellipse": "Sun2024 椭圆轨迹（Vmax/amax/n）",
            "circle": "圆形轨迹（Circle）",
            "csv_import": "从 CSV 导入位置/速度轨迹，并按速度上限自动时间缩放",
        }
        if self._is_user_template(traj_name):
            base = self._template_base_kind(traj_name)
            self.task_hint_label.setText(
                hints.get(base, f"自定义模板（基于 {base}）")
            )
        else:
            self.task_hint_label.setText(hints.get(str(traj_name), ""))
        # Selection should immediately drive the parameter panel below.
        self._apply_task_to_planning(emit_log=False)
        self._schedule_plan_panel_height_refresh()
        self._sync_plots_for_task_selection()

    def _sync_gazebo_launch_with_task(self) -> None:
        robot = self.task_robot_combo.currentText() if hasattr(self, "task_robot_combo") else "s500_uam"
        if not hasattr(self, "gz_launch_combo"):
            return
        if robot == "s500":
            self.gz_launch_combo.setCurrentText("s500_sitl.launch")
            if hasattr(self, "gz_model_combo"):
                self.gz_model_combo.setCurrentText("s500")
            if hasattr(self, "gz_pkg_combo") and not self.gz_pkg_combo.currentText().strip():
                self.gz_pkg_combo.setCurrentText("eagle_mpc_python")
        else:
            self.gz_launch_combo.setCurrentText("s500_uam_sitl.launch")
            if hasattr(self, "gz_model_combo"):
                self.gz_model_combo.setCurrentText("s500_uam")
            if hasattr(self, "gz_pkg_combo") and not self.gz_pkg_combo.currentText().strip():
                self.gz_pkg_combo.setCurrentText("eagle_mpc_python")

    def _apply_task_to_planning(self, emit_log: bool = True) -> None:
        robot = self.task_robot_combo.currentText()
        traj_id = self._current_trajectory_template_id()
        user_def = self._user_templates.get(traj_id)
        traj = self._template_base_kind(traj_id)
        if robot == "s500":
            if self.wp_table.rowCount() >= 1:
                self.wp_table.setItem(0, 4, QTableWidgetItem("0.0"))
                self.wp_table.setItem(0, 5, QTableWidgetItem("0.0"))
            if self.wp_table.rowCount() >= 2:
                self.wp_table.setItem(1, 4, QTableWidgetItem("0.0"))
                self.wp_table.setItem(1, 5, QTableWidgetItem("0.0"))
        if traj == "full_state_crocoddyl":
            self.plan_mode_combo.setCurrentIndex(0)
            self._restore_wp_rows(_full_wp_default_rows())
            if "crocoddyl" in getattr(self, "_method_ids", []):
                self.method_combo.setCurrentIndex(self._method_ids.index("crocoddyl"))
            elif "crocoddyl_actuator_ocp" in getattr(self, "_method_ids", []):
                self.method_combo.setCurrentIndex(self._method_ids.index("crocoddyl_actuator_ocp"))
            else:
                QMessageBox.warning(self, "Notice", "当前环境不可用 Crocoddyl，已保留 Full-state 模式。")
        elif traj == "full_state_default":
            self.plan_mode_combo.setCurrentIndex(0)
            self._restore_wp_rows(_full_wp_default_rows())
        elif traj == "wp3_joint_opt":
            self.plan_mode_combo.setCurrentIndex(0)
            if "acados_wp3_joint_opt" in getattr(self, "_method_ids", []):
                self.method_combo.setCurrentIndex(self._method_ids.index("acados_wp3_joint_opt"))
            else:
                QMessageBox.warning(self, "Notice", "当前环境不可用 acados_wp3_joint_opt，已保留 Full-state 模式。")
        elif traj == "figure8":
            self.plan_mode_combo.setCurrentIndex(1)
            self.ee_plan_type_combo.setCurrentIndex(1)
        elif traj == "sun_ellipse":
            self.plan_mode_combo.setCurrentIndex(1)
            self.ee_plan_type_combo.setCurrentIndex(2)
        elif traj == "circle":
            self.plan_mode_combo.setCurrentIndex(1)
            self.ee_plan_type_combo.setCurrentIndex(3)
        elif traj == "csv_import":
            self.plan_mode_combo.setCurrentIndex(1)
            self.ee_plan_type_combo.setCurrentIndex(4)
        elif traj == "acc_track":
            self.plan_mode_combo.setCurrentIndex(2)
        else:  # minimum_snap
            self.plan_mode_combo.setCurrentIndex(1)
            self.ee_plan_type_combo.setCurrentIndex(0)
        # User-defined templates restore their saved parameter snapshot on top of the base.
        if isinstance(user_def, dict) and isinstance(user_def.get("params"), dict):
            self._apply_plan_template_snapshot(user_def["params"])
        # Per-template saved control points (highest priority user-edited override).
        saved_cp = self._template_control_points.get(
            self._template_config_key(robot, traj_id)
        )
        if isinstance(saved_cp, dict):
            self._apply_plan_template_snapshot(saved_cp)
        self._on_plan_mode()
        self._refresh_plan_actuator_taus_enabled()
        self._schedule_plan_panel_height_refresh()
        if emit_log:
            self.log(f"Task applied: robot={robot}, trajectory={self._template_label(traj_id)}")

    def _template_config_key(self, robot: str, template_id: str) -> str:
        return f"{robot}::{template_id}"

    def _save_template_control_points(self) -> None:
        robot = self.task_robot_combo.currentText() if hasattr(self, "task_robot_combo") else "s500"
        tid = self._current_trajectory_template_id()
        if not tid:
            QMessageBox.information(self, "Save control points", "请先选择一个 Trajectory template。")
            return
        try:
            params = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "Save control points", f"读取当前配置失败：{e!r}")
            return
        # _read_wp_table already returns v2-format angle columns (j1,j2,yaw); tag the
        # snapshot as v2 so restore does not re-run the v1->v2 column swap.
        params["version"] = 2
        key = self._template_config_key(robot, tid)
        self._template_control_points[key] = params
        try:
            _write_template_control_points(self._template_control_points)
        except Exception as e:
            QMessageBox.critical(self, "Save control points", f"保存失败：{e!r}")
            return
        self.log(
            f"[template] Saved control points for {robot} / {self._template_label(tid)}."
        )

    def _apply_plan_template_snapshot(self, p: dict) -> None:
        """Apply only the planning-related widgets from a saved param snapshot."""
        if not isinstance(p, dict):
            return
        if isinstance(p.get("wp_rows"), list):
            rows = p["wp_rows"]
            if int(p.get("version", 1)) < 2:
                rows = _migrate_mixed_wp_rows_v1_to_v2(rows)
            self._restore_wp_rows(rows)
        if isinstance(p.get("ee_wp_rows"), list):
            self._set_table_from_rows(self.ee_wp_table, p["ee_wp_rows"], 5)

        plan_spins = {
            "dt_plan": self.dt_plan,
            "max_iter_plan": self.max_iter_plan,
            "state_w": self.state_w,
            "ctrl_w": self.ctrl_w,
            "wp_mult": self.wp_mult,
            "ee_knot_w": self.ee_knot_w,
            "ee_knot_state_reg_w": self.ee_knot_state_reg_w,
            "ee_knot_rot_w": self.ee_knot_rot_w,
            "ee_knot_vel_w": self.ee_knot_vel_w,
            "ee_knot_vel_pitch_w": self.ee_knot_vel_pitch_w,
            "dt_ee_sample": self.dt_ee_sample,
            "ee_eight_a": self.ee_eight_a,
            "ee_eight_period": self.ee_eight_period,
            "ee_eight_tdur": self.ee_eight_tdur,
            "ee_sun_vmax": self.ee_sun_vmax,
            "ee_sun_amax": self.ee_sun_amax,
            "ee_sun_n": self.ee_sun_n,
            "ee_sun_loops": self.ee_sun_loops,
            "ee_sun_yaw_const_deg": self.ee_sun_yaw_const,
            "ee_sun_buffer_s": self.ee_sun_buffer,
            "ee_circle_r": self.ee_circle_r,
            "ee_circle_period": self.ee_circle_period,
            "ee_circle_loops": self.ee_circle_loops,
            "ee_circle_tdur": self.ee_circle_tdur,
            "ee_circle_yaw_const_deg": self.ee_circle_yaw_const,
            "ee_circle_buffer_s": self.ee_circle_buffer,
            "ee_csv_vmax_limit": self.ee_csv_vmax_limit,
            "ee_csv_z_offset_m": self.ee_csv_z_offset,
            "ee_csv_yaw_const_deg": self.ee_csv_yaw_const,
            "plan_tau_motor": self.plan_tau_motor,
            "plan_tau_joint": self.plan_tau_joint,
            "acc_track_px": self.acc_track_px,
            "acc_track_py": self.acc_track_py,
            "acc_track_pz": self.acc_track_pz,
            "acc_track_yaw_deg": self.acc_track_yaw_deg,
            "acc_track_duration": self.acc_track_duration,
            "acc_track_step_time": self.acc_track_step_time,
            "acc_track_a_before": self.acc_track_a_before,
            "acc_track_a_after": self.acc_track_a_after,
            "acc_track_pulse_end": self.acc_track_pulse_end,
            "acc_track_sin_amp": self.acc_track_sin_amp,
            "acc_track_sin_freq": self.acc_track_sin_freq,
            "acc_track_sin_phase_deg": self.acc_track_sin_phase_deg,
            "acc_track_rotor_tau_s": self.acc_track_rotor_tau,
        }
        for key, widget in plan_spins.items():
            if key in p:
                try:
                    widget.setValue(p[key])
                except Exception:
                    pass

        for key, widgets in (
            ("ee_eight_center", (self.ee_eight_cx, self.ee_eight_cy, self.ee_eight_cz)),
            ("ee_sun_center", (self.ee_sun_cx, self.ee_sun_cy, self.ee_sun_cz)),
            ("ee_circle_center", (self.ee_circle_cx, self.ee_circle_cy, self.ee_circle_cz)),
        ):
            v = p.get(key)
            if isinstance(v, list) and len(v) >= 3:
                for w, val in zip(widgets, v):
                    w.setValue(float(val))

        for key, chk in (
            ("ee_sun_yaw_hold", self.ee_sun_yaw_hold),
            ("ee_circle_yaw_hold", self.ee_circle_yaw_hold),
            ("ee_csv_yaw_hold", self.ee_csv_yaw_hold),
            ("acc_track_brake_to_rest", self.acc_track_brake_to_rest),
            ("acc_track_rotor_dyn", self.acc_track_rotor_dyn_chk),
            ("plan_croc_use_actuator_first_order", self.plan_croc_use_actuator_first_order),
        ):
            if key in p:
                chk.setChecked(bool(p[key]))

        if "ee_csv_path" in p:
            self.ee_csv_path.setText(str(p["ee_csv_path"]))

        for key, combo in (
            ("method_index", self.method_combo),
            ("ee_plan_type_index", self.ee_plan_type_combo),
            ("ee_sun_plane_index", self.ee_sun_plane_combo),
            ("acc_track_shape_index", self.acc_track_shape_combo),
            ("acc_track_axis_index", self.acc_track_axis_combo),
        ):
            if key in p:
                idx = int(p[key])
                if 0 <= idx < combo.count():
                    combo.setCurrentIndex(idx)

        self.acc_track_rotor_tau.setEnabled(self.acc_track_rotor_dyn_chk.isChecked())
        self._on_ee_plan_type_changed()
        self._on_acc_track_shape_changed()

    def _generate_task_trajectory_now(self) -> None:
        # Do not re-apply task template on every Generate click.
        # Template application resets waypoint inputs to defaults and can wipe user edits.
        idx = int(self.plan_mode_combo.currentIndex())
        if idx == 0:
            self._run_plan()
        elif idx == 1:
            self._run_ee_plan()
        else:
            self._run_acc_track_plan()

    def _refresh_plan_actuator_taus_enabled(self) -> None:
        if not hasattr(self, "method_combo"):
            return
        mid = int(self.method_combo.currentIndex())
        method = self._method_ids[mid] if 0 <= mid < len(self._method_ids) else "none"
        is_croc = method in ("crocoddyl", "crocoddyl_actuator_ocp")
        is_ocp = method == "crocoddyl_actuator_ocp"
        use_lag = bool(is_ocp or self.plan_croc_use_actuator_first_order.isChecked())

        if hasattr(self, "plan_croc_use_actuator_first_order"):
            self.plan_croc_use_actuator_first_order.blockSignals(True)
            if is_ocp:
                self.plan_croc_use_actuator_first_order.setChecked(True)
            self.plan_croc_use_actuator_first_order.blockSignals(False)
            self.plan_croc_use_actuator_first_order.setEnabled(method == "crocoddyl")

        if hasattr(self, "plan_tau_motor"):
            self.plan_tau_motor.setEnabled(is_croc and use_lag)
        if hasattr(self, "plan_tau_joint"):
            self.plan_tau_joint.setEnabled(is_croc and use_lag)
        if hasattr(self, "wp3_group"):
            self.wp3_group.setVisible(method == "acados_wp3_joint_opt")
        self._schedule_plan_panel_height_refresh()

    def _on_ee_plan_type_changed(self):
        idx = int(self.ee_plan_type_combo.currentIndex())
        self.ee_wp_table.setVisible(idx == 0)
        self.ee_eight_group.setVisible(idx == 1)
        self.ee_sun_group.setVisible(idx == 2)
        self.ee_circle_group.setVisible(idx == 3)
        self.ee_csv_group.setVisible(idx == 4)
        self._refresh_trajectory_setting_height()

    def _on_acc_track_shape_changed(self) -> None:
        if not hasattr(self, "acc_track_step_group"):
            return
        step = int(self.acc_track_shape_combo.currentIndex()) == 0
        self.acc_track_step_group.setVisible(step)
        self.acc_track_sin_group.setVisible(not step)
        self._refresh_trajectory_setting_height()

    def _browse_ee_csv_file(self):
        default_dir = str(Path(__file__).resolve().parent / "trajectory")
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV trajectory file",
            default_dir,
            "CSV files (*.csv);;All files (*)",
        )
        if filepath:
            self.ee_csv_path.setText(filepath)

    def _plan_bundle_snapshot_meta(self, pb: dict | None = None) -> dict:
        pb = pb if pb is not None else self._plan_bundle
        robot = self.task_robot_combo.currentText() if hasattr(self, "task_robot_combo") else ""
        traj = self._current_trajectory_template_id()
        traj_label = (
            self.task_traj_combo.currentText() if hasattr(self, "task_traj_combo") else traj
        )
        kind = str(pb.get("kind", "")) if isinstance(pb, dict) else ""
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        return {
            "robot": robot,
            "trajectory_template": traj,
            "trajectory_template_label": traj_label,
            "kind": kind,
            "updated_at": now,
        }

    def _refresh_saved_trajectory_combo(self) -> None:
        if not hasattr(self, "saved_traj_combo"):
            return
        index = _load_saved_trajectory_index()
        items: dict = index.get("items") or {}
        rows = sorted(
            items.values(),
            key=lambda it: str(it.get("updated_at", it.get("created_at", ""))),
            reverse=True,
        )
        cur_id = self.saved_traj_combo.currentData()
        self.saved_traj_combo.blockSignals(True)
        self.saved_traj_combo.clear()
        for it in rows:
            tid = str(it.get("id", ""))
            if not tid:
                continue
            label = str(it.get("name", tid))
            robot = it.get("robot", "")
            tmpl = it.get("trajectory_template", "")
            tmpl_label = it.get("trajectory_template_label") or tmpl
            if robot or tmpl_label:
                label = f"{label}  ({robot}/{tmpl_label})"
            self.saved_traj_combo.addItem(label, tid)
        if cur_id is not None:
            i = self.saved_traj_combo.findData(cur_id)
            if i >= 0:
                self.saved_traj_combo.setCurrentIndex(i)
        elif index.get("last_loaded_id"):
            i = self.saved_traj_combo.findData(index.get("last_loaded_id"))
            if i >= 0:
                self.saved_traj_combo.setCurrentIndex(i)
        self.saved_traj_combo.blockSignals(False)

    def _apply_plan_bundle_to_gui(self, pb: dict, *, log_msg: str | None = None) -> bool:
        if not isinstance(pb, dict) or not pb.get("kind"):
            return False
        self._plan_bundle = pb
        self._last_track_res = None
        # Planning tab may call this before Tracking widgets (sim_dt, etc.) exist.
        if hasattr(self, "fig_states") and hasattr(self, "sim_dt"):
            self._redraw_combined_views(None)
        self._update_track_mode_enabled()
        has_plan = True
        self.run_track_btn.setEnabled(has_plan)
        self.meshcat_plan_btn.setEnabled(has_plan)
        self.meshcat_track_btn.setEnabled(False)
        _full_plan = pb.get("kind") in ("full_croc", "full_acados", "ee_snap")
        if hasattr(self, "rn_launch_btn"):
            self.rn_launch_btn.setEnabled(_full_plan)
        if log_msg:
            self.log(log_msg)
        return True

    def _persist_plan_bundle(
        self,
        pb: dict,
        *,
        traj_id: str,
        display_name: str,
        created_at: str | None = None,
    ) -> None:
        index = _load_saved_trajectory_index()
        items: dict = index.setdefault("items", {})
        prev = items.get(traj_id, {}) if isinstance(items.get(traj_id), dict) else {}
        entry = dict(prev)
        entry.update(self._plan_bundle_snapshot_meta(pb))
        entry["id"] = traj_id
        entry["name"] = display_name
        entry.setdefault("created_at", created_at or entry.get("created_at") or entry["updated_at"])
        items[traj_id] = entry
        index["last_loaded_id"] = traj_id
        _write_plan_bundle_files(traj_id, pb, entry)
        _write_saved_trajectory_index(index)
        self._refresh_saved_trajectory_combo()
        i = self.saved_traj_combo.findData(traj_id)
        if i >= 0:
            self.saved_traj_combo.setCurrentIndex(i)

    def _autosave_plan_bundle(self, pb: dict) -> None:
        if not isinstance(pb, dict) or not pb.get("kind"):
            return
        robot = self.task_robot_combo.currentText() if hasattr(self, "task_robot_combo") else "s500"
        tmpl = self._current_trajectory_template_id()
        slot_id = _plan_slot_autosave_id(robot, tmpl)
        name = self._current_trajectory_save_name()
        try:
            self._persist_plan_bundle(
                pb,
                traj_id=slot_id,
                display_name=f"{name} (auto)",
            )
        except Exception as e:
            self.log(f"[trajectory] Autosave failed: {e!r}")

    def _find_saved_plan_entry_for_selection(self) -> str | None:
        """查找与当前 robot + template 匹配的已保存轨迹（优先该槽位 autosave）。"""
        if not hasattr(self, "task_robot_combo") or not hasattr(self, "task_traj_combo"):
            return None
        robot = str(self.task_robot_combo.currentText())
        tmpl = str(self._current_trajectory_template_id())
        if not robot or not tmpl:
            return None
        slot_id = _plan_slot_autosave_id(robot, tmpl)
        if _read_plan_bundle_files(slot_id) is not None:
            return slot_id
        index = _load_saved_trajectory_index()
        best_id: str | None = None
        best_ts = ""
        for tid, entry in (index.get("items") or {}).items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("robot", "")) != robot:
                continue
            if str(entry.get("trajectory_template", "")) != tmpl:
                continue
            if str(tid) == AUTOSAVE_TRAJECTORY_ID:
                continue
            ts = str(entry.get("updated_at", entry.get("created_at", "")))
            if ts >= best_ts:
                best_ts = ts
                best_id = str(tid)
        if best_id and _read_plan_bundle_files(best_id) is not None:
            return best_id
        # 兼容旧版单一 autosave（仅当 robot/template 一致时）
        legacy = (index.get("items") or {}).get(AUTOSAVE_TRAJECTORY_ID)
        if isinstance(legacy, dict):
            if (
                str(legacy.get("robot", "")) == robot
                and str(legacy.get("trajectory_template", "")) == tmpl
                and _read_plan_bundle_files(AUTOSAVE_TRAJECTORY_ID) is not None
            ):
                return AUTOSAVE_TRAJECTORY_ID
        return None

    def _clear_plan_plots(self) -> None:
        """无已保存轨迹时清空右侧绘图与 plan bundle。"""
        if not hasattr(self, "fig_states"):
            self._plan_bundle = None
            self._full_plan_result = None
            self._last_track_res = None
            return
        self._plan_bundle = None
        self._full_plan_result = None
        self._last_track_res = None
        for fig, cv in (
            (self.fig_states, self.cv_states),
            (self.fig_control, self.cv_control),
            (self.fig_3d_track, self.cv_3d_track),
            (self.fig_traj_dash, self.cv_traj_dash),
            (self.fig_cost_analysis, self.cv_cost_analysis),
        ):
            if fig is not None:
                fig.clear()
            if cv is not None:
                cv.draw()
        self._update_track_mode_enabled()
        if hasattr(self, "run_track_btn"):
            self.run_track_btn.setEnabled(False)
        if hasattr(self, "meshcat_plan_btn"):
            self.meshcat_plan_btn.setEnabled(False)
        if hasattr(self, "meshcat_track_btn"):
            self.meshcat_track_btn.setEnabled(False)
        if hasattr(self, "rn_launch_btn"):
            self.rn_launch_btn.setEnabled(False)

    def _sync_plots_for_task_selection(self, *, log_if_loaded: bool = False) -> None:
        """切换 robot / template 后：有保存结果则显示，否则清空绘图。"""
        if not hasattr(self, "fig_states"):
            return
        tid = self._find_saved_plan_entry_for_selection()
        if tid:
            pb = _read_plan_bundle_files(tid)
            if pb is not None:
                msg = None
                if log_if_loaded:
                    index = _load_saved_trajectory_index()
                    entry = (index.get("items") or {}).get(tid, {})
                    name = entry.get("name", tid) if isinstance(entry, dict) else tid
                    msg = f"[trajectory] Restored plot data for “{name}”."
                self._apply_plan_bundle_to_gui(pb, log_msg=msg)
                if hasattr(self, "saved_traj_combo"):
                    self._refresh_saved_trajectory_combo()
                    i = self.saved_traj_combo.findData(tid)
                    if i >= 0:
                        self.saved_traj_combo.blockSignals(True)
                        self.saved_traj_combo.setCurrentIndex(i)
                        self.saved_traj_combo.blockSignals(False)
                return
        self._clear_plan_plots()

    def _save_trajectory_as_named(self) -> None:
        if self._plan_bundle is None:
            QMessageBox.information(self, "Save trajectory", "请先生成一条轨迹（Generate trajectory）。")
            return
        default = self._current_trajectory_save_name()
        name, ok = QInputDialog.getText(
            self,
            "Save trajectory",
            "轨迹名称（可含中文/空格，将显示在下拉列表中）：",
            text=default,
        )
        if not ok:
            return
        name = str(name).strip()
        if not name:
            QMessageBox.warning(self, "Save trajectory", "名称不能为空。")
            return
        traj_id = uuid.uuid4().hex[:12]
        try:
            self._persist_plan_bundle(self._plan_bundle, traj_id=traj_id, display_name=name)
            self.log(f"[trajectory] Saved as “{name}” (id={traj_id})")
        except Exception as e:
            QMessageBox.critical(self, "Save trajectory", f"保存失败：{e!r}")

    def _load_selected_saved_trajectory(self) -> None:
        traj_id = self.saved_traj_combo.currentData()
        if not traj_id:
            QMessageBox.information(self, "Load trajectory", "请先选择一条已保存轨迹。")
            return
        self._load_saved_trajectory_by_id(str(traj_id))

    def _load_saved_trajectory_by_id(self, traj_id: str) -> bool:
        pb = _read_plan_bundle_files(traj_id)
        if pb is None:
            QMessageBox.warning(self, "Load trajectory", f"无法读取轨迹文件：{traj_id}")
            return False
        index = _load_saved_trajectory_index()
        entry = (index.get("items") or {}).get(traj_id, {})
        robot = entry.get("robot") if isinstance(entry, dict) else None
        tmpl = entry.get("trajectory_template") if isinstance(entry, dict) else None
        if robot and hasattr(self, "task_robot_combo"):
            self.task_robot_combo.blockSignals(True)
            i = self.task_robot_combo.findText(str(robot))
            if i >= 0 and i != self.task_robot_combo.currentIndex():
                self.task_robot_combo.setCurrentIndex(i)
                self._lazy_pin_planner = None
                self.planner = None
            self.task_robot_combo.blockSignals(False)
            self._apply_robot_mode_ui_labels(str(robot))
            self._populate_task_traj_combo(str(robot), select_id=str(tmpl) if tmpl else None)
        if tmpl and hasattr(self, "task_traj_combo"):
            self.task_traj_combo.blockSignals(True)
            i = self.task_traj_combo.findData(str(tmpl))
            if i < 0:
                i = self.task_traj_combo.findText(str(tmpl))
            if i >= 0:
                self.task_traj_combo.setCurrentIndex(i)
            self.task_traj_combo.blockSignals(False)
        self._apply_task_to_planning(emit_log=False)
        self._schedule_plan_panel_height_refresh()
        name = entry.get("name", traj_id) if isinstance(entry, dict) else traj_id
        ok = self._apply_plan_bundle_to_gui(
            pb,
            log_msg=f"[trajectory] Loaded “{name}” from library.",
        )
        if ok:
            index["last_loaded_id"] = traj_id
            _write_saved_trajectory_index(index)
            self._refresh_saved_trajectory_combo()
            i = self.saved_traj_combo.findData(traj_id)
            if i >= 0:
                self.saved_traj_combo.setCurrentIndex(i)
        return ok

    def _try_restore_last_saved_trajectory(self) -> None:
        """启动后按当前 robot + template 恢复绘图（无则留空）。"""
        self._sync_plots_for_task_selection(log_if_loaded=True)

    def _current_trajectory_save_name(self) -> str:
        tid = self._current_trajectory_template_id() or "trajectory"
        if self._is_user_template(tid):
            name = self._template_label(tid)
        else:
            name = tid
            if hasattr(self, "task_traj_combo"):
                label = self.task_traj_combo.currentText().strip()
                if label and label != name:
                    name = f"{name}_{label}"
        if str(name) == "csv_import":
            p = self.ee_csv_path.text().strip() if hasattr(self, "ee_csv_path") else ""
            if p:
                name = Path(p).stem
            vmax = float(self.ee_csv_vmax_limit.value()) if hasattr(self, "ee_csv_vmax_limit") else 0.0
            yaw_hold = bool(self.ee_csv_yaw_hold.isChecked()) if hasattr(self, "ee_csv_yaw_hold") else False
            if yaw_hold:
                yaw_const = float(self.ee_csv_yaw_const.value()) if hasattr(self, "ee_csv_yaw_const") else 0.0
                name = f"{name}_vmax{vmax:g}_yaw_fixed_{yaw_const:g}deg"
            else:
                name = f"{name}_vmax{vmax:g}_yaw_free"
        return _safe_name_token(name)

    def _save_generated_plan_csv(self, pb: dict) -> Path | None:
        if not isinstance(pb, dict):
            return None
        out_dir = Path(__file__).resolve().parent / "tracking_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_name = self._current_trajectory_save_name()
        out = out_dir / f"{traj_name}_plan.csv"

        kind = str(pb.get("kind", ""))
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if kind in ("full_croc", "full_acados"):
                t = np.asarray(pb.get("t_plan", []), dtype=float).reshape(-1)
                x = np.asarray(pb.get("x_plan", []), dtype=float)
                u = np.asarray(pb.get("u_plan", []), dtype=float) if pb.get("u_plan") is not None else np.zeros((0, 0))
                nx = int(x.shape[1]) if x.ndim == 2 and x.size > 0 else 0
                nu = int(u.shape[1]) if u.ndim == 2 and u.size > 0 else 0
                header = ["t"] + [f"x_{i}" for i in range(nx)] + [f"u_{j}" for j in range(nu)]
                w.writerow(header)
                n = int(t.size)
                for i in range(n):
                    row = [float(t[i])]
                    if nx > 0:
                        row.extend(np.asarray(x[i], dtype=float).tolist())
                    if nu > 0:
                        if i < int(u.shape[0]):
                            row.extend(np.asarray(u[i], dtype=float).tolist())
                        else:
                            row.extend([float("nan")] * nu)
                    w.writerow(row)
            elif kind == "ee_snap":
                t = np.asarray(pb.get("t_ref", []), dtype=float).reshape(-1)
                p = np.asarray(pb.get("p_ref", []), dtype=float)
                yaw = np.asarray(pb.get("yaw_ref", []), dtype=float).reshape(-1)
                dp = np.asarray(pb.get("dp_ref"), dtype=float) if pb.get("dp_ref") is not None else None
                ddp = np.asarray(pb.get("ddp_ref"), dtype=float) if pb.get("ddp_ref") is not None else None
                dyaw = np.asarray(pb.get("dyaw_ref"), dtype=float).reshape(-1) if pb.get("dyaw_ref") is not None else None
                w.writerow(["t", "px", "py", "pz", "yaw", "vx", "vy", "vz", "ax", "ay", "az", "dyaw"])
                for i in range(int(t.size)):
                    row = [float(t[i])]
                    row.extend(np.asarray(p[i], dtype=float).tolist() if p.ndim == 2 and i < p.shape[0] else [float("nan")] * 3)
                    row.append(float(yaw[i]) if i < yaw.size else float("nan"))
                    row.extend(np.asarray(dp[i], dtype=float).tolist() if dp is not None and dp.ndim == 2 and i < dp.shape[0] else [float("nan")] * 3)
                    row.extend(np.asarray(ddp[i], dtype=float).tolist() if ddp is not None and ddp.ndim == 2 and i < ddp.shape[0] else [float("nan")] * 3)
                    row.append(float(dyaw[i]) if dyaw is not None and i < dyaw.size else float("nan"))
                    w.writerow(row)
            else:
                return None
        self.log(f"[planning] Generated trajectory saved: {out}")
        return out

    def _on_reg_mode_changed(self):
        mode = int(self.reg_mode_combo.currentIndex()) if hasattr(self, "reg_mode_combo") else 0
        full = mode == 0
        self.reg_full_state_label.setVisible(full)
        self.reg_full_state_table.setVisible(full)
        self.reg_ee_state_label.setVisible(not full)
        self.reg_ee_state_table.setVisible(not full)
        self.reg_ee_pose_label.setVisible(not full)
        self.reg_ee_pose_table.setVisible(not full)

    def _relayout_algo_grid(self, visible_widgets) -> None:
        """Pack only the visible (label, widget) pairs into a compact 2-column flow."""
        grid = getattr(self, "_algo_grid", None)
        if grid is None:
            return
        # Detach everything from the grid first (widgets keep their parent).
        while grid.count():
            item = grid.takeAt(0)
            wdg = item.widget()
            if wdg is not None:
                wdg.hide()
        i = 0
        for lb, w in getattr(self, "_algo_rows", []):
            if w not in visible_widgets:
                lb.hide()
                w.hide()
                continue
            row = i // 2
            col = (i % 2) * 2
            grid.addWidget(lb, row, col)
            grid.addWidget(w, row, col + 1)
            lb.show()
            w.show()
            i += 1

    def _on_track_mode_changed(self):
        idx = int(self.track_mode_combo.currentIndex())
        if not hasattr(self, "track_algo_group"):
            return
        visible_widgets = {self.dt_mpc}
        if idx in (0, 1):
            visible_widgets.update(
                {
                    self.w_state_track,
                    self.w_state_reg,
                    self.w_control,
                    self.w_terminal_track,
                    self.w_pos,
                    self.w_att,
                    self.w_joint,
                    self.w_vel,
                    self.w_omega,
                    self.w_joint_vel,
                    self.w_u_thrust,
                    self.w_u_joint_torque,
                }
            )
            if idx == 0:
                visible_widgets.update(
                    {self.croc_horizon, self.croc_mpc_iter}
                )
            else:
                visible_widgets.update(
                    {
                        self.N_mpc,
                        self.mpc_max_iter,
                        self.mpc_log_iv,
                        self.control_mode_track,
                    }
                )
        elif idx == 2:
            visible_widgets.update(
                {
                    self.N_mpc,
                    self.w_ee,
                    self.w_ee_yaw,
                    self.mpc_max_iter,
                    self.mpc_log_iv,
                    self.control_mode_track,
                }
            )
        else:
            visible_widgets.update(
                {
                    self.N_mpc,
                    self.croc_ee_w_pos,
                    self.croc_ee_w_rot_rp,
                    self.croc_ee_w_rot_yaw,
                    self.croc_ee_w_vel_lin,
                    self.croc_ee_w_vel_ang_rp,
                    self.croc_ee_w_vel_ang_yaw,
                    self.croc_ee_w_u,
                    self.w_state_reg,
                    self.w_state_track,
                    self.croc_ee_w_terminal,
                    self.mpc_max_iter,
                    self.croc_ee_use_thrust_constraints,
                }
            )
        self._relayout_algo_grid(visible_widgets)
        if idx == 0:
            self.track_algo_group.setTitle(
                "Algorithm parameters (Crocoddyl full-state tracking)"
            )
        elif idx == 1:
            self.track_algo_group.setTitle(
                "Algorithm parameters (Acados full-state tracking; shared cost weights with Croc)"
            )
        elif idx == 2:
            self.track_algo_group.setTitle(
                "Algorithm parameters (Acados EE-centric tracking)"
            )
        else:
            self.track_algo_group.setTitle(
                "Algorithm parameters (Crocoddyl EE pose tracking)"
            )
        self._refresh_sim_plant_controls_state()

    def _refresh_sim_plant_controls_state(self) -> None:
        """Enable/disable plant-only simulator widgets (u lag, payload) by tracking mode."""
        idx = int(self.track_mode_combo.currentIndex()) if hasattr(self, "track_mode_combo") else 0
        croc_plant_lag = idx in (0, 3)
        use_lag = (
            bool(self.croc_use_actuator_first_order.isChecked())
            if hasattr(self, "croc_use_actuator_first_order")
            else False
        )
        if hasattr(self, "_track_sim_actuator_hint"):
            self._track_sim_actuator_hint.setVisible(croc_plant_lag)
        if hasattr(self, "croc_use_actuator_first_order"):
            self.croc_use_actuator_first_order.setEnabled(croc_plant_lag)
        if hasattr(self, "tau_thrust_track"):
            self.tau_thrust_track.setEnabled(croc_plant_lag and use_lag)
        if hasattr(self, "tau_theta_track"):
            self.tau_theta_track.setEnabled(croc_plant_lag and use_lag)
        stack_idx = (
            int(self.track_sim_control_stack.currentIndex())
            if hasattr(self, "track_sim_control_stack")
            else 0
        )
        px4_on = stack_idx == 1
        if hasattr(self, "track_sim_control_stack"):
            self.track_sim_control_stack.setEnabled(croc_plant_lag)
        if hasattr(self, "_px4_gain_row"):
            self._px4_gain_row.setEnabled(croc_plant_lag and px4_on)
        if hasattr(self, "px4_rate_Kp_track"):
            self.px4_rate_Kp_track.setEnabled(croc_plant_lag and px4_on)
        if hasattr(self, "px4_rate_Kd_track"):
            self.px4_rate_Kd_track.setEnabled(croc_plant_lag and px4_on)
        croc_payload = idx in (0, 3)
        if hasattr(self, "sim_payload_enable"):
            self.sim_payload_enable.setEnabled(croc_payload)
        if hasattr(self, "sim_payload_row"):
            self.sim_payload_row.setEnabled(
                croc_payload and self.sim_payload_enable.isChecked()
            )
        if hasattr(self, "_sim_payload_label"):
            self._sim_payload_label.setEnabled(croc_payload)

    def _refresh_track_sim_actuator_taus_enabled(self) -> None:
        self._refresh_sim_plant_controls_state()

    def _on_sim_payload_enable_toggled(self, on: bool) -> None:
        self._refresh_sim_plant_controls_state()

    def _refresh_sim_payload_inertia_hint(self) -> None:
        if not hasattr(self, "sim_payload_inertia_lbl"):
            return
        try:
            from s500_uam_crocoddyl_ee_pose_tracking_mpc import solid_sphere_principal_inertias

            m = float(self.sim_payload_mass.value())
            r = 0.02
            ii, _, _ = solid_sphere_principal_inertias(m, r)
            self.sim_payload_inertia_lbl.setText(f"→ I=⅖mr² (r=2cm) ≈ {ii:.4g} kg·m²")
        except Exception:
            self.sim_payload_inertia_lbl.setText("")

    def _set_table_from_rows(self, table: QTableWidget, rows: list[list[float]], n_cols: int) -> None:
        table.setRowCount(max(0, len(rows)))
        for r, row in enumerate(rows):
            for c in range(n_cols):
                v = float(row[c]) if c < len(row) else 0.0
                table.setItem(r, c, QTableWidgetItem(f"{v:g}"))

    def _read_reg_state_table_row(self, table: QTableWidget, row: int) -> dict[str, float]:
        keys = ["x", "y", "z", "j1", "j2", "yaw"]
        out: dict[str, float] = {}
        for c, k in enumerate(keys):
            it = table.item(row, c)
            if it is None or not str(it.text()).strip():
                out[k] = 0.0
            else:
                try:
                    out[k] = float(it.text())
                except ValueError:
                    out[k] = 0.0
        return out

    def _set_reg_state_table_row(self, table: QTableWidget, row: int, d: dict) -> None:
        keys = ["x", "y", "z", "j1", "j2", "yaw"]
        for c, k in enumerate(keys):
            v = float(d.get(k, 0.0))
            table.setItem(row, c, QTableWidgetItem(f"{v:g}"))

    def _read_reg_ee_pose_table_row(self) -> dict[str, float]:
        keys = ["x", "y", "z", "yaw"]
        out: dict[str, float] = {}
        for c, k in enumerate(keys):
            it = self.reg_ee_pose_table.item(0, c)
            if it is None or not str(it.text()).strip():
                out[k] = 0.0
            else:
                try:
                    out[k] = float(it.text())
                except ValueError:
                    out[k] = 0.0
        return out

    def _reg_table_row_to_uam_state(self, table: QTableWidget, row: int) -> np.ndarray:
        from s500_uam_trajectory_planner import make_uam_state

        keys = ["x", "y", "z", "j1", "j2", "yaw"]
        vals: list[float] = []
        for c, _k in enumerate(keys):
            it = table.item(row, c)
            if it is None or not str(it.text()).strip():
                vals.append(0.0)
            else:
                try:
                    vals.append(float(it.text()))
                except ValueError:
                    vals.append(0.0)
        x, y, z, j1, j2, yaw = vals
        x_uam = np.asarray(
            make_uam_state(
                float(x),
                float(y),
                float(z),
                j1=np.deg2rad(float(j1)),
                j2=np.deg2rad(float(j2)),
                yaw=np.deg2rad(float(yaw)),
            ),
            dtype=float,
        ).flatten()
        # Keep regulation state size consistent with active robot model.
        try:
            rm, _ = self._robot_model_and_ee()
            nx = int(rm.nq + rm.nv)
            if nx > 0:
                if x_uam.size >= nx:
                    return x_uam[:nx].copy()
                out = np.zeros(nx, dtype=float)
                out[: x_uam.size] = x_uam
                return out
        except Exception:
            pass
        return x_uam

    def _collect_params(self) -> dict:
        if hasattr(self, "_rn_mpc_weight_profiles"):
            m = self.rn_controller_combo.currentText()
            if m in self._rn_mpc_weight_profiles:
                self._rn_mpc_weight_profiles[m] = self._rn_mpc_weight_snapshot()
        return {
            "version": 1,
            "task_robot_index": int(self.task_robot_combo.currentIndex()),
            "task_traj_index": int(self.task_traj_combo.currentIndex()),
            "plan_mode_index": int(self.plan_mode_combo.currentIndex()),
            "method_index": int(self.method_combo.currentIndex()),
            "track_mode_index": int(self.track_mode_combo.currentIndex()),
            "track_mode_layout_version": 2,
            "reg_mode_index": int(self.reg_mode_combo.currentIndex()),
            "control_mode_track_index": int(self.control_mode_track.currentIndex()),
            "wp_rows": [
                list(r) + [bool(self._wp_row_zero_v(i))]
                for i, r in enumerate(self._read_wp_table())
            ],
            "ee_wp_rows": self._read_ee_rows(),
            "dt_plan": float(self.dt_plan.value()),
            "max_iter_plan": int(self.max_iter_plan.value()),
            "state_w": float(self.state_w.value()),
            "ctrl_w": float(self.ctrl_w.value()),
            "wp_mult": float(self.wp_mult.value()),
            "ee_knot_w": float(self.ee_knot_w.value()),
            "ee_knot_state_reg_w": float(self.ee_knot_state_reg_w.value()),
            "ee_knot_rot_w": float(self.ee_knot_rot_w.value()),
            "ee_knot_vel_w": float(self.ee_knot_vel_w.value()),
            "ee_knot_vel_pitch_w": float(self.ee_knot_vel_pitch_w.value()),
            "dt_ee_sample": float(self.dt_ee_sample.value()),
            "ee_plan_type_index": int(self.ee_plan_type_combo.currentIndex()),
            "ee_eight_center": [
                float(self.ee_eight_cx.value()),
                float(self.ee_eight_cy.value()),
                float(self.ee_eight_cz.value()),
            ],
            "ee_eight_a": float(self.ee_eight_a.value()),
            "ee_eight_period": float(self.ee_eight_period.value()),
            "ee_eight_tdur": float(self.ee_eight_tdur.value()),
            "ee_sun_plane_index": int(self.ee_sun_plane_combo.currentIndex()),
            "ee_sun_vmax": float(self.ee_sun_vmax.value()),
            "ee_sun_amax": float(self.ee_sun_amax.value()),
            "ee_sun_n": float(self.ee_sun_n.value()),
            "ee_sun_loops": int(self.ee_sun_loops.value()),
            "ee_sun_center": [
                float(self.ee_sun_cx.value()),
                float(self.ee_sun_cy.value()),
                float(self.ee_sun_cz.value()),
            ],
            "ee_sun_yaw_const_deg": float(self.ee_sun_yaw_const.value()),
            "ee_sun_yaw_hold": bool(self.ee_sun_yaw_hold.isChecked()),
            "ee_sun_buffer_s": float(self.ee_sun_buffer.value()),
            "ee_circle_center": [
                float(self.ee_circle_cx.value()),
                float(self.ee_circle_cy.value()),
                float(self.ee_circle_cz.value()),
            ],
            "ee_circle_r": float(self.ee_circle_r.value()),
            "ee_circle_period": float(self.ee_circle_period.value()),
            "ee_circle_loops": int(self.ee_circle_loops.value()),
            "ee_circle_tdur": float(self.ee_circle_tdur.value()),
            "ee_circle_yaw_const_deg": float(self.ee_circle_yaw_const.value()),
            "ee_circle_yaw_hold": bool(self.ee_circle_yaw_hold.isChecked()),
            "ee_circle_buffer_s": float(self.ee_circle_buffer.value()),
            "ee_csv_path": self.ee_csv_path.text().strip(),
            "ee_csv_vmax_limit": float(self.ee_csv_vmax_limit.value()),
            "ee_csv_z_offset_m": float(self.ee_csv_z_offset.value()),
            "ee_csv_yaw_const_deg": float(self.ee_csv_yaw_const.value()),
            "ee_csv_yaw_hold": bool(self.ee_csv_yaw_hold.isChecked()),
            "acc_track_shape_index": int(self.acc_track_shape_combo.currentIndex()),
            "acc_track_axis_index": int(self.acc_track_axis_combo.currentIndex()),
            "acc_track_px": float(self.acc_track_px.value()),
            "acc_track_py": float(self.acc_track_py.value()),
            "acc_track_pz": float(self.acc_track_pz.value()),
            "acc_track_yaw_deg": float(self.acc_track_yaw_deg.value()),
            "acc_track_duration": float(self.acc_track_duration.value()),
            "acc_track_step_time": float(self.acc_track_step_time.value()),
            "acc_track_a_before": float(self.acc_track_a_before.value()),
            "acc_track_a_after": float(self.acc_track_a_after.value()),
            "acc_track_pulse_end": float(self.acc_track_pulse_end.value()),
            "acc_track_brake_to_rest": bool(self.acc_track_brake_to_rest.isChecked()),
            "acc_track_sin_amp": float(self.acc_track_sin_amp.value()),
            "acc_track_sin_freq": float(self.acc_track_sin_freq.value()),
            "acc_track_sin_phase_deg": float(self.acc_track_sin_phase_deg.value()),
            "acc_track_rotor_dyn": bool(self.acc_track_rotor_dyn_chk.isChecked()),
            "acc_track_rotor_tau_s": float(self.acc_track_rotor_tau.value()),
            "T_sim": float(self.T_sim.value()),
            "sim_dt": float(self.sim_dt.value()),
            "control_dt": float(self.control_dt.value()),
            "dt_mpc": float(self.dt_mpc.value()),
            "N_mpc": int(self.N_mpc.value()),
            "w_ee": float(self.w_ee.value()),
            "w_ee_yaw": float(self.w_ee_yaw.value()),
            "croc_ee_w_pos": float(self.croc_ee_w_pos.value()),
            "croc_ee_w_rot_rp": float(self.croc_ee_w_rot_rp.value()),
            "croc_ee_w_rot_yaw": float(self.croc_ee_w_rot_yaw.value()),
            "croc_ee_w_vel_lin": float(self.croc_ee_w_vel_lin.value()),
            "croc_ee_w_vel_ang_rp": float(self.croc_ee_w_vel_ang_rp.value()),
            "croc_ee_w_vel_ang_yaw": float(self.croc_ee_w_vel_ang_yaw.value()),
            "croc_ee_w_u": float(self.croc_ee_w_u.value()),
            "croc_ee_w_terminal": float(self.croc_ee_w_terminal.value()),
            "mpc_max_iter": int(self.mpc_max_iter.value()),
            "mpc_log_iv": int(self.mpc_log_iv.value()),
            "tau_thrust_track": float(self.tau_thrust_track.value()),
            "tau_theta_track": float(self.tau_theta_track.value()),
            "track_sim_control_stack_index": int(self.track_sim_control_stack.currentIndex()),
            "px4_rate_Kp_track": float(self.px4_rate_Kp_track.value()),
            "px4_rate_Kd_track": float(self.px4_rate_Kd_track.value()),
            "croc_horizon": int(self.croc_horizon.value()),
            "croc_mpc_iter": int(self.croc_mpc_iter.value()),
            "w_state_track": float(self.w_state_track.value()),
            "w_state_reg": float(self.w_state_reg.value()),
            "w_control": float(self.w_control.value()),
            "w_terminal_track": float(self.w_terminal_track.value()),
            "w_pos": float(self.w_pos.value()),
            "w_att": float(self.w_att.value()),
            "w_joint": float(self.w_joint.value()),
            "w_vel": float(self.w_vel.value()),
            "w_omega": float(self.w_omega.value()),
            "w_joint_vel": float(self.w_joint_vel.value()),
            "w_u_thrust": float(self.w_u_thrust.value()),
            "w_u_joint_torque": float(self.w_u_joint_torque.value()),
            "croc_use_actuator_first_order": bool(self.croc_use_actuator_first_order.isChecked()),
            "croc_ee_use_thrust_constraints": bool(self.croc_ee_use_thrust_constraints.isChecked()),
            "sim_payload_enable": bool(self.sim_payload_enable.isChecked()),
            "sim_payload_t_grasp": float(self.sim_payload_t_grasp.value()),
            "sim_payload_mass": float(self.sim_payload_mass.value()),
            "plan_croc_use_actuator_first_order": bool(self.plan_croc_use_actuator_first_order.isChecked()),
            "plan_tau_motor": float(self.plan_tau_motor.value()),
            "plan_tau_joint": float(self.plan_tau_joint.value()),
            "reg_full_x0": self._read_reg_state_table_row(self.reg_full_state_table, 0),
            "reg_full_xref": self._read_reg_state_table_row(self.reg_full_state_table, 1),
            "reg_ee_x0": self._read_reg_state_table_row(self.reg_ee_state_table, 0),
            "reg_ee_xref": self._read_reg_state_table_row(self.reg_ee_state_table, 1),
            "reg_ee_target_pose": self._read_reg_ee_pose_table_row(),
            # ── ROS Tracking 独立 MPC 参数 ──────────────────────────────────
            "rn_max_thrust_total": float(self.rn_max_thrust_total.value()),
            "rn_dt_mpc": float(self.rn_dt_mpc.value()),
            "rn_horizon": int(self.rn_horizon.value()),
            "rn_mpc_max_iter": int(self.rn_mpc_max_iter.value()),
            "rn_w_state_track": float(self.rn_w_state_track.value()),
            "rn_w_state_reg": float(self.rn_w_state_reg.value()),
            "rn_w_control": float(self.rn_w_control.value()),
            "rn_w_terminal_track": float(self.rn_w_terminal_track.value()),
            "rn_w_pos": float(self.rn_w_pos.value()),
            "rn_w_att": float(self.rn_w_att.value()),
            "rn_w_joint": float(self.rn_w_joint.value()),
            "rn_w_vel": float(self.rn_w_vel.value()),
            "rn_w_omega": float(self.rn_w_omega.value()),
            "rn_w_joint_vel": float(self.rn_w_joint_vel.value()),
            "rn_w_u_thrust": float(self.rn_w_u_thrust.value()),
            "rn_w_u_joint_torque": float(self.rn_w_u_joint_torque.value()),
            "rn_mpc_weight_profiles": {
                k: dict(v) for k, v in self._rn_mpc_weight_profiles.items()
            },
            "rn_acados_solver_mode": self.rn_acados_solver_mode.currentText(),
            "rn_acados_integrator": self.rn_acados_integrator.currentText(),
            "rn_acados_hpipm_mode": self.rn_acados_hpipm.currentText(),
            "rn_acados_qp_iter_max": int(self.rn_acados_qp_iter.value()),
            "rn_ee_w_pos": float(self.rn_ee_w_pos.value()),
            "rn_ee_w_rot_rp": float(self.rn_ee_w_rot_rp.value()),
            "rn_ee_w_rot_yaw": float(self.rn_ee_w_rot_yaw.value()),
            "rn_ee_w_vel_lin": float(self.rn_ee_w_vel_lin.value()),
            "rn_ee_w_vel_ang_rp": float(self.rn_ee_w_vel_ang_rp.value()),
            "rn_ee_w_vel_ang_yaw": float(self.rn_ee_w_vel_ang_yaw.value()),
            "rn_ee_w_u": float(self.rn_ee_w_u.value()),
            "rn_ee_w_terminal": float(self.rn_ee_w_terminal.value()),
            "rn_geo_kp_pos": float(self.rn_geo_kp_pos.value()),
            "rn_geo_kd_vel": float(self.rn_geo_kd_vel.value()),
            "rn_geo_kR": float(self.rn_geo_kR.value()),
            "rn_geo_kOmega": float(self.rn_geo_kOmega.value()),
            "rn_geo_max_tilt_deg": float(self.rn_geo_max_tilt_deg.value()),
            "gz_pkg": self.gz_pkg_combo.currentText().strip(),
            "gz_launch_file": self.gz_launch_combo.currentText().strip(),
            "gz_model": self.gz_model_combo.currentText().strip(),
            "gz_model_type": self.gz_model_type_combo.currentText().strip(),
            "gz_world": self.gz_world_combo.currentText().strip(),
            "gz_model_index": int(self.gz_model_combo.currentIndex()),
            "gz_model_type_index": int(self.gz_model_type_combo.currentIndex()),
            "gz_world_index": int(self.gz_world_combo.currentIndex()),
            "gz_launch_args": self.gz_args_edit.text().strip(),
        }

    def _apply_params(self, p: dict) -> None:
        if not isinstance(p, dict):
            raise ValueError("Parameter file format is invalid (root must be a JSON object).")

        if isinstance(p.get("wp_rows"), list):
            wp_rows = p["wp_rows"]
            if int(p.get("version", 1)) < 2:
                wp_rows = _migrate_mixed_wp_rows_v1_to_v2(wp_rows)
            self._restore_wp_rows(wp_rows)
        if isinstance(p.get("ee_wp_rows"), list):
            self._set_table_from_rows(self.ee_wp_table, p["ee_wp_rows"], 5)

        def _set_spin(name: str, widget):
            if name in p:
                widget.setValue(p[name])
        def _set_text(name: str, widget):
            if name in p and hasattr(widget, "setText"):
                widget.setText(str(p[name]))
        def _set_combo_text(name: str, widget):
            if name in p and hasattr(widget, "setCurrentText"):
                widget.setCurrentText(str(p[name]))

        _set_spin("dt_plan", self.dt_plan)
        _set_spin("max_iter_plan", self.max_iter_plan)
        _set_spin("state_w", self.state_w)
        _set_spin("ctrl_w", self.ctrl_w)
        _set_spin("wp_mult", self.wp_mult)
        _set_spin("ee_knot_w", self.ee_knot_w)
        _set_spin("ee_knot_state_reg_w", self.ee_knot_state_reg_w)
        _set_spin("ee_knot_rot_w", self.ee_knot_rot_w)
        _set_spin("ee_knot_vel_w", self.ee_knot_vel_w)
        _set_spin("dt_ee_sample", self.dt_ee_sample)
        _set_spin("ee_eight_a", self.ee_eight_a)
        _set_spin("ee_eight_period", self.ee_eight_period)
        _set_spin("ee_eight_tdur", self.ee_eight_tdur)
        _set_spin("ee_sun_vmax", self.ee_sun_vmax)
        _set_spin("ee_sun_amax", self.ee_sun_amax)
        _set_spin("ee_sun_n", self.ee_sun_n)
        _set_spin("ee_sun_loops", self.ee_sun_loops)
        _set_spin("ee_sun_yaw_const_deg", self.ee_sun_yaw_const)
        _set_spin("ee_sun_buffer_s", self.ee_sun_buffer)
        _set_spin("ee_circle_r", self.ee_circle_r)
        _set_spin("ee_circle_period", self.ee_circle_period)
        _set_spin("ee_circle_loops", self.ee_circle_loops)
        _set_spin("ee_circle_tdur", self.ee_circle_tdur)
        _set_spin("ee_circle_yaw_const_deg", self.ee_circle_yaw_const)
        _set_spin("ee_circle_buffer_s", self.ee_circle_buffer)
        _set_spin("ee_csv_vmax_limit", self.ee_csv_vmax_limit)
        _set_spin("ee_csv_z_offset_m", self.ee_csv_z_offset)
        _set_spin("ee_csv_yaw_const_deg", self.ee_csv_yaw_const)
        _set_text("ee_csv_path", self.ee_csv_path)
        if "ee_sun_yaw_hold" in p:
            self.ee_sun_yaw_hold.setChecked(bool(p["ee_sun_yaw_hold"]))
        if "ee_circle_yaw_hold" in p:
            self.ee_circle_yaw_hold.setChecked(bool(p["ee_circle_yaw_hold"]))
        if "ee_csv_yaw_hold" in p:
            self.ee_csv_yaw_hold.setChecked(bool(p["ee_csv_yaw_hold"]))
        _set_spin("T_sim", self.T_sim)
        _set_spin("sim_dt", self.sim_dt)
        _set_spin("control_dt", self.control_dt)
        _set_spin("dt_mpc", self.dt_mpc)
        _set_spin("N_mpc", self.N_mpc)
        _set_spin("w_ee", self.w_ee)
        _set_spin("w_ee_yaw", self.w_ee_yaw)
        _set_spin("croc_ee_w_pos", self.croc_ee_w_pos)
        _set_spin("croc_ee_w_rot_rp", self.croc_ee_w_rot_rp)
        _set_spin("croc_ee_w_rot_yaw", self.croc_ee_w_rot_yaw)
        _set_spin("croc_ee_w_vel_lin", self.croc_ee_w_vel_lin)
        _set_spin("croc_ee_w_vel_ang_rp", self.croc_ee_w_vel_ang_rp)
        _set_spin("croc_ee_w_vel_ang_yaw", self.croc_ee_w_vel_ang_yaw)
        _set_spin("croc_ee_w_u", self.croc_ee_w_u)
        _set_spin("croc_ee_w_terminal", self.croc_ee_w_terminal)
        if "sim_payload_enable" in p:
            self.sim_payload_enable.setChecked(bool(p["sim_payload_enable"]))
        _set_spin("sim_payload_t_grasp", self.sim_payload_t_grasp)
        _set_spin("sim_payload_mass", self.sim_payload_mass)
        if "sim_payload_enable" not in p and "sim_payload_t_grasp" in p:
            tg = float(p.get("sim_payload_t_grasp", -1.0))
            m = float(p.get("sim_payload_mass", 0.0))
            self.sim_payload_enable.setChecked(tg >= 0.0 and m > 1e-9)
        if hasattr(self, "sim_payload_enable"):
            self._on_sim_payload_enable_toggled(self.sim_payload_enable.isChecked())
            self._refresh_sim_payload_inertia_hint()
        _set_spin("mpc_max_iter", self.mpc_max_iter)
        _set_spin("mpc_log_iv", self.mpc_log_iv)
        _set_spin("tau_thrust_track", self.tau_thrust_track)
        _set_spin("tau_theta_track", self.tau_theta_track)
        _set_spin("px4_rate_Kp_track", self.px4_rate_Kp_track)
        _set_spin("px4_rate_Kd_track", self.px4_rate_Kd_track)
        _set_spin("croc_horizon", self.croc_horizon)
        _set_spin("croc_mpc_iter", self.croc_mpc_iter)
        _set_spin("w_state_track", self.w_state_track)
        _set_spin("w_state_reg", self.w_state_reg)
        _set_spin("w_control", self.w_control)
        _set_spin("w_terminal_track", self.w_terminal_track)
        _set_spin("w_pos", self.w_pos)
        _set_spin("w_att", self.w_att)
        _set_spin("w_joint", self.w_joint)
        _set_spin("w_vel", self.w_vel)
        _set_spin("w_omega", self.w_omega)
        _set_spin("w_joint_vel", self.w_joint_vel)
        _set_spin("w_u_thrust", self.w_u_thrust)
        _set_spin("w_u_joint_torque", self.w_u_joint_torque)
        _set_spin("plan_tau_motor", self.plan_tau_motor)
        _set_spin("plan_tau_joint", self.plan_tau_joint)
        # Backward compatibility with earlier naming.
        if "plan_tau_motor" not in p and "plan_tau_thrust" in p:
            self.plan_tau_motor.setValue(float(p["plan_tau_thrust"]))
        if "plan_tau_joint" not in p and "plan_tau_theta" in p:
            self.plan_tau_joint.setValue(float(p["plan_tau_theta"]))

        # Older parameter files only had w_ee / w_ee_yaw for EE tracking.
        if "croc_ee_w_pos" not in p and "w_ee" in p:
            self.croc_ee_w_pos.setValue(float(p["w_ee"]))
        if "croc_ee_w_rot_yaw" not in p and "w_ee_yaw" in p:
            self.croc_ee_w_rot_yaw.setValue(float(p["w_ee_yaw"]))

        # ── ROS Tracking 独立 MPC 参数 ────────────────────────────────────────
        _set_spin("rn_max_thrust_total", self.rn_max_thrust_total)
        _set_spin("rn_dt_mpc", self.rn_dt_mpc)
        _set_spin("rn_horizon", self.rn_horizon)
        _set_spin("rn_mpc_max_iter", self.rn_mpc_max_iter)
        _set_spin("rn_w_state_track", self.rn_w_state_track)
        _set_spin("rn_w_state_reg", self.rn_w_state_reg)
        _set_spin("rn_w_control", self.rn_w_control)
        _set_spin("rn_w_terminal_track", self.rn_w_terminal_track)
        _set_spin("rn_w_pos", self.rn_w_pos)
        _set_spin("rn_w_att", self.rn_w_att)
        _set_spin("rn_w_joint", self.rn_w_joint)
        _set_spin("rn_w_vel", self.rn_w_vel)
        _set_spin("rn_w_omega", self.rn_w_omega)
        _set_spin("rn_w_joint_vel", self.rn_w_joint_vel)
        _set_spin("rn_w_u_thrust", self.rn_w_u_thrust)
        _set_spin("rn_w_u_joint_torque", self.rn_w_u_joint_torque)
        _set_spin("rn_ee_w_pos", self.rn_ee_w_pos)
        _set_spin("rn_ee_w_rot_rp", self.rn_ee_w_rot_rp)
        _set_spin("rn_ee_w_rot_yaw", self.rn_ee_w_rot_yaw)
        _set_spin("rn_ee_w_vel_lin", self.rn_ee_w_vel_lin)
        _set_spin("rn_ee_w_vel_ang_rp", self.rn_ee_w_vel_ang_rp)
        _set_spin("rn_ee_w_vel_ang_yaw", self.rn_ee_w_vel_ang_yaw)
        _set_spin("rn_ee_w_u", self.rn_ee_w_u)
        _set_spin("rn_ee_w_terminal", self.rn_ee_w_terminal)
        _set_spin("rn_geo_kp_pos", self.rn_geo_kp_pos)
        _set_spin("rn_geo_kd_vel", self.rn_geo_kd_vel)
        _set_spin("rn_geo_kR", self.rn_geo_kR)
        _set_spin("rn_geo_kOmega", self.rn_geo_kOmega)
        _set_spin("rn_geo_max_tilt_deg", self.rn_geo_max_tilt_deg)
        if isinstance(p.get("rn_mpc_weight_profiles"), dict) and hasattr(
            self, "_rn_mpc_weight_profiles"
        ):
            for mk, prof in p["rn_mpc_weight_profiles"].items():
                if mk in self._rn_mpc_weight_profiles and isinstance(prof, dict):
                    self._rn_mpc_weight_profiles[mk] = dict(prof)
            self._rn_on_controller_mode_changed(self.rn_controller_combo.currentIndex())
        else:
            if "rn_acados_solver_mode" in p:
                self.rn_acados_solver_mode.setCurrentText(str(p["rn_acados_solver_mode"]))
            if "rn_acados_integrator" in p:
                self.rn_acados_integrator.setCurrentText(str(p["rn_acados_integrator"]))
            if "rn_acados_hpipm_mode" in p:
                self.rn_acados_hpipm.setCurrentText(str(p["rn_acados_hpipm_mode"]))
            _set_spin("rn_acados_qp_iter_max", self.rn_acados_qp_iter)
        _set_combo_text("gz_pkg", self.gz_pkg_combo)
        _set_combo_text("gz_launch_file", self.gz_launch_combo)
        _set_combo_text("gz_model", self.gz_model_combo)
        _set_combo_text("gz_model_type", self.gz_model_type_combo)
        _set_combo_text("gz_world", self.gz_world_combo)
        _set_text("gz_launch_args", self.gz_args_edit)

        def _set_combo(name: str, widget):
            if name in p:
                idx = int(p[name])
                if 0 <= idx < widget.count():
                    widget.setCurrentIndex(idx)

        def _set_check(name: str, widget: QCheckBox):
            if name in p:
                widget.setChecked(bool(p[name]))

        _set_combo("task_robot_index", self.task_robot_combo)
        _set_combo("task_traj_index", self.task_traj_combo)
        _set_combo("plan_mode_index", self.plan_mode_combo)
        _set_combo("ee_plan_type_index", self.ee_plan_type_combo)
        _set_combo("ee_sun_plane_index", self.ee_sun_plane_combo)
        _set_combo("method_index", self.method_combo)
        if "track_mode_index" in p:
            idx = int(p["track_mode_index"])
            if int(p.get("track_mode_layout_version", 1)) < 2:
                # v1: 0=croc full, 1=ee acados, 2=croc ee → v2 inserts acados full at index 1
                if idx == 1:
                    idx = 2
                elif idx == 2:
                    idx = 3
            if 0 <= idx < self.track_mode_combo.count():
                self.track_mode_combo.setCurrentIndex(idx)
        _set_combo("reg_mode_index", self.reg_mode_combo)
        _set_combo("control_mode_track_index", self.control_mode_track)
        _set_combo("track_sim_control_stack_index", self.track_sim_control_stack)
        _set_combo("gz_model_index", self.gz_model_combo)
        _set_combo("gz_model_type_index", self.gz_model_type_combo)
        _set_combo("gz_world_index", self.gz_world_combo)
        _set_check("croc_use_actuator_first_order", self.croc_use_actuator_first_order)
        _set_check("croc_ee_use_thrust_constraints", self.croc_ee_use_thrust_constraints)
        _set_check("plan_croc_use_actuator_first_order", self.plan_croc_use_actuator_first_order)
        if isinstance(p.get("reg_full_x0"), dict):
            self._set_reg_state_table_row(self.reg_full_state_table, 0, p["reg_full_x0"])
        if isinstance(p.get("reg_full_xref"), dict):
            self._set_reg_state_table_row(self.reg_full_state_table, 1, p["reg_full_xref"])
        if isinstance(p.get("reg_ee_x0"), dict):
            self._set_reg_state_table_row(self.reg_ee_state_table, 0, p["reg_ee_x0"])
        if isinstance(p.get("reg_ee_xref"), dict):
            self._set_reg_state_table_row(self.reg_ee_state_table, 1, p["reg_ee_xref"])
        rp = p.get("reg_ee_target_pose")
        if isinstance(rp, dict):
            keys = ["x", "y", "z", "yaw"]
            for c, k in enumerate(keys):
                if k in rp:
                    self.reg_ee_pose_table.setItem(
                        0, c, QTableWidgetItem(f"{float(rp[k]):g}")
                    )
        ec = p.get("ee_eight_center")
        if isinstance(ec, list) and len(ec) >= 3:
            self.ee_eight_cx.setValue(float(ec[0]))
            self.ee_eight_cy.setValue(float(ec[1]))
            self.ee_eight_cz.setValue(float(ec[2]))
        sc = p.get("ee_sun_center")
        if isinstance(sc, list) and len(sc) >= 3:
            self.ee_sun_cx.setValue(float(sc[0]))
            self.ee_sun_cy.setValue(float(sc[1]))
            self.ee_sun_cz.setValue(float(sc[2]))
        cc = p.get("ee_circle_center")
        if isinstance(cc, list) and len(cc) >= 3:
            self.ee_circle_cx.setValue(float(cc[0]))
            self.ee_circle_cy.setValue(float(cc[1]))
            self.ee_circle_cz.setValue(float(cc[2]))
        if hasattr(self, "acc_track_shape_combo"):
            _set_combo("acc_track_shape_index", self.acc_track_shape_combo)
            _set_combo("acc_track_axis_index", self.acc_track_axis_combo)
            for name, spin in (
                ("acc_track_px", self.acc_track_px),
                ("acc_track_py", self.acc_track_py),
                ("acc_track_pz", self.acc_track_pz),
                ("acc_track_yaw_deg", self.acc_track_yaw_deg),
                ("acc_track_duration", self.acc_track_duration),
                ("acc_track_step_time", self.acc_track_step_time),
                ("acc_track_a_before", self.acc_track_a_before),
                ("acc_track_a_after", self.acc_track_a_after),
                ("acc_track_pulse_end", self.acc_track_pulse_end),
                ("acc_track_sin_amp", self.acc_track_sin_amp),
                ("acc_track_sin_freq", self.acc_track_sin_freq),
                ("acc_track_sin_phase_deg", self.acc_track_sin_phase_deg),
                ("acc_track_rotor_tau_s", self.acc_track_rotor_tau),
            ):
                if name in p:
                    spin.setValue(float(p[name]))
            _set_check("acc_track_brake_to_rest", self.acc_track_brake_to_rest)
            _set_check("acc_track_rotor_dyn", self.acc_track_rotor_dyn_chk)
            self.acc_track_rotor_tau.setEnabled(self.acc_track_rotor_dyn_chk.isChecked())
            self._on_acc_track_shape_changed()
        self._refresh_task_selection_ui()
        self._on_ee_plan_type_changed()
        self._on_reg_mode_changed()
        self._update_track_mode_enabled()

    def _load_params_from_path(self, path: Path, silent_if_missing: bool = False) -> bool:
        if not path.exists():
            if not silent_if_missing:
                self.log(f"Parameter file not found: {path}")
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._apply_params(data)
            self._params_path = path
            self.log(f"Loaded parameters: {path}")
            return True
        except Exception as e:
            msg = f"Failed to load parameters from {path}: {e}"
            self.log(msg)
            QMessageBox.critical(self, "Error", msg[:2000])
            return False

    def _write_params_to_path(self, path: Path) -> bool:
        try:
            payload = self._collect_params()
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._params_path = path
            self.log(f"Parameters saved: {path}")
            return True
        except Exception as e:
            msg = f"Failed to save parameters to {path}: {e}"
            self.log(msg)
            QMessageBox.critical(self, "Error", msg[:2000])
            return False

    def _param_keys_for_tab(self, tab_id: str) -> set[str]:
        if tab_id == TAB_PLAN:
            return {
                "version",
                "task_robot_index",
                "task_traj_index",
                "plan_mode_index",
                "method_index",
                "wp_rows",
                "ee_wp_rows",
                "dt_plan",
                "max_iter_plan",
                "state_w",
                "ctrl_w",
                "wp_mult",
                "ee_knot_w",
                "ee_knot_state_reg_w",
                "ee_knot_rot_w",
                "ee_knot_vel_w",
                "ee_knot_vel_pitch_w",
                "dt_ee_sample",
                "ee_plan_type_index",
                "ee_eight_center",
                "ee_eight_a",
                "ee_eight_period",
                "ee_eight_tdur",
                "ee_sun_plane_index",
                "ee_sun_vmax",
                "ee_sun_amax",
                "ee_sun_n",
                "ee_sun_loops",
                "ee_sun_center",
                "ee_sun_yaw_const_deg",
                "ee_sun_yaw_hold",
                "ee_sun_buffer_s",
                "ee_circle_center",
                "ee_circle_r",
                "ee_circle_period",
                "ee_circle_loops",
                "ee_circle_tdur",
                "ee_circle_yaw_const_deg",
                "ee_circle_yaw_hold",
                "ee_circle_buffer_s",
                "ee_csv_path",
                "ee_csv_vmax_limit",
                "ee_csv_z_offset_m",
                "ee_csv_yaw_const_deg",
                "ee_csv_yaw_hold",
                "acc_track_shape_index",
                "acc_track_axis_index",
                "acc_track_px",
                "acc_track_py",
                "acc_track_pz",
                "acc_track_yaw_deg",
                "acc_track_duration",
                "acc_track_step_time",
                "acc_track_a_before",
                "acc_track_a_after",
                "acc_track_pulse_end",
                "acc_track_brake_to_rest",
                "acc_track_sin_amp",
                "acc_track_sin_freq",
                "acc_track_sin_phase_deg",
                "acc_track_rotor_dyn",
                "acc_track_rotor_tau_s",
                "plan_croc_use_actuator_first_order",
                "plan_tau_motor",
                "plan_tau_joint",
            }
        if tab_id == TAB_TRACK:
            return {
                "version",
                "track_mode_index",
                "track_mode_layout_version",
                "reg_mode_index",
                "control_mode_track_index",
                "T_sim",
                "sim_dt",
                "control_dt",
                "dt_mpc",
                "N_mpc",
                "w_ee",
                "w_ee_yaw",
                "croc_ee_w_pos",
                "croc_ee_w_rot_rp",
                "croc_ee_w_rot_yaw",
                "croc_ee_w_vel_lin",
                "croc_ee_w_vel_ang_rp",
                "croc_ee_w_vel_ang_yaw",
                "croc_ee_w_u",
                "croc_ee_w_terminal",
                "mpc_max_iter",
                "mpc_log_iv",
                "tau_thrust_track",
                "tau_theta_track",
                "track_sim_control_stack_index",
                "px4_rate_Kp_track",
                "px4_rate_Kd_track",
                "croc_horizon",
                "croc_mpc_iter",
                "w_state_track",
                "w_state_reg",
                "w_control",
                "w_terminal_track",
                "w_pos",
                "w_att",
                "w_joint",
                "w_vel",
                "w_omega",
                "w_joint_vel",
                "w_u_thrust",
                "w_u_joint_torque",
                "croc_use_actuator_first_order",
                "croc_ee_use_thrust_constraints",
                "sim_payload_enable",
                "sim_payload_t_grasp",
                "sim_payload_mass",
                "reg_full_x0",
                "reg_full_xref",
                "reg_ee_x0",
                "reg_ee_xref",
                "reg_ee_target_pose",
            }
        if tab_id == TAB_ROS:
            return {
                "version",
                "rn_max_thrust_total",
                "rn_dt_mpc",
                "rn_horizon",
                "rn_mpc_max_iter",
                "rn_w_state_track",
                "rn_w_state_reg",
                "rn_w_control",
                "rn_w_terminal_track",
                "rn_w_pos",
                "rn_w_att",
                "rn_w_joint",
                "rn_w_vel",
                "rn_w_omega",
                "rn_w_joint_vel",
                "rn_w_u_thrust",
                "rn_w_u_joint_torque",
                "rn_ee_w_pos",
                "rn_ee_w_rot_rp",
                "rn_ee_w_rot_yaw",
                "rn_ee_w_vel_lin",
                "rn_ee_w_vel_ang_rp",
                "rn_ee_w_vel_ang_yaw",
                "rn_ee_w_u",
                "rn_ee_w_terminal",
                "rn_geo_kp_pos",
                "rn_geo_kd_vel",
                "rn_geo_kR",
                "rn_geo_kOmega",
                "rn_geo_max_tilt_deg",
                "gz_pkg",
                "gz_launch_file",
                "gz_model",
                "gz_model_type",
                "gz_world",
                "gz_model_index",
                "gz_model_type_index",
                "gz_world_index",
                "gz_launch_args",
            }
        return {"version"}

    def _save_tab_params_to_path(self, tab_id: str, path: Path) -> bool:
        try:
            current = {}
            if path.exists():
                current = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(current, dict):
                    current = {}
            collected = self._collect_params()
            keys = self._param_keys_for_tab(tab_id)
            for k in keys:
                if k in collected:
                    current[k] = collected[k]
            path.write_text(json.dumps(current, indent=2), encoding="utf-8")
            self._params_path = path
            self.log(f"Saved {tab_id} parameters: {path}")
            return True
        except Exception as e:
            msg = f"Failed to save {tab_id} parameters to {path}: {e}"
            self.log(msg)
            QMessageBox.critical(self, "Error", msg[:2000])
            return False

    def _save_params(self):
        self._write_params_to_path(self._params_path)

    def _save_params_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameters as",
            str(self._params_path),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        self._write_params_to_path(Path(path))

    def _save_tab_params(self, tab_id: str):
        self._save_tab_params_to_path(tab_id, self._params_path)

    def _save_tab_params_as(self, tab_id: str):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameters as",
            str(self._params_path),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        self._save_tab_params_to_path(tab_id, Path(path))

    def _make_wp_type_combo(self, type_value: str = "Base") -> QComboBox:
        cb = QComboBox()
        # Display label + canonical value (data). EEp gets a clearer, explicit label.
        cb.addItem("Base", "Base")
        cb.addItem("EE (位姿: 位置+姿态)", "EE")
        cb.addItem("EEp (仅位置, 不约束姿态)", "EEp")
        cb.setItemData(0, "基座位姿 + 关节角", Qt.ToolTipRole)
        cb.setItemData(1, "末端执行器位置 + 姿态(roll/pitch/yaw)", Qt.ToolTipRole)
        cb.setItemData(2, "仅约束末端执行器位置；姿态不约束，角度列禁用", Qt.ToolTipRole)
        canon = _normalize_wp_type_for_combo(type_value)
        idx = cb.findData(canon)
        cb.setCurrentIndex(idx if idx >= 0 else 0)
        cb.currentIndexChanged.connect(self._refresh_wp_rows_angle_enabled)
        return cb

    def _wp_combo_value(self, cb: QComboBox) -> str:
        data = cb.currentData() if cb is not None else None
        if data:
            return str(data)
        return cb.currentText() if cb is not None else "Base"

    def _refresh_wp_rows_angle_enabled(self) -> None:
        """EEp 行不约束姿态：禁用并灰显 j1/roll, j2/pitch, yaw 三列。"""
        if not hasattr(self, "wp_table"):
            return
        from PyQt5.QtGui import QColor

        for r in range(self.wp_table.rowCount()):
            w0 = self.wp_table.cellWidget(r, 0)
            kind = _normalize_wp_type_for_combo(
                self._wp_combo_value(w0) if isinstance(w0, QComboBox) else "Base"
            )
            disable = kind == "EEp"
            for c in (4, 5, 6):
                it = self.wp_table.item(r, c)
                if it is None:
                    it = QTableWidgetItem("0")
                    self.wp_table.setItem(r, c, it)
                if disable:
                    it.setFlags(Qt.ItemIsSelectable)
                    it.setBackground(QColor(225, 225, 225))
                    it.setToolTip("EEp 仅约束位置，姿态不可输入")
                else:
                    it.setFlags(
                        Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
                    )
                    it.setBackground(QColor(255, 255, 255))
                    it.setToolTip("")

    def _make_wp_zero_v_widget(self, checked: bool = True) -> QWidget:
        """Centered checkbox container for the wp_table 'Zero v' column."""
        cont = QWidget()
        lay = QHBoxLayout(cont)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setAlignment(Qt.AlignCenter)
        cb = QCheckBox()
        cb.setChecked(bool(checked))
        cb.setToolTip(
            "勾选：在该航点约束速度=0；取消：该航点速度不被约束\n"
            "(仅在 full_state_crocoddyl 模板下生效)"
        )
        cont.setProperty("zero_v_checkbox", True)
        lay.addWidget(cb)
        cont._wp_zero_v_box = cb  # type: ignore[attr-defined]
        return cont

    def _wp_row_zero_v(self, r: int) -> bool:
        w = self.wp_table.cellWidget(r, 8) if hasattr(self, "wp_table") else None
        if w is None:
            return True
        cb = getattr(w, "_wp_zero_v_box", None)
        if isinstance(cb, QCheckBox):
            return bool(cb.isChecked())
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        return True

    def _read_wp_table(self) -> list[list]:
        rows = []
        for r in range(self.wp_table.rowCount()):
            w0 = self.wp_table.cellWidget(r, 0)
            if isinstance(w0, QComboBox):
                mode = self._wp_combo_value(w0) or "Base"
            else:
                it0 = self.wp_table.item(r, 0)
                mode = (it0.text().strip() if it0 else "Base") or "Base"
            nums = []
            for c in range(1, 8):
                it = self.wp_table.item(r, c)
                nums.append(float(it.text()) if it else 0.0)
            rows.append([mode] + nums)
        return rows

    def _read_wp_zero_v_flags(self) -> list[bool]:
        return [self._wp_row_zero_v(r) for r in range(self.wp_table.rowCount())]

    def _restore_wp_rows(self, rows: list) -> None:
        self.wp_table.setRowCount(max(0, len(rows)))
        for r, row in enumerate(rows):
            if not isinstance(row, (list, tuple)):
                continue
            zero_v = True
            if len(row) >= 9:
                mode = str(row[0])
                nums = [float(row[i]) for i in range(1, 8)]
                zero_v = bool(row[8])
            elif len(row) >= 8:
                mode = str(row[0])
                nums = [float(row[i]) for i in range(1, 8)]
            elif len(row) >= 7:
                mode = "Base"
                nums = [float(row[i]) for i in range(7)]
            else:
                mode = "Base"
                nums = [0.0] * 7
            self.wp_table.setCellWidget(r, 0, self._make_wp_type_combo(mode))
            for c, v in enumerate(nums):
                self.wp_table.setItem(r, c + 1, QTableWidgetItem(f"{v:g}"))
            self.wp_table.setCellWidget(r, 8, self._make_wp_zero_v_widget(zero_v))
        self._refresh_wp_rows_angle_enabled()

    def _add_wp_row(self):
        r = self.wp_table.rowCount()
        self.wp_table.insertRow(r)
        self.wp_table.setCellWidget(r, 0, self._make_wp_type_combo("Base"))
        for c in range(1, 8):
            self.wp_table.setItem(r, c, QTableWidgetItem("0"))
        self.wp_table.setCellWidget(r, 8, self._make_wp_zero_v_widget(True))
        self._refresh_wp_rows_angle_enabled()

    def _del_wp_row(self):
        if self.wp_table.rowCount() > 2:
            self.wp_table.removeRow(self.wp_table.rowCount() - 1)

    def _mixed_rows_to_waypoints7(self, sorted_rows: list[list]) -> list[list[float]]:
        """Acados multi-waypoint: 7 floats [x,y,z, j1°, j2°, yaw°, t] (consistent with wp_to_state)."""
        out: list[list[float]] = []
        d2r = np.pi / 180.0
        rk_fn = self._mixed_wp_row_kind
        if rk_fn is None:
            from s500_uam_trajectory_gui import mixed_wp_row_kind

            rk_fn = mixed_wp_row_kind
        import pinocchio as pin

        for row in sorted_rows:
            rk = rk_fn(row[0])
            x, y, z, a, b, c, t = (float(row[i]) for i in range(1, 8))
            if rk == "base":
                out.append([x, y, z, a, b, c, t])
            elif rk == "ee_pos" and self.planner is not None and self._make_uam_state is not None:
                st0 = self._make_uam_state(0.0, 0.0, 1.0, j1=a * d2r, j2=b * d2r, yaw=c * d2r)
                st = self.planner.align_state_ee_to_world_point(
                    st0, np.array([x, y, z], dtype=float)
                )
                out.append([float(st[0]), float(st[1]), float(st[2]), a, b, c, t])
            elif rk == "ee_pose" and self.planner is not None:
                st0 = np.zeros(17)
                st0[2] = 1.0
                rpy = np.array([a, b, c], dtype=float) * d2r
                R = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
                quat = pin.Quaternion(R)
                st0[3], st0[4], st0[5], st0[6] = quat.x, quat.y, quat.z, quat.w
                st = self.planner.align_state_ee_to_world_point(
                    st0, np.array([x, y, z], dtype=float)
                )
                rpy_s = _quat_to_euler_row(st[3:7])
                out.append(
                    [
                        float(st[0]),
                        float(st[1]),
                        float(st[2]),
                        float(np.degrees(st[7])),
                        float(np.degrees(st[8])),
                        float(np.degrees(rpy_s[2])),
                        t,
                    ]
                )
            else:
                out.append([x, y, z, a, b, c, t])
        return out

    def _read_ee_rows(self) -> list[list[float]]:
        rows = []
        for r in range(self.ee_wp_table.rowCount()):
            row = []
            for c in range(5):
                it = self.ee_wp_table.item(r, c)
                row.append(float(it.text()) if it else 0.0)
            rows.append(row)
        return rows

    def _run_plan(self):
        if int(self.plan_mode_combo.currentIndex()) != 0:
            return
        if self.OptimizationWorker is None or self._wp_to_state is None:
            QMessageBox.warning(self, "Error", "Unable to import trajectory_gui / solver.")
            return
        mid = self.method_combo.currentIndex()
        method = self._method_ids[mid] if mid < len(self._method_ids) else "none"
        if method == "acados_wp3_joint_opt":
            params = {
                "dt": self.dt_plan.value(),
                "max_iter": self.max_iter_plan.value(),
                "state_weight": self.state_w.value(),
                "control_weight": self.ctrl_w.value(),
                "waypoint_multiplier": self.wp_mult.value(),
                "wp3_config": {
                    "ocp_mode": self.wp3_mode_combo.currentText(),
                    "dt": self.dt_plan.value(),
                    "total_time": self.wp3_total_time.value(),
                    "grasp_time": self.wp3_grasp_time.value(),
                    "grasp_ee_pos": np.array([self.wp3_gx.value(), self.wp3_gy.value(), self.wp3_gz.value()], dtype=float),
                    "grasp_ee_euler_deg": np.array([self.wp3_gr.value(), self.wp3_gp.value(), self.wp3_gyaw.value()], dtype=float),
                    "grasp_ee_vel": np.zeros(3, dtype=float),
                    "pos_err_gain": np.array([self.wp3_kx.value(), self.wp3_ky.value(), self.wp3_kz.value()], dtype=float),
                    "grasp_pos_err_max": np.array([self.wp3_ex.value(), self.wp3_ey.value(), self.wp3_ez.value()], dtype=float),
                    "state_weight": self.state_w.value(),
                    "control_weight": self.ctrl_w.value(),
                    "terminal_scale": self.wp_mult.value(),
                    "max_iter": self.max_iter_plan.value(),
                    "wp0": np.array([self.wp3_w0x.value(), self.wp3_w0y.value(), self.wp3_w0z.value(), self.wp3_w0j1.value(), self.wp3_w0j2.value(), self.wp3_w0yaw.value()], dtype=float),
                    "wp2": np.array([self.wp3_w2x.value(), self.wp3_w2y.value(), self.wp3_w2z.value(), self.wp3_w2j1.value(), self.wp3_w2j2.value(), self.wp3_w2yaw.value()], dtype=float),
                },
            }
            self.task_generate_btn.setEnabled(False)
            self.log("Planning started: acados_wp3_joint_opt")
            self._plan_worker = self.OptimizationWorker("acados_wp3_joint_opt", params)
            self._plan_worker.finished.connect(self._on_plan_finished)
            self._plan_worker.start()
            return
        rows = self._read_wp_table()
        zero_v_flags_raw = self._read_wp_zero_v_flags()
        if len(zero_v_flags_raw) != len(rows):
            zero_v_flags_raw = [True] * len(rows)
        rows_with_flag = [list(r) + [bool(zero_v_flags_raw[i])] for i, r in enumerate(rows)]
        sorted_rows_full = sorted(rows_with_flag, key=lambda x: float(x[7]))
        sorted_rows = [r[:8] for r in sorted_rows_full]
        zero_v_flags = [bool(r[8]) for r in sorted_rows_full]
        if len(sorted_rows) < 2:
            QMessageBox.warning(self, "Error", "At least 2 waypoints are required.")
            return
        self._last_plan_sorted_wp_rows = copy.deepcopy(sorted_rows)
        self._last_plan_zero_v_flags = list(zero_v_flags)
        durs = []
        for i in range(len(sorted_rows) - 1):
            d = float(sorted_rows[i + 1][7]) - float(sorted_rows[i][7])
            durs.append(d if d > 1e-6 else 1.0)
        if method == "none":
            QMessageBox.warning(self, "Error", "No available solver.")
            return
        if method in ("crocoddyl", "crocoddyl_actuator_ocp") and self.planner is None:
            self._init_croc_planner()
            if self.planner is None:
                QMessageBox.warning(self, "Error", "Crocoddyl planner is not initialized.")
                return
        worker_method = "crocoddyl" if method == "crocoddyl_actuator_ocp" else method
        use_actuator_first_order = bool(
            method == "crocoddyl_actuator_ocp"
            or (method == "crocoddyl" and self.plan_croc_use_actuator_first_order.isChecked())
        )
        wps7 = self._mixed_rows_to_waypoints7(sorted_rows)
        params = {
            "mixed_wp_rows": sorted_rows,
            "zero_velocity_flags": list(zero_v_flags),
            "waypoints": wps7,
            "durations": durs,
            "dt": self.dt_plan.value(),
            "max_iter": self.max_iter_plan.value(),
            "state_weight": self.state_w.value(),
            "control_weight": self.ctrl_w.value(),
            "waypoint_multiplier": self.wp_mult.value(),
            "ee_knot_weight": self.ee_knot_w.value(),
            "ee_knot_state_reg_weight": self.ee_knot_state_reg_w.value(),
            "ee_knot_rotation_weight": self.ee_knot_rot_w.value(),
            "ee_knot_velocity_weight": self.ee_knot_vel_w.value(),
            "ee_knot_velocity_pitch_weight": self.ee_knot_vel_pitch_w.value(),
            "planner": self.planner,
            "tau_cmd": np.array(
                [
                    self.plan_tau_motor.value(),
                    self.plan_tau_motor.value(),
                    self.plan_tau_motor.value(),
                    self.plan_tau_motor.value(),
                    self.plan_tau_joint.value(),
                    self.plan_tau_joint.value(),
                ],
                dtype=float,
            ),
            "use_actuator_first_order": use_actuator_first_order,
        }
        self.task_generate_btn.setEnabled(False)
        self.log(f"Planning started: {method}, {len(sorted_rows)} waypoints")
        self._plan_worker = self.OptimizationWorker(worker_method, params)
        self._plan_worker.finished.connect(self._on_plan_finished)
        self._plan_worker.start()

    def _start_meshcat_playback(
        self, X: np.ndarray, dt: float, traj_points: dict[str, np.ndarray] | None = None
    ):
        if self._meshcat_worker is not None and self._meshcat_worker.isRunning():
            # Browser-close events are not always detectable across backends;
            # proactively stop any previous playback worker before starting a new one.
            self._meshcat_worker.requestInterruption()
            if not self._meshcat_worker.wait(1200):
                self._meshcat_worker.terminate()
                self._meshcat_worker.wait(500)
            self.log("Stopped previous Meshcat playback worker.")
        urdf_path = None
        if self.planner is not None and getattr(self.planner, "urdf_path", None):
            urdf_path = str(self.planner.urdf_path)
        if urdf_path is None:
            urdf_path = self._selected_robot_urdf_path()
        self._meshcat_worker = MeshcatPlaybackWorker(urdf_path, X, dt, traj_points=traj_points)
        self._meshcat_worker.finished.connect(self._on_meshcat_finished)
        self._meshcat_worker.start()
        self.log("Started Meshcat playback...")

    def _on_meshcat_finished(self, ok: bool, err: str):
        if not ok:
            self.log(err)
            QMessageBox.critical(self, "Meshcat error", err[:2000])
            return
        if err:
            self.log(err)
        self.log("Meshcat playback finished.")

    def _visualize_planned_meshcat(self):
        pb = self._plan_bundle
        if pb is None:
            QMessageBox.warning(self, "Notice", "Please run planning first.")
            return
        if pb["kind"] in ("full_croc", "full_acados"):
            X = np.asarray(pb["x_plan"], dtype=float)
            t = np.asarray(pb["t_plan"], dtype=float).flatten()
        elif pb["kind"] == "ee_snap":
            try:
                ep = self._full_state_ros_plan_from_ee_snap(pb)
                X = np.asarray(ep["x_plan"], dtype=float)
                t = np.asarray(ep["t_plan"], dtype=float).flatten()
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Notice",
                    f"无法从 EE 参考构造全身状态用于 Meshcat：{e}",
                )
                return
        else:
            QMessageBox.warning(self, "Notice", "No plannable trajectory available.")
            return
        if X.ndim != 2 or X.shape[0] < 2:
            QMessageBox.warning(self, "Notice", "Planned state trajectory is empty.")
            return
        dt = float(np.median(np.diff(t))) if t.size >= 2 else float(self.dt_plan.value())
        traj = {"base": np.asarray(X[:, :3], dtype=float)}
        if pb.get("kind") == "full_croc":
            try:
                rm, eid = self._robot_model_and_ee()
                data = rm.createData()
                from s500_uam_trajectory_planner import compute_ee_kinematics_along_trajectory

                ee, _, _, _ = compute_ee_kinematics_along_trajectory(X, rm, data, eid)
                traj["ee"] = np.asarray(ee, dtype=float)
            except Exception:
                pass
        self._start_meshcat_playback(X, dt, traj_points=traj)

    def _visualize_tracked_meshcat(self):
        if self._last_track_res is None:
            QMessageBox.warning(self, "Notice", "Please run closed-loop tracking first.")
            return
        X = _extract_x17(self._last_track_res)
        t = np.asarray(self._last_track_res.get("t", []), dtype=float).flatten()
        if X.ndim != 2 or X.shape[0] < 2:
            QMessageBox.warning(self, "Notice", "Tracked state trajectory is empty.")
            return
        dt = float(np.median(np.diff(t))) if t.size >= 2 else float(self.sim_dt.value())
        traj = {"base": np.asarray(X[:, :3], dtype=float)}
        try:
            traj["ee"] = np.asarray(self._last_track_res.get("ee"), dtype=float)
            traj["ref"] = np.asarray(self._last_track_res.get("p_ref"), dtype=float)
        except Exception:
            pass
        self._start_meshcat_playback(X, dt, traj_points=traj)

    def _on_plan_finished(self, ok: bool, err: str, result_data: object):
        self.task_generate_btn.setEnabled(True)
        if not ok:
            self.log("Planning failed:\n" + err)
            QMessageBox.critical(self, "Planning failed", err[:2000])
            return
        assert isinstance(result_data, dict)
        self._full_plan_result = result_data
        dt = float(self.dt_plan.value())

        if result_data.get("method") == "crocoddyl":
            pl = result_data.get("planner")
            cache = result_data.get("_plot_cache") or getattr(pl, "_plot_cache", None)
            if cache is not None and cache.get("xs") is not None:
                xs = [np.array(x, dtype=float).flatten() for x in cache["xs"]]
                us = [
                    np.array(u, dtype=float).flatten()
                    for u in (cache.get("us") or [])
                ]
            else:
                xs = [np.array(x, dtype=float).flatten() for x in pl.solver.xs]
                us = [np.array(u, dtype=float).flatten() for u in pl.solver.us]
            t_plan = np.arange(len(xs), dtype=float) * dt
            x_plan = np.vstack(xs)
            _wpr = getattr(self, "_last_plan_sorted_wp_rows", None)
            self._plan_bundle = {
                "kind": "full_croc",
                "t_plan": t_plan,
                "x_plan": x_plan,
                "u_plan": np.vstack(us) if len(us) else np.zeros((0, 6), dtype=float),
                # Crocoddyl full-state convention: linear/angular velocities are body-frame.
                "velocity_frame": "body",
                "plan_mixed_wp_rows": copy.deepcopy(_wpr) if _wpr else None,
            }
        elif result_data.get("method") in ("acados", "acados_cascade", "acados_wp3_joint_opt"):
            t_plan = np.asarray(result_data["time_arr"], dtype=float).flatten()
            x_plan = np.asarray(result_data["simX"], dtype=float)
            u_plan = np.asarray(result_data.get("simU"), dtype=float)
            _wpr = getattr(self, "_last_plan_sorted_wp_rows", None)
            self._plan_bundle = {
                "kind": "full_acados",
                "t_plan": t_plan,
                "x_plan": x_plan,
                "u_plan": u_plan if u_plan.ndim == 2 else np.zeros((0, 6), dtype=float),
                # Acados full-state export in this project also follows body-frame velocity convention.
                "velocity_frame": "body",
                "plan_mixed_wp_rows": copy.deepcopy(_wpr) if _wpr else None,
            }
        else:
            self._plan_bundle = None

        cache = None
        if result_data.get("method") == "crocoddyl" and result_data.get("planner"):
            pl = result_data["planner"]
            if getattr(pl, "_plot_cache", None) is not None:
                cache = pl._plot_cache
        self._last_track_res = None
        self._redraw_combined_views(None)
        self._update_track_mode_enabled()
        self.run_track_btn.setEnabled(self._plan_bundle is not None)
        self.meshcat_plan_btn.setEnabled(self._plan_bundle is not None)
        self.meshcat_track_btn.setEnabled(False)
        try:
            if self._plan_bundle is not None:
                self._save_generated_plan_csv(self._plan_bundle)
                self._autosave_plan_bundle(self._plan_bundle)
        except Exception as e:
            self.log(f"[planning] Failed to save generated CSV: {e!r}")
        _full_plan = self._plan_bundle is not None and (
            self._plan_bundle["kind"] in ("full_croc", "full_acados", "ee_snap")
        )
        self.rn_launch_btn.setEnabled(_full_plan)
        self.log("Planning finished. You can run the closed loop on the \"Tracking\" tab.")

    def _run_ee_plan(self):
        self.task_generate_btn.setEnabled(False)
        self.log("Generating EE reference…")
        pt = int(self.ee_plan_type_combo.currentIndex())
        if pt == 0:
            params = {
                "mode": "snap",
                "rows": self._read_ee_rows(),
                "dt_sample": self.dt_ee_sample.value(),
            }
        elif pt == 1:
            params = {
                "mode": "eight",
                "dt_sample": self.dt_ee_sample.value(),
                "eight_center": [
                    self.ee_eight_cx.value(),
                    self.ee_eight_cy.value(),
                    self.ee_eight_cz.value(),
                ],
                "eight_a": self.ee_eight_a.value(),
                "eight_period": self.ee_eight_period.value(),
                "t_duration": self.ee_eight_tdur.value(),
            }
        elif pt == 2:
            params = {
                "mode": "sun_ellipse",
                "dt_sample": self.dt_ee_sample.value(),
                "plane": self.ee_sun_plane_combo.currentText(),
                "vmax": self.ee_sun_vmax.value(),
                "amax": self.ee_sun_amax.value(),
                "ellipticity": self.ee_sun_n.value(),
                "loops": self.ee_sun_loops.value(),
                "center": [
                    self.ee_sun_cx.value(),
                    self.ee_sun_cy.value(),
                    self.ee_sun_cz.value(),
                ],
                "yaw_const_deg": self.ee_sun_yaw_const.value(),
                "yaw_hold": self.ee_sun_yaw_hold.isChecked(),
                "buffer_s": self.ee_sun_buffer.value(),
            }
        elif pt == 3:
            params = {
                "mode": "circle",
                "dt_sample": self.dt_ee_sample.value(),
                "center": [
                    self.ee_circle_cx.value(),
                    self.ee_circle_cy.value(),
                    self.ee_circle_cz.value(),
                ],
                "radius": self.ee_circle_r.value(),
                "period": self.ee_circle_period.value(),
                "loops": self.ee_circle_loops.value(),
                "duration": self.ee_circle_period.value() * self.ee_circle_loops.value(),
                "yaw_const_deg": self.ee_circle_yaw_const.value(),
                "yaw_hold": self.ee_circle_yaw_hold.isChecked(),
                "buffer_s": self.ee_circle_buffer.value(),
            }
        else:
            params = {
                "mode": "csv_import",
                "csv_path": self.ee_csv_path.text().strip(),
                "vmax_limit": self.ee_csv_vmax_limit.value(),
                "z_offset_m": self.ee_csv_z_offset.value(),
                "yaw_const_deg": self.ee_csv_yaw_const.value(),
                "yaw_hold": self.ee_csv_yaw_hold.isChecked(),
            }
        self._plan_worker = EeRefPlanWorker(params)
        self._plan_worker.finished.connect(self._on_ee_plan_finished)
        self._plan_worker.start()

    def _run_acc_track_plan(self) -> None:
        """Build world-frame acceleration test reference; integrate to p,v for EE tracking."""
        dt = float(self.dt_plan.value())
        T = float(self.acc_track_duration.value())
        if dt <= 1e-9 or T <= 1e-9:
            QMessageBox.warning(self, "Error", "Sampling dt and duration must be positive.")
            return
        shape_idx = int(self.acc_track_shape_combo.currentIndex())
        t_step = float(self.acc_track_step_time.value())
        merge_list = [0.0, T, t_step]
        t1_brake: float | None = None
        if shape_idx == 0 and bool(self.acc_track_brake_to_rest.isChecked()):
            t1 = float(self.acc_track_pulse_end.value())
            t1 = max(t_step + 1e-6, min(t1, T - 1e-3))
            if T - t1 < max(0.5 * dt, 5e-4):
                QMessageBox.warning(
                    self,
                    "Error",
                    "制动段太短：请增大总时长，或减小「脉冲结束 t₁」，使 T−t₁ 明显大于采样 dt。",
                )
                return
            merge_list.append(t1)
            t1_brake = t1
        base_grid = np.arange(0.0, T + 0.5 * dt, dt, dtype=float)
        t_nodes = np.unique(np.concatenate([base_grid, np.asarray(merge_list, dtype=float)]))
        t_nodes = np.sort(t_nodes[(t_nodes >= -1e-12) & (t_nodes <= T + 1e-12)])
        if t_nodes.size < 2:
            t_nodes = np.array([0.0, max(T, dt)], dtype=float)
        if float(t_nodes[-1]) < T - 1e-9:
            t_nodes = np.unique(np.append(t_nodes, T))
        axis = int(self.acc_track_axis_combo.currentIndex())
        axis = max(0, min(2, axis))
        p0 = np.array(
            [
                float(self.acc_track_px.value()),
                float(self.acc_track_py.value()),
                float(self.acc_track_pz.value()),
            ],
            dtype=float,
        )
        yaw0 = float(np.deg2rad(self.acc_track_yaw_deg.value()))
        n = int(t_nodes.size)
        a_w = np.zeros((n, 3), dtype=float)
        if shape_idx == 0:
            a_lo = float(self.acc_track_a_before.value())
            a_hi = float(self.acc_track_a_after.value())
            brake_on = bool(self.acc_track_brake_to_rest.isChecked())
            if brake_on and t1_brake is not None:
                t1 = float(t1_brake)
                v_pulse = 0.0
                for j in range(n - 1):
                    ta = float(t_nodes[j])
                    tb = float(t_nodes[j + 1])
                    if tb <= t1 + 1e-12:
                        mid = 0.5 * (ta + tb)
                        ak = a_lo if mid < t_step else a_hi
                        v_pulse += ak * (tb - ta)
                    else:
                        break
                a_brake = -v_pulse / max(float(T - t1), 1e-9)
                if abs(a_brake) > 80.0:
                    self.log(
                        f"[planning] Acc track step: |a_brake|={abs(a_brake):.2f} m/s² 较大，"
                        "可延长制动段 (增大 T−t₁) 或减小阶跃加速度。"
                    )
                for j in range(n - 1):
                    ta = float(t_nodes[j])
                    tb = float(t_nodes[j + 1])
                    mid = 0.5 * (ta + tb)
                    if mid < t_step:
                        a_w[j, axis] = a_lo
                    elif mid < t1:
                        a_w[j, axis] = a_hi
                    else:
                        a_w[j, axis] = a_brake
            else:
                for j in range(n - 1):
                    ta = float(t_nodes[j])
                    tb = float(t_nodes[j + 1])
                    mid = 0.5 * (ta + tb)
                    a_w[j, axis] = a_lo if mid < t_step else a_hi
        else:
            amp = float(self.acc_track_sin_amp.value())
            f_hz = float(self.acc_track_sin_freq.value())
            phi = float(np.deg2rad(self.acc_track_sin_phase_deg.value()))
            w = 2.0 * np.pi * f_hz
            a_w[:, axis] = amp * np.sin(w * t_nodes + phi)

        if bool(self.acc_track_rotor_dyn_chk.isChecked()):
            tau_act = float(self.acc_track_rotor_tau.value())
            if tau_act > 1e-9:
                a_cmd_w = np.asarray(a_w, dtype=float).copy()
                a_w = _first_order_accel_response_piecewise(t_nodes, a_cmd_w, tau_act)
                self.log(
                    f"[planning] acc_track: 已应用旋翼/推力一阶模型 τ={tau_act:.4f} s（τ·da/dt+a=a_cmd，a(0)=0），再积分 p,v。"
                )

        if n >= 2:
            a_w[n - 1] = a_w[n - 2]

        v_w = np.zeros_like(a_w)
        p_w = np.zeros_like(a_w)
        p_w[0] = p0
        for i in range(1, n):
            dti = float(t_nodes[i] - t_nodes[i - 1])
            if dti <= 0.0:
                continue
            v_w[i] = v_w[i - 1] + a_w[i - 1] * dti
            p_w[i] = p_w[i - 1] + v_w[i - 1] * dti

        yaw_ref = np.full(n, yaw0, dtype=float)
        dyaw_ref = np.zeros(n, dtype=float)
        self._full_plan_result = None
        self._plan_bundle = {
            "kind": "ee_snap",
            "ee_track_kind": "acc_track",
            "t_ref": t_nodes,
            "p_ref": p_w,
            "yaw_ref": yaw_ref,
            "dp_ref": v_w,
            "ddp_ref": a_w,
            "dyaw_ref": dyaw_ref,
            "waypoints": None,
            "t_wp": None,
        }
        self._last_track_res = None
        self._redraw_combined_views(None)
        self._update_track_mode_enabled()
        self.run_track_btn.setEnabled(True)
        self.meshcat_plan_btn.setEnabled(True)
        self.meshcat_track_btn.setEnabled(False)
        try:
            self._save_generated_plan_csv(self._plan_bundle)
            self._autosave_plan_bundle(self._plan_bundle)
        except Exception as e:
            self.log(f"[planning] Failed to save generated CSV: {e!r}")
        self.rn_launch_btn.setEnabled(True)
        msg = (
            "[planning] Acc tracking reference generated (world a → integrated p,v). "
            "建议使用 Tracking 页的 Acados EE-centric 或 Crocoddyl EE pose。"
        )
        if shape_idx == 0 and bool(self.acc_track_brake_to_rest.isChecked()):
            msg += " Step：t₀ 前/后加速度脉冲，t₁ 后常值制动至终点 v≈0。"
        if bool(self.acc_track_rotor_dyn_chk.isChecked()) and float(self.acc_track_rotor_tau.value()) > 1e-9:
            msg += f" 一阶执行器 τ={float(self.acc_track_rotor_tau.value()):.3f} s。"
        self.log(msg)

    def _on_ee_plan_finished(self, ok: bool, err: str, payload: object):
        self.task_generate_btn.setEnabled(True)
        if not ok:
            self.log(err)
            QMessageBox.critical(self, "Error", err[:2000])
            return
        assert isinstance(payload, dict)
        self._full_plan_result = None
        self._plan_bundle = {
            "kind": "ee_snap",
            "ee_track_kind": payload.get("track_kind", "snap"),
            "t_ref": payload["t_ref"],
            "p_ref": payload["p_ref"],
            "yaw_ref": payload["yaw_ref"],
            "dp_ref": payload.get("dp_ref"),
            "ddp_ref": payload.get("ddp_ref"),
            "dyaw_ref": payload.get("dyaw_ref"),
            "waypoints": payload["waypoints_xyz_yaw"],
            "t_wp": payload["t_wp"],
        }
        self._last_track_res = None
        self._redraw_combined_views(None)
        self._update_track_mode_enabled()
        self.run_track_btn.setEnabled(True)
        self.meshcat_plan_btn.setEnabled(True)
        self.meshcat_track_btn.setEnabled(False)
        try:
            self._save_generated_plan_csv(self._plan_bundle)
            self._autosave_plan_bundle(self._plan_bundle)
        except Exception as e:
            self.log(f"[planning] Failed to save generated CSV: {e!r}")
        self.rn_launch_btn.setEnabled(True)
        if payload.get("track_kind") == "csv_import" and isinstance(payload.get("meta"), dict):
            m = payload["meta"]
            self.log(
                "[csv_import] "
                f"raw vmax={float(m.get('vmax_raw', 0.0)):.3f} m/s, "
                f"limit={float(m.get('vmax_limit', 0.0)):.3f} m/s, "
                f"time_scale={float(m.get('time_scale', 1.0)):.3f}"
            )
        self.log("EE reference generated. We recommend Acados EE-centric tracking.")

    def _update_track_mode_enabled(self):
        """Crocoddyl tracking along the trajectory is selectable only when full-state planning is available."""
        if self._plan_bundle is None:
            self._on_track_mode_changed()
            return
        full = self._plan_bundle["kind"] in ("full_croc", "full_acados")
        try:
            it = self.track_mode_combo.model().item(0)
            if it is not None:
                it.setEnabled(full)
            it_acados_full = self.track_mode_combo.model().item(1)
            if it_acados_full is not None:
                it_acados_full.setEnabled(full and self._EE_MPC_OK)
            it2 = self.track_mode_combo.model().item(3)
            if it2 is not None:
                it2.setEnabled(self._CROC_EE_OK)
        except Exception:
            pass
        if not full and self.track_mode_combo.currentIndex() in (0, 1):
            self.track_mode_combo.setCurrentIndex(2)
        self._on_track_mode_changed()

    def _run_regulation(self):
        mode = int(self.reg_mode_combo.currentIndex())
        T_sim = float(self.T_sim.value())
        sim_dt = float(self.sim_dt.value())
        self._manual_ref_overlay = None
        if mode == 0:
            x0 = self._reg_table_row_to_uam_state(self.reg_full_state_table, 0)
            x_ref = self._reg_table_row_to_uam_state(self.reg_full_state_table, 1)
            t_ref = np.arange(0.0, T_sim + 1e-12, sim_dt, dtype=float)
            if t_ref.size < 2:
                t_ref = np.array([0.0, max(T_sim, sim_dt)], dtype=float)
            x_ref_traj = np.tile(x_ref.reshape(1, -1), (t_ref.size, 1))
            self._manual_ref_overlay = {
                "ref_time_states": t_ref,
                "ref_states": x_ref_traj.copy(),
                "ref_time_controls": None,
                "ref_controls": None,
                "waypoints": None,
            }
            params = {
                "x0": x0,
                "t_plan": np.array([0.0, T_sim], dtype=float),
                "x_plan": np.vstack([x_ref, x_ref]),
                "x_nom": x_ref.copy(),
                "T_sim": T_sim,
                "sim_dt": sim_dt,
                "control_dt": self.control_dt.value(),
                "dt_mpc": self.dt_mpc.value(),
                "horizon": self.croc_horizon.value(),
                "mpc_max_iter": self.croc_mpc_iter.value(),
                "w_state_track": self.w_state_track.value(),
                "w_state_reg": self.w_state_reg.value(),
                "w_control": self.w_control.value(),
                "w_terminal_track": self.w_terminal_track.value(),
                "w_pos": self.w_pos.value(),
                "w_att": self.w_att.value(),
                "w_joint": self.w_joint.value(),
                "w_vel": self.w_vel.value(),
                "w_omega": self.w_omega.value(),
                "w_joint_vel": self.w_joint_vel.value(),
                "w_u_thrust": self.w_u_thrust.value(),
                "w_u_joint_torque": self.w_u_joint_torque.value(),
                "use_actuator_first_order": self.croc_use_actuator_first_order.isChecked(),
                "tau_thrust": float(self.tau_thrust_track.value()),
                "tau_theta": float(self.tau_theta_track.value()),
                "sim_control_stack": (
                    "px4_rate"
                    if self.track_sim_control_stack.currentIndex() == 1
                    else "direct"
                ),
                "px4_rate_Kp": float(self.px4_rate_Kp_track.value()),
                "px4_rate_Kd": float(self.px4_rate_Kd_track.value()),
                "sim_payload_enable": bool(self.sim_payload_enable.isChecked()),
                "sim_payload_t_grasp": float(self.sim_payload_t_grasp.value()),
                "sim_payload_mass": float(self.sim_payload_mass.value()),
                "sim_payload_sphere_r": 0.02,
            }
            self.reg_run_btn.setEnabled(False)
            self.log("Crocoddyl full-state regulation closed loop...")
            self._track_worker = TrackCrocAlongPlanWorker(params)
            self._track_worker.finished.connect(self._on_track_croc_finished)
            self._track_worker.start()
            return

        if not self._CROC_EE_OK or self._croc_ee_mpc is None:
            QMessageBox.warning(self, "Error", "Crocoddyl EE pose tracking is unavailable.")
            return
        x0 = self._reg_table_row_to_uam_state(self.reg_ee_state_table, 0)
        x_ref = self._reg_table_row_to_uam_state(self.reg_ee_state_table, 1)
        t_ref = np.arange(0.0, T_sim + 1e-12, sim_dt, dtype=float)
        if t_ref.size < 2:
            t_ref = np.array([0.0, max(T_sim, sim_dt)], dtype=float)
        pose = self._read_reg_ee_pose_table_row()
        p_goal = np.array(
            [pose["x"], pose["y"], pose["z"]],
            dtype=float,
        )
        yaw_goal = np.deg2rad(float(pose["yaw"]))
        p_ref = np.tile(p_goal.reshape(1, 3), (t_ref.size, 1))
        yaw_ref = np.full(t_ref.size, yaw_goal, dtype=float)
        x_ref_traj = np.tile(x_ref.reshape(1, -1), (t_ref.size, 1))
        self._manual_ref_overlay = {
            "ref_time_states": t_ref,
            "ref_states": x_ref_traj.copy(),
            "ref_time_controls": None,
            "ref_controls": None,
            "waypoints": None,
        }
        params_croc_ee = {
            "x0": x0,
            "t_ref": t_ref,
            "p_ref": p_ref,
            "yaw_ref": yaw_ref,
            "sim_dt": sim_dt,
            "control_dt": self.control_dt.value(),
            "dt_mpc": self.dt_mpc.value(),
            "N_mpc": self.N_mpc.value(),
            "croc_ee_w_pos": float(self.croc_ee_w_pos.value()),
            "croc_ee_w_rot_rp": float(self.croc_ee_w_rot_rp.value()),
            "croc_ee_w_rot_yaw": float(self.croc_ee_w_rot_yaw.value()),
            "croc_ee_w_vel_lin": float(self.croc_ee_w_vel_lin.value()),
            "croc_ee_w_vel_ang_rp": float(self.croc_ee_w_vel_ang_rp.value()),
            "croc_ee_w_vel_ang_yaw": float(self.croc_ee_w_vel_ang_yaw.value()),
            "croc_ee_w_u": float(self.croc_ee_w_u.value()),
            "croc_ee_w_terminal": float(self.croc_ee_w_terminal.value()),
            "w_state_reg": float(self.w_state_reg.value()),
            "w_state_track": float(self.w_state_track.value()),
            "mpc_max_iter": self.mpc_max_iter.value(),
            "use_thrust_constraints": self.croc_ee_use_thrust_constraints.isChecked(),
            "use_actuator_first_order": self.croc_use_actuator_first_order.isChecked(),
            "tau_thrust": float(self.tau_thrust_track.value()),
            "tau_theta": float(self.tau_theta_track.value()),
            "t_plan": np.array([0.0, T_sim], dtype=float),
            "x_plan": np.vstack([x_ref, x_ref]),
            "sim_payload_enable": bool(self.sim_payload_enable.isChecked()),
            "sim_payload_t_grasp": float(self.sim_payload_t_grasp.value()),
            "sim_payload_mass": float(self.sim_payload_mass.value()),
            "sim_payload_sphere_r": 0.02,
        }
        self.reg_run_btn.setEnabled(False)
        self.log("Crocoddyl EE pose regulation closed loop...")
        self._track_worker = TrackEeCrocWorker(params_croc_ee)
        self._track_worker.finished.connect(self._on_track_croc_ee_finished)
        self._track_worker.start()

    def _rn_fs_weight_widgets(self) -> dict:
        return {
            "w_state_track": self.rn_w_state_track,
            "w_state_reg": self.rn_w_state_reg,
            "w_control": self.rn_w_control,
            "w_terminal_track": self.rn_w_terminal_track,
            "w_pos": self.rn_w_pos,
            "w_att": self.rn_w_att,
            "w_joint": self.rn_w_joint,
            "w_vel": self.rn_w_vel,
            "w_omega": self.rn_w_omega,
            "w_joint_vel": self.rn_w_joint_vel,
            "w_u_thrust": self.rn_w_u_thrust,
            "w_u_joint_torque": self.rn_w_u_joint_torque,
            "mpc_max_iter": self.rn_mpc_max_iter,
        }

    def _rn_acados_solver_snapshot(self) -> dict:
        return {
            "acados_solver_mode": self.rn_acados_solver_mode.currentText(),
            "acados_integrator": self.rn_acados_integrator.currentText(),
            "acados_hpipm_mode": self.rn_acados_hpipm.currentText(),
            "acados_qp_iter_max": int(self.rn_acados_qp_iter.value()),
        }

    def _rn_extra_param_widgets(self) -> dict:
        """除 full-state 权重/acados 选项外，每个算法各自维护的其它控制器参数。"""
        return {
            "dt_mpc": self.rn_dt_mpc,
            "horizon": self.rn_horizon,
            "control_rate": self.rn_control_rate,
            "max_thrust_total": self.rn_max_thrust_total,
            "ee_w_pos": self.rn_ee_w_pos,
            "ee_w_rot_rp": self.rn_ee_w_rot_rp,
            "ee_w_rot_yaw": self.rn_ee_w_rot_yaw,
            "ee_w_vel_lin": self.rn_ee_w_vel_lin,
            "ee_w_vel_ang_rp": self.rn_ee_w_vel_ang_rp,
            "ee_w_vel_ang_yaw": self.rn_ee_w_vel_ang_yaw,
            "ee_w_u": self.rn_ee_w_u,
            "ee_w_terminal": self.rn_ee_w_terminal,
            "geo_kp_pos": self.rn_geo_kp_pos,
            "geo_kd_vel": self.rn_geo_kd_vel,
            "geo_kR": self.rn_geo_kR,
            "geo_kOmega": self.rn_geo_kOmega,
            "geo_max_tilt_deg": self.rn_geo_max_tilt_deg,
        }

    def _rn_mpc_weight_snapshot(self) -> dict:
        snap = {}
        for k, w in self._rn_fs_weight_widgets().items():
            snap[k] = (
                int(w.value()) if isinstance(w, QSpinBox) else float(w.value())
            )
        snap.update(self._rn_acados_solver_snapshot())
        for k, w in self._rn_extra_param_widgets().items():
            snap[k] = (
                int(w.value()) if isinstance(w, QSpinBox) else float(w.value())
            )
        return snap

    def _rn_set_spin_value(self, widget, val) -> None:
        if isinstance(widget, QSpinBox):
            widget.setValue(int(round(float(val))))
        else:
            widget.setValue(float(val))

    def _rn_mpc_weight_apply(self, data: dict) -> None:
        extra = self._rn_extra_param_widgets()
        widgets = list(self._rn_fs_weight_widgets().values())
        widgets += [
            self.rn_acados_solver_mode, self.rn_acados_integrator, self.rn_acados_hpipm,
            self.rn_acados_qp_iter,
        ]
        widgets += list(extra.values())
        for w in widgets:
            w.blockSignals(True)
        try:
            for k, w in self._rn_fs_weight_widgets().items():
                if k in data:
                    self._rn_set_spin_value(w, data[k])
            if "acados_solver_mode" in data:
                self.rn_acados_solver_mode.setCurrentText(str(data["acados_solver_mode"]))
            if "acados_integrator" in data:
                self.rn_acados_integrator.setCurrentText(str(data["acados_integrator"]))
            if "acados_hpipm_mode" in data:
                self.rn_acados_hpipm.setCurrentText(str(data["acados_hpipm_mode"]))
            if "acados_qp_iter_max" in data:
                self.rn_acados_qp_iter.setValue(int(data["acados_qp_iter_max"]))
            for k, w in extra.items():
                if k in data:
                    self._rn_set_spin_value(w, data[k])
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _rn_init_mpc_weight_profiles(self) -> None:
        croc = self._rn_mpc_weight_snapshot()
        self._rn_mpc_weight_profiles["croc_full_state"] = dict(croc)
        acados = dict(croc)
        acados["w_terminal_track"] = 100.0
        acados["mpc_max_iter"] = 40
        acados["acados_solver_mode"] = "rti"
        acados["acados_integrator"] = "ERK"
        acados["acados_hpipm_mode"] = "SPEED"
        acados["acados_qp_iter_max"] = 20
        self._rn_mpc_weight_profiles["acados_full_state"] = acados

    def _rn_save_controller_profiles(self) -> None:
        """将每个算法各自的控制器参数持久化到磁盘（跨重启复用，无需重新调参）。"""
        # 先把当前算法的最新设置存入内存 profile。
        cur = self.rn_controller_combo.currentText()
        if cur in self._rn_mpc_weight_profiles:
            self._rn_mpc_weight_profiles[cur] = self._rn_mpc_weight_snapshot()
        try:
            import json as _json

            payload = {
                "version": 1,
                "profiles": {
                    k: dict(v) for k, v in self._rn_mpc_weight_profiles.items() if v
                },
            }
            RN_CONTROLLER_PROFILES_PATH.write_text(
                _json.dumps(payload, indent=2), encoding="utf-8"
            )
            self.log(
                f"[controller params] 已保存各算法控制器参数 → "
                f"{RN_CONTROLLER_PROFILES_PATH.name}（当前算法：{cur}）"
            )
            if hasattr(self, "rn_save_ctrl_params_btn"):
                self.rn_save_ctrl_params_btn.setText("Save controller parameters ✓")
                QTimer.singleShot(
                    1500,
                    lambda: self.rn_save_ctrl_params_btn.setText(
                        "Save controller parameters"
                    ),
                )
        except Exception as e:
            self.log(f"[controller params] 保存失败: {e}")
            QMessageBox.warning(self, "Save failed", f"无法保存控制器参数：\n{e}")

    def _rn_load_controller_profiles(self) -> None:
        """启动时从磁盘加载各算法控制器参数到内存 profile（不直接应用）。"""
        try:
            if not RN_CONTROLLER_PROFILES_PATH.exists():
                return
            import json as _json

            data = _json.loads(RN_CONTROLLER_PROFILES_PATH.read_text(encoding="utf-8"))
            profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
            n = 0
            for mode, prof in profiles.items():
                if mode in self._rn_mpc_weight_profiles and isinstance(prof, dict):
                    self._rn_mpc_weight_profiles[mode].update(prof)
                    n += 1
            if n:
                self.log(
                    f"[controller params] 已加载 {n} 个算法的已保存控制器参数 "
                    f"({RN_CONTROLLER_PROFILES_PATH.name})"
                )
        except Exception as e:
            self.log(f"[controller params] 加载失败（忽略）: {e}")

    def _rn_on_controller_mode_changed(self, index: int = 0) -> None:
        """切换算法时保存当前 profile 并加载对应 Crocoddyl/Acados 权重。"""
        new_mode = self.rn_controller_combo.currentText()
        prev = self._rn_mpc_profile_mode
        if prev in self._rn_mpc_weight_profiles:
            self._rn_mpc_weight_profiles[prev] = self._rn_mpc_weight_snapshot()
        if new_mode in self._rn_mpc_weight_profiles:
            self._rn_mpc_weight_apply(self._rn_mpc_weight_profiles[new_mode])
        self._rn_mpc_profile_mode = new_mode
        if new_mode == "acados_full_state":
            self.rn_fs_weights_title.setText("Cost weights — Acados profile")
        elif new_mode == "croc_full_state":
            self.rn_fs_weights_title.setText("Cost weights — Crocoddyl profile")
        else:
            self.rn_fs_weights_title.setText("Cost weights (full-state MPC)")
        self._rn_update_mpc_panel(index)

    def _rn_update_mpc_panel(self, _index: int = 0) -> None:
        """根据 controller mode 切换 full-state / EE 参数面板可见性。"""
        mode = self.rn_controller_combo.currentText()
        is_full = mode in ("croc_full_state", "acados_full_state")
        is_ee = mode == "croc_ee_pose"
        is_geo = mode == "geometric"
        is_acados = mode == "acados_full_state"
        self._rn_fs_panel.setVisible(is_full)
        self._rn_acados_panel.setVisible(is_acados)
        self.rn_fs_weights_title.setVisible(is_full)
        self._rn_ee_panel.setVisible(is_ee)
        self._rn_geo_panel.setVisible(is_geo)
        # px4 / geometric 模式不使用此处 MPC 参数
        use_mpc_params = mode in ("croc_full_state", "acados_full_state", "croc_ee_pose")
        self.rn_dt_mpc.setEnabled(use_mpc_params)
        self.rn_horizon.setEnabled(use_mpc_params)
        self.rn_mpc_max_iter.setEnabled(use_mpc_params)
        # 切换模式后刷新依赖节点的服务按钮可用性（仅在节点运行时启用）
        if hasattr(self, "rn_start_svc_btn"):
            running = self._rn_process is not None and self._rn_process.poll() is None
            self._set_node_service_buttons_enabled(bool(running))

    def _start_rviz_viz_node(self) -> None:
        """Launch the standalone robot/EE RViz visualizer (decoupled from MPC node)."""
        if self._viz_process is not None and self._viz_process.poll() is None:
            return
        root = Path(__file__).resolve().parent
        script = root / "suite_rviz_state_node.py"
        if not script.exists():
            self.log(f"[viz] visualization node not found: {script}")
            return
        is_s500 = self._is_s500_mode()
        robot_name = (
            self.gz_model_combo.currentText().strip()
            if hasattr(self, "gz_model_combo") and self.gz_model_combo.currentText().strip()
            else self.task_robot_combo.currentText()
        )
        odom_src = (
            self.rn_odom_combo.currentText() if hasattr(self, "rn_odom_combo") else "gazebo"
        )
        use_sim = (
            "true"
            if (hasattr(self, "rn_use_sim_check") and self.rn_use_sim_check.isChecked())
            else "false"
        )
        cmd = [
            sys.executable,
            str(script),
            "__name:=suite_rviz_state_node",
            f"_urdf_path:={self._selected_robot_urdf_path()}",
            f"_robot_name:={robot_name}",
            f"_arm_enabled:={'false' if is_s500 else 'true'}",
            f"_odom_source:={odom_src}",
            f"_use_simulation:={use_sim}",
        ]
        try:
            self._viz_process = subprocess.Popen(
                cmd, cwd=str(root), env=os.environ.copy()
            )
            self.log(
                f"[viz] RViz robot/EE visualizer started (PID={self._viz_process.pid}) "
                f"-> /suite_mpc/robot_markers, /suite_mpc/ee_axes"
            )
        except Exception as e:
            self.log(f"[viz] failed to start visualizer: {e!r}")

    def _restart_rviz_viz_node(self) -> None:
        """重启 viz 节点并 DELETEALL，清除 tracking 曾留下的 latched marker。"""
        self._stop_rviz_viz_node()
        self._start_rviz_viz_node()

    def _current_gazebo_robot_name(self) -> str:
        if hasattr(self, "gz_model_combo") and self.gz_model_combo.currentText().strip():
            return self.gz_model_combo.currentText().strip()
        return self.task_robot_combo.currentText().strip() or "s500_uam"

    def _ensure_gazebo_state_subscription(self) -> None:
        """订阅 /gazebo/model_states，用于 Drone Status 与（可选）尽早启动显示。"""
        if getattr(self, "_gazebo_states_subscribed", False):
            if not getattr(self, "_status_monitor_inited", False):
                self._init_drone_status_monitor()
            return
        if not self._ensure_ros_node():
            return
        try:
            import time as _time
            import rospy
            from gazebo_msgs.msg import ModelStates

            robot_name = self._current_gazebo_robot_name()

            def _gz_cb(msg):
                try:
                    idx = msg.name.index(robot_name)
                except ValueError:
                    return
                p = msg.pose[idx].position
                self._gazebo_pose = (float(p.x), float(p.y), float(p.z))
                self._gazebo_pose_t = _time.monotonic()
                if not getattr(self, "_status_monitor_inited", False):
                    self._init_drone_status_monitor()

            rospy.Subscriber(
                "/gazebo/model_states", ModelStates, _gz_cb, queue_size=10
            )
            self._gazebo_states_subscribed = True
            self.log(
                f"[status] 已订阅 /gazebo/model_states（模型 '{robot_name}'）"
            )
            self._init_drone_status_monitor()
        except Exception as e:
            self.log(f"[status] gazebo/model_states 订阅失败: {e}")

    def _stop_rviz_viz_node(self) -> None:
        if self._viz_process is None:
            return
        try:
            self._viz_process.terminate()
            self._viz_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                self._viz_process.kill()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            self._viz_process = None

    def _start_ros_gazebo(self) -> None:
        pkg = self.gz_pkg_combo.currentText().strip() if hasattr(self, "gz_pkg_combo") else ""
        launch_file = self.gz_launch_combo.currentText().strip() if hasattr(self, "gz_launch_combo") else ""
        args = self.gz_args_edit.text().strip() if hasattr(self, "gz_args_edit") else ""
        model = self.gz_model_combo.currentText().strip() if hasattr(self, "gz_model_combo") else ""
        model_type = self.gz_model_type_combo.currentText().strip() if hasattr(self, "gz_model_type_combo") else ""
        world = self.gz_world_combo.currentText().strip() if hasattr(self, "gz_world_combo") else ""
        if not pkg or not launch_file:
            QMessageBox.warning(self, "Error", "Gazebo 启动参数不完整（package/launch file）。")
            return
        if self._gazebo_process is not None and self._gazebo_process.poll() is None:
            QMessageBox.information(self, "Notice", "Gazebo 进程已在运行。")
            return
        cmd = ["roslaunch", pkg, launch_file]
        if world:
            cmd.append(f"world_name:={world}")
        if model_type:
            cmd.append(f"model_type:={model_type}")
        if model:
            cmd.append(f"robot_model:={model}")
        if args:
            cmd.extend(args.split())
        sdf_path, sdf_note = _predict_gazebo_spawn_sdf_path(launch_file, model_type)
        if sdf_path:
            self.log(f"[Gazebo] Spawn SDF (expected): {sdf_path} ({sdf_note})")
        else:
            self.log(f"[Gazebo] SDF path: {sdf_note}")
        if args and ("sdf:=" in args or "sdf_file:=" in args):
            self.log(
                "[Gazebo] Note: extra args may override the default SDF; "
                "the path above matches the launch file defaults only."
            )
        # 把本项目 models/ 目录加入 GAZEBO_MODEL_PATH，使 SDF 里的 model://s500_uam
        # 等 mesh 资源能被解析，不再依赖 eagle_mpc_debugger 的环境设置。
        gz_env = os.environ.copy()
        models_dir = str((Path(__file__).resolve().parent / "models").resolve())
        prev_model_path = gz_env.get("GAZEBO_MODEL_PATH", "")
        if models_dir not in prev_model_path.split(os.pathsep):
            gz_env["GAZEBO_MODEL_PATH"] = (
                models_dir + (os.pathsep + prev_model_path if prev_model_path else "")
            )
        try:
            self._gazebo_process = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent), env=gz_env)
            self.log(f"Started Gazebo: {' '.join(cmd)} (PID={self._gazebo_process.pid})")
        except Exception as e:
            QMessageBox.critical(self, "Launch failed", str(e)[:2000])
            self.log(f"Gazebo launch failed: {e!r}")
            return
        # Gazebo 位姿 + RViz 可视化：model_states 可用后即订阅/显示（不依赖 tracking node）
        for _delay in (1500, 3000, 5000):
            QTimer.singleShot(_delay, self._ensure_gazebo_state_subscription)
        QTimer.singleShot(2000, self._start_rviz_viz_node)

    def _stop_ros_gazebo(self) -> None:
        try:
            self._gazebo_states_subscribed = False
            self._gazebo_pose = None
            self._gazebo_pose_t = 0.0
            # 0) 关闭独立 RViz 可视化节点
            self._stop_rviz_viz_node()
            subprocess.run(
                ["rosnode", "kill", "/suite_rviz_state_node"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            # 1) 优先关闭当前 GUI 记录的 roslaunch 进程
            if self._gazebo_process is not None:
                try:
                    self._gazebo_process.terminate()
                    self._gazebo_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._gazebo_process.kill()
                    self._gazebo_process.wait(timeout=2)
                except Exception:
                    try:
                        self._gazebo_process.kill()
                    except Exception:
                        pass
                finally:
                    self._gazebo_process = None

            # 2) 清理常见仿真相关 ROS 节点（忽略不存在的节点）
            ros_nodes = [
                "/gazebo",
                "/gazebo_gui",
                "/rviz",
                "/rviz_gui",
                "/robot_state_publisher",
                "/joint_state_publisher",
                "/joint_state_publisher_gui",
                "/move_group",
                "/controller_spawner",
                "/controller_manager",
                "/groundtruth_pub",
            ]
            subprocess.run(
                ["rosnode", "kill", *ros_nodes],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            # 3) 清理残留进程（避免下次启动端口/资源冲突）
            subprocess.run(
                ["killall", "-q", "-9", "gazebo", "gzserver", "gzclient", "gazebo_gui"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                ["killall", "-q", "-9", "rviz", "rviz_gui"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                ["killall", "-q", "-9", "px4"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                ["pkill", "-9", "-f", "python.*simulation"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            # 给系统一点时间完成资源回收
            time.sleep(1.0)
            self.log("Gazebo stopped (clean shutdown + residual process cleanup).")
        except Exception as e:
            self.log(f"Failed to stop Gazebo cleanly: {e!r}")

    def _full_state_ros_plan_from_ee_snap(self, pb: dict) -> dict:
        """Convert ee_snap (world p, dp, ddp, yaw) to a full-state plan for ROS / Meshcat."""
        import pinocchio as pin

        urdf = self._selected_robot_urdf_path()
        rm = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
        nq, nv = int(rm.nq), int(rm.nv)

        t_ref = np.asarray(pb.get("t_ref", []), dtype=float).flatten()
        p_ref = np.asarray(pb.get("p_ref", []), dtype=float)
        yaw_ref = np.asarray(pb.get("yaw_ref", []), dtype=float).flatten()
        if t_ref.size == 0 or p_ref.ndim != 2 or p_ref.shape[1] != 3:
            raise ValueError("Invalid ee_snap bundle for full-state export")
        if yaw_ref.size != t_ref.size:
            yaw_ref = np.zeros_like(t_ref)
        dp_ref = np.asarray(pb.get("dp_ref"), dtype=float) if pb.get("dp_ref") is not None else None
        if dp_ref is None or dp_ref.shape != p_ref.shape:
            if t_ref.size >= 2:
                dp_ref = np.gradient(p_ref, t_ref, axis=0)
            else:
                dp_ref = np.zeros_like(p_ref)
        dyaw_ref = np.asarray(pb.get("dyaw_ref"), dtype=float).flatten() if pb.get("dyaw_ref") is not None else None
        if dyaw_ref is None or dyaw_ref.size != t_ref.size:
            if t_ref.size >= 2:
                dyaw_ref = np.gradient(yaw_ref, t_ref)
            else:
                dyaw_ref = np.zeros_like(t_ref)

        if t_ref.size >= 2:
            ddp_ref = np.asarray(pb.get("ddp_ref"), dtype=float) if pb.get("ddp_ref") is not None else None
            if ddp_ref is None or ddp_ref.shape != dp_ref.shape:
                ddp_ref = np.gradient(dp_ref, t_ref, axis=0)
        else:
            ddp_ref = np.zeros_like(dp_ref)

        def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            nrm = float(np.linalg.norm(v))
            if nrm < 1e-9:
                return np.asarray(fallback, dtype=float).copy()
            return (np.asarray(v, dtype=float) / nrm).copy()

        R_list: list[np.ndarray] = []
        x_plan = np.zeros((t_ref.size, nq + nv), dtype=float)
        for i in range(t_ref.size):
            q = pin.neutral(rm)
            q[:3] = p_ref[i]

            # Build attitude from desired acceleration + yaw:
            # b3 follows thrust direction, yaw sets heading around b3.
            a_des = np.asarray(ddp_ref[i], dtype=float) + np.array([0.0, 0.0, 9.81], dtype=float)
            b3 = _normalize(a_des, np.array([0.0, 0.0, 1.0], dtype=float))
            yaw_i = float(yaw_ref[i])
            b1_yaw = np.array([np.cos(yaw_i), np.sin(yaw_i), 0.0], dtype=float)
            b2 = np.cross(b3, b1_yaw)
            if np.linalg.norm(b2) < 1e-9:
                # Near singular case when b3 and heading align.
                b2 = np.array([-np.sin(yaw_i), np.cos(yaw_i), 0.0], dtype=float)
            b2 = _normalize(b2, np.array([0.0, 1.0, 0.0], dtype=float))
            b1 = _normalize(np.cross(b2, b3), np.array([1.0, 0.0, 0.0], dtype=float))
            R = np.column_stack([b1, b2, b3])
            R_list.append(R)
            quat = pin.Quaternion(R)
            q[3:7] = np.array([quat.x, quat.y, quat.z, quat.w], dtype=float)

            v = np.zeros(nv, dtype=float)
            if nv >= 3:
                # Crocoddyl state convention uses body-frame linear velocity.
                # dp_ref from GUI trajectories is typically world-frame, so convert here.
                v[:3] = R.T @ np.asarray(dp_ref[i], dtype=float)
            x_plan[i, :nq] = q
            x_plan[i, nq:] = v

        # Fill angular velocity reference (body frame) from attitude time derivative.
        if nv >= 6 and len(R_list) > 0:
            omega_ref = np.zeros((t_ref.size, 3), dtype=float)
            if t_ref.size >= 2:
                for i in range(t_ref.size):
                    if i == 0:
                        dt = float(max(t_ref[1] - t_ref[0], 1e-9))
                        Rdot = (R_list[1] - R_list[0]) / dt
                    elif i == t_ref.size - 1:
                        dt = float(max(t_ref[-1] - t_ref[-2], 1e-9))
                        Rdot = (R_list[-1] - R_list[-2]) / dt
                    else:
                        dt = float(max(t_ref[i + 1] - t_ref[i - 1], 1e-9))
                        Rdot = (R_list[i + 1] - R_list[i - 1]) / dt
                    W = R_list[i].T @ Rdot
                    omega_ref[i, 0] = 0.5 * (W[2, 1] - W[1, 2])
                    omega_ref[i, 1] = 0.5 * (W[0, 2] - W[2, 0])
                    omega_ref[i, 2] = 0.5 * (W[1, 0] - W[0, 1])
            else:
                omega_ref[:, 2] = dyaw_ref
            x_plan[:, nq + 3 : nq + 6] = omega_ref

        nu_pad = 4 if self._is_s500_mode() else 6
        return {
            "kind": "full_acados",
            "t_plan": t_ref,
            "x_plan": x_plan,
            "u_plan": np.zeros((max(0, t_ref.size - 1), nu_pad), dtype=float),
            "ddp_plan": ddp_ref,
            "velocity_frame": "body",
        }

    def _launch_tracking_node(self):
        """导出规划并启动 run_tracking_controller.py 子进程。"""
        if self._plan_bundle is None:
            return
        export_pb = self._prepare_ros_export_plan_bundle()
        if export_pb is None:
            return

        # 若已有子进程在运行，先询问是否重启
        if self._rn_process is not None and self._rn_process.poll() is None:
            ret = QMessageBox.question(
                self,
                "节点已在运行",
                "run_tracking_controller 进程仍在运行，是否终止并重新启动？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
            self._kill_tracking_node()

        root = Path(__file__).resolve().parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        # 导出规划 npz
        try:
            from suite_plan_export import export_suite_plan_npz
        except ImportError as e:
            QMessageBox.critical(self, "Import error", f"无法导入 export_suite_plan_npz:\n{e}")
            return

        export_dir = root / ".suite_ros_export"
        export_dir.mkdir(exist_ok=True)
        npz_path = export_dir / "last_suite_plan.npz"
        try:
            export_suite_plan_npz(npz_path, export_pb, dt_plan_fallback_s=float(self.dt_plan.value()))
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e)[:2000])
            self.log(f"ROS export failed: {e!r}")
            return

        ctrl_mode = self.rn_controller_combo.currentText()
        odom_src = self.rn_odom_combo.currentText()
        ctrl_rate = float(self.rn_control_rate.value())
        arm_mode = self.rn_arm_mode_combo.currentText()
        traj_name = self._current_trajectory_save_name()
        use_sim = "true" if self.rn_use_sim_check.isChecked() else "false"
        is_s500 = self._is_s500_mode()
        if is_s500 and ctrl_mode == "croc_ee_pose":
            QMessageBox.warning(self, "Notice", "s500 模式无机械臂，不支持 croc_ee_pose，请选择 croc_full_state / px4 / geometric。")
            return

        script = root / "run_tracking_controller.py"
        cmd = [
            sys.executable,
            str(script),
            "__name:=suite_tracking_controller",
            "_trajectory_source:=suite_npz",
            f"_suite_plan_path:={npz_path}",
            f"_robot_name:={self.task_robot_combo.currentText()}",
            f"_arm_enabled:={'false' if is_s500 else 'true'}",
            f"_trajectory_name:={traj_name}",
            f"_controller_mode:={ctrl_mode}",
            f"_odom_source:={odom_src}",
            f"_control_rate:={ctrl_rate}",
            f"_arm_control_mode:={arm_mode}",
            f"_use_simulation:={use_sim}",
            f"_max_thrust:={self.rn_max_thrust_total.value()}",
            # ── ROS-specific MPC parameters (independent from Tracking tab) ──
            f"_dt_mpc:={self.rn_dt_mpc.value()}",
            f"_horizon:={self.rn_horizon.value()}",
            f"_mpc_max_iter:={self.rn_mpc_max_iter.value()}",
            f"_w_state_track:={self.rn_w_state_track.value()}",
            f"_w_state_reg:={self.rn_w_state_reg.value()}",
            f"_w_control:={self.rn_w_control.value()}",
            f"_w_terminal_track:={self.rn_w_terminal_track.value()}",
            f"_w_pos:={self.rn_w_pos.value()}",
            f"_w_att:={self.rn_w_att.value()}",
            f"_w_joint:={self.rn_w_joint.value()}",
            f"_w_vel:={self.rn_w_vel.value()}",
            f"_w_omega:={self.rn_w_omega.value()}",
            f"_w_joint_vel:={self.rn_w_joint_vel.value()}",
            f"_w_u_thrust:={self.rn_w_u_thrust.value()}",
            f"_w_u_joint_torque:={self.rn_w_u_joint_torque.value()}",
            f"_acados_solver_mode:={self.rn_acados_solver_mode.currentText()}",
            f"_acados_integrator:={self.rn_acados_integrator.currentText()}",
            f"_acados_hpipm_mode:={self.rn_acados_hpipm.currentText()}",
            f"_acados_qp_iter_max:={self.rn_acados_qp_iter.value()}",
            f"_ee_w_pos:={self.rn_ee_w_pos.value()}",
            f"_ee_w_rot_rp:={self.rn_ee_w_rot_rp.value()}",
            f"_ee_w_rot_yaw:={self.rn_ee_w_rot_yaw.value()}",
            f"_ee_w_vel_lin:={self.rn_ee_w_vel_lin.value()}",
            f"_ee_w_vel_ang_rp:={self.rn_ee_w_vel_ang_rp.value()}",
            f"_ee_w_vel_ang_yaw:={self.rn_ee_w_vel_ang_yaw.value()}",
            f"_ee_w_u:={self.rn_ee_w_u.value()}",
            f"_ee_w_terminal:={self.rn_ee_w_terminal.value()}",
            f"_geo_kp_pos:={self.rn_geo_kp_pos.value()}",
            f"_geo_kd_vel:={self.rn_geo_kd_vel.value()}",
            f"_geo_kR:={self.rn_geo_kR.value()}",
            f"_geo_kOmega:={self.rn_geo_kOmega.value()}",
            f"_geo_max_tilt_deg:={self.rn_geo_max_tilt_deg.value()}",
            "_viz_robot_markers:=false",
            "_viz_ee_axes:=false",
        ]
        # RViz 机器人/EE 始终由 suite_rviz_state_node 根据 Gazebo 发布；
        # tracking node 关闭 mesh/axes 发布，避免 Kill 后 latched 旧位姿卡住 RViz。
        self._ensure_gazebo_state_subscription()
        if self._viz_process is None or self._viz_process.poll() is not None:
            QTimer.singleShot(500, self._start_rviz_viz_node)
        try:
            self._rn_process = subprocess.Popen(
                cmd, cwd=str(root), env=os.environ.copy()
            )
        except Exception as e:
            QMessageBox.critical(self, "Launch failed", str(e)[:2000])
            self.log(f"ROS tracking launch failed: {e!r}")
            return

        self.rn_status_label.setText(f"节点状态：运行中  (PID {self._rn_process.pid})")
        self.rn_status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        self.rn_kill_btn.setEnabled(True)
        self._set_node_service_buttons_enabled(True)
        self._init_drone_status_monitor()
        self.log(
            f"Launched run_tracking_controller.py | PID={self._rn_process.pid} | "
            f"mode={ctrl_mode} odom={odom_src} rate={ctrl_rate}Hz | plan={npz_path}\n"
            "节点就绪后，可点 Take off 自动 OFFBOARD+解锁+归位起点，或手动 OFFBOARD 解锁后点 /start_tracking。"
        )

    def _prepare_ros_export_plan_bundle(self) -> dict | None:
        """Prepare a ROS-trackable full-state plan bundle from current planning result."""
        if self._plan_bundle is None:
            QMessageBox.warning(self, "Notice", "当前没有可导出的规划轨迹。")
            return None
        pb = self._plan_bundle
        export_pb = pb
        if pb.get("kind") == "ee_snap":
            try:
                export_pb = self._full_state_ros_plan_from_ee_snap(pb)
            except Exception as e:
                QMessageBox.warning(self, "Notice", f"EE 参考转换为 ROS full-state 失败：{e}")
                return None
        if export_pb["kind"] not in ("full_croc", "full_acados"):
            QMessageBox.warning(
                self,
                "Notice",
                "ROS Tracking Node 需要先完成 Full state 规划（Crocoddyl 或 Acados）。",
            )
            return None
        return export_pb

    def _kill_tracking_node(self):
        """终止 ROS Tracking 子进程（新节点或 PX4 入口）。"""
        if self._rn_process is None:
            return
        try:
            self._rn_process.terminate()
            self._rn_process.wait(timeout=3)
        except Exception:
            try:
                self._rn_process.kill()
            except Exception:
                pass
        self._rn_process = None
        self.rn_status_label.setText("节点状态：已停止")
        self.rn_status_label.setStyleSheet("color: gray;")
        self.rn_kill_btn.setEnabled(False)
        self._set_node_service_buttons_enabled(False)
        self.log("ROS tracking process terminated.")
        # 确保 Gazebo 驱动的 RViz 可视化仍在运行（重启 viz 以 DELETEALL 旧 latched marker）
        if self._gazebo_process is not None and self._gazebo_process.poll() is None:
            self._ensure_gazebo_state_subscription()
            QTimer.singleShot(200, self._restart_rviz_viz_node)

    def _ensure_ros_node(self) -> bool:
        """Initialize a rospy node once (for subscribers/status). Safe to call repeatedly.

        Returns False (without blocking) when no ROS master is reachable.
        """
        if getattr(self, "_ros_node_inited", False):
            return True
        try:
            import rosgraph

            if not rosgraph.is_master_online():
                return False
        except Exception:
            return False
        try:
            import rospy

            rospy.init_node(
                "uam_flight_studio_gui", anonymous=True, disable_signals=True
            )
            self._ros_node_inited = True
            return True
        except Exception as e:
            self.log(f"[mavros] rospy init failed: {e}")
            return False

    def _init_drone_status_monitor(self) -> None:
        """Subscribe to MAVROS state/pose topics and refresh the status labels periodically.

        No-op (retryable) if there is no ROS master yet.
        """
        if getattr(self, "_status_monitor_inited", False):
            return
        self._mav_state = None
        self._mav_local_pose = None
        self._mav_vision_pose = None
        self._gazebo_pose = getattr(self, "_gazebo_pose", None)
        self._gazebo_pose_t = getattr(self, "_gazebo_pose_t", 0.0)
        self._mav_state_t = 0.0
        self._mav_local_t = 0.0
        self._mav_vision_t = 0.0
        self._mpc_stats = None
        self._mpc_stats_t = 0.0
        if not self._ensure_ros_node():
            return
        try:
            import time as _time
            import json as _json
            import rospy
            from mavros_msgs.msg import State
            from geometry_msgs.msg import PoseStamped
            from std_msgs.msg import String as _StringMsg

            def _state_cb(msg):
                self._mav_state = msg
                self._mav_state_t = _time.monotonic()

            def _local_cb(msg):
                self._mav_local_pose = msg
                self._mav_local_t = _time.monotonic()

            def _vision_cb(msg):
                self._mav_vision_pose = msg
                self._mav_vision_t = _time.monotonic()

            def _stats_cb(msg):
                try:
                    self._mpc_stats = _json.loads(msg.data)
                    self._mpc_stats_t = _time.monotonic()
                except Exception:
                    pass

            rospy.Subscriber("/mavros/state", State, _state_cb, queue_size=5)
            rospy.Subscriber(
                "/mavros/local_position/pose", PoseStamped, _local_cb, queue_size=5
            )
            rospy.Subscriber(
                "/mavros/vision_pose/pose", PoseStamped, _vision_cb, queue_size=5
            )
            rospy.Subscriber(
                "/suite_mpc/stats", _StringMsg, _stats_cb, queue_size=5
            )
        except Exception as e:
            self.log(f"[mavros] status subscribe failed: {e}")
            return
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_drone_status)
        self._status_timer.start(300)
        self._status_monitor_inited = True
        self.log("[mavros] drone status monitor active.")

    def _update_drone_status(self) -> None:
        import time as _time

        now = _time.monotonic()
        # 超过该时长未收到对应话题，视为数据失效（链路断开 / Gazebo 已停止）。
        # /mavros/state 默认 1Hz，阈值放宽；位姿话题高频，阈值收紧。
        STATE_STALE = 3.0
        POSE_STALE = 1.5

        def _pos_str(ps, t_recv):
            if ps is None or (now - t_recv) > POSE_STALE:
                return "—"
            p = ps.pose.position
            return f"{p.x:+.2f}, {p.y:+.2f}, {p.z:+.2f}"

        st = self._mav_state
        state_fresh = st is not None and (now - self._mav_state_t) <= STATE_STALE
        if not state_fresh:
            self.rn_st_conn.setText("no link")
            self.rn_st_conn.setStyleSheet("font-weight: bold; color: #b71c1c;")
            self.rn_st_arm.setText("—")
            self.rn_st_arm.setStyleSheet("font-weight: bold; color: gray;")
            self.rn_st_mode.setText("—")
        else:
            conn = bool(getattr(st, "connected", False))
            armed = bool(getattr(st, "armed", False))
            self.rn_st_conn.setText("connected" if conn else "no link")
            self.rn_st_conn.setStyleSheet(
                "font-weight: bold; color: %s;" % ("#2e7d32" if conn else "#b71c1c")
            )
            self.rn_st_arm.setText("ARMED" if armed else "disarmed")
            self.rn_st_arm.setStyleSheet(
                "font-weight: bold; color: %s;" % ("#e65100" if armed else "gray")
            )
            self.rn_st_mode.setText(str(getattr(st, "mode", "—")) or "—")

        # EKF pose 来自 MAVROS /mavros/local_position/pose（PX4 EKF2 融合输出）。
        # Vision pose 来自 /mavros/vision_pose/pose；仿真下 vision = Gazebo 真值，
        # 因此定位一致性始终比较 EKF vs Vision。
        ekf_fresh = (
            self._mav_local_pose is not None
            and (now - self._mav_local_t) <= POSE_STALE
        )
        vis_fresh = (
            self._mav_vision_pose is not None
            and (now - self._mav_vision_t) <= POSE_STALE
        )
        self.rn_st_ekf.setText(_pos_str(self._mav_local_pose, self._mav_local_t))
        self.rn_st_ekf.setStyleSheet("font-weight: bold;")
        self.rn_st_vision.setText(_pos_str(self._mav_vision_pose, self._mav_vision_t))

        # ── Localization：EKF vs Vision 一致性 ──────────────────────────────
        if not ekf_fresh and not vis_fresh:
            self.rn_st_loc.setText("no data")
            self.rn_st_loc.setStyleSheet("font-weight: bold; color: gray;")
        elif not ekf_fresh:
            self.rn_st_loc.setText("UNAVAILABLE (no EKF)")
            self.rn_st_loc.setStyleSheet("font-weight: bold; color: #b71c1c;")
        elif not vis_fresh:
            self.rn_st_loc.setText("UNAVAILABLE (no Vision)")
            self.rn_st_loc.setStyleSheet("font-weight: bold; color: #b71c1c;")
        else:
            pe = self._mav_local_pose.pose.position
            pv = self._mav_vision_pose.pose.position
            err = (
                (float(pe.x) - float(pv.x)) ** 2
                + (float(pe.y) - float(pv.y)) ** 2
                + (float(pe.z) - float(pv.z)) ** 2
            ) ** 0.5
            if err > 0.50:
                self.rn_st_loc.setText(f"UNAVAILABLE  err={err*100:.0f} cm")
                self.rn_st_loc.setStyleSheet("font-weight: bold; color: #b71c1c;")
            elif err > 0.10:
                self.rn_st_loc.setText(f"WARNING  err={err*100:.0f} cm")
                self.rn_st_loc.setStyleSheet("font-weight: bold; color: #e65100;")
            else:
                self.rn_st_loc.setText(f"OK  err={err*100:.1f} cm")
                self.rn_st_loc.setStyleSheet("font-weight: bold; color: #2e7d32;")

        # ── MPC runtime stats ────────────────────────────────────────────────
        ms = getattr(self, "_mpc_stats", None)
        ms_fresh = ms is not None and (now - self._mpc_stats_t) <= 1.0
        if not ms_fresh:
            for lbl in (
                self.rn_ms_hz, self.rn_ms_phase, self.rn_ms_solve,
                self.rn_ms_solve_stat, self.rn_ms_iters, self.rn_ms_horizon,
                self.rn_ms_cost, self.rn_ms_err_yaw, self.rn_ms_err_xyz,
                self.rn_ms_err_pos,
            ):
                lbl.setText("—")
                lbl.setStyleSheet("font-weight: bold; color: gray;")
        else:
            hz = float(ms.get("loop_hz", 0.0))
            tgt = float(ms.get("target_hz", 0.0))
            self.rn_ms_hz.setText(f"{hz:.1f} / {tgt:.0f} Hz")
            ok_hz = tgt <= 0 or hz >= 0.8 * tgt
            self.rn_ms_hz.setStyleSheet(
                "font-weight: bold; color: %s;" % ("#2e7d32" if ok_hz else "#e65100")
            )
            status = int(ms.get("status", 0))
            status_ok = status in (0, 2)
            self.rn_ms_phase.setText(f"{ms.get('mode', '?')} · {ms.get('phase', '?')} · s{status}")
            self.rn_ms_phase.setStyleSheet(
                "font-weight: bold; color: %s;" % ("#2e7d32" if status_ok else "#b71c1c")
            )
            self.rn_ms_solve.setText(f"{float(ms.get('solve_ms', 0.0)):.2f}")
            self.rn_ms_solve.setStyleSheet("font-weight: bold;")
            self.rn_ms_solve_stat.setText(
                f"{float(ms.get('solve_ms_avg', 0.0)):.2f} / {float(ms.get('solve_ms_max', 0.0)):.2f}"
            )
            self.rn_ms_solve_stat.setStyleSheet("font-weight: bold;")
            self.rn_ms_iters.setText(f"{int(ms.get('iters', 0))} / {int(ms.get('qp_iters', 0))}")
            self.rn_ms_iters.setStyleSheet("font-weight: bold;")
            self.rn_ms_horizon.setText(
                f"N={int(ms.get('horizon', 0))}, dt={float(ms.get('dt_mpc', 0.0)):.3f}s"
            )
            self.rn_ms_horizon.setStyleSheet("font-weight: bold;")

            cost = ms.get("cost", None)
            self.rn_ms_cost.setText("—" if cost is None else f"{float(cost):.3g}")
            self.rn_ms_cost.setStyleSheet("font-weight: bold;")

            yaw_err = float(ms.get("err_yaw_deg", 0.0))
            self.rn_ms_err_yaw.setText(f"{yaw_err:+.2f}")
            self.rn_ms_err_yaw.setStyleSheet(
                "font-weight: bold; color: %s;"
                % ("#2e7d32" if abs(yaw_err) <= 5.0 else "#e65100")
            )

            ex = float(ms.get("err_x", 0.0))
            ey = float(ms.get("err_y", 0.0))
            ez = float(ms.get("err_z", 0.0))
            self.rn_ms_err_xyz.setText(f"{ex:+.3f}, {ey:+.3f}, {ez:+.3f}")
            self.rn_ms_err_xyz.setStyleSheet("font-weight: bold;")

            epos = float(ms.get("err_pos", 0.0))
            self.rn_ms_err_pos.setText(f"{epos:.3f}")
            self.rn_ms_err_pos.setStyleSheet(
                "font-weight: bold; color: %s;"
                % ("#2e7d32" if epos <= 0.10 else ("#e65100" if epos <= 0.30 else "#b71c1c"))
            )
        running = self._rn_process is not None and self._rn_process.poll() is None
        self.rn_st_track.setText("node on" if running else "node off")
        self.rn_st_track.setStyleSheet(
            "font-weight: bold; color: %s;" % ("#2e7d32" if running else "gray")
        )

    def _call_set_flight_mode(self, mode: str) -> None:
        """调用 /mavros/set_mode 切换 PX4 飞行模式（OFFBOARD / POSCTL ...）。"""
        import threading

        def _call():
            try:
                import rospy
                from mavros_msgs.srv import SetMode

                rospy.wait_for_service("/mavros/set_mode", timeout=3.0)
                svc = rospy.ServiceProxy("/mavros/set_mode", SetMode)
                resp = svc(base_mode=0, custom_mode=mode)
                ok = bool(getattr(resp, "mode_sent", False))
                msg = f"[set_mode {mode}] {'OK' if ok else 'FAIL'}"
            except Exception as e:
                msg = f"[set_mode {mode}] ERROR: {e}"
            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG

            QMetaObject.invokeMethod(self, "log", Qt.QueuedConnection, Q_ARG(str, msg))

        threading.Thread(target=_call, daemon=True).start()

    def _call_arm(self, arm: bool) -> None:
        """调用 /mavros/cmd/arming 解锁/上锁。"""
        import threading

        def _call():
            try:
                import rospy
                from mavros_msgs.srv import CommandBool

                rospy.wait_for_service("/mavros/cmd/arming", timeout=3.0)
                svc = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
                resp = svc(value=bool(arm))
                ok = bool(getattr(resp, "success", False))
                msg = f"[{'arm' if arm else 'disarm'}] {'OK' if ok else 'FAIL'}"
            except Exception as e:
                msg = f"[{'arm' if arm else 'disarm'}] ERROR: {e}"
            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG

            QMetaObject.invokeMethod(self, "log", Qt.QueuedConnection, Q_ARG(str, msg))

        threading.Thread(target=_call, daemon=True).start()

    def _call_gazebo_takeoff(self, target_z: float = 1.0) -> None:
        """仅依赖 Gazebo + PX4 SITL + MAVROS 的一键起飞（PX4 AUTO.TAKEOFF）。

        使用 PX4 内置的 AUTO.TAKEOFF 模式：设定起飞高度参数 → 解锁 → 切
        AUTO.TAKEOFF。这是一次性指令，PX4 自主爬升到目标高度后会自动转入
        AUTO.LOITER 悬停。**不占用 OFFBOARD、不持续发布设定点流**，因此不会
        与后续 MPC 的 OFFBOARD 接管产生冲突。可在仅启动 Gazebo / PX4 SITL 后使用。
        """
        import threading
        from PyQt5.QtCore import QMetaObject, Qt, Q_ARG

        def _log(msg):
            QMetaObject.invokeMethod(self, "log", Qt.QueuedConnection, Q_ARG(str, msg))

        if not self._ensure_ros_node():
            QMessageBox.warning(
                self, "Notice", "ROS master 不可用，请先启动 Gazebo / PX4 SITL。"
            )
            return

        def _run():
            try:
                import rospy
                from mavros_msgs.srv import SetMode, CommandBool, ParamSet
                from mavros_msgs.msg import ParamValue
            except Exception as e:
                _log(f"[takeoff] import 失败: {e}")
                return

            # 1) 设置 PX4 起飞高度参数 MIS_TAKEOFF_ALT（米，相对起飞点）
            try:
                rospy.wait_for_service("/mavros/param/set", timeout=3.0)
                param_set = rospy.ServiceProxy("/mavros/param/set", ParamSet)
                val = ParamValue(integer=0, real=float(target_z))
                r = param_set(param_id="MIS_TAKEOFF_ALT", value=val)
                _log(
                    f"[takeoff] set MIS_TAKEOFF_ALT={target_z:.2f}m "
                    f"{'OK' if getattr(r, 'success', False) else 'FAIL'}"
                )
            except Exception as e:
                _log(f"[takeoff] set MIS_TAKEOFF_ALT ERROR（忽略，继续）: {e}")

            # 2) 解锁
            try:
                rospy.wait_for_service("/mavros/cmd/arming", timeout=3.0)
                arming = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
                r = arming(value=True)
                if not getattr(r, "success", False):
                    _log("[takeoff] arm FAIL（请检查是否满足解锁条件）")
                    return
                _log("[takeoff] arm OK")
            except Exception as e:
                _log(f"[takeoff] arm ERROR: {e}")
                return

            # 3) 切 AUTO.TAKEOFF（一次性，PX4 自主爬升后自动转 LOITER）
            try:
                rospy.wait_for_service("/mavros/set_mode", timeout=3.0)
                set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
                r = set_mode(base_mode=0, custom_mode="AUTO.TAKEOFF")
                ok = bool(getattr(r, "mode_sent", False))
                _log(f"[takeoff] set AUTO.TAKEOFF {'OK' if ok else 'FAIL'}")
                if ok:
                    _log(
                        f"[takeoff] 已触发自主起飞到约 {target_z:.2f} m，"
                        "到达后 PX4 自动 LOITER 悬停（未占用 OFFBOARD）。"
                    )
            except Exception as e:
                _log(f"[takeoff] set AUTO.TAKEOFF ERROR: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _set_node_service_buttons_enabled(self, enabled: bool) -> None:
        """Enable/disable all ROS-node-dependent service buttons in one place."""
        ctrl_mode = (
            self.rn_controller_combo.currentText()
            if hasattr(self, "rn_controller_combo")
            else ""
        )
        support_reg = ctrl_mode in (
            "croc_full_state", "acados_full_state", "croc_ee_pose", "px4", "geometric"
        )
        for name in (
            "rn_start_svc_btn", "rn_stop_svc_btn", "rn_save_svc_btn",
            "rn_update_ctrl_btn", "rn_update_traj_btn",
        ):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(enabled)
        for name in ("rn_takeoff_btn", "rn_reset_svc_btn", "rn_set_reg_btn"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(enabled and support_reg)

    def _call_tracking_service(self, srv_name: str):
        """在后台线程中调用一个无参数的 ROS Trigger 服务，结果通过日志反馈。"""
        import threading

        def _call():
            try:
                import rospy
                from std_srvs.srv import Trigger
                rospy.wait_for_service(srv_name, timeout=3.0)
                svc = rospy.ServiceProxy(srv_name, Trigger)
                resp = svc()
                msg = f"[{srv_name}] {'OK' if resp.success else 'FAIL'}: {resp.message}"
            except Exception as e:
                msg = f"[{srv_name}] ERROR: {e}"
            rospy.loginfo(msg) if "OK" in msg else rospy.logwarn(msg)
            # Log 到 GUI（需在主线程；用 QMetaObject 保证线程安全）
            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(self, "log", Qt.QueuedConnection, Q_ARG(str, msg))

        threading.Thread(target=_call, daemon=True).start()

    def _call_start_tracking_service(self):
        self._call_tracking_service("/start_tracking")

    def _call_stop_tracking_service(self):
        self._call_tracking_service("/stop_tracking")

    def _call_save_data_service(self):
        self._call_tracking_service("/save_data")

    def _call_update_trajectory_service(self):
        """Export current GUI plan, then call /update_trajectory for hot reload."""
        if self._rn_process is None or self._rn_process.poll() is not None:
            QMessageBox.warning(self, "Notice", "ROS Tracking Node 未运行，无法在线更新轨迹。")
            return
        export_pb = self._prepare_ros_export_plan_bundle()
        if export_pb is None:
            return
        try:
            from suite_plan_export import export_suite_plan_npz
            import rospy
            root = Path(__file__).resolve().parent
            export_dir = root / ".suite_ros_export"
            export_dir.mkdir(exist_ok=True)
            npz_path = export_dir / "last_suite_plan.npz"
            export_suite_plan_npz(npz_path, export_pb, dt_plan_fallback_s=float(self.dt_plan.value()))
            traj_name = self._current_trajectory_save_name()
            rospy.set_param("/suite_tracking_controller/trajectory_name", traj_name)
            rospy.set_param("/suite_tracking_controller/suite_plan_path", str(npz_path))
            self.log(f"[update_trajectory] 已导出最新规划到 {npz_path}")
            self.log(f"[update_trajectory] trajectory_name 已更新为: {traj_name}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e)[:2000])
            self.log(f"[update_trajectory] 轨迹导出失败: {e!r}")
            return
        self._call_tracking_service("/update_trajectory")

    def _call_reset_to_initial_service(self):
        self._call_tracking_service("/reset_to_initial")

    def _call_take_off(self):
        """导出最新规划、刷新节点轨迹，再调用 /take_off 执行两阶段起飞悬停。"""
        if self._rn_process is None or self._rn_process.poll() is not None:
            QMessageBox.warning(self, "Notice", "ROS Tracking Node 未运行，无法 Take off。")
            return
        export_pb = self._prepare_ros_export_plan_bundle()
        if export_pb is None:
            return

        import threading
        from PyQt5.QtCore import QMetaObject, Qt, Q_ARG

        def _run():
            try:
                import rospy
                from std_srvs.srv import Trigger
                from suite_plan_export import export_suite_plan_npz

                root = Path(__file__).resolve().parent
                export_dir = root / ".suite_ros_export"
                export_dir.mkdir(exist_ok=True)
                npz_path = export_dir / "last_suite_plan.npz"
                export_suite_plan_npz(
                    npz_path, export_pb, dt_plan_fallback_s=float(self.dt_plan.value())
                )
                traj_name = self._current_trajectory_save_name()
                rospy.set_param("/suite_tracking_controller/trajectory_name", traj_name)
                rospy.set_param("/suite_tracking_controller/suite_plan_path", str(npz_path))

                if self._rn_process is None or self._rn_process.poll() is not None:
                    msg = "[/take_off] ERROR: tracking 节点已退出，请重新 Launch"
                    QMetaObject.invokeMethod(
                        self, "log", Qt.QueuedConnection, Q_ARG(str, msg)
                    )
                    return

                # 快速重载轨迹（节点内默认不重建 acados，避免数秒编译导致崩溃）
                rospy.wait_for_service("/update_trajectory", timeout=10.0)
                upd = rospy.ServiceProxy("/update_trajectory", Trigger)
                upd_resp = upd()
                if not upd_resp.success:
                    msg = f"[/update_trajectory] FAIL: {upd_resp.message}"
                    QMetaObject.invokeMethod(
                        self, "log", Qt.QueuedConnection, Q_ARG(str, msg)
                    )
                    return

                if self._rn_process.poll() is not None:
                    msg = "[/take_off] ERROR: update_trajectory 后节点已退出（请查看终端日志）"
                    QMetaObject.invokeMethod(
                        self, "log", Qt.QueuedConnection, Q_ARG(str, msg)
                    )
                    return

                rospy.wait_for_service("/take_off", timeout=10.0)
                to_resp = rospy.ServiceProxy("/take_off", Trigger)()
                msg = (
                    f"[/take_off] {'OK' if to_resp.success else 'FAIL'}: {to_resp.message}"
                )
            except Exception as e:
                msg = f"[/take_off] ERROR: {e}"
                if self._rn_process is not None and self._rn_process.poll() is not None:
                    msg += "（tracking 节点已退出，请查看 Launch 终端）"
            QMetaObject.invokeMethod(self, "log", Qt.QueuedConnection, Q_ARG(str, msg))

        self.log("[take_off] 导出规划并刷新轨迹，随后启动起飞序列…")
        threading.Thread(target=_run, daemon=True).start()

    def _build_ros_controller_update_cfg(self) -> dict:
        """组装 /update_controller_params 所需配置（含当前 Croc/Acados profile）。"""
        mode = self.rn_controller_combo.currentText()
        if hasattr(self, "_rn_mpc_weight_profiles") and mode in self._rn_mpc_weight_profiles:
            self._rn_mpc_weight_profiles[mode] = self._rn_mpc_weight_snapshot()
        return {
            "controller_mode": mode,
            "control_rate": float(self.rn_control_rate.value()),
            "max_thrust": float(self.rn_max_thrust_total.value()),
            "dt_mpc": float(self.rn_dt_mpc.value()),
            "horizon": int(self.rn_horizon.value()),
            "mpc_max_iter": int(self.rn_mpc_max_iter.value()),
            "w_state_track": float(self.rn_w_state_track.value()),
            "w_state_reg": float(self.rn_w_state_reg.value()),
            "w_control": float(self.rn_w_control.value()),
            "w_terminal_track": float(self.rn_w_terminal_track.value()),
            "w_pos": float(self.rn_w_pos.value()),
            "w_att": float(self.rn_w_att.value()),
            "w_joint": float(self.rn_w_joint.value()),
            "w_vel": float(self.rn_w_vel.value()),
            "w_omega": float(self.rn_w_omega.value()),
            "w_joint_vel": float(self.rn_w_joint_vel.value()),
            "w_u_thrust": float(self.rn_w_u_thrust.value()),
            "w_u_joint_torque": float(self.rn_w_u_joint_torque.value()),
            "acados_solver_mode": self.rn_acados_solver_mode.currentText(),
            "acados_integrator": self.rn_acados_integrator.currentText(),
            "acados_hpipm_mode": self.rn_acados_hpipm.currentText(),
            "acados_qp_iter_max": int(self.rn_acados_qp_iter.value()),
            "ee_w_pos": float(self.rn_ee_w_pos.value()),
            "ee_w_rot_rp": float(self.rn_ee_w_rot_rp.value()),
            "ee_w_rot_yaw": float(self.rn_ee_w_rot_yaw.value()),
            "ee_w_vel_lin": float(self.rn_ee_w_vel_lin.value()),
            "ee_w_vel_ang_rp": float(self.rn_ee_w_vel_ang_rp.value()),
            "ee_w_vel_ang_yaw": float(self.rn_ee_w_vel_ang_yaw.value()),
            "ee_w_u": float(self.rn_ee_w_u.value()),
            "ee_w_terminal": float(self.rn_ee_w_terminal.value()),
            "geo_kp_pos": float(self.rn_geo_kp_pos.value()),
            "geo_kd_vel": float(self.rn_geo_kd_vel.value()),
            "geo_kR": float(self.rn_geo_kR.value()),
            "geo_kOmega": float(self.rn_geo_kOmega.value()),
            "geo_max_tilt_deg": float(self.rn_geo_max_tilt_deg.value()),
        }

    def _call_update_controller_params(self):
        """
        将当前 ROS Tracking 参数写入节点私有参数后，
        调用 /update_controller_params 在线更新控制器。
        """
        import threading

        cfg = self._build_ros_controller_update_cfg()

        def _run():
            try:
                import rospy
                from std_srvs.srv import Trigger

                if not self._ensure_ros_node():
                    log_msg = "[update_controller_params] ERROR: ROS master 不可用"
                    from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                    QMetaObject.invokeMethod(
                        self, "log", Qt.QueuedConnection, Q_ARG(str, log_msg)
                    )
                    return

                param_path = "/suite_tracking_controller/controller_update_data"
                rospy.set_param(param_path, cfg)

                svc_name = "/update_controller_params"
                rospy.wait_for_service(svc_name, timeout=60.0)
                svc = rospy.ServiceProxy(svc_name, Trigger)
                resp = svc()
                if resp.success:
                    log_msg = f"[update_controller_params] OK: {resp.message}"
                else:
                    log_msg = f"[update_controller_params] FAIL: {resp.message}"
            except Exception as e:
                log_msg = f"[update_controller_params] ERROR: {e}"

            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(self, "log", Qt.QueuedConnection, Q_ARG(str, log_msg))

        threading.Thread(target=_run, daemon=True).start()

    def _call_set_regulation_target(self):
        """
        先通过 rospy.set_param 设置目标，再调用 /set_regulation_target 服务。
        """
        import threading

        x       = self.rn_reg_x.value()
        y       = self.rn_reg_y.value()
        z       = self.rn_reg_z.value()
        yaw_deg = self.rn_reg_yaw.value()
        j1_deg  = self.rn_reg_j1.value()
        j2_deg  = self.rn_reg_j2.value()

        def _run():
            try:
                import rospy
                from std_srvs.srv import Trigger

                # 1. 将目标写入节点私有参数（与 _svc_set_regulation_target 读取的路径一致）
                param_path = "/suite_tracking_controller/regulation_target_data"
                rospy.set_param(param_path, [x, y, z, yaw_deg, j1_deg, j2_deg])

                # 2. 调用服务
                svc_name = "/set_regulation_target"
                rospy.wait_for_service(svc_name, timeout=3.0)
                svc = rospy.ServiceProxy(svc_name, Trigger)
                resp = svc()
                if resp.success:
                    log_msg = (
                        f"[regulation_target] 已设置: "
                        f"x={x:.2f} y={y:.2f} z={z:.2f} "
                        f"yaw={yaw_deg:.1f}° j1={j1_deg:.1f}° j2={j2_deg:.1f}°\n"
                        f"  节点回复: {resp.message}"
                    )
                else:
                    log_msg = f"[regulation_target] 服务返回失败: {resp.message}"
            except Exception as e:
                log_msg = f"[regulation_target] 错误: {e}"
            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(self, "log", Qt.QueuedConnection, Q_ARG(str, log_msg))

        threading.Thread(target=_run, daemon=True).start()

    # =========================================================================
    # ROS Tracking 结果绘图
    # =========================================================================

    def _plot_ros_tracking_data(self):
        """弹出文件对话框，加载 run_tracking_controller.py 保存的 npz 并绘图。"""
        default_dir = str(
            Path(__file__).resolve().parent / "results" / "suite_tracking"
        )
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load ROS Tracking Data",
            default_dir,
            "NumPy archives (*.npz);;All files (*)",
        )
        if not filepath:
            return
        try:
            self._render_ros_tracking_figures(filepath)
            self.log(f"[plot] Loaded and rendered: {filepath}")
            if hasattr(self, "_right_plot_tabs"):
                self._right_plot_tabs.setCurrentIndex(0)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            self.log(f"[plot] ERROR rendering {filepath}:\n{tb}")
            QMessageBox.critical(
                self,
                "Plot failed",
                f"无法绘制该 npz（键名或维度需与 run_tracking_controller 录制一致）。\n\n{e}\n\n{tb[:1200]}",
            )

    def _render_ros_tracking_figures(self, npz_path: str) -> None:
        """
        从 run_tracking_controller.py 保存的 npz 构建 res dict，
        直接调用 _render_tracking_figures(res, "direct")，
        与 Run Closed Loop Tracking 完全相同的绘图 API 和 Tab 布局。
        """
        import traceback as _tb

        d = np.load(npz_path, allow_pickle=True)

        def _arr(key):
            if key in d and d[key].size > 0:
                return np.asarray(d[key], dtype=float)
            return np.zeros((0,), dtype=float)

        t       = _arr("time").flatten()
        pos     = _arr("position")                   # (N,3)
        vel     = _arr("velocity")                   # (N,3) body linear v
        omg     = _arr("angular_velocity")           # (N,3) body angular v
        ori     = _arr("orientation")                # (N,4) qx qy qz qw
        j_pos   = _arr("arm_joint_positions")        # (N,nj)
        j_vel   = _arr("arm_joint_velocities")       # (N,nj)
        u_mpc   = _arr("mpc_control")                # (N,nu)
        t_solve = _arr("mpc_solve_time").flatten()   # ms
        r_pos   = _arr("reference_position")         # (N,3)
        r_ori   = _arr("reference_orientation")      # (N,4)
        r_jpos  = _arr("reference_arm_positions")    # (N,nj)
        r_vel   = _arr("reference_velocity")
        r_omg   = _arr("reference_angular_velocity")
        brate   = _arr("body_rate_commands")          # (N,3) CTBR body-rate cmd [rad/s]
        thr_cmd = _arr("thrust_command").flatten()    # (N,) normalized thrust cmd

        N = len(t)
        if N < 2:
            self.log("[plot] Not enough data points to plot.")
            return

        nj = j_pos.shape[1] if (j_pos.ndim == 2 and j_pos.shape[0] == N and j_pos.shape[1] > 0) else 0
        nq = 7 + nj
        nv = 6 + nj

        if vel.ndim != 2 or vel.shape[0] != N:
            vel = np.zeros((N, 3), dtype=float)
        if omg.ndim != 2 or omg.shape[0] != N:
            omg = np.zeros((N, 3), dtype=float)

        # ── 重建完整状态 x = [q; v]，与 run_tracking_controller._record / plot_acados 约定一致 ──
        q_part = np.hstack([pos, ori, j_pos[:N]]) if nj > 0 else np.hstack([pos, ori])
        if j_vel.ndim == 2 and j_vel.shape[0] >= N and nj > 0:
            jv = np.asarray(j_vel[:N, :nj], dtype=float)
        else:
            jv = np.zeros((N, max(0, nj)), dtype=float)
        v_body = np.hstack([vel[:N, :3], omg[:N, :3], jv])
        x_act = np.hstack([q_part, v_body])
        if x_act.shape[1] != nq + nv:
            raise ValueError(
                f"Rebuilt state width mismatch: got {x_act.shape[1]}, expect nq+nv={nq}+{nv}={nq + nv}"
            )

        def _pad_freeflyer13_to_uam17(x_mat: np.ndarray) -> np.ndarray:
            """plot_acados 需要 ≥15 列 twist；s500 录制为 13 维 [q7; v6]，补零关节位/速到 UAM 17 维。"""
            x_mat = np.asarray(x_mat, dtype=float)
            if x_mat.ndim != 2 or x_mat.shape[1] != 13:
                return x_mat
            nw = int(x_mat.shape[0])
            out = np.zeros((nw, 17), dtype=float)
            out[:, :7] = x_mat[:, :7]
            out[:, 9:15] = x_mat[:, 7:13]
            return out

        x_act = _pad_freeflyer13_to_uam17(x_act)

        # ── 参考状态（含参考速度，供 plot_acados 虚线）──────────────────────
        has_ref = r_pos.ndim == 2 and r_pos.shape[0] == N
        if has_ref:
            if r_ori.ndim != 2 or r_ori.shape[0] != N or r_ori.shape[1] < 4:
                r_ori = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float), (N, 1))
            if nj > 0 and r_jpos.ndim == 2 and r_jpos.shape[0] >= N:
                jr = np.asarray(r_jpos[:N, :nj], dtype=float)
            else:
                jr = np.zeros((N, max(0, nj)), dtype=float)
            qr = np.hstack([r_pos, r_ori, jr]) if nj > 0 else np.hstack([r_pos, r_ori])
            if r_vel.ndim == 2 and r_vel.shape[0] == N:
                rv_lin = r_vel[:N, :3]
            else:
                rv_lin = np.zeros((N, 3), dtype=float)
            if r_omg.ndim == 2 and r_omg.shape[0] == N:
                rv_omg = r_omg[:N, :3]
            else:
                rv_omg = np.zeros((N, 3), dtype=float)
            v_ref = np.hstack([rv_lin, rv_omg, np.zeros((N, max(0, nj)), dtype=float)])
            x_ref_states = np.hstack([qr, v_ref])
            x_ref_states = _pad_freeflyer13_to_uam17(x_ref_states)
        else:
            x_ref_states = None

        # ── EE FK ──────────────────────────────────────────────────────────
        if self._is_s500_mode():
            ee_act = np.full((N, 3), np.nan, dtype=float)
            ee_ref = np.full((N, 3), np.nan, dtype=float)
            ee_yaw_act = np.full(N, np.nan, dtype=float)
            yaw_ref = np.full(N, np.nan, dtype=float)
            err = np.full(N, np.nan, dtype=float)
            err_yaw = np.full(N, np.nan, dtype=float)
        else:
            try:
                from s500_uam_acados_trajectory_plot import compute_ee_kinematics_along_trajectory

                rm, eid = self._robot_model_and_ee()
                pin_data = rm.createData()
                ee_act_raw, _, ee_rpy_a, _ = compute_ee_kinematics_along_trajectory(x_act, rm, pin_data, eid)
                ee_act      = np.asarray(ee_act_raw, dtype=float)
                ee_yaw_act  = np.unwrap(np.asarray(ee_rpy_a[:, 2], dtype=float).flatten())
                if x_ref_states is not None:
                    ee_ref_raw, _, ee_rpy_r, _ = compute_ee_kinematics_along_trajectory(x_ref_states, rm, pin_data, eid)
                    ee_ref  = np.asarray(ee_ref_raw, dtype=float)
                    yaw_ref = np.unwrap(np.asarray(ee_rpy_r[:, 2], dtype=float).flatten())
                else:
                    ee_ref  = ee_act.copy()
                    yaw_ref = ee_yaw_act.copy()
            except Exception:
                self.log(f"[plot] EE FK failed (using base pos as fallback):\n{_tb.format_exc()}")
                ee_act     = pos.copy()
                ee_yaw_act = np.zeros(N)
                ee_ref     = r_pos.copy() if has_ref else pos.copy()
                yaw_ref    = np.zeros(N)

            err     = np.linalg.norm(ee_act - ee_ref, axis=1) if ee_act.ndim == 2 and ee_ref.ndim == 2 else np.zeros(N)
            err_yaw = (ee_yaw_act - yaw_ref + np.pi) % (2.0 * np.pi) - np.pi

        # ── 控制量对齐 ────────────────────────────────────────────────────
        if u_mpc.ndim == 2 and u_mpc.shape[0] in (N, N - 1):
            u_out = u_mpc
        else:
            u_out = np.zeros((max(0, N - 1), 4), dtype=float)
        N_u = u_out.shape[0]

        # ── MPC 求解统计（与 closed-loop tracking 相同结构）─────────────
        mpc_wall       = np.zeros(N_u, dtype=float)
        mpc_iter       = np.zeros(N_u, dtype=int)
        mpc_stat       = np.zeros(N_u, dtype=int)
        mpc_total_cost = np.full(N_u, np.nan, dtype=float)
        if t_solve.size >= N_u > 0:
            mpc_wall[:] = t_solve[:N_u] / 1000.0   # ms → s

        dt = float(t[1] - t[0]) if N >= 2 else 0.1

        # ── 组装 res dict（与 TrackEeCrocWorker 完全相同的 key/格式）─────
        res = {
            "t":            t,
            "x":            x_act,
            "u":            u_out,
            "ee":           ee_act,
            "p_ref":        ee_ref,
            "err":          err,
            "ee_yaw":       ee_yaw_act,
            "yaw_ref":      yaw_ref,
            "err_yaw":      err_yaw,
            "control_mode": "direct",
            "sim_dt":       dt,
            "control_dt":   dt,
            "mpc_stride":   1,
            "mpc_solve": {
                "nlp_iter":   mpc_iter,
                "cpu_s":      mpc_wall.copy(),
                "wall_s":     mpc_wall,
                "status":     mpc_stat,
                "total_cost": mpc_total_cost,
            },
            "mpc_cost_t":      t[:N_u] if N_u > 0 else np.array([]),
            "mpc_cost_total":  np.full(N_u, np.nan, dtype=float),
            "mpc_cost_terms":  {"solve_ms": t_solve[:N_u]} if t_solve.size >= N_u > 0 else {},
            "mpc_cost_groups": {},
            "mpc_cost_weights": {},
        }

        # ── 临时设置参考状态叠加层 ────────────────────────────────────────
        old_manual = self._manual_ref_overlay
        if x_ref_states is not None:
            self._manual_ref_overlay = {
                "ref_time_states": t,
                "ref_states":      x_ref_states,
            }
        try:
            self._render_tracking_figures(res, "direct")
        finally:
            self._manual_ref_overlay = old_manual

        # ── CTBR / Feedback 专用图：实际 vs 规划电机力、CTBR 指令、角速度反馈 ──
        try:
            self._render_ctbr_feedback_figure(
                t=t,
                u_mpc=u_out,
                omg=omg[:N] if omg.ndim == 2 and omg.shape[0] >= N else None,
                r_omg=r_omg[:N] if r_omg.ndim == 2 and r_omg.shape[0] >= N else None,
                brate=brate[:N] if brate.ndim == 2 and brate.shape[0] >= N else None,
                thr_cmd=thr_cmd[:N] if thr_cmd.size >= N else None,
            )
        except Exception:
            self.log(f"[plot] CTBR/feedback figure failed:\n{_tb.format_exc()}")

        if self._is_s500_mode():
            self.log(
                f"[plot] Rendered {N} steps (s500 base-only) | "
                + (f"solve mean={t_solve.mean():.1f} ms" if t_solve.size else "no solve time")
            )
        else:
            self.log(
                f"[plot] Rendered {N} steps | "
                f"EE err mean={float(np.nanmean(err)):.3f} m | "
                + (f"solve mean={t_solve.mean():.1f} ms" if t_solve.size else "no solve time")
            )

    def _render_ctbr_feedback_figure(
        self,
        *,
        t: np.ndarray,
        u_mpc: np.ndarray,
        omg: Optional[np.ndarray],
        r_omg: Optional[np.ndarray],
        brate: Optional[np.ndarray],
        thr_cmd: Optional[np.ndarray],
    ) -> None:
        """Control 图（合并控制输入 + CTBR/反馈，2 列 3 行）。

        布局（行优先）：
          1) 电机力：MPC 实际输出 vs 规划      2) CTBR 推力指令（归一化）
          3) roll rate p：指令/反馈/参考       4) pitch rate q：指令/反馈/参考
          5) yaw rate r：指令/反馈/参考        6) 角速度跟踪误差（指令 - 反馈）
        """
        fig = self.fig_control
        fig.clear()
        t = np.asarray(t, dtype=float).flatten()
        N = t.size
        if N < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            self.cv_control.draw()
            return

        # 规划电机力（从当前 plan bundle 重采样到记录时间轴）
        u_plan_interp = None
        pb = self._plan_bundle
        try:
            if pb is not None and pb.get("kind") in ("full_croc", "full_acados"):
                up = np.asarray(pb.get("u_plan"), dtype=float)
                tp = np.asarray(pb.get("t_plan"), dtype=float).flatten()
                if up.ndim == 2 and up.shape[0] >= 1 and tp.size >= 2:
                    tpc = tp[: up.shape[0]] - tp[0]
                    nrot = min(4, up.shape[1])
                    u_plan_interp = np.column_stack([
                        np.interp(t, tpc, up[: tpc.size, j]) for j in range(nrot)
                    ])
        except Exception:
            u_plan_interp = None

        u_mpc = np.asarray(u_mpc, dtype=float)
        n_rot = min(4, u_mpc.shape[1]) if u_mpc.ndim == 2 and u_mpc.shape[0] > 0 else 0
        t_u = t[: u_mpc.shape[0]] if (u_mpc.ndim == 2 and u_mpc.shape[0] <= N) else t

        # 2 列 3 行（行优先索引：1..6）
        axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        # ── 面板 1：电机力（实际 vs 规划）────────────────────────────────────
        ax = axes[0]
        if n_rot > 0:
            for j in range(n_rot):
                ax.plot(t_u, u_mpc[: t_u.size, j], color=colors[j], lw=1.1,
                        label=f"MPC rotor {j+1}")
            if u_plan_interp is not None:
                for j in range(min(n_rot, u_plan_interp.shape[1])):
                    ax.plot(t, u_plan_interp[:, j], color=colors[j], lw=1.0,
                            ls="--", alpha=0.7,
                            label=f"plan rotor {j+1}" if j == 0 else None)
        ax.set_title("Rotor forces — MPC (solid) vs plan (dashed)", fontsize=_mpl_pt(9))
        ax.set_ylabel("force [N]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=_mpl_pt(6), ncol=2)

        # ── 面板 2：CTBR 推力指令 ────────────────────────────────────────────
        ax = axes[1]
        if thr_cmd is not None and np.asarray(thr_cmd).size >= 2:
            tc = np.asarray(thr_cmd, dtype=float).flatten()
            ax.plot(t[: tc.size], tc[: t.size], color="k", lw=1.1, label="CTBR thrust cmd")
        ax.set_title("CTBR normalized thrust command (→ PX4)", fontsize=_mpl_pt(9))
        ax.set_ylabel("thrust [0..1]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=_mpl_pt(6))

        # ── 面板 3-5：三轴角速度（指令 vs 反馈 vs 参考）─────────────────────
        labels = ["roll rate p", "pitch rate q", "yaw rate r"]
        for k in range(3):
            ax = axes[2 + k]
            if brate is not None and np.asarray(brate).ndim == 2:
                br = np.asarray(brate, dtype=float)
                ax.plot(t[: br.shape[0]], br[: t.size, k], color="tab:red", lw=1.2,
                        label="CTBR cmd")
            if omg is not None and np.asarray(omg).ndim == 2:
                ow = np.asarray(omg, dtype=float)
                ax.plot(t[: ow.shape[0]], ow[: t.size, k], color="tab:blue", lw=1.1,
                        label="feedback")
            if r_omg is not None and np.asarray(r_omg).ndim == 2:
                rw = np.asarray(r_omg, dtype=float)
                ax.plot(t[: rw.shape[0]], rw[: t.size, k], color="tab:green", lw=1.0,
                        ls="--", alpha=0.8, label="reference")
            ax.set_title(f"Angular velocity — {labels[k]} [rad/s]", fontsize=_mpl_pt(9))
            ax.set_ylabel("[rad/s]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=_mpl_pt(6), ncol=3)

        # ── 面板 6：角速度跟踪误差（CTBR 指令 - 反馈），直观体现 PX4 延迟 ──────
        ax = axes[5]
        if (
            brate is not None and np.asarray(brate).ndim == 2
            and omg is not None and np.asarray(omg).ndim == 2
        ):
            br = np.asarray(brate, dtype=float)
            ow = np.asarray(omg, dtype=float)
            n = min(br.shape[0], ow.shape[0], t.size)
            err_labels = ["p", "q", "r"]
            err_colors = ["tab:blue", "tab:orange", "tab:green"]
            for k in range(3):
                ax.plot(t[:n], br[:n, k] - ow[:n, k], color=err_colors[k], lw=1.0,
                        label=f"{err_labels[k]} err")
            ax.axhline(0.0, color="gray", lw=0.6, alpha=0.6)
        ax.set_title("Body-rate tracking error (CTBR cmd − feedback)", fontsize=_mpl_pt(9))
        ax.set_ylabel("[rad/s]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=_mpl_pt(6), ncol=3)

        for ax in (axes[4], axes[5]):
            ax.set_xlabel("t [s]")

        fig.suptitle("Control — rotor forces, CTBR commands & body-rate feedback",
                     fontsize=_mpl_pt(11), y=0.995)
        try:
            fig.tight_layout(rect=(0, 0, 1, 0.98))
        except Exception:
            pass
        self.cv_control.draw()

    def _run_track(self):
        if self._plan_bundle is None:
            return
        self._manual_ref_overlay = None
        mode = self.track_mode_combo.currentIndex()
        if mode == 0:
            if self._plan_bundle["kind"] not in ("full_croc", "full_acados"):
                QMessageBox.warning(
                    self,
                    "Notice",
                    "Tracking along the full-state plan requires doing the \"Full state\" planning first.",
                )
                return
            from s500_uam_crocoddyl_state_tracking_mpc import default_hover_nominal

            pb = self._plan_bundle
            x0 = np.asarray(pb["x_plan"][0], dtype=float).flatten()
            params = {
                "x0": x0,
                "t_plan": pb["t_plan"],
                "x_plan": pb["x_plan"],
                "x_nom": default_hover_nominal(),
                "T_sim": self.T_sim.value(),
                "sim_dt": self.sim_dt.value(),
                "control_dt": self.control_dt.value(),
                "dt_mpc": self.dt_mpc.value(),
                "horizon": self.croc_horizon.value(),
                "mpc_max_iter": self.croc_mpc_iter.value(),
                "w_state_track": self.w_state_track.value(),
                "w_state_reg": self.w_state_reg.value(),
                "w_control": self.w_control.value(),
                "w_terminal_track": self.w_terminal_track.value(),
                "w_pos": self.w_pos.value(),
                "w_att": self.w_att.value(),
                "w_joint": self.w_joint.value(),
                "w_vel": self.w_vel.value(),
                "w_omega": self.w_omega.value(),
                "w_joint_vel": self.w_joint_vel.value(),
                "w_u_thrust": self.w_u_thrust.value(),
                "w_u_joint_torque": self.w_u_joint_torque.value(),
                "use_actuator_first_order": self.croc_use_actuator_first_order.isChecked(),
                "tau_thrust": float(self.tau_thrust_track.value()),
                "tau_theta": float(self.tau_theta_track.value()),
                "sim_control_stack": (
                    "px4_rate"
                    if self.track_sim_control_stack.currentIndex() == 1
                    else "direct"
                ),
                "px4_rate_Kp": float(self.px4_rate_Kp_track.value()),
                "px4_rate_Kd": float(self.px4_rate_Kd_track.value()),
                "sim_payload_enable": bool(self.sim_payload_enable.isChecked()),
                "sim_payload_t_grasp": float(self.sim_payload_t_grasp.value()),
                "sim_payload_mass": float(self.sim_payload_mass.value()),
                "sim_payload_sphere_r": 0.02,
            }
            self.run_track_btn.setEnabled(False)
            self.log("Crocoddyl closed-loop tracking along the plan…")
            self._track_worker = TrackCrocAlongPlanWorker(params)
            self._track_worker.finished.connect(self._on_track_croc_finished)
            self._track_worker.start()
            return

        if mode == 1:
            if not self._EE_MPC_OK:
                QMessageBox.warning(self, "Error", "Acados MPC is unavailable.")
                return
            pb = self._plan_bundle
            x0 = np.asarray(pb["x_plan"][0], dtype=float).flatten()
            cm = (
                "direct"
                if self.control_mode_track.currentIndex() == 0
                else "actuator_first_order"
            )
            params = {
                "x0": x0,
                "t_plan": pb["t_plan"],
                "x_plan": pb["x_plan"],
                "T_sim": self.T_sim.value(),
                "sim_dt": self.sim_dt.value(),
                "control_dt": self.control_dt.value(),
                "dt_mpc": self.dt_mpc.value(),
                "N": self.N_mpc.value(),
                "mpc_max_iter": self.mpc_max_iter.value(),
                "mpc_log_interval": self.mpc_log_iv.value(),
                "control_mode": cm,
                "w_state_track": self.w_state_track.value(),
                "w_state_reg": self.w_state_reg.value(),
                "w_control": self.w_control.value(),
                "w_terminal_track": self.w_terminal_track.value(),
                "w_pos": self.w_pos.value(),
                "w_att": self.w_att.value(),
                "w_joint": self.w_joint.value(),
                "w_vel": self.w_vel.value(),
                "w_omega": self.w_omega.value(),
                "w_joint_vel": self.w_joint_vel.value(),
                "w_u_thrust": self.w_u_thrust.value(),
                "w_u_joint_torque": self.w_u_joint_torque.value(),
            }
            self.run_track_btn.setEnabled(False)
            self.log("Acados closed-loop tracking along the plan (shared Croc cost weights)…")
            self._track_worker = TrackAcadosAlongPlanWorker(params)
            self._track_worker.finished.connect(self._on_track_acados_full_finished)
            self._track_worker.start()
            return

        pb = self._plan_bundle
        sim_dt = self.sim_dt.value()
        if pb["kind"] == "ee_snap":
            t_ref = np.asarray(pb["t_ref"], dtype=float).flatten()
            p_ref = np.asarray(pb["p_ref"], dtype=float)
            yaw_ref = np.asarray(pb["yaw_ref"], dtype=float).flatten()
            waypoints = pb.get("waypoints")
            t_wp = pb.get("t_wp")
            ek = pb.get("ee_track_kind")
            if ek == "eight":
                plan_title = "EE figure-eight (plan ref)"
            elif ek == "acc_track":
                plan_title = "Acc tracking test (plan ref)"
            else:
                plan_title = "EE minimum snap (plan ref)"
            x0_for_ee = self._aligned_x0_from_ee_ref(p_ref, yaw_ref, x_seed=None)
        else:
            rm, eid = self._robot_model_and_ee()
            t_ref, p_ref, yaw_ref = build_ee_ref_from_full_state(
                pb["t_plan"],
                pb["x_plan"],
                rm,
                eid,
                self.T_sim.value(),
                sim_dt,
            )
            waypoints = None
            t_wp = None
            plan_title = "EE ref (from full-state plan FK)"
            x0_for_ee = self._aligned_x0_from_ee_ref(
                p_ref, yaw_ref, x_seed=np.asarray(pb["x_plan"][0], dtype=float).flatten()[:17]
            )

        if mode == 3:
            if not self._CROC_EE_OK or self._croc_ee_mpc is None:
                QMessageBox.warning(self, "Error", "Crocoddyl EE pose tracking is unavailable.")
                return
            t_plan_ee = None
            x_plan_ee = None
            if pb.get("kind") != "ee_snap" and pb.get("t_plan") is not None and pb.get("x_plan") is not None:
                t_plan_ee = np.asarray(pb["t_plan"], dtype=float)
                x_plan_ee = np.asarray(pb["x_plan"], dtype=float)
            params_croc_ee = {
                "x0": x0_for_ee,
                "t_ref": t_ref,
                "p_ref": p_ref,
                "yaw_ref": yaw_ref,
                "sim_dt": sim_dt,
                "control_dt": self.control_dt.value(),
                "dt_mpc": self.dt_mpc.value(),
                "N_mpc": self.N_mpc.value(),
                "croc_ee_w_pos": float(self.croc_ee_w_pos.value()),
                "croc_ee_w_rot_rp": float(self.croc_ee_w_rot_rp.value()),
                "croc_ee_w_rot_yaw": float(self.croc_ee_w_rot_yaw.value()),
                "croc_ee_w_vel_lin": float(self.croc_ee_w_vel_lin.value()),
                "croc_ee_w_vel_ang_rp": float(self.croc_ee_w_vel_ang_rp.value()),
                "croc_ee_w_vel_ang_yaw": float(self.croc_ee_w_vel_ang_yaw.value()),
                "croc_ee_w_u": float(self.croc_ee_w_u.value()),
                "croc_ee_w_terminal": float(self.croc_ee_w_terminal.value()),
                "w_state_reg": float(self.w_state_reg.value()),
                "w_state_track": float(self.w_state_track.value()),
                "mpc_max_iter": self.mpc_max_iter.value(),
                "use_thrust_constraints": self.croc_ee_use_thrust_constraints.isChecked(),
                "use_actuator_first_order": self.croc_use_actuator_first_order.isChecked(),
                "tau_thrust": float(self.tau_thrust_track.value()),
                "tau_theta": float(self.tau_theta_track.value()),
                "t_plan": t_plan_ee,
                "x_plan": x_plan_ee,
                "sim_payload_enable": bool(self.sim_payload_enable.isChecked()),
                "sim_payload_t_grasp": float(self.sim_payload_t_grasp.value()),
                "sim_payload_mass": float(self.sim_payload_mass.value()),
                "sim_payload_sphere_r": 0.02,
            }
            self.run_track_btn.setEnabled(False)
            self.log("Crocoddyl EE pose closed loop…")
            self._track_worker = TrackEeCrocWorker(params_croc_ee)
            self._track_worker.finished.connect(self._on_track_croc_ee_finished)
            self._track_worker.start()
            return

        if not self._EE_MPC_OK or self._ee_mpc is None:
            QMessageBox.warning(self, "Error", "Acados EE MPC is unavailable.")
            return

        cm = (
            "direct"
            if self.control_mode_track.currentIndex() == 0
            else "actuator_first_order"
        )
        params = {
            "t_ref": t_ref,
            "p_ref": p_ref,
            "yaw_ref": yaw_ref,
            "x0_init": x0_for_ee,
            "T_sim": self.T_sim.value(),
            "sim_dt": sim_dt,
            "control_dt": self.control_dt.value(),
            "dt_mpc": self.dt_mpc.value(),
            "N_mpc": self.N_mpc.value(),
            "w_ee": self.w_ee.value(),
            "w_ee_yaw": self.w_ee_yaw.value(),
            "mpc_max_iter": self.mpc_max_iter.value(),
            "mpc_log_interval": self.mpc_log_iv.value(),
            "control_mode": cm,
            "plan_title": plan_title,
            "waypoints": waypoints,
            "t_wp": t_wp,
            "track_label": "suite_ee",
        }
        self.run_track_btn.setEnabled(False)
        self.log("Acados EE-centric closed loop…")
        self._track_worker = TrackEeAcadosWorker(params)
        self._track_worker.finished.connect(self._on_track_ee_finished)
        self._track_worker.start()

    def _on_track_acados_full_finished(self, ok: bool, err: str, payload: object):
        self.run_track_btn.setEnabled(True)
        if hasattr(self, "reg_run_btn"):
            self.reg_run_btn.setEnabled(True)
        if not ok:
            self.log(err)
            QMessageBox.critical(self, "Error", err[:2000])
            return
        assert isinstance(payload, dict)
        res = payload["res"]
        cm = payload.get("control_mode", "direct")
        self._render_tracking_figures(res, cm, payload.get("out"))
        self.log(
            f"Acados full-state tracking finished | EE error (final) {res['err'][-1]:.4f} m | "
            f"yaw err {res['err_yaw'][-1]:.4f} rad"
        )
        self.meshcat_track_btn.setEnabled(True)

    def _on_track_croc_finished(self, ok: bool, err: str, payload: object):
        self.run_track_btn.setEnabled(True)
        if hasattr(self, "reg_run_btn"):
            self.reg_run_btn.setEnabled(True)
        if not ok:
            self.log(err)
            QMessageBox.critical(self, "Error", err[:2000])
            return
        assert isinstance(payload, dict)
        res = payload["res"]
        self._render_tracking_figures(res, "direct")
        self.log(
            f"Croc tracking finished | EE error (final) {res['err'][-1]:.4f} m | "
            f"yaw err {res['err_yaw'][-1]:.4f} rad"
        )
        self.meshcat_track_btn.setEnabled(True)

    def _on_track_croc_ee_finished(self, ok: bool, err: str, payload: object):
        self.run_track_btn.setEnabled(True)
        if hasattr(self, "reg_run_btn"):
            self.reg_run_btn.setEnabled(True)
        if not ok:
            self.log(err)
            QMessageBox.critical(self, "Error", err[:2000])
            return
        assert isinstance(payload, dict)
        res = payload["res"]
        self._render_tracking_figures(res, "direct")
        self.log(
            f"Croc EE tracking finished | EE error (final) {res['err'][-1]:.4f} m | "
            f"yaw err {res['err_yaw'][-1]:.4f} rad"
        )
        self.meshcat_track_btn.setEnabled(True)

    def _on_track_ee_finished(self, ok: bool, err: str, out: object):
        self.run_track_btn.setEnabled(True)
        if hasattr(self, "reg_run_btn"):
            self.reg_run_btn.setEnabled(True)
        if not ok:
            self.log(err)
            QMessageBox.critical(self, "Error", err[:2000])
            return
        assert isinstance(out, dict)
        res = out["res"]
        cm = out["control_mode"]
        self._render_tracking_figures(res, cm, out)
        self.log(
            f"EE tracking finished | pos error (final) {res['err'][-1]:.4f} m | "
            f"yaw err {res['err_yaw'][-1]:.4f} rad"
        )
        self.meshcat_track_btn.setEnabled(True)

    def _render_tracking_figures(self, res: dict, control_mode: str, out: dict | None = None):
        self._last_track_res = res
        em = self._ee_mpc
        if em is None:
            self.fig_states.clear()
            ax = self.fig_states.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "States plot unavailable (missing s500_uam_ee_snap_tracking_mpc)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            self.fig_control.clear()
            axc = self.fig_control.add_subplot(111)
            axc.axis("off")
            self.cv_states.draw()
            self.cv_control.draw()

            self.fig_3d_track.clear()
            ax3 = self.fig_3d_track.add_subplot(111, projection="3d")
            ax3.text2D(0.2, 0.5, "3D plot unavailable", transform=ax3.transAxes)
            self.cv_3d_track.draw()

            self.fig_traj_dash.clear()
            ax = self.fig_traj_dash.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "MPC overview unavailable (missing s500_uam_ee_snap_tracking_mpc)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            self.cv_traj_dash.draw()
            self._render_cost_analysis_figure(res)
            return

        wp = None
        if out is not None and out.get("waypoints") is not None:
            wp = out["waypoints"]
        elif self._plan_bundle and self._plan_bundle.get("kind") == "ee_snap":
            wp = self._plan_bundle.get("waypoints")
        if self._is_s500_mode():
            wp = None
        ref_time_states = None
        ref_states = None
        ref_time_controls = None
        ref_controls = None
        pb = self._plan_bundle
        if pb is not None and pb.get("kind") in ("full_croc", "full_acados"):
            try:
                tp = np.asarray(pb.get("t_plan"), dtype=float).flatten()
                xp = np.asarray(pb.get("x_plan"), dtype=float)
                if tp.size >= 2 and xp.ndim == 2 and xp.shape[0] == tp.size:
                    ref_time_states = tp - tp[0]
                    ref_states = xp[:, :17] if xp.shape[1] > 17 else xp
                up = np.asarray(pb.get("u_plan"), dtype=float)
                if up.ndim == 2 and up.shape[0] > 0 and ref_time_states is not None:
                    n_u = int(up.shape[0])
                    if n_u == max(0, len(ref_time_states) - 1):
                        ref_time_controls = ref_time_states[:-1]
                    else:
                        ref_time_controls = np.linspace(
                            float(ref_time_states[0]),
                            float(ref_time_states[-1]),
                            n_u,
                        )
                    ref_controls = up
            except Exception:
                ref_time_states = None
                ref_states = None
                ref_time_controls = None
                ref_controls = None
        manual = self._manual_ref_overlay
        if isinstance(manual, dict):
            ref_time_states = manual.get("ref_time_states")
            ref_states = manual.get("ref_states")
            ref_time_controls = manual.get("ref_time_controls")
            ref_controls = manual.get("ref_controls")
            if wp is None:
                wp = manual.get("waypoints")
        fs = self.fig_states if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_into_figure else None
        f3 = self.fig_3d_track if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_3d_into_figure else None
        if fs is None:
            self.fig_states.clear()
            ax = self.fig_states.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "States plot unavailable\n(needs pinocchio and s500_uam_acados_trajectory)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
        if f3 is None:
            self.fig_3d_track.clear()
            ax = self.fig_3d_track.add_subplot(111, projection="3d")
            ax.text2D(0.2, 0.5, "3D plot unavailable", transform=ax.transAxes)
        plot_res = self._s500_plot_sanitize_res(res) if self._is_s500_mode() else res
        # s500：与规划页一致，仅机体 state/control + 基座 3D/摘要，不画 EE 总览图。
        if self._is_s500_mode():
            Xs = _ensure_uam17_for_s500_base_plot(np.asarray(plot_res["x"], dtype=float))
            tplt = np.asarray(plot_res["t"], dtype=float).flatten()
            if Xs.ndim == 2 and Xs.shape[0] == tplt.size:
                uplt = np.asarray(plot_res.get("u"), dtype=float)
                t_ref_plt = (
                    np.asarray(ref_time_states, dtype=float).flatten()
                    if ref_time_states is not None
                    else None
                )
                x_ref_plt = np.asarray(ref_states, dtype=float) if ref_states is not None else None
                t_u_ref_plt = (
                    np.asarray(ref_time_controls, dtype=float).flatten()
                    if ref_time_controls is not None
                    else None
                )
                u_ref_plt = np.asarray(ref_controls, dtype=float) if ref_controls is not None else None
                self._render_s500_base_only_planning_figures(
                    tplt,
                    Xs,
                    title_prefix="MPC closed-loop",
                    u_plan=uplt,
                    t_ref=t_ref_plt,
                    x_ref=x_ref_plt,
                    t_u_ref=t_u_ref_plt,
                    u_ref=u_ref_plt,
                    mpc_solve=plot_res.get("mpc_solve"),
                )
            else:
                self.fig_states.clear()
                ax = self.fig_states.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "s500 绘图：t 与 x 行数不一致",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                self.fig_3d_track.clear()
                ax3 = self.fig_3d_track.add_subplot(111, projection="3d")
                ax3.text2D(0.2, 0.5, "—", transform=ax3.transAxes)
                self.fig_traj_dash.clear()
                axd = self.fig_traj_dash.add_subplot(111)
                axd.axis("off")
                self.fig_control.clear()
                axc = self.fig_control.add_subplot(111)
                axc.axis("off")
        else:
            self._set_control_tab_uam_placeholder()
            em.render_ee_tracking_results_to_figures(
                plot_res,
                fs,
                f3,
                self.fig_traj_dash,
                control_mode=control_mode,
                plan_waypoints_xyz=wp,
                states_title="MPC closed-loop",
                ref_time_states=ref_time_states,
                ref_states=ref_states,
                ref_time_controls=ref_time_controls,
                ref_controls=ref_controls,
            )
        self.cv_states.draw()
        self.cv_control.draw()
        self.cv_3d_track.draw()
        self.cv_traj_dash.draw()
        self._render_cost_analysis_figure(res)

    def _render_cost_analysis_figure(self, res: dict) -> None:
        fig = self.fig_cost_analysis
        fig.clear()
        t = np.asarray(res.get("mpc_cost_t", []), dtype=float).flatten()
        total = np.asarray(res.get("mpc_cost_total", []), dtype=float).flatten()
        groups = res.get("mpc_cost_groups", {})
        terms = res.get("mpc_cost_terms", {})
        weights = res.get("mpc_cost_weights", {})

        has_total = bool(t.size and total.size == t.size and np.isfinite(total).any())
        term_keys = []
        if isinstance(terms, dict):
            for k in sorted(terms.keys()):
                v = np.asarray(terms.get(k, []), dtype=float).flatten()
                if v.size == t.size and v.size > 0 and np.isfinite(v).any():
                    term_keys.append(k)

        n_panels = (1 if has_total else 0) + len(term_keys)
        if n_panels <= 0:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No MPC cost breakdown available for this result",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            self.cv_cost_analysis.draw()
            return

        ncols = 2 if n_panels > 1 else 1
        nrows = int(math.ceil(float(n_panels) / float(ncols)))
        axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(n_panels)]
        idx = 0
        if has_total:
            ax = axes[idx]
            idx += 1
            ax.plot(t, total, "k-", lw=1.3)
            ax.set_title("total", fontsize=_mpl_pt(10))
            ax.set_xlabel("t [s]")
            ax.set_ylabel("cost")
            ax.grid(True, alpha=0.3)
        for key in term_keys:
            ax = axes[idx]
            idx += 1
            v = np.asarray(terms.get(key, []), dtype=float).flatten()
            ax.plot(t, v, lw=1.0, color="tab:orange")
            w = float(weights.get(key, float("nan"))) if isinstance(weights, dict) else float("nan")
            if np.isfinite(w):
                ax.set_title(f"{key} (w={w:g})", fontsize=_mpl_pt(10))
            else:
                ax.set_title(str(key), fontsize=_mpl_pt(10))
            ax.set_xlabel("t [s]")
            ax.set_ylabel("cost")
            ax.grid(True, alpha=0.3)

        fig.suptitle("MPC cost analysis (total + weighted term costs)", fontsize=_mpl_pt(12), y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.cv_cost_analysis.draw()


def _apply_display_scaling(app) -> float:
    """界面字体与 matplotlib 绘图字号缩放。

    Qt 控件默认 scale=1（可用 UAM_GUI_SCALE 覆盖）。
    绘图在 4K 屏上自动放大（可用 UAM_PLOT_SCALE 覆盖）。
    """
    global _MPL_FONT_SCALE

    env_scale = os.environ.get("UAM_GUI_SCALE", "").strip()
    if env_scale:
        try:
            gui_scale = max(0.8, min(3.0, float(env_scale)))
        except ValueError:
            gui_scale = 1.0
    else:
        gui_scale = 1.0

    plot_scale = _detect_plot_font_scale(app)
    _MPL_FONT_SCALE = plot_scale

    base_pt = 10.0
    f = app.font()
    f.setPointSizeF(base_pt * gui_scale)
    app.setFont(f)

    try:
        import matplotlib

        matplotlib.rcParams.update(
            {
                "font.size": 10.0 * plot_scale,
                "axes.titlesize": 11.0 * plot_scale,
                "axes.labelsize": 10.0 * plot_scale,
                "xtick.labelsize": 9.5 * plot_scale,
                "ytick.labelsize": 9.5 * plot_scale,
                "legend.fontsize": 9.0 * plot_scale,
            }
        )
    except Exception:
        pass

    return gui_scale


def _fit_window_to_screen(app, w) -> None:
    """让初始窗口不超过标准 1080p，并适配当前屏幕可用区域后居中显示。"""
    screen = app.primaryScreen()
    if screen is None:
        w.resize(1600, 900)
        return
    avail = screen.availableGeometry()
    # 逻辑像素上限：标准 1080p 显示器。
    cap_w, cap_h = 1920, 1080
    target_w = min(1600, cap_w, int(avail.width() * 0.96))
    target_h = min(900, cap_h, int(avail.height() * 0.94))
    w.setMaximumSize(min(avail.width(), cap_w), min(avail.height(), cap_h))
    w.resize(target_w, target_h)
    fg = w.frameGeometry()
    fg.moveCenter(avail.center())
    w.move(fg.topLeft())


def main():
    # High-DPI 适配必须在创建 QApplication 之前设置。
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    _icon = _app_icon()
    if _icon is not None:
        app.setWindowIcon(_icon)
    app.setStyle("Fusion")
    _apply_display_scaling(app)
    w = UamSuiteGUI()
    _fit_window_to_screen(app, w)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
