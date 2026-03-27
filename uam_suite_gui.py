#!/usr/bin/env python3
"""
S500 UAM integrated GUI: trajectory planning (full-state / EE-only) + closed-loop tracking (Crocoddyl along the plan / Acados EE-centric).

Single-page main plot on the right, "States + 3D": time-domain states on the left (dashed ref = plan, solid real = closed loop),
3D base/EE trajectory comparison on the right; an additional page "MPC / Error overview" reuses the EE tracking dashboard.

Usage:
  python uam_suite_gui.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

DEFAULT_PARAMS_PATH = Path(__file__).resolve().with_name("uam_suite_gui_params.json")


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


def _extract_x17(res: dict) -> np.ndarray:
    x = np.asarray(res["x"], dtype=float)
    n = min(17, x.shape[1])
    return x[:, :n].copy()


def _snap_default_rows() -> list[list[float]]:
    return [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.35, 0.95, 14.3, 2.5],
        [0.85, -0.15, 1.05, -22.9, 5.0],
        [1.0, 0.2, 0.9, 8.6, 8.0],
    ]


def _full_wp_default_rows() -> list[list[float]]:
    return [
        [0.0, 0.0, 1.0, 0.0, -68.8, -34.4, 0.0],
        [1.0, 0.5, 1.2, 45.0, -45.8, -17.2, 5.0],
    ]


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
            if mode == "eight":
                center = np.asarray(p["eight_center"], dtype=float).reshape(3)
                t_grid, p_ref, yaw_ref, _ = sample_ee_figure_eight_trajectory(
                    t_duration=float(p["t_duration"]),
                    dt_sample=float(p["dt_sample"]),
                    center=center,
                    semi_axis=float(p["eight_a"]),
                    period=float(p["eight_period"]),
                )
                payload = {
                    "kind": "ee_ref",
                    "track_kind": "eight",
                    "t_ref": t_grid,
                    "p_ref": p_ref,
                    "yaw_ref": yaw_ref,
                    "waypoints_xyz_yaw": None,
                    "t_wp": None,
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
                t_grid, p_ref, yaw_ref, _ = sample_ee_minimum_snap_trajectory(
                    wp, tw, float(p["dt_sample"])
                )
                payload = {
                    "kind": "ee_ref",
                    "track_kind": "snap",
                    "t_ref": t_grid,
                    "p_ref": p_ref,
                    "yaw_ref": yaw_ref,
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
                mpc_max_iter=p.get("mpc_max_iter", 60),
                use_thrust_constraints=p.get("use_thrust_constraints", True),
                use_actuator_first_order=p.get("use_actuator_first_order", True),
                tau_thrust=p.get("tau_thrust", 0.06),
                tau_theta=p.get("tau_theta", 0.05),
                s500_yaml_path=p.get("s500_yaml_path"),
                urdf_path=p.get("urdf_path"),
                verbose=False,
            )
            res = crocoddyl_closed_loop_to_ee_tracking_res(out)
            self.finished.emit(True, "", {"out": out, "res": res})
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
                w_pos=p.get("w_ee", 400.0),
                w_rot_yaw=p.get("w_ee_yaw", 200.0),
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
            mpc_stub = np.zeros(n_mpc, dtype=float)

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
                    "nlp_iter": np.zeros(n_mpc, dtype=int),
                    "cpu_s": mpc_stub.copy(),
                    "wall_s": mpc_stub,
                    "status": np.zeros(n_mpc, dtype=int),
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
        self._params_path: Path = DEFAULT_PARAMS_PATH

        try:
            from s500_uam_trajectory_gui import (
                ACADOS_AVAILABLE,
                CASCADE_TRAJ_AVAILABLE,
                CROCODDYL_AVAILABLE,
                OptimizationWorker,
                wp_to_state,
            )

            self._ACADOS_AVAILABLE = ACADOS_AVAILABLE
            self._CASCADE_TRAJ_AVAILABLE = CASCADE_TRAJ_AVAILABLE
            self._CROCODDYL_AVAILABLE = CROCODDYL_AVAILABLE
            self.OptimizationWorker = OptimizationWorker
            self._wp_to_state = wp_to_state
        except Exception as e:
            self._ACADOS_AVAILABLE = False
            self._CASCADE_TRAJ_AVAILABLE = False
            self._CROCODDYL_AVAILABLE = False
            self.OptimizationWorker = None
            self._wp_to_state = None
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
        self._init_croc_planner()
        self._build_ui()
        self._load_params_from_path(self._params_path, silent_if_missing=True)

    def _init_croc_planner(self):
        if not self._CROCODDYL_AVAILABLE:
            return
        try:
            from s500_uam_trajectory_planner import S500UAMTrajectoryPlanner

            self.planner = S500UAMTrajectoryPlanner()
        except Exception:
            self.planner = None

    def _robot_model_and_ee(self):
        if self._lazy_pin_planner is None:
            from s500_uam_trajectory_planner import S500UAMTrajectoryPlanner

            self._lazy_pin_planner = S500UAMTrajectoryPlanner()
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
        self.setWindowTitle("S500 UAM — Planning + Tracking Overview")
        self.resize(1680, 960)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left_tabs = QTabWidget()
        root.addWidget(left_tabs, stretch=0)

        # ----- Plan tab -----
        tab_plan = QWidget()
        plan_layout = QVBoxLayout(tab_plan)
        left_tabs.addTab(tab_plan, "Planning")

        self.plan_mode_combo = QComboBox()
        self.plan_mode_combo.addItems(["Full state (default)", "EE only (Minimum snap)"])
        self.plan_mode_combo.currentIndexChanged.connect(self._on_plan_mode)
        plan_layout.addWidget(QLabel("Planning type"))
        plan_layout.addWidget(self.plan_mode_combo)

        self.plan_stack = QStackedWidget()
        plan_layout.addWidget(self.plan_stack)

        # Stack 0: full state
        w_full = QWidget()
        g_full = QVBoxLayout(w_full)

        method_row = QHBoxLayout()
        self.method_combo = QComboBox()
        self._method_ids: list[str] = []
        if self._CROCODDYL_AVAILABLE:
            self.method_combo.addItem("Crocoddyl (BoxDDP)")
            self._method_ids.append("crocoddyl")
        if self._ACADOS_AVAILABLE:
            self.method_combo.addItem("Acados (thrusters + τ)")
            self._method_ids.append("acados")
            if self._CASCADE_TRAJ_AVAILABLE:
                self.method_combo.addItem("Acados (ω,T,θ + 1st-order)")
                self._method_ids.append("acados_cascade")
        if not self._method_ids:
            self.method_combo.addItem("(No solver available)")
            self._method_ids.append("none")
        method_row.addWidget(QLabel("Method"))
        method_row.addWidget(self.method_combo)
        g_full.addLayout(method_row)

        g_full.addWidget(QLabel("Waypoints [x,y,z m | yaw° | j1° | j2° | arrival time s]"))
        self.wp_table = QTableWidget(2, 7)
        self.wp_table.setHorizontalHeaderLabels(
            ["x", "y", "z", "yaw°", "j1°", "j2°", "t [s]"]
        )
        wp_header = self.wp_table.horizontalHeader()
        wp_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        wp_header.setStretchLastSection(False)
        wp_header.setMinimumSectionSize(56)
        for r, row in enumerate(_full_wp_default_rows()):
            for c, val in enumerate(row):
                self.wp_table.setItem(r, c, QTableWidgetItem(f"{val:g}"))
        g_full.addWidget(self.wp_table)
        wp_btn = QHBoxLayout()
        add_r = QPushButton("Add row")
        add_r.clicked.connect(self._add_wp_row)
        del_r = QPushButton("Delete last row")
        del_r.clicked.connect(self._del_wp_row)
        wp_btn.addWidget(add_r)
        wp_btn.addWidget(del_r)
        g_full.addLayout(wp_btn)

        cost_g = QGridLayout()
        self.dt_plan = QDoubleSpinBox()
        self.dt_plan.setRange(0.001, 0.5)
        self.dt_plan.setValue(0.02)
        self.max_iter_plan = QSpinBox()
        self.max_iter_plan.setRange(10, 2000)
        self.max_iter_plan.setValue(200)
        self.state_w = QDoubleSpinBox()
        self.state_w.setRange(1e-4, 1e4)
        self.state_w.setValue(1.0)
        self.ctrl_w = QDoubleSpinBox()
        self.ctrl_w.setRange(1e-8, 1.0)
        self.ctrl_w.setValue(1e-5)
        self.wp_mult = QDoubleSpinBox()
        self.wp_mult.setRange(1, 1e6)
        self.wp_mult.setValue(1000.0)
        self.plan_croc_use_actuator_first_order = QCheckBox("Enable")
        self.plan_croc_use_actuator_first_order.setChecked(True)
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
        cost_g.addWidget(QLabel("dt"), 0, 0)
        cost_g.addWidget(self.dt_plan, 0, 1)
        cost_g.addWidget(QLabel("max_iter"), 0, 2)
        cost_g.addWidget(self.max_iter_plan, 0, 3)
        cost_g.addWidget(QLabel("state_w"), 1, 0)
        cost_g.addWidget(self.state_w, 1, 1)
        cost_g.addWidget(QLabel("ctrl_w"), 1, 2)
        cost_g.addWidget(self.ctrl_w, 1, 3)
        cost_g.addWidget(QLabel("wp_mult"), 2, 0)
        cost_g.addWidget(self.wp_mult, 2, 1)
        cost_g.addWidget(QLabel("Croc actuator 1st-order"), 2, 2)
        cost_g.addWidget(self.plan_croc_use_actuator_first_order, 2, 3)
        cost_g.addWidget(QLabel("tau motor thrust [s]"), 3, 0)
        cost_g.addWidget(self.plan_tau_motor, 3, 1)
        cost_g.addWidget(QLabel("tau joint torque [s]"), 3, 2)
        cost_g.addWidget(self.plan_tau_joint, 3, 3)
        wg = QGroupBox("Full-state optimization parameters")
        wg.setLayout(cost_g)
        g_full.addWidget(wg)

        self.run_plan_btn = QPushButton("Run planning")
        self.run_plan_btn.clicked.connect(self._run_plan)
        g_full.addWidget(self.run_plan_btn)

        self.plan_stack.addWidget(w_full)

        # Stack 1: EE snap only
        w_ee = QWidget()
        g_ee = QVBoxLayout(w_ee)
        ee_type_row = QHBoxLayout()
        self.ee_plan_type_combo = QComboBox()
        self.ee_plan_type_combo.addItems(
            ["Minimum snap (waypoints)", "Figure-eight (figure-8)"]
        )
        ee_type_row.addWidget(QLabel("EE trajectory type"))
        ee_type_row.addWidget(self.ee_plan_type_combo)
        g_ee.addLayout(ee_type_row)
        g_ee.addWidget(QLabel("EE waypoints (x,y,z m, yaw°, time s) — consistent with the EE tracking GUI"))
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
        self.dt_ee_sample = QDoubleSpinBox()
        self.dt_ee_sample.setRange(0.005, 0.2)
        self.dt_ee_sample.setValue(0.02)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Sampling dt"))
        hl.addWidget(self.dt_ee_sample)
        g_ee.addLayout(hl)
        self.run_ee_plan_btn = QPushButton("Generate EE reference (from planning)")
        self.run_ee_plan_btn.clicked.connect(self._run_ee_plan)
        g_ee.addWidget(self.run_ee_plan_btn)
        self.ee_plan_type_combo.currentIndexChanged.connect(self._on_ee_plan_type_changed)
        self._on_ee_plan_type_changed()
        self.plan_stack.addWidget(w_ee)

        self._on_plan_mode()
        self.meshcat_plan_btn = QPushButton("Visualize planned trajectory (Meshcat)")
        self.meshcat_plan_btn.clicked.connect(self._visualize_planned_meshcat)
        self.meshcat_plan_btn.setEnabled(False)
        plan_layout.addWidget(self.meshcat_plan_btn)

        # ----- Track tab -----
        tab_track = QWidget()
        tk = QVBoxLayout(tab_track)
        left_tabs.addTab(tab_track, "Tracking")

        self.track_mode_combo = QComboBox()
        self.track_mode_combo.addItems(
            [
                "Crocoddyl — track along the full-state plan",
                "Acados — EE-centric tracking",
                "Crocoddyl — EE pose tracking",
            ]
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
        self.mpc_max_iter = QSpinBox()
        self.mpc_max_iter.setRange(1, 200)
        self.mpc_max_iter.setValue(20)
        self.mpc_log_iv = QSpinBox()
        self.mpc_log_iv.setRange(0, 1000)
        self.mpc_log_iv.setValue(0)
        self.control_mode_track = QComboBox()
        self.control_mode_track.addItems(["direct (thrust + τ)", "actuator_first_order (ω, T, θ)"])

        # Crocoddyl full-state tracking: optional first-order actuator response (different τ for thrust vs joint torques)
        # Note: Acados' actuator_first_order already includes a first-order actuator model; the τ below is only used for the Crocoddyl branch.
        self.tau_thrust_track = QDoubleSpinBox()
        self.tau_thrust_track.setRange(0.001, 2.0)
        self.tau_thrust_track.setDecimals(3)
        self.tau_thrust_track.setSingleStep(0.005)
        self.tau_thrust_track.setValue(0.06)

        self.tau_theta_track = QDoubleSpinBox()
        self.tau_theta_track.setRange(0.001, 2.0)
        self.tau_theta_track.setDecimals(3)
        self.tau_theta_track.setSingleStep(0.005)
        self.tau_theta_track.setValue(0.05)

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
        self.croc_use_actuator_first_order = QCheckBox("enable")
        self.croc_use_actuator_first_order.setChecked(True)
        self.croc_ee_use_thrust_constraints = QCheckBox("enable")
        self.croc_ee_use_thrust_constraints.setChecked(True)

        # Common simulation parameters (algorithm-independent)
        sim_g = QGridLayout()
        sim_g.addWidget(QLabel("T_sim [s]"), 0, 0)
        sim_g.addWidget(self.T_sim, 0, 1)
        sim_g.addWidget(QLabel("sim_dt"), 1, 0)
        sim_g.addWidget(self.sim_dt, 1, 1)
        sim_g.addWidget(QLabel("control_dt"), 2, 0)
        sim_g.addWidget(self.control_dt, 2, 1)
        sg = QGroupBox("Simulation parameters")
        sg.setLayout(sim_g)
        tk.addWidget(sg)

        tk.addWidget(QLabel("Tracking method"))
        tk.addWidget(self.track_mode_combo)

        # Algorithm-dependent parameters (single panel, dynamic visibility by algorithm)
        algo_grid = QGridLayout()
        self._algo_rows: list[tuple[QLabel, QWidget]] = []
        for r, (lab, w) in enumerate(
            [
                ("dt_mpc", self.dt_mpc),
                ("N (horizon)", self.N_mpc),
                ("w_ee", self.w_ee),
                ("w_ee_yaw", self.w_ee_yaw),
                ("mpc max_iter", self.mpc_max_iter),
                ("mpc log ivl", self.mpc_log_iv),
                ("Control mode", self.control_mode_track),
                ("Croc horizon steps", self.croc_horizon),
                ("Croc MPC max_iter", self.croc_mpc_iter),
                ("tau_thrust_track [s]", self.tau_thrust_track),
                ("tau_theta_track [s]", self.tau_theta_track),
                ("w_state_track", self.w_state_track),
                ("w_state_reg", self.w_state_reg),
                ("w_control", self.w_control),
                ("w_terminal_track", self.w_terminal_track),
                ("use actuator 1st-order", self.croc_use_actuator_first_order),
                ("use thrust constraints", self.croc_ee_use_thrust_constraints),
            ]
        ):
            lb = QLabel(lab)
            algo_grid.addWidget(lb, r, 0)
            algo_grid.addWidget(w, r, 1)
            self._algo_rows.append((lb, w))

        self.track_algo_group = QGroupBox("Algorithm parameters")
        algo_wrap = QVBoxLayout()
        algo_wrap.addLayout(algo_grid)
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

        params_btns = QHBoxLayout()
        self.save_params_btn = QPushButton("Save parameters")
        self.save_params_btn.clicked.connect(self._save_params)
        self.save_params_as_btn = QPushButton("Save parameters as")
        self.save_params_as_btn.clicked.connect(self._save_params_as)
        params_btns.addWidget(self.save_params_btn)
        params_btns.addWidget(self.save_params_as_btn)
        tk.addLayout(params_btns)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(140)
        tk.addWidget(self.log_text)

        # ----- Right: plots -----
        right = QTabWidget()
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

        self.fig_states, self.cv_states = embed_fig("States / controls", (12, 9))
        self.fig_3d_track, self.cv_3d_track = embed_fig("Base 3D", (10, 8))
        self.fig_traj_dash, self.cv_traj_dash = embed_fig("Tracking / MPC", (12, 10))
        # Backward-compatible aliases for existing planning preview rendering.
        self.fig_combined, self.cv_combined = self.fig_states, self.cv_states

        if self._import_err:
            self.log(f"trajectory_gui import warning: {self._import_err!r}")
        if not self._EE_MPC_OK:
            self.log("EE MPC (Acados) is unavailable: EE-centric tracking will fail.")
        if not self._CROC_EE_OK:
            self.log("Crocoddyl EE pose tracking is unavailable.")

    def log(self, msg: str) -> None:
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

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

        wp_xyz = None
        fs = self.fig_states if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_into_figure else None
        f3 = self.fig_3d_track if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_3d_into_figure else None

        # Render into the same 3 figures as the tracking GUI.
        em.render_ee_tracking_results_to_figures(
            res_ref,
            fs,
            f3,
            self.fig_traj_dash,
            control_mode="direct",
            plan_waypoints_xyz=wp_xyz,
            states_title="Planned reference",
        )
        self.cv_states.draw()
        self.cv_3d_track.draw()
        self.cv_traj_dash.draw()

    def _render_planning_reference_ee_snap(self) -> None:
        """Render the EE-only (minimum-snap) planning reference using the same dashboard framework."""
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

        # Construct a plausible robot state sequence by:
        # - using a fixed nominal joint configuration
        # - setting base yaw from yaw_ref
        # - translating the base so that FK EE position matches p_ref at each time
        rm, _ = self._robot_model_and_ee()
        nq = int(getattr(rm, "nq", 0))
        nv = int(getattr(rm, "nv", 0))
        if nq <= 0 or nv <= 0:
            raise ValueError(f"Invalid pinocchio model dims: nq={nq}, nv={nv}")

        from s500_uam_trajectory_planner import make_uam_state
        from s500_uam_ee_snap_tracking_mpc import align_uam_state_ee_to_world_position

        x_nom_base = np.asarray(make_uam_state(0.0, 0.0, 1.0, j1=0.0, j2=0.0, yaw=0.0), dtype=float)
        x_nom = x_nom_base.copy()

        xs: list[np.ndarray] = []
        for i in range(n):
            xi = x_nom.copy()
            # Update quaternion yaw component (indices follow make_uam_state convention).
            # We simply rebuild full state for each time for clarity.
            xi = make_uam_state(
                float(x_nom[0]),
                float(x_nom[1]),
                float(x_nom[2]),
                j1=0.0,
                j2=0.0,
                yaw=float(yaw_ref_u[i]),
            )
            xi_aligned = align_uam_state_ee_to_world_position(
                xi, rm, p_ref[i], nq=nq, nv=nv
            )
            xs.append(np.asarray(xi_aligned, dtype=float).flatten())

        simX = np.asarray(xs, dtype=float)

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

        wp_xyz = pb.get("waypoints")

        em.render_ee_tracking_results_to_figures(
            res_ref,
            self.fig_states if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_into_figure else None,
            self.fig_3d_track if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_3d_into_figure else None,
            self.fig_traj_dash,
            control_mode="direct",
            plan_waypoints_xyz=wp_xyz,
            states_title="Planned reference (EE-only)",
        )
        self.cv_states.draw()
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
                fontsize=12,
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

        t_ref = base_ref = X_r = ee_ref = None
        if pb["kind"] in ("full_croc", "full_acados"):
            tp = np.asarray(pb["t_plan"], dtype=float).flatten()
            t_ref = tp - tp[0]
            X_r = np.asarray(pb["x_plan"], dtype=float)
            if X_r.shape[1] > 17:
                X_r = X_r[:, :17]
            base_ref = X_r[:, :3]
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
            ee_ref = np.asarray(pb["p_ref"], dtype=float)

        t_m = X_m = base_m = ee_m = None
        if has_real:
            assert res is not None
            t_m = np.asarray(res["t"], dtype=float).flatten()
            X_m = _extract_x17(res)
            base_m = X_m[:, :3]
            ee_m = np.asarray(res["ee"], dtype=float)

        def _style_leg(ax):
            ax.legend(loc="upper right", fontsize=6, framealpha=0.88, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=8)

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
        ax.set_title("Base position", fontsize=9)
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
        ax.set_title("Base orientation (Euler ZYX)", fontsize=9)
        _style_leg(ax)

        # 2: EE position
        ax = axes[2]
        if ee_ref is not None:
            ax.plot(t_ref, ee_ref[:, 0], "r--", lw=1.0, alpha=0.9, label="ref x")
            ax.plot(t_ref, ee_ref[:, 1], "g--", lw=1.0, alpha=0.9, label="ref y")
            ax.plot(t_ref, ee_ref[:, 2], "b--", lw=1.0, alpha=0.9, label="ref z")
        if ee_m is not None:
            ax.plot(t_m, ee_m[:, 0], "r-", lw=1.1, label="real x")
            ax.plot(t_m, ee_m[:, 1], "g-", lw=1.1, label="real y")
            ax.plot(t_m, ee_m[:, 2], "b-", lw=1.1, label="real z")
        ax.set_xlabel("t [s]", **tinfo)
        ax.set_ylabel("m", **tinfo)
        ax.set_title("EE position (FK ref / meas. real)", fontsize=9)
        _style_leg(ax)

        # 3: arm joints
        ax = axes[3]
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
        ax.set_ylabel("deg", **tinfo)
        ax.set_title("Arm joints", fontsize=9)
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
        if pb.get("kind") == "ee_snap" and pb.get("waypoints") is not None:
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
        ax3d.set_title("3D: ref (dashed) · real (solid)", fontsize=10)
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
            ax.tick_params(axis="both", labelsize=8)
        ax3d.tick_params(axis="both", labelsize=8)
        subt = "(only ref)" if not has_real else "(ref + real)"
        fig.suptitle(f"Plan ref (dashed) · closed-loop real (solid) {subt}", fontsize=12, y=0.98)

    def _on_plan_mode(self):
        self.plan_stack.setCurrentIndex(self.plan_mode_combo.currentIndex())

    def _on_ee_plan_type_changed(self):
        snap = self.ee_plan_type_combo.currentIndex() == 0
        self.ee_wp_table.setVisible(snap)
        self.ee_eight_group.setVisible(not snap)

    def _on_track_mode_changed(self):
        idx = int(self.track_mode_combo.currentIndex())
        if not hasattr(self, "track_algo_group"):
            return
        visible_widgets = {self.dt_mpc}
        if idx == 0:
            visible_widgets.update(
                {
                    self.croc_horizon,
                    self.croc_mpc_iter,
                    self.tau_thrust_track,
                    self.tau_theta_track,
                    self.w_state_track,
                    self.w_state_reg,
                    self.w_control,
                    self.w_terminal_track,
                    self.croc_use_actuator_first_order,
                }
            )
        elif idx == 1:
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
                    self.w_ee,
                    self.w_ee_yaw,
                    self.mpc_max_iter,
                    self.croc_ee_use_thrust_constraints,
                }
            )
        for lb, w in getattr(self, "_algo_rows", []):
            show = w in visible_widgets
            lb.setVisible(show)
            w.setVisible(show)
        if idx == 0:
            self.track_algo_group.setTitle(
                "Algorithm parameters (Crocoddyl full-state tracking)"
            )
        elif idx == 1:
            self.track_algo_group.setTitle(
                "Algorithm parameters (Acados EE-centric tracking)"
            )
        else:
            self.track_algo_group.setTitle(
                "Algorithm parameters (Crocoddyl EE pose tracking)"
            )

    def _set_table_from_rows(self, table: QTableWidget, rows: list[list[float]], n_cols: int) -> None:
        table.setRowCount(max(0, len(rows)))
        for r, row in enumerate(rows):
            for c in range(n_cols):
                v = float(row[c]) if c < len(row) else 0.0
                table.setItem(r, c, QTableWidgetItem(f"{v:g}"))

    def _collect_params(self) -> dict:
        return {
            "version": 1,
            "plan_mode_index": int(self.plan_mode_combo.currentIndex()),
            "method_index": int(self.method_combo.currentIndex()),
            "track_mode_index": int(self.track_mode_combo.currentIndex()),
            "control_mode_track_index": int(self.control_mode_track.currentIndex()),
            "wp_rows": self._read_wp_table(),
            "ee_wp_rows": self._read_ee_rows(),
            "dt_plan": float(self.dt_plan.value()),
            "max_iter_plan": int(self.max_iter_plan.value()),
            "state_w": float(self.state_w.value()),
            "ctrl_w": float(self.ctrl_w.value()),
            "wp_mult": float(self.wp_mult.value()),
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
            "T_sim": float(self.T_sim.value()),
            "sim_dt": float(self.sim_dt.value()),
            "control_dt": float(self.control_dt.value()),
            "dt_mpc": float(self.dt_mpc.value()),
            "N_mpc": int(self.N_mpc.value()),
            "w_ee": float(self.w_ee.value()),
            "w_ee_yaw": float(self.w_ee_yaw.value()),
            "mpc_max_iter": int(self.mpc_max_iter.value()),
            "mpc_log_iv": int(self.mpc_log_iv.value()),
            "tau_thrust_track": float(self.tau_thrust_track.value()),
            "tau_theta_track": float(self.tau_theta_track.value()),
            "croc_horizon": int(self.croc_horizon.value()),
            "croc_mpc_iter": int(self.croc_mpc_iter.value()),
            "w_state_track": float(self.w_state_track.value()),
            "w_state_reg": float(self.w_state_reg.value()),
            "w_control": float(self.w_control.value()),
            "w_terminal_track": float(self.w_terminal_track.value()),
            "croc_use_actuator_first_order": bool(self.croc_use_actuator_first_order.isChecked()),
            "croc_ee_use_thrust_constraints": bool(self.croc_ee_use_thrust_constraints.isChecked()),
            "plan_croc_use_actuator_first_order": bool(self.plan_croc_use_actuator_first_order.isChecked()),
            "plan_tau_motor": float(self.plan_tau_motor.value()),
            "plan_tau_joint": float(self.plan_tau_joint.value()),
        }

    def _apply_params(self, p: dict) -> None:
        if not isinstance(p, dict):
            raise ValueError("Parameter file format is invalid (root must be a JSON object).")

        if isinstance(p.get("wp_rows"), list):
            self._set_table_from_rows(self.wp_table, p["wp_rows"], 7)
        if isinstance(p.get("ee_wp_rows"), list):
            self._set_table_from_rows(self.ee_wp_table, p["ee_wp_rows"], 5)

        def _set_spin(name: str, widget):
            if name in p:
                widget.setValue(p[name])

        _set_spin("dt_plan", self.dt_plan)
        _set_spin("max_iter_plan", self.max_iter_plan)
        _set_spin("state_w", self.state_w)
        _set_spin("ctrl_w", self.ctrl_w)
        _set_spin("wp_mult", self.wp_mult)
        _set_spin("dt_ee_sample", self.dt_ee_sample)
        _set_spin("ee_eight_a", self.ee_eight_a)
        _set_spin("ee_eight_period", self.ee_eight_period)
        _set_spin("ee_eight_tdur", self.ee_eight_tdur)
        _set_spin("T_sim", self.T_sim)
        _set_spin("sim_dt", self.sim_dt)
        _set_spin("control_dt", self.control_dt)
        _set_spin("dt_mpc", self.dt_mpc)
        _set_spin("N_mpc", self.N_mpc)
        _set_spin("w_ee", self.w_ee)
        _set_spin("w_ee_yaw", self.w_ee_yaw)
        _set_spin("mpc_max_iter", self.mpc_max_iter)
        _set_spin("mpc_log_iv", self.mpc_log_iv)
        _set_spin("tau_thrust_track", self.tau_thrust_track)
        _set_spin("tau_theta_track", self.tau_theta_track)
        _set_spin("croc_horizon", self.croc_horizon)
        _set_spin("croc_mpc_iter", self.croc_mpc_iter)
        _set_spin("w_state_track", self.w_state_track)
        _set_spin("w_state_reg", self.w_state_reg)
        _set_spin("w_control", self.w_control)
        _set_spin("w_terminal_track", self.w_terminal_track)
        _set_spin("plan_tau_motor", self.plan_tau_motor)
        _set_spin("plan_tau_joint", self.plan_tau_joint)
        # Backward compatibility with earlier naming.
        if "plan_tau_motor" not in p and "plan_tau_thrust" in p:
            self.plan_tau_motor.setValue(float(p["plan_tau_thrust"]))
        if "plan_tau_joint" not in p and "plan_tau_theta" in p:
            self.plan_tau_joint.setValue(float(p["plan_tau_theta"]))

        def _set_combo(name: str, widget):
            if name in p:
                idx = int(p[name])
                if 0 <= idx < widget.count():
                    widget.setCurrentIndex(idx)

        def _set_check(name: str, widget: QCheckBox):
            if name in p:
                widget.setChecked(bool(p[name]))

        _set_combo("plan_mode_index", self.plan_mode_combo)
        _set_combo("ee_plan_type_index", self.ee_plan_type_combo)
        _set_combo("method_index", self.method_combo)
        _set_combo("track_mode_index", self.track_mode_combo)
        _set_combo("control_mode_track_index", self.control_mode_track)
        _set_check("croc_use_actuator_first_order", self.croc_use_actuator_first_order)
        _set_check("croc_ee_use_thrust_constraints", self.croc_ee_use_thrust_constraints)
        _set_check("plan_croc_use_actuator_first_order", self.plan_croc_use_actuator_first_order)
        ec = p.get("ee_eight_center")
        if isinstance(ec, list) and len(ec) >= 3:
            self.ee_eight_cx.setValue(float(ec[0]))
            self.ee_eight_cy.setValue(float(ec[1]))
            self.ee_eight_cz.setValue(float(ec[2]))
        self._on_plan_mode()
        self._on_ee_plan_type_changed()
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

    def _read_wp_table(self) -> list[list[float]]:
        rows = []
        for r in range(self.wp_table.rowCount()):
            row = []
            for c in range(7):
                it = self.wp_table.item(r, c)
                row.append(float(it.text()) if it else 0.0)
            rows.append(row)
        return rows

    def _add_wp_row(self):
        r = self.wp_table.rowCount()
        self.wp_table.insertRow(r)
        for c in range(7):
            self.wp_table.setItem(r, c, QTableWidgetItem("0"))

    def _del_wp_row(self):
        if self.wp_table.rowCount() > 2:
            self.wp_table.removeRow(self.wp_table.rowCount() - 1)

    def _read_ee_rows(self) -> list[list[float]]:
        rows = []
        for r in range(self.ee_wp_table.rowCount()):
            row = []
            for c in range(5):
                it = self.ee_wp_table.item(r, c)
                row.append(float(it.text()) if it else 0.0)
            rows.append(row)
        return rows

    def _sorted_full_wps_and_durations(self, rows: list[list[float]]):
        wps = sorted(rows, key=lambda x: float(x[6]))
        durs = []
        for i in range(len(wps) - 1):
            d = float(wps[i + 1][6]) - float(wps[i][6])
            durs.append(d if d > 0 else 1.0)
        return wps, durs

    def _run_plan(self):
        if self.plan_mode_combo.currentIndex() == 1:
            return
        if self.OptimizationWorker is None or self._wp_to_state is None:
            QMessageBox.warning(self, "Error", "Unable to import trajectory_gui / solver.")
            return
        rows = self._read_wp_table()
        wps, durs = self._sorted_full_wps_and_durations(rows)
        if len(wps) < 2:
            QMessageBox.warning(self, "Error", "At least 2 waypoints are required.")
            return
        mid = self.method_combo.currentIndex()
        method = self._method_ids[mid] if mid < len(self._method_ids) else "none"
        if method == "none":
            QMessageBox.warning(self, "Error", "No available solver.")
            return
        if method == "crocoddyl" and self.planner is None:
            QMessageBox.warning(self, "Error", "Crocoddyl planner is not initialized.")
            return
        params = {
            "waypoints": wps,
            "durations": durs,
            "dt": self.dt_plan.value(),
            "max_iter": self.max_iter_plan.value(),
            "state_weight": self.state_w.value(),
            "control_weight": self.ctrl_w.value(),
            "waypoint_multiplier": self.wp_mult.value(),
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
            "use_actuator_first_order": bool(
                method == "crocoddyl" and self.plan_croc_use_actuator_first_order.isChecked()
            ),
        }
        self.run_plan_btn.setEnabled(False)
        self.log(f"Planning started: {method}, {len(wps)} waypoints")
        self._plan_worker = self.OptimizationWorker(method, params)
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
            urdf_path = str(Path(__file__).resolve().parent / "models" / "urdf" / "s500_uam_simple.urdf")
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
            QMessageBox.warning(self, "Notice", "EE-only planning has no full robot state trajectory for Meshcat playback.")
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
        self.run_plan_btn.setEnabled(True)
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
            self._plan_bundle = {
                "kind": "full_croc",
                "t_plan": t_plan,
                "x_plan": x_plan,
                "u_plan": np.vstack(us) if len(us) else np.zeros((0, 6), dtype=float),
            }
        elif result_data.get("method") in ("acados", "acados_cascade"):
            t_plan = np.asarray(result_data["time_arr"], dtype=float).flatten()
            x_plan = np.asarray(result_data["simX"], dtype=float)
            u_plan = np.asarray(result_data.get("simU"), dtype=float)
            self._plan_bundle = {
                "kind": "full_acados",
                "t_plan": t_plan,
                "x_plan": x_plan,
                "u_plan": u_plan if u_plan.ndim == 2 else np.zeros((0, 6), dtype=float),
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
        self.log("Planning finished. You can run the closed loop on the \"Tracking\" tab.")

    def _run_ee_plan(self):
        self.run_ee_plan_btn.setEnabled(False)
        self.log("Generating EE reference…")
        if self.ee_plan_type_combo.currentIndex() == 0:
            params = {
                "mode": "snap",
                "rows": self._read_ee_rows(),
                "dt_sample": self.dt_ee_sample.value(),
            }
        else:
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
        self._plan_worker = EeRefPlanWorker(params)
        self._plan_worker.finished.connect(self._on_ee_plan_finished)
        self._plan_worker.start()

    def _on_ee_plan_finished(self, ok: bool, err: str, payload: object):
        self.run_ee_plan_btn.setEnabled(True)
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
            "waypoints": payload["waypoints_xyz_yaw"],
            "t_wp": payload["t_wp"],
        }
        self._last_track_res = None
        self._redraw_combined_views(None)
        self._update_track_mode_enabled()
        self.run_track_btn.setEnabled(True)
        self.meshcat_plan_btn.setEnabled(True)
        self.meshcat_track_btn.setEnabled(False)
        self.log("EE reference generated. We recommend Acados EE-centric tracking.")

    def _update_track_mode_enabled(self):
        """Crocoddyl tracking along the trajectory is selectable only when full-state planning is available."""
        if self._plan_bundle is None:
            return
        full = self._plan_bundle["kind"] in ("full_croc", "full_acados")
        try:
            it = self.track_mode_combo.model().item(0)
            if it is not None:
                it.setEnabled(full)
            it2 = self.track_mode_combo.model().item(2)
            if it2 is not None:
                it2.setEnabled(self._CROC_EE_OK)
        except Exception:
            pass
        if not full and self.track_mode_combo.currentIndex() == 0:
            self.track_mode_combo.setCurrentIndex(1)
        self._on_track_mode_changed()

    def _run_track(self):
        if self._plan_bundle is None:
            return
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
                "use_actuator_first_order": self.croc_use_actuator_first_order.isChecked(),
                "tau_thrust": float(self.tau_thrust_track.value()),
                "tau_theta": float(self.tau_theta_track.value()),
            }
            self.run_track_btn.setEnabled(False)
            self.log("Crocoddyl closed-loop tracking along the plan…")
            self._track_worker = TrackCrocAlongPlanWorker(params)
            self._track_worker.finished.connect(self._on_track_croc_finished)
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
            if pb.get("ee_track_kind") == "eight":
                plan_title = "EE figure-eight (plan ref)"
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

        if mode == 2:
            if not self._CROC_EE_OK or self._croc_ee_mpc is None:
                QMessageBox.warning(self, "Error", "Crocoddyl EE pose tracking is unavailable.")
                return
            params_croc_ee = {
                "x0": x0_for_ee,
                "t_ref": t_ref,
                "p_ref": p_ref,
                "yaw_ref": yaw_ref,
                "sim_dt": sim_dt,
                "control_dt": self.control_dt.value(),
                "dt_mpc": self.dt_mpc.value(),
                "N_mpc": self.N_mpc.value(),
                "w_ee": self.w_ee.value(),
                "w_ee_yaw": self.w_ee_yaw.value(),
                "mpc_max_iter": self.mpc_max_iter.value(),
                "use_thrust_constraints": self.croc_ee_use_thrust_constraints.isChecked(),
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

    def _on_track_croc_finished(self, ok: bool, err: str, payload: object):
        self.run_track_btn.setEnabled(True)
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
            self.cv_states.draw()

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
            return

        wp = None
        if out is not None and out.get("waypoints") is not None:
            wp = out["waypoints"]
        elif self._plan_bundle and self._plan_bundle.get("kind") == "ee_snap":
            wp = self._plan_bundle.get("waypoints")
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
        em.render_ee_tracking_results_to_figures(
            res,
            fs,
            f3,
            self.fig_traj_dash,
            control_mode=control_mode,
            plan_waypoints_xyz=wp,
            states_title="MPC closed-loop",
        )
        self.cv_states.draw()
        self.cv_3d_track.draw()
        self.cv_traj_dash.draw()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = UamSuiteGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
