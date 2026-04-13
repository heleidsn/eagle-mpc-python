#!/usr/bin/env python3
"""
S500 UAM Trajectory Planning GUI

A graphical interface for S500 UAM trajectory optimization:
- Supports Crocoddyl and Acados (direct thrust/torque, or ω+T+θ with cascaded 1st-order dynamics)
- Waypoint add/remove/update (like tvc_traj_opt_gui)
- Task selection: Point-to-point or Grasp
- Cost parameter tuning
- Plot visualization

Usage:
    python s500_uam_trajectory_gui.py

Requires: PyQt5, matplotlib, numpy, pinocchio, crocoddyl
Optional: acados_template (for Acados method)
"""

import sys
import os
import json
from pathlib import Path

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QTextEdit, QTabWidget, QScrollArea, QGridLayout,
    QSplitter, QFrame, QMessageBox, QFileDialog, QListWidget, QListWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    from s500_uam_trajectory_planner import (
        S500UAMTrajectoryPlanner,
        make_uam_state,
    )
    CROCODDYL_AVAILABLE = True
except ImportError:
    S500UAMTrajectoryPlanner = None
    make_uam_state = None
    CROCODDYL_AVAILABLE = False

try:
    from s500_uam_acados_trajectory import (
        run_simple_trajectory,
        run_multiwaypoint_trajectory,
        run_simple_trajectory_cascade,
        run_multiwaypoint_trajectory_cascade,
        plot_acados_into_figure,
        plot_acados_3d_into_figure,
        ACADOS_AVAILABLE,
        CASCADE_TRAJ_AVAILABLE,
    )
except (ImportError, Exception):
    run_simple_trajectory = None
    run_multiwaypoint_trajectory = None
    run_simple_trajectory_cascade = None
    run_multiwaypoint_trajectory_cascade = None
    plot_acados_into_figure = None
    plot_acados_3d_into_figure = None
    ACADOS_AVAILABLE = False
    CASCADE_TRAJ_AVAILABLE = False

# Fixed path for saving/loading parameters
DEFAULT_PARAMS_PATH = Path(__file__).parent.parent / "config" / "yaml" / "trajectories" / "s500_uam_trajectory_params.json"


def mixed_wp_row_kind(cell0) -> str:
    """Parse full-state waypoint Type cell.

    Returns:
        ``base`` — x,y,z = base origin (m); A,B,C = j1°, j2°, base yaw°.
        ``ee_pose`` — x,y,z = EE position (world); A,B,C = roll°, pitch°, yaw° (ZYX, Pinocchio).
        ``ee_pos`` — x,y,z = EE position (world); A,B,C = j1°, j2°, yaw° seeds for alignment.
    """
    s = str(cell0).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if s.startswith("eep") or s in ("eepos", "eeposition"):
        return "ee_pos"
    if s.startswith("e"):
        return "ee_pose"
    return "base"


def wp_to_state(wp):
    """Convert waypoint [x,y,z,j1_deg,j2_deg,yaw_deg,time] to 17-dim state."""
    if make_uam_state is None:
        return None
    deg2rad = np.pi / 180.0
    return make_uam_state(
        wp[0], wp[1], wp[2],
        j1=wp[3] * deg2rad, j2=wp[4] * deg2rad,
        yaw=wp[5] * deg2rad,
    )


class OptimizationWorker(QThread):
    """Background worker for trajectory optimization (Crocoddyl or Acados)."""
    finished = pyqtSignal(bool, str, object)  # converged, error_msg, result_data

    def __init__(self, method, params):
        super().__init__()
        self.method = method  # "crocoddyl" | "acados" | "acados_cascade"
        self.params = params

    def run(self):
        try:
            if self.method == "crocoddyl":
                self._run_crocoddyl()
            elif self.method in ("acados", "acados_cascade"):
                self._run_acados()
        except Exception as e:
            import traceback
            self.finished.emit(False, traceback.format_exc(), None)

    def _run_crocoddyl(self):
        if S500UAMTrajectoryPlanner is None or make_uam_state is None:
            self.finished.emit(False, "Crocoddyl not available. Install pinocchio, crocoddyl.", None)
            return
        planner = self.params.get("planner")
        if planner is None:
            self.finished.emit(False, "Planner not initialized.", None)
            return
        if self.params.get("grasp"):
            planner.create_trajectory_problem(
                start_state=self.params["start_state"],
                grasp_position=self.params["grasp_position"],
                target_state=self.params["target_state"],
                durations=self.params["durations"],
                dt=self.params["dt"],
                grasp_ee_weight=self.params.get("grasp_ee_weight", 5000),
                waypoint_multiplier=self.params.get("waypoint_multiplier", 1000),
                state_weight=self.params.get("state_weight", 1.0),
                control_weight=self.params.get("control_weight", 1e-5),
                use_actuator_first_order=self.params.get("use_actuator_first_order", False),
                tau_cmd=self.params.get("tau_cmd"),
            )
        else:
            mixed_rows = self.params.get("mixed_wp_rows")
            if mixed_rows is not None and len(mixed_rows) >= 2 and make_uam_state is not None:
                import pinocchio as pin

                deg2rad = np.pi / 180.0
                sorted_rows = sorted(mixed_rows, key=lambda r: float(r[7]))
                modes: list[str] = []
                resolved: list = []
                ee_targets: list = []
                ee_pose_rpy_world: list = []
                durs = []
                for row in sorted_rows:
                    rk = mixed_wp_row_kind(row[0])
                    x, y, z, a, b, c, t = (float(row[i]) for i in range(1, 8))
                    if rk == "base":
                        modes.append("base")
                        st = wp_to_state([x, y, z, a, b, c, t])
                        if st is None:
                            self.finished.emit(False, "make_uam_state not available.", None)
                            return
                        resolved.append(st)
                        ee_targets.append(None)
                        ee_pose_rpy_world.append(None)
                    elif rk == "ee_pos":
                        modes.append("ee_pos")
                        st0 = make_uam_state(
                            0.0, 0.0, 1.0, j1=a * deg2rad, j2=b * deg2rad, yaw=c * deg2rad
                        )
                        st = planner.align_state_ee_to_world_point(
                            st0, np.array([x, y, z], dtype=float)
                        )
                        resolved.append(st)
                        ee_targets.append(np.array([x, y, z], dtype=float))
                        ee_pose_rpy_world.append(None)
                    else:
                        modes.append("ee_pose")
                        st0 = np.zeros(17)
                        st0[2] = 1.0
                        rpy = np.array([a, b, c], dtype=float) * deg2rad
                        R = pin.rpy.rpyToMatrix(
                            float(rpy[0]), float(rpy[1]), float(rpy[2])
                        )
                        quat = pin.Quaternion(R)
                        st0[3], st0[4], st0[5], st0[6] = quat.x, quat.y, quat.z, quat.w
                        st = planner.align_state_ee_to_world_point(
                            st0, np.array([x, y, z], dtype=float)
                        )
                        resolved.append(st)
                        ee_targets.append(np.array([x, y, z], dtype=float))
                        ee_pose_rpy_world.append(rpy.copy())
                for i in range(len(sorted_rows) - 1):
                    d = float(sorted_rows[i + 1][7]) - float(sorted_rows[i][7])
                    durs.append(d if d > 1e-6 else 1.0)
                planner.create_trajectory_problem_mixed_waypoints(
                    modes=modes,
                    resolved_states=resolved,
                    ee_targets=ee_targets,
                    durations=durs,
                    dt=self.params["dt"],
                    waypoint_multiplier=self.params.get("waypoint_multiplier", 1000),
                    state_weight=self.params.get("state_weight", 1.0),
                    control_weight=self.params.get("control_weight", 1e-5),
                    ee_knot_weight=self.params.get("ee_knot_weight", 5000.0),
                    ee_knot_state_reg_weight=self.params.get("ee_knot_state_reg_weight", 0.0),
                    ee_pose_rpy_world=ee_pose_rpy_world,
                    ee_knot_rotation_weight=float(
                        self.params.get("ee_knot_rotation_weight", 0.0)
                    ),
                    ee_knot_velocity_weight=float(
                        self.params.get("ee_knot_velocity_weight", 200.0)
                    ),
                    ee_knot_velocity_pitch_weight=float(
                        self.params.get("ee_knot_velocity_pitch_weight", 0.0)
                    ),
                    use_actuator_first_order=self.params.get("use_actuator_first_order", False),
                    tau_cmd=self.params.get("tau_cmd"),
                )
            else:
                waypoints = self.params.get("waypoints", [])
                durations = self.params.get("durations", [])
                if len(waypoints) < 2:
                    self.finished.emit(False, "Need at least 2 waypoints.", None)
                    return
                states = [wp_to_state(wp) for wp in waypoints]
                if any(s is None for s in states):
                    self.finished.emit(False, "make_uam_state not available.", None)
                    return
                planner.create_trajectory_problem_waypoints(
                    waypoints=states,
                    durations=durations,
                    dt=self.params["dt"],
                    waypoint_multiplier=self.params.get("waypoint_multiplier", 1000),
                    state_weight=self.params.get("state_weight", 1.0),
                    control_weight=self.params.get("control_weight", 1e-5),
                    use_actuator_first_order=self.params.get("use_actuator_first_order", False),
                    tau_cmd=self.params.get("tau_cmd"),
                )
        import time as _time
        t0 = _time.perf_counter()
        converged = planner.solve_trajectory(max_iter=self.params.get("max_iter", 200), verbose=False)
        elapsed = _time.perf_counter() - t0
        n_iter = int(planner.solver.iter) if planner.solver else 0
        timing = {
            "n_iter": n_iter,
            "total_s": elapsed,
            "avg_ms_per_iter": (elapsed * 1000.0 / n_iter) if n_iter > 0 else 0.0,
        }
        self.finished.emit(
            converged,
            "",
            {
                "planner": planner,
                "method": "crocoddyl",
                "timing": timing,
                "_plot_cache": getattr(planner, "_plot_cache", None),
            },
        )

    def _run_acados(self):
        waypoints = self.params.get("waypoints", [])
        durations = self.params.get("durations", [])
        if len(waypoints) < 2:
            self.finished.emit(False, "Need at least 2 waypoints.", None)
            return
        states = [wp_to_state(wp) for wp in waypoints]
        if any(s is None for s in states):
            self.finished.emit(False, "make_uam_state not available.", None)
            return
        kw = {
            "state_weight": self.params.get("state_weight", 1.0),
            "control_weight": self.params.get("control_weight", 1e-5),
            "waypoint_multiplier": self.params.get("waypoint_multiplier", 1000),
            "max_iter": self.params.get("max_iter", 200),
        }
        if self.method == "acados_cascade":
            if not ACADOS_AVAILABLE or not CASCADE_TRAJ_AVAILABLE or run_simple_trajectory_cascade is None:
                self.finished.emit(
                    False,
                    "Acados cascade model not available (need acados, pinocchio, casadi).",
                    None,
                )
                return
            tc = self.params.get("tau_cmd")
            if tc is not None:
                kw["tau_cmd"] = np.asarray(tc, dtype=float).reshape(6)
            ta = self.params.get("tau_act")
            if ta is not None:
                kw["tau_act"] = np.asarray(ta, dtype=float).reshape(6)
            if len(waypoints) == 2:
                simX, simU, time_arr, _, timing = run_simple_trajectory_cascade(
                    states[0], states[1],
                    duration=durations[0],
                    dt=self.params["dt"],
                    **kw,
                )
            else:
                simX, simU, time_arr, _, timing = run_multiwaypoint_trajectory_cascade(
                    states, durations, dt=self.params["dt"], **kw
                )
            method_tag = "acados_cascade"
            control_layout = "high_level"
        else:
            if not ACADOS_AVAILABLE or run_simple_trajectory is None:
                self.finished.emit(False, "Acados not available. Install acados_template and build acados.", None)
                return
            if len(waypoints) == 2:
                simX, simU, time_arr, _, timing = run_simple_trajectory(
                    states[0], states[1],
                    duration=durations[0],
                    dt=self.params["dt"],
                    **kw,
                )
            else:
                simX, simU, time_arr, _, timing = run_multiwaypoint_trajectory(
                    states, durations, dt=self.params["dt"], **kw
                )
            method_tag = "acados"
            control_layout = "direct"
        if simX is None:
            self.finished.emit(False, "Acados optimization failed.", None)
            return
        wp_positions = [[w[0], w[1], w[2]] for w in waypoints]
        wp_times = [w[6] for w in waypoints]
        self.finished.emit(True, "", {
            "method": method_tag,
            "simX": simX,
            "simU": simU,
            "time_arr": time_arr,
            "waypoint_positions": wp_positions,
            "waypoint_times": wp_times,
            "timing": timing or {},
            "control_layout": control_layout,
        })


class S500UAMTrajectoryGUI(QMainWindow):
    """Main GUI window for S500 UAM trajectory planning."""

    def __init__(self):
        super().__init__()
        self.planner = None
        self.worker = None
        self.last_result = None
        self.init_ui()
        self.init_planner()
        self.load_params_from_default()

    def init_ui(self):
        self.setWindowTitle("S500 UAM Trajectory Planning")
        self.setMinimumSize(1100, 750)
        self.resize(1920, 1080)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Optimization method
        method_group = QGroupBox("Optimization Method")
        method_layout = QHBoxLayout()
        self.method_combo = QComboBox()
        self._method_ids = []
        items = []
        if CROCODDYL_AVAILABLE:
            items.append("Crocoddyl (BoxDDP)")
            self._method_ids.append("crocoddyl")
        if ACADOS_AVAILABLE:
            items.append("Acados (thrusters + τ)")
            self._method_ids.append("acados")
            if CASCADE_TRAJ_AVAILABLE:
                items.append("Acados (ω,T,θ + 1st-order)")
                self._method_ids.append("acados_cascade")
        if not items:
            items.append("(No solver available)")
            self._method_ids.append("none")
        self.method_combo.addItems(items)
        self._solver_available = CROCODDYL_AVAILABLE or ACADOS_AVAILABLE
        self.method_combo.setCurrentIndex(0)
        method_layout.addWidget(QLabel("Method:"))
        method_layout.addWidget(self.method_combo)
        method_group.setLayout(method_layout)
        left_layout.addWidget(method_group)

        self.cascade_tau_group = QGroupBox(
            "Cascade: first-order time constant τ (s) for high-level commands — used only by the ω, T, θ methods; ωx/ωy correspond to the roll/pitch angular-rate channels"
        )
        ct_layout = QGridLayout()
        self.tau_omega_xy = QDoubleSpinBox()
        self.tau_omega_xy.setRange(0.001, 2.0)
        self.tau_omega_xy.setDecimals(3)
        self.tau_omega_xy.setSingleStep(0.005)
        self.tau_omega_xy.setValue(0.03)
        self.tau_omega_yaw = QDoubleSpinBox()
        self.tau_omega_yaw.setRange(0.001, 2.0)
        self.tau_omega_yaw.setDecimals(3)
        self.tau_omega_yaw.setSingleStep(0.005)
        self.tau_omega_yaw.setValue(0.08)
        self.tau_thrust = QDoubleSpinBox()
        self.tau_thrust.setRange(0.001, 2.0)
        self.tau_thrust.setDecimals(3)
        self.tau_thrust.setSingleStep(0.005)
        self.tau_thrust.setValue(0.06)
        self.tau_theta = QDoubleSpinBox()
        self.tau_theta.setRange(0.001, 2.0)
        self.tau_theta.setDecimals(3)
        self.tau_theta.setSingleStep(0.005)
        self.tau_theta.setValue(0.05)
        ct_layout.addWidget(QLabel("τ ωx,ωy (fast):"), 0, 0)
        ct_layout.addWidget(self.tau_omega_xy, 0, 1)
        ct_layout.addWidget(QLabel("τ ωz (yaw):"), 1, 0)
        ct_layout.addWidget(self.tau_omega_yaw, 1, 1)
        ct_layout.addWidget(QLabel("τ total thrust:"), 2, 0)
        ct_layout.addWidget(self.tau_thrust, 2, 1)
        ct_layout.addWidget(QLabel("τ joint angles θ1,θ2:"), 3, 0)
        ct_layout.addWidget(self.tau_theta, 3, 1)
        self.cascade_tau_group.setLayout(ct_layout)
        self.cascade_tau_group.setVisible(CASCADE_TRAJ_AVAILABLE)
        left_layout.addWidget(self.cascade_tau_group)

        # Task selection (for backward compat, can hide if using waypoints)
        task_group = QGroupBox("Task Type")
        task_layout = QVBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Multi-Waypoint", "Point-to-Point (2 WP)", "Grasp (Start → EE → Target)"])
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        task_layout.addWidget(self.task_combo)
        task_group.setLayout(task_layout)
        left_layout.addWidget(task_group)

        # Waypoints (list + add/remove/update)
        self.waypoint_group = QGroupBox("Waypoints")
        wp_layout = QVBoxLayout()
        self.waypoint_list = QListWidget()
        self.waypoint_list.setMaximumHeight(100)
        self.waypoint_list.itemSelectionChanged.connect(self.on_waypoint_selected)
        wp_layout.addWidget(self.waypoint_list)
        wp_btn_layout = QHBoxLayout()
        self.add_wp_btn = QPushButton("Add Waypoint")
        self.add_wp_btn.clicked.connect(self.add_waypoint)
        self.remove_wp_btn = QPushButton("Remove Selected")
        self.remove_wp_btn.clicked.connect(self.remove_waypoint)
        wp_btn_layout.addWidget(self.add_wp_btn)
        wp_btn_layout.addWidget(self.remove_wp_btn)
        wp_layout.addLayout(wp_btn_layout)
        # Current waypoint editor
        wp_edit_group = QGroupBox("Edit Waypoint")
        wp_edit_layout = QGridLayout()
        self.wp_x = QDoubleSpinBox()
        self.wp_x.setRange(-100, 100)
        self.wp_x.setValue(0.0)
        self.wp_x.setDecimals(2)
        self.wp_y = QDoubleSpinBox()
        self.wp_y.setRange(-100, 100)
        self.wp_y.setValue(0.0)
        self.wp_y.setDecimals(2)
        self.wp_z = QDoubleSpinBox()
        self.wp_z.setRange(-100, 100)
        self.wp_z.setValue(1.0)
        self.wp_z.setDecimals(2)
        self.wp_yaw = QDoubleSpinBox()
        self.wp_yaw.setRange(-180, 180)
        self.wp_yaw.setValue(0.0)
        self.wp_yaw.setDecimals(1)
        self.wp_j1 = QDoubleSpinBox()
        self.wp_j1.setRange(-180, 180)
        self.wp_j1.setValue(-68.8)
        self.wp_j1.setDecimals(1)
        self.wp_j2 = QDoubleSpinBox()
        self.wp_j2.setRange(-180, 180)
        self.wp_j2.setValue(-34.4)
        self.wp_j2.setDecimals(1)
        self.wp_time = QDoubleSpinBox()
        self.wp_time.setRange(0, 1000)
        self.wp_time.setValue(0.0)
        self.wp_time.setDecimals(2)
        wp_edit_layout.addWidget(QLabel("X (m):"), 0, 0)
        wp_edit_layout.addWidget(self.wp_x, 0, 1)
        wp_edit_layout.addWidget(QLabel("Y (m):"), 0, 2)
        wp_edit_layout.addWidget(self.wp_y, 0, 3)
        wp_edit_layout.addWidget(QLabel("Z (m):"), 1, 0)
        wp_edit_layout.addWidget(self.wp_z, 1, 1)
        wp_edit_layout.addWidget(QLabel("J1 (°):"), 1, 2)
        wp_edit_layout.addWidget(self.wp_j1, 1, 3)
        wp_edit_layout.addWidget(QLabel("J2 (°):"), 2, 0)
        wp_edit_layout.addWidget(self.wp_j2, 2, 1)
        wp_edit_layout.addWidget(QLabel("Yaw (°):"), 2, 2)
        wp_edit_layout.addWidget(self.wp_yaw, 2, 3)
        wp_edit_layout.addWidget(QLabel("Arrival Time (s):"), 3, 0)
        wp_edit_layout.addWidget(self.wp_time, 3, 1)
        self.update_wp_btn = QPushButton("Update Selected")
        self.update_wp_btn.clicked.connect(self.update_waypoint)
        wp_edit_layout.addWidget(self.update_wp_btn, 3, 2, 1, 2)
        wp_edit_group.setLayout(wp_edit_layout)
        wp_layout.addWidget(wp_edit_group)
        self.waypoint_group.setLayout(wp_layout)
        left_layout.addWidget(self.waypoint_group)

        # Default waypoints: [x, y, z, j1_deg, j2_deg, yaw_deg, arrival_time]
        self.waypoints = [
            [0.0, 0.0, 1.0, -68.8, -34.4, 0.0, 0.0],
            [1.0, 0.5, 1.2, -45.8, -17.2, 45.0, 5.0],
        ]
        self.update_waypoint_list()

        # Grasp group (for Grasp task only)
        self.grasp_group = QGroupBox("Grasp (for Grasp task only)")
        grasp_layout = QGridLayout()
        grasp_pairs = [
            ("Start x", "0"), ("Start y", "0"), ("Start z", "1.0"),
            ("Grasp x", "0.5"), ("Grasp y", "0"), ("Grasp z", "0.7"),
            ("Target x", "1.0"), ("Target y", "0.5"), ("Target z", "1.2"),
            ("Duration to grasp", "3.0"), ("Duration to target", "3.0")
        ]
        self.grasp_inputs = {}
        for i, (label, def_val) in enumerate(grasp_pairs):
            r, c = i // 3, (i % 3) * 2
            grasp_layout.addWidget(QLabel(label + ":"), r, c)
            le = QLineEdit()
            le.setText(def_val)
            grasp_layout.addWidget(le, r, c + 1)
            self.grasp_inputs[label.replace(" ", "_").lower()] = le
        self.grasp_group.setLayout(grasp_layout)
        self.grasp_group.setVisible(False)
        left_layout.addWidget(self.grasp_group)

        # Cost parameters
        cost_group = QGroupBox("Cost Parameters")
        cost_layout = QGridLayout()
        cost_layout.addWidget(QLabel("State weight:"), 0, 0)
        self.state_weight = QDoubleSpinBox()
        self.state_weight.setRange(1e-4, 1e4)
        self.state_weight.setValue(1.0)
        self.state_weight.setDecimals(1)
        cost_layout.addWidget(self.state_weight, 0, 1)
        cost_layout.addWidget(QLabel("Control weight:"), 1, 0)
        self.control_weight = QDoubleSpinBox()
        self.control_weight.setRange(1e-8, 50.0)
        self.control_weight.setValue(1e-5)
        self.control_weight.setDecimals(1)
        cost_layout.addWidget(self.control_weight, 1, 1)
        cost_layout.addWidget(QLabel("EE position weight:"), 2, 0)
        self.ee_weight = QDoubleSpinBox()
        self.ee_weight.setRange(0, 1e6)
        self.ee_weight.setValue(5000.0)
        self.ee_weight.setDecimals(1)
        cost_layout.addWidget(self.ee_weight, 2, 1)
        cost_layout.addWidget(QLabel("Waypoint multiplier:"), 3, 0)
        self.waypoint_mult = QDoubleSpinBox()
        self.waypoint_mult.setRange(1, 1e6)
        self.waypoint_mult.setValue(1000.0)
        cost_layout.addWidget(self.waypoint_mult, 3, 1)
        cost_group.setLayout(cost_layout)
        left_layout.addWidget(cost_group)

        # Solver
        solver_group = QGroupBox("Solver")
        solver_layout = QGridLayout()
        solver_layout.addWidget(QLabel("Time step (dt):"), 0, 0)
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 0.5)
        self.dt_spin.setValue(0.02)
        self.dt_spin.setSingleStep(0.01)
        solver_layout.addWidget(self.dt_spin, 0, 1)
        solver_layout.addWidget(QLabel("Max iterations:"), 1, 0)
        self.max_iter = QSpinBox()
        self.max_iter.setRange(10, 2000)
        self.max_iter.setValue(200)
        solver_layout.addWidget(self.max_iter, 1, 1)
        solver_group.setLayout(solver_layout)
        left_layout.addWidget(solver_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.clicked.connect(self.run_optimization)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        btn_layout.addWidget(self.run_btn)
        self.save_btn = QPushButton("Save Plot")
        self.save_btn.clicked.connect(self.save_plot)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)
        self.save_data_btn = QPushButton("Save Data")
        self.save_data_btn.clicked.connect(self.save_trajectory_data)
        self.save_data_btn.setEnabled(False)
        btn_layout.addWidget(self.save_data_btn)
        left_layout.addLayout(btn_layout)
        param_btn_layout = QHBoxLayout()
        self.save_params_btn = QPushButton("Save Params")
        self.save_params_btn.clicked.connect(self.save_params)
        self.load_params_btn = QPushButton("Load Params")
        self.load_params_btn.clicked.connect(self.load_params)
        param_btn_layout.addWidget(self.save_params_btn)
        param_btn_layout.addWidget(self.load_params_btn)
        left_layout.addLayout(param_btn_layout)

        # Log
        log_group = QGroupBox("Log")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_group.setLayout(QVBoxLayout())
        log_group.layout().addWidget(self.log_text)
        left_layout.addWidget(log_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Plot area: 4 tabs - Current, Previous, Current 3D, Previous 3D
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.plot_tabs = QTabWidget()
        self.prev_result = None

        # Tab 1: Current (2D trajectory)
        curr_traj = QWidget()
        curr_traj_layout = QVBoxLayout(curr_traj)
        self.plot_canvas = FigureCanvas(Figure(figsize=(20, 16)))
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        curr_traj_layout.addWidget(self.plot_toolbar)
        curr_traj_layout.addWidget(self.plot_canvas)
        self.plot_tabs.addTab(curr_traj, "Current")

        # Tab 2: Previous (2D trajectory)
        prev_traj = QWidget()
        prev_traj_layout = QVBoxLayout(prev_traj)
        self.plot_canvas_prev = FigureCanvas(Figure(figsize=(20, 16)))
        self.plot_toolbar_prev = NavigationToolbar(self.plot_canvas_prev, self)
        prev_traj_layout.addWidget(self.plot_toolbar_prev)
        prev_traj_layout.addWidget(self.plot_canvas_prev)
        self.plot_tabs.addTab(prev_traj, "Previous")

        # Tab 3: Current 3D
        curr_3d = QWidget()
        curr_3d_layout = QVBoxLayout(curr_3d)
        self.plot_canvas_3d = FigureCanvas(Figure(figsize=(10, 8)))
        self.plot_toolbar_3d = NavigationToolbar(self.plot_canvas_3d, self)
        curr_3d_layout.addWidget(self.plot_toolbar_3d)
        curr_3d_layout.addWidget(self.plot_canvas_3d)
        self.plot_tabs.addTab(curr_3d, "Current 3D")

        # Tab 4: Previous 3D
        prev_3d = QWidget()
        prev_3d_layout = QVBoxLayout(prev_3d)
        self.plot_canvas_3d_prev = FigureCanvas(Figure(figsize=(10, 8)))
        self.plot_toolbar_3d_prev = NavigationToolbar(self.plot_canvas_3d_prev, self)
        prev_3d_layout.addWidget(self.plot_toolbar_3d_prev)
        prev_3d_layout.addWidget(self.plot_canvas_3d_prev)
        self.plot_tabs.addTab(prev_3d, "Previous 3D")
        right_layout.addWidget(self.plot_tabs)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)

    def on_task_changed(self):
        idx = self.task_combo.currentIndex()
        self.grasp_group.setVisible(idx == 2)
        self.waypoint_group.setVisible(idx != 2)

    def update_waypoint_list(self):
        self.waypoint_list.clear()
        for i, wp in enumerate(self.waypoints):
            if len(wp) < 7:
                wp = list(wp) + [0.0] * (7 - len(wp))
            label = "Start" if i == 0 else f"WP {i}"
            t = wp[6] if len(wp) > 6 else 0
            item_text = (
                f"{label}: [{wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}] "
                f"j=[{wp[3]:.0f},{wp[4]:.0f}]° yaw={wp[5]:.0f}° @ t={t:.2f}s"
            )
            self.waypoint_list.addItem(QListWidgetItem(item_text))

    def add_waypoint(self):
        new_wp = [
            self.wp_x.value(),
            self.wp_y.value(),
            self.wp_z.value(),
            self.wp_j1.value(),
            self.wp_j2.value(),
            self.wp_yaw.value(),
            self.wp_time.value(),
        ]
        self.waypoints.append(new_wp)
        self.update_waypoint_list()
        self.waypoint_list.setCurrentRow(len(self.waypoints) - 1)

    def remove_waypoint(self):
        row = self.waypoint_list.currentRow()
        if row < 0 or row >= len(self.waypoints):
            return
        if row == 0:
            QMessageBox.warning(self, "Warning", "Cannot remove start point")
            return
        self.waypoints.pop(row)
        self.update_waypoint_list()
        if row > 0:
            self.waypoint_list.setCurrentRow(row - 1)

    def update_waypoint(self):
        row = self.waypoint_list.currentRow()
        if row < 0 or row >= len(self.waypoints):
            return
        self.waypoints[row] = [
            self.wp_x.value(),
            self.wp_y.value(),
            self.wp_z.value(),
            self.wp_j1.value(),
            self.wp_j2.value(),
            self.wp_yaw.value(),
            self.wp_time.value(),
        ]
        self.update_waypoint_list()
        self.waypoint_list.setCurrentRow(row)

    def on_waypoint_selected(self):
        row = self.waypoint_list.currentRow()
        if row >= 0 and row < len(self.waypoints):
            wp = self.waypoints[row]
            if len(wp) < 7:
                wp = list(wp) + [0.0] * (7 - len(wp))
            self.wp_x.setValue(wp[0])
            self.wp_y.setValue(wp[1])
            self.wp_z.setValue(wp[2])
            self.wp_j1.setValue(wp[3] if len(wp) > 3 else 0)
            self.wp_j2.setValue(wp[4] if len(wp) > 4 else 0)
            self.wp_yaw.setValue(wp[5] if len(wp) > 5 else 0)
            self.wp_time.setValue(wp[6] if len(wp) > 6 else 0)

    def get_waypoints_and_durations(self):
        """Get waypoints and segment durations from list. Sorts by arrival time."""
        if len(self.waypoints) < 2:
            return [], []
        wps = []
        for wp in self.waypoints:
            if len(wp) < 7:
                wp = list(wp) + [0.0] * (7 - len(wp))
            wps.append(wp[:7])
        wps = sorted(wps, key=lambda x: float(x[6]))
        durations = []
        for i in range(len(wps) - 1):
            d = wps[i + 1][6] - wps[i][6]
            if d <= 0:
                d = 1.0
            durations.append(d)
        return wps, durations

    def get_params_dict(self):
        d = {
            "task_type": self.task_combo.currentIndex(),
            "method": self.method_combo.currentIndex(),
            "traj_gui_wp_version": 2,
            "waypoints": [list(w) for w in self.waypoints],
            "dt": self.dt_spin.value(),
            "max_iter": self.max_iter.value(),
            "state_weight": self.state_weight.value(),
            "control_weight": self.control_weight.value(),
            "waypoint_multiplier": self.waypoint_mult.value(),
            "grasp_ee_weight": self.ee_weight.value(),
            "tau_omega_xy": self.tau_omega_xy.value(),
            "tau_omega_yaw": self.tau_omega_yaw.value(),
            "tau_thrust": self.tau_thrust.value(),
            "tau_theta": self.tau_theta.value(),
        }
        for k, le in self.grasp_inputs.items():
            try:
                d[k] = float(le.text())
            except (ValueError, AttributeError):
                pass
        return d

    def set_params_from_dict(self, d):
        self.task_combo.setCurrentIndex(d.get("task_type", 0))
        mi = int(d.get("method", 0))
        mc = self.method_combo.count()
        if mc > 0:
            self.method_combo.setCurrentIndex(max(0, min(mi, mc - 1)))
        if "waypoints" in d:
            wps = [list(w) for w in d["waypoints"]]
            if int(d.get("traj_gui_wp_version", 1)) < 2:
                for w in wps:
                    if len(w) >= 6:
                        yaw, j1, j2 = float(w[3]), float(w[4]), float(w[5])
                        w[3], w[4], w[5] = j1, j2, yaw
            self.waypoints = wps
            self.update_waypoint_list()
        if "dt" in d:
            self.dt_spin.setValue(d["dt"])
        if "max_iter" in d:
            self.max_iter.setValue(d["max_iter"])
        if "state_weight" in d:
            self.state_weight.setValue(d["state_weight"])
        if "control_weight" in d:
            self.control_weight.setValue(d["control_weight"])
        if "waypoint_multiplier" in d:
            self.waypoint_mult.setValue(d["waypoint_multiplier"])
        if "grasp_ee_weight" in d:
            self.ee_weight.setValue(d["grasp_ee_weight"])
        for k, le in self.grasp_inputs.items():
            if k in d:
                le.setText(str(d[k]))
        if "tau_omega_xy" in d:
            self.tau_omega_xy.setValue(float(d["tau_omega_xy"]))
        if "tau_omega_yaw" in d:
            self.tau_omega_yaw.setValue(float(d["tau_omega_yaw"]))
        if "tau_thrust" in d:
            self.tau_thrust.setValue(float(d["tau_thrust"]))
        if "tau_theta" in d:
            self.tau_theta.setValue(float(d["tau_theta"]))
        self.grasp_group.setVisible(d.get("task_type", 0) == 2)

    def init_planner(self):
        if S500UAMTrajectoryPlanner is None:
            self.log("WARNING: Crocoddyl planner not available.")
        else:
            try:
                self.planner = S500UAMTrajectoryPlanner()
                self.log("Planner (Crocoddyl) initialized.")
            except Exception as e:
                self.log(f"Failed to init planner: {e}")
        if ACADOS_AVAILABLE:
            self.log("Acados available.")
        if not self._solver_available:
            self.log("ERROR: No solver available. Install pinocchio+crocoddyl or acados.")
            self.run_btn.setEnabled(False)

    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def run_optimization(self):
        task = self.task_combo.currentIndex()
        if task == 2:  # Grasp
            self._run_grasp()
            return
        # Multi-waypoint or Point-to-point
        wps, durations = self.get_waypoints_and_durations()
        if len(wps) < 2:
            self.log("ERROR: Need at least 2 waypoints")
            QMessageBox.warning(self, "Error", "Need at least 2 waypoints")
            return
        mid = self.method_combo.currentIndex()
        method = self._method_ids[mid] if 0 <= mid < len(self._method_ids) else "none"
        if method == "none":
            self.log("ERROR: No solver available")
            QMessageBox.warning(self, "Error", "No solver available")
            return
        if method == "acados_cascade" and not CASCADE_TRAJ_AVAILABLE:
            self.log("ERROR: Acados cascade model not available")
            QMessageBox.warning(self, "Error", "Acados cascade model not available")
            return
        if method in ("acados", "acados_cascade") and not ACADOS_AVAILABLE:
            self.log("ERROR: Acados not available")
            QMessageBox.warning(self, "Error", "Acados not available")
            return
        if method == "crocoddyl" and self.planner is None:
            self.log("ERROR: Crocoddyl planner not initialized")
            QMessageBox.warning(self, "Error", "Planner not initialized")
            return
        tau_cmd = np.array([
            self.tau_omega_xy.value(),
            self.tau_omega_xy.value(),
            self.tau_omega_yaw.value(),
            self.tau_thrust.value(),
            self.tau_theta.value(),
            self.tau_theta.value(),
        ], dtype=float)
        params = {
            "waypoints": wps,
            "durations": durations,
            "dt": self.dt_spin.value(),
            "max_iter": self.max_iter.value(),
            "state_weight": self.state_weight.value(),
            "control_weight": self.control_weight.value(),
            "waypoint_multiplier": self.waypoint_mult.value(),
            "planner": self.planner,
            "tau_cmd": tau_cmd,
            "use_actuator_first_order": bool(method == "crocoddyl"),
        }
        self.run_btn.setEnabled(False)
        self.log(f"Running {method} optimization ({len(wps)} waypoints)...")
        self.worker = OptimizationWorker(method, params)
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.start()

    def _run_grasp(self):
        if self.planner is None:
            self.log("ERROR: Grasp task requires Crocoddyl planner")
            return
        if make_uam_state is None:
            self.log("ERROR: make_uam_state not available")
            return
        def get_grasp_float(key, default):
            try:
                return float(self.grasp_inputs[key].text())
            except (KeyError, ValueError):
                return default
        deg2rad = np.pi / 180
        sx = get_grasp_float("start_x", 0)
        sy = get_grasp_float("start_y", 0)
        sz = get_grasp_float("start_z", 1.0)
        start = make_uam_state(sx, sy, sz, j1=-1.2, j2=-0.6, yaw=0)
        gx = get_grasp_float("grasp_x", 0.5)
        gy = get_grasp_float("grasp_y", 0)
        gz = get_grasp_float("grasp_z", 0.7)
        grasp_pos = np.array([gx, gy, gz])
        tx = get_grasp_float("target_x", 1.0)
        ty = get_grasp_float("target_y", 0.5)
        tz = get_grasp_float("target_z", 1.2)
        target = make_uam_state(tx, ty, tz, j1=-0.8, j2=-0.3, yaw=45*deg2rad)
        d1 = get_grasp_float("duration_to_grasp", 3.0)
        d2 = get_grasp_float("duration_to_target", 3.0)
        self.run_btn.setEnabled(False)
        self.log("Running Crocoddyl grasp optimization...")
        self.worker = OptimizationWorker("crocoddyl", {
            "planner": self.planner,
            "grasp": True,
            "start_state": start,
            "grasp_position": grasp_pos,
            "target_state": target,
            "durations": [d1, d2],
            "dt": self.dt_spin.value(),
            "max_iter": self.max_iter.value(),
            "grasp_ee_weight": self.ee_weight.value(),
        })
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.start()

    def _draw_result_to_figures(self, result_data, fig_main, fig_3d, title_suffix=""):
        """Draw result (Crocoddyl or Acados) to given figures."""
        if result_data is None:
            return
        timing = result_data.get("timing")
        if result_data.get("method") in ("acados", "acados_cascade"):
            simX = result_data["simX"]
            simU = result_data["simU"]
            time_arr = result_data["time_arr"]
            wp_times = result_data.get("waypoint_times", [])
            wp_pos = result_data.get("waypoint_positions", [])
            ac_tag = "Acados cascade" if result_data.get("method") == "acados_cascade" else "Acados"
            if plot_acados_into_figure:
                plot_acados_into_figure(
                    simX, simU, time_arr, fig_main,
                    title=f"{ac_tag}{title_suffix}",
                    waypoint_times=wp_times,
                    timing_info=timing,
                    control_layout=result_data.get("control_layout", "direct"),
                )
            if plot_acados_3d_into_figure:
                plot_acados_3d_into_figure(simX, fig_3d, waypoint_positions=wp_pos)
        else:
            planner = result_data.get("planner")
            cache = result_data.get("_plot_cache")
            if cache and planner:
                planner._plot_cache = cache
                planner.get_plot_figure(title=f"Crocoddyl{title_suffix}", fig=fig_main, timing_info=timing)
                planner.get_3d_plot_figure(fig=fig_3d)
                planner._plot_cache = None
            elif planner and hasattr(planner, "solver") and planner.solver:
                planner.get_plot_figure(title=f"Crocoddyl{title_suffix}", fig=fig_main, timing_info=timing)
                planner.get_3d_plot_figure(fig=fig_3d)

    def on_optimization_finished(self, converged, error_msg, result_data):
        self.run_btn.setEnabled(True)
        if error_msg:
            self.log(f"ERROR:\n{error_msg}")
            QMessageBox.critical(self, "Error", error_msg[:500])
            return
        # Move current to previous before overwriting (copy trajectory for Crocoddyl)
        if self.last_result is not None:
            prev = self.last_result
            if prev.get("method") == "crocoddyl":
                planner = prev.get("planner")
                if planner and hasattr(planner, "solver") and planner.solver:
                    self.prev_result = {
                        "method": "crocoddyl",
                        "planner": planner,
                        "timing": prev.get("timing"),
                        "_plot_cache": {
                            "xs": [x.copy() for x in planner.solver.xs],
                            "us": [u.copy() for u in planner.solver.us],
                            "dt": planner.dt or 0.02,
                            "waypoint_times": list(getattr(planner, "_waypoint_times", [])),
                            "waypoint_positions": [np.asarray(p, dtype=float).copy() for p in getattr(planner, "_waypoint_positions", [])],
                            "waypoint_labels": list(getattr(planner, "_waypoint_labels", [])),
                            "waypoint_ee_positions": [np.asarray(p, dtype=float).copy() for p in getattr(planner, "_waypoint_ee_positions", [])],
                            "cost_logger": getattr(planner, "_cost_logger", None),
                            "ee_positions": planner.get_ee_positions(),
                        },
                    }
                else:
                    self.prev_result = prev
            else:
                self.prev_result = prev
            self._draw_result_to_figures(
                self.prev_result,
                self.plot_canvas_prev.figure,
                self.plot_canvas_3d_prev.figure,
                title_suffix=" (Previous)"
            )
            self.plot_canvas_prev.draw()
            self.plot_canvas_3d_prev.draw()
        self.last_result = result_data
        status = "Converged" if converged else "Not converged"
        self.log(f"Optimization finished: {status}")
        timing = result_data.get("timing") if result_data else None
        if result_data and result_data.get("method") in ("acados", "acados_cascade"):
            simX = result_data["simX"]
            simU = result_data["simU"]
            time_arr = result_data["time_arr"]
            wp_times = result_data.get("waypoint_times", [])
            wp_pos = result_data.get("waypoint_positions", [])
            ac_title = "Acados cascade" if result_data.get("method") == "acados_cascade" else "Acados"
            plot_acados_into_figure(
                simX, simU, time_arr,
                self.plot_canvas.figure,
                title=f"S500 UAM Trajectory ({status}) [{ac_title}]",
                waypoint_times=wp_times,
                timing_info=timing,
                control_layout=result_data.get("control_layout", "direct"),
            )
            plot_acados_3d_into_figure(simX, self.plot_canvas_3d.figure, waypoint_positions=wp_pos)
        else:
            planner = result_data.get("planner") if result_data else self.planner
            if planner and hasattr(planner, "solver") and planner.solver:
                cost = planner.solver.cost
                self.log(f"Final cost: {cost:.6f}")
                planner.get_plot_figure(
                    title=f"S500 UAM Trajectory ({status}) [Crocoddyl]",
                    fig=self.plot_canvas.figure,
                    timing_info=timing,
                )
                planner.get_3d_plot_figure(fig=self.plot_canvas_3d.figure)
        self.plot_canvas.draw()
        self.plot_canvas_3d.draw()
        self.save_btn.setEnabled(True)
        self.save_data_btn.setEnabled(True)

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG (*.png);;All (*)")
        if path:
            import matplotlib.pyplot as plt
            self.plot_canvas.figure.savefig(path, dpi=300, bbox_inches='tight')
            self.log(f"Plot saved to {path}")
            path_3d = path.replace('.png', '_3d.png') if path.endswith('.png') else path + '_3d.png'
            self.plot_canvas_3d.figure.savefig(path_3d, dpi=300, bbox_inches='tight')
            self.log(f"3D plot saved to {path_3d}")

    def save_trajectory_data(self):
        if self.last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Trajectory Data", "", "NPZ (*.npz);;All (*)")
        if path:
            if self.last_result.get("method") in ("acados", "acados_cascade"):
                np.savez(path,
                        states=self.last_result["simX"],
                        controls=self.last_result["simU"],
                        time_arr=self.last_result["time_arr"])
            else:
                planner = self.last_result.get("planner") or self.planner
                if planner and planner.solver:
                    planner.save_trajectory(path)
            self.log(f"Trajectory data saved to {path}")

    def save_params(self):
        try:
            d = self.get_params_dict()
            DEFAULT_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_PARAMS_PATH, "w") as f:
                json.dump(d, f, indent=2)
            self.log(f"Parameters saved to {DEFAULT_PARAMS_PATH}")
        except Exception as e:
            self.log(f"Failed to save params: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def load_params(self):
        self.load_params_from_default()

    def load_params_from_default(self):
        if not DEFAULT_PARAMS_PATH.exists():
            return
        try:
            with open(DEFAULT_PARAMS_PATH, "r") as f:
                d = json.load(f)
            self.set_params_from_dict(d)
            self.log(f"Parameters loaded from {DEFAULT_PARAMS_PATH}")
        except Exception as e:
            self.log(f"Failed to load params: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = S500UAMTrajectoryGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
