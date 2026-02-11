#!/usr/bin/env python3
"""
S500 UAM Trajectory Planning GUI

A graphical interface for S500 UAM trajectory optimization:
- Task selection: Point-to-point or Grasp
- Cost parameter tuning
- Waypoint configuration
- Plot visualization

Usage:
    python s500_uam_trajectory_gui.py

Requires: PyQt5, matplotlib, numpy, pinocchio, crocoddyl
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
    QSplitter, QFrame, QMessageBox, QFileDialog
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
except ImportError:
    S500UAMTrajectoryPlanner = None
    make_uam_state = None

# Fixed path for saving/loading parameters
DEFAULT_PARAMS_PATH = Path(__file__).parent.parent / "config" / "yaml" / "trajectories" / "s500_uam_trajectory_params.json"


class OptimizationWorker(QThread):
    """Background worker for trajectory optimization."""
    finished = pyqtSignal(bool, str)

    def __init__(self, planner, task_type, params):
        super().__init__()
        self.planner = planner
        self.task_type = task_type
        self.params = params

    def run(self):
        try:
            if self.task_type == "point_to_point":
                self.planner.create_trajectory_problem_simple(
                    start_state=self.params["start_state"],
                    target_state=self.params["target_state"],
                    duration=self.params["duration"],
                    dt=self.params["dt"],
                    waypoint_multiplier=self.params["waypoint_multiplier"],
                    state_weight=self.params["state_weight"],
                    control_weight=self.params["control_weight"],
                )
            else:  # grasp
                self.planner.create_trajectory_problem(
                    start_state=self.params["start_state"],
                    grasp_position=self.params["grasp_position"],
                    target_state=self.params["target_state"],
                    durations=self.params["durations"],
                    dt=self.params["dt"],
                    grasp_ee_weight=self.params["grasp_ee_weight"],
                    waypoint_multiplier=self.params["waypoint_multiplier"],
                    state_weight=self.params["state_weight"],
                    control_weight=self.params["control_weight"],
                )
            converged = self.planner.solve_trajectory(
                max_iter=self.params["max_iter"],
                verbose=False
            )
            self.finished.emit(converged, "")
        except Exception as e:
            import traceback
            self.finished.emit(False, traceback.format_exc())


class S500UAMTrajectoryGUI(QMainWindow):
    """Main GUI window for S500 UAM trajectory planning."""

    def __init__(self):
        super().__init__()
        self.planner = None
        self.worker = None
        self.init_ui()
        self.init_planner()
        self.load_params_from_default()

    def init_ui(self):
        self.setWindowTitle("S500 UAM Trajectory Planning")
        self.setMinimumSize(1100, 750)
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Task selection
        task_group = QGroupBox("Task Type")
        task_layout = QVBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Point-to-Point", "Grasp (Start → EE → Target)"])
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        task_layout.addWidget(self.task_combo)
        task_group.setLayout(task_layout)
        left_layout.addWidget(task_group)

        # Waypoints
        self.waypoint_group = QGroupBox("Waypoints")
        self.waypoint_layout = QGridLayout()
        self.add_waypoint_fields()
        self.waypoint_group.setLayout(self.waypoint_layout)
        left_layout.addWidget(self.waypoint_group)

        # Cost parameters
        cost_group = QGroupBox("Cost Parameters")
        cost_layout = QGridLayout()
        cost_layout.addWidget(QLabel("State weight:"), 0, 0)
        self.state_weight = QDoubleSpinBox()
        self.state_weight.setRange(1e-4, 1e4)
        self.state_weight.setValue(1.0)
        self.state_weight.setDecimals(1)
        self.state_weight.setSingleStep(0.1)
        cost_layout.addWidget(self.state_weight, 0, 1)
        cost_layout.addWidget(QLabel("Control weight:"), 1, 0)
        self.control_weight = QDoubleSpinBox()
        self.control_weight.setRange(1e-8, 1.0)
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

        # Plot area (tabbed: Trajectory + 3D)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.plot_tabs = QTabWidget()
        self.plot_canvas = FigureCanvas(Figure(figsize=(14, 11)))
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        traj_widget = QWidget()
        traj_layout = QVBoxLayout(traj_widget)
        traj_layout.addWidget(self.plot_toolbar)
        traj_layout.addWidget(self.plot_canvas)
        self.plot_tabs.addTab(traj_widget, "Trajectory")
        self.plot_canvas_3d = FigureCanvas(Figure(figsize=(10, 8)))
        self.plot_toolbar_3d = NavigationToolbar(self.plot_canvas_3d, self)
        traj3d_widget = QWidget()
        traj3d_layout = QVBoxLayout(traj3d_widget)
        traj3d_layout.addWidget(self.plot_toolbar_3d)
        traj3d_layout.addWidget(self.plot_canvas_3d)
        self.plot_tabs.addTab(traj3d_widget, "3D Trajectory")
        right_layout.addWidget(self.plot_tabs)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)

    def add_waypoint_fields(self):
        """Add input fields for waypoints.
        Row 1: Start x, y, z, Start joint (角度)
        Row 2: Target x, y, z, Target joint (角度)
        Row 3: Duration
        """
        self.wp_inputs = {}
        # Row 1: Start x, y, z, Start j1, j2 (key 必须与 get_params 中的 get_float 一致)
        row1 = [
            ("Start x", "0", "start_x"), ("Start y", "0", "start_y"), ("Start z", "1.0", "start_z"),
            ("Start j1 (°)", "-68.8", "start_j1"), ("Start j2 (°)", "-34.4", "start_j2")
        ]
        for col, (label, def_val, key) in enumerate(row1):
            self.waypoint_layout.addWidget(QLabel(label + ":"), 0, col * 2)
            le = QLineEdit()
            le.setText(def_val)
            self.waypoint_layout.addWidget(le, 0, col * 2 + 1)
            self.wp_inputs[key] = le
        # Row 2: Target x, y, z, Target j1, j2
        row2 = [
            ("Target x", "1.0", "target_x"), ("Target y", "0.5", "target_y"), ("Target z", "1.2", "target_z"),
            ("Target j1 (°)", "-45.8", "target_j1"), ("Target j2 (°)", "-17.2", "target_j2")
        ]
        for col, (label, def_val, key) in enumerate(row2):
            self.waypoint_layout.addWidget(QLabel(label + ":"), 1, col * 2)
            le = QLineEdit()
            le.setText(def_val)
            self.waypoint_layout.addWidget(le, 1, col * 2 + 1)
            self.wp_inputs[key] = le
        # Row 3: Duration
        self.waypoint_layout.addWidget(QLabel("Duration:"), 2, 0)
        le = QLineEdit()
        le.setText("5.0")
        self.waypoint_layout.addWidget(le, 2, 1)
        self.wp_inputs["duration"] = le
        self.grasp_group = QGroupBox("Grasp (for Grasp task only)")
        grasp_layout = QGridLayout()
        grasp_pairs = [("Grasp x", "0.5"), ("Grasp y", "0"), ("Grasp z", "0.7"),
                      ("Duration to grasp", "3.0"), ("Duration to target", "3.0")]
        for i, (label, def_val) in enumerate(grasp_pairs):
            grasp_layout.addWidget(QLabel(label + ":"), i // 2, (i % 2) * 2)
            le = QLineEdit()
            le.setText(def_val)
            grasp_layout.addWidget(le, i // 2, (i % 2) * 2 + 1)
            self.wp_inputs[label.replace(" ", "_").lower()] = le
        self.grasp_group.setLayout(grasp_layout)
        self.waypoint_layout.addWidget(self.grasp_group, 3, 0, 1, 10)
        self.grasp_group.setVisible(False)

    def on_task_changed(self):
        idx = self.task_combo.currentIndex()
        self.grasp_group.setVisible(idx == 1)

    def get_float(self, key, default=0.0):
        try:
            return float(self.wp_inputs[key].text())
        except (KeyError, ValueError):
            return default

    def get_params_dict(self):
        """Get serializable params dict for save/load (no numpy arrays)."""
        return {
            "task_type": self.task_combo.currentIndex(),
            "start_x": self.get_float("start_x", 0),
            "start_y": self.get_float("start_y", 0),
            "start_z": self.get_float("start_z", 1),
            "start_j1": self.get_float("start_j1", -68.8),
            "start_j2": self.get_float("start_j2", -34.4),
            "target_x": self.get_float("target_x", 1),
            "target_y": self.get_float("target_y", 0.5),
            "target_z": self.get_float("target_z", 1.2),
            "target_j1": self.get_float("target_j1", -45.8),
            "target_j2": self.get_float("target_j2", -17.2),
            "grasp_x": self.get_float("grasp_x", 0.5),
            "grasp_y": self.get_float("grasp_y", 0),
            "grasp_z": self.get_float("grasp_z", 0.7),
            "duration": self.get_float("duration", 5.0),
            "duration_to_grasp": self.get_float("duration_to_grasp", 3.0),
            "duration_to_target": self.get_float("duration_to_target", 3.0),
            "dt": self.dt_spin.value(),
            "max_iter": self.max_iter.value(),
            "state_weight": self.state_weight.value(),
            "control_weight": self.control_weight.value(),
            "waypoint_multiplier": self.waypoint_mult.value(),
            "grasp_ee_weight": self.ee_weight.value(),
        }

    def set_params_from_dict(self, d):
        """Load params from dict into GUI."""
        self.task_combo.setCurrentIndex(d.get("task_type", 0))
        for key in ["start_x", "start_y", "start_z", "start_j1", "start_j2",
                    "target_x", "target_y", "target_z", "target_j1", "target_j2",
                    "grasp_x", "grasp_y", "grasp_z",
                    "duration", "duration_to_grasp", "duration_to_target"]:
            if key in d and key in self.wp_inputs:
                self.wp_inputs[key].setText(str(d[key]))
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
        self.grasp_group.setVisible(d.get("task_type", 0) == 1)

    def get_params(self):
        if make_uam_state is None:
            return None
        # j1, j2 输入为角度(°)，需转为弧度
        deg2rad = np.pi / 180.0
        params = {
            "start_state": make_uam_state(
                self.get_float("start_x", 0), self.get_float("start_y", 0), self.get_float("start_z", 1),
                self.get_float("start_j1", -68.8) * deg2rad, self.get_float("start_j2", -34.4) * deg2rad
            ),
            "target_state": make_uam_state(
                self.get_float("target_x", 1), self.get_float("target_y", 0.5), self.get_float("target_z", 1.2),
                self.get_float("target_j1", -45.8) * deg2rad, self.get_float("target_j2", -17.2) * deg2rad
            ),
            "grasp_position": np.array([
                self.get_float("grasp_x", 0.5), self.get_float("grasp_y", 0), self.get_float("grasp_z", 0.7)
            ]),
            "duration": self.get_float("duration", 5.0),
            "durations": [
                self.get_float("duration_to_grasp", 3.0),
                self.get_float("duration_to_target", 3.0)
            ],
            "dt": self.dt_spin.value(),
            "max_iter": self.max_iter.value(),
            "state_weight": self.state_weight.value(),
            "control_weight": self.control_weight.value(),
            "waypoint_multiplier": self.waypoint_mult.value(),
            "grasp_ee_weight": self.ee_weight.value(),
        }
        return params

    def init_planner(self):
        if S500UAMTrajectoryPlanner is None:
            self.log("ERROR: Could not import S500UAMTrajectoryPlanner. Check dependencies.")
            self.run_btn.setEnabled(False)
            return
        try:
            self.planner = S500UAMTrajectoryPlanner()
            self.log("Planner initialized successfully.")
        except Exception as e:
            self.log(f"Failed to init planner: {e}")
            self.run_btn.setEnabled(False)

    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def run_optimization(self):
        if self.planner is None:
            return
        params = self.get_params()
        if params is None:
            self.log("ERROR: make_uam_state not available")
            return
        task_type = "point_to_point" if self.task_combo.currentIndex() == 0 else "grasp"
        self.run_btn.setEnabled(False)
        self.log(f"Running {task_type} optimization...")
        self.worker = OptimizationWorker(self.planner, task_type, params)
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.start()

    def on_optimization_finished(self, converged, error_msg):
        self.run_btn.setEnabled(True)
        if error_msg:
            self.log(f"ERROR:\n{error_msg}")
            QMessageBox.critical(self, "Error", error_msg[:500])
            return
        status = "Converged" if converged else "Not converged"
        self.log(f"Optimization finished: {status}")
        cost = self.planner.solver.cost if self.planner.solver else 0
        self.log(f"Final cost: {cost:.6f}")
        self.planner.get_plot_figure(
            title=f"S500 UAM Trajectory ({status})",
            fig=self.plot_canvas.figure
        )
        self.planner.get_3d_plot_figure(fig=self.plot_canvas_3d.figure)
        self.plot_canvas.draw()
        self.plot_canvas_3d.draw()
        self.save_btn.setEnabled(True)
        self.save_data_btn.setEnabled(True)

    def save_plot(self):
        if self.planner is None or self.planner.solver is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG (*.png);;All (*)"
        )
        if path:
            import matplotlib.pyplot as plt
            fig_main = self.planner.get_plot_figure()
            if fig_main:
                fig_main.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig_main)
                self.log(f"Plot saved to {path}")
            path_3d = path.replace('.png', '_3d.png') if path.endswith('.png') else path + '_3d.png'
            fig_3d = self.planner.get_3d_plot_figure()
            if fig_3d:
                fig_3d.savefig(path_3d, dpi=300, bbox_inches='tight')
                plt.close(fig_3d)
                self.log(f"3D plot saved to {path_3d}")

    def save_trajectory_data(self):
        if self.planner is None or self.planner.solver is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Trajectory Data", "", "NPZ (*.npz);;All (*)"
        )
        if path:
            self.planner.save_trajectory(path)
            self.log(f"Trajectory data saved to {path}")

    def save_params(self):
        """Save parameters to fixed path."""
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
        """Load parameters from fixed path."""
        self.load_params_from_default()

    def load_params_from_default(self):
        """Load parameters from default path if exists."""
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
