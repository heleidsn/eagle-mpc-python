#!/usr/bin/env python3
"""
S500 UAM — EE reference trajectory + Acados tracking MPC GUI

Workflow (three tabs, aligned with the trajectory_gui style):
  1) Reference trajectory: Minimum snap (waypoints x,y,z,yaw°, time) or Figure-eight (center, a, period)
  2) Tracking method: direct thrust + torques / high-level ω, T, θ + first-order actuator
  3) MPC and simulation: dt_mpc, N, sim_dt, control_dt (ZOH / MPC period), T_sim, w_ee, etc.

Dependencies: PyQt5, matplotlib, numpy, pinocchio, casadi, acados_template (same as s500_uam_ee_snap_tracking_mpc.py)

Usage:
    python s500_uam_ee_tracking_gui.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
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

_ee_mpc_import_err: Exception | None = None
try:
    import s500_uam_ee_snap_tracking_mpc as ee_mpc

    EE_MPC_OK = bool(ee_mpc.ACADOS_AVAILABLE and ee_mpc.PINOCCHIO_AVAILABLE and ee_mpc.DEPS_OK)
except Exception as e:
    ee_mpc = None  # type: ignore
    EE_MPC_OK = False
    _ee_mpc_import_err = e

DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "yaml" / "trajectories" / "s500_uam_ee_tracking_gui.json"
)


def _snap_default_table() -> list[list[float]]:
    """[x, y, z, yaw_deg, t_s] per row."""
    return [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.35, 0.95, 14.3, 2.5],
        [0.85, -0.15, 1.05, -22.9, 5.0],
        [1.0, 0.2, 0.9, 8.6, 8.0],
    ]


class EeTrackingWorker(QThread):
    finished = pyqtSignal(bool, str, object)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            import s500_uam_ee_snap_tracking_mpc as em

            p = self.params
            track = p["track"]
            deg = np.pi / 180.0
            if track == "snap":
                rows = p["snap_rows"]
                wp = np.zeros((len(rows), 4), dtype=float)
                tw = np.zeros(len(rows), dtype=float)
                for i, r in enumerate(rows):
                    wp[i, 0] = r[0]
                    wp[i, 1] = r[1]
                    wp[i, 2] = r[2]
                    wp[i, 3] = r[3] * deg
                    tw[i] = r[4]
                out = em.run_ee_tracking_pipeline(
                    "snap",
                    waypoints_xyz_yaw=wp,
                    times_wp=tw,
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
                )
            else:
                out = em.run_ee_tracking_pipeline(
                    "eight",
                    eight_center=np.array(p["eight_center"], dtype=float),
                    eight_a=p["eight_a"],
                    eight_period=p["eight_period"],
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
                )
            self.finished.emit(True, "", out)
        except Exception:
            self.finished.emit(False, traceback.format_exc(), None)


class S500UAMEeTrackingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker: EeTrackingWorker | None = None
        self._last_out: dict | None = None
        self._prev_out: dict | None = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("S500 UAM — EE Tracking MPC")
        self.setMinimumSize(1120, 780)
        self.resize(1680, 980)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QWidget()
        left_l = QVBoxLayout(left)
        root.addWidget(left, stretch=0)

        self.tabs = QTabWidget()
        left_l.addWidget(self.tabs)

        # ----- Tab 1: Trajectory -----
        tab_traj = QWidget()
        t1 = QVBoxLayout(tab_traj)
        self.tabs.addTab(tab_traj, "1. Reference trajectory")

        traj_type = QGroupBox("Trajectory type")
        tt_l = QHBoxLayout()
        self.traj_combo = QComboBox()
        self.traj_combo.addItems(["Minimum snap (waypoints)", "Figure-eight (figure-8)"])
        tt_l.addWidget(QLabel("Type:"))
        tt_l.addWidget(self.traj_combo)
        traj_type.setLayout(tt_l)
        t1.addWidget(traj_type)

        self.snap_group = QGroupBox("Minimum snap waypoints (x,y,z m, yaw °, time s)")
        sg = QVBoxLayout()
        self.snap_table = QTableWidget(4, 5)
        self.snap_table.setHorizontalHeaderLabels(["x", "y", "z", "yaw°", "t [s]"])
        for r, row in enumerate(_snap_default_table()):
            for c, val in enumerate(row):
                self.snap_table.setItem(r, c, QTableWidgetItem(f"{val:g}"))
        self.snap_table.resizeColumnsToContents()
        sg.addWidget(self.snap_table)
        self.snap_group.setLayout(sg)
        t1.addWidget(self.snap_group)

        self.eight_group = QGroupBox("Figure-eight parameters")
        eg = QGridLayout()
        self.eight_cx = QDoubleSpinBox()
        self.eight_cy = QDoubleSpinBox()
        self.eight_cz = QDoubleSpinBox()
        for w, v in zip((self.eight_cx, self.eight_cy, self.eight_cz), (0.55, 0.05, 0.92)):
            w.setRange(-20, 20)
            w.setDecimals(3)
            w.setSingleStep(0.05)
            w.setValue(v)
        self.eight_a = QDoubleSpinBox()
        self.eight_a.setRange(0.05, 2.0)
        self.eight_a.setDecimals(3)
        self.eight_a.setValue(0.22)
        self.eight_period = QDoubleSpinBox()
        self.eight_period.setRange(0.5, 120.0)
        self.eight_period.setDecimals(2)
        self.eight_period.setValue(6.0)
        eg.addWidget(QLabel("Center cx"), 0, 0)
        eg.addWidget(self.eight_cx, 0, 1)
        eg.addWidget(QLabel("cy"), 0, 2)
        eg.addWidget(self.eight_cy, 0, 3)
        eg.addWidget(QLabel("cz"), 1, 0)
        eg.addWidget(self.eight_cz, 1, 1)
        eg.addWidget(QLabel("Half-width a [m]"), 1, 2)
        eg.addWidget(self.eight_a, 1, 3)
        eg.addWidget(QLabel("Period [s]"), 2, 0)
        eg.addWidget(self.eight_period, 2, 1)
        self.eight_group.setLayout(eg)
        t1.addWidget(self.eight_group)

        self.traj_combo.currentIndexChanged.connect(self._on_traj_type)
        self._on_traj_type()

        # ----- Tab 2: Tracking -----
        tab_track = QWidget()
        t2 = QVBoxLayout(tab_track)
        self.tabs.addTab(tab_track, "2. Tracking method")

        cm = QGroupBox("MPC control input (consistent with the -c command-line option)")
        cm_l = QVBoxLayout()
        self.control_combo = QComboBox()
        self.control_combo.addItems(
            [
                "Direct — quadrotor thrust + joint torques [T1..T4, τ1, τ2]",
                "High-level — body ω + total thrust + joint angle commands + first-order actuator",
            ]
        )
        cm_l.addWidget(self.control_combo)
        cm.setLayout(cm_l)
        t2.addWidget(cm)
        t2.addStretch()

        # ----- Tab 3: MPC -----
        tab_mpc = QWidget()
        t3 = QVBoxLayout(tab_mpc)
        self.tabs.addTab(tab_mpc, "3. MPC / Simulation")

        mpc_g = QGroupBox("MPC and simulation")
        mg = QGridLayout()
        r = 0

        def add_row(label, widget, row):
            mg.addWidget(QLabel(label), row, 0)
            mg.addWidget(widget, row, 1)

        self.dt_mpc = QDoubleSpinBox()
        self.dt_mpc.setRange(0.01, 0.5)
        self.dt_mpc.setDecimals(3)
        self.dt_mpc.setSingleStep(0.01)
        self.dt_mpc.setValue(0.05)
        add_row("dt_mpc [s]", self.dt_mpc, r)
        r += 1

        self.N_mpc = QSpinBox()
        self.N_mpc.setRange(5, 200)
        self.N_mpc.setValue(35)
        add_row("Horizon N", self.N_mpc, r)
        r += 1

        self.sim_dt = QDoubleSpinBox()
        self.sim_dt.setRange(0.0005, 0.2)
        self.sim_dt.setDecimals(4)
        self.sim_dt.setSingleStep(0.001)
        self.sim_dt.setValue(0.001)
        add_row("sim_dt [s] (RK4)", self.sim_dt, r)
        r += 1

        self.control_dt = QDoubleSpinBox()
        self.control_dt.setRange(0.001, 0.5)
        self.control_dt.setDecimals(4)
        self.control_dt.setSingleStep(0.005)
        self.control_dt.setValue(0.01)
        add_row("control_dt [s] (MPC period / ZOH)", self.control_dt, r)
        r += 1

        self.T_sim = QDoubleSpinBox()
        self.T_sim.setRange(0.5, 300.0)
        self.T_sim.setDecimals(2)
        self.T_sim.setValue(8.0)
        add_row("T_sim [s]", self.T_sim, r)
        r += 1

        self.w_ee = QDoubleSpinBox()
        self.w_ee.setRange(1.0, 1e5)
        self.w_ee.setDecimals(1)
        self.w_ee.setValue(400.0)
        add_row("w_ee (position LS)", self.w_ee, r)
        r += 1

        self.w_ee_yaw = QDoubleSpinBox()
        self.w_ee_yaw.setRange(0.0, 1e4)
        self.w_ee_yaw.setDecimals(1)
        self.w_ee_yaw.setValue(200.0)
        add_row("w_ee_yaw (heading cos/sin)", self.w_ee_yaw, r)
        r += 1

        self.mpc_max_iter = QSpinBox()
        self.mpc_max_iter.setRange(1, 500)
        self.mpc_max_iter.setValue(50)
        add_row("SQP max_iter / steps", self.mpc_max_iter, r)
        r += 1

        self.mpc_log_interval = QSpinBox()
        self.mpc_log_interval.setRange(0, 1000)
        self.mpc_log_interval.setValue(0)
        add_row("mpc_log_interval (0=silent)", self.mpc_log_interval, r)
        r += 1

        mpc_g.setLayout(mg)
        t3.addWidget(mpc_g)
        t3.addStretch()

        # ----- Global buttons (left panel bottom) -----
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run closed-loop tracking")
        self.run_btn.clicked.connect(self.run_tracking)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        if not EE_MPC_OK:
            self.run_btn.setEnabled(False)
            self.run_btn.setToolTip("Missing dependencies such as acados / pinocchio")
        btn_row.addWidget(self.run_btn)

        self.save_plot_btn = QPushButton("Save figures")
        self.save_plot_btn.clicked.connect(self.save_plots)
        self.save_plot_btn.setEnabled(False)
        btn_row.addWidget(self.save_plot_btn)

        self.save_params_btn = QPushButton("Save parameters")
        self.save_params_btn.clicked.connect(self.save_params)
        btn_row.addWidget(self.save_params_btn)

        self.load_params_btn = QPushButton("Load parameters")
        self.load_params_btn.clicked.connect(self.load_params)
        btn_row.addWidget(self.load_params_btn)

        left_l.addLayout(btn_row)

        log_g = QGroupBox("Log")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(140)
        log_l = QVBoxLayout()
        log_l.addWidget(self.log_text)
        log_g.setLayout(log_l)
        left_l.addWidget(log_g)

        if not EE_MPC_OK:
            err = repr(_ee_mpc_import_err) if _ee_mpc_import_err else "unknown"
            self.log(f"Warning: EE MPC module is unavailable ({err})")

        # ----- Right: plots -----
        right = QWidget()
        rl = QVBoxLayout(right)
        root.addWidget(right, stretch=1)

        self.plot_tabs = QTabWidget()
        rl.addWidget(self.plot_tabs)

        def make_tab(title: str):
            w = QWidget()
            l = QVBoxLayout(w)
            fig = Figure(figsize=(11, 8))
            canvas = FigureCanvas(fig)
            tb = NavigationToolbar(canvas, w)
            l.addWidget(tb)
            l.addWidget(canvas)
            self.plot_tabs.addTab(w, title)
            return fig, canvas

        self.fig_states, self.canvas_states = make_tab("States / controls")
        self.fig_3d, self.canvas_3d = make_tab("Base 3D")
        self.fig_dash, self.canvas_dash = make_tab("Tracking / MPC")
        self.fig_states_p, self.canvas_states_p = make_tab("Prev. States")
        self.fig_3d_p, self.canvas_3d_p = make_tab("Prev. 3D")
        self.fig_dash_p, self.canvas_dash_p = make_tab("Prev. Tracking")

    def log(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _on_traj_type(self):
        is_snap = self.traj_combo.currentIndex() == 0
        self.snap_group.setVisible(is_snap)
        self.eight_group.setVisible(not is_snap)

    def _read_snap_rows(self) -> list[list[float]]:
        rows = []
        for r in range(self.snap_table.rowCount()):
            row = []
            for c in range(5):
                it = self.snap_table.item(r, c)
                txt = it.text().strip() if it else "0"
                row.append(float(txt))
            rows.append(row)
        return rows

    def _collect_params(self) -> dict:
        track = "snap" if self.traj_combo.currentIndex() == 0 else "eight"
        cm = "direct" if self.control_combo.currentIndex() == 0 else "actuator_first_order"
        p = {
            "track": track,
            "snap_rows": self._read_snap_rows(),
            "eight_center": [self.eight_cx.value(), self.eight_cy.value(), self.eight_cz.value()],
            "eight_a": self.eight_a.value(),
            "eight_period": self.eight_period.value(),
            "control_mode": cm,
            "dt_mpc": self.dt_mpc.value(),
            "N_mpc": self.N_mpc.value(),
            "sim_dt": self.sim_dt.value(),
            "control_dt": self.control_dt.value(),
            "T_sim": self.T_sim.value(),
            "w_ee": self.w_ee.value(),
            "w_ee_yaw": self.w_ee_yaw.value(),
            "mpc_max_iter": self.mpc_max_iter.value(),
            "mpc_log_interval": self.mpc_log_interval.value(),
        }
        return p

    def run_tracking(self):
        if not EE_MPC_OK:
            QMessageBox.warning(self, "Error", "Dependencies not satisfied; cannot run.")
            return
        params = self._collect_params()
        self.run_btn.setEnabled(False)
        self.log("Starting closed-loop simulation…")
        self.worker = EeTrackingWorker(params)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, ok: bool, err: str, out: object):
        self.run_btn.setEnabled(True)
        if not ok:
            self.log("Failed:\n" + err)
            QMessageBox.critical(self, "Error", err[:1200])
            return

        assert isinstance(out, dict)
        if self._last_out is not None:
            self._prev_out = self._last_out
            self._render_out(self._prev_out, self.fig_states_p, self.fig_3d_p, self.fig_dash_p, suffix=" (previous)")
            self.canvas_states_p.draw()
            self.canvas_3d_p.draw()
            self.canvas_dash_p.draw()

        self._last_out = out
        self._render_out(out, self.fig_states, self.fig_3d, self.fig_dash, suffix="")
        self.canvas_states.draw()
        self.canvas_3d.draw()
        self.canvas_dash.draw()

        res = out["res"]
        self.log(
            f"Done | pos error (final) {res['err'][-1]:.4f} m | "
            f"yaw error (final) {res['err_yaw'][-1]:.4f} rad"
        )
        self.save_plot_btn.setEnabled(True)

    def _render_out(self, out: dict, fig_s, fig_3d, fig_dash, suffix: str = ""):
        import s500_uam_ee_snap_tracking_mpc as em

        res = out["res"]
        cm = out["control_mode"]
        wp = out["waypoints"] if out["track"] == "snap" else None

        fs = fig_s if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_into_figure else None
        f3 = fig_3d if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_3d_into_figure else None
        if fs is None:
            fig_s.clear()
            ax = fig_s.add_subplot(111)
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
            fig_3d.clear()
            ax = fig_3d.add_subplot(111, projection="3d")
            ax.text2D(0.2, 0.5, "3D plot unavailable", transform=ax.transAxes)
        em.render_ee_tracking_results_to_figures(
            res,
            fs,
            f3,
            fig_dash,
            control_mode=cm,
            plan_waypoints_xyz=wp,
            states_title=f"MPC closed-loop{suffix}",
        )

    def save_plots(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save States plot", "", "PNG (*.png)")
        if not path:
            return
        self.fig_states.savefig(path, dpi=200, bbox_inches="tight")
        stem = path[:-4] if path.lower().endswith(".png") else path
        self.fig_3d.savefig(stem + "_3d.png", dpi=200, bbox_inches="tight")
        self.fig_dash.savefig(stem + "_tracking.png", dpi=200, bbox_inches="tight")
        self.log(f"Saved: {path}, {stem}_3d.png, {stem}_tracking.png")

    def get_params_dict(self) -> dict:
        d = self._collect_params()
        d["traj_combo_index"] = self.traj_combo.currentIndex()
        d["control_combo_index"] = self.control_combo.currentIndex()
        return d

    def set_params_from_dict(self, d: dict):
        if "traj_combo_index" in d:
            self.traj_combo.setCurrentIndex(int(d["traj_combo_index"]))
        if "control_combo_index" in d:
            self.control_combo.setCurrentIndex(int(d["control_combo_index"]))
        rows = d.get("snap_rows")
        if rows and isinstance(rows, list):
            self.snap_table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, val in enumerate(row[:5]):
                    self.snap_table.setItem(r, c, QTableWidgetItem(str(val)))
        ec = d.get("eight_center")
        if ec and len(ec) >= 3:
            self.eight_cx.setValue(float(ec[0]))
            self.eight_cy.setValue(float(ec[1]))
            self.eight_cz.setValue(float(ec[2]))
        if "eight_a" in d:
            self.eight_a.setValue(float(d["eight_a"]))
        if "eight_period" in d:
            self.eight_period.setValue(float(d["eight_period"]))
        mapping = [
            ("dt_mpc", self.dt_mpc, float),
            ("N_mpc", self.N_mpc, int),
            ("sim_dt", self.sim_dt, float),
            ("control_dt", self.control_dt, float),
            ("T_sim", self.T_sim, float),
            ("w_ee", self.w_ee, float),
            ("w_ee_yaw", self.w_ee_yaw, float),
            ("mpc_max_iter", self.mpc_max_iter, int),
            ("mpc_log_interval", self.mpc_log_interval, int),
        ]
        for key, spin, cast in mapping:
            if key in d:
                spin.setValue(cast(d[key]))
        self._on_traj_type()

    def save_params(self):
        try:
            DEFAULT_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_PARAMS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.get_params_dict(), f, indent=2, ensure_ascii=False)
            self.log(f"Parameters saved: {DEFAULT_PARAMS_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_params(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load parameters", str(DEFAULT_PARAMS_PATH.parent), "JSON (*.json)")
        if not path:
            if DEFAULT_PARAMS_PATH.exists():
                path = str(DEFAULT_PARAMS_PATH)
            else:
                return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.set_params_from_dict(d)
            self.log(f"Loaded parameters: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = S500UAMEeTrackingGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
