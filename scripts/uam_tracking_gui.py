#!/usr/bin/env python3
"""
UAM full-state tracking GUI (Crocoddyl MPC + numeric simulation)

 - Reference trajectory: a single constant full-state (built from pose + joints + zero velocity)
 - Control: Crocoddyl Box-FDDP; running = state tracking + state regularization + control regularization; terminal = state tracking
 - Simulation: given T_sim, sim_dt, control_dt (ZOH), consistent with the run_numeric_sim approach

Usage:
    python uam_tracking_gui.py

Dependencies: PyQt5, matplotlib, numpy, pinocchio, crocoddyl
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

_MPC_IMPORT_ERR: Exception | None = None
try:
    import s500_uam_crocoddyl_state_tracking_mpc as st_mpc
    from s500_uam_trajectory_planner import make_uam_state

    MPC_OK = True
except Exception as e:
    st_mpc = None  # type: ignore
    make_uam_state = None  # type: ignore
    MPC_OK = False
    _MPC_IMPORT_ERR = e

_EE_RENDER_ERR: Exception | None = None
try:
    import s500_uam_ee_snap_tracking_mpc as ee_render_mpc

    EE_RENDER_OK = bool(getattr(ee_render_mpc, "render_ee_tracking_results_to_figures", None))
except Exception as e:
    ee_render_mpc = None  # type: ignore
    EE_RENDER_OK = False
    _EE_RENDER_ERR = e

DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "yaml"
    / "trajectories"
    / "uam_tracking_gui_params.json"
)


def _full_state_from_ui(
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
    j1_deg: float,
    j2_deg: float,
) -> np.ndarray:
    deg = np.pi / 180.0
    qv = make_uam_state(x, y, z, j1=j1_deg * deg, j2=j2_deg * deg, yaw=yaw_deg * deg)
    return qv


class TrackingWorker(QThread):
    finished = pyqtSignal(bool, str, object)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            import s500_uam_crocoddyl_state_tracking_mpc as sm

            p = self.params
            x_ref = _full_state_from_ui(
                p["ref_x"],
                p["ref_y"],
                p["ref_z"],
                p["ref_yaw_deg"],
                p["ref_j1_deg"],
                p["ref_j2_deg"],
            )
            x0 = _full_state_from_ui(
                p["x0_x"],
                p["x0_y"],
                p["x0_z"],
                p["x0_yaw_deg"],
                p["x0_j1_deg"],
                p["x0_j2_deg"],
            )
            x_nom = sm.default_hover_nominal()
            out = sm.run_closed_loop_state_tracking(
                x0=x0,
                x_ref=x_ref,
                x_nom=x_nom,
                T_sim=p["T_sim"],
                sim_dt=p["sim_dt"],
                control_dt=p["control_dt"],
                dt_mpc=p["dt_mpc"],
                horizon=p["N_mpc"],
                w_state_track=p["w_track"],
                w_state_reg=p["w_reg"],
                w_control=p["w_u"],
                w_terminal_track=p["w_term"],
                mpc_max_iter=p["mpc_max_iter"],
                use_thrust_constraints=p["thrust_constraints"],
                s500_yaml_path=p.get("s500_yaml"),
                urdf_path=p.get("urdf_path"),
                verbose=False,
            )
            self.finished.emit(True, "", out)
        except Exception:
            self.finished.emit(False, traceback.format_exc(), None)


def _ensure_res_dict(out: dict) -> dict:
    """Add the ``res`` field for ``render_ee_tracking_results_to_figures``."""
    if "res" not in out and st_mpc is not None:
        out = dict(out)
        out["res"] = st_mpc.crocoddyl_closed_loop_to_ee_tracking_res(out)
    return out


def render_uam_tracking_to_figures(
    out: dict,
    fig_states: Figure,
    fig_3d: Figure,
    fig_dash: Figure,
    suffix: str = "",
) -> None:
    """Same pipeline as ``s500_uam_ee_tracking_gui._render_out`` (acados 4x4 + 3D + dashboard)."""
    out = _ensure_res_dict(out)
    res = out["res"]
    cm = res["control_mode"]
    x_ref = np.asarray(out["x_ref"], dtype=float)
    wp = np.array([[x_ref[0], x_ref[1], x_ref[2]]], dtype=float)

    if not EE_RENDER_OK or ee_render_mpc is None:
        fig_states.clear()
        ax = fig_states.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Plot module unavailable\n(needs s500_uam_ee_snap_tracking_mpc)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig_3d.clear()
        ax3 = fig_3d.add_subplot(111, projection="3d")
        ax3.text2D(0.2, 0.5, "3D unavailable", transform=ax3.transAxes)
        fig_dash.clear()
        axd = fig_dash.add_subplot(111)
        axd.text(0.5, 0.5, "Dashboard unavailable", ha="center", va="center", transform=axd.transAxes)
        axd.axis("off")
        return

    em = ee_render_mpc
    fs = fig_states if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_into_figure else None
    f3 = fig_3d if em.PLOT_ACADOS_GUI_STYLE and em.plot_acados_3d_into_figure else None
    if fs is None:
        fig_states.clear()
        ax = fig_states.add_subplot(111)
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
        states_title=f"Crocoddyl full-state MPC closed loop{suffix}",
    )


class UamTrackingGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker: TrackingWorker | None = None
        self._last_out: dict | None = None
        self._prev_out: dict | None = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("UAM — Full-state tracking MPC (Crocoddyl)")
        self.setMinimumSize(1100, 760)
        self.resize(1500, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QWidget()
        ll = QVBoxLayout(left)
        root.addWidget(left, stretch=0)

        tabs = QTabWidget()
        ll.addWidget(tabs)

        # --- Tab 1: Reference & initial ---
        t1 = QWidget()
        g1 = QGridLayout(t1)
        row = 0

        def add_spin(r, label, spin):
            g1.addWidget(QLabel(label), r, 0)
            g1.addWidget(spin, r, 1)

        g1.addWidget(QLabel("<b>Reference state (full-state tracking target)</b>"), row, 0, 1, 2)
        row += 1
        self.ref_x = QDoubleSpinBox()
        self.ref_y = QDoubleSpinBox()
        self.ref_z = QDoubleSpinBox()
        for w, v in zip((self.ref_x, self.ref_y, self.ref_z), (0.8, 0.0, 1.2)):
            w.setRange(-50, 50)
            w.setDecimals(3)
            w.setSingleStep(0.05)
            w.setValue(v)
        add_spin(row, "ref x [m]", self.ref_x)
        row += 1
        add_spin(row, "ref y [m]", self.ref_y)
        row += 1
        add_spin(row, "ref z [m]", self.ref_z)
        row += 1
        self.ref_yaw = QDoubleSpinBox()
        self.ref_j1 = QDoubleSpinBox()
        self.ref_j2 = QDoubleSpinBox()
        self.ref_yaw.setRange(-180, 180)
        self.ref_j1.setRange(-180, 180)
        self.ref_j2.setRange(-180, 180)
        for w, v in zip((self.ref_yaw, self.ref_j1, self.ref_j2), (0.0, -45.0, -30.0)):
            w.setDecimals(2)
            w.setValue(v)
        add_spin(row, "ref yaw [deg]", self.ref_yaw)
        row += 1
        add_spin(row, "ref j1 [deg]", self.ref_j1)
        row += 1
        add_spin(row, "ref j2 [deg]", self.ref_j2)
        row += 1

        g1.addWidget(QLabel("<b>Initial state</b>"), row, 0, 1, 2)
        row += 1
        self.x0_x = QDoubleSpinBox()
        self.x0_y = QDoubleSpinBox()
        self.x0_z = QDoubleSpinBox()
        for w, v in zip((self.x0_x, self.x0_y, self.x0_z), (0.0, 0.0, 1.0)):
            w.setRange(-50, 50)
            w.setDecimals(3)
            w.setSingleStep(0.05)
            w.setValue(v)
        add_spin(row, "x0 x [m]", self.x0_x)
        row += 1
        add_spin(row, "x0 y [m]", self.x0_y)
        row += 1
        add_spin(row, "x0 z [m]", self.x0_z)
        row += 1
        self.x0_yaw = QDoubleSpinBox()
        self.x0_j1 = QDoubleSpinBox()
        self.x0_j2 = QDoubleSpinBox()
        self.x0_yaw.setRange(-180, 180)
        self.x0_j1.setRange(-180, 180)
        self.x0_j2.setRange(-180, 180)
        for w, v in zip((self.x0_yaw, self.x0_j1, self.x0_j2), (0.0, -45.0, -30.0)):
            w.setDecimals(2)
            w.setValue(v)
        add_spin(row, "x0 yaw [deg]", self.x0_yaw)
        row += 1
        add_spin(row, "x0 j1 [deg]", self.x0_j1)
        row += 1
        add_spin(row, "x0 j2 [deg]", self.x0_j2)
        row += 1

        tabs.addTab(t1, "1. Reference / Initial")

        # --- Tab 2: Sim + MPC ---
        t2 = QWidget()
        g2 = QGridLayout(t2)
        r2 = 0

        def add2(label, w, rr):
            g2.addWidget(QLabel(label), rr, 0)
            g2.addWidget(w, rr, 1)

        self.T_sim = QDoubleSpinBox()
        self.T_sim.setRange(0.5, 120.0)
        self.T_sim.setDecimals(2)
        self.T_sim.setValue(6.0)
        add2("T_sim [s]", self.T_sim, r2)
        r2 += 1

        self.sim_dt = QDoubleSpinBox()
        self.sim_dt.setRange(0.0005, 0.05)
        self.sim_dt.setDecimals(4)
        self.sim_dt.setValue(0.002)
        add2("sim_dt [s]", self.sim_dt, r2)
        r2 += 1

        self.control_dt = QDoubleSpinBox()
        self.control_dt.setRange(0.005, 0.5)
        self.control_dt.setDecimals(4)
        self.control_dt.setValue(0.04)
        add2("control_dt [s] (ZOH)", self.control_dt, r2)
        r2 += 1

        self.dt_mpc = QDoubleSpinBox()
        self.dt_mpc.setRange(0.01, 0.3)
        self.dt_mpc.setDecimals(3)
        self.dt_mpc.setValue(0.04)
        add2("dt_mpc [s]", self.dt_mpc, r2)
        r2 += 1

        self.N_mpc = QSpinBox()
        self.N_mpc.setRange(5, 80)
        self.N_mpc.setValue(25)
        add2("Horizon N", self.N_mpc, r2)
        r2 += 1

        self.w_track = QDoubleSpinBox()
        self.w_track.setRange(0.01, 1e4)
        self.w_track.setDecimals(3)
        self.w_track.setValue(15.0)
        add2("w state tracking (running)", self.w_track, r2)
        r2 += 1

        self.w_reg = QDoubleSpinBox()
        self.w_reg.setRange(0.0, 100.0)
        self.w_reg.setDecimals(4)
        self.w_reg.setValue(0.2)
        add2("w state regularization → nominal hover", self.w_reg, r2)
        r2 += 1

        self.w_u = QDoubleSpinBox()
        self.w_u.setRange(1e-6, 1.0)
        self.w_u.setDecimals(6)
        self.w_u.setValue(2e-3)
        add2("w control regularization", self.w_u, r2)
        r2 += 1

        self.w_term = QDoubleSpinBox()
        self.w_term.setRange(0.1, 1e5)
        self.w_term.setDecimals(2)
        self.w_term.setValue(200.0)
        add2("w terminal tracking", self.w_term, r2)
        r2 += 1

        self.mpc_max_iter = QSpinBox()
        self.mpc_max_iter.setRange(10, 300)
        self.mpc_max_iter.setValue(80)
        add2("MPC max_iter", self.mpc_max_iter, r2)
        r2 += 1

        tabs.addTab(t2, "2. Simulation / MPC")

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run closed-loop simulation")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        if not MPC_OK:
            self.run_btn.setEnabled(False)
        btn_row.addWidget(self.run_btn)

        self.save_btn = QPushButton("Save figures")
        self.save_btn.clicked.connect(self._save_plots)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.save_btn)

        self.save_p_btn = QPushButton("Save parameters")
        self.save_p_btn.clicked.connect(self._save_params)
        btn_row.addWidget(self.save_p_btn)

        self.load_p_btn = QPushButton("Load parameters")
        self.load_p_btn.clicked.connect(self._load_params)
        btn_row.addWidget(self.load_p_btn)

        ll.addLayout(btn_row)

        log_g = QGroupBox("Log")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_g.setLayout(QVBoxLayout())
        log_g.layout().addWidget(self.log_text)
        ll.addWidget(log_g)

        if not MPC_OK:
            err = repr(_MPC_IMPORT_ERR) if _MPC_IMPORT_ERR else "unknown"
            self._log(f"Unable to load MPC module: {err}")
        if not EE_RENDER_OK:
            er2 = repr(_EE_RENDER_ERR) if _EE_RENDER_ERR else "unknown"
            self._log(f"Note: EE plot module not ready; using placeholder plot ({er2})")

        # --- Right: figures ---
        right = QWidget()
        rl = QVBoxLayout(right)
        root.addWidget(right, stretch=1)

        self.plot_tabs = QTabWidget()
        rl.addWidget(self.plot_tabs)

        def make_tab(title: str):
            w = QWidget()
            l = QVBoxLayout(w)
            fig = Figure(figsize=(11, 8))
            cv = FigureCanvas(fig)
            l.addWidget(NavigationToolbar(cv, w))
            l.addWidget(cv)
            self.plot_tabs.addTab(w, title)
            return fig, cv

        self.fig_s, self.cv_s = make_tab("States / Controls")
        self.fig_3d, self.cv_3d = make_tab("Base 3D")
        self.fig_mpc, self.cv_mpc = make_tab("Tracking / MPC")
        self.fig_s_p, self.cv_s_p = make_tab("Prev. States")
        self.fig_3d_p, self.cv_3d_p = make_tab("Prev. 3D")
        self.fig_mpc_p, self.cv_mpc_p = make_tab("Prev. Tracking")

    def _log(self, s: str):
        self.log_text.append(s)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _collect_params(self) -> dict:
        root = Path(__file__).resolve().parent.parent
        return {
            "ref_x": self.ref_x.value(),
            "ref_y": self.ref_y.value(),
            "ref_z": self.ref_z.value(),
            "ref_yaw_deg": self.ref_yaw.value(),
            "ref_j1_deg": self.ref_j1.value(),
            "ref_j2_deg": self.ref_j2.value(),
            "x0_x": self.x0_x.value(),
            "x0_y": self.x0_y.value(),
            "x0_z": self.x0_z.value(),
            "x0_yaw_deg": self.x0_yaw.value(),
            "x0_j1_deg": self.x0_j1.value(),
            "x0_j2_deg": self.x0_j2.value(),
            "T_sim": self.T_sim.value(),
            "sim_dt": self.sim_dt.value(),
            "control_dt": self.control_dt.value(),
            "dt_mpc": self.dt_mpc.value(),
            "N_mpc": self.N_mpc.value(),
            "w_track": self.w_track.value(),
            "w_reg": self.w_reg.value(),
            "w_u": self.w_u.value(),
            "w_term": self.w_term.value(),
            "mpc_max_iter": self.mpc_max_iter.value(),
            "thrust_constraints": True,
            "s500_yaml": str(root / "config" / "yaml" / "multicopter" / "s500.yaml"),
            "urdf_path": str(root / "models" / "urdf" / "s500_uam_simple.urdf"),
        }

    def _run(self):
        if not MPC_OK:
            QMessageBox.warning(self, "Error", "Dependencies not satisfied (pinocchio / crocoddyl).")
            return
        self.run_btn.setEnabled(False)
        self._log("Starting simulation…")
        self.worker = TrackingWorker(self._collect_params())
        self.worker.finished.connect(self._on_done)
        self.worker.start()

    def _on_done(self, ok: bool, err: str, out: object):
        self.run_btn.setEnabled(True)
        if not ok:
            self._log(err[:2000])
            QMessageBox.critical(self, "Error", err[:1500])
            return
        assert isinstance(out, dict)
        out = _ensure_res_dict(out)
        if self._last_out is not None:
            self._prev_out = self._last_out
            render_uam_tracking_to_figures(
                self._prev_out, self.fig_s_p, self.fig_3d_p, self.fig_mpc_p, suffix=" (previous)"
            )
            self.cv_s_p.draw()
            self.cv_3d_p.draw()
            self.cv_mpc_p.draw()

        self._last_out = out
        render_uam_tracking_to_figures(out, self.fig_s, self.fig_3d, self.fig_mpc, suffix="")
        self.cv_s.draw()
        self.cv_3d.draw()
        self.cv_mpc.draw()

        res = out["res"]
        self._log(
            f"Done | EE position error (final) {res['err'][-1]:.4f} m | "
            f"yaw error (final) {res['err_yaw'][-1]:.4f} rad | "
            f"full-state norm (final) {out['track_norm'][-1]:.4f}"
        )
        self.save_btn.setEnabled(True)

    def _save_plots(self):
        if self._last_out is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save States figure", "", "PNG (*.png)")
        if not path:
            return
        self.fig_s.savefig(path, dpi=200, bbox_inches="tight")
        stem = path[:-4] if path.lower().endswith(".png") else path
        self.fig_3d.savefig(stem + "_3d.png", dpi=200, bbox_inches="tight")
        self.fig_mpc.savefig(stem + "_tracking.png", dpi=200, bbox_inches="tight")
        self._log(f"Saved: {path}, {stem}_3d.png, {stem}_tracking.png")

    def _save_params(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters", str(DEFAULT_PARAMS_PATH), "JSON (*.json)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._collect_params(), f, indent=2)
        self._log(f"Saved parameters: {path}")

    def _load_params(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load parameters", str(DEFAULT_PARAMS_PATH.parent))
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        m = {
            "ref_x": self.ref_x,
            "ref_y": self.ref_y,
            "ref_z": self.ref_z,
            "ref_yaw_deg": self.ref_yaw,
            "ref_j1_deg": self.ref_j1,
            "ref_j2_deg": self.ref_j2,
            "x0_x": self.x0_x,
            "x0_y": self.x0_y,
            "x0_z": self.x0_z,
            "x0_yaw_deg": self.x0_yaw,
            "x0_j1_deg": self.x0_j1,
            "x0_j2_deg": self.x0_j2,
            "T_sim": self.T_sim,
            "sim_dt": self.sim_dt,
            "control_dt": self.control_dt,
            "dt_mpc": self.dt_mpc,
            "N_mpc": self.N_mpc,
            "w_track": self.w_track,
            "w_reg": self.w_reg,
            "w_u": self.w_u,
            "w_term": self.w_term,
            "mpc_max_iter": self.mpc_max_iter,
        }
        for k, w in m.items():
            if k not in d:
                continue
            if isinstance(w, QSpinBox):
                w.setValue(int(d[k]))
            else:
                w.setValue(float(d[k]))
        self._log(f"Loaded: {path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = UamTrackingGui()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
