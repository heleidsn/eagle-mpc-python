#!/usr/bin/env python3
"""
S500 UAM — Crocoddyl tracking MPC (unified dynamics, two cost modes)

- **Full-state**: running / terminal costs on ``ResidualModelState`` (track + reg + control).
- **EE pose**: costs on EE translation / orientation / velocity (+ optional state reg / plan state track).

Same multibody dynamics and actuation; only the optimal-control objective differs. Use
``UAMCrocoddylTrackingMPC`` with ``mode=MODE_FULL_STATE`` or ``MODE_EE_POSE``, or the thin subclasses
``UAMCrocoddylStateTrackingMPC`` / ``UAMEEPoseTrackingCrocoddylMPC``.

Simulation: ``sim_dt`` integration, ``control_dt`` ZOH; optional actuator first-order lag before the plant step;
optional **sim-only** EE payload (same sphere-inertia hack as EE-pose closed loop; MPC keeps nominal model).
Optional ``sim_control_stack="px4_rate"`` inserts a PX4-like layer (integrated body-rate setpoint, rate PD, thruster mixer)
before the plant; see ``s500_uam_px4_style_rate_sim``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
import crocoddyl

from s500_uam_closed_loop_plant import (
    CrocoddylEulerPlant,
    PayloadSchedulePlant,
    crocoddyl_euler_step,
    mpc_inner_stride,
)

from s500_uam_trajectory_planner import (
    S500UAMTrajectoryPlanner,
    compute_ee_kinematics_along_trajectory,
    make_uam_state,
)

from s500_uam_px4_style_rate_sim import (
    px4_rate_compute_plant_u,
    bodyrate_from_horizon,
    ctbr_rate_pid_mix,
)


def _apply_first_order_actuator(
    u_act: np.ndarray,
    u_cmd: np.ndarray,
    *,
    tau_thrust: float,
    tau_theta: float,
    dt: float,
) -> np.ndarray:
    """
    First-order actuator discrete update (Euler):
      du/dt = (u_cmd - u_act) / tau
    Here thrust (T1..T4) and joint torques (tau1..tau2) use different tau values.
    """
    u_act = np.asarray(u_act, dtype=float).reshape(-1)
    u_cmd = np.asarray(u_cmd, dtype=float).reshape(-1)
    if u_act.shape != u_cmd.shape:
        raise ValueError(f"u_act/u_cmd shape mismatch: {u_act.shape} vs {u_cmd.shape}")
    if u_act.size < 6:
        raise ValueError(f"expect at least 6 control dims (T1..T4, tau1..tau2), got {u_act.size}")

    out = u_act.copy()
    if tau_thrust is not None and tau_thrust > 0:
        out[:4] = out[:4] + dt * (u_cmd[:4] - out[:4]) / float(tau_thrust)
    else:
        out[:4] = u_cmd[:4]

    if tau_theta is not None and tau_theta > 0:
        out[4:6] = out[4:6] + dt * (u_cmd[4:6] - out[4:6]) / float(tau_theta)
    else:
        out[4:6] = u_cmd[4:6]

    return out


def interp_full_state_piecewise(
    t_query: float,
    t_nodes: np.ndarray,
    x_nodes: np.ndarray,
    robot_model: pin.Model,
) -> np.ndarray:
    """Piecewise geodesic interp in q, linear in v, clamped to [t_nodes[0], t_nodes[-1]]."""
    t_nodes = np.asarray(t_nodes, dtype=float).flatten()
    x_nodes = np.asarray(x_nodes, dtype=float)
    if len(t_nodes) < 2 or len(x_nodes) < 2:
        return np.asarray(x_nodes[0], dtype=float).flatten()
    tq = float(np.clip(t_query, t_nodes[0], t_nodes[-1]))
    idx = int(np.searchsorted(t_nodes, tq, side="right"))
    i = max(1, min(idx, len(t_nodes) - 1))
    t0, t1 = float(t_nodes[i - 1]), float(t_nodes[i])
    alpha = (tq - t0) / (t1 - t0) if t1 > t0 + 1e-15 else 0.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    nq = robot_model.nq
    nv = robot_model.nv
    qa = x_nodes[i - 1][:nq]
    qb = x_nodes[i][:nq]
    va = x_nodes[i - 1][nq : nq + nv]
    vb = x_nodes[i][nq : nq + nv]
    q = pin.interpolate(robot_model, qa, qb, alpha)
    v = (1.0 - alpha) * va + alpha * vb
    return np.concatenate([q, v])


@dataclass
class EETrackingWeights:
    """Running-cost scales for EE-pose mode (see ``UAMCrocoddylTrackingMPC.MODE_EE_POSE``)."""

    w_pos: float = 10.0
    w_rot_rp: float = 1.0
    w_rot_yaw: float = 1.0
    w_vel_lin: float = 1.0
    w_vel_ang_rp: float = 1.0
    w_vel_ang_yaw: float = 1.0
    w_u: float = 0.0
    w_terminal_scale: float = 3.0
    # 弱基座参考（沿 x_plan，不锁关节）：抑制 EE 零空间漂移
    w_base_pos: float = 0.0
    w_base_yaw: float = 0.0
    w_state_reg: float = 0.0
    w_state_track: float = 0.0
    # x_plan 分项激活（仅当 w_state_track > 0）；关节权宜小，避免锁死臂
    w_st_pos: float = 1.0
    w_st_att: float = 1.0
    w_st_joint: float = 0.2
    w_st_vel: float = 0.1
    w_st_omega: float = 0.1
    w_st_joint_vel: float = 0.05


BASE_FRAME_NAME = "base_link"


def _yaw_from_full_state(x: np.ndarray) -> float:
    q = np.asarray(x, dtype=float).flatten()[3:7]
    R = pin.Quaternion(q).toRotationMatrix()
    return float(np.arctan2(float(R[1, 0]), float(R[0, 0])))


def interp_ref_pose(
    tq: float,
    t_ref: np.ndarray,
    p_ref: np.ndarray,
    yaw_ref: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Linear interpolation for EE position and yaw (yaw unwrapped)."""
    t_ref = np.asarray(t_ref, dtype=float).flatten()
    p_ref = np.asarray(p_ref, dtype=float)
    yaw_ref = np.asarray(yaw_ref, dtype=float).flatten()
    if len(t_ref) < 2:
        raise ValueError("t_ref must have at least 2 points")
    if p_ref.shape != (len(t_ref), 3):
        raise ValueError(f"p_ref must have shape (len(t_ref),3); got {p_ref.shape}")
    if len(yaw_ref) != len(t_ref):
        raise ValueError("yaw_ref length mismatch with t_ref")
    tq = float(np.clip(tq, t_ref[0], t_ref[-1]))
    px = float(np.interp(tq, t_ref, p_ref[:, 0]))
    py = float(np.interp(tq, t_ref, p_ref[:, 1]))
    pz = float(np.interp(tq, t_ref, p_ref[:, 2]))
    yaw_u = np.unwrap(yaw_ref)
    yaw = float(np.interp(tq, t_ref, yaw_u))
    return np.array([px, py, pz], dtype=float), yaw


def _cost_group_from_name(name: str) -> str:
    s = str(name).lower()
    if "u_" in s or "control" in s or "act" in s:
        return "action"
    if s.startswith("x_") or "state" in s:
        return "state"
    if s.startswith("ee_") or "frame" in s or "pose" in s or "yaw" in s:
        return "ee"
    return "other"


def _extract_solver_cost_terms(
    solver: crocoddyl.SolverBoxFDDP,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Aggregate weighted cost terms by stage/group and return per-term coefficients."""
    out: Dict[str, float] = {}
    grp: Dict[str, float] = {}
    coeffs: Dict[str, float] = {}

    def _iter_cost_items(costs_obj):
        if costs_obj is None:
            return []
        if hasattr(costs_obj, "items"):
            try:
                return list(costs_obj.items())
            except Exception:
                pass
        if hasattr(costs_obj, "todict"):
            try:
                d = costs_obj.todict()
                if isinstance(d, dict):
                    return list(d.items())
            except Exception:
                pass
        if hasattr(costs_obj, "keys"):
            try:
                ks = list(costs_obj.keys())
                return [(k, costs_obj[k]) for k in ks]
            except Exception:
                pass
        try:
            d = dict(costs_obj)
            return list(d.items())
        except Exception:
            return []

    def _accumulate(stage: str, cost_model_sum, cost_data_sum) -> None:
        model_costs = getattr(cost_model_sum, "costs", None)
        costs = getattr(cost_data_sum, "costs", None)
        if costs is None and model_costs is None:
            return
        data_map = {str(k): v for k, v in _iter_cost_items(costs)}
        model_map = {str(k): v for k, v in _iter_cost_items(model_costs)}
        for name in sorted(set(data_map.keys()) | set(model_map.keys())):
            m_item = model_map.get(name)
            d_item = data_map.get(name)
            w = float(getattr(m_item, "weight", 1.0)) if m_item is not None else 1.0
            c_raw = float(getattr(d_item, "cost", 0.0)) if d_item is not None else 0.0
            c = w * c_raw
            key = f"{stage}/{name}"
            out[key] = float(out.get(key, 0.0)) + c
            coeffs[key] = float(w)

    problem = solver.problem
    running_models = list(getattr(problem, "runningModels", []))
    running_datas = list(getattr(problem, "runningDatas", []))
    for m, d in zip(running_models, running_datas):
        mcost = getattr(getattr(m, "differential", None), "costs", None)
        dcost = getattr(getattr(d, "differential", None), "costs", None)
        _accumulate(
            "running",
            mcost if mcost is not None else getattr(m, "costs", None),
            dcost if dcost is not None else getattr(d, "costs", None),
        )
    term_model = getattr(problem, "terminalModel", None)
    td = getattr(problem, "terminalData", None)
    if td is not None:
        mcost = getattr(getattr(term_model, "differential", None), "costs", None)
        dcost = getattr(getattr(td, "differential", None), "costs", None)
        _accumulate(
            "terminal",
            mcost if mcost is not None else getattr(term_model, "costs", None),
            dcost if dcost is not None else getattr(td, "costs", None),
        )
    # Enforce consistency once per solve (across running + terminal together).
    s = float(np.nansum(np.asarray(list(out.values()), dtype=float))) if out else 0.0
    total = float(getattr(solver, "cost", np.nan))
    if out and np.isfinite(total) and np.isfinite(s) and abs(s) > 1e-12:
        scale = total / s
        for k in list(out.keys()):
            out[k] = float(out[k]) * float(scale)
    # Build grouped costs from the (possibly scaled) term costs.
    for key, c in out.items():
        stage, name = key.split("/", 1) if "/" in key else ("running", key)
        gk = f"{stage}/{_cost_group_from_name(name)}"
        grp[gk] = float(grp.get(gk, 0.0)) + float(c)
    return out, grp, coeffs


def _yaw_to_rotation_matrix(yaw: float, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    return pin.rpy.rpyToMatrix(roll, pitch, yaw)


def _parent_joint_id_for_frame(model: pin.Model, frame_id: int) -> int:
    f = model.frames[frame_id]
    if hasattr(f, "parentJoint"):
        return int(f.parentJoint)
    return int(f.parent)


def solid_sphere_principal_inertias(mass: float, radius: float) -> Tuple[float, float, float]:
    """Ixx = Iyy = Izz = 2/5 * m * r^2 (uniform solid sphere)."""
    m = float(mass)
    r = float(radius)
    ii = 0.4 * m * r * r
    return ii, ii, ii


def _apply_payload_inertia_on_plant_model(
    model: pin.Model,
    ee_frame_id: int,
    mass: float,
    com_local: np.ndarray,
    ixx: float,
    iyy: float,
    izz: float,
) -> None:
    m = float(mass)
    if m <= 1e-9:
        return
    com = np.asarray(com_local, dtype=float).reshape(3)
    eps = 1e-6
    d_ixx = max(float(ixx), eps)
    d_iyy = max(float(iyy), eps)
    d_izz = max(float(izz), eps)
    I3 = np.diag([d_ixx, d_iyy, d_izz])
    jid = _parent_joint_id_for_frame(model, ee_frame_id)
    I_add = pin.Inertia(m, com, I3)
    model.inertias[jid] = model.inertias[jid] + I_add


class UAMCrocoddylTrackingMPC:
    """Receding-horizon Box-FDDP MPC: full-state **or** EE-pose costs; same Pinocchio/Crocoddyl plant."""

    MODE_FULL_STATE = "full_state"
    MODE_EE_POSE = "ee_pose"

    def __init__(
        self,
        *,
        mode: str = "full_state",
        s500_yaml_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        dt_mpc: float = 0.05,
        horizon: int = 25,
        use_thrust_constraints: bool = True,
        w_state_track: float = 10.0,
        w_state_reg: float = 0.1,
        w_control: float = 1e-3,
        w_terminal_track: float = 100.0,
        w_pos: float = 1.0,
        w_att: float = 1.0,
        w_joint: float = 1.0,
        w_vel: float = 1.0,
        w_omega: float = 1.0,
        w_joint_vel: float = 1.0,
        w_u_thrust: float = 1.0,
        w_u_joint_torque: float = 1.0,
        ee_weights: Optional[EETrackingWeights] = None,
    ):
        if mode not in (self.MODE_FULL_STATE, self.MODE_EE_POSE):
            raise ValueError(f"Unknown mode {mode!r}; use MODE_FULL_STATE or MODE_EE_POSE")
        self.mode = mode
        self.dt_mpc = float(dt_mpc)
        self.horizon = int(horizon)
        self.use_thrust_constraints = bool(use_thrust_constraints)

        self._planner = S500UAMTrajectoryPlanner(
            s500_yaml_path=s500_yaml_path, urdf_path=urdf_path
        )
        self.state = self._planner.state
        self.actuation = self._planner.actuation
        self.robot_model = self._planner.robot_model
        self.robot_data = self._planner.robot_data
        self.s500_config = self._planner.s500_config
        self.ee_frame_id = self._planner.ee_frame_id
        try:
            self.base_frame_id = int(self.robot_model.getFrameId(BASE_FRAME_NAME))
        except Exception:
            self.base_frame_id = -1
        if self.base_frame_id < 0 or self.base_frame_id >= len(self.robot_model.frames):
            self.base_frame_id = -1
        self.nu = self.actuation.nu
        self.nq = self.robot_model.nq
        self.nv = self.robot_model.nv

        if mode == self.MODE_EE_POSE:
            self.w: EETrackingWeights = ee_weights if ee_weights is not None else EETrackingWeights()
        else:
            self.w_state_track = float(w_state_track)
            self.w_state_reg = float(w_state_reg)
            self.w_control = float(w_control)
            self.w_terminal_track = float(w_terminal_track)
            # Per-group state/control unit-balancing weights for full-state MPC.
            self.w_pos = float(w_pos)
            self.w_att = float(w_att)
            self.w_joint = float(w_joint)
            self.w_vel = float(w_vel)
            self.w_omega = float(w_omega)
            self.w_joint_vel = float(w_joint_vel)
            self.w_u_thrust = float(w_u_thrust)
            self.w_u_joint_torque = float(w_u_joint_torque)

        # Hover thrust reference should use the full system mass (base + all arm links).
        mass = float(sum(inertia.mass for inertia in self.robot_model.inertias))
        self._hover_thrust = mass * 9.81 / 4.0
        self._u_ref = np.array([self._hover_thrust] * 4 + [0.0] * (self.nu - 4))

        if use_thrust_constraints:
            p = self.s500_config["platform"]
            lb = [p["min_thrust"]] * min(4, self.nu)
            ub = [p["max_thrust"]] * min(4, self.nu)
            if self.nu > 4:
                lb += [-2.0] * (self.nu - 4)
                ub += [2.0] * (self.nu - 4)
            self._u_lb = np.asarray(lb, dtype=float)
            self._u_ub = np.asarray(ub, dtype=float)
        else:
            self._u_lb = -1e6 * np.ones(self.nu)
            self._u_ub = 1e6 * np.ones(self.nu)

    # ----- Full-state costs -----

    def _state_track_activation_weights(self) -> np.ndarray:
        """Per-tangent weights for optional x_plan state track (EE or full-state)."""
        n_joint_q = max(0, int(self.nq) - 7)
        n_joint_v = max(0, int(self.nv) - 6)
        if self.mode == self.MODE_EE_POSE:
            w = self.w
            w_pos = float(w.w_st_pos)
            w_att = float(w.w_st_att)
            w_joint = float(w.w_st_joint)
            w_vel = float(w.w_st_vel)
            w_omega = float(w.w_st_omega)
            w_joint_vel = float(w.w_st_joint_vel)
        else:
            w_pos = float(self.w_pos)
            w_att = float(self.w_att)
            w_joint = float(self.w_joint)
            w_vel = float(self.w_vel)
            w_omega = float(self.w_omega)
            w_joint_vel = float(self.w_joint_vel)
        act = (
            [w_pos] * 3
            + [w_att] * 3
            + [w_joint] * n_joint_q
            + [w_vel] * 3
            + [w_omega] * 3
            + [w_joint_vel] * n_joint_v
        )
        act = np.asarray(act, dtype=np.float64)
        if act.size != int(self.state.ndx):
            raise ValueError(
                f"state track activation size mismatch: got {act.size}, expected ndx={self.state.ndx}"
            )
        return act

    def _full_state_activation_weights(self) -> np.ndarray:
        """Weights on state tangent [pos(3), att(3), joints, vel(3), omega(3), joint_vels]."""
        n_joint_q = max(0, int(self.nq) - 7)
        n_joint_v = max(0, int(self.nv) - 6)
        w = (
            [self.w_pos] * 3
            + [self.w_att] * 3
            + [self.w_joint] * n_joint_q
            + [self.w_vel] * 3
            + [self.w_omega] * 3
            + [self.w_joint_vel] * n_joint_v
        )
        w = np.asarray(w, dtype=np.float64)
        if w.size != int(self.state.ndx):
            raise ValueError(
                f"state activation weight size mismatch: got {w.size}, expected ndx={self.state.ndx}"
            )
        return w

    def _control_activation_weights(self) -> np.ndarray:
        """Weights on control [T1,T2,T3,T4,tau1,tau2]."""
        w_u = np.ones(self.nu, dtype=np.float64)
        if self.nu >= 4:
            w_u[:4] = float(self.w_u_thrust)
        if self.nu >= 6:
            w_u[4:6] = float(self.w_u_joint_torque)
        return w_u

    def update_cost_weights(self, **kw) -> None:
        """在线更新代价权重（下一拍 build_shooting_problem 即生效，无需重建模型）。"""
        if self.mode == self.MODE_FULL_STATE:
            for name in (
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
            ):
                if name in kw and kw[name] is not None:
                    setattr(self, name, float(kw[name]))
            if "dt_mpc" in kw and kw["dt_mpc"] is not None:
                self.dt_mpc = float(kw["dt_mpc"])
            if "horizon" in kw and kw["horizon"] is not None:
                self.horizon = int(kw["horizon"])
            return
        w = self.w
        for name in (
            "w_pos",
            "w_rot_rp",
            "w_rot_yaw",
            "w_vel_lin",
            "w_vel_ang_rp",
            "w_vel_ang_yaw",
            "w_u",
            "w_terminal_scale",
            "w_base_pos",
            "w_base_yaw",
            "w_state_reg",
            "w_state_track",
            "w_st_pos",
            "w_st_att",
            "w_st_joint",
            "w_st_vel",
            "w_st_omega",
            "w_st_joint_vel",
        ):
            if name in kw and kw[name] is not None:
                setattr(w, name, float(kw[name]))
        if "dt_mpc" in kw and kw["dt_mpc"] is not None:
            self.dt_mpc = float(kw["dt_mpc"])
        if "horizon" in kw and kw["horizon"] is not None:
            self.horizon = int(kw["horizon"])

    def _make_running_cost_state(self, x_ref: np.ndarray, x_nom: np.ndarray) -> crocoddyl.CostModelSum:
        """Full-state tracking 的 running cost（每个 MPC 时域节点都会用）.

        结构:
          1) x_track:  跟踪参考状态 x_ref（主任务项）
          2) x_reg:    向名义状态 x_nom 正则（抑制漂移/病态解）
          3) u_reg:    控制正则，参考悬停推力 self._u_ref

        说明:
          - 三项最终由 CostModelSum 做加权求和，权重分别是
            self.w_state_track / self.w_state_reg / self.w_control。
          - 对关节力矩通道 (u[4], u[5]) 额外放大激活权重（100x），
            用于抑制关节力矩过大导致的抖动。
        """
        nu = self.nu
        c = crocoddyl.CostModelSum(self.state, nu)
        act_x = crocoddyl.ActivationModelWeightedQuad(self._full_state_activation_weights())
        c.addCost(
            "x_track",
            crocoddyl.CostModelResidual(
                self.state,
                act_x,
                crocoddyl.ResidualModelState(self.state, np.asarray(x_ref, dtype=float), nu),
            ),
            self.w_state_track,
        )
        c.addCost(
            "x_reg",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(self._full_state_activation_weights()),
                crocoddyl.ResidualModelState(self.state, np.asarray(x_nom, dtype=float), nu),
            ),
            self.w_state_reg,
        )
        act_u = crocoddyl.ActivationModelWeightedQuad(self._control_activation_weights())
        c.addCost(
            "u_reg",
            crocoddyl.CostModelResidual(
                self.state,
                act_u,
                crocoddyl.ResidualModelControl(self.state, self._u_ref.copy()),
            ),
            self.w_control,
        )
        return c

    def _make_terminal_cost_state(self, x_ref: np.ndarray) -> crocoddyl.CostModelSum:
        """Full-state tracking 的 terminal cost（时域末端节点）.

        终端仅保留 x_track_term（对 x_ref 的状态跟踪），
        常见作用是给 MPC 末端一个“收敛方向”，避免短视控制。
        权重由 self.w_terminal_track 控制。
        """
        nu = self.nu
        c = crocoddyl.CostModelSum(self.state, nu)
        c.addCost(
            "x_track_term",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(self._full_state_activation_weights()),
                crocoddyl.ResidualModelState(self.state, np.asarray(x_ref, dtype=float), nu),
            ),
            self.w_terminal_track,
        )
        return c

    def _make_integrated_running_state(self, x_ref: np.ndarray, x_nom: np.ndarray):
        cost = self._make_running_cost_state(x_ref, x_nom)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost
        )
        inte = crocoddyl.IntegratedActionModelEuler(diff, self.dt_mpc)
        inte.u_lb = self._u_lb.copy()
        inte.u_ub = self._u_ub.copy()
        return inte

    def _make_integrated_terminal_state(self, x_ref: np.ndarray):
        cost = self._make_terminal_cost_state(x_ref)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost
        )
        return crocoddyl.IntegratedActionModelEuler(diff, 0.0)

    # ----- EE-pose costs -----

    def _make_running_cost_ee(
        self,
        p_des: np.ndarray,
        yaw_des: float,
        v_lin_des: np.ndarray,
        w_ang_des: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
        p_base_des: Optional[np.ndarray] = None,
        yaw_base_des: Optional[float] = None,
        *,
        cost_scale: float = 1.0,
    ) -> crocoddyl.CostModelSum:
        """EE pose tracking 的 running cost（用于 croc_ee_pose 模式）.

        可包含的项（按权重开关）:
          - ee_pos: EE 位置跟踪
          - ee_rot: EE 姿态跟踪（roll/pitch/yaw 可分权）
          - ee_vel: EE 线速度/角速度跟踪（通常用于抓取前减速）
          - base_pos / base_yaw: 弱基座参考（沿 plan，不锁关节）
          - u_reg:  控制正则（参考悬停）
          - x_reg:  状态正则（默认更弱）
          - x_track_ref: 可选全状态参考跟踪
        其中 cost_scale 用于统一放大（终端项会用更大的 scale）。
        """
        nu = self.nu
        w = self.w
        c = crocoddyl.CostModelSum(self.state, nu)
        sc = float(cost_scale)
        R_des = _yaw_to_rotation_matrix(yaw_des)
        p_des = np.asarray(p_des, dtype=float).reshape(3)
        T_des = pin.SE3(R_des, p_des)
        if w.w_pos > 0:
            trans_res = crocoddyl.ResidualModelFrameTranslation(
                self.state, self.ee_frame_id, p_des, nu
            )
            c.addCost(
                "ee_pos",
                crocoddyl.CostModelResidual(self.state, trans_res),
                sc * float(w.w_pos),
            )
        if w.w_rot_rp > 0 or w.w_rot_yaw > 0:
            rot_act = crocoddyl.ActivationModelWeightedQuad(
                np.array(
                    [0.0, 0.0, 0.0, float(w.w_rot_rp), float(w.w_rot_rp), float(w.w_rot_yaw)],
                    dtype=float,
                )
            )
            ee_pl_res = crocoddyl.ResidualModelFramePlacement(
                self.state, self.ee_frame_id, T_des, nu
            )
            c.addCost(
                "ee_rot",
                crocoddyl.CostModelResidual(self.state, rot_act, ee_pl_res),
                sc,
            )
        if w.w_vel_lin > 0 or w.w_vel_ang_rp > 0 or w.w_vel_ang_yaw > 0:
            rf = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            vel_act = crocoddyl.ActivationModelWeightedQuad(
                np.array(
                    [
                        w.w_vel_lin,
                        w.w_vel_lin,
                        w.w_vel_lin,
                        w.w_vel_ang_rp,
                        w.w_vel_ang_rp,
                        w.w_vel_ang_yaw,
                    ],
                    dtype=float,
                )
            )
            v_lin_des = np.asarray(v_lin_des, dtype=float).reshape(3)
            w_ang_des = np.asarray(w_ang_des, dtype=float).reshape(3)
            vel_motion_ref = pin.Motion(v_lin_des, w_ang_des)
            vel_res = None
            if hasattr(crocoddyl, "ResidualModelFrameVelocityTpl"):
                try:
                    vel_res = crocoddyl.ResidualModelFrameVelocityTpl(
                        self.state, self.ee_frame_id, vel_motion_ref, rf, nu
                    )
                except Exception:
                    vel_res = None
            if vel_res is None:
                vel_res = crocoddyl.ResidualModelFrameVelocity(
                    self.state, self.ee_frame_id, vel_motion_ref, rf, nu
                )
            c.addCost(
                "ee_vel",
                crocoddyl.CostModelResidual(self.state, vel_act, vel_res),
                sc,
            )
        if (
            w.w_base_pos > 0
            and p_base_des is not None
            and int(getattr(self, "base_frame_id", -1)) >= 0
        ):
            p_b = np.asarray(p_base_des, dtype=float).reshape(3)
            base_pos_res = crocoddyl.ResidualModelFrameTranslation(
                self.state, self.base_frame_id, p_b, nu
            )
            c.addCost(
                "base_pos",
                crocoddyl.CostModelResidual(self.state, base_pos_res),
                sc * float(w.w_base_pos),
            )
        if (
            w.w_base_yaw > 0
            and yaw_base_des is not None
            and int(getattr(self, "base_frame_id", -1)) >= 0
        ):
            R_b = _yaw_to_rotation_matrix(float(yaw_base_des))
            T_b = pin.SE3(R_b, np.zeros(3, dtype=float))
            base_yaw_act = crocoddyl.ActivationModelWeightedQuad(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
            )
            base_rot_res = crocoddyl.ResidualModelFramePlacement(
                self.state, self.base_frame_id, T_b, nu
            )
            c.addCost(
                "base_yaw",
                crocoddyl.CostModelResidual(self.state, base_yaw_act, base_rot_res),
                sc * float(w.w_base_yaw),
            )
        if w.w_u > 0:
            w_u = np.ones(nu, dtype=np.float64)
            if nu >= 6:
                w_u[4] = 100.0
                w_u[5] = 100.0
            u_act = crocoddyl.ActivationModelWeightedQuad(w_u)
            u_res = crocoddyl.ResidualModelControl(self.state, self._u_ref.copy())
            c.addCost(
                "u_reg",
                crocoddyl.CostModelResidual(self.state, u_act, u_res),
                sc * float(w.w_u),
            )
        if w.w_state_reg > 0:
            x_nom = np.zeros(self.nq + self.nv, dtype=float)
            x_nom[2] = 1.0
            x_nom[6] = 1.0
            x_act_weights = np.ones(int(self.state.ndx), dtype=float)
            if self.nv > 0 and self.state.ndx >= self.nv:
                x_act_weights[-self.nv :] = 0.0
            x_act = crocoddyl.ActivationModelWeightedQuad(x_act_weights)
            x_res = crocoddyl.ResidualModelState(self.state, x_nom, nu)
            c.addCost(
                "x_reg",
                crocoddyl.CostModelResidual(self.state, x_act, x_res),
                sc * float(w.w_state_reg),
            )
        if x_ref is not None and float(w.w_state_track) > 0.0:
            xr = np.asarray(x_ref, dtype=float).flatten()
            act_x = crocoddyl.ActivationModelWeightedQuad(
                self._state_track_activation_weights()
            )
            x_track_res = crocoddyl.ResidualModelState(self.state, xr, nu)
            c.addCost(
                "x_track_ref",
                crocoddyl.CostModelResidual(self.state, act_x, x_track_res),
                sc * float(w.w_state_track),
            )
        return c

    def _make_terminal_cost_ee(
        self,
        p_des: np.ndarray,
        yaw_des: float,
        v_lin_des: np.ndarray,
        w_ang_des: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
        p_base_des: Optional[np.ndarray] = None,
        yaw_base_des: Optional[float] = None,
    ) -> crocoddyl.CostModelSum:
        ts = float(self.w.w_terminal_scale)
        return self._make_running_cost_ee(
            p_des,
            yaw_des,
            v_lin_des,
            w_ang_des,
            x_ref=x_ref,
            p_base_des=p_base_des,
            yaw_base_des=yaw_base_des,
            cost_scale=ts,
        )

    def _make_integrated_running_ee(
        self,
        p_des: np.ndarray,
        yaw_des: float,
        v_lin_des: np.ndarray,
        w_ang_des: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
        p_base_des: Optional[np.ndarray] = None,
        yaw_base_des: Optional[float] = None,
    ) -> crocoddyl.IntegratedActionModelEuler:
        cost = self._make_running_cost_ee(
            p_des,
            yaw_des,
            v_lin_des,
            w_ang_des,
            x_ref=x_ref,
            p_base_des=p_base_des,
            yaw_base_des=yaw_base_des,
            cost_scale=1.0,
        )
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, cost)
        inte = crocoddyl.IntegratedActionModelEuler(diff, self.dt_mpc)
        inte.u_lb = self._u_lb.copy()
        inte.u_ub = self._u_ub.copy()
        return inte

    def _make_integrated_terminal_ee(
        self,
        p_des: np.ndarray,
        yaw_des: float,
        v_lin_des: np.ndarray,
        w_ang_des: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
        p_base_des: Optional[np.ndarray] = None,
        yaw_base_des: Optional[float] = None,
    ) -> crocoddyl.IntegratedActionModelEuler:
        cost = self._make_terminal_cost_ee(
            p_des,
            yaw_des,
            v_lin_des,
            w_ang_des,
            x_ref=x_ref,
            p_base_des=p_base_des,
            yaw_base_des=yaw_base_des,
        )
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, cost)
        return crocoddyl.IntegratedActionModelEuler(diff, 0.0)

    def build_shooting_problem(
        self, x0: np.ndarray, x_ref: np.ndarray, x_nom: np.ndarray
    ) -> crocoddyl.ShootingProblem:
        if self.mode != self.MODE_FULL_STATE:
            raise RuntimeError("build_shooting_problem is only valid in full-state mode")
        x0 = np.asarray(x0, dtype=float).flatten()
        running = [
            self._make_integrated_running_state(x_ref, x_nom) for _ in range(self.horizon)
        ]
        term = self._make_integrated_terminal_state(x_ref)
        return crocoddyl.ShootingProblem(x0, running, term)

    def build_shooting_problem_along_plan(
        self,
        x0: np.ndarray,
        x_nom: np.ndarray,
        t_start: float,
        t_plan: np.ndarray,
        x_plan: np.ndarray,
    ) -> crocoddyl.ShootingProblem:
        if self.mode != self.MODE_FULL_STATE:
            raise RuntimeError("build_shooting_problem_along_plan is only valid in full-state mode")
        x0 = np.asarray(x0, dtype=float).flatten()
        running = []
        for k in range(self.horizon):
            tk = t_start + k * self.dt_mpc
            xrk = interp_full_state_piecewise(tk, t_plan, x_plan, self.robot_model)
            running.append(self._make_integrated_running_state(xrk, x_nom))
        tkN = t_start + self.horizon * self.dt_mpc
        xN = interp_full_state_piecewise(tkN, t_plan, x_plan, self.robot_model)
        term = self._make_integrated_terminal_state(xN)
        return crocoddyl.ShootingProblem(x0, running, term)

    def build_shooting_problem_along_ee_ref(
        self,
        x0: np.ndarray,
        t_start: float,
        t_ref: np.ndarray,
        p_ref: np.ndarray,
        yaw_ref: np.ndarray,
        dp_ref: np.ndarray,
        dyaw_ref: np.ndarray,
        t_plan: Optional[np.ndarray] = None,
        x_plan: Optional[np.ndarray] = None,
    ) -> crocoddyl.ShootingProblem:
        if self.mode != self.MODE_EE_POSE:
            raise RuntimeError("build_shooting_problem_along_ee_ref is only valid in EE-pose mode")
        x0 = np.asarray(x0, dtype=float).flatten()
        use_plan_aux = (
            t_plan is not None
            and x_plan is not None
            and (
                float(self.w.w_state_track) > 0.0
                or float(self.w.w_base_pos) > 0.0
                or float(self.w.w_base_yaw) > 0.0
            )
        )
        if use_plan_aux:
            t_plan = np.asarray(t_plan, dtype=float).flatten()
            x_plan = np.asarray(x_plan, dtype=float)

        def _plan_aux_at(tk: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
            if not use_plan_aux:
                return None, None, None
            xk = interp_full_state_piecewise(tk, t_plan, x_plan, self.robot_model)
            x_ref_k = xk if float(self.w.w_state_track) > 0.0 else None
            p_base_k = (
                np.asarray(xk[:3], dtype=float).reshape(3)
                if float(self.w.w_base_pos) > 0.0
                else None
            )
            yaw_base_k = (
                _yaw_from_full_state(xk) if float(self.w.w_base_yaw) > 0.0 else None
            )
            return x_ref_k, p_base_k, yaw_base_k

        running: List[crocoddyl.IntegratedActionModelEuler] = []
        for k in range(self.horizon):
            tk = float(t_start + k * self.dt_mpc)
            p_des_k, yaw_des_k = interp_ref_pose(tk, t_ref, p_ref, yaw_ref)
            v_lin_k = np.array(
                [
                    np.interp(tk, t_ref, dp_ref[:, 0]),
                    np.interp(tk, t_ref, dp_ref[:, 1]),
                    np.interp(tk, t_ref, dp_ref[:, 2]),
                ],
                dtype=float,
            )
            yaw_rate_k = float(np.interp(tk, t_ref, dyaw_ref))
            w_ang_k = np.array([0.0, 0.0, yaw_rate_k], dtype=float)
            x_ref_k, p_base_k, yaw_base_k = _plan_aux_at(tk)
            running.append(
                self._make_integrated_running_ee(
                    p_des_k,
                    yaw_des_k,
                    v_lin_k,
                    w_ang_k,
                    x_ref=x_ref_k,
                    p_base_des=p_base_k,
                    yaw_base_des=yaw_base_k,
                )
            )
        tN = float(t_start + self.horizon * self.dt_mpc)
        p_des_N, yaw_des_N = interp_ref_pose(tN, t_ref, p_ref, yaw_ref)
        v_lin_N = np.array(
            [
                np.interp(tN, t_ref, dp_ref[:, 0]),
                np.interp(tN, t_ref, dp_ref[:, 1]),
                np.interp(tN, t_ref, dp_ref[:, 2]),
            ],
            dtype=float,
        )
        yaw_rate_N = float(np.interp(tN, t_ref, dyaw_ref))
        w_ang_N = np.array([0.0, 0.0, yaw_rate_N], dtype=float)
        x_ref_N, p_base_N, yaw_base_N = _plan_aux_at(tN)
        terminal = self._make_integrated_terminal_ee(
            p_des_N,
            yaw_des_N,
            v_lin_N,
            w_ang_N,
            x_ref=x_ref_N,
            p_base_des=p_base_N,
            yaw_base_des=yaw_base_N,
        )
        return crocoddyl.ShootingProblem(x0, running, terminal)

    def build_shooting_problem_along_ref(
        self,
        x0: np.ndarray,
        t_start: float,
        t_ref: np.ndarray,
        p_ref: np.ndarray,
        yaw_ref: np.ndarray,
        dp_ref: np.ndarray,
        dyaw_ref: np.ndarray,
        t_plan: Optional[np.ndarray] = None,
        x_plan: Optional[np.ndarray] = None,
    ) -> crocoddyl.ShootingProblem:
        """Backward-compatible name for :meth:`build_shooting_problem_along_ee_ref`."""
        return self.build_shooting_problem_along_ee_ref(
            x0,
            t_start,
            t_ref,
            p_ref,
            yaw_ref,
            dp_ref,
            dyaw_ref,
            t_plan=t_plan,
            x_plan=x_plan,
        )

    def make_sim_integrator(
        self,
        sim_dt: float,
        x_ref: Optional[np.ndarray] = None,
        x_nom: Optional[np.ndarray] = None,
    ):
        if self.mode == self.MODE_FULL_STATE:
            if x_ref is None or x_nom is None:
                raise ValueError("full-state make_sim_integrator requires x_ref and x_nom")
            cost = self._make_running_cost_state(x_ref, x_nom)
        else:
            cost = crocoddyl.CostModelSum(self.state, self.nu)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost
        )
        inte = crocoddyl.IntegratedActionModelEuler(diff, float(sim_dt))
        return inte, inte.createData()

    def integrate_one(
        self, sim_int: crocoddyl.IntegratedActionModelEuler, sim_data, x: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        return crocoddyl_euler_step(sim_int, sim_data, x, u)


class UAMCrocoddylStateTrackingMPC(UAMCrocoddylTrackingMPC):
    """Full-state tracking (constant or sampled references); backward-compatible constructor."""

    def __init__(
        self,
        s500_yaml_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        dt_mpc: float = 0.05,
        horizon: int = 25,
        w_state_track: float = 10.0,
        w_state_reg: float = 0.1,
        w_control: float = 1e-3,
        w_terminal_track: float = 100.0,
        w_pos: float = 1.0,
        w_att: float = 1.0,
        w_joint: float = 1.0,
        w_vel: float = 1.0,
        w_omega: float = 1.0,
        w_joint_vel: float = 1.0,
        w_u_thrust: float = 1.0,
        w_u_joint_torque: float = 1.0,
        use_thrust_constraints: bool = True,
    ):
        super().__init__(
            mode=UAMCrocoddylTrackingMPC.MODE_FULL_STATE,
            s500_yaml_path=s500_yaml_path,
            urdf_path=urdf_path,
            dt_mpc=dt_mpc,
            horizon=horizon,
            use_thrust_constraints=use_thrust_constraints,
            w_state_track=w_state_track,
            w_state_reg=w_state_reg,
            w_control=w_control,
            w_terminal_track=w_terminal_track,
            w_pos=w_pos,
            w_att=w_att,
            w_joint=w_joint,
            w_vel=w_vel,
            w_omega=w_omega,
            w_joint_vel=w_joint_vel,
            w_u_thrust=w_u_thrust,
            w_u_joint_torque=w_u_joint_torque,
        )


class UAMEEPoseTrackingCrocoddylMPC(UAMCrocoddylTrackingMPC):
    """EE pose + velocity tracking; backward-compatible constructor."""

    def __init__(
        self,
        *,
        s500_yaml_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        dt_mpc: float = 0.05,
        horizon: int = 25,
        u_weights: EETrackingWeights = EETrackingWeights(),
        use_thrust_constraints: bool = True,
    ):
        super().__init__(
            mode=UAMCrocoddylTrackingMPC.MODE_EE_POSE,
            s500_yaml_path=s500_yaml_path,
            urdf_path=urdf_path,
            dt_mpc=dt_mpc,
            horizon=horizon,
            use_thrust_constraints=use_thrust_constraints,
            ee_weights=u_weights,
        )


def _full_state_closed_loop_plant(
    mpc: UAMCrocoddylStateTrackingMPC,
    sim_dt: float,
    x_ref_for_placeholder: np.ndarray,
    x_nom: np.ndarray,
    *,
    sim_payload_enable: bool = False,
    sim_payload_t_grasp: float = 1.0,
    sim_payload_mass: float = 0.2,
    sim_payload_sphere_radius: float = 0.02,
):
    """
    Plant integrator for full-state tracking: nominal dynamics, or a Pinocchio copy with optional
    sphere payload applied once at ``t_grasp`` (simulation only; MPC model unchanged).
    """
    t_grasp = max(0.0, float(sim_payload_t_grasp))
    m_pay = float(sim_payload_mass)
    r_sph = max(1e-6, float(sim_payload_sphere_radius))
    use_sim_plant_payload = bool(sim_payload_enable) and m_pay > 1e-9

    if not use_sim_plant_payload:
        sim_int, sim_data = mpc.make_sim_integrator(sim_dt, x_ref_for_placeholder, x_nom)
        return CrocoddylEulerPlant(sim_int, sim_data)

    ixx_p, iyy_p, izz_p = solid_sphere_principal_inertias(m_pay, r_sph)
    com_pl = np.zeros(3, dtype=float)
    sim_model = pin.Model(mpc.robot_model)
    sim_state, sim_actuation = mpc._planner.thruster_actuation_for_model(sim_model)
    cost0 = crocoddyl.CostModelSum(sim_state, mpc.nu)
    diff0 = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        sim_state, sim_actuation, cost0
    )
    sim_int = crocoddyl.IntegratedActionModelEuler(diff0, float(sim_dt))
    sim_data = sim_int.createData()
    base_plant = CrocoddylEulerPlant(sim_int, sim_data)

    def _apply_payload_once() -> None:
        _apply_payload_inertia_on_plant_model(
            sim_model,
            mpc.ee_frame_id,
            m_pay,
            com_pl,
            ixx_p,
            iyy_p,
            izz_p,
        )

    return PayloadSchedulePlant(base_plant, t_grasp, _apply_payload_once)


def run_closed_loop_state_tracking(
    x0: np.ndarray,
    x_ref: np.ndarray,
    x_nom: np.ndarray,
    T_sim: float,
    sim_dt: float,
    control_dt: float,
    dt_mpc: float,
    horizon: int,
    w_state_track: float = 10.0,
    w_state_reg: float = 0.1,
    w_control: float = 1e-3,
    w_terminal_track: float = 100.0,
    w_pos: float = 1.0,
    w_att: float = 1.0,
    w_joint: float = 1.0,
    w_vel: float = 1.0,
    w_omega: float = 1.0,
    w_joint_vel: float = 1.0,
    w_u_thrust: float = 1.0,
    w_u_joint_torque: float = 1.0,
    mpc_max_iter: int = 60,
    use_thrust_constraints: bool = True,
    use_actuator_first_order: bool = False,
    tau_thrust: float = 0.06,
    tau_theta: float = 0.05,
    sim_payload_enable: bool = False,
    sim_payload_t_grasp: float = 1.0,
    sim_payload_mass: float = 0.2,
    sim_payload_sphere_radius: float = 0.02,
    sim_control_stack: str = "direct",
    px4_rate_Kp: float = 12.0,
    px4_rate_Kd: float = 1.5,
    s500_yaml_path: Optional[str] = None,
    urdf_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Closed-loop simulation: solve the MPC once every control_dt with ZOH in between; propagate the state using sim_dt
    forward integration. MPC dynamics use commanded ``u``; if ``use_actuator_first_order``, the plant integrates with
    lagged ``u_act`` (see module docstring).

    If ``sim_control_stack == "px4_rate"``, the plant command is built from MPC ``u`` via a body-rate setpoint integrator,
    a rate loop, and a 4-rotor mixer (arm torques still follow ``u_mpc[4:6]``).
    """
    sim_control_stack = str(sim_control_stack).strip().lower()
    if sim_control_stack not in ("direct", "px4_rate"):
        raise ValueError(f"unknown sim_control_stack: {sim_control_stack!r}")

    mpc = UAMCrocoddylStateTrackingMPC(
        s500_yaml_path=s500_yaml_path,
        urdf_path=urdf_path,
        dt_mpc=dt_mpc,
        horizon=horizon,
        w_state_track=w_state_track,
        w_state_reg=w_state_reg,
        w_control=w_control,
        w_terminal_track=w_terminal_track,
        w_pos=w_pos,
        w_att=w_att,
        w_joint=w_joint,
        w_vel=w_vel,
        w_omega=w_omega,
        w_joint_vel=w_joint_vel,
        w_u_thrust=w_u_thrust,
        w_u_joint_torque=w_u_joint_torque,
        use_thrust_constraints=use_thrust_constraints,
    )

    sim_dt = float(sim_dt)
    control_dt = float(control_dt)
    if sim_dt <= 0 or control_dt <= 0:
        raise ValueError("sim_dt and control_dt must be positive")
    n_inner = mpc_inner_stride(control_dt, sim_dt)

    plant = _full_state_closed_loop_plant(
        mpc,
        sim_dt,
        x_ref,
        x_nom,
        sim_payload_enable=sim_payload_enable,
        sim_payload_t_grasp=sim_payload_t_grasp,
        sim_payload_mass=sim_payload_mass,
        sim_payload_sphere_radius=sim_payload_sphere_radius,
    )

    n_total = max(1, int(np.ceil(T_sim / sim_dt)))
    t_arr = np.arange(n_total, dtype=float) * sim_dt

    x = np.asarray(x0, dtype=float).copy().flatten()
    time_data: List[float] = []
    state_data: List[np.ndarray] = []
    ctrl_data: List[np.ndarray] = []
    mpc_costs: List[float] = []
    mpc_cost_terms_hist: Dict[str, List[float]] = {}
    mpc_cost_groups_hist: Dict[str, List[float]] = {}
    mpc_cost_weights: Dict[str, float] = {}
    mpc_iters: List[int] = []
    mpc_solve_t: List[float] = []
    mpc_solve_steps: List[int] = []
    mpc_wall_s: List[float] = []
    track_norm: List[float] = []

    xs_guess: Optional[List[np.ndarray]] = None
    us_guess: Optional[List[np.ndarray]] = None

    u_cmd_hold = mpc._u_ref.copy()
    u_act = u_cmd_hold.copy()
    nq = mpc.robot_model.nq
    act_data = mpc.actuation.createData()
    omega_sp = np.asarray(x0, dtype=float).reshape(-1)[nq + 3 : nq + 6].copy()

    for step in range(n_total):
        t = step * sim_dt
        time_data.append(t)
        state_data.append(x.copy())
        plant.on_pre_step(t, step)
        # ctrl_data records u_act applied to the plant (after optional px4_rate stack and optional 1st-order lag).

        dq = pin.difference(mpc.robot_model, x_ref[:nq], x[:nq])
        dv = x[nq:] - x_ref[nq:]
        track_norm.append(float(np.linalg.norm(np.concatenate([dq, dv]))))

        if step % n_inner == 0:
            prob = mpc.build_shooting_problem(x, x_ref, x_nom)
            solver = crocoddyl.SolverBoxFDDP(prob)
            solver.convergence_init = 1e-9
            solver.convergence_stop = 1e-7
            try:
                solver.setCallbacks([])
            except Exception:
                pass

            if xs_guess is None:
                xs_init = [x.copy() for _ in range(horizon + 1)]
                us_init = [mpc._u_ref.copy() for _ in range(horizon)]
            else:
                xs_init = xs_guess
                us_init = us_guess

            t_solve0 = time.perf_counter()
            converged = solver.solve(xs_init, us_init, mpc_max_iter)
            wall_s = time.perf_counter() - t_solve0
            if verbose and step % (5 * n_inner) == 0:
                print(
                    f"t={t:.3f} converged={converged} cost={solver.cost:.4f} iters={solver.iter}"
                )

            u_cmd_hold = np.array(solver.us[0], dtype=float).copy()
            solve_idx = len(mpc_costs)
            mpc_costs.append(float(solver.cost))
            terms, groups, coeffs = _extract_solver_cost_terms(solver)
            if coeffs:
                mpc_cost_weights.update({k: float(v) for k, v in coeffs.items()})
            all_keys = set(mpc_cost_terms_hist.keys()) | set(terms.keys())
            for k in all_keys:
                if k not in mpc_cost_terms_hist:
                    mpc_cost_terms_hist[k] = [float("nan")] * solve_idx
                mpc_cost_terms_hist[k].append(float(terms.get(k, float("nan"))))
            grp_keys = set(mpc_cost_groups_hist.keys()) | set(groups.keys())
            for k in grp_keys:
                if k not in mpc_cost_groups_hist:
                    mpc_cost_groups_hist[k] = [float("nan")] * solve_idx
                mpc_cost_groups_hist[k].append(float(groups.get(k, float("nan"))))
            mpc_iters.append(int(solver.iter))
            mpc_solve_t.append(t)
            mpc_solve_steps.append(step)
            mpc_wall_s.append(float(wall_s))

            xs_guess = [solver.xs[i + 1].copy() for i in range(horizon)] + [
                solver.xs[-1].copy()
            ]
            xs_guess[0] = x.copy()
            us_guess = [solver.us[i + 1].copy() for i in range(horizon - 1)] + [
                solver.us[-1].copy()
            ]

        if sim_control_stack == "px4_rate":
            u_plant, omega_sp = px4_rate_compute_plant_u(
                mpc,
                act_data,
                x,
                u_cmd_hold,
                omega_sp,
                sim_dt=sim_dt,
                rate_Kp=px4_rate_Kp,
                rate_Kd=px4_rate_Kd,
            )
            u_after_stack = u_plant
        else:
            u_after_stack = u_cmd_hold

        # Update the first-order actuator response (ZOH: u_cmd_hold stays constant within n_inner)
        if use_actuator_first_order:
            u_act = _apply_first_order_actuator(
                u_act,
                u_after_stack,
                tau_thrust=tau_thrust,
                tau_theta=tau_theta,
                dt=sim_dt,
            )
        else:
            u_act = u_after_stack.copy()

        ctrl_data.append(u_act.copy())

        if step < n_total - 1:
            x = plant.step(x, u_act)

    return {
        "time": np.array(time_data),
        "states": np.array(state_data),
        "controls": np.array(ctrl_data),
        "sim_control_stack": sim_control_stack,
        "px4_rate_Kp": float(px4_rate_Kp),
        "px4_rate_Kd": float(px4_rate_Kd),
        "x_ref": np.asarray(x_ref, dtype=float).copy(),
        "x_nom": np.asarray(x_nom, dtype=float).copy(),
        "track_norm": np.array(track_norm),
        "mpc_costs": np.array(mpc_costs),
        "mpc_cost_terms": {k: np.asarray(v, dtype=float) for k, v in mpc_cost_terms_hist.items()},
        "mpc_cost_groups": {k: np.asarray(v, dtype=float) for k, v in mpc_cost_groups_hist.items()},
        "mpc_cost_weights": {k: float(v) for k, v in mpc_cost_weights.items()},
        "mpc_iters": np.array(mpc_iters, dtype=int),
        "mpc_solve_t": np.array(mpc_solve_t),
        "mpc_solve_steps": np.array(mpc_solve_steps, dtype=int),
        "mpc_wall_s": np.array(mpc_wall_s, dtype=float),
        "sim_dt": sim_dt,
        "control_dt": control_dt,
        "n_inner": n_inner,
        "mpc": mpc,
        "sim_plant_payload_applied": bool(
            isinstance(plant, PayloadSchedulePlant) and plant.schedule_applied
        ),
    }


def run_closed_loop_track_full_state_plan(
    x0: np.ndarray,
    t_plan: np.ndarray,
    x_plan: np.ndarray,
    x_nom: np.ndarray,
    T_sim: float,
    sim_dt: float,
    control_dt: float,
    dt_mpc: float,
    horizon: int,
    w_state_track: float = 10.0,
    w_state_reg: float = 0.1,
    w_control: float = 1e-3,
    w_terminal_track: float = 100.0,
    w_pos: float = 1.0,
    w_att: float = 1.0,
    w_joint: float = 1.0,
    w_vel: float = 1.0,
    w_omega: float = 1.0,
    w_joint_vel: float = 1.0,
    w_u_thrust: float = 1.0,
    w_u_joint_torque: float = 1.0,
    mpc_max_iter: int = 60,
    use_thrust_constraints: bool = True,
    use_actuator_first_order: bool = False,
    tau_thrust: float = 0.06,
    tau_theta: float = 0.05,
    sim_payload_enable: bool = False,
    sim_payload_t_grasp: float = 1.0,
    sim_payload_mass: float = 0.2,
    sim_payload_sphere_radius: float = 0.02,
    sim_control_stack: str = "direct",
    px4_rate_Kp: float = 12.0,
    px4_rate_Kd: float = 1.5,
    ctbr: Optional[Dict[str, Any]] = None,
    s500_yaml_path: Optional[str] = None,
    urdf_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Track a planned full-state trajectory (t_plan, x_plan) with Crocoddyl receding-horizon tracking; the reference within the horizon is sampled with a time shift.

    ``sim_control_stack="ctbr"``：取 MPC horizon 在 lookahead 处的规划角速度做设定点，
    经角速度 PID（rp/yaw 分组）+ 四旋翼混控 + 电机一阶滞后施加到 plant（参数见 ``ctbr``）。
    """
    sim_control_stack = str(sim_control_stack).strip().lower()
    if sim_control_stack not in ("direct", "px4_rate", "ctbr"):
        raise ValueError(f"unknown sim_control_stack: {sim_control_stack!r}")

    # CTBR 内环配置（与 acados/几何 共用同一组参数语义）。
    ctbr_cfg = dict(ctbr) if ctbr else {}
    ctbr_lookahead_s = float(ctbr_cfg.get("lookahead_ms", dt_mpc * 1000.0)) / 1000.0
    ctbr_kp = np.array([
        float(ctbr_cfg.get("kp_rp", 12.0)), float(ctbr_cfg.get("kp_rp", 12.0)),
        float(ctbr_cfg.get("kp_yaw", 4.0)),
    ], dtype=float)
    ctbr_ki = np.array([
        float(ctbr_cfg.get("ki_rp", 0.0)), float(ctbr_cfg.get("ki_rp", 0.0)),
        float(ctbr_cfg.get("ki_yaw", 0.0)),
    ], dtype=float)
    ctbr_kd = np.array([
        float(ctbr_cfg.get("kd_rp", 0.0)), float(ctbr_cfg.get("kd_rp", 0.0)),
        float(ctbr_cfg.get("kd_yaw", 0.0)),
    ], dtype=float)
    ctbr_max_rate = float(ctbr_cfg.get("max_rate", 8.0))
    ctbr_max_torque = float(ctbr_cfg.get("max_torque", 0.0))
    ctbr_motor_tau = float(ctbr_cfg.get("motor_tau", 0.02))
    ctbr_int_limit = float(ctbr_cfg.get("int_limit", 2.0))

    t_plan = np.asarray(t_plan, dtype=float).flatten()
    x_plan = np.asarray(x_plan, dtype=float)
    if x_plan.ndim != 2 or len(t_plan) != len(x_plan):
        raise ValueError("x_plan must be 2D with same length as t_plan")

    mpc = UAMCrocoddylStateTrackingMPC(
        s500_yaml_path=s500_yaml_path,
        urdf_path=urdf_path,
        dt_mpc=dt_mpc,
        horizon=horizon,
        w_state_track=w_state_track,
        w_state_reg=w_state_reg,
        w_control=w_control,
        w_terminal_track=w_terminal_track,
        w_pos=w_pos,
        w_att=w_att,
        w_joint=w_joint,
        w_vel=w_vel,
        w_omega=w_omega,
        w_joint_vel=w_joint_vel,
        w_u_thrust=w_u_thrust,
        w_u_joint_torque=w_u_joint_torque,
        use_thrust_constraints=use_thrust_constraints,
    )

    sim_dt = float(sim_dt)
    control_dt = float(control_dt)
    n_inner = mpc_inner_stride(control_dt, sim_dt)

    t_mid = 0.5 * (float(t_plan[0]) + float(t_plan[-1]))
    x_mid = interp_full_state_piecewise(t_mid, t_plan, x_plan, mpc.robot_model)
    plant = _full_state_closed_loop_plant(
        mpc,
        sim_dt,
        x_mid,
        x_nom,
        sim_payload_enable=sim_payload_enable,
        sim_payload_t_grasp=sim_payload_t_grasp,
        sim_payload_mass=sim_payload_mass,
        sim_payload_sphere_radius=sim_payload_sphere_radius,
    )

    n_total = max(1, int(np.ceil(T_sim / sim_dt)))
    x = np.asarray(x0, dtype=float).copy().flatten()
    time_data: List[float] = []
    state_data: List[np.ndarray] = []
    ctrl_data: List[np.ndarray] = []
    mpc_costs: List[float] = []
    mpc_cost_terms_hist: Dict[str, List[float]] = {}
    mpc_cost_groups_hist: Dict[str, List[float]] = {}
    mpc_cost_weights: Dict[str, float] = {}
    mpc_iters: List[int] = []
    mpc_solve_t: List[float] = []
    mpc_solve_steps: List[int] = []
    mpc_wall_s: List[float] = []
    track_norm: List[float] = []

    xs_guess: Optional[List[np.ndarray]] = None
    us_guess: Optional[List[np.ndarray]] = None
    u_cmd_hold = mpc._u_ref.copy()
    u_act = u_cmd_hold.copy()
    nq = mpc.robot_model.nq
    act_data = mpc.actuation.createData()
    omega_sp = np.asarray(x0, dtype=float).reshape(-1)[nq + 3 : nq + 6].copy()
    # CTBR 内环状态。
    xs_horizon: Optional[List[np.ndarray]] = None
    ctbr_int = np.zeros(3, dtype=float)
    ctbr_prev_err: Optional[np.ndarray] = None
    u_motor = u_cmd_hold.copy()
    ctbr_rate_sp_data: List[np.ndarray] = []

    for step in range(n_total):
        t = step * sim_dt
        time_data.append(t)
        state_data.append(x.copy())
        plant.on_pre_step(t, step)
        # ctrl_data records u_act applied to the plant (after optional px4_rate stack and optional 1st-order lag).

        xr = interp_full_state_piecewise(t, t_plan, x_plan, mpc.robot_model)
        dq = pin.difference(mpc.robot_model, xr[:nq], x[:nq])
        dv = x[nq:] - xr[nq:]
        track_norm.append(float(np.linalg.norm(np.concatenate([dq, dv]))))

        if step % n_inner == 0:
            prob = mpc.build_shooting_problem_along_plan(
                x, x_nom, t, t_plan, x_plan
            )
            solver = crocoddyl.SolverBoxFDDP(prob)
            solver.convergence_init = 1e-9
            solver.convergence_stop = 1e-7
            try:
                solver.setCallbacks([])
            except Exception:
                pass

            if xs_guess is None:
                xs_init = [x.copy() for _ in range(horizon + 1)]
                us_init = [mpc._u_ref.copy() for _ in range(horizon)]
            else:
                xs_init = xs_guess
                us_init = us_guess

            t_solve0 = time.perf_counter()
            converged = solver.solve(xs_init, us_init, mpc_max_iter)
            wall_s = time.perf_counter() - t_solve0
            if verbose and step % (5 * n_inner) == 0:
                print(
                    f"t={t:.3f} converged={converged} cost={solver.cost:.4f} iters={solver.iter}"
                )

            u_cmd_hold = np.array(solver.us[0], dtype=float).copy()
            solve_idx = len(mpc_costs)
            mpc_costs.append(float(solver.cost))
            terms, groups, coeffs = _extract_solver_cost_terms(solver)
            if coeffs:
                mpc_cost_weights.update({k: float(v) for k, v in coeffs.items()})
            all_keys = set(mpc_cost_terms_hist.keys()) | set(terms.keys())
            for k in all_keys:
                if k not in mpc_cost_terms_hist:
                    mpc_cost_terms_hist[k] = [float("nan")] * solve_idx
                mpc_cost_terms_hist[k].append(float(terms.get(k, float("nan"))))
            grp_keys = set(mpc_cost_groups_hist.keys()) | set(groups.keys())
            for k in grp_keys:
                if k not in mpc_cost_groups_hist:
                    mpc_cost_groups_hist[k] = [float("nan")] * solve_idx
                mpc_cost_groups_hist[k].append(float(groups.get(k, float("nan"))))
            mpc_iters.append(int(solver.iter))
            mpc_solve_t.append(t)
            mpc_solve_steps.append(step)
            mpc_wall_s.append(float(wall_s))

            xs_guess = [solver.xs[i + 1].copy() for i in range(horizon)] + [
                solver.xs[-1].copy()
            ]
            xs_guess[0] = x.copy()
            us_guess = [solver.us[i + 1].copy() for i in range(horizon - 1)] + [
                solver.us[-1].copy()
            ]
            xs_horizon = [np.asarray(s, dtype=float).copy() for s in solver.xs]

        ctbr_used = False
        if sim_control_stack == "px4_rate":
            u_plant, omega_sp = px4_rate_compute_plant_u(
                mpc,
                act_data,
                x,
                u_cmd_hold,
                omega_sp,
                sim_dt=sim_dt,
                rate_Kp=px4_rate_Kp,
                rate_Kd=px4_rate_Kd,
            )
            u_after_stack = u_plant
        elif sim_control_stack == "ctbr":
            # 规划角速度设定点（horizon 在 lookahead 处插值）。
            omega_sp_plan = bodyrate_from_horizon(
                xs_horizon, ctbr_lookahead_s, dt_mpc, nq
            )
            if omega_sp_plan is None:
                omega_sp_plan = x[nq + 3 : nq + 6].copy()
            if ctbr_max_rate > 0.0:
                omega_sp_plan = np.clip(omega_sp_plan, -ctbr_max_rate, ctbr_max_rate)
            ctbr_rate_sp_data.append(np.asarray(omega_sp_plan, dtype=float).copy())
            u_plant, ctbr_int, ctbr_prev_err, _M_des = ctbr_rate_pid_mix(
                mpc, act_data, x, u_cmd_hold, omega_sp_plan,
                kp=ctbr_kp, ki=ctbr_ki, kd=ctbr_kd,
                integ=ctbr_int, prev_err=ctbr_prev_err, dt=sim_dt,
                int_limit=ctbr_int_limit, max_torque=ctbr_max_torque,
            )
            # 电机一阶滞后（仅旋翼；机械臂关节直通）。
            alpha_m = (
                1.0 - np.exp(-sim_dt / max(ctbr_motor_tau, 1e-9))
                if ctbr_motor_tau > 1e-9 else 1.0
            )
            u_motor[:4] = u_motor[:4] + alpha_m * (u_plant[:4] - u_motor[:4])
            if u_motor.size > 4:
                u_motor[4:] = u_plant[4:]
            u_act = u_motor.copy()
            ctbr_used = True
        else:
            u_after_stack = u_cmd_hold

        # Update the first-order actuator response (ZOH: u_cmd_hold stays constant within n_inner)
        if ctbr_used:
            pass  # CTBR 已含电机一阶，直接用 u_act
        elif use_actuator_first_order:
            u_act = _apply_first_order_actuator(
                u_act,
                u_after_stack,
                tau_thrust=tau_thrust,
                tau_theta=tau_theta,
                dt=sim_dt,
            )
        else:
            u_act = u_after_stack.copy()

        ctrl_data.append(u_act.copy())

        if step < n_total - 1:
            x = plant.step(x, u_act)

    return {
        "time": np.array(time_data),
        "states": np.array(state_data),
        "controls": np.array(ctrl_data),
        "sim_control_stack": sim_control_stack,
        "px4_rate_Kp": float(px4_rate_Kp),
        "px4_rate_Kd": float(px4_rate_Kd),
        "ctbr_active": sim_control_stack == "ctbr",
        "ctbr_rate_sp": (
            np.array(ctbr_rate_sp_data) if ctbr_rate_sp_data else None
        ),
        "t_plan": t_plan.copy(),
        "x_plan": x_plan.copy(),
        "x_nom": np.asarray(x_nom, dtype=float).copy(),
        "track_mode": "full_state_trajectory",
        "track_norm": np.array(track_norm),
        "mpc_costs": np.array(mpc_costs),
        "mpc_cost_terms": {k: np.asarray(v, dtype=float) for k, v in mpc_cost_terms_hist.items()},
        "mpc_cost_groups": {k: np.asarray(v, dtype=float) for k, v in mpc_cost_groups_hist.items()},
        "mpc_cost_weights": {k: float(v) for k, v in mpc_cost_weights.items()},
        "mpc_iters": np.array(mpc_iters, dtype=int),
        "mpc_solve_t": np.array(mpc_solve_t),
        "mpc_solve_steps": np.array(mpc_solve_steps, dtype=int),
        "mpc_wall_s": np.array(mpc_wall_s, dtype=float),
        "sim_dt": sim_dt,
        "control_dt": control_dt,
        "n_inner": n_inner,
        "mpc": mpc,
        "sim_plant_payload_applied": bool(
            isinstance(plant, PayloadSchedulePlant) and plant.schedule_applied
        ),
    }


def crocoddyl_closed_loop_to_ee_tracking_res(out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the output of ``run_closed_loop_state_tracking`` or ``run_closed_loop_track_full_state_plan`` into
    the required ``res`` structure for ``s500_uam_ee_snap_tracking_mpc.render_ee_tracking_results_to_figures``.

    If ``out["track_mode"] == "full_state_trajectory"``, then ``p_ref`` / ``yaw_ref`` are interpolated along ``t_plan``/``x_plan``
    to match the planned reference.
    """
    mpc = out["mpc"]
    t = np.asarray(out["time"], dtype=float).flatten()
    X = np.asarray(out["states"], dtype=float)
    U = np.asarray(out["controls"], dtype=float)
    n_t = len(t)
    n_inner = int(out["n_inner"])

    data = mpc.robot_model.createData()
    fid = mpc._planner.ee_frame_id
    nframes = len(mpc.robot_model.frames)
    # s500 (no gripper frame) has nq<=7: there is no end-effector, so EE channels
    # fall back to the floating-base origin pose (position + yaw) instead of FK.
    use_base_pose = (
        fid is None
        or int(mpc.robot_model.nq) <= 7
        or not (0 <= int(fid) < nframes)
    )

    def _pos_rpy(states: np.ndarray):
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if use_base_pose:
            pos = states[:, 0:3].copy()
            rpy = np.zeros((states.shape[0], 3), dtype=float)
            for i in range(states.shape[0]):
                qx, qy, qz, qw = states[i, 3], states[i, 4], states[i, 5], states[i, 6]
                siny = 2.0 * (qw * qz + qx * qy)
                cosy = 1.0 - 2.0 * (qy**2 + qz**2)
                rpy[i, 2] = np.arctan2(siny, cosy)
            return pos, rpy
        p, _, rpy, _ = compute_ee_kinematics_along_trajectory(
            states, mpc.robot_model, data, fid
        )
        return p, rpy

    ee_pos, ee_rpy = _pos_rpy(X)

    if out.get("track_mode") == "full_state_trajectory":
        t_plan = np.asarray(out["t_plan"], dtype=float).flatten()
        x_plan = np.asarray(out["x_plan"], dtype=float)
        Xr = np.array(
            [
                interp_full_state_piecewise(float(ti), t_plan, x_plan, mpc.robot_model)
                for ti in t
            ]
        )
        p_ref, prpy_ref = _pos_rpy(Xr)
        yaw_ref = prpy_ref[:, 2].astype(float)
    else:
        x_ref = np.asarray(out["x_ref"], dtype=float).flatten()
        xr = x_ref.reshape(1, -1)
        pref0, prpy0 = _pos_rpy(xr)
        p_ref = np.repeat(pref0, n_t, axis=0)
        yaw_ref_val = float(prpy0[0, 2])
        yaw_ref = np.full(n_t, yaw_ref_val, dtype=float)

    err = np.linalg.norm(ee_pos - p_ref, axis=1)

    ee_yaw = ee_rpy[:, 2].astype(float)
    err_yaw = ee_yaw - yaw_ref
    err_yaw = (err_yaw + np.pi) % (2.0 * np.pi) - np.pi

    n_mpc = max(0, n_t - 1)
    nit = np.zeros(n_mpc, dtype=int)
    wall = np.zeros(n_mpc, dtype=float)
    stat = np.zeros(n_mpc, dtype=int)
    steps = out.get("mpc_solve_steps")
    iters = out.get("mpc_iters")
    walls = out.get("mpc_wall_s")
    costs = out.get("mpc_costs")
    if steps is not None and iters is not None and len(steps) == len(iters):
        walls_arr = walls if walls is not None and len(walls) == len(iters) else None
        costs_arr = costs if costs is not None and len(costs) == len(iters) else None
        total_cost = np.full(n_mpc, np.nan, dtype=float)
        for j in range(len(steps)):
            st = int(steps[j])
            if n_mpc <= 0:
                break
            si = min(max(st, 0), n_mpc - 1)
            nit[si] = int(iters[j])
            if walls_arr is not None:
                wall[si] = float(walls_arr[j])
            if costs_arr is not None:
                total_cost[si] = float(costs_arr[j])
            stat[si] = 0
    else:
        total_cost = np.full(n_mpc, np.nan, dtype=float)

    mpc_cost_terms = out.get("mpc_cost_terms")
    if not isinstance(mpc_cost_terms, dict):
        mpc_cost_terms = {}
    mpc_cost_groups = out.get("mpc_cost_groups")
    if not isinstance(mpc_cost_groups, dict):
        mpc_cost_groups = {}
    mpc_cost_weights = out.get("mpc_cost_weights")
    if not isinstance(mpc_cost_weights, dict):
        mpc_cost_weights = {}

    return {
        "t": t,
        "x": X,
        "u": U,
        "ee": ee_pos,
        "p_ref": p_ref,
        "err": err,
        "ee_yaw": ee_yaw,
        "yaw_ref": yaw_ref,
        "err_yaw": err_yaw,
        "control_mode": "direct",
        "sim_dt": float(out["sim_dt"]),
        "control_dt": float(out["control_dt"]),
        "mpc_stride": n_inner,
        "mpc_solve": {
            "nlp_iter": nit,
            "cpu_s": wall.copy(),
            "wall_s": wall,
            "status": stat,
            "total_cost": total_cost,
        },
        "mpc_cost_t": np.asarray(out.get("mpc_solve_t", []), dtype=float),
        "mpc_cost_total": np.asarray(out.get("mpc_costs", []), dtype=float),
        "mpc_cost_terms": {k: np.asarray(v, dtype=float) for k, v in mpc_cost_terms.items()},
        "mpc_cost_groups": {k: np.asarray(v, dtype=float) for k, v in mpc_cost_groups.items()},
        "mpc_cost_weights": {k: float(v) for k, v in mpc_cost_weights.items()},
    }


def default_hover_nominal() -> np.ndarray:
    """Nominal state used for weak regularization: body origin height 1 m, level attitude, joints 0, velocity 0."""
    x = make_uam_state(0.0, 0.0, 1.0, j1=0.0, j2=0.0, yaw=0.0)
    return x
