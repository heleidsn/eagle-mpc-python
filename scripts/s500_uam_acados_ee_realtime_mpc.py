#!/usr/bin/env python3
"""Real-time Acados EE-centric NMPC for ROS tracking.

Cost (NONLINEAR_LS, diagonal W) — two layers
---------------------------------------------
**EE task** (always, terminal × ``w_terminal_scale``):
  ``[p_ee(3), yaw_ee, roll_ee, pitch_ee]``  weights ``w_ee_pos``, ``w_ee_yaw``, ``w_ee_rot_rp`` (×2)

**State reference** (optional, same layout as ``acados_full_state``; ``w_state_track=0`` → off):
  ``[pos(3), yaw, roll, pitch, q_joint, v_lin(3), ω(3), q̇_joint]`` along ``x_plan``
  weights ``w_state_track × {w_pos, w_att, w_joint, w_vel, w_omega, w_joint_vel}``;
  terminal state block × ``w_terminal_track``.

**Control reg** (running only):
  ``[u]``  ``w_control × w_u_thrust / w_u_joint_torque``

Counterpart of offline ``s500_uam_ee_snap_tracking_mpc`` EE task + full-state aux;
integrated with ``run_tracking_controller`` mode ``acados_ee_pose``.
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from acados_runtime import preload_acados_shared_libs

    preload_acados_shared_libs()
    from acados_template import AcadosOcp, AcadosOcpSolver

    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

try:
    import casadi as ca
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    PINOCCHIO_AVAILABLE = False
    _pin_err = e

try:
    from pinocchio import casadi as cpin

    from s500_uam_acados_model import load_s500_config
    from s500_uam_acados_state_tracking_mpc import (
        _ACADOS_CODEGEN_NLP_MAX_ITER,
        _ACADOS_CODEGEN_QP_ITER_MAX,
        _align_ocp_with_cached_json,
        _apply_runtime_solver_options,
        ACADOS_OK_STATUSES,
        MpcRefErrorLimits,
        acados_mpc_extract_solution,
        acados_mpc_run_solve,
        acados_solution_finite,
        clamp_mpc_ee_pose_reference,
        clamp_mpc_state_reference,
        sanitize_mpc_state,
    )
    from s500_uam_acados_trajectory import (
        STATE_LIMITS,
        _acados_ocp_generate_build_flags,
    )
    from s500_uam_ee_snap_tracking_mpc import (
        EE_FRAME_NAME,
        _casadi_ee_rpy_zyx_expr,
        _casadi_ee_translation_expr,
    )

    DEPS_OK = True
except ImportError as e:  # pragma: no cover
    DEPS_OK = False
    _deps_err = e

REPO_ROOT = Path(__file__).resolve().parent.parent

# Joint torque channel scale (matches full-state acados tracking).
_JOINT_TORQUE_W_SCALE = 1.0e4


@dataclass(frozen=True)
class _CostSegment:
    """One block in the NONLINEAR_LS output vector ``y``."""

    name: str
    dim: int
    weights: Tuple[float, ...]


def _casadi_state_y_expr(x_sym: "ca.SX", nq: int, n_arm: int) -> "ca.SX":
    """Full-state LS output (no controls); matches ``acados_full_state`` layout."""
    from s500_uam_acados_state_tracking_mpc import _quat_to_euler_zyx_ca

    quat = x_sym[3:7]
    roll, pitch, yaw = _quat_to_euler_zyx_ca(quat)
    pieces: List["ca.SX"] = [x_sym[0:3], ca.vertcat(yaw, roll, pitch)]
    if n_arm > 0:
        pieces.append(x_sym[7 : 7 + n_arm])
    pieces.append(x_sym[nq : nq + 6])
    if n_arm > 0:
        pieces.append(x_sym[nq + 6 : nq + 6 + n_arm])
    return ca.vertcat(*pieces)


def _codegen_state_suffix(*, w_state_track: float, n_arm: int) -> str:
    """Codegen cache dir tag when optional state-reference block is enabled."""
    if float(w_state_track) <= 0.0:
        return ""
    return f"_st{int(n_arm)}"


def _interp_scalar(t: float, t_grid: np.ndarray, y_grid: np.ndarray) -> float:
    tg = np.asarray(t_grid, dtype=float).flatten()
    yg = np.asarray(y_grid, dtype=float).flatten()
    if tg.size == 0 or yg.size == 0:
        return 0.0
    if tg.size == 1:
        return float(yg[0])
    return float(np.interp(float(t), tg, yg))


def _interp_vec3(t: float, t_grid: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
    p = np.asarray(p_grid, dtype=float)
    if p.ndim == 1:
        return p.reshape(3).copy()
    return np.array(
        [_interp_scalar(t, t_grid, p[:, i]) for i in range(min(3, p.shape[1]))],
        dtype=float,
    )


def _apply_plan_joint_override(
    x_ref: Optional[np.ndarray],
    *,
    enabled: bool,
    joint_q: Tuple[float, float],
) -> Optional[np.ndarray]:
    """Replace planned arm joint angles in ``x_ref`` when override is enabled."""
    if not enabled or x_ref is None:
        return x_ref
    x = np.asarray(x_ref, dtype=float).reshape(-1).copy()
    if x.size >= 9:
        x[7] = float(joint_q[0])
        x[8] = float(joint_q[1])
    return x


def _interp_rows(t: float, t_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Interpolate each column of ``y_grid`` (shape ``(len(t_grid), n)``) at time ``t``."""
    y = np.asarray(y_grid, dtype=float)
    if y.ndim == 1:
        return y.reshape(-1).copy()
    return np.array(
        [_interp_scalar(t, t_grid, y[:, j]) for j in range(y.shape[1])],
        dtype=float,
    )


class AcadosEECentricRealtimeMPC:
    """Online Acados NMPC: EE position + heading + control regularization."""

    def __init__(
        self,
        *,
        urdf_path: str,
        dt_mpc: float,
        horizon: int,
        w_ee_pos: float = 500.0,
        w_ee_yaw: float = 200.0,
        w_ee_rot_rp: float = 1.0,
        w_state_track: float = 0.0,
        w_pos: float = 1.0,
        w_att: float = 1.0,
        w_joint: float = 1.0,
        w_vel: float = 1.0,
        w_omega: float = 1.0,
        w_joint_vel: float = 1.0,
        w_control: float = 1e-4,
        w_u_thrust: float = 1.0,
        w_u_joint_torque: float = 1.0,
        w_terminal_scale: float = 3.0,
        w_terminal_track: float = 100.0,
        max_iter: int = 40,
        solver_mode: str = "rti",
        integrator_type: str = "ERK",
        hpipm_mode: str = "SPEED",
        qp_iter_max: int = 20,
        qp_warm_start: int = 1,
        sim_num_stages: int = 4,
        sim_num_steps: int = 1,
        as_rti_iter: int = 0,
        dist_aware: bool = False,
        ref_error_limits: Optional[MpcRefErrorLimits] = None,
    ):
        if not ACADOS_AVAILABLE:
            raise ImportError("acados_template not installed")
        if not PINOCCHIO_AVAILABLE:
            raise ImportError(f"pinocchio/casadi required: {_pin_err}")
        if not DEPS_OK:
            raise ImportError(f"project deps unavailable: {_deps_err}")

        self.dt_mpc = float(dt_mpc)
        self.horizon = int(horizon)
        self.solver_mode = str(solver_mode).lower()
        self.integrator_type = str(integrator_type).upper()
        self.hpipm_mode = str(hpipm_mode).upper()
        self.qp_iter_max = int(qp_iter_max)
        self.qp_warm_start = int(qp_warm_start)
        self.sim_num_stages = int(sim_num_stages)
        self.sim_num_steps = int(sim_num_steps)
        self.as_rti_iter = int(as_rti_iter)
        self.dist_aware = bool(dist_aware)
        self.ref_error_limits = ref_error_limits

        self.s500_config = load_s500_config()
        if self.dist_aware:
            from s500_uam_acados_state_tracking_mpc import _build_robot_acados_model

            acados_model, pin_model, nq, nv, nu = _build_robot_acados_model(
                urdf_path, wholebody=False
            )
            self.n_dist_param = 6
        else:
            acados_model, pin_model, nq, nv, nu = self._build_robot_acados_model(urdf_path)
            self.n_dist_param = 0

        self.robot_model = pin_model
        self.acados_model = acados_model
        self.nq = int(nq)
        self.nv = int(nv)
        self.nu = int(nu)
        self.nx = self.nq + self.nv
        self.n_arm = self.nq - 7
        if self.n_arm <= 0:
            raise ValueError("acados_ee_pose requires an arm model (nq > 7)")

        self._planner = type("_PlannerShim", (), {"urdf_path": str(urdf_path)})()
        fid = int(pin_model.getFrameId(EE_FRAME_NAME))
        if fid < 0 or fid >= pin_model.nframes:
            raise ValueError(f"Frame '{EE_FRAME_NAME}' not found in URDF")
        self.ee_frame_id = fid
        self._pin_data = pin_model.createData()

        plat = self.s500_config["platform"]
        self.min_thrust = float(plat["min_thrust"])
        self.max_thrust = float(plat["max_thrust"])

        self._store_cost_weights(
            w_ee_pos=w_ee_pos,
            w_ee_yaw=w_ee_yaw,
            w_ee_rot_rp=w_ee_rot_rp,
            w_state_track=w_state_track,
            w_pos=w_pos,
            w_att=w_att,
            w_joint=w_joint,
            w_vel=w_vel,
            w_omega=w_omega,
            w_joint_vel=w_joint_vel,
            w_control=w_control,
            w_u_thrust=w_u_thrust,
            w_u_joint_torque=w_u_joint_torque,
            w_terminal_scale=w_terminal_scale,
            w_terminal_track=w_terminal_track,
        )
        self._build_solver(max_iter=max_iter)

        g = 9.81
        m = sum(self.robot_model.inertias[i].mass for i in range(1, len(self.robot_model.inertias)))
        t_each = float(np.clip(m * g / 4.0, self.min_thrust, self.max_thrust))
        self.u_hover = np.zeros(self.nu, dtype=float)
        self.u_hover[:4] = t_each

        self._xs_guess: Optional[list] = None
        self._us_guess: Optional[list] = None
        self.last_xs: Optional[list] = None
        self.last_status: int = 0
        self.last_sqp_iter: int = 0
        self.last_qp_iter: int = 0
        self.last_cost: float = float("nan")
        self.last_cpu_time_ms: float = float("nan")
        self.last_cost_terms: dict = {}

    def _build_robot_acados_model(self, urdf_path: str):
        from acados_template import AcadosModel
        from s500_uam_acados_model import _quat_prod, _quat_to_R

        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()

        platform = self.s500_config["platform"]
        cm_cf = platform["cm"] / platform["cf"]
        rotors = platform["$rotors"]

        nq, nv = model.nq, model.nv
        n_arm = nq - 7
        n_thrust = 4
        nu = n_thrust + n_arm

        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nv)
        u = ca.SX.sym("u", nu)

        thrusts = u[:n_thrust]
        arm_tau = u[n_thrust:]

        Fz = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
        Mx = My = Mz = 0.0
        for i, r in enumerate(rotors):
            pos = r["translation"]
            spin = r["spin_direction"][0]
            T = thrusts[i]
            Mx += -pos[1] * T
            My += pos[0] * T
            Mz += spin * cm_cf * T
        tau_base = ca.vertcat(0, 0, Fz, Mx, My, Mz)
        tau = ca.vertcat(tau_base, arm_tau)

        a = cpin.aba(cmodel, cdata, q, v, tau)

        quat = q[3:7]
        v_lin = v[:3]
        v_ang = v[3:6]
        R = _quat_to_R(quat)
        pos_dot = ca.mtimes(R, v_lin)
        quat_dot = 0.5 * _quat_prod(quat, ca.vertcat(v_ang[0], v_ang[1], v_ang[2], 0))
        q_dot = ca.vertcat(pos_dot, quat_dot, v[6 : 6 + n_arm])

        x = ca.vertcat(q, v)
        x_dot = ca.vertcat(q_dot, a)

        acados_model = AcadosModel()
        acados_model.name = "s500_uam_ee_rt"
        acados_model.x = x
        acados_model.u = u
        acados_model.xdot = ca.SX.sym("xdot", x.rows())
        acados_model.f_impl_expr = acados_model.xdot - x_dot
        acados_model.f_expl_expr = x_dot
        return acados_model, model, nq, nv, nu

    def _store_cost_weights(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, float(v))

    def _use_state_track(self) -> bool:
        return float(self.w_state_track) > 0.0

    def _state_codegen_kwargs(self) -> dict:
        return dict(w_state_track=float(self.w_state_track), n_arm=self.n_arm)

    def _state_cost_segments(self, *, terminal: bool) -> List[_CostSegment]:
        if not self._use_state_track():
            return []
        s = float(self.w_state_track)
        if terminal:
            s *= float(self.w_terminal_track)
        na = self.n_arm
        segs: List[_CostSegment] = [
            _CostSegment("st_pos", 3, (float(self.w_pos) * s,) * 3),
            _CostSegment("st_att", 3, (float(self.w_att) * s,) * 3),
        ]
        if na > 0:
            wj = float(self.w_joint) * s
            segs.append(_CostSegment("st_joint", na, (wj,) * na))
        segs.append(_CostSegment("st_vel", 3, (float(self.w_vel) * s,) * 3))
        segs.append(_CostSegment("st_omega", 3, (float(self.w_omega) * s,) * 3))
        if na > 0:
            wjv = float(self.w_joint_vel) * s
            segs.append(_CostSegment("st_joint_vel", na, (wjv,) * na))
        return segs

    def _task_aux_segments(self, *, terminal: bool) -> List[_CostSegment]:
        scale = float(self.w_terminal_scale) if terminal else 1.0
        segs: List[_CostSegment] = [
            _CostSegment(
                "ee_pos",
                3,
                (float(self.w_ee_pos) * scale,) * 3,
            ),
            _CostSegment(
                "ee_att",
                3,
                (
                    float(self.w_ee_yaw) * scale,
                    float(self.w_ee_rot_rp) * scale,
                    float(self.w_ee_rot_rp) * scale,
                ),
            ),
        ]
        segs.extend(self._state_cost_segments(terminal=terminal))
        return segs

    def _terminal_cost_segments(self) -> List[_CostSegment]:
        return self._task_aux_segments(terminal=True)

    def _running_cost_segments(self) -> List[_CostSegment]:
        segs = self._task_aux_segments(terminal=False)
        r_thrust = float(self.w_control) * float(self.w_u_thrust)
        r_torque = float(self.w_control) * float(self.w_u_joint_torque) * _JOINT_TORQUE_W_SCALE
        na = self.n_arm
        segs.append(_CostSegment("u_thrust", 4, (r_thrust,) * 4))
        segs.append(_CostSegment("u_torque", na, (r_torque,) * na))
        return segs

    def _W_from_segments(self, segments: List[_CostSegment]) -> np.ndarray:
        ny = sum(s.dim for s in segments)
        W = np.zeros((ny, ny))
        off = 0
        for seg in segments:
            for j, w in enumerate(seg.weights):
                W[off + j, off + j] = float(w)
            off += seg.dim
        return W

    def describe_cost_layout(self) -> str:
        """Human-readable cost layout (for logs / GUI)."""
        run_segs = self._running_cost_segments()
        term_segs = self._terminal_cost_segments()
        run_names = " | ".join(f"{s.name}({s.dim})" for s in run_segs)
        term_names = " | ".join(f"{s.name}({s.dim})" for s in term_segs)
        return (
            f"running y=[{run_names}]  (ny={sum(s.dim for s in run_segs)}); "
            f"terminal y_e=[{term_names}]  (ny_e={sum(s.dim for s in term_segs)})"
        )

    def _build_cost_y_exprs(self, ocp: "AcadosOcp") -> Tuple["ca.SX", "ca.SX"]:
        """Assemble CasADi ``cost_y`` / ``cost_y_e`` from EE task + optional state ref (+ u)."""
        ee_p = _casadi_ee_translation_expr(ocp.model, self.robot_model, self.ee_frame_id)
        ee_rpy = _casadi_ee_rpy_zyx_expr(ocp.model, self.robot_model, self.ee_frame_id)
        task: List["ca.SX"] = [ee_p, ee_rpy]
        if self._use_state_track():
            task.append(_casadi_state_y_expr(ocp.model.x, self.nq, self.n_arm))
        cost_y_e = ca.vertcat(*task)
        cost_y = ca.vertcat(cost_y_e, ocp.model.u)
        return cost_y, cost_y_e

    def _cost_W_and_W_e(self) -> Tuple[np.ndarray, np.ndarray]:
        W = self._W_from_segments(self._running_cost_segments())
        W_e = self._W_from_segments(self._terminal_cost_segments())
        return W, W_e

    def update_cost_weights(
        self,
        *,
        w_ee_pos: Optional[float] = None,
        w_ee_yaw: Optional[float] = None,
        w_ee_rot_rp: Optional[float] = None,
        w_state_track: Optional[float] = None,
        w_pos: Optional[float] = None,
        w_att: Optional[float] = None,
        w_joint: Optional[float] = None,
        w_vel: Optional[float] = None,
        w_omega: Optional[float] = None,
        w_joint_vel: Optional[float] = None,
        w_control: Optional[float] = None,
        w_u_thrust: Optional[float] = None,
        w_u_joint_torque: Optional[float] = None,
        w_terminal_scale: Optional[float] = None,
        w_terminal_track: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> None:
        loc = {
            "w_ee_pos": w_ee_pos,
            "w_ee_yaw": w_ee_yaw,
            "w_ee_rot_rp": w_ee_rot_rp,
            "w_state_track": w_state_track,
            "w_pos": w_pos,
            "w_att": w_att,
            "w_joint": w_joint,
            "w_vel": w_vel,
            "w_omega": w_omega,
            "w_joint_vel": w_joint_vel,
            "w_control": w_control,
            "w_u_thrust": w_u_thrust,
            "w_u_joint_torque": w_u_joint_torque,
            "w_terminal_scale": w_terminal_scale,
            "w_terminal_track": w_terminal_track,
        }
        for k, v in loc.items():
            if v is not None:
                setattr(self, k, float(v))
        mi = int(max_iter) if max_iter is not None else int(getattr(self, "_last_max_iter", 40))
        new_suffix = _codegen_state_suffix(**self._state_codegen_kwargs())
        old_suffix = str(getattr(self, "_last_state_suffix", new_suffix))
        need_rebuild = (
            (self.solver_mode != "rti" and mi != int(getattr(self, "_last_max_iter", mi)))
            or new_suffix != old_suffix
        )
        applied_runtime = False
        if not need_rebuild:
            try:
                self._apply_runtime_solver_weights()
                applied_runtime = True
                self._last_max_iter = mi
            except Exception as exc:
                import rospy

                rospy.logwarn("[acados-ee] runtime cost_set failed (%s); rebuilding", exc)
        if not applied_runtime:
            self._build_solver(max_iter=mi)
        self.reset_warm_start()

    def _build_solver(self, *, max_iter: int):
        self._last_max_iter = int(max_iter)
        self._last_state_suffix = _codegen_state_suffix(**self._state_codegen_kwargs())
        nq, nv, nu, nx, na = self.nq, self.nv, self.nu, self.nx, self.n_arm

        ocp = AcadosOcp()
        ocp.model = self.acados_model
        if self.dist_aware and self.n_dist_param > 0:
            ocp.parameter_values = np.zeros(self.n_dist_param)

        cost_y, cost_y_e = self._build_cost_y_exprs(ocp)

        ocp.model.cost_y_expr = cost_y
        ocp.model.cost_y_expr_e = cost_y_e
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        W, W_e = self._cost_W_and_W_e()
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        self._ny = int(cost_y.shape[0])
        self._ny_e = int(cost_y_e.shape[0])
        ocp.cost.yref = np.zeros(self._ny)
        ocp.cost.yref_e = np.zeros(self._ny_e)

        N = self.horizon
        ocp.dims.N = int(N)
        ocp.solver_options.tf = float(N) * self.dt_mpc
        codegen_nlp_iter = max(_ACADOS_CODEGEN_NLP_MAX_ITER, int(max_iter))
        codegen_qp_iter = max(_ACADOS_CODEGEN_QP_ITER_MAX, int(max_iter) * 2)
        ocp.solver_options.nlp_solver_max_iter = codegen_nlp_iter
        if hasattr(ocp.solver_options, "N_horizon"):
            ocp.solver_options.N_horizon = int(N)

        v_max = STATE_LIMITS["v_max"]
        om_max = STATE_LIMITS["omega_max"]
        j_max = STATE_LIMITS["j_angle_max"]
        jv_max = STATE_LIMITS["j_vel_max"]

        ocp.constraints.lbu = np.array([self.min_thrust] * 4 + [-2.0] * na)
        ocp.constraints.ubu = np.array([self.max_thrust] * 4 + [2.0] * na)
        ocp.constraints.idxbu = np.arange(nu)

        lbx = np.concatenate([
            np.array([-50.0, -50.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([-j_max] * na),
            np.array([-v_max, -v_max, -v_max, -om_max, -om_max, -om_max]),
            np.array([-jv_max] * na),
        ])
        ubx = np.concatenate([
            np.array([50.0, 50.0, 20.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([j_max] * na),
            np.array([v_max, v_max, v_max, om_max, om_max, om_max]),
            np.array([jv_max] * na),
        ])
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx = lbx
        ocp.constraints.ubx = ubx
        ocp.constraints.x0 = np.zeros(nx)

        # 软化全部状态箱约束（与 full-state RT MPC 一致）
        ocp.constraints.idxsbx = np.arange(nx)
        ns = nx
        z_quad, z_lin = 1.0e2, 1.0e1
        ocp.cost.Zl = z_quad * np.ones(ns)
        ocp.cost.Zu = z_quad * np.ones(ns)
        ocp.cost.zl = z_lin * np.ones(ns)
        ocp.cost.zu = z_lin * np.ones(ns)
        ocp.constraints.idxbx_e = np.arange(nx)
        ocp.constraints.lbx_e = lbx.copy()
        ocp.constraints.ubx_e = ubx.copy()
        ocp.constraints.idxsbx_e = np.arange(nx)
        ocp.cost.Zl_e = z_quad * np.ones(ns)
        ocp.cost.Zu_e = z_quad * np.ones(ns)
        ocp.cost.zl_e = z_lin * np.ones(ns)
        ocp.cost.zu_e = z_lin * np.ones(ns)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = self.integrator_type
        ocp.solver_options.print_level = 0
        if self.solver_mode == "rti":
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
        else:
            ocp.solver_options.nlp_solver_type = "SQP"
            if hasattr(ocp.solver_options, "nlp_solver_max_iter"):
                ocp.solver_options.nlp_solver_max_iter = int(max_iter)
            if hasattr(ocp.solver_options, "globalization"):
                ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        if hasattr(ocp.solver_options, "qp_solver_iter_max"):
            ocp.solver_options.qp_solver_iter_max = int(codegen_qp_iter)
        if hasattr(ocp.solver_options, "qp_solver_warm_start"):
            ocp.solver_options.qp_solver_warm_start = int(self.qp_warm_start)
        if hasattr(ocp.solver_options, "hpipm_mode"):
            ocp.solver_options.hpipm_mode = self.hpipm_mode
        if hasattr(ocp.solver_options, "levenberg_marquardt"):
            ocp.solver_options.levenberg_marquardt = 1e-3
        if hasattr(ocp.solver_options, "qp_solver_cond_N"):
            ocp.solver_options.qp_solver_cond_N = max(1, min(int(N), 5))
        if hasattr(ocp.solver_options, "sim_method_num_stages"):
            ocp.solver_options.sim_method_num_stages = int(self.sim_num_stages)
        if hasattr(ocp.solver_options, "sim_method_num_steps"):
            ocp.solver_options.sim_method_num_steps = int(self.sim_num_steps)
        if self.solver_mode == "rti" and int(self.as_rti_iter) > 0:
            if hasattr(ocp.solver_options, "as_rti_iter"):
                ocp.solver_options.as_rti_iter = int(self.as_rti_iter)
            if hasattr(ocp.solver_options, "as_rti_level"):
                ocp.solver_options.as_rti_level = 3

        da_tag = "_da" if self.dist_aware else ""
        aux_tag = self._last_state_suffix
        code_export_dir = REPO_ROOT / "c_generated_code" / f"s500_uam_ee_rt_rpy6_mpc{da_tag}{aux_tag}"
        code_export_dir.mkdir(parents=True, exist_ok=True)
        json_path = code_export_dir / "ocp.json"
        ocp.code_gen_opts.code_export_directory = str(code_export_dir)
        ocp.code_gen_opts.json_file = str(json_path)

        # 权重/迭代上界等会进入 OCP 哈希；对齐已缓存 json 后仅加载 .so（<1s），
        # 运行时用 cost_set / options_set 覆盖当前权重与 max_iter（与 full-state RT 一致）。
        _align_ocp_with_cached_json(ocp, json_path)
        gen, bld = _acados_ocp_generate_build_flags(code_export_dir, self.acados_model.name)
        t_solver0 = time.perf_counter()
        try:
            self.solver = AcadosOcpSolver(
                ocp,
                json_file=str(json_path),
                generate=gen,
                build=bld,
                verbose=False,
                check_reuse_possible=True,
            )
        except TypeError:
            self.solver = AcadosOcpSolver(
                ocp,
                json_file=str(json_path),
                generate=gen,
                build=bld,
            )
        solver_setup_s = time.perf_counter() - t_solver0
        if solver_setup_s > 1.0 or getattr(self.solver, "generated", False):
            tag = "codegen+build" if getattr(self.solver, "generated", False) else "build"
            print(
                f"[acados-ee rt] OCP {tag} {solver_setup_s:.1f}s "
                f"dir={code_export_dir.name} (结构变化才需重编；同结构二次启动应 <1s)\n"
                f"  cost: {self.describe_cost_layout()}"
            )
        self._apply_runtime_solver_weights()
        nlp_type = str(getattr(ocp.solver_options, "nlp_solver_type", "SQP_RTI"))
        _apply_runtime_solver_options(
            self.solver,
            int(max_iter),
            solver_type=nlp_type,
            qp_iter_max=int(self.qp_iter_max),
            levenberg_marquardt=1e-3,
        )

    def _apply_runtime_solver_weights(self) -> None:
        W, W_e = self._cost_W_and_W_e()
        for i in range(self.horizon):
            self.solver.cost_set(i, "W", W)
        self.solver.cost_set(self.horizon, "W", W_e)

    def _ee_to_y(
        self,
        p_ee: np.ndarray,
        yaw: float,
        roll: float,
        pitch: float,
        *,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        parts: List[np.ndarray] = [
            np.asarray(p_ee, dtype=float).reshape(3),
            np.array([float(yaw), float(roll), float(pitch)], dtype=float),
        ]
        if self._use_state_track() and x_ref is not None:
            from s500_uam_acados_state_tracking_mpc import _state_to_y_np

            parts.append(
                _state_to_y_np(x_ref, self.nq, self.n_arm)
            )
        y_e = np.concatenate(parts)
        if u_ref is None:
            u_ref = self.u_hover
        y = np.concatenate([y_e, np.asarray(u_ref, dtype=float).reshape(self.nu)])
        return y, y_e

    def _current_ee_pose(self, x_now: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """FK: EE position + roll/pitch/yaw (rad, Pinocchio ZYX)."""
        q = np.asarray(x_now, dtype=float).flatten()[: self.nq].copy()
        qn = float(np.linalg.norm(q[3:7]))
        if qn > 1e-12:
            q[3:7] /= qn
        pin.forwardKinematics(self.robot_model, self._pin_data, q)
        pin.updateFramePlacements(self.robot_model, self._pin_data)
        oMf = self._pin_data.oMf[self.ee_frame_id]
        rpy = pin.rpy.matrixToRpy(oMf.rotation)
        return (
            np.asarray(oMf.translation, dtype=float).reshape(3),
            float(rpy[2]),
            float(rpy[0]),
            float(rpy[1]),
        )

    def solve_step(
        self,
        x_now: np.ndarray,
        t_query: float,
        t_ee_ref: np.ndarray,
        p_ee_ref: np.ndarray,
        yaw_ee_ref: np.ndarray,
        roll_ee_ref: np.ndarray,
        pitch_ee_ref: np.ndarray,
        *,
        t_plan: Optional[np.ndarray] = None,
        x_plan: Optional[np.ndarray] = None,
        joint_override: bool = False,
        joint_override_q: Tuple[float, float] = (0.0, 0.0),
        p_dist: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """One MPC step. EE refs along ``t_ee_ref``; state ref from ``x_plan`` when enabled."""
        from s500_uam_crocoddyl_state_tracking_mpc import interp_full_state_piecewise

        x_now = sanitize_mpc_state(x_now, nq=self.nq, nv=self.nv)
        N = self.horizon
        dt = self.dt_mpc
        use_xplan = (
            self._use_state_track()
            and t_plan is not None
            and x_plan is not None
        )

        if self.dist_aware and self.n_dist_param > 0:
            p_stage = np.zeros(self.n_dist_param, dtype=float)
            if p_dist is not None:
                pd = np.asarray(p_dist, dtype=float).reshape(-1)
                n_take = min(self.n_dist_param, pd.size)
                p_stage[:n_take] = pd[:n_take]

        p_ee_now, yaw_ee_now, roll_ee_now, pitch_ee_now = self._current_ee_pose(x_now)

        for i in range(N):
            t_i = float(t_query) + i * dt
            p_i = _interp_vec3(t_i, t_ee_ref, p_ee_ref)
            yaw_i = _interp_scalar(t_i, t_ee_ref, yaw_ee_ref)
            roll_i = _interp_scalar(t_i, t_ee_ref, roll_ee_ref)
            pitch_i = _interp_scalar(t_i, t_ee_ref, pitch_ee_ref)
            p_i, yaw_i, roll_i, pitch_i = clamp_mpc_ee_pose_reference(
                p_i,
                yaw_i,
                roll_i,
                pitch_i,
                p_ee_now,
                yaw_ee_now,
                roll_ee_now,
                pitch_ee_now,
                self.ref_error_limits,
            )
            x_ref_i = None
            if use_xplan:
                x_ref_i = interp_full_state_piecewise(
                    t_i, t_plan, x_plan, self.robot_model
                )
                x_ref_i = _apply_plan_joint_override(
                    x_ref_i,
                    enabled=joint_override,
                    joint_q=joint_override_q,
                )
                x_ref_i = clamp_mpc_state_reference(
                    x_ref_i, x_now, self.ref_error_limits, nq=self.nq, nv=self.nv
                )
            yref, _ = self._ee_to_y(p_i, yaw_i, roll_i, pitch_i, x_ref=x_ref_i)
            self.solver.set(i, "yref", yref)
            if self.dist_aware and self.n_dist_param > 0:
                self.solver.set(i, "p", p_stage)

        t_N = float(t_query) + N * dt
        p_N = _interp_vec3(t_N, t_ee_ref, p_ee_ref)
        yaw_N = _interp_scalar(t_N, t_ee_ref, yaw_ee_ref)
        roll_N = _interp_scalar(t_N, t_ee_ref, roll_ee_ref)
        pitch_N = _interp_scalar(t_N, t_ee_ref, pitch_ee_ref)
        p_N, yaw_N, roll_N, pitch_N = clamp_mpc_ee_pose_reference(
            p_N,
            yaw_N,
            roll_N,
            pitch_N,
            p_ee_now,
            yaw_ee_now,
            roll_ee_now,
            pitch_ee_now,
            self.ref_error_limits,
        )
        x_ref_N = None
        if use_xplan:
            x_ref_N = interp_full_state_piecewise(
                t_N, t_plan, x_plan, self.robot_model
            )
            x_ref_N = _apply_plan_joint_override(
                x_ref_N,
                enabled=joint_override,
                joint_q=joint_override_q,
            )
            x_ref_N = clamp_mpc_state_reference(
                x_ref_N, x_now, self.ref_error_limits, nq=self.nq, nv=self.nv
            )
        _, yref_e = self._ee_to_y(p_N, yaw_N, roll_N, pitch_N, x_ref=x_ref_N)
        self.solver.set(N, "yref", yref_e)

        self.solver.set(0, "lbx", x_now)
        self.solver.set(0, "ubx", x_now)

        if self._xs_guess is None:
            for i in range(N + 1):
                self.solver.set(i, "x", x_now.copy())
            for i in range(N):
                self.solver.set(i, "u", self.u_hover.copy())
        else:
            self._xs_guess[0] = x_now.copy()
            for i in range(N + 1):
                self.solver.set(i, "x", self._xs_guess[i])
            for i in range(N):
                self.solver.set(i, "u", self._us_guess[i])

        status = acados_mpc_run_solve(self.solver)
        u_opt, x_next, xs, us = acados_mpc_extract_solution(self.solver, N)

        if status not in ACADOS_OK_STATUSES or not acados_solution_finite(
            u_opt, x_next, xs
        ):
            for i in range(N + 1):
                self.solver.set(i, "x", x_now.copy())
            for i in range(N):
                self.solver.set(i, "u", self.u_hover.copy())
            status = acados_mpc_run_solve(self.solver, retry_lm=0.1)
            u_opt, x_next, xs, us = acados_mpc_extract_solution(self.solver, N)

        self.last_status = status
        self.last_sqp_iter = int(self._get_stat("sqp_iter"))
        self.last_qp_iter = int(self._get_stat("qp_iter"))
        self.last_cpu_time_ms = float(self._get_stat("time_tot")) * 1000.0
        try:
            self.last_cost = float(self.solver.get_cost())
        except Exception:
            self.last_cost = float("nan")

        if status not in ACADOS_OK_STATUSES or not acados_solution_finite(
            u_opt, x_next, xs
        ):
            self._xs_guess = None
            self._us_guess = None
            self.last_xs = None
            self.last_cost_terms = {}
            return None, None, status

        self.last_xs = xs
        self._xs_guess = xs[1:] + [xs[-1].copy()]
        self._us_guess = us[1:] + [us[-1].copy()]
        self.last_cost_terms = {}
        return u_opt, x_next, status

    def _get_stat(self, name: str) -> float:
        try:
            v = self.solver.get_stats(name)
            arr = np.asarray(v).flatten()
            return float(arr[-1]) if arr.size else 0.0
        except Exception:
            return 0.0

    def warmup(
        self,
        x0: np.ndarray,
        t_ee_ref: np.ndarray,
        p_ee_ref: np.ndarray,
        yaw_ee_ref: np.ndarray,
        roll_ee_ref: Optional[np.ndarray] = None,
        pitch_ee_ref: Optional[np.ndarray] = None,
        iters: int = 5,
    ):
        x0 = np.asarray(x0, dtype=float).flatten()[: self.nx]
        t0 = float(np.asarray(t_ee_ref, dtype=float).flatten()[0]) if np.asarray(t_ee_ref).size else 0.0
        yaw_ee_ref = np.asarray(yaw_ee_ref, dtype=float).flatten()
        if roll_ee_ref is None:
            roll_ee_ref = np.zeros_like(yaw_ee_ref)
        if pitch_ee_ref is None:
            pitch_ee_ref = np.zeros_like(yaw_ee_ref)
        for _ in range(max(1, int(iters))):
            try:
                self.solve_step(
                    x0,
                    t0,
                    t_ee_ref,
                    p_ee_ref,
                    yaw_ee_ref,
                    roll_ee_ref,
                    pitch_ee_ref,
                )
            except Exception:
                break

    def reset_warm_start(self):
        self._xs_guess = None
        self._us_guess = None
