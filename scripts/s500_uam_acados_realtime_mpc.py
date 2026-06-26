#!/usr/bin/env python3
"""Real-time (single-step) Acados full-state tracking MPC.

This is the *online* counterpart of ``s500_uam_acados_state_tracking_mpc`` (which
runs an offline closed-loop simulation). It is robot-aware so it works for both:

  * s500      : nq=7, nv=6, nu=4  (4 rotor thrusts, no arm)
  * s500_uam  : nq=9, nv=8, nu=6  (4 thrusts + 2 joint torques)

The cost layout / weights mirror the Crocoddyl full-state tracking weights so that
``acados_full_state`` and ``croc_full_state`` in the ROS controller are comparable.

Usage (ROS controller):
    mpc = AcadosFullStateRealtimeMPC(urdf_path=..., dt_mpc=..., horizon=..., ...)
    u_opt, x_next, status = mpc.solve_step(x_now, t_query, t_plan, x_plan)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

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

    from s500_uam_acados_model import _quat_prod, _quat_to_R, load_s500_config
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
        clamp_mpc_state_reference,
        sanitize_mpc_state,
    )
    from s500_uam_acados_trajectory import (
        STATE_LIMITS,
        _acados_ocp_generate_build_flags,
    )
    from s500_uam_crocoddyl_state_tracking_mpc import interp_full_state_piecewise

    DEPS_OK = True
except ImportError as e:  # pragma: no cover
    DEPS_OK = False
    _deps_err = e

REPO_ROOT = Path(__file__).resolve().parent.parent


def _quat_to_euler_zyx_ca(quat):
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    roll = ca.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = 2 * (qw * qy - qz * qx)
    sinp = ca.fmin(1, ca.fmax(-1, sinp))
    pitch = ca.asin(sinp)
    yaw = ca.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


def _quat_to_euler_zyx_np(qx, qy, qz, qw):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


class AcadosFullStateRealtimeMPC:
    """Robot-aware online Acados NMPC tracking a full-state plan ``(t_plan, x_plan)``."""

    def __init__(
        self,
        *,
        urdf_path: str,
        dt_mpc: float,
        horizon: int,
        w_pos: float = 1.0,
        w_att: float = 1.0,
        w_joint: float = 1.0,
        w_vel: float = 1.0,
        w_omega: float = 1.0,
        w_joint_vel: float = 1.0,
        w_control: float = 1e-3,
        w_u_thrust: float = 1.0,
        w_u_joint_torque: float = 1.0,
        w_state_track: float = 10.0,
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
        # 求解器实时性配置（默认面向 100Hz：RTI + ERK + HPIPM SPEED）
        self.solver_mode = str(solver_mode).lower()          # "rti" | "sqp"
        self.integrator_type = str(integrator_type).upper()  # "ERK" | "IRK"
        self.hpipm_mode = str(hpipm_mode).upper()             # "SPEED" | "BALANCE" | "ROBUST"
        self.qp_iter_max = int(qp_iter_max)
        # 进一步的实时/精度调参
        self.qp_warm_start = int(qp_warm_start)   # QP 热启动（1=用上一拍 QP 解，减少 qp_iter）
        self.sim_num_stages = int(sim_num_stages) # 积分器级数（ERK: 4=RK4）
        self.sim_num_steps = int(sim_num_steps)   # 每个 shooting 区间内积分子步数（>1 提升离散精度）
        self.as_rti_iter = int(as_rti_iter)       # AS-RTI 迭代次数（0=普通 RTI）
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
        if self.n_arm < 0:
            raise ValueError(f"Unexpected nq={self.nq} (<7)")

        # Visualization shims expected by the ROS controller (robot_model FK + EE).
        self._planner = type("_PlannerShim", (), {"urdf_path": str(urdf_path)})()
        self.ee_frame_id = None
        if self.n_arm > 0:
            try:
                from s500_uam_trajectory_planner import S500UAMTrajectoryPlanner

                ee_name = getattr(S500UAMTrajectoryPlanner, "EE_FRAME_NAME", "gripper_link")
            except Exception:
                ee_name = "gripper_link"
            fid = self.robot_model.getFrameId(ee_name)
            if 0 <= int(fid) < len(self.robot_model.frames):
                self.ee_frame_id = int(fid)

        plat = self.s500_config["platform"]
        self.min_thrust = float(plat["min_thrust"])
        self.max_thrust = float(plat["max_thrust"])

        self._store_cost_weights(
            w_pos=w_pos, w_att=w_att, w_joint=w_joint, w_vel=w_vel,
            w_omega=w_omega, w_joint_vel=w_joint_vel, w_control=w_control,
            w_u_thrust=w_u_thrust, w_u_joint_torque=w_u_joint_torque,
            w_state_track=w_state_track, w_terminal_track=w_terminal_track,
        )
        self._build_solver(max_iter=max_iter)

        # hover control reference for control regularization
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

    # ------------------------------------------------------------------ model
    def _build_robot_acados_model(self, urdf_path: str):
        """Robot-aware AcadosModel (handles n_arm = nq-7, nu = 4 + n_arm)."""
        from acados_template import AcadosModel

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
        arm_tau = u[n_thrust:] if n_arm > 0 else None

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
        tau = ca.vertcat(tau_base, arm_tau) if n_arm > 0 else tau_base

        a = cpin.aba(cmodel, cdata, q, v, tau)

        quat = q[3:7]
        v_lin = v[:3]
        v_ang = v[3:6]
        R = _quat_to_R(quat)
        pos_dot = ca.mtimes(R, v_lin)
        quat_dot = 0.5 * _quat_prod(quat, ca.vertcat(v_ang[0], v_ang[1], v_ang[2], 0))
        if n_arm > 0:
            q_dot = ca.vertcat(pos_dot, quat_dot, v[6 : 6 + n_arm])
        else:
            q_dot = ca.vertcat(pos_dot, quat_dot)

        x = ca.vertcat(q, v)
        x_dot = ca.vertcat(q_dot, a)

        acados_model = AcadosModel()
        acados_model.name = "s500_uam_rt" if n_arm > 0 else "s500_rt"
        acados_model.x = x
        acados_model.u = u
        acados_model.xdot = ca.SX.sym("xdot", x.rows())
        acados_model.f_impl_expr = acados_model.xdot - x_dot
        acados_model.f_expl_expr = x_dot
        return acados_model, model, nq, nv, nu

    # ------------------------------------------------------------------ build
    def _store_cost_weights(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, float(v))

    def _state_cost_weights(self) -> np.ndarray:
        s = float(self.w_state_track)
        diag = [self.w_pos * s] * 3 + [self.w_att * s] * 3
        diag += [self.w_joint * s] * self.n_arm
        diag += [self.w_vel * s] * 3 + [self.w_omega * s] * 3
        diag += [self.w_joint_vel * s] * self.n_arm
        return np.diag(np.asarray(diag, dtype=float))

    def _cost_W_and_W_e(self) -> Tuple[np.ndarray, np.ndarray]:
        na = self.n_arm
        W_state = self._state_cost_weights()
        r_thrust = float(self.w_control) * float(self.w_u_thrust)
        r_torque = float(self.w_control) * float(self.w_u_joint_torque) * 10000.0
        R = np.diag([r_thrust] * 4 + [r_torque] * na)
        W = np.diag(np.concatenate([np.diag(W_state), np.diag(R)]))
        W_e = W_state * float(self.w_terminal_track)
        return W, W_e

    def update_cost_weights(
        self,
        *,
        w_pos: Optional[float] = None,
        w_att: Optional[float] = None,
        w_joint: Optional[float] = None,
        w_vel: Optional[float] = None,
        w_omega: Optional[float] = None,
        w_joint_vel: Optional[float] = None,
        w_control: Optional[float] = None,
        w_u_thrust: Optional[float] = None,
        w_u_joint_torque: Optional[float] = None,
        w_state_track: Optional[float] = None,
        w_terminal_track: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> None:
        """在线更新代价权重（重建 OCP 求解器，复用已生成的 C 代码目录）。"""
        loc = {
            "w_pos": w_pos, "w_att": w_att, "w_joint": w_joint,
            "w_vel": w_vel, "w_omega": w_omega, "w_joint_vel": w_joint_vel,
            "w_control": w_control, "w_u_thrust": w_u_thrust,
            "w_u_joint_torque": w_u_joint_torque,
            "w_state_track": w_state_track, "w_terminal_track": w_terminal_track,
        }
        for k, v in loc.items():
            if v is not None:
                setattr(self, k, float(v))
        mi = int(max_iter) if max_iter is not None else int(getattr(self, "_last_max_iter", 40))

        # For NONLINEAR_LS the weight matrix W is a *runtime* parameter, so update it
        # in-place via cost_set on the existing solver. This is instant and, unlike
        # re-creating AcadosOcpSolver (which reuses cached/compiled code via
        # check_reuse_possible and silently keeps the OLD W), it is guaranteed to take
        # effect. A full rebuild is only needed when max_iter must change in SQP mode.
        need_rebuild = (
            self.solver_mode != "rti"
            and mi != int(getattr(self, "_last_max_iter", mi))
        )
        applied_runtime = False
        if not need_rebuild:
            try:
                self._apply_runtime_solver_weights()
                applied_runtime = True
                self._last_max_iter = mi
            except Exception as exc:
                applied_runtime = False
                import rospy
                rospy.logwarn(
                    "[acados] runtime cost_set failed (%s); rebuilding solver", exc
                )
        if not applied_runtime:
            self._build_solver(max_iter=mi)
        self.reset_warm_start()

    def _build_solver(self, *, max_iter: int):
        self._last_max_iter = int(max_iter)
        nq, nv, nu, nx, na = self.nq, self.nv, self.nu, self.nx, self.n_arm

        ocp = AcadosOcp()
        ocp.model = self.acados_model
        if self.dist_aware and self.n_dist_param > 0:
            ocp.parameter_values = np.zeros(self.n_dist_param)

        x = ocp.model.x
        u = ocp.model.u
        pos = x[0:3]
        quat = x[3:7]
        roll, pitch, yaw = _quat_to_euler_zyx_ca(quat)
        jq = x[7 : 7 + na] if na > 0 else None
        v_lin = x[nq : nq + 3]
        omega = x[nq + 3 : nq + 6]
        jv = x[nq + 6 : nq + 6 + na] if na > 0 else None

        pieces = [pos, ca.vertcat(yaw, roll, pitch)]
        if na > 0:
            pieces.append(jq)
        pieces += [v_lin, omega]
        if na > 0:
            pieces.append(jv)
        cost_y_e = ca.vertcat(*pieces)
        cost_y = ca.vertcat(cost_y_e, u)

        ocp.model.cost_y_expr = cost_y
        ocp.model.cost_y_expr_e = cost_y_e
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        W, W_e = self._cost_W_and_W_e()
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        ny = int(cost_y.shape[0])
        ny_e = int(cost_y_e.shape[0])
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        self._ny = ny
        self._ny_e = ny_e

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

        # NOTE: z 下界放宽到 -1.0（允许地面起飞 z≈0），其余状态保留物理上限。
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

        # 软化全部状态箱约束（slack）：保证 QP 在大初始误差（如地面→1m 起飞）下
        # 始终可行，避免 HPIPM ACADOS_MINSTEP / NaN 求解失败。
        ocp.constraints.idxsbx = np.arange(nx)
        ns = nx
        z_quad = 1.0e2
        z_lin = 1.0e1
        ocp.cost.Zl = z_quad * np.ones(ns)
        ocp.cost.Zu = z_quad * np.ones(ns)
        ocp.cost.zl = z_lin * np.ones(ns)
        ocp.cost.zu = z_lin * np.ones(ns)
        # 终端状态约束同样软化（需先定义终端箱约束）
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
        # ERK（显式）积分器对四旋翼这类非刚性动力学足够，且比 IRK 快很多。
        ocp.solver_options.integrator_type = self.integrator_type
        ocp.solver_options.print_level = 0
        # 实时迭代（RTI）：每个控制步只做 1 次 SQP 迭代，依赖跨步 warm-start，
        # 是实时 NMPC（>=100Hz）的标准做法；full SQP 每步迭代多次故慢得多。
        if self.solver_mode == "rti":
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
        else:
            ocp.solver_options.nlp_solver_type = "SQP"
            if hasattr(ocp.solver_options, "nlp_solver_max_iter"):
                ocp.solver_options.nlp_solver_max_iter = int(max_iter)
            # full SQP 才需要线搜索做全局化
            if hasattr(ocp.solver_options, "globalization"):
                ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        if hasattr(ocp.solver_options, "qp_solver_iter_max"):
            ocp.solver_options.qp_solver_iter_max = int(codegen_qp_iter)
        # QP 热启动：用上一拍 QP 解作为本拍 QP 初值，显著减少 HPIPM 迭代数。
        # RTI 跨步 warm-start 配合 QP warm-start 是实时 NMPC 的常见组合。
        if hasattr(ocp.solver_options, "qp_solver_warm_start"):
            ocp.solver_options.qp_solver_warm_start = int(self.qp_warm_start)
        # HPIPM 模式：SPEED 最快；大初始误差不收敛时可调 BALANCE/ROBUST。
        if hasattr(ocp.solver_options, "hpipm_mode"):
            ocp.solver_options.hpipm_mode = self.hpipm_mode
        # 轻正则，保持 KKT 良态（开销极小）。
        if hasattr(ocp.solver_options, "levenberg_marquardt"):
            ocp.solver_options.levenberg_marquardt = 1e-3
        # 部分凝聚的 horizon（对小 N 取较小值可减小 QP 规模、提速）。
        if hasattr(ocp.solver_options, "qp_solver_cond_N"):
            ocp.solver_options.qp_solver_cond_N = max(1, min(int(N), 5))
        # 积分器精度：每个 shooting 区间内的级数/子步数。dt_mpc 较大（如 50ms）
        # 且轨迹激烈时，单步 RK4 的离散误差会让 MPC 解“切角”、贴不住离线 plan；
        # 把 num_steps 提到 2~4 可在 dt_mpc 不变的前提下改善离散精度（开销线性增加）。
        if hasattr(ocp.solver_options, "sim_method_num_stages"):
            ocp.solver_options.sim_method_num_stages = int(self.sim_num_stages)
        if hasattr(ocp.solver_options, "sim_method_num_steps"):
            ocp.solver_options.sim_method_num_steps = int(self.sim_num_steps)
        # AS-RTI（advanced-step real-time iteration）：在保持实时的前提下，通过
        # 额外的内层迭代提升 RTI 对激烈/非线性轨迹的收敛质量；0=普通单次 RTI。
        if self.solver_mode == "rti" and int(self.as_rti_iter) > 0:
            if hasattr(ocp.solver_options, "as_rti_iter"):
                ocp.solver_options.as_rti_iter = int(self.as_rti_iter)
            if hasattr(ocp.solver_options, "as_rti_level"):
                # acados: LEVEL-A=0, B=1, C=2, LEVEL-D=3, STANDARD_RTI=4。
                # LEVEL-D(3) 对全部 KKT 做 advanced-step 更新，质量最好（配合
                # as_rti_iter>0 才有意义；4 等价普通 RTI）。
                ocp.solver_options.as_rti_level = 3

        robot_tag = "uam" if na > 0 else "s500"
        da_tag = "_da" if self.dist_aware else ""
        code_export_dir = (
            REPO_ROOT / "c_generated_code" / f"s500_{robot_tag}_full_state_rt_mpc{da_tag}"
        )
        code_export_dir.mkdir(parents=True, exist_ok=True)
        json_path = code_export_dir / "ocp.json"
        ocp.code_gen_opts.code_export_directory = str(code_export_dir)
        ocp.code_gen_opts.json_file = str(json_path)

        # 权重/迭代上界等会进入 OCP 哈希；对齐已缓存 json 后仅加载 .so（<1s），
        # 运行时用 cost_set / options_set 覆盖当前权重与 max_iter。
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
                f"[acados rt] OCP {tag} {solver_setup_s:.1f}s "
                f"dir={code_export_dir.name} (结构变化才需重编；同结构二次启动应 <1s)"
            )
        self._apply_runtime_solver_weights()
        nlp_type = str(
            getattr(ocp.solver_options, "nlp_solver_type", "SQP_RTI")
        )
        _apply_runtime_solver_options(
            self.solver,
            int(max_iter),
            solver_type=nlp_type,
            qp_iter_max=int(self.qp_iter_max),
            levenberg_marquardt=1e-3,
        )

    def _apply_runtime_solver_weights(self) -> None:
        """Push current W / W_e to loaded solver (no codegen)."""
        W, W_e = self._cost_W_and_W_e()
        for i in range(self.horizon):
            self.solver.cost_set(i, "W", W)
        self.solver.cost_set(self.horizon, "W", W_e)

    # ------------------------------------------------------------------ refs
    def _state_to_y(self, x_ref: np.ndarray) -> np.ndarray:
        x_ref = np.asarray(x_ref, dtype=float).flatten()
        nq, na = self.nq, self.n_arm
        roll, pitch, yaw = _quat_to_euler_zyx_np(x_ref[3], x_ref[4], x_ref[5], x_ref[6])
        parts = [x_ref[0:3], np.array([yaw, roll, pitch])]
        if na > 0:
            parts.append(x_ref[7 : 7 + na])
        parts.append(x_ref[nq : nq + 6])  # v_lin(3) + omega(3)
        if na > 0:
            parts.append(x_ref[nq + 6 : nq + 6 + na])
        return np.concatenate(parts)

    # ------------------------------------------------------------------ solve
    def solve_step(
        self,
        x_now: np.ndarray,
        t_query: float,
        t_plan: np.ndarray,
        x_plan: np.ndarray,
        p_dist: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Solve one MPC step. Returns (u_opt(nu), x_next(nx), status).

        ``p_dist`` (6,) 世界系 base wrench [Fx,Fy,Fz,Mx,My,Mz]；仅 ``dist_aware=True``
        时写入 OCP 参数，实现增广 MPC（offset-free）。
        """
        x_now = sanitize_mpc_state(x_now, nq=self.nq, nv=self.nv)
        N = self.horizon
        dt = self.dt_mpc
        rm = self.robot_model
        if self.dist_aware and self.n_dist_param > 0:
            p_stage = np.zeros(self.n_dist_param, dtype=float)
            if p_dist is not None:
                pd = np.asarray(p_dist, dtype=float).reshape(-1)
                n_take = min(self.n_dist_param, pd.size)
                p_stage[:n_take] = pd[:n_take]

        # Stage references along the plan
        yref_running = []
        for i in range(N):
            x_ref_i = interp_full_state_piecewise(
                t_query + i * dt, t_plan, x_plan, rm
            )
            x_ref_i = clamp_mpc_state_reference(
                x_ref_i, x_now, self.ref_error_limits, nq=self.nq, nv=self.nv
            )
            yref = np.concatenate([self._state_to_y(x_ref_i), self.u_hover])
            yref_running.append(yref.copy())
            self.solver.set(i, "yref", yref)
            if self.dist_aware and self.n_dist_param > 0:
                self.solver.set(i, "p", p_stage)
        x_ref_N = interp_full_state_piecewise(t_query + N * dt, t_plan, x_plan, rm)
        x_ref_N = clamp_mpc_state_reference(
            x_ref_N, x_now, self.ref_error_limits, nq=self.nq, nv=self.nv
        )
        yref_terminal = self._state_to_y(x_ref_N)
        self.solver.set(N, "yref", yref_terminal)

        # Initial-state constraint
        self.solver.set(0, "lbx", x_now)
        self.solver.set(0, "ubx", x_now)

        # Warm start
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

        # Reject a diverged/failed solve: never publish NaN and never poison the
        # warm start. Drop the guess (cold-start next time) and return None so the
        # controller keeps the previous _u_hold.
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
        # Shift warm start for next iteration
        self._xs_guess = xs[1:] + [xs[-1].copy()]
        self._us_guess = us[1:] + [us[-1].copy()]

        try:
            from s500_uam_acados_state_tracking_mpc import _acados_cost_term_breakdown

            W_state_diag = np.diag(self._state_cost_weights())
            W_full, _ = self._cost_W_and_W_e()
            R_diag = np.diag(W_full)[-self.nu :]
            self.last_cost_terms = _acados_cost_term_breakdown(
                self.solver,
                N,
                self.nq,
                self.n_arm,
                W_state_diag,
                R_diag,
                self.w_terminal_track,
                yref_running,
                yref_terminal,
            )
        except Exception:
            self.last_cost_terms = {}

        return u_opt, x_next, status

    def _get_stat(self, name: str) -> float:
        try:
            v = self.solver.get_stats(name)
            arr = np.asarray(v).flatten()
            return float(arr[-1]) if arr.size else 0.0
        except Exception:
            return 0.0

    def warmup(self, x0: np.ndarray, t_plan: np.ndarray, x_plan: np.ndarray, iters: int = 5):
        """Prime the solver (codegen lib load + factorization + warm start) before the
        real-time loop, so the first in-loop solve does not stall and cause a control gap.
        """
        x0 = np.asarray(x0, dtype=float).flatten()[: self.nx]
        t_plan = np.asarray(t_plan, dtype=float).flatten()
        x_plan = np.asarray(x_plan, dtype=float)
        t0 = float(t_plan[0]) if t_plan.size else 0.0
        for _ in range(max(1, int(iters))):
            try:
                self.solve_step(x0, t0, t_plan, x_plan)
            except Exception:
                break

    def reset_warm_start(self):
        self._xs_guess = None
        self._us_guess = None
