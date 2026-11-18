"""
acados OCP for DynamicTubeMPC2D: augmented state + discrete dynamics matching dtmpc_prototype.rollout.
Requires: casadi, acados_template, built acados.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import casadi as ca
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

    ACADOS_AVAILABLE = True
except ImportError:
    ca = None
    AcadosOcp = None
    AcadosOcpSolver = None
    AcadosModel = None
    ACADOS_AVAILABLE = False


def _embed_cfg_constants(cfg) -> dict:
    oc = np.asarray(cfg.obstacle_centers, dtype=float)
    return {
        "dt": float(cfg.dt),
        "mass": float(cfg.mass),
        "drag_hat": float(cfg.drag_hat),
        "drag_bar": float(cfg.drag_bar),
        "disturbance_bound": float(cfg.disturbance_bound),
        "eta": float(cfg.eta),
        "lambda_s": float(cfg.lambda_s),
        "alpha_min": float(cfg.alpha_min),
        "alpha_max": float(cfg.alpha_max),
        "vel_max": float(cfg.vel_max),
        "u_max": float(cfg.u_max),
        "q_pos": float(cfg.q_pos),
        "q_vel": float(cfg.q_vel),
        "r_u": float(cfg.r_u),
        "r_alpha": float(cfg.r_alpha),
        "r_v": float(cfg.r_v),
        "w_obstacle": float(cfg.w_obstacle),
        "w_terminal": float(cfg.w_terminal),
        "slack_penalty": float(cfg.slack_penalty),
        "obstacle_radius": float(cfg.obstacle_radius),
        "robot_radius": float(cfg.robot_radius),
        "c0x": float(oc[0, 0]),
        "c0y": float(oc[0, 1]),
        "c1x": float(oc[1, 0]),
        "c1y": float(oc[1, 1]),
    }


def export_dtmpc_acados_model(cfg) -> "AcadosModel":
    if not ACADOS_AVAILABLE:
        raise ImportError("casadi / acados_template not available")

    C = _embed_cfg_constants(cfg)
    dt = C["dt"]
    oc = np.asarray(cfg.obstacle_centers, dtype=float)
    margin = float(cfg.obstacle_soft_margin)
    min_cl = float(cfg.acados_obstacle_min_clearance)
    use_h = bool(cfg.acados_use_obstacle_constraints)
    soft_scale = float(cfg.acados_obstacle_soft_scale)

    model = AcadosModel()
    model.name = "dtmpc_mpc"

    x = ca.SX.sym("x", 8)
    u = ca.SX.sym("u", 3)
    p = ca.SX.sym("p", 4)
    model.x = x
    model.u = u
    model.p = p

    px, py, vx, vy = x[0], x[1], x[2], x[3]
    phi, alpha, omega_p, omega_v = x[4], x[5], x[6], x[7]
    u_ax, u_ay, v_alpha = u[0], u[1], u[2]

    eps = 1e-9
    speed = ca.sqrt(vx * vx + vy * vy + eps)
    drag_x = -C["drag_hat"] * speed * vx
    drag_y = -C["drag_hat"] * speed * vy
    acc_x = drag_x + u_ax / C["mass"]
    acc_y = drag_y + u_ay / C["mass"]

    px_n = px + dt * vx
    py_n = py + dt * vy
    vx_n = vx + dt * acc_x
    vy_n = vy + dt * acc_y

    delta_drag = ca.fmax(0.0, C["drag_bar"] - C["drag_hat"]) * speed * speed
    phi_dot = -alpha * phi + delta_drag + C["disturbance_bound"] + C["eta"]
    phi_n = ca.fmax(phi + dt * phi_dot, 1e-6)

    alpha_n = ca.fmin(
        ca.fmax(alpha + dt * v_alpha, C["alpha_min"]),
        C["alpha_max"],
    )

    omega_p_n = ca.fmax(omega_p + dt * omega_v, 0.0)
    omega_v_n = ca.fmax(omega_v + dt * (-C["lambda_s"] * omega_v + phi), 0.0)

    model.disc_dyn_expr = ca.vertcat(px_n, py_n, vx_n, vy_n, phi_n, alpha_n, omega_p_n, omega_v_n)

    pos_err_x = px - p[0]
    pos_err_y = py - p[1]
    vel_err_x = vx - p[2]
    vel_err_y = vy - p[3]
    pos_err = ca.vertcat(pos_err_x, pos_err_y)
    vel_err = ca.vertcat(vel_err_x, vel_err_y)

    h_rows = []
    obs_pen = 0.0
    for j in range(oc.shape[0]):
        cx = float(oc[j, 0])
        cy = float(oc[j, 1])
        safe_r = C["obstacle_radius"] + C["robot_radius"] + omega_p
        dist = ca.sqrt((px - cx) ** 2 + (py - cy) ** 2 + eps)
        clr = dist - safe_r
        if use_h:
            h_rows.append(clr - min_cl)
        else:
            obs_pen += C["w_obstacle"] * (ca.fmax(0.0, margin - clr)) ** 2
    if use_h and h_rows:
        model.con_h_expr = ca.vertcat(*h_rows)
        model.con_h_expr_e = ca.vertcat(*h_rows)

    speed_norm = ca.sqrt(vx * vx + vy * vy + eps)
    vel_slack = speed_norm + omega_v - C["vel_max"]
    vel_pen = C["slack_penalty"] * (ca.fmax(0.0, vel_slack)) ** 2

    u_mag = ca.sqrt(u_ax * u_ax + u_ay * u_ay + eps)
    fb_bound = C["lambda_s"] * omega_v + alpha * phi
    u_slack = u_mag + fb_bound - C["u_max"]
    u_pen = C["slack_penalty"] * (ca.fmax(0.0, u_slack)) ** 2

    stage_core = (
        C["q_pos"] * ca.dot(pos_err, pos_err)
        + C["q_vel"] * ca.dot(vel_err, vel_err)
        + C["r_u"] * (u_ax * u_ax + u_ay * u_ay)
        + C["r_alpha"] * (alpha - C["alpha_min"]) ** 2
        + C["r_v"] * v_alpha * v_alpha
    )
    model.cost_expr_ext_cost = dt * stage_core + obs_pen + vel_pen + u_pen

    px_e, py_e, vx_e, vy_e = x[0], x[1], x[2], x[3]
    omega_p_e, omega_v_e = x[6], x[7]
    pe_x = px_e - p[0]
    pe_y = py_e - p[1]
    ve_x = vx_e - p[2]
    ve_y = vy_e - p[3]
    obs_pen_e = 0.0
    for j in range(oc.shape[0]):
        cx = float(oc[j, 0])
        cy = float(oc[j, 1])
        safe_r_e = C["obstacle_radius"] + C["robot_radius"] + omega_p_e
        clr_e = ca.sqrt((px_e - cx) ** 2 + (py_e - cy) ** 2 + eps) - safe_r_e
        if use_h:
            if soft_scale > 0.0:
                obs_pen_e += soft_scale * C["w_obstacle"] * (ca.fmax(0.0, margin - clr_e)) ** 2
        else:
            obs_pen_e += C["w_obstacle"] * (ca.fmax(0.0, margin - clr_e)) ** 2

    model.cost_expr_ext_cost_e = (
        C["w_terminal"] * (
            pe_x * pe_x + pe_y * pe_y + 0.5 * (ve_x * ve_x + ve_y * ve_y)
        )
        + 10.0 * (omega_p_e ** 2 + omega_v_e ** 2)
        + obs_pen_e
    )

    return model


def formulate_ocp(cfg, code_export_dir: Path) -> "AcadosOcp":
    if not ACADOS_AVAILABLE:
        raise ImportError("acados / casadi not available")

    ocp = AcadosOcp()
    ocp.model = export_dtmpc_acados_model(cfg)

    N = int(cfg.horizon)
    Tf = N * float(cfg.dt)
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.integrator_type = "DISCRETE"

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    nu = 3
    ocp.constraints.lbu = np.array([-cfg.u_max, -cfg.u_max, -cfg.v_max])
    ocp.constraints.ubu = np.array([cfg.u_max, cfg.u_max, cfg.v_max])
    ocp.constraints.idxbu = np.arange(nu)

    nx = 8
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0 = -1e3 * np.ones(nx)
    ocp.constraints.ubx_0 = 1e3 * np.ones(nx)

    ocp.solver_options.qp_solver = str(cfg.acados_qp_solver)
    ha = str(cfg.acados_hessian_approx).upper()
    ocp.solver_options.hessian_approx = ha
    if ha == "EXACT":
        ocp.solver_options.ext_cost_num_hess = 0
    else:
        ocp.solver_options.ext_cost_num_hess = int(cfg.acados_ext_cost_num_hess)

    if cfg.acados_use_obstacle_constraints:
        n_h = int(cfg.obstacle_centers.shape[0])
        big = 1e9
        ocp.constraints.lh = np.zeros(n_h)
        ocp.constraints.uh = big * np.ones(n_h)
        ocp.constraints.lh_e = np.zeros(n_h)
        ocp.constraints.uh_e = big * np.ones(n_h)
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = int(cfg.acados_nlp_iter)
    ocp.solver_options.tol = float(cfg.acados_tol)
    ocp.solver_options.print_level = int(cfg.acados_print_level)
    ocp.solver_options.reg_epsilon = float(cfg.acados_reg_epsilon)
    ocp.solver_options.hpipm_mode = str(cfg.acados_hpipm_mode)

    ocp.parameter_values = np.zeros(4)

    code_export_dir = Path(code_export_dir)
    code_export_dir.mkdir(parents=True, exist_ok=True)
    ocp.code_gen_opts.code_export_directory = str(code_export_dir)
    ocp.code_gen_opts.json_file = str(code_export_dir / "dtmpc_mpc_ocp.json")

    return ocp


class DTMPCAcadosSolver:
    """Lazy AcadosOcpSolver; rebuild if cfg signature changes."""

    def __init__(self, cfg, code_export_dir: Path | None = None):
        self.cfg = cfg
        self._code_dir = (
            Path(code_export_dir)
            if code_export_dir is not None
            else Path(__file__).resolve().parent / "c_generated_dtmpc"
        )
        self._ocp = formulate_ocp(cfg, self._code_dir)
        self._solver: AcadosOcpSolver | None = None
        self._signature = self._cfg_signature(cfg)

    @staticmethod
    def _cfg_signature(cfg) -> tuple:
        return (
            cfg.horizon,
            cfg.dt,
            float(cfg.u_max),
            float(cfg.v_max),
            tuple(np.asarray(cfg.obstacle_centers).flatten()),
            float(cfg.obstacle_radius),
            int(cfg.acados_nlp_iter),
            float(cfg.acados_tol),
            float(cfg.acados_reg_epsilon),
            str(cfg.acados_hpipm_mode),
            str(cfg.acados_qp_solver),
            str(cfg.acados_hessian_approx),
            int(cfg.acados_ext_cost_num_hess),
            bool(cfg.acados_use_obstacle_constraints),
            float(cfg.acados_obstacle_min_clearance),
            float(cfg.obstacle_soft_margin),
            float(cfg.acados_obstacle_soft_scale),
        )

    def _ensure_solver(self):
        if self._solver is not None:
            return
        json_path = self._ocp.code_gen_opts.json_file
        self._solver = AcadosOcpSolver(self._ocp, json_file=json_path)

    def solve(self, controller, x_aug: np.ndarray, x_goal: np.ndarray, z_guess: np.ndarray | None):
        """
        controller: DynamicTubeMPC2D (for rollout warm-start).
        x_aug: (8,) initial augmented state.
        """
        self._ensure_solver()
        solver = self._solver
        N = self.cfg.horizon
        p_val = np.asarray(x_goal, dtype=float).flatten()[:4]

        for i in range(N + 1):
            solver.set(i, "p", p_val)

        solver.set(0, "lbx", x_aug)
        solver.set(0, "ubx", x_aug)

        if z_guess is not None:
            U = z_guess[: N * 2].reshape(N, 2)
            V_a = z_guess[N * 2 :]
            x0 = x_aug[:4]
            phi0 = float(x_aug[4])
            alpha0 = float(x_aug[5])
            omega_p0 = float(x_aug[6])
            omega_v0 = float(x_aug[7])
            X, PHI, ALPHA, OMEGA_P, OMEGA_V = controller.rollout(
                x0, U, V_a, phi0, alpha0, omega_p0, omega_v0
            )
            for i in range(N):
                solver.set(i, "u", np.array([U[i, 0], U[i, 1], V_a[i]]))
            for i in range(N + 1):
                xi = np.concatenate(
                    [X[i], [PHI[i], ALPHA[i], OMEGA_P[i], OMEGA_V[i]]]
                )
                solver.set(i, "x", xi)
        else:
            U = np.zeros((N, 2))
            U[:, 0] = 0.35
            U[:, 1] = 0.15
            V_a = np.zeros(N)
            x0 = x_aug[:4]
            phi0 = float(x_aug[4])
            alpha0 = float(x_aug[5])
            omega_p0 = float(x_aug[6])
            omega_v0 = float(x_aug[7])
            X, PHI, ALPHA, OMEGA_P, OMEGA_V = controller.rollout(
                x0, U, V_a, phi0, alpha0, omega_p0, omega_v0
            )
            for i in range(N):
                solver.set(i, "u", np.array([U[i, 0], U[i, 1], V_a[i]]))
            for i in range(N + 1):
                xi = np.concatenate(
                    [X[i], [PHI[i], ALPHA[i], OMEGA_P[i], OMEGA_V[i]]]
                )
                solver.set(i, "x", xi)

        status = solver.solve()
        msg = f"acados status {status}"
        U_out = np.zeros((N, 2))
        V_out = np.zeros(N)
        for i in range(N):
            ui = solver.get(i, "u")
            U_out[i, 0] = float(ui[0])
            U_out[i, 1] = float(ui[1])
            V_out[i] = float(ui[2])

        return U_out, V_out, int(status), msg


def acados_available() -> bool:
    return ACADOS_AVAILABLE
