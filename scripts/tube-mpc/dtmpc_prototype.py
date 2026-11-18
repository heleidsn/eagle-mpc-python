import sys
"""
DTMPC 原型：默认用 scipy.optimize (L-BFGS-B) 解有限维 NLP。
设置 DTMPCConfig.mpc_backend = \"acados\" 时使用 CasADi 离散模型 + acados SQP（需安装 acados_template 与编译好的 acados）。
"""
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


@dataclass
class DTMPCConfig:
    dt: float = 0.1
    horizon: int = 20
    mass: float = 1.0
    drag_hat: float = 0.10
    drag_bar: float = 0.20
    disturbance_bound: float = 0.10
    eta: float = 0.05
    lambda_s: float = 2.0
    alpha_min: float = 0.5
    alpha_max: float = 4.0
    v_max: float = 2.0
    u_max: float = 4.0
    vel_max: float = 3.0
    q_pos: float = 30.0
    q_vel: float = 3.0
    r_u: float = 0.2
    r_alpha: float = 0.1
    r_v: float = 0.05
    w_obstacle: float = 800.0
    # 与 acados 软障碍项一致：w * max(0, margin - clr)^2
    obstacle_soft_margin: float = 0.15
    w_terminal: float = 200.0
    # 两个圆形障碍，沿 y 轴分列在 x≈1.45，中间留缝供轨迹穿过（见 run_closed_loop_demo 示意图）
    obstacle_centers: np.ndarray = field(
        default_factory=lambda: np.array([[1.45, 0.62], [1.45, -0.62]])
    )
    obstacle_radius: float = 0.52
    robot_radius: float = 0.08
    phi0_pos: float = 0.02
    phi0_vel: float = 0.02
    alpha0: float = 1.0
    slack_penalty: float = 1e5
    # Total simulated time = receding_steps * dt; must be enough to reach x_goal after detour.
    receding_steps: int = 65
    # 仅盒约束时 L-BFGS-B 通常明显快于 SLSQP；若收敛差可改回 "SLSQP" 或调大 mpc_maxiter
    mpc_solver: str = "L-BFGS-B"
    mpc_maxiter: int = 200
    # L-BFGS-B 会限制 f/g 求值次数，维数高时需足够大，否则会早停并报 “F,G EVALUATIONS EXCEEDS LIMIT”
    mpc_maxfun: int = 30000
    mpc_ftol: float = 1e-4
    # mpc_backend: "scipy" = scipy.optimize.minimize（默认）； "acados" = CasADi+acados SQP
    mpc_backend: str = "scipy"
    # acados：对名义位置加强制避障（含 tube omega_p），避免仅靠软惩罚时 SQP 收敛差导致穿障
    acados_use_obstacle_constraints: bool = True
    acados_obstacle_min_clearance: float = 0.0
    # 与硬约束并存时保留一小部分软惩罚，便于 SQP 获得平滑梯度、改善到达目标
    acados_obstacle_soft_scale: float = 0.25
    acados_nlp_iter: int = 400
    acados_tol: float = 1e-3
    acados_print_level: int = 0
    acados_reg_epsilon: float = 1e-1
    acados_hpipm_mode: str = "ROBUST"
    acados_qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    # 含非线性不等式约束时 GAUSS_NEWTON + 数值 Hessian 往往比 EXACT 更稳
    acados_hessian_approx: str = "GAUSS_NEWTON"
    acados_ext_cost_num_hess: int = 1
    # acados 常返回 status 2（达到最大 SQP 迭代）但仍可用当前迭代解；与 scipy 的 best-effort 对齐
    acados_count_max_iter_as_success: bool = True
    # None 时使用 scripts/tube-mpc/c_generated_dtmpc（生成 C 代码与 JSON）
    acados_code_export_dir: str | None = None


class DynamicTubeMPC2D:
    """
    A compact DTMPC prototype for a 2D double-integrator-like vehicle with
    quadratic drag. This is a simplified educational implementation inspired by
    the paper "Dynamic Tube MPC for Nonlinear Systems".

    State x = [px, py, vx, vy]
    Control u = [ax, ay]

    Tube dynamics:
        phi_dot = -alpha * phi + Delta(x*) + D + eta
        alpha_dot = v_alpha

    Error-bound dynamics (conservative proxy for Theorem 1 tube evolution):
        omega_p_dot = omega_v
        omega_v_dot = -lambda_s * omega_v + phi

    We optimize the nominal trajectory, control bandwidth alpha, and its rate v.
    Tightened constraints are enforced approximately via nonlinear penalties.
    """

    def __init__(self, cfg: DTMPCConfig):
        self.cfg = cfg
        self.nx = 4
        self.nu = 2
        self._acados_wrapper = None
        self._acados_sig = None

    def nominal_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        px, py, vx, vy = x
        v = np.array([vx, vy])
        speed = np.linalg.norm(v)
        drag = -self.cfg.drag_hat * speed * v
        acc = drag + u / self.cfg.mass
        return np.array([vx, vy, acc[0], acc[1]])

    def uncertainty_bound(self, x: np.ndarray) -> float:
        """
        State-dependent uncertainty bound. Following the paper's drag example,
        uncertainty scales with speed^2. We use the mismatch between the upper
        drag coefficient and the nominal drag estimate plus an additive bound.
        """
        v = x[2:4]
        speed = np.linalg.norm(v)
        delta_drag = max(self.cfg.drag_bar - self.cfg.drag_hat, 0.0) * speed * speed
        return delta_drag

    def step_nominal(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x + self.cfg.dt * self.nominal_dynamics(x, u)

    def step_phi(self, phi: float, alpha: float, x_nom: np.ndarray) -> float:
        phi_dot = -alpha * phi + self.uncertainty_bound(x_nom) + self.cfg.disturbance_bound + self.cfg.eta
        return max(phi + self.cfg.dt * phi_dot, 1e-6)

    def step_alpha(self, alpha: float, v_alpha: float) -> float:
        alpha_next = alpha + self.cfg.dt * v_alpha
        return float(np.clip(alpha_next, self.cfg.alpha_min, self.cfg.alpha_max))

    def step_omega(self, omega_p: float, omega_v: float, phi: float) -> tuple[float, float]:
        omega_p_dot = omega_v
        omega_v_dot = -self.cfg.lambda_s * omega_v + phi
        omega_p_next = max(omega_p + self.cfg.dt * omega_p_dot, 0.0)
        omega_v_next = max(omega_v + self.cfg.dt * omega_v_dot, 0.0)
        return omega_p_next, omega_v_next

    def rollout(self, x0: np.ndarray, U: np.ndarray, V_alpha: np.ndarray,
                phi0: float, alpha0: float, omega_p0: float, omega_v0: float):
        N = U.shape[0]
        X = np.zeros((N + 1, self.nx))
        PHI = np.zeros(N + 1)
        ALPHA = np.zeros(N + 1)
        OMEGA_P = np.zeros(N + 1)
        OMEGA_V = np.zeros(N + 1)

        X[0] = x0
        PHI[0] = phi0
        ALPHA[0] = alpha0
        OMEGA_P[0] = omega_p0
        OMEGA_V[0] = omega_v0

        for k in range(N):
            X[k + 1] = self.step_nominal(X[k], U[k])
            PHI[k + 1] = self.step_phi(PHI[k], ALPHA[k], X[k])
            ALPHA[k + 1] = self.step_alpha(ALPHA[k], V_alpha[k])
            OMEGA_P[k + 1], OMEGA_V[k + 1] = self.step_omega(OMEGA_P[k], OMEGA_V[k], PHI[k])

        return X, PHI, ALPHA, OMEGA_P, OMEGA_V

    def obstacle_clearance(self, p: np.ndarray, omega_p: float, obstacle_index: int) -> float:
        c = self.cfg.obstacle_centers[obstacle_index]
        safe_r = self.cfg.obstacle_radius + self.cfg.robot_radius + omega_p
        return float(np.linalg.norm(p - c) - safe_r)

    def obstacle_penalty(self, p: np.ndarray, omega_p: float) -> float:
        """平滑软惩罚，与 dtmpc_acados 中 CasADi 形式一致：w * max(0, margin - clr)^2。"""
        pen = 0.0
        m = self.cfg.obstacle_soft_margin
        for i in range(self.cfg.obstacle_centers.shape[0]):
            clr = self.obstacle_clearance(p, omega_p, i)
            d = max(0.0, m - clr)
            pen += self.cfg.w_obstacle * d * d
        return pen

    def pack(self, U: np.ndarray, V_alpha: np.ndarray) -> np.ndarray:
        return np.hstack([U.reshape(-1), V_alpha.reshape(-1)])

    def unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = self.cfg.horizon
        U = z[:N * self.nu].reshape(N, self.nu)
        V_alpha = z[N * self.nu:].reshape(N, 1)
        return U, V_alpha[:, 0]

    def objective(self, z: np.ndarray, x0: np.ndarray, x_goal: np.ndarray,
                  phi0: float, alpha0: float, omega_p0: float, omega_v0: float) -> float:
        U, V_alpha = self.unpack(z)
        X, PHI, ALPHA, OMEGA_P, OMEGA_V = self.rollout(x0, U, V_alpha, phi0, alpha0, omega_p0, omega_v0)

        J = 0.0
        for k in range(self.cfg.horizon):
            pos_err = X[k, 0:2] - x_goal[0:2]
            vel_err = X[k, 2:4] - x_goal[2:4]
            J += self.cfg.dt * (
                self.cfg.q_pos * float(pos_err @ pos_err)
                + self.cfg.q_vel * float(vel_err @ vel_err)
                + self.cfg.r_u * float(U[k] @ U[k])
                + self.cfg.r_alpha * (ALPHA[k] - self.cfg.alpha_min) ** 2
                + self.cfg.r_v * V_alpha[k] ** 2
            )

            # soft obstacle avoidance (multiple disks)
            J += self.obstacle_penalty(X[k, 0:2], OMEGA_P[k])

            # tightened velocity bound
            speed = np.linalg.norm(X[k, 2:4])
            vel_slack = speed + OMEGA_V[k] - self.cfg.vel_max
            if vel_slack > 0:
                J += self.cfg.slack_penalty * vel_slack ** 2

            # approximate tightened actuator bound
            fb_bound = self.cfg.lambda_s * OMEGA_V[k] + ALPHA[k] * PHI[k]
            u_mag = np.linalg.norm(U[k])
            u_slack = u_mag + fb_bound - self.cfg.u_max
            if u_slack > 0:
                J += self.cfg.slack_penalty * u_slack ** 2

        J += self.obstacle_penalty(X[-1, 0:2], OMEGA_P[-1])

        pos_terminal = X[-1, 0:2] - x_goal[0:2]
        vel_terminal = X[-1, 2:4] - x_goal[2:4]
        J += self.cfg.w_terminal * (float(pos_terminal @ pos_terminal) + 0.5 * float(vel_terminal @ vel_terminal))
        J += 10.0 * (OMEGA_P[-1] ** 2 + OMEGA_V[-1] ** 2)
        return J

    def solve(self, x0: np.ndarray, x_goal: np.ndarray,
              phi0: float, alpha0: float, omega_p0: float, omega_v0: float,
              z_guess: np.ndarray | None = None):
        N = self.cfg.horizon
        backend = (self.cfg.mpc_backend or "scipy").lower().strip()

        if backend == "acados":
            from dtmpc_acados import DTMPCAcadosSolver, acados_available

            if not acados_available():
                raise ImportError(
                    "mpc_backend='acados' 需要安装 casadi 与 acados_template，并正确编译 acados。"
                )
            x_aug = np.concatenate(
                [np.asarray(x0, dtype=float).flatten()[:4],
                 [phi0, alpha0, omega_p0, omega_v0]]
            )
            export_dir = self.cfg.acados_code_export_dir
            path_dir = Path(export_dir) if export_dir else (_SCRIPT_DIR / "c_generated_dtmpc")
            sig = DTMPCAcadosSolver._cfg_signature(self.cfg)
            if self._acados_wrapper is None or self._acados_sig != sig:
                self._acados_wrapper = DTMPCAcadosSolver(self.cfg, path_dir)
                self._acados_sig = sig
            U, V_alpha, status, msg = self._acados_wrapper.solve(
                self, x_aug, x_goal, z_guess
            )
            X, PHI, ALPHA, OMEGA_P, OMEGA_V = self.rollout(
                x0[:4], U, V_alpha, phi0, alpha0, omega_p0, omega_v0
            )
            z_opt = self.pack(U, np.asarray(V_alpha).reshape(N, 1))
            cost = float(
                self.objective(z_opt, x0, x_goal, phi0, alpha0, omega_p0, omega_v0)
            )
            ok = status == 0 or (
                self.cfg.acados_count_max_iter_as_success and status == 2
            )
            return {
                "success": ok,
                "message": msg,
                "acados_status": status,
                "cost": cost,
                "U": U,
                "V_alpha": V_alpha,
                "X": X,
                "PHI": PHI,
                "ALPHA": ALPHA,
                "OMEGA_P": OMEGA_P,
                "OMEGA_V": OMEGA_V,
                "z_opt": z_opt,
            }

        if z_guess is None:
            U0 = np.zeros((N, self.nu))
            # 前向 + 轻微 y 向，便于从两障碍中间穿过
            U0[:, 0] = 0.35
            U0[:, 1] = 0.15
            V0 = np.zeros((N, 1))
            z_guess = self.pack(U0, V0)

        bounds = []
        for _ in range(N):
            bounds.extend([(-self.cfg.u_max, self.cfg.u_max), (-self.cfg.u_max, self.cfg.u_max)])
        for _ in range(N):
            bounds.append((-self.cfg.v_max, self.cfg.v_max))

        def fun(z):
            return self.objective(z, x0, x_goal, phi0, alpha0, omega_p0, omega_v0)

        opt = {
            "maxiter": self.cfg.mpc_maxiter,
            "ftol": self.cfg.mpc_ftol,
            "disp": False,
        }
        if self.cfg.mpc_solver.upper() in ("L-BFGS-B", "TNC"):
            opt["maxfun"] = self.cfg.mpc_maxfun

        result = minimize(
            fun=fun,
            x0=z_guess,
            method=self.cfg.mpc_solver,
            bounds=bounds,
            options=opt,
        )

        U, V_alpha = self.unpack(result.x)
        X, PHI, ALPHA, OMEGA_P, OMEGA_V = self.rollout(x0, U, V_alpha, phi0, alpha0, omega_p0, omega_v0)
        return {
            "success": bool(result.success),
            "message": result.message,
            "cost": float(result.fun),
            "U": U,
            "V_alpha": V_alpha,
            "X": X,
            "PHI": PHI,
            "ALPHA": ALPHA,
            "OMEGA_P": OMEGA_P,
            "OMEGA_V": OMEGA_V,
            "z_opt": result.x,
            "acados_status": None,
        }


def shift_initial_guess(z_opt: np.ndarray, cfg: DTMPCConfig) -> np.ndarray:
    N = cfg.horizon
    nu = 2
    U = z_opt[:N * nu].reshape(N, nu)
    V = z_opt[N * nu:].reshape(N, 1)
    U_shift = np.vstack([U[1:], U[-1:]])
    V_shift = np.vstack([V[1:], V[-1:]])
    return np.hstack([U_shift.reshape(-1), V_shift.reshape(-1)])


def run_closed_loop_demo(save_path: str | None = None, cfg: DTMPCConfig | None = None):
    cfg = cfg if cfg is not None else DTMPCConfig()
    controller = DynamicTubeMPC2D(cfg)

    # 起点在左下，终点在右上；中间需穿过两圆之间的通道（约在 x≈1.45, y≈0 附近）
    x = np.array([0.0, -0.75, 0.0, 0.0])
    x_goal = np.array([3.0, 0.75, 0.0, 0.0])

    phi = cfg.phi0_pos
    alpha = cfg.alpha0
    omega_p = cfg.phi0_pos
    omega_v = cfg.phi0_vel

    xs = [x.copy()]
    phis = [phi]
    alphas = [alpha]
    op_list = [omega_p]
    ov_list = [omega_v]
    z_guess = None

    for t in range(cfg.receding_steps):
        sol = controller.solve(x, x_goal, phi, alpha, omega_p, omega_v, z_guess=z_guess)
        if not sol["success"]:
            print(f"Warning: solver reported '{sol['message']}' at step {t}, using best effort solution.")

        u_ff = sol["U"][0]
        # simplified ancillary feedback is omitted in closed-loop simulation because
        # we are illustrating the nominal/tube optimizer itself. Disturbance is applied.
        u = u_ff.copy()
        d = np.array([0.08 * np.sin(0.25 * t), -0.06 * np.cos(0.2 * t)])
        xdot = controller.nominal_dynamics(x, u) + np.array([0.0, 0.0, d[0], d[1]])
        x = x + cfg.dt * xdot

        phi = controller.step_phi(phi, alpha, x)
        alpha = controller.step_alpha(alpha, sol["V_alpha"][0])
        omega_p, omega_v = controller.step_omega(omega_p, omega_v, phi)

        xs.append(x.copy())
        phis.append(phi)
        alphas.append(alpha)
        op_list.append(omega_p)
        ov_list.append(omega_v)

        z_guess = shift_initial_guess(sol["z_opt"], cfg)

    xs = np.array(xs)
    phis = np.array(phis)
    alphas = np.array(alphas)
    op_list = np.array(op_list)
    ov_list = np.array(ov_list)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(xs[:, 0], xs[:, 1], '-o', ms=3, label='closed-loop trajectory')
    ax.scatter([x_goal[0]], [x_goal[1]], marker='*', s=150, label='goal')
    for i, c in enumerate(cfg.obstacle_centers):
        lab = 'obstacles' if i == 0 else None
        circ = plt.Circle(c, cfg.obstacle_radius, color='gray', alpha=0.4, label=lab)
        ax.add_patch(circ)
    for i in range(0, len(xs), max(1, len(xs)//10)):
        tube = plt.Circle(xs[i, 0:2], op_list[i], fill=False, color='black', alpha=0.25)
        ax.add_patch(tube)
    ax.set_title('Trajectory with dynamic tube radius')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc='best')

    ax = axes[1]
    ax.plot(phis, label='phi')
    ax.plot(op_list, label='omega_pos')
    ax.plot(ov_list, label='omega_vel')
    ax.set_title('Tube states')
    ax.set_xlabel('step')
    ax.grid(True)
    ax.legend()

    ax = axes[2]
    ax.plot(alphas, label='alpha')
    ax.set_title('Optimized bandwidth state')
    ax.set_xlabel('step')
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches='tight')

    p_final = xs[-1, 0:2]
    dist_goal = float(np.linalg.norm(p_final - x_goal[0:2]))
    return {
        'trajectory': xs,
        'phi': phis,
        'alpha': alphas,
        'omega_pos': op_list,
        'omega_vel': ov_list,
        'figure': fig,
        'dist_to_goal': dist_goal,
        'x_goal': x_goal,
    }


if __name__ == '__main__':
    _default_png = Path(__file__).resolve().parent / 'dtmpc_demo.png'
    _t0 = time.perf_counter()
    out = run_closed_loop_demo(save_path=str(_default_png))
    _elapsed = time.perf_counter() - _t0
    print(f'Saved figure to {_default_png}')
    print(f'Closed-loop wall time: {_elapsed:.2f} s')
    print(
        f'Final |position - goal|: {out["dist_to_goal"]:.4f} m '
        f'(goal {out["x_goal"][0:2]})'
    )
