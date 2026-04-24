from casadi import MX, DM, Function, vertcat, veccat, nlpsol, norm_2
import numpy as np


class SegmentPlanner:
    """
    Time-optimal segment-wise multiple shooting planner.
    Dynamics: double-integrator (position + velocity) with bounded acceleration.
    This keeps the NLP robust and fast while using mass from URDF to scale bounds.
    """

    def __init__(self, quad, track, options=None):
        self.quad = quad
        self.track = track
        self.options = options or {}

        self.wp = DM(track["gates"]).T
        if track.get("end_pos") is not None:
            from casadi import horzcat

            self.wp = horzcat(self.wp, DM(track["end_pos"]))

        self.p_init = DM(track["init_pos"])
        self.v_init = DM(track.get("init_vel", [0.0, 0.0, 0.0]))
        self.v_end = track.get("end_vel", [0.0, 0.0, 0.0])

        self.NX = 6  # p(3), v(3)
        self.NU = 3  # accel command in world frame
        self.NW = self.wp.shape[1]
        self.NPS = int(self.options.get("nodes_per_segment", 20))
        self.vel_guess = float(self.options.get("vel_guess", 3.0))
        self.accel_penalty = float(self.options.get("accel_penalty", 0.01))

        self.solver_type = self.options.get("solver_type", "ipopt")
        self.solver_options = self.options.get(
            "solver_options",
            {
                "ipopt": {
                    "max_iter": 2000,
                    "tol": 1e-6,
                    "acceptable_tol": 1e-4,
                    "acceptable_iter": 20,
                }
            },
        )

        # RK4 dynamics for xdot = f(x,u)
        x = MX.sym("x", self.NX)
        u = MX.sym("u", self.NU)
        p = x[0:3]
        v = x[3:6]
        xdot = vertcat(v, u)
        self.f = Function("f", [x, u], [xdot])

        self.var_slices = {}
        self.x_sol = None

    def _append_var(self, name, sym, guess, var_list, guess_list):
        start = sum(v.numel() for v in var_list)
        var_list.append(sym)
        guess_list.append(guess)
        self.var_slices[name] = (start, start + sym.numel())

    def _rk4_step(self, xk, uk, dt):
        k1 = self.f(xk, uk)
        k2 = self.f(xk + 0.5 * dt * k1, uk)
        k3 = self.f(xk + 0.5 * dt * k2, uk)
        k4 = self.f(xk + dt * k3, uk)
        return xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def setup(self):
        x = []
        xg = []
        g = []
        lb = []
        ub = []
        J = 0
        prev_end = None

        for s in range(self.NW):
            p_from = self.p_init if s == 0 else self.wp[:, s - 1]
            p_to = self.wp[:, s]
            d = p_to - p_from
            dist = float(norm_2(d))
            v_guess = self.vel_guess * d / dist if dist > 1e-8 else DM.zeros(3, 1)
            T_guess = max(dist / max(self.vel_guess, 1e-3), 0.2)

            Ts = MX.sym(f"T_{s}", 1)
            self._append_var(f"T_{s}", Ts, [T_guess], x, xg)
            g += [Ts]
            lb += [0.05]
            ub += [200.0]
            J += Ts

            Xs = []
            Us = []
            for k in range(self.NPS + 1):
                Xk = MX.sym(f"X_{s}_{k}", self.NX)
                alpha = k / self.NPS
                p_guess = (1 - alpha) * p_from + alpha * p_to
                x_guess = vertcat(p_guess, v_guess)
                self._append_var(f"X_{s}_{k}", Xk, x_guess, x, xg)
                Xs.append(Xk)

                # Keep altitude physically valid
                g += [Xk[2]]
                lb += [0.1]
                ub += [200.0]

            for k in range(self.NPS):
                Uk = MX.sym(f"U_{s}_{k}", self.NU)
                self._append_var(f"U_{s}_{k}", Uk, [0.0, 0.0, 0.0], x, xg)
                Us.append(Uk)
                g += [Uk]
                lb += [-self.quad["a_max_xy"], -self.quad["a_max_xy"], -self.quad["a_max_z"]]
                ub += [self.quad["a_max_xy"], self.quad["a_max_xy"], self.quad["a_max_z"]]
                J += self.accel_penalty * Ts / self.NPS * (Uk[0] ** 2 + Uk[1] ** 2 + Uk[2] ** 2)

            for k in range(self.NPS):
                dt = Ts / self.NPS
                xn = self._rk4_step(Xs[k], Us[k], dt)
                g += [Xs[k + 1] - xn]
                lb += [0.0] * self.NX
                ub += [0.0] * self.NX

            if s == 0:
                g += [Xs[0][0:3] - self.p_init]
                lb += [0.0, 0.0, 0.0]
                ub += [0.0, 0.0, 0.0]
                g += [Xs[0][3:6] - self.v_init]
                lb += [0.0, 0.0, 0.0]
                ub += [0.0, 0.0, 0.0]

            # hard pass-through gate/end point
            g += [Xs[-1][0:3] - p_to]
            lb += [0.0, 0.0, 0.0]
            ub += [0.0, 0.0, 0.0]

            if prev_end is not None:
                g += [Xs[0] - prev_end]
                lb += [0.0] * self.NX
                ub += [0.0] * self.NX
            prev_end = Xs[-1]

        if self.v_end is not None:
            g += [prev_end[3:6] - DM(self.v_end)]
            lb += [0.0, 0.0, 0.0]
            ub += [0.0, 0.0, 0.0]

        self.x = vertcat(*x)
        self.xg = veccat(*xg)
        self.g = vertcat(*g)
        self.lb = veccat(*lb)
        self.ub = veccat(*ub)
        self.nlp = {"f": J, "x": self.x, "g": self.g}

    def solve(self):
        solver = nlpsol("solver", self.solver_type, self.nlp, self.solver_options)
        sol = solver(x0=self.xg, lbg=self.lb, ubg=self.ub)
        self.x_sol = sol["x"].full().flatten()
        return self.x_sol

    def get_var(self, name):
        s, e = self.var_slices[name]
        return self.x_sol[s:e]

    def extract_position_velocity_trajectory(self):
        t = []
        p = []
        v = []
        t_now = 0.0
        for s in range(self.NW):
            Ts = float(self.get_var(f"T_{s}")[0])
            dt = Ts / self.NPS
            for k in range(self.NPS + 1):
                if s > 0 and k == 0:
                    continue
                Xk = self.get_var(f"X_{s}_{k}")
                t.append(t_now + k * dt)
                p.append(Xk[0:3])
                v.append(Xk[3:6])
            t_now += Ts
        return np.asarray(t), np.asarray(p), np.asarray(v)