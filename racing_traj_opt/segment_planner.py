from casadi import MX, DM, Function, vertcat, veccat, nlpsol, norm_2, fabs, sqrt, dot, atan2
import numpy as np
import joblib


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
        self.planning_mode = str(self.options.get("planning_mode", "segmented")).lower()
        if self.planning_mode not in {"segmented", "monolithic"}:
            raise ValueError(f"Unsupported planning_mode: {self.planning_mode}")
        self.dynamics_model = str(self.options.get("dynamics_model", "simple")).lower()
        if self.dynamics_model not in {"simple", "full"}:
            raise ValueError(f"Unsupported dynamics_model: {self.dynamics_model}")

        self.wp = DM(track["gates"]).T
        self.n_gates = len(track.get("gates", []))
        self.has_end_pos = track.get("end_pos") is not None
        if track.get("end_pos") is not None:
            from casadi import horzcat

            self.wp = horzcat(self.wp, DM(track["end_pos"]))

        self.p_init = DM(track["init_pos"])
        self.v_init = DM(track.get("init_vel", [0.0, 0.0, 0.0]))
        self.q_init = DM(track.get("init_att", [1.0, 0.0, 0.0, 0.0]))
        self.w_init = DM(track.get("init_omega", [0.0, 0.0, 0.0]))
        self.v_end = track.get("end_vel", [0.0, 0.0, 0.0])
        self.q_end = track.get("end_att", None)
        self.w_end = track.get("end_omega", None)

        if self.dynamics_model == "full":
            self.NX = 13  # p(3), v(3), q(4, wxyz), w(3)
            self.NU = 4   # four rotor thrusts
        else:
            self.NX = 6   # p(3), v(3)
            self.NU = 3   # accel command in world frame
        self.NW = self.wp.shape[1]
        self.NPS = int(self.options.get("nodes_per_segment", 20))
        self.N_TOTAL = max(int(self.options.get("nodes_total", self.NPS * max(self.NW, 1))), 2)
        self.vel_guess = float(self.options.get("vel_guess", 3.0))
        self.accel_penalty = float(self.options.get("accel_penalty", 0.01))
        self.omega_max = float(self.options.get("omega_max", np.deg2rad(200.0)))
        self.tilt_max = float(self.options.get("tilt_max", np.deg2rad(70.0)))
        fixed_yaw_opt = self.options.get("fixed_yaw", None)
        self.fixed_yaw = None if fixed_yaw_opt is None else float(fixed_yaw_opt)
        self.enable_omega_limit = self.omega_max > 0.0
        self.enable_tilt_limit = self.tilt_max > 0.0
        self.enable_fixed_yaw = self.fixed_yaw is not None
        self.waypoint_tolerance = float(self.options.get("waypoint_tolerance", 0.25))
        self.waypoint_penalty = float(self.options.get("waypoint_penalty", 2000.0))

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

        self.f = self._build_dynamics()

        self.var_slices = {}
        self.x_sol = None
        self.error_model = self.options.get("error_model", None)
        self.error_model_type = str(self.options.get("error_model_type", "nn")).lower()
        self.linear_error_k = float(self.options.get("linear_error_k", 0.05))
        self.error_bound = float(self.options.get("error_bound", 0.10))
        self.error_penalty = float(self.options.get("error_penalty", 5000.0))
        self.use_error_model = bool(self.options.get("use_error_model", False))
        self.nn_params = None
        if self.use_error_model:
            if self.error_model_type == "nn":
                if self.error_model is None:
                    raise ValueError("error_model_type='nn' requires a loaded error_model object")
                self.nn_params = self._load_nn_params(self.error_model)
            elif self.error_model_type == "linear_speed":
                if self.linear_error_k < 0.0:
                    raise ValueError("linear_error_k must be non-negative")
            else:
                raise ValueError(
                    f"Unsupported error_model_type: {self.error_model_type}. "
                    "Use 'nn' or 'linear_speed'."
                )
    def _build_dynamics(self):
        if self.dynamics_model == "simple":
            x = MX.sym("x", self.NX)
            u = MX.sym("u", self.NU)
            v = x[3:6]
            xdot = vertcat(v, u)
            return Function("f_simple", [x, u], [xdot])

        # Full quadrotor rigid-body dynamics.
        x = MX.sym("x", self.NX)
        u = MX.sym("u", self.NU)
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]  # wxyz
        w = x[10:13]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Rotation matrix body->world from quaternion (w,x,y,z)
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qw * qz)
        r02 = 2 * (qx * qz + qw * qy)
        r10 = 2 * (qx * qy + qw * qz)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qw * qx)
        r20 = 2 * (qx * qz - qw * qy)
        r21 = 2 * (qy * qz + qw * qx)
        r22 = 1 - 2 * (qx * qx + qy * qy)

        thrust_total = u[0] + u[1] + u[2] + u[3]
        thrust_world = vertcat(r02 * thrust_total, r12 * thrust_total, r22 * thrust_total)
        gvec = vertcat(0, 0, self.quad["g"])
        m = self.quad["mass"]
        vdot = thrust_world / m - gvec

        # Quaternion kinematics qdot = 0.5 * Omega(w) * q
        wx, wy, wz = w[0], w[1], w[2]
        qdot = 0.5 * vertcat(
            -qx * wx - qy * wy - qz * wz,
            qw * wx + qy * wz - qz * wy,
            qw * wy + qz * wx - qx * wz,
            qw * wz + qx * wy - qy * wx,
        )

        # Rotor geometry and body torques.
        rotor_pos = np.asarray(self.quad.get("rotor_pos", [[0.171, 0.171, 0.0], [-0.171, 0.171, 0.0], [-0.171, -0.171, 0.0], [0.171, -0.171, 0.0]]), dtype=float)
        yaw_dirs = np.asarray(self.quad.get("rotor_yaw_dirs", [1.0, -1.0, 1.0, -1.0]), dtype=float)
        k_yaw = float(self.quad.get("k_yaw", 0.01))
        tau_x = 0
        tau_y = 0
        tau_z = 0
        for i in range(4):
            rx, ry = rotor_pos[i, 0], rotor_pos[i, 1]
            fi = u[i]
            tau_x += ry * fi
            tau_y += -rx * fi
            tau_z += yaw_dirs[i] * k_yaw * fi
        tau = vertcat(tau_x, tau_y, tau_z)

        I = np.asarray(self.quad["inertia"], dtype=float)
        Ixx, Iyy, Izz = float(I[0, 0]), float(I[1, 1]), float(I[2, 2])
        Iw = vertcat(Ixx * wx, Iyy * wy, Izz * wz)
        wxIw = vertcat(
            wy * Iw[2] - wz * Iw[1],
            wz * Iw[0] - wx * Iw[2],
            wx * Iw[1] - wy * Iw[0],
        )
        wdot = vertcat(
            (tau[0] - wxIw[0]) / Ixx,
            (tau[1] - wxIw[1]) / Iyy,
            (tau[2] - wxIw[2]) / Izz,
        )

        xdot = vertcat(v, vdot, qdot, wdot)
        return Function("f_full", [x, u], [xdot])


    def _append_var(self, name, sym, guess, var_list, guess_list):
        start = sum(v.numel() for v in var_list)
        var_list.append(sym)
        guess_list.append(guess)
        self.var_slices[name] = (start, start + sym.numel())

    def _load_nn_params(self, model_obj):
        scaler = model_obj["scaler"]
        mlp = model_obj["mlp"]
        window_size = int(model_obj.get("window_size", 1))
        if window_size != 1:
            raise ValueError(f"Only window_size=1 supported in optimizer constraint, got {window_size}")
        default_features = [
            "ep_x", "ep_y", "ep_z",
            "ev_x", "ev_y", "ev_z",
            "a_ref_x", "a_ref_y", "a_ref_z",
            "j_ref_x", "j_ref_y", "j_ref_z",
            "yaw_ref", "yaw_rate_ref",
            "v_world_x", "v_world_y", "v_world_z",
            "omega_x", "omega_y", "omega_z",
            "body_z_world_x", "body_z_world_y", "body_z_world_z",
            "u_thrust", "thrust_margin",
        ]
        feature_names = list(model_obj.get("feature_names_per_frame") or default_features)
        target_names = list(
            model_obj.get("target_names")
            or ["ep_x", "ep_y", "ep_z", "ev_x", "ev_y", "ev_z"]
        )
        scale = np.asarray(scaler.scale_, dtype=float).reshape(-1)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        if len(feature_names) != int(scale.size):
            raise ValueError(
                "Error model feature count mismatch: "
                f"{len(feature_names)} names vs scaler dimension {int(scale.size)}"
            )
        return {
            "mean": np.asarray(scaler.mean_, dtype=float).reshape(-1),
            "scale": scale,
            "coefs": [np.asarray(w, dtype=float) for w in mlp.coefs_],
            "biases": [np.asarray(b, dtype=float).reshape(-1) for b in mlp.intercepts_],
            "feature_names": feature_names,
            "target_names": target_names,
        }

    def _nn_forward_ep_pred(self, x_feat):
        # x_feat shape follows the loaded model's feature_names_per_frame.
        p = self.nn_params
        z = (x_feat - DM(p["mean"]).reshape((x_feat.shape[0], 1))) / DM(p["scale"]).reshape((x_feat.shape[0], 1))
        for i, (w, b) in enumerate(zip(p["coefs"], p["biases"])):
            z = DM(w).T @ z + DM(b).reshape((b.shape[0], 1))
            if i < len(p["coefs"]) - 1:
                # Smooth ReLU to keep NLP derivatives well-behaved.
                z = 0.5 * (z + sqrt(z * z + 1e-8))
        # Prefer named outputs when available; legacy models use [ep_x, ep_y, ep_z, ev_x, ev_y, ev_z].
        idx = []
        names = p.get("target_names") or []
        for name in ("ep_x", "ep_y", "ep_z"):
            idx.append(names.index(name) if name in names else len(idx))
        return vertcat(z[idx[0]], z[idx[1]], z[idx[2]])

    def _rk4_step(self, xk, uk, dt):
        k1 = self.f(xk, uk)
        k2 = self.f(xk + 0.5 * dt * k1, uk)
        k3 = self.f(xk + 0.5 * dt * k2, uk)
        k4 = self.f(xk + dt * k3, uk)
        return xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _build_error_model_features(self, p_to, x_end, u_end):
        # Build the feature vector requested by the loaded model. Legacy models use
        # the original 25-D layout; model_3 uses plan-only [v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref].
        ep = p_to - x_end[0:3]
        v_world = x_end[3:6]
        v_ref = v_world
        ev = -v_world  # gate terminal reference velocity is near zero

        if self.dynamics_model == "simple":
            a_ref = u_end
            omega = DM.zeros(3, 1)
            body_z_world = DM([0.0, 0.0, 1.0])
            a_norm = sqrt(dot(a_ref, a_ref) + 1e-8)
            u_thrust = DM([0.5]) + 0.2 * a_norm / self.quad["g"]
            thrust_margin = DM([1.0]) - u_thrust
            yaw_ref = DM([0.0])
            yaw_rate_ref = DM([0.0])
        else:
            qw, qx, qy, qz = x_end[6], x_end[7], x_end[8], x_end[9]
            r02 = 2 * (qx * qz + qw * qy)
            r12 = 2 * (qy * qz - qw * qx)
            r22 = 1 - 2 * (qx * qx + qy * qy)
            body_z_world = vertcat(r02, r12, r22)
            thrust_total = u_end[0] + u_end[1] + u_end[2] + u_end[3]
            a_ref = body_z_world * thrust_total / self.quad["mass"] - DM([0.0, 0.0, self.quad["g"]])
            omega = x_end[10:13]
            denom = max(float(self.quad.get("T_max", 1.0)) * 4.0, 1e-6)
            u_thrust = DM([1.0 / denom]) * thrust_total
            thrust_margin = DM([1.0]) - u_thrust
            siny = 2 * (qw * qz + qx * qy)
            cosy = 1 - 2 * (qy * qy + qz * qz)
            yaw_ref = vertcat(atan2(siny, cosy))
            # Approximate ZYX yaw rate from body rates. This keeps model_3's plan-only
            # yaw_rate_ref feature consistent with the quaternion state used by the full dynamics.
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll_den = sqrt(sinr_cosp * sinr_cosp + cosr_cosp * cosr_cosp + 1e-8)
            sin_roll = sinr_cosp / roll_den
            cos_roll = cosr_cosp / roll_den
            sin_pitch = 2 * (qw * qy - qz * qx)
            cos_pitch_sq = 1 - sin_pitch * sin_pitch
            # Smooth max(cos_pitch_sq, 0) avoids NaNs during intermediate IPOPT
            # iterations where the quaternion may be slightly off the unit sphere.
            cos_pitch = sqrt(0.5 * (cos_pitch_sq + sqrt(cos_pitch_sq * cos_pitch_sq + 1e-8)) + 1e-8)
            yaw_rate_ref = vertcat((sin_roll * omega[1] + cos_roll * omega[2]) / cos_pitch)

        j_ref = DM.zeros(3, 1)
        feature_map = {
            "ep_x": ep[0], "ep_y": ep[1], "ep_z": ep[2],
            "ev_x": ev[0], "ev_y": ev[1], "ev_z": ev[2],
            "v_ref_x": v_ref[0], "v_ref_y": v_ref[1], "v_ref_z": v_ref[2],
            "a_ref_x": a_ref[0], "a_ref_y": a_ref[1], "a_ref_z": a_ref[2],
            "j_ref_x": j_ref[0], "j_ref_y": j_ref[1], "j_ref_z": j_ref[2],
            "yaw_ref": yaw_ref[0], "yaw_rate_ref": yaw_rate_ref[0],
            "v_world_x": v_world[0], "v_world_y": v_world[1], "v_world_z": v_world[2],
            "omega_x": omega[0], "omega_y": omega[1], "omega_z": omega[2],
            "body_z_world_x": body_z_world[0],
            "body_z_world_y": body_z_world[1],
            "body_z_world_z": body_z_world[2],
            "u_thrust": u_thrust[0],
            "thrust_margin": thrust_margin[0],
        }
        names = (self.nn_params or {}).get("feature_names") or []
        missing = [name for name in names if name not in feature_map]
        if missing:
            raise ValueError(f"Unsupported error model feature(s): {missing}")
        return vertcat(*[feature_map[name] for name in names])

    def setup(self):
        if self.planning_mode == "monolithic":
            self._setup_monolithic()
            return

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
                if self.dynamics_model == "full":
                    x_guess = vertcat(p_guess, v_guess, self.q_init, self.w_init)
                else:
                    x_guess = vertcat(p_guess, v_guess)
                self._append_var(f"X_{s}_{k}", Xk, x_guess, x, xg)
                Xs.append(Xk)

                # Keep altitude physically valid
                g += [Xk[2]]
                lb += [0.1]
                ub += [200.0]
                if self.dynamics_model == "full":
                    # quaternion unit norm
                    g += [dot(Xk[6:10], Xk[6:10])]
                    lb += [1.0]
                    ub += [1.0]
                    if self.enable_tilt_limit:
                        # Tilt bound: angle(body-z, world-z) <= tilt_max
                        # cos(tilt) = r22 = 1 - 2*(qx^2 + qy^2)
                        qx = Xk[7]
                        qy = Xk[8]
                        r22 = 1 - 2 * (qx * qx + qy * qy)
                        g += [r22]
                        lb += [np.cos(self.tilt_max)]
                        ub += [1.0]
                    if self.enable_omega_limit:
                        # body angular-rate bounds [rad/s]
                        g += [Xk[10:13]]
                        lb += [-self.omega_max, -self.omega_max, -self.omega_max]
                        ub += [self.omega_max, self.omega_max, self.omega_max]
                    if self.enable_fixed_yaw:
                        qw = Xk[6]
                        qx = Xk[7]
                        qy = Xk[8]
                        qz = Xk[9]
                        siny = 2 * (qw * qz + qx * qy)
                        cosy = 1 - 2 * (qy * qy + qz * qz)
                        g += [siny, cosy]
                        lb += [float(np.sin(self.fixed_yaw)), float(np.cos(self.fixed_yaw))]
                        ub += [float(np.sin(self.fixed_yaw)), float(np.cos(self.fixed_yaw))]

            for k in range(self.NPS):
                Uk = MX.sym(f"U_{s}_{k}", self.NU)
                if self.dynamics_model == "full":
                    u_guess = [self.quad["mass"] * self.quad["g"] / 4.0] * 4
                else:
                    u_guess = [0.0, 0.0, 0.0]
                self._append_var(f"U_{s}_{k}", Uk, u_guess, x, xg)
                Us.append(Uk)
                g += [Uk]
                if self.dynamics_model == "full":
                    lb += [self.quad["T_min"]] * 4
                    ub += [self.quad["T_max"]] * 4
                    J += self.accel_penalty * Ts / self.NPS * dot(Uk, Uk)
                else:
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
                if self.dynamics_model == "full":
                    g += [Xs[0][6:10] - self.q_init]
                    lb += [0.0, 0.0, 0.0, 0.0]
                    ub += [0.0, 0.0, 0.0, 0.0]
                    g += [Xs[0][10:13] - self.w_init]
                    lb += [0.0, 0.0, 0.0]
                    ub += [0.0, 0.0, 0.0]

            # hard pass-through gate/end point
            g += [Xs[-1][0:3] - p_to]
            lb += [0.0, 0.0, 0.0]
            ub += [0.0, 0.0, 0.0]

            # Optional NN-based gate passing error bound.
            if self.use_error_model:
                u_end = Us[-1] if len(Us) > 0 else DM.zeros(self.NU, 1)
                feat = self._build_error_model_features(p_to, Xs[-1], u_end)
                if self.error_model_type == "nn":
                    ep_pred = self._nn_forward_ep_pred(feat)
                else:
                    # Simple baseline for debugging convergence:
                    # assume |e_p_pred| grows linearly with speed magnitude.
                    v_world = Xs[-1][3:6]
                    ep_pred = DM([self.linear_error_k]) * v_world
                ep_norm = sqrt(dot(ep_pred, ep_pred) + 1e-8)
                # Soft gate-error constraint: ep_norm <= error_bound + slack, slack >= 0
                Es = MX.sym(f"Eslack_{s}", 1)
                self._append_var(f"Eslack_{s}", Es, [0.01], x, xg)
                g += [Es]
                lb += [0.0]
                ub += [5.0]
                g += [ep_norm - self.error_bound - Es]
                lb += [-1e6]
                ub += [0.0]
                J += self.error_penalty * Es[0] * Es[0]

            if prev_end is not None:
                g += [Xs[0] - prev_end]
                lb += [0.0] * self.NX
                ub += [0.0] * self.NX
            prev_end = Xs[-1]

        if self.v_end is not None:
            g += [prev_end[3:6] - DM(self.v_end)]
            lb += [0.0, 0.0, 0.0]
            ub += [0.0, 0.0, 0.0]
        if self.dynamics_model == "full":
            if self.q_end is not None:
                g += [prev_end[6:10] - DM(self.q_end)]
                lb += [0.0, 0.0, 0.0, 0.0]
                ub += [0.0, 0.0, 0.0, 0.0]
            if self.w_end is not None:
                g += [prev_end[10:13] - DM(self.w_end)]
                lb += [0.0, 0.0, 0.0]
                ub += [0.0, 0.0, 0.0]

        self.x = vertcat(*x)
        self.xg = veccat(*xg)
        self.g = vertcat(*g)
        self.lb = veccat(*lb)
        self.ub = veccat(*ub)
        self.nlp = {"f": J, "x": self.x, "g": self.g}

    def _setup_monolithic(self):
        x = []
        xg = []
        g = []
        lb = []
        ub = []
        J = 0

        p_start = self.p_init
        p_final = self.wp[:, -1]
        d_total = p_final - p_start
        dist_total = float(norm_2(d_total))
        v_guess = self.vel_guess * d_total / dist_total if dist_total > 1e-8 else DM.zeros(3, 1)
        T_guess = max(dist_total / max(self.vel_guess, 1e-3), 0.2)

        Tm = MX.sym("T_total", 1)
        self._append_var("T_total", Tm, [T_guess], x, xg)
        g += [Tm]
        lb += [0.05]
        ub += [300.0]
        J += Tm

        Xs = []
        Us = []
        for k in range(self.N_TOTAL + 1):
            Xk = MX.sym(f"X_m_{k}", self.NX)
            alpha = k / self.N_TOTAL
            p_guess = (1 - alpha) * p_start + alpha * p_final
            if self.dynamics_model == "full":
                x_guess = vertcat(p_guess, v_guess, self.q_init, self.w_init)
            else:
                x_guess = vertcat(p_guess, v_guess)
            self._append_var(f"X_m_{k}", Xk, x_guess, x, xg)
            Xs.append(Xk)

            g += [Xk[2]]
            lb += [0.1]
            ub += [200.0]
            if self.dynamics_model == "full":
                g += [dot(Xk[6:10], Xk[6:10])]
                lb += [1.0]
                ub += [1.0]
                if self.enable_tilt_limit:
                    # Tilt bound: angle(body-z, world-z) <= tilt_max
                    qx = Xk[7]
                    qy = Xk[8]
                    r22 = 1 - 2 * (qx * qx + qy * qy)
                    g += [r22]
                    lb += [np.cos(self.tilt_max)]
                    ub += [1.0]
                if self.enable_omega_limit:
                    # body angular-rate bounds [rad/s]
                    g += [Xk[10:13]]
                    lb += [-self.omega_max, -self.omega_max, -self.omega_max]
                    ub += [self.omega_max, self.omega_max, self.omega_max]
                if self.enable_fixed_yaw:
                    qw = Xk[6]
                    qx = Xk[7]
                    qy = Xk[8]
                    qz = Xk[9]
                    siny = 2 * (qw * qz + qx * qy)
                    cosy = 1 - 2 * (qy * qy + qz * qz)
                    g += [siny, cosy]
                    lb += [float(np.sin(self.fixed_yaw)), float(np.cos(self.fixed_yaw))]
                    ub += [float(np.sin(self.fixed_yaw)), float(np.cos(self.fixed_yaw))]

        for k in range(self.N_TOTAL):
            Uk = MX.sym(f"U_m_{k}", self.NU)
            if self.dynamics_model == "full":
                u_guess = [self.quad["mass"] * self.quad["g"] / 4.0] * 4
            else:
                u_guess = [0.0, 0.0, 0.0]
            self._append_var(f"U_m_{k}", Uk, u_guess, x, xg)
            Us.append(Uk)
            g += [Uk]
            if self.dynamics_model == "full":
                lb += [self.quad["T_min"]] * 4
                ub += [self.quad["T_max"]] * 4
                J += self.accel_penalty * Tm / self.N_TOTAL * dot(Uk, Uk)
            else:
                lb += [-self.quad["a_max_xy"], -self.quad["a_max_xy"], -self.quad["a_max_z"]]
                ub += [self.quad["a_max_xy"], self.quad["a_max_xy"], self.quad["a_max_z"]]
                J += self.accel_penalty * Tm / self.N_TOTAL * (Uk[0] ** 2 + Uk[1] ** 2 + Uk[2] ** 2)

        for k in range(self.N_TOTAL):
            dt = Tm / self.N_TOTAL
            xn = self._rk4_step(Xs[k], Us[k], dt)
            g += [Xs[k + 1] - xn]
            lb += [0.0] * self.NX
            ub += [0.0] * self.NX

        g += [Xs[0][0:3] - self.p_init]
        lb += [0.0, 0.0, 0.0]
        ub += [0.0, 0.0, 0.0]
        g += [Xs[0][3:6] - self.v_init]
        lb += [0.0, 0.0, 0.0]
        ub += [0.0, 0.0, 0.0]
        if self.dynamics_model == "full":
            g += [Xs[0][6:10] - self.q_init]
            lb += [0.0, 0.0, 0.0, 0.0]
            ub += [0.0, 0.0, 0.0, 0.0]
            g += [Xs[0][10:13] - self.w_init]
            lb += [0.0, 0.0, 0.0]
            ub += [0.0, 0.0, 0.0]

        # Waypoint pass constraints (soft): ||p(k_i)-wp_i|| <= tol + slack_i
        for i in range(self.NW):
            k_i = int(round((i + 1) * self.N_TOTAL / self.NW))
            k_i = max(1, min(self.N_TOTAL, k_i))
            err = Xs[k_i][0:3] - self.wp[:, i]
            err_norm = sqrt(dot(err, err) + 1e-8)
            Es = MX.sym(f"Wslack_{i}", 1)
            self._append_var(f"Wslack_{i}", Es, [0.01], x, xg)
            g += [Es]
            lb += [0.0]
            ub += [10.0]
            g += [err_norm - self.waypoint_tolerance - Es]
            lb += [-1e6]
            ub += [0.0]
            J += self.waypoint_penalty * Es[0] * Es[0]

        if self.v_end is not None:
            g += [Xs[-1][3:6] - DM(self.v_end)]
            lb += [0.0, 0.0, 0.0]
            ub += [0.0, 0.0, 0.0]
        if self.dynamics_model == "full":
            if self.q_end is not None:
                g += [Xs[-1][6:10] - DM(self.q_end)]
                lb += [0.0, 0.0, 0.0, 0.0]
                ub += [0.0, 0.0, 0.0, 0.0]
            if self.w_end is not None:
                g += [Xs[-1][10:13] - DM(self.w_end)]
                lb += [0.0, 0.0, 0.0]
                ub += [0.0, 0.0, 0.0]

        self.x = vertcat(*x)
        self.xg = veccat(*xg)
        self.g = vertcat(*g)
        self.lb = veccat(*lb)
        self.ub = veccat(*ub)
        self.nlp = {"f": J, "x": self.x, "g": self.g}

    def solve(self, allow_partial: bool = False):
        solver = nlpsol("solver", self.solver_type, self.nlp, self.solver_options)
        sol = solver(x0=self.xg, lbg=self.lb, ubg=self.ub)
        stats = solver.stats()
        self.solver_stats = stats
        status = str(stats.get("return_status", ""))
        success = bool(stats.get("success", False)) or ("Solve_Succeeded" in status) or ("Optimal Solution Found" in status)
        self.solve_status = status
        if not success and not allow_partial:
            raise RuntimeError(f"NLP solve failed. return_status={status}")
        if not success and allow_partial:
            print(f"[WARN] NLP not fully converged, using partial solution. return_status={status}")
        self.x_sol = sol["x"].full().flatten()
        return self.x_sol

    def get_var(self, name):
        s, e = self.var_slices[name]
        return self.x_sol[s:e]

    def extract_position_velocity_trajectory(self):
        if self.planning_mode == "monolithic":
            t = []
            p = []
            v = []
            Tm = float(self.get_var("T_total")[0])
            dt = Tm / self.N_TOTAL
            for k in range(self.N_TOTAL + 1):
                Xk = self.get_var(f"X_m_{k}")
                t.append(k * dt)
                p.append(Xk[0:3])
                v.append(Xk[3:6])
            return np.asarray(t), np.asarray(p), np.asarray(v)

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

    def extract_full_trajectory(self):
        if self.planning_mode == "monolithic":
            t = []
            p = []
            v = []
            q = []
            w = []
            u = []
            Tm = float(self.get_var("T_total")[0])
            dt = Tm / self.N_TOTAL
            for k in range(self.N_TOTAL + 1):
                Xk = self.get_var(f"X_m_{k}")
                t.append(k * dt)
                p.append(Xk[0:3])
                v.append(Xk[3:6])
                if self.dynamics_model == "full":
                    q.append(Xk[6:10])
                    w.append(Xk[10:13])
                    uk_idx = min(k, self.N_TOTAL - 1)
                    Uk = self.get_var(f"U_m_{uk_idx}")
                    u.append(Uk[0:4])
                else:
                    q.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                    w.append(np.array([np.nan, np.nan, np.nan]))
                    uk_idx = min(k, self.N_TOTAL - 1)
                    Uk = self.get_var(f"U_m_{uk_idx}")
                    u.append(np.array([Uk[0], Uk[1], Uk[2], np.nan]))
            return (
                np.asarray(t),
                np.asarray(p),
                np.asarray(v),
                np.asarray(q),
                np.asarray(w),
                np.asarray(u),
            )

        t = []
        p = []
        v = []
        q = []
        w = []
        u = []
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
                if self.dynamics_model == "full":
                    q.append(Xk[6:10])
                    w.append(Xk[10:13])
                    uk_idx = min(k, self.NPS - 1)
                    Uk = self.get_var(f"U_{s}_{uk_idx}")
                    u.append(Uk[0:4])
                else:
                    q.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                    w.append(np.array([np.nan, np.nan, np.nan]))
                    uk_idx = min(k, self.NPS - 1)
                    Uk = self.get_var(f"U_{s}_{uk_idx}")
                    u.append(np.array([Uk[0], Uk[1], Uk[2], np.nan]))
            t_now += Ts
        return (
            np.asarray(t),
            np.asarray(p),
            np.asarray(v),
            np.asarray(q),
            np.asarray(w),
            np.asarray(u),
        )

    def extract_gate_events(self):
        """Return planned gate/end crossing events as dicts with cumulative time and xyz.

        In segmented mode each segment terminal point is hard-constrained to the
        corresponding gate/end, so cumulative segment durations give exact planned
        crossing times. In monolithic mode, use the same nominal waypoint indices
        used by the soft waypoint constraints.
        """
        if self.x_sol is None:
            raise RuntimeError("Solve trajectory first")

        events = []
        if self.planning_mode == "monolithic":
            Tm = float(self.get_var("T_total")[0])
            dt = Tm / self.N_TOTAL
            for i in range(self.NW):
                k_i = int(round((i + 1) * self.N_TOTAL / self.NW))
                k_i = max(1, min(self.N_TOTAL, k_i))
                Xk = self.get_var(f"X_m_{k_i}")
                kind = "end" if self.has_end_pos and i == self.NW - 1 else "gate"
                idx = 0 if kind == "end" else i
                events.append(
                    {
                        "kind": kind,
                        "index": int(idx),
                        "t": float(k_i * dt),
                        "x": float(Xk[0]),
                        "y": float(Xk[1]),
                        "z": float(Xk[2]),
                    }
                )
            return events

        t_now = 0.0
        for i in range(self.NW):
            Ts = float(self.get_var(f"T_{i}")[0])
            t_now += Ts
            Xk = self.get_var(f"X_{i}_{self.NPS}")
            kind = "end" if self.has_end_pos and i == self.NW - 1 else "gate"
            idx = 0 if kind == "end" else i
            events.append(
                {
                    "kind": kind,
                    "index": int(idx),
                    "t": float(t_now),
                    "x": float(Xk[0]),
                    "y": float(Xk[1]),
                    "z": float(Xk[2]),
                }
            )
        return events