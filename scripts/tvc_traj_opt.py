#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization using Crocoddyl

Usage:
    python -u tvc_traj_opt.py
    
Note: Use -u flag (unbuffered output) to see real-time iteration information during solving
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import crocoddyl

# -------- quaternion utils --------
def quat_mul(q1, q2):
    # q = [w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z])

def quat_norm(q):
    return q / np.linalg.norm(q)

def quat_exp(dtheta):
    # exp map from so(3) to quaternion, dtheta is 3-vector
    a = np.linalg.norm(dtheta)
    if a < 1e-12:
        return np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
    axis = dtheta / a
    s = np.sin(0.5*a)
    return np.array([np.cos(0.5*a), axis[0]*s, axis[1]*s, axis[2]*s])

def so3_log_from_quat(q):
    # q must be unit, returns rotation vector
    q = quat_norm(q)
    w, v = q[0], q[1:]
    nv = np.linalg.norm(v)
    w = np.clip(w, -1.0, 1.0)
    if nv < 1e-12:
        return np.zeros(3)
    angle = 2.0*np.arctan2(nv, w)
    return angle * (v / nv)

def R_from_quat(q):
    # q=[w,x,y,z]
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])

def Ry(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])

# -------- Crocoddyl Action Model --------
class TVCRocketActionModel(crocoddyl.ActionModelAbstract):
    """
    x = [p(3), v(3), q(4), w(3), u_prev(4)]  => nx=17
    u = [th_p, th_r, T, tau_yaw]             => nu=4
    """
    def __init__(self, dt, m, I_body, r_thrust_body,
                 g=9.81, tvc_order="pitch_roll",
                 x_goal=None, u_ref=None,
                 weights=None, bounds=None):
        self.dt = float(dt)
        self.m = float(m)
        self.I = np.array(I_body, dtype=float).reshape(3,3)
        self.Iinv = np.linalg.inv(self.I)
        self.r = np.array(r_thrust_body, dtype=float).reshape(3,)
        self.g = float(g)
        self.tvc_order = tvc_order

        # Define dimensions as local variables first
        nx, nu = 17, 4
        state = crocoddyl.StateVector(nx)
        super().__init__(state, nu)

        # Use state.nx and self.nu (set by base class) for dimension access
        self.x_goal = np.zeros(self.state.nx) if x_goal is None else x_goal.copy()
        self.u_ref  = np.zeros(self.nu) if u_ref  is None else u_ref.copy()

        # weights: dict of scalars
        self.w = {
            "p": 1.0, "v": 0.2, "R": 0.5, "w": 0.1,
            "u": 1e-3, "du": 1e-2,
            "terminal_scale": 0.0  # Terminal can use separate model for scaling
        }
        if weights is not None:
            self.w.update(weights)

        # bounds: dict (control constraints)
        self.b = {
            "th_p": (-0.4, 0.4),
            "th_r": (-0.4, 0.4),
            "T": (0.0, 30.0),
            "tau_yaw": (-2.0, 2.0),
            "k_bound": 200.0
        }
        if bounds is not None:
            self.b.update(bounds)
        
        # state_bounds: dict (state constraints)
        self.state_b = {
            "v_horizontal_max": 20.0,  # Maximum horizontal velocity magnitude (m/s)
            "v_vertical_max": 20.0,    # Maximum vertical velocity magnitude (m/s)
            "roll_max": np.radians(45.0),   # Maximum roll angle (rad)
            "pitch_max": np.radians(45.0),  # Maximum pitch angle (rad)
            "yaw_max": np.radians(180.0),   # Maximum yaw angle (rad)
            "w_max": 2.0,            # Maximum angular velocity magnitude (rad/s)
            "k_state_bound": 200.0   # State constraint penalty coefficient
        }
        # Allow state bounds to be passed via bounds dict with "state_" prefix
        if bounds is not None:
            for key, value in bounds.items():
                if key.startswith("state_"):
                    state_key = key[6:]  # Remove "state_" prefix
                    if state_key in self.state_b:
                        self.state_b[state_key] = value
                    # Backward compatibility: convert old v_max to both horizontal and vertical
                    if state_key == "v_max":
                        self.state_b["v_horizontal_max"] = value
                        self.state_b["v_vertical_max"] = value

        self.unone = np.zeros(self.nu)

    def _Rtvc(self, th_p, th_r):
        if self.tvc_order == "pitch_roll":
            return Ry(th_p) @ Rx(th_r)
        elif self.tvc_order == "roll_pitch":
            return Rx(th_r) @ Ry(th_p)
        else:
            raise ValueError("Bad tvc_order")

    def _step(self, x, u):
        dt = self.dt
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]

        th_p, th_r, T, tau_yaw = u
        Rwb = R_from_quat(q)
        Rtvc = self._Rtvc(th_p, th_r)

        Fb = Rtvc @ np.array([0., 0., T])      # body thrust vector
        Fw = Rwb @ Fb

        # torque in body
        tau = np.cross(self.r, Fb) + np.array([0., 0., tau_yaw])

        # dynamics (semi-implicit Euler + quat exp update)
        a = (1.0/self.m)*Fw + np.array([0., 0., -self.g])
        v_next = v + dt*a
        p_next = p + dt*v_next

        w_dot = self.Iinv @ (tau - np.cross(w, self.I @ w))
        w_next = w + dt*w_dot

        dq = quat_exp(w_next * dt)             # use w_next improves stability
        q_next = quat_norm(quat_mul(dq, q))

        x_next = np.zeros_like(x)
        x_next[0:3] = p_next
        x_next[3:6] = v_next
        x_next[6:10] = q_next
        x_next[10:13] = w_next
        x_next[13:17] = u                      # u_prev <- u
        return x_next

    def _bound_pen(self, val, lb, ub, k):
        if val < lb: return k*(lb - val)**2
        if val > ub: return k*(val - ub)**2
        return 0.0
    
    def _quat_to_euler(self, q):
        """Convert quaternion to Euler angles (ZYX order)"""
        w, x, y, z = q[0], q[1], q[2], q[3]
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    def _compute_cost(self, x, u):
        """
        Compute cost (helper method for numerical differentiation)
        Does not modify data object, only returns cost value
        """
        p = x[0:3]; v = x[3:6]; q = x[6:10]; w = x[10:13]
        u_prev = x[13:17]
        du = u - u_prev

        p_g = self.x_goal[0:3]
        v_g = self.x_goal[3:6]
        q_g = self.x_goal[6:10]
        w_g = self.x_goal[10:13]

        e_p = p - p_g
        e_v = v - v_g
        q_e = quat_mul(q_g, quat_conj(q))
        e_R = so3_log_from_quat(q_e)
        e_w = w - w_g

        cost = 0.0
        cost += self.w["p"] * e_p.dot(e_p)
        cost += self.w["v"] * e_v.dot(e_v)
        cost += self.w["R"] * e_R.dot(e_R)
        cost += self.w["w"] * e_w.dot(e_w)

        e_u = u - self.u_ref
        cost += self.w["u"]  * e_u.dot(e_u)
        cost += self.w["du"] * du.dot(du)

        # Control constraints
        kB = self.b["k_bound"]
        th_p, th_r, T, tau_yaw = u
        cost += self._bound_pen(th_p, *self.b["th_p"], kB)
        cost += self._bound_pen(th_r, *self.b["th_r"], kB)
        cost += self._bound_pen(T,    *self.b["T"],    kB)
        cost += self._bound_pen(tau_yaw, *self.b["tau_yaw"], kB)
        
        # State constraints
        kSB = self.state_b["k_state_bound"]
        # Velocity constraints (horizontal and vertical)
        v_horizontal = np.sqrt(v[0]**2 + v[1]**2)  # Horizontal velocity magnitude
        v_vertical = abs(v[2])  # Vertical velocity magnitude
        cost += self._bound_pen(v_horizontal, 0.0, self.state_b["v_horizontal_max"], kSB)
        cost += self._bound_pen(v_vertical, 0.0, self.state_b["v_vertical_max"], kSB)
        # Euler angle constraints
        euler = self._quat_to_euler(q)
        cost += self._bound_pen(abs(euler[0]), 0.0, self.state_b["roll_max"], kSB)  # Roll
        cost += self._bound_pen(abs(euler[1]), 0.0, self.state_b["pitch_max"], kSB)  # Pitch
        cost += self._bound_pen(abs(euler[2]), 0.0, self.state_b["yaw_max"], kSB)    # Yaw
        # Angular velocity magnitude constraint
        w_mag = np.linalg.norm(w)
        cost += self._bound_pen(w_mag, 0.0, self.state_b["w_max"], kSB)
        
        return cost

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone

        data.xnext = self._step(x, u)
        data.cost = self._compute_cost(x, u)

    def calcDiff(self, data, x, u=None):
        """
        Compute Jacobian matrices (optimized numerical differentiation)
        
        Performance optimizations:
        1. Reuse data object to avoid repeated creation
        2. Optimize numerical differentiation computation flow
        3. Reduce unnecessary memory allocation
        
        Note: This is a numerical differentiation implementation, slower than analytical Jacobian,
        but more efficient than previous implementation. Can be further optimized to analytical
        Jacobian for best performance in the future.
        """
        if u is None:
            u = self.unone
        
        dt = self.dt
        nx, nu = self.state.nx, self.nu
        
        # First call calc to compute next state and cost
        self.calc(data, x, u)
        
        # Use numerical differentiation to compute Jacobians
        # Optimization: use smaller perturbation step size, but be careful with numerical precision
        eps = 1e-6
        eps_inv = 1.0 / eps
        
        # Pre-allocate memory
        Fx = np.zeros((nx, nx))
        Fu = np.zeros((nx, nu))
        Lx = np.zeros(nx)
        Lu = np.zeros(nu)
        
        # Cache base values
        x_next_base = data.xnext.copy()
        cost_base = data.cost
        
        # Optimization: reuse temporary variables to reduce memory allocation
        x_pert = x.copy()
        u_pert = u.copy()
        
        # Fx: dynamics Jacobian w.r.t. state (nx x nx)
        # Optimization: directly modify x_pert to avoid repeated copying
        for i in range(nx):
            x_pert[i] = x[i] + eps
            x_next_pert = self._step(x_pert, u)
            Fx[:, i] = (x_next_pert - x_next_base) * eps_inv
            x_pert[i] = x[i]  # Restore original value
        
        # Fu: dynamics Jacobian w.r.t. control (nx x nu)
        for i in range(nu):
            u_pert[i] = u[i] + eps
            x_next_pert = self._step(x, u_pert)
            Fu[:, i] = (x_next_pert - x_next_base) * eps_inv
            u_pert[i] = u[i]  # Restore original value
        
        # Lx: cost gradient w.r.t. state (nx,)
        # Optimization: use helper method to directly compute cost, avoid modifying data object
        for i in range(nx):
            x_pert[i] = x[i] + eps
            cost_pert = self._compute_cost(x_pert, u)
            Lx[i] = (cost_pert - cost_base) * eps_inv
            x_pert[i] = x[i]  # Restore original value
        
        # Lu: cost gradient w.r.t. control (nu,)
        for i in range(nu):
            u_pert[i] = u[i] + eps
            cost_pert = self._compute_cost(x, u_pert)
            Lu[i] = (cost_pert - cost_base) * eps_inv
            u_pert[i] = u[i]  # Restore original value
        
        # Lxx: cost Hessian w.r.t. state (nx x nx) - simplified to zero matrix
        # Note: for quadratic cost, analytical Hessian can be computed, but simplified here
        Lxx = np.zeros((nx, nx))
        
        # Luu: cost Hessian w.r.t. control (nu x nu)
        # Optimization: directly compute analytical Hessian for control regularization term
        Luu = np.zeros((nu, nu))
        # Analytical Hessian for control regularization term
        Luu += 2 * self.w["u"] * np.eye(nu)
        Luu += 2 * self.w["du"] * np.eye(nu)
        # Note: Hessian for boundary penalty term needs to be computed based on whether inside boundary
        # Simplified here, only includes regularization term
        
        # Lxu: cost mixed Hessian w.r.t. state and control (nx x nu)
        Lxu = np.zeros((nx, nu))
        
        # Store to data
        data.Fx = Fx
        data.Fu = Fu
        data.Lx = Lx
        data.Lu = Lu
        data.Lxx = Lxx
        data.Luu = Luu
        data.Lxu = Lxu

    def createData(self):
        data = super().createData()
        data.xnext = np.zeros(self.state.nx)
        return data


# -------- Custom Callback for Progress Display --------
class ProgressCallback(crocoddyl.CallbackAbstract):
    """Custom callback class for displaying solving progress"""
    def __init__(self):
        crocoddyl.CallbackAbstract.__init__(self)
        self.iter_count = 0
        
    def __call__(self, solver):
        self.iter_count += 1
        # Use \r to update same line, \033[K to clear to end of line
        print(f"\rIteration {self.iter_count}: Cost = {solver.cost:.6e}, Stop Condition = {solver.stop:.6e}", end='', flush=True)
        sys.stdout.flush()


# -------- build + solve --------
def solve_once(dt=0.02, N=100, max_iter=100):
    """
    Solve trajectory optimization problem
    
    Parameters:
        dt: Time step (default 0.02s)
        N: Number of time steps (default 100, reducing can speed up but lower accuracy)
        max_iter: Maximum number of iterations (default 100)
    """

    m = 0.6
    I = np.diag([0.02, 0.02, 0.01])
    r_thrust = np.array([0.0, 0.0, -0.2])

    # initial state
    x0 = np.zeros(17)
    x0[6:10] = np.array([1.,0.,0.,0.])  # q0

    # goal
    xg = np.zeros(17)
    xg[0:3]  = np.array([0.,0.,10.])
    xg[6:10] = np.array([1.,0.,0.,0.])  # upright

    # reference control (hover-like)
    uref = np.array([0.0, 0.0, m*9.81, 0.0])

    running = TVCRocketActionModel(dt, m, I, r_thrust,
                                   tvc_order="pitch_roll",
                                   x_goal=xg, u_ref=uref,
                                   weights={"du": 5e-2, "u": 1e-3},
                                   bounds={"T": (0.0, 25.0),
                                           "th_p": (-0.35, 0.35),
                                           "th_r": (-0.35, 0.35),
                                           "tau_yaw": (-1.0, 1.0)})
    # Terminal model: amplify terminal error
    terminal = TVCRocketActionModel(dt, m, I, r_thrust,
                                    tvc_order="pitch_roll",
                                    x_goal=xg, u_ref=uref,
                                    weights={"p": 200.0, "v": 50.0, "R": 200.0, "w": 20.0,
                                             "u": 0.0, "du": 0.0},
                                    bounds=running.b)

    # Directly use model (calcDiff method already implemented)
    # Note: calcDiff currently uses numerical differentiation, but avoids overhead of ActionModelNumDiff
    # Can implement full analytical Jacobian in the future for further speedup
    problem = crocoddyl.ShootingProblem(x0, [running]*N, terminal)
    solver  = crocoddyl.SolverFDDP(problem)
    
    # Optimize solver parameters for speed
    solver.th_stop = 1e-4  # Relax stop condition (default 1e-6)
    solver.reg_min = 1e-9  # Minimum regularization
    solver.reg_max = 1e6   # Maximum regularization

    # Add callbacks to display solving process
    # Use CallbackLogger to record data, CallbackVerbose to display progress
    # Note: If real-time output is not visible, use python -u to run script (unbuffered output mode)
    logger = crocoddyl.CallbackLogger()
    callbacks = [
        crocoddyl.CallbackVerbose(),  # Display detailed iteration information
        logger  # Record data for subsequent analysis
    ]
    solver.setCallbacks(callbacks)

    # Initial guess
    xs_init = [x0.copy() for _ in range(N+1)]
    us_init = [uref.copy() for _ in range(N)]

    print("Starting trajectory optimization problem...", flush=True)
    print(f"  - Number of time steps: {N}", flush=True)
    print(f"  - Time step: {dt} s", flush=True)
    print(f"  - Total duration: {N*dt:.2f} s", flush=True)
    print(f"  - Maximum iterations: {max_iter}", flush=True)
    print("  - Tip: If iteration progress is not visible, use 'python -u' to run script", flush=True)
    print("", flush=True)  # Empty line
    
    import time
    start_time = time.time()
    
    # Call solve
    # CallbackVerbose should print information at each iteration
    # If output is not visible, may be output buffering issue, using python -u can solve it
    solver.solve(xs_init, us_init, max_iter, False)
    
    solve_time = time.time() - start_time
    
    print("", flush=True)  # Empty line
    print(f"Solving completed!", flush=True)
    print(f"  - Solving time: {solve_time:.2f} seconds", flush=True)
    print(f"  - Final cost: {solver.cost:.6e}", flush=True)
    print(f"  - Iterations: {solver.iter}", flush=True)
    print(f"  - Stop condition: {solver.stop:.6e}", flush=True)
    
    # If recorded data exists, display convergence curve information
    if len(logger.costs) > 0:
        print(f"  - Cost change: {logger.costs[0]:.6e} -> {logger.costs[-1]:.6e}", flush=True)
        print(f"  - Average time per iteration: {solve_time/solver.iter:.3f} seconds", flush=True)

    return solver.xs, solver.us, logger

# xs, us = solve_once()

def plot_trajectory(xs, us, dt, logger=None, x_goal=None, waypoints=None):
    """
    Plot trajectory optimization results - all states, controls and cost on one page
    
    Args:
        xs: State trajectory list
        us: Control input list
        dt: Time step
        logger: Callback logger (optional, for plotting convergence curve)
        x_goal: Target state (optional)
        waypoints: List of waypoint positions (optional, for plotting waypoints)
    """
    # Convert to numpy arrays
    xs_array = np.array(xs)
    us_array = np.array(us)
    
    # Time axis
    time_states = np.arange(len(xs)) * dt
    time_controls = np.arange(len(us)) * dt
    
    # Extract states
    positions = xs_array[:, 0:3]      # p(3)
    velocities = xs_array[:, 3:6]     # v(3)
    quaternions = xs_array[:, 6:10]    # q(4)
    angular_velocities = xs_array[:, 10:13]  # w(3)
    
    # Extract control inputs
    th_p = us_array[:, 0]  # pitch angle
    th_r = us_array[:, 1]  # roll angle
    T = us_array[:, 2]     # thrust
    tau_yaw = us_array[:, 3]  # yaw torque
    
    # Convert quaternion to Euler angles (for display)
    def quat_to_euler(q):
        """Convert quaternion to Euler angles (ZYX order)"""
        w, x, y, z = q[0], q[1], q[2], q[3]
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])
    
    euler_angles = np.array([quat_to_euler(q) for q in quaternions])
    
    # Create figure - use GridSpec for flexible layout, all content on one page
    # Layout: 4 rows x 4 columns, 16 subplot positions
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('TVC Rocket Trajectory Optimization - Complete States, Controls and Cost', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # First row: 3D trajectory and convergence curve
    # 1. 3D position trajectory (occupies 2 positions)
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                color='green', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                color='red', s=100, marker='*', label='End')
    
    # Collect all points for unified scaling
    all_x = positions[:, 0].tolist()
    all_y = positions[:, 1].tolist()
    all_z = positions[:, 2].tolist()
    
    if waypoints is not None and len(waypoints) > 0:
        # Plot waypoints with smaller markers and numbered labels
        for i, wp in enumerate(waypoints):
            if len(wp) >= 3:
                # Use smaller, clearer marker (triangle up)
                ax1.scatter(wp[0], wp[1], wp[2], 
                           color='orange', s=50, marker='^', 
                           edgecolors='darkorange', linewidths=1.5, 
                           label=f'WP {i+1}', zorder=5, alpha=0.8)
                # Add text label with waypoint number
                ax1.text(wp[0], wp[1], wp[2], f' {i+1}', 
                        fontsize=9, color='darkorange', 
                        fontweight='bold', zorder=6)
                all_x.append(wp[0])
                all_y.append(wp[1])
                all_z.append(wp[2])
    elif x_goal is not None:
        ax1.scatter(x_goal[0], x_goal[1], x_goal[2], 
                   color='orange', s=100, marker='x', label='Target', linewidths=3)
        all_x.append(x_goal[0])
        all_y.append(x_goal[1])
        all_z.append(x_goal[2])
    
    # Calculate unified scale for all axes
    x_range = max(all_x) - min(all_x) if len(all_x) > 0 else 1.0
    y_range = max(all_y) - min(all_y) if len(all_y) > 0 else 1.0
    z_range = max(all_z) - min(all_z) if len(all_z) > 0 else 1.0
    
    # Use the maximum range for all axes to ensure equal scaling
    max_range = max(x_range, y_range, z_range)
    if max_range == 0:
        max_range = 1.0
    
    x_center = (max(all_x) + min(all_x)) / 2 if len(all_x) > 0 else 0.0
    y_center = (max(all_y) + min(all_y)) / 2 if len(all_y) > 0 else 0.0
    z_center = (max(all_z) + min(all_z)) / 2 if len(all_z) > 0 else 0.0
    
    # Set equal limits for all axes
    half_range = max_range / 2.0
    ax1.set_xlim([x_center - half_range, x_center + half_range])
    ax1.set_ylim([y_center - half_range, y_center + half_range])
    ax1.set_zlim([z_center - half_range, z_center + half_range])
    
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.set_title('3D Position Trajectory', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Cost convergence curve (occupies 2 positions)
    ax_cost = fig.add_subplot(gs[0, 2:4])
    if logger is not None and len(logger.costs) > 0:
        iterations = np.arange(len(logger.costs))
        ax_cost.semilogy(iterations, logger.costs, 'b-', linewidth=2.5, label='Cost', marker='o', markersize=3)
        ax_cost.set_xlabel('Iteration', fontsize=10)
        ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        ax_cost.legend(fontsize=9)
        ax_cost.grid(True, alpha=0.3)
        # Add final cost text
        final_cost = logger.costs[-1]
        ax_cost.text(0.02, 0.98, f'Final Cost: {final_cost:.4e}', 
                    transform=ax_cost.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_cost.text(0.5, 0.5, 'No convergence data', 
                    ha='center', va='center', transform=ax_cost.transAxes, fontsize=12)
        ax_cost.set_title('Cost Convergence', fontsize=11, fontweight='bold')
    
    # Second row: position states
    # 3. Position vs time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_states, positions[:, 0], 'r-', label='x', linewidth=2)
    ax2.plot(time_states, positions[:, 1], 'g-', label='y', linewidth=2)
    ax2.plot(time_states, positions[:, 2], 'b-', label='z', linewidth=2)
    if x_goal is not None:
        ax2.axhline(y=x_goal[0], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.axhline(y=x_goal[1], color='g', linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.axhline(y=x_goal[2], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.set_xlabel('Time (s)', fontsize=9)
    ax2.set_ylabel('Position (m)', fontsize=9)
    ax2.set_title('Position', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 4. Velocity
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_states, velocities[:, 0], 'r-', label='vx', linewidth=2)
    ax3.plot(time_states, velocities[:, 1], 'g-', label='vy', linewidth=2)
    ax3.plot(time_states, velocities[:, 2], 'b-', label='vz', linewidth=2)
    if x_goal is not None and len(x_goal) > 3:
        ax3.axhline(y=x_goal[3], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
        ax3.axhline(y=x_goal[4], color='g', linestyle='--', alpha=0.5, linewidth=1.5)
        ax3.axhline(y=x_goal[5], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.set_xlabel('Time (s)', fontsize=9)
    ax3.set_ylabel('Velocity (m/s)', fontsize=9)
    ax3.set_title('Linear Velocity', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 5. Angular velocity
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(time_states, angular_velocities[:, 0], 'r-', label='ωx', linewidth=2)
    ax4.plot(time_states, angular_velocities[:, 1], 'g-', label='ωy', linewidth=2)
    ax4.plot(time_states, angular_velocities[:, 2], 'b-', label='ωz', linewidth=2)
    if x_goal is not None and len(x_goal) > 10:
        ax4.axhline(y=x_goal[10], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
        ax4.axhline(y=x_goal[11], color='g', linestyle='--', alpha=0.5, linewidth=1.5)
        ax4.axhline(y=x_goal[12], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.set_xlabel('Time (s)', fontsize=9)
    ax4.set_ylabel('Angular Vel (rad/s)', fontsize=9)
    ax4.set_title('Angular Velocity', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 6. Euler angles
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.plot(time_states, np.degrees(euler_angles[:, 0]), 'r-', label='Roll', linewidth=2)
    ax5.plot(time_states, np.degrees(euler_angles[:, 1]), 'g-', label='Pitch', linewidth=2)
    ax5.plot(time_states, np.degrees(euler_angles[:, 2]), 'b-', label='Yaw', linewidth=2)
    ax5.set_xlabel('Time (s)', fontsize=9)
    ax5.set_ylabel('Euler Angles (deg)', fontsize=9)
    ax5.set_title('Attitude (Euler)', fontsize=10, fontweight='bold')
    ax5.legend(fontsize=8, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Third row: control inputs
    # 7. TVC Pitch angle
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(time_controls, np.degrees(th_p), 'b-', linewidth=2, label='θ_pitch', marker='o', markersize=2)
    ax6.set_xlabel('Time (s)', fontsize=9)
    ax6.set_ylabel('Angle (deg)', fontsize=9)
    ax6.set_title('TVC Pitch Angle', fontsize=10, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 8. TVC Roll angle
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(time_controls, np.degrees(th_r), 'r-', linewidth=2, label='θ_roll', marker='o', markersize=2)
    ax7.set_xlabel('Time (s)', fontsize=9)
    ax7.set_ylabel('Angle (deg)', fontsize=9)
    ax7.set_title('TVC Roll Angle', fontsize=10, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 9. Thrust
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(time_controls, T, 'g-', linewidth=2, label='Thrust', marker='o', markersize=2)
    ax8.set_xlabel('Time (s)', fontsize=9)
    ax8.set_ylabel('Thrust (N)', fontsize=9)
    ax8.set_title('Thrust', fontsize=10, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # 10. Yaw torque
    ax9 = fig.add_subplot(gs[2, 3])
    ax9.plot(time_controls, tau_yaw, 'm-', linewidth=2, label='τ_yaw', marker='o', markersize=2)
    ax9.set_xlabel('Time (s)', fontsize=9)
    ax9.set_ylabel('Torque (N·m)', fontsize=9)
    ax9.set_title('Yaw Torque', fontsize=10, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # Fourth row: auxiliary information
    # 11. Altitude vs horizontal distance
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2), positions[:, 2], 
              'b-', linewidth=2, label='Trajectory')
    ax10.scatter(0, positions[0, 2], color='green', s=80, marker='o', label='Start', zorder=5)
    ax10.scatter(np.sqrt(positions[-1, 0]**2 + positions[-1, 1]**2), positions[-1, 2],
                color='red', s=80, marker='*', label='End', zorder=5)
    if waypoints is not None and len(waypoints) > 0:
        # Plot waypoints
        for i, wp in enumerate(waypoints):
            if len(wp) >= 3:
                h_dist = np.sqrt(wp[0]**2 + wp[1]**2)
                ax10.scatter(h_dist, wp[2], color='orange', s=100, marker='s', 
                           label=f'WP {i+1}', linewidths=2, zorder=5)
    elif x_goal is not None:
        ax10.scatter(np.sqrt(x_goal[0]**2 + x_goal[1]**2), x_goal[2],
                    color='orange', s=80, marker='x', label='Target', linewidths=2, zorder=5)
    ax10.set_xlabel('Horizontal Distance (m)', fontsize=9)
    ax10.set_ylabel('Altitude (m)', fontsize=9)
    ax10.set_title('Altitude vs Horizontal', fontsize=10, fontweight='bold')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # 12. Speed magnitude
    ax11 = fig.add_subplot(gs[3, 1])
    speed = np.linalg.norm(velocities, axis=1)
    ax11.plot(time_states, speed, 'purple', linewidth=2, label='Speed')
    ax11.set_xlabel('Time (s)', fontsize=9)
    ax11.set_ylabel('Speed (m/s)', fontsize=9)
    ax11.set_title('Speed Magnitude', fontsize=10, fontweight='bold')
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)
    
    # 13. Control inputs summary (all controls in one plot)
    ax12 = fig.add_subplot(gs[3, 2])
    ax12_twin = ax12.twinx()
    ax12.plot(time_controls, np.degrees(th_p), 'b-', linewidth=2, label='θ_pitch (deg)', alpha=0.7)
    ax12.plot(time_controls, np.degrees(th_r), 'r-', linewidth=2, label='θ_roll (deg)', alpha=0.7)
    ax12_twin.plot(time_controls, T, 'g-', linewidth=2, label='Thrust (N)', alpha=0.7)
    ax12_twin.plot(time_controls, tau_yaw, 'm-', linewidth=2, label='τ_yaw (N·m)', alpha=0.7)
    ax12.set_xlabel('Time (s)', fontsize=9)
    ax12.set_ylabel('TVC Angles (deg)', fontsize=9, color='black')
    ax12_twin.set_ylabel('Thrust & Torque', fontsize=9, color='black')
    ax12.set_title('All Controls', fontsize=10, fontweight='bold')
    ax12.tick_params(axis='y', labelcolor='black')
    ax12_twin.tick_params(axis='y', labelcolor='black')
    # Merge legends
    lines1, labels1 = ax12.get_legend_handles_labels()
    lines2, labels2 = ax12_twin.get_legend_handles_labels()
    ax12.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
    ax12.grid(True, alpha=0.3)
    
    # 14. State summary (position and velocity in one plot)
    ax13 = fig.add_subplot(gs[3, 3])
    ax13_twin = ax13.twinx()
    ax13.plot(time_states, positions[:, 2], 'b-', linewidth=2, label='z (m)', alpha=0.7)
    ax13_twin.plot(time_states, velocities[:, 2], 'r-', linewidth=2, label='vz (m/s)', alpha=0.7)
    if x_goal is not None:
        ax13.axhline(y=x_goal[2], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
        if len(x_goal) > 5:
            ax13_twin.axhline(y=x_goal[5], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    ax13.set_xlabel('Time (s)', fontsize=9)
    ax13.set_ylabel('Altitude (m)', fontsize=9, color='b')
    ax13_twin.set_ylabel('Vertical Vel (m/s)', fontsize=9, color='r')
    ax13.set_title('Position & Velocity', fontsize=10, fontweight='bold')
    ax13.tick_params(axis='y', labelcolor='b')
    ax13_twin.tick_params(axis='y', labelcolor='r')
    lines1, labels1 = ax13.get_legend_handles_labels()
    lines2, labels2 = ax13_twin.get_legend_handles_labels()
    ax13.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
    ax13.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

if __name__ == "__main__":
    # Can adjust parameters to balance speed and accuracy
    # Smaller N is faster, but trajectory accuracy may decrease
    # Recommend starting with N=50 for testing, then increase as needed
    xs, us, logger = solve_once(dt=0.02, N=100, max_iter=100)
    print("\n" + "="*50)
    print("Solved. N =", len(us))
    print("x0 =", xs[0])
    print("xN =", xs[-1])
    
    # Plot results
    print("\nPlotting trajectory...")
    x_goal = np.array([0., 0., 10.])  # Target position
    fig = plot_trajectory(xs, us, dt=0.02, logger=logger, x_goal=x_goal)
    plt.show()