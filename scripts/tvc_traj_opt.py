#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization using Crocoddyl

运行方式：
    python -u tvc_traj_opt.py
    
注意：使用 -u 参数（无缓冲输出）可以实时看到求解过程的迭代信息
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
            "terminal_scale": 0.0  # 终端可用单独模型放大
        }
        if weights is not None:
            self.w.update(weights)

        # bounds: dict
        self.b = {
            "th_p": (-0.4, 0.4),
            "th_r": (-0.4, 0.4),
            "T": (0.0, 30.0),
            "tau_yaw": (-2.0, 2.0),
            "k_bound": 200.0
        }
        if bounds is not None:
            self.b.update(bounds)

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

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone

        data.xnext = self._step(x, u)

        # ----- running cost -----
        p = x[0:3]; v = x[3:6]; q = x[6:10]; w = x[10:13]
        u_prev = x[13:17]
        du = u - u_prev

        p_g = self.x_goal[0:3]
        v_g = self.x_goal[3:6]
        q_g = self.x_goal[6:10]
        w_g = self.x_goal[10:13]

        e_p = p - p_g
        e_v = v - v_g
        # attitude error: q_e = q_g * inv(q)
        q_e = quat_mul(q_g, quat_conj(q))
        e_R = so3_log_from_quat(q_e)
        e_w = w - w_g

        cost = 0.0
        cost += self.w["p"] * e_p.dot(e_p)
        cost += self.w["v"] * e_v.dot(e_v)
        cost += self.w["R"] * e_R.dot(e_R)
        cost += self.w["w"] * e_w.dot(e_w)

        # control regularization
        e_u = u - self.u_ref
        cost += self.w["u"]  * e_u.dot(e_u)
        cost += self.w["du"] * du.dot(du)

        # soft bounds
        kB = self.b["k_bound"]
        th_p, th_r, T, tau_yaw = u
        cost += self._bound_pen(th_p, *self.b["th_p"], kB)
        cost += self._bound_pen(th_r, *self.b["th_r"], kB)
        cost += self._bound_pen(T,    *self.b["T"],    kB)
        cost += self._bound_pen(tau_yaw, *self.b["tau_yaw"], kB)

        data.cost = cost

    def calcDiff(self, data, x, u=None):
        """
        计算雅可比矩阵（解析导数）
        这是关键方法，提供解析雅可比可以避免数值微分，大幅提升速度
        """
        if u is None:
            u = self.unone
        
        dt = self.dt
        nx, nu = self.state.nx, self.nu
        
        # 先调用 calc 计算下一步状态和代价
        self.calc(data, x, u)
        
        # 使用数值微分计算雅可比（作为临时方案）
        # 注意：完整的解析雅可比需要手动推导，这里先用数值微分
        # 但至少不需要 ActionModelNumDiff 包装了
        eps = 1e-6
        
        # Fx: 动力学对状态的雅可比 (nx x nx)
        Fx = np.zeros((nx, nx))
        x_pert = x.copy()
        for i in range(nx):
            x_pert[i] += eps
            x_next_pert = self._step(x_pert, u)
            Fx[:, i] = (x_next_pert - data.xnext) / eps
            x_pert[i] = x[i]
        
        # Fu: 动力学对控制的雅可比 (nx x nu)
        Fu = np.zeros((nx, nu))
        u_pert = u.copy()
        for i in range(nu):
            u_pert[i] += eps
            x_next_pert = self._step(x, u_pert)
            Fu[:, i] = (x_next_pert - data.xnext) / eps
            u_pert[i] = u[i]
        
        # Lx: 代价对状态的梯度 (nx,)
        Lx = np.zeros(nx)
        x_pert = x.copy()
        cost_base = data.cost
        for i in range(nx):
            x_pert[i] += eps
            data_pert = self.createData()
            self.calc(data_pert, x_pert, u)
            Lx[i] = (data_pert.cost - cost_base) / eps
            x_pert[i] = x[i]
        
        # Lu: 代价对控制的梯度 (nu,)
        Lu = np.zeros(nu)
        u_pert = u.copy()
        for i in range(nu):
            u_pert[i] += eps
            data_pert = self.createData()
            self.calc(data_pert, x, u_pert)
            Lu[i] = (data_pert.cost - cost_base) / eps
            u_pert[i] = u[i]
        
        # Lxx: 代价对状态的 Hessian (nx x nx) - 简化为零矩阵
        Lxx = np.zeros((nx, nx))
        
        # Luu: 代价对控制的 Hessian (nu x nu) - 简化
        Luu = np.zeros((nu, nu))
        # 控制正则化项的 Hessian
        Luu += 2 * self.w["u"] * np.eye(nu)
        Luu += 2 * self.w["du"] * np.eye(nu)
        
        # Lxu: 代价对状态和控制的混合 Hessian (nx x nu)
        Lxu = np.zeros((nx, nu))
        
        # 存储到 data
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
    """自定义回调类，用于显示求解进度"""
    def __init__(self):
        crocoddyl.CallbackAbstract.__init__(self)
        self.iter_count = 0
        
    def __call__(self, solver):
        self.iter_count += 1
        # 使用 \r 实现同一行更新，\033[K 清除到行尾
        print(f"\r迭代 {self.iter_count}: 成本 = {solver.cost:.6e}, 停止条件 = {solver.stop:.6e}", end='', flush=True)
        sys.stdout.flush()


# -------- build + solve --------
def solve_once(dt=0.02, N=100, max_iter=100):
    """
    求解轨迹优化问题
    
    参数:
        dt: 时间步长 (默认 0.02s)
        N: 时间步数 (默认 100，减少可加快速度但降低精度)
        max_iter: 最大迭代次数 (默认 100)
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
    # terminal model：把终端误差放大
    terminal = TVCRocketActionModel(dt, m, I, r_thrust,
                                    tvc_order="pitch_roll",
                                    x_goal=xg, u_ref=uref,
                                    weights={"p": 200.0, "v": 50.0, "R": 200.0, "w": 20.0,
                                             "u": 0.0, "du": 0.0},
                                    bounds=running.b)

    # 直接使用模型（已实现 calcDiff 方法）
    # 注意：calcDiff 中目前使用数值微分，但避免了 ActionModelNumDiff 的额外开销
    # 未来可以实现完整的解析雅可比以进一步提升速度
    problem = crocoddyl.ShootingProblem(x0, [running]*N, terminal)
    solver  = crocoddyl.SolverFDDP(problem)
    
    # 优化求解器参数以提高速度
    solver.th_stop = 1e-4  # 放宽停止条件（默认 1e-6）
    solver.reg_min = 1e-9  # 最小正则化
    solver.reg_max = 1e6   # 最大正则化

    # 添加回调以显示求解过程
    # 使用 CallbackLogger 记录数据，CallbackVerbose 显示进度
    # 注意：如果看不到实时输出，请使用 python -u 运行脚本（无缓冲输出模式）
    logger = crocoddyl.CallbackLogger()
    callbacks = [
        crocoddyl.CallbackVerbose(),  # 显示详细迭代信息
        logger  # 记录数据用于后续分析
    ]
    solver.setCallbacks(callbacks)

    # initial guess
    xs_init = [x0.copy() for _ in range(N+1)]
    us_init = [uref.copy() for _ in range(N)]

    print("开始求解轨迹优化问题...", flush=True)
    print(f"  - 时间步数: {N}", flush=True)
    print(f"  - 时间步长: {dt} s", flush=True)
    print(f"  - 总时长: {N*dt:.2f} s", flush=True)
    print(f"  - 最大迭代次数: {max_iter}", flush=True)
    print("  - 提示：如果看不到迭代进度，请使用 'python -u' 运行脚本", flush=True)
    print("", flush=True)  # 空行
    
    import time
    start_time = time.time()
    
    # 调用 solve
    # CallbackVerbose 应该会在每次迭代时打印信息
    # 如果看不到输出，可能是输出缓冲问题，使用 python -u 可以解决
    solver.solve(xs_init, us_init, max_iter, False)
    
    solve_time = time.time() - start_time
    
    print("", flush=True)  # 空行
    print(f"求解完成!", flush=True)
    print(f"  - 求解时间: {solve_time:.2f} 秒", flush=True)
    print(f"  - 最终成本: {solver.cost:.6e}", flush=True)
    print(f"  - 迭代次数: {solver.iter}", flush=True)
    print(f"  - 停止条件: {solver.stop:.6e}", flush=True)
    
    # 如果有记录的数据，显示收敛曲线信息
    if len(logger.costs) > 0:
        print(f"  - 成本变化: {logger.costs[0]:.6e} -> {logger.costs[-1]:.6e}", flush=True)
        print(f"  - 平均每次迭代时间: {solve_time/solver.iter:.3f} 秒", flush=True)

    return solver.xs, solver.us, logger

# xs, us = solve_once()

def plot_trajectory(xs, us, dt, logger=None, x_goal=None):
    """
    绘制轨迹优化结果
    
    参数:
        xs: 状态轨迹列表
        us: 控制输入列表
        dt: 时间步长
        logger: 回调记录器（可选，用于绘制收敛曲线）
        x_goal: 目标状态（可选）
    """
    # 转换为 numpy 数组
    xs_array = np.array(xs)
    us_array = np.array(us)
    
    # 时间轴
    time_states = np.arange(len(xs)) * dt
    time_controls = np.arange(len(us)) * dt
    
    # 提取状态
    positions = xs_array[:, 0:3]      # p(3)
    velocities = xs_array[:, 3:6]     # v(3)
    quaternions = xs_array[:, 6:10]    # q(4)
    angular_velocities = xs_array[:, 10:13]  # w(3)
    
    # 提取控制输入
    th_p = us_array[:, 0]  # pitch angle
    th_r = us_array[:, 1]  # roll angle
    T = us_array[:, 2]     # thrust
    tau_yaw = us_array[:, 3]  # yaw torque
    
    # 将四元数转换为欧拉角（用于显示）
    def quat_to_euler(q):
        """四元数转欧拉角 (ZYX顺序)"""
        w, x, y, z = q[0], q[1], q[2], q[3]
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])
    
    euler_angles = np.array([quat_to_euler(q) for q in quaternions])
    
    # 创建图形
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('TVC Rocket Trajectory Optimization Results', fontsize=16, fontweight='bold')
    
    # 1. 3D 位置轨迹
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='轨迹')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                color='green', s=100, marker='o', label='起点')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                color='red', s=100, marker='*', label='终点')
    if x_goal is not None:
        ax1.scatter(x_goal[0], x_goal[1], x_goal[2], 
                   color='orange', s=100, marker='x', label='目标', linewidths=3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Position Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 位置 vs 时间
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(time_states, positions[:, 0], 'r-', label='x', linewidth=2)
    ax2.plot(time_states, positions[:, 1], 'g-', label='y', linewidth=2)
    ax2.plot(time_states, positions[:, 2], 'b-', label='z', linewidth=2)
    if x_goal is not None:
        ax2.axhline(y=x_goal[0], color='r', linestyle='--', alpha=0.5, label='目标 x')
        ax2.axhline(y=x_goal[1], color='g', linestyle='--', alpha=0.5, label='目标 y')
        ax2.axhline(y=x_goal[2], color='b', linestyle='--', alpha=0.5, label='目标 z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 速度
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(time_states, velocities[:, 0], 'r-', label='vx', linewidth=2)
    ax3.plot(time_states, velocities[:, 1], 'g-', label='vy', linewidth=2)
    ax3.plot(time_states, velocities[:, 2], 'b-', label='vz', linewidth=2)
    if x_goal is not None and len(x_goal) > 3:
        ax3.axhline(y=x_goal[3], color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=x_goal[4], color='g', linestyle='--', alpha=0.5)
        ax3.axhline(y=x_goal[5], color='b', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Linear Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 角速度
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(time_states, angular_velocities[:, 0], 'r-', label='ωx', linewidth=2)
    ax4.plot(time_states, angular_velocities[:, 1], 'g-', label='ωy', linewidth=2)
    ax4.plot(time_states, angular_velocities[:, 2], 'b-', label='ωz', linewidth=2)
    if x_goal is not None and len(x_goal) > 10:
        ax4.axhline(y=x_goal[10], color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=x_goal[11], color='g', linestyle='--', alpha=0.5)
        ax4.axhline(y=x_goal[12], color='b', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.set_title('Angular Velocity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 欧拉角
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(time_states, np.degrees(euler_angles[:, 0]), 'r-', label='Roll', linewidth=2)
    ax5.plot(time_states, np.degrees(euler_angles[:, 1]), 'g-', label='Pitch', linewidth=2)
    ax5.plot(time_states, np.degrees(euler_angles[:, 2]), 'b-', label='Yaw', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Euler Angles (deg)')
    ax5.set_title('Attitude (Euler Angles)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. TVC 角度 - Pitch
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(time_controls, np.degrees(th_p), 'b-', linewidth=2, label='θ_pitch')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Angle (deg)')
    ax6.set_title('TVC Pitch Angle')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. TVC 角度 - Roll
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(time_controls, np.degrees(th_r), 'r-', linewidth=2, label='θ_roll')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Angle (deg)')
    ax7.set_title('TVC Roll Angle')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 推力
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(time_controls, T, 'g-', linewidth=2, label='Thrust')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Thrust (N)')
    ax8.set_title('Thrust')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Yaw 扭矩
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(time_controls, tau_yaw, 'm-', linewidth=2, label='τ_yaw')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Torque (N·m)')
    ax9.set_title('Yaw Torque')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. 高度 vs 水平距离
    ax10 = plt.subplot(3, 4, 10)
    ax10.plot(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2), positions[:, 2], 
              'b-', linewidth=2, label='轨迹')
    ax10.scatter(0, positions[0, 2], color='green', s=100, marker='o', label='起点')
    ax10.scatter(np.sqrt(positions[-1, 0]**2 + positions[-1, 1]**2), positions[-1, 2],
                color='red', s=100, marker='*', label='终点')
    if x_goal is not None:
        ax10.scatter(np.sqrt(x_goal[0]**2 + x_goal[1]**2), x_goal[2],
                    color='orange', s=100, marker='x', label='目标', linewidths=3)
    ax10.set_xlabel('Horizontal Distance (m)')
    ax10.set_ylabel('Altitude (m)')
    ax10.set_title('Altitude vs Horizontal Distance')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. 速度大小
    ax11 = plt.subplot(3, 4, 11)
    speed = np.linalg.norm(velocities, axis=1)
    ax11.plot(time_states, speed, 'purple', linewidth=2, label='Speed')
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('Speed (m/s)')
    ax11.set_title('Speed Magnitude')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. 收敛曲线（如果有 logger）
    ax12 = plt.subplot(3, 4, 12)
    if logger is not None and len(logger.costs) > 0:
        iterations = np.arange(len(logger.costs))
        ax12.semilogy(iterations, logger.costs, 'b-', linewidth=2, label='Cost')
        ax12.set_xlabel('Iteration')
        ax12.set_ylabel('Cost (log scale)')
        ax12.set_title('Convergence')
        ax12.legend()
        ax12.grid(True, alpha=0.3)
    else:
        ax12.text(0.5, 0.5, 'No convergence data', 
                 ha='center', va='center', transform=ax12.transAxes)
        ax12.set_title('Convergence')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 可以调整参数来平衡速度和精度
    # N 越小速度越快，但轨迹精度可能降低
    # 建议从 N=50 开始测试，然后根据需要增加
    xs, us, logger = solve_once(dt=0.02, N=100, max_iter=100)
    print("\n" + "="*50)
    print("Solved. N =", len(us))
    print("x0 =", xs[0])
    print("xN =", xs[-1])
    
    # 绘制结果
    print("\n正在绘制轨迹图...")
    x_goal = np.array([0., 0., 10.])  # 目标位置
    fig = plot_trajectory(xs, us, dt=0.02, logger=logger, x_goal=x_goal)
    plt.show()