import os
import signal
import sys
import time
import example_robot_data
import numpy as np
import pinocchio
import crocoddyl
import matplotlib.pyplot as plt

signal.signal(signal.SIGINT, signal.SIG_DFL)

# 1️⃣ 加载模型
hector = example_robot_data.load("hector")
robot_model = hector.model

target_pos = np.array([1.0, 0.0, 1.0])
target_quat = pinocchio.Quaternion(1.0, 0.0, 0.0, 0.0)
state = crocoddyl.StateMultibody(robot_model)

# 2️⃣ 定义致动器
d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
ps = [
    crocoddyl.Thruster(pinocchio.SE3(np.eye(3), np.array([d_cog, 0, 0])), cm / cf, crocoddyl.ThrusterType.CCW),
    crocoddyl.Thruster(pinocchio.SE3(np.eye(3), np.array([0, d_cog, 0])), cm / cf, crocoddyl.ThrusterType.CW),
    crocoddyl.Thruster(pinocchio.SE3(np.eye(3), np.array([-d_cog, 0, 0])), cm / cf, crocoddyl.ThrusterType.CCW),
    crocoddyl.Thruster(pinocchio.SE3(np.eye(3), np.array([0, -d_cog, 0])), cm / cf, crocoddyl.ThrusterType.CW),
]
actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, ps)
nu = actuation.nu

# 3️⃣ 定义 cost
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1.0]*3 + [1.0]*3 + [1.0]*robot_model.nv)
)
uResidual = crocoddyl.ResidualModelControl(state, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    robot_model.getFrameId("base_link"),
    pinocchio.SE3(target_quat.matrix(), target_pos),
    nu,
)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)

runningCostModel.addCost("xReg", xRegCost, 1e-6)
runningCostModel.addCost("uReg", uRegCost, 1e-6)
runningCostModel.addCost("trackPose", goalTrackingCost, 1.0)
terminalCostModel.addCost("goalPose", goalTrackingCost, 1.0)

# 4️⃣ 定义 running & terminal 模型
dt = 0.01
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt
)
# 注意 terminal 的 dt=0
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.0
)

# 5️⃣ 构建 Problem
x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
T = 100
problem = crocoddyl.ShootingProblem(x0, [runningModel]*T, terminalModel)

# 6️⃣ 初始 guess
xs_init = [x0.copy() for _ in range(T+1)]
us_init = [np.zeros(nu) for _ in range(T)]

# 7️⃣ Solver
solver = crocoddyl.SolverFDDP(problem)

# === 关键：添加 plot 回调 ===
log = crocoddyl.CallbackLogger()
# plot = crocoddyl.CallbackPlot()
solver.setCallbacks([log])

solver.solve(xs_init, us_init, 100, False)

# 8️⃣ 获取优化结果
xs = solver.xs
us = solver.us
print(f"✓ 优化完成，最终成本: {solver.cost:.6f}")
print(f"  - 迭代次数: {solver.iter}")
print(f"  - 收敛状态: {'已收敛' if solver.stop < 1e-6 else '未完全收敛'}")

# 9️⃣ 提取状态数据
time_states = np.arange(len(xs)) * dt
time_controls = np.arange(len(us)) * dt

# 提取位置、姿态、速度、角速度
positions = np.array([x[:3] for x in xs])  # x, y, z
quaternions = np.array([x[3:7] for x in xs])  # qx, qy, qz, qw
velocities = np.array([x[7:10] for x in xs])  # vx, vy, vz
angular_velocities = np.array([x[10:13] for x in xs])  # wx, wy, wz

# 提取控制输入
controls = np.array(us)  # 4个推进器的推力

# 🔟 绘制所有状态
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Quadrotor Trajectory Optimization Results', fontsize=16)

# 1. 位置轨迹
ax1 = plt.subplot(3, 3, 1)
ax1.plot(time_states, positions[:, 0], 'r-', label='x', linewidth=2)
ax1.plot(time_states, positions[:, 1], 'g-', label='y', linewidth=2)
ax1.plot(time_states, positions[:, 2], 'b-', label='z', linewidth=2)
ax1.axhline(y=target_pos[0], color='r', linestyle='--', alpha=0.5, label='目标 x')
ax1.axhline(y=target_pos[1], color='g', linestyle='--', alpha=0.5, label='目标 y')
ax1.axhline(y=target_pos[2], color='b', linestyle='--', alpha=0.5, label='目标 z')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)')
ax1.set_title('Position Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 线速度
ax2 = plt.subplot(3, 3, 2)
ax2.plot(time_states, velocities[:, 0], 'r-', label='vx', linewidth=2)
ax2.plot(time_states, velocities[:, 1], 'g-', label='vy', linewidth=2)
ax2.plot(time_states, velocities[:, 2], 'b-', label='vz', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Linear Velocity (m/s)')
ax2.set_title('Linear Velocity')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 角速度
ax3 = plt.subplot(3, 3, 3)
ax3.plot(time_states, angular_velocities[:, 0], 'r-', label='ωx', linewidth=2)
ax3.plot(time_states, angular_velocities[:, 1], 'g-', label='ωy', linewidth=2)
ax3.plot(time_states, angular_velocities[:, 2], 'b-', label='ωz', linewidth=2)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Angular Velocity (rad/s)')
ax3.set_title('Angular Velocity')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 四元数
ax4 = plt.subplot(3, 3, 4)
ax4.plot(time_states, quaternions[:, 0], 'r-', label='qx', linewidth=2)
ax4.plot(time_states, quaternions[:, 1], 'g-', label='qy', linewidth=2)
ax4.plot(time_states, quaternions[:, 2], 'b-', label='qz', linewidth=2)
ax4.plot(time_states, quaternions[:, 3], 'orange', label='qw', linewidth=2)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Quaternion')
ax4.set_title('Orientation (Quaternion)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 控制输入（推力）
ax5 = plt.subplot(3, 3, 5)
colors = ['r', 'g', 'b', 'orange']
for i in range(min(4, controls.shape[1])):
    ax5.plot(time_controls, controls[:, i], color=colors[i], label=f'Thruster {i+1}', linewidth=2)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Thrust (N)')
ax5.set_title('Control Inputs (Thrusters)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 成本收敛
ax6 = plt.subplot(3, 3, 6)
ax6.semilogy(log.costs, 'b-', linewidth=2)
ax6.set_xlabel('Iteration')
ax6.set_ylabel('Cost')
ax6.set_title('Cost Convergence')
ax6.grid(True, alpha=0.3)

# 7. 3D轨迹
ax7 = plt.subplot(3, 3, 7, projection='3d')
ax7.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=3, label='Trajectory')
ax7.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
           color='g', s=100, label='Start', marker='o')
ax7.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
           color='r', s=100, label='End', marker='s')
ax7.scatter(target_pos[0], target_pos[1], target_pos[2], 
           color='orange', s=150, label='Target', marker='*')
ax7.set_xlabel('X (m)')
ax7.set_ylabel('Y (m)')
ax7.set_zlabel('Z (m)')
ax7.set_title('3D Trajectory')
ax7.legend()

# 8. XY平面投影
ax8 = plt.subplot(3, 3, 8)
ax8.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
ax8.scatter(positions[0, 0], positions[0, 1], color='g', s=100, label='Start', marker='o', zorder=5)
ax8.scatter(positions[-1, 0], positions[-1, 1], color='r', s=100, label='End', marker='s', zorder=5)
ax8.scatter(target_pos[0], target_pos[1], color='orange', s=150, label='Target', marker='*', zorder=5)
ax8.set_xlabel('X (m)')
ax8.set_ylabel('Y (m)')
ax8.set_title('XY Projection')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.set_aspect('equal', adjustable='box')

# 9. XZ平面投影
ax9 = plt.subplot(3, 3, 9)
ax9.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
ax9.scatter(positions[0, 0], positions[0, 2], color='g', s=100, label='Start', marker='o', zorder=5)
ax9.scatter(positions[-1, 0], positions[-1, 2], color='r', s=100, label='End', marker='s', zorder=5)
ax9.scatter(target_pos[0], target_pos[2], color='orange', s=150, label='Target', marker='*', zorder=5)
ax9.set_xlabel('X (m)')
ax9.set_ylabel('Z (m)')
ax9.set_title('XZ Projection')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

print(f"\n✓ 绘图完成，显示了所有状态信息")
