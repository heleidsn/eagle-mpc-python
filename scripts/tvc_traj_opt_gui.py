#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization GUI using PyQt5

使用 PyQt5 创建用户界面，支持：
- 指定初始位置和目标位置
- 调整 cost 权重参数
- 实时显示优化过程和结果

运行方式：
    python tvc_traj_opt_gui.py
    
安装依赖：
    如果遇到 PyQt5 导入错误，请安装：
    - 使用 conda: conda install pyqt
    - 使用 pip: pip install PyQt5
    
注意：需要先激活 conda 环境
    conda activate eagle_mpc
"""

import sys
import os

# 确保可以导入 tvc_traj_opt 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np
import matplotlib

# 检查并导入 Qt 后端
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                                 QGroupBox, QGridLayout, QTextEdit, QTabWidget,
                                 QDoubleSpinBox, QSpinBox, QMessageBox, QProgressBar)
    from PyQt5.QtCore import QThread, pyqtSignal, Qt
    from PyQt5.QtGui import QFont
    QT_AVAILABLE = True
except ImportError:
    try:
        import PySide2
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                      QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                                      QGroupBox, QGridLayout, QTextEdit, QTabWidget,
                                      QDoubleSpinBox, QSpinBox, QMessageBox, QProgressBar)
        from PySide2.QtCore import QThread, Signal as pyqtSignal, Qt
        from PySide2.QtGui import QFont
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        print("=" * 60)
        print("错误：未找到 PyQt5 或 PySide2")
        print("=" * 60)
        print("请安装 PyQt5 或 PySide2：")
        print("  使用 pip:  pip install PyQt5")
        print("  使用 conda: conda install pyqt")
        print("")
        print("如果使用 conda 环境，请运行：")
        print("  conda activate eagle_mpc")
        print("  conda install pyqt")
        print("=" * 60)
        sys.exit(1)

from matplotlib.figure import Figure
import crocoddyl
# 导入 TVC 模型（从同一目录）
from tvc_traj_opt import TVCRocketActionModel, plot_trajectory


class OptimizationThread(QThread):
    """优化线程，在后台运行优化过程"""
    # 信号定义
    iteration_update = pyqtSignal(int, float, float)  # 迭代次数, cost, stop
    state_update = pyqtSignal(list, list)  # xs, us
    finished = pyqtSignal(list, list, object)  # xs, us, logger
    error = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.running = True
        
    def run(self):
        """运行优化"""
        try:
            # 解包参数
            dt = self.params['dt']
            N = self.params['N']
            max_iter = self.params['max_iter']
            x0 = self.params['x0']
            xg = self.params['xg']
            weights = self.params['weights']
            bounds = self.params['bounds']
            m = self.params['m']
            I = self.params['I']
            r_thrust = self.params['r_thrust']
            
            # 创建模型
            uref = np.array([0.0, 0.0, m*9.81, 0.0])
            
            running = TVCRocketActionModel(dt, m, I, r_thrust,
                                         tvc_order="pitch_roll",
                                         x_goal=xg, u_ref=uref,
                                         weights=weights,
                                         bounds=bounds)
            
            terminal = TVCRocketActionModel(dt, m, I, r_thrust,
                                          tvc_order="pitch_roll",
                                          x_goal=xg, u_ref=uref,
                                          weights={**weights, 
                                                  "p": 200.0, "v": 50.0, 
                                                  "R": 200.0, "w": 20.0,
                                                  "u": 0.0, "du": 0.0},
                                          bounds=bounds)
            
            # 创建问题
            problem = crocoddyl.ShootingProblem(x0, [running]*N, terminal)
            solver = crocoddyl.SolverFDDP(problem)
            
            # 设置回调
            logger = crocoddyl.CallbackLogger()
            
            # 自定义回调用于实时更新
            class RealTimeCallback(crocoddyl.CallbackAbstract):
                def __init__(self, thread):
                    crocoddyl.CallbackAbstract.__init__(self)
                    self.thread = thread
                    
                def __call__(self, solver):
                    if self.thread.running:
                        self.thread.iteration_update.emit(
                            solver.iter, solver.cost, solver.stop
                        )
                        # 发送当前状态（每5次迭代发送一次以降低频率）
                        if solver.iter % 5 == 0:
                            # 将 C++ 类型转换为 Python 列表
                            xs_list = [np.array(x) for x in solver.xs]
                            us_list = [np.array(u) for u in solver.us]
                            self.thread.state_update.emit(xs_list, us_list)
            
            callback = RealTimeCallback(self)
            solver.setCallbacks([callback, logger])
            
            # 初始猜测
            xs_init = [x0.copy() for _ in range(N+1)]
            us_init = [uref.copy() for _ in range(N)]
            
            # 求解
            solver.solve(xs_init, us_init, max_iter, False)
            
            if self.running:
                # 将 C++ 类型转换为 Python 列表
                xs_list = [np.array(x) for x in solver.xs]
                us_list = [np.array(u) for u in solver.us]
                self.finished.emit(xs_list, us_list, logger)
                
        except Exception as e:
            if self.running:
                self.error.emit(str(e))
    
    def stop(self):
        """停止优化"""
        self.running = False


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.opt_thread = None
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('TVC Rocket Trajectory Optimization')
        self.setGeometry(100, 100, 1400, 900)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：参数设置面板
        left_panel = self.create_parameter_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右侧：显示面板
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_parameter_panel(self):
        """创建参数设置面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel('参数设置')
        title.setFont(QFont('Arial', 14, QFont.Bold))
        layout.addWidget(title)
        
        # 初始状态
        init_group = QGroupBox('初始状态')
        init_layout = QGridLayout()
        
        self.x0_x = QDoubleSpinBox()
        self.x0_x.setRange(-100, 100)
        self.x0_x.setValue(0.0)
        self.x0_x.setDecimals(2)
        
        self.x0_y = QDoubleSpinBox()
        self.x0_y.setRange(-100, 100)
        self.x0_y.setValue(0.0)
        self.x0_y.setDecimals(2)
        
        self.x0_z = QDoubleSpinBox()
        self.x0_z.setRange(-100, 100)
        self.x0_z.setValue(0.0)
        self.x0_z.setDecimals(2)
        
        init_layout.addWidget(QLabel('X (m):'), 0, 0)
        init_layout.addWidget(self.x0_x, 0, 1)
        init_layout.addWidget(QLabel('Y (m):'), 1, 0)
        init_layout.addWidget(self.x0_y, 1, 1)
        init_layout.addWidget(QLabel('Z (m):'), 2, 0)
        init_layout.addWidget(self.x0_z, 2, 1)
        
        init_group.setLayout(init_layout)
        layout.addWidget(init_group)
        
        # 目标状态
        goal_group = QGroupBox('目标状态')
        goal_layout = QGridLayout()
        
        self.xg_x = QDoubleSpinBox()
        self.xg_x.setRange(-100, 100)
        self.xg_x.setValue(0.0)
        self.xg_x.setDecimals(2)
        
        self.xg_y = QDoubleSpinBox()
        self.xg_y.setRange(-100, 100)
        self.xg_y.setValue(0.0)
        self.xg_y.setDecimals(2)
        
        self.xg_z = QDoubleSpinBox()
        self.xg_z.setRange(-100, 100)
        self.xg_z.setValue(10.0)
        self.xg_z.setDecimals(2)
        
        goal_layout.addWidget(QLabel('X (m):'), 0, 0)
        goal_layout.addWidget(self.xg_x, 0, 1)
        goal_layout.addWidget(QLabel('Y (m):'), 1, 0)
        goal_layout.addWidget(self.xg_y, 1, 1)
        goal_layout.addWidget(QLabel('Z (m):'), 2, 0)
        goal_layout.addWidget(self.xg_z, 2, 1)
        
        goal_group.setLayout(goal_layout)
        layout.addWidget(goal_group)
        
        # 优化参数
        opt_group = QGroupBox('优化参数')
        opt_layout = QGridLayout()
        
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 0.1)
        self.dt_spin.setValue(0.02)
        self.dt_spin.setSingleStep(0.01)
        self.dt_spin.setDecimals(3)
        
        self.N_spin = QSpinBox()
        self.N_spin.setRange(10, 500)
        self.N_spin.setValue(100)
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setValue(100)
        
        opt_layout.addWidget(QLabel('时间步长 (s):'), 0, 0)
        opt_layout.addWidget(self.dt_spin, 0, 1)
        opt_layout.addWidget(QLabel('时间步数:'), 1, 0)
        opt_layout.addWidget(self.N_spin, 1, 1)
        opt_layout.addWidget(QLabel('最大迭代:'), 2, 0)
        opt_layout.addWidget(self.max_iter_spin, 2, 1)
        
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        
        # Cost 权重
        cost_group = QGroupBox('Cost 权重')
        cost_layout = QGridLayout()
        
        self.w_p = QDoubleSpinBox()
        self.w_p.setRange(0, 1000)
        self.w_p.setValue(1.0)
        self.w_p.setDecimals(3)
        
        self.w_v = QDoubleSpinBox()
        self.w_v.setRange(0, 1000)
        self.w_v.setValue(0.2)
        self.w_v.setDecimals(3)
        
        self.w_R = QDoubleSpinBox()
        self.w_R.setRange(0, 1000)
        self.w_R.setValue(0.5)
        self.w_R.setDecimals(3)
        
        self.w_w = QDoubleSpinBox()
        self.w_w.setRange(0, 1000)
        self.w_w.setValue(0.1)
        self.w_w.setDecimals(3)
        
        self.w_u = QDoubleSpinBox()
        self.w_u.setRange(0, 1)
        self.w_u.setValue(0.001)
        self.w_u.setDecimals(6)
        
        self.w_du = QDoubleSpinBox()
        self.w_du.setRange(0, 1)
        self.w_du.setValue(0.05)
        self.w_du.setDecimals(6)
        
        cost_layout.addWidget(QLabel('位置权重 (p):'), 0, 0)
        cost_layout.addWidget(self.w_p, 0, 1)
        cost_layout.addWidget(QLabel('速度权重 (v):'), 1, 0)
        cost_layout.addWidget(self.w_v, 1, 1)
        cost_layout.addWidget(QLabel('姿态权重 (R):'), 2, 0)
        cost_layout.addWidget(self.w_R, 2, 1)
        cost_layout.addWidget(QLabel('角速度权重 (w):'), 3, 0)
        cost_layout.addWidget(self.w_w, 3, 1)
        cost_layout.addWidget(QLabel('控制权重 (u):'), 4, 0)
        cost_layout.addWidget(self.w_u, 4, 1)
        cost_layout.addWidget(QLabel('控制变化权重 (du):'), 5, 0)
        cost_layout.addWidget(self.w_du, 5, 1)
        
        cost_group.setLayout(cost_layout)
        layout.addWidget(cost_group)
        
        # 控制约束
        bounds_group = QGroupBox('控制约束')
        bounds_layout = QGridLayout()
        
        self.th_p_max = QDoubleSpinBox()
        self.th_p_max.setRange(0, 1)
        self.th_p_max.setValue(0.35)
        self.th_p_max.setDecimals(3)
        
        self.th_r_max = QDoubleSpinBox()
        self.th_r_max.setRange(0, 1)
        self.th_r_max.setValue(0.35)
        self.th_r_max.setDecimals(3)
        
        self.T_max = QDoubleSpinBox()
        self.T_max.setRange(0, 100)
        self.T_max.setValue(25.0)
        self.T_max.setDecimals(2)
        
        bounds_layout.addWidget(QLabel('Pitch 最大角度 (rad):'), 0, 0)
        bounds_layout.addWidget(self.th_p_max, 0, 1)
        bounds_layout.addWidget(QLabel('Roll 最大角度 (rad):'), 1, 0)
        bounds_layout.addWidget(self.th_r_max, 1, 1)
        bounds_layout.addWidget(QLabel('最大推力 (N):'), 2, 0)
        bounds_layout.addWidget(self.T_max, 2, 1)
        
        bounds_group.setLayout(bounds_layout)
        layout.addWidget(bounds_group)
        
        # 控制按钮
        self.run_btn = QPushButton('开始优化')
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.start_optimization)
        layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton('停止优化')
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_optimization)
        layout.addWidget(self.stop_btn)
        
        # 进度条
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # 状态信息
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(QLabel('状态信息:'))
        layout.addWidget(self.status_text)
        
        layout.addStretch()
        
        return panel
    
    def create_display_panel(self):
        """创建显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标签页
        tabs = QTabWidget()
        
        # Cost convergence curve
        self.cost_fig = Figure(figsize=(8, 6))
        self.cost_canvas = FigureCanvas(self.cost_fig)
        self.cost_ax = self.cost_fig.add_subplot(111)
        self.cost_ax.set_xlabel('Iteration')
        self.cost_ax.set_ylabel('Cost (log scale)')
        self.cost_ax.set_title('Cost Convergence')
        self.cost_ax.grid(True, alpha=0.3)
        self.cost_canvas.draw()
        tabs.addTab(self.cost_canvas, 'Cost Convergence')
        
        # Position trajectory
        self.pos_fig = Figure(figsize=(8, 6))
        self.pos_canvas = FigureCanvas(self.pos_fig)
        self.pos_ax = self.pos_fig.add_subplot(111)
        self.pos_ax.set_xlabel('Time (s)')
        self.pos_ax.set_ylabel('Position (m)')
        self.pos_ax.set_title('Position Trajectory')
        self.pos_ax.grid(True, alpha=0.3)
        self.pos_canvas.draw()
        tabs.addTab(self.pos_canvas, 'Position')
        
        # Velocity
        self.vel_fig = Figure(figsize=(8, 6))
        self.vel_canvas = FigureCanvas(self.vel_fig)
        self.vel_ax = self.vel_fig.add_subplot(111)
        self.vel_ax.set_xlabel('Time (s)')
        self.vel_ax.set_ylabel('Velocity (m/s)')
        self.vel_ax.set_title('Velocity')
        self.vel_ax.grid(True, alpha=0.3)
        self.vel_canvas.draw()
        tabs.addTab(self.vel_canvas, 'Velocity')
        
        # Control input
        self.ctrl_fig = Figure(figsize=(8, 6))
        self.ctrl_canvas = FigureCanvas(self.ctrl_fig)
        self.ctrl_ax = self.ctrl_fig.add_subplot(111)
        self.ctrl_ax.set_xlabel('Time (s)')
        self.ctrl_ax.set_ylabel('Control Input')
        self.ctrl_ax.set_title('Control Input')
        self.ctrl_ax.grid(True, alpha=0.3)
        self.ctrl_canvas.draw()
        tabs.addTab(self.ctrl_canvas, 'Control')
        
        layout.addWidget(tabs)
        
        # 数据存储
        self.iterations = []
        self.costs = []
        self.stops = []
        self.current_xs = None
        self.current_us = None
        
        return panel
    
    def get_parameters(self):
        """获取参数"""
        # 初始状态
        x0 = np.zeros(17)
        x0[0] = self.x0_x.value()
        x0[1] = self.x0_y.value()
        x0[2] = self.x0_z.value()
        x0[6:10] = np.array([1., 0., 0., 0.])  # 初始四元数
        
        # 目标状态
        xg = np.zeros(17)
        xg[0] = self.xg_x.value()
        xg[1] = self.xg_y.value()
        xg[2] = self.xg_z.value()
        xg[6:10] = np.array([1., 0., 0., 0.])  # 目标四元数
        
        # Cost 权重
        weights = {
            "p": self.w_p.value(),
            "v": self.w_v.value(),
            "R": self.w_R.value(),
            "w": self.w_w.value(),
            "u": self.w_u.value(),
            "du": self.w_du.value()
        }
        
        # 控制约束
        bounds = {
            "th_p": (-self.th_p_max.value(), self.th_p_max.value()),
            "th_r": (-self.th_r_max.value(), self.th_r_max.value()),
            "T": (0.0, self.T_max.value()),
            "tau_yaw": (-1.0, 1.0),
            "k_bound": 200.0
        }
        
        # 物理参数
        m = 0.6
        I = np.diag([0.02, 0.02, 0.01])
        r_thrust = np.array([0.0, 0.0, -0.2])
        
        return {
            'dt': self.dt_spin.value(),
            'N': self.N_spin.value(),
            'max_iter': self.max_iter_spin.value(),
            'x0': x0,
            'xg': xg,
            'weights': weights,
            'bounds': bounds,
            'm': m,
            'I': I,
            'r_thrust': r_thrust
        }
    
    def start_optimization(self):
        """开始优化"""
        if self.opt_thread and self.opt_thread.isRunning():
            QMessageBox.warning(self, '警告', '优化正在进行中，请先停止当前优化')
            return
        
        # 重置显示
        self.iterations = []
        self.costs = []
        self.stops = []
        self.cost_ax.clear()
        self.cost_ax.set_xlabel('迭代次数')
        self.cost_ax.set_ylabel('Cost (log scale)')
        self.cost_ax.set_title('Cost 收敛曲线')
        self.cost_ax.grid(True, alpha=0.3)
        self.cost_canvas.draw()
        
        # 获取参数
        params = self.get_parameters()
        
        # 创建优化线程
        self.opt_thread = OptimizationThread(params)
        self.opt_thread.iteration_update.connect(self.update_iteration)
        self.opt_thread.state_update.connect(self.update_state)
        self.opt_thread.finished.connect(self.optimization_finished)
        self.opt_thread.error.connect(self.optimization_error)
        
        # 更新按钮状态
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setMaximum(params['max_iter'])
        self.progress.setValue(0)
        
        # 启动优化
        self.status_text.append('开始优化...')
        self.opt_thread.start()
    
    def stop_optimization(self):
        """停止优化"""
        if self.opt_thread and self.opt_thread.isRunning():
            self.opt_thread.stop()
            self.opt_thread.wait()
            self.status_text.append('优化已停止')
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def update_iteration(self, iter_num, cost, stop):
        """更新迭代信息"""
        self.iterations.append(iter_num)
        self.costs.append(cost)
        self.stops.append(stop)
        
        # 更新进度条
        self.progress.setValue(iter_num)
        
        # 更新状态文本
        self.status_text.clear()
        self.status_text.append(f'迭代: {iter_num}')
        self.status_text.append(f'Cost: {cost:.6e}')
        self.status_text.append(f'停止条件: {stop:.6e}')
        
        # Update Cost curve
        self.cost_ax.clear()
        if len(self.iterations) > 0:
            self.cost_ax.semilogy(self.iterations, self.costs, 'b-', linewidth=2)
            self.cost_ax.set_xlabel('Iteration')
            self.cost_ax.set_ylabel('Cost (log scale)')
            self.cost_ax.set_title('Cost Convergence')
            self.cost_ax.grid(True, alpha=0.3)
        self.cost_canvas.draw()
    
    def update_state(self, xs, us):
        """更新状态显示"""
        self.current_xs = xs
        self.current_us = us
        
        if xs is None or len(xs) == 0:
            return
        
        dt = self.dt_spin.value()
        time_states = np.arange(len(xs)) * dt
        time_controls = np.arange(len(us)) * dt
        
        # Update position plot
        xs_array = np.array(xs)
        positions = xs_array[:, 0:3]
        
        self.pos_ax.clear()
        self.pos_ax.plot(time_states, positions[:, 0], 'r-', label='x', linewidth=2)
        self.pos_ax.plot(time_states, positions[:, 1], 'g-', label='y', linewidth=2)
        self.pos_ax.plot(time_states, positions[:, 2], 'b-', label='z', linewidth=2)
        self.pos_ax.set_xlabel('Time (s)')
        self.pos_ax.set_ylabel('Position (m)')
        self.pos_ax.set_title('Position Trajectory')
        self.pos_ax.legend()
        self.pos_ax.grid(True, alpha=0.3)
        self.pos_canvas.draw()
        
        # Update velocity plot
        velocities = xs_array[:, 3:6]
        self.vel_ax.clear()
        self.vel_ax.plot(time_states, velocities[:, 0], 'r-', label='vx', linewidth=2)
        self.vel_ax.plot(time_states, velocities[:, 1], 'g-', label='vy', linewidth=2)
        self.vel_ax.plot(time_states, velocities[:, 2], 'b-', label='vz', linewidth=2)
        self.vel_ax.set_xlabel('Time (s)')
        self.vel_ax.set_ylabel('Velocity (m/s)')
        self.vel_ax.set_title('Velocity')
        self.vel_ax.legend()
        self.vel_ax.grid(True, alpha=0.3)
        self.vel_canvas.draw()
        
        # Update control plot
        us_array = np.array(us)
        self.ctrl_ax.clear()
        self.ctrl_ax.plot(time_controls, us_array[:, 0], 'b-', label='θ_pitch', linewidth=2)
        self.ctrl_ax.plot(time_controls, us_array[:, 1], 'r-', label='θ_roll', linewidth=2)
        self.ctrl_ax.plot(time_controls, us_array[:, 2], 'g-', label='Thrust', linewidth=2)
        self.ctrl_ax.plot(time_controls, us_array[:, 3], 'm-', label='τ_yaw', linewidth=2)
        self.ctrl_ax.set_xlabel('Time (s)')
        self.ctrl_ax.set_ylabel('Control Input')
        self.ctrl_ax.set_title('Control Input')
        self.ctrl_ax.legend()
        self.ctrl_ax.grid(True, alpha=0.3)
        self.ctrl_canvas.draw()
    
    def optimization_finished(self, xs, us, logger):
        """优化完成"""
        self.status_text.append('优化完成！')
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(self.progress.maximum())
        
        # 显示最终结果
        if logger and len(logger.costs) > 0:
            final_cost = logger.costs[-1]
            self.status_text.append(f'最终 Cost: {final_cost:.6e}')
            self.status_text.append(f'总迭代次数: {len(logger.costs)}')
        
        # 更新最终状态
        self.update_state(xs, us)
        
        # 询问是否显示完整轨迹图
        reply = QMessageBox.question(self, '优化完成', 
                                    '优化已完成！是否显示完整的轨迹图？',
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.show_full_trajectory(xs, us, logger)
    
    def optimization_error(self, error_msg):
        """优化错误"""
        QMessageBox.critical(self, '错误', f'优化过程中出现错误：\n{error_msg}')
        self.status_text.append(f'错误: {error_msg}')
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def show_full_trajectory(self, xs, us, logger):
        """显示完整轨迹图"""
        try:
            from tvc_traj_opt import plot_trajectory
            import matplotlib.pyplot as plt
            
            x_goal = np.array([self.xg_x.value(), self.xg_y.value(), self.xg_z.value()])
            dt = self.dt_spin.value()
            fig = plot_trajectory(xs, us, dt, logger, x_goal)
            plt.show()
        except Exception as e:
            QMessageBox.warning(self, '警告', f'无法显示完整轨迹图：{str(e)}')


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
