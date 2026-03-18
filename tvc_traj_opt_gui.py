#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization GUI using PyQt5

Create user interface using PyQt5, supports:
- Specify initial and target positions
- Adjust cost weight parameters
- Real-time display of optimization process and results

Usage:
    python tvc_traj_opt_gui.py
    
Installation:
    If PyQt5 import error occurs, please install:
    - Using conda: conda install pyqt
    - Using pip: pip install PyQt5
    
Note: Need to activate conda environment first
    conda activate eagle_mpc
"""

import sys
import os
import time

# Ensure tvc_traj_opt module can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np
import matplotlib

# Check and import Qt backend
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                                 QGroupBox, QGridLayout, QTextEdit, QTabWidget,
                                 QDoubleSpinBox, QSpinBox, QMessageBox, QProgressBar, QComboBox, QCheckBox)
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
                                      QDoubleSpinBox, QSpinBox, QMessageBox, QProgressBar, QComboBox, QCheckBox)
        from PySide2.QtCore import QThread, Signal as pyqtSignal, Qt
        from PySide2.QtGui import QFont
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        print("=" * 60)
        print("Error: PyQt5 or PySide2 not found")
        print("=" * 60)
        print("Please install PyQt5 or PySide2:")
        print("  Using pip:  pip install PyQt5")
        print("  Using conda: conda install pyqt")
        print("")
        print("If using conda environment, please run:")
        print("  conda activate eagle_mpc")
        print("  conda install pyqt")
        print("=" * 60)
        sys.exit(1)

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import crocoddyl
# Import TVC model and common utils (from same directory)
from tvc_traj_opt import TVCRocketActionModel, plot_trajectory
from tvc_common import quat_to_euler, yaw_to_quaternion, SEGMENT_COLORS, segment_boundaries_from_waypoints
from tvc_traj_opt_pinocchio import (solve_with_pinocchio_waypoints, solve_with_pinocchio_waypoints_unified,
                                    convert_pinocchio_state_to_method1)
try:
    from tvc_traj_opt_acados import (solve_with_acados_waypoints, solve_with_acados_waypoints_unified,
                                     ACADOS_AVAILABLE)
except ImportError:
    solve_with_acados_waypoints = None
    solve_with_acados_waypoints_unified = None
    ACADOS_AVAILABLE = False


class OptimizationThread(QThread):
    """Optimization thread, runs optimization process in background"""
    # Signal definitions
    iteration_update = pyqtSignal(int, float, float, int)  # iteration number, cost, stop, segment_index
    state_update = pyqtSignal(list, list)  # xs, us
    finished = pyqtSignal(list, list, list, dict)  # xs, us, all_loggers, timing_info
    error = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.running = True
        
    def run(self):
        """Run optimization with multiple waypoints and time segments"""
        try:
            # Unpack parameters
            dt = self.params['dt']
            max_iter = self.params['max_iter']
            weights = self.params['weights']
            terminal_weights = self.params.get('terminal_weights', None)
            bounds = self.params['bounds']
            m = self.params['m']
            I = self.params['I']
            r_thrust = self.params['r_thrust']
            waypoints = self.params.get('waypoints', [])
            method = self.params.get('method', 0)  # 0=Method1, 1=Method2 FDDP, 2=Method3 BoxFDDP, 3=Method4 Acados
            use_box_solver = (method == 2)  # BoxFDDP uses native control bounds
            unified = self.params.get('unified', False)  # Merge all segments into one problem
            
            # Method 4: Acados (native constraints)
            if method == 3:
                if not ACADOS_AVAILABLE:
                    self.error.emit("Acados not available. Install: pip install casadi; build acados from source.")
                    return
                def callback_acados(solver, seg_idx, current_xs, current_us, completed_xs, completed_us):
                    if self.running and current_xs:
                        # Same as Pinocchio: send completed segments + current, show full trajectory
                        combined_xs, combined_us = [], []
                        for i, (seg_xs, seg_us) in enumerate(zip(completed_xs, completed_us)):
                            if i == 0:
                                combined_xs.extend(seg_xs)
                                combined_us.extend(seg_us)
                            else:
                                combined_xs.extend(seg_xs[1:])
                                combined_us.extend(seg_us)
                        if len(combined_xs) > 0 and seg_idx > 0:
                            combined_xs.extend(current_xs[1:])
                        else:
                            combined_xs.extend(current_xs)
                        combined_us.extend(current_us)
                        self.state_update.emit(combined_xs, combined_us)
                def iteration_callback_acados(iter, cost, stop, seg_idx):
                    if self.running:
                        self.iteration_update.emit(iter, cost, stop, seg_idx)
                # When multi-waypoint and need waypoint terminal cost: use segment mode (terminal cost at each segment end)
                use_segment_for_waypoints = (
                    len(waypoints) > 2 and unified and
                    weights.get("waypoint_terminal_cost", True)
                )
                if use_segment_for_waypoints:
                    unified = False  # Switch to segment mode for terminal cost at each intermediate point
                solver_fn = solve_with_acados_waypoints_unified if unified else solve_with_acados_waypoints
                t0 = time.perf_counter()
                combined_xs, combined_us, all_loggers = solver_fn(
                    dt=dt, waypoints=waypoints, m=m, I=I, r_thrust=r_thrust,
                    weights=weights, terminal_weights=terminal_weights, bounds=bounds,
                    max_iter=max_iter, use_box_solver=False, callback=callback_acados,
                    running_flag=lambda: self.running,
                    iteration_callback=iteration_callback_acados,
                    verbose_solve=True  # For store_iterates and cost curve plotting
                )
                total_time = time.perf_counter() - t0
                total_iters = sum(len(logger.costs) for logger in all_loggers) if all_loggers else 0
                timing_info = {"total_time": total_time, "total_iters": total_iters,
                              "avg_time_per_iter": total_time / total_iters if total_iters > 0 else 0.0,
                              "method": "Method 4 (Acados)"}
                if self.running:
                    self.finished.emit(combined_xs, combined_us, all_loggers, timing_info)
                return
            
            # Check if we have waypoints with times
            if len(waypoints) < 2:
                self.error.emit("Need at least 2 waypoints (start and at least one waypoint)")
                return
            
            # Ensure all waypoints have all required fields and convert to lists
            # Format: [x, y, z, yaw_deg, time]
            waypoints_list = []
            for wp in waypoints:
                wp_list = list(wp) if isinstance(wp, (list, tuple, np.ndarray)) else [wp]
                # Ensure waypoint has 5 elements [x, y, z, yaw_deg, time]
                if len(wp_list) == 4:
                    # Old format: [x, y, z, time] -> convert to [x, y, z, yaw=0, time]
                    wp_list = [wp_list[0], wp_list[1], wp_list[2], 0.0, wp_list[3]]
                while len(wp_list) < 5:
                    wp_list.append(0.0)
                waypoints_list.append(wp_list[:5])  # Keep only first 5 elements
            
            # Sort waypoints by time (index 4)
            waypoints = sorted(waypoints_list, key=lambda x: float(x[4]))
            
            # Calculate segment durations
            durations = []
            for i in range(len(waypoints) - 1):
                duration = waypoints[i+1][4] - waypoints[i][4]  # time is at index 4
                if duration <= 0:
                    self.error.emit(f"Waypoint {i+1} time must be greater than waypoint {i} time")
                    return
                durations.append(duration)
            
            uref = np.array([0.0, 0.0, m*9.81, 0.0])
            
            # Check which method to use
            if method >= 1:  # Method 2: Pinocchio (FDDP or BoxFDDP)
                # Use Pinocchio-based optimization
                def callback_func(solver, seg_idx, current_xs, current_us, completed_xs, completed_us):
                    """Callback for real-time updates in Pinocchio method"""
                    if self.running:
                        # Emit iteration update
                        self.iteration_update.emit(solver.iter, solver.cost, solver.stop, seg_idx)
                        
                        # Update state display periodically
                        if solver.iter % 5 == 0:
                            # Combine with completed segments
                            combined_xs = []
                            combined_us = []
                            
                            # Add all completed segments
                            for i, (seg_xs, seg_us) in enumerate(zip(completed_xs, completed_us)):
                                if i == 0:
                                    combined_xs.extend(seg_xs)
                                    combined_us.extend(seg_us)
                                else:
                                    combined_xs.extend(seg_xs[1:])
                                    combined_us.extend(seg_us)
                            
                            # Add current segment
                            if len(combined_xs) > 0:
                                if seg_idx > 0:
                                    combined_xs.extend(current_xs[1:])
                                else:
                                    combined_xs.extend(current_xs)
                            else:
                                combined_xs.extend(current_xs)
                            combined_us.extend(current_us)
                            
                            if len(combined_xs) > 0:
                                self.state_update.emit(combined_xs, combined_us)
                
                # Use Pinocchio method (FDDP or BoxFDDP), segmented or unified
                solver_fn = solve_with_pinocchio_waypoints_unified if unified else solve_with_pinocchio_waypoints
                t0 = time.perf_counter()
                combined_xs, combined_us, all_loggers = solver_fn(
                    dt=dt,
                    waypoints=waypoints,
                    m=m,
                    I=I,
                    r_thrust=r_thrust,
                    weights=weights,
                    terminal_weights=terminal_weights,
                    bounds=bounds,
                    max_iter=max_iter,
                    use_box_solver=use_box_solver,
                    callback=callback_func,
                    running_flag=lambda: self.running
                )
                total_time = time.perf_counter() - t0
                total_iters = sum(len(logger.costs) for logger in all_loggers) if all_loggers else 0
                avg_time_per_iter = total_time / total_iters if total_iters > 0 else 0.0
                method_name = "Method 3 (BoxFDDP)" if use_box_solver else "Method 2 (FDDP)"
                timing_info = {"total_time": total_time, "total_iters": total_iters,
                              "avg_time_per_iter": avg_time_per_iter, "method": method_name}
                if self.running:
                    self.finished.emit(combined_xs, combined_us, all_loggers, timing_info)
                return
            
            # Method 1: Custom calcDiff (original implementation)
            # Store all segments' trajectories
            all_xs = []
            all_us = []
            all_loggers = []
            t0_method1 = time.perf_counter()
            total_iters_method1 = 0
            
            # Custom callback for real-time updates
            class RealTimeCallback(crocoddyl.CallbackAbstract):
                def __init__(self, thread, seg_idx, completed_segments_xs, completed_segments_us):
                    crocoddyl.CallbackAbstract.__init__(self)
                    self.thread = thread
                    self.seg_idx = seg_idx
                    self.completed_segments_xs = completed_segments_xs  # List of completed segment trajectories
                    self.completed_segments_us = completed_segments_us  # List of completed segment controls
                    self.last_update_iter = -1
                    
                def __call__(self, solver):
                    if self.thread.running:
                        # Emit iteration update with segment index
                        self.thread.iteration_update.emit(
                            solver.iter, solver.cost, solver.stop, self.seg_idx
                        )
                        
                        # Update state display periodically
                        if solver.iter % 5 == 0 and solver.iter != self.last_update_iter:
                            self.last_update_iter = solver.iter
                            # Try to get current solver's states for display
                            try:
                                if hasattr(solver, 'xs') and len(solver.xs) > 0:
                                    # Get current segment's trajectory
                                    current_xs = [np.array(x) for x in solver.xs]
                                    current_us = [np.array(u) for u in solver.us]
                                    
                                    # Combine with completed segments
                                    combined_xs = []
                                    combined_us = []
                                    
                                    # Add all completed segments
                                    for i, (seg_xs, seg_us) in enumerate(zip(self.completed_segments_xs, self.completed_segments_us)):
                                        if i == 0:
                                            # First segment: include all states and controls
                                            combined_xs.extend(seg_xs)
                                            combined_us.extend(seg_us)
                                        else:
                                            # Subsequent segments: skip first state (duplicate)
                                            combined_xs.extend(seg_xs[1:])
                                            combined_us.extend(seg_us)
                                    
                                    # Add current segment being optimized
                                    if len(combined_xs) > 0:
                                        # Skip first state of current segment if it's not the first segment
                                        if self.seg_idx > 0:
                                            combined_xs.extend(current_xs[1:])
                                        else:
                                            combined_xs.extend(current_xs)
                                    else:
                                        combined_xs.extend(current_xs)
                                    combined_us.extend(current_us)
                                    
                                    if len(combined_xs) > 0:
                                        self.thread.state_update.emit(combined_xs, combined_us)
                            except Exception:
                                pass  # Ignore errors in callback
            
            self.all_xs = []
            self.all_us = []
            
            # Initial state for first segment
            x0_seg = np.zeros(17)
            if len(waypoints) > 0:
                first_wp = waypoints[0]
                x0_seg[0:3] = [float(first_wp[0]), float(first_wp[1]), float(first_wp[2])]  # Position
                # Convert yaw to quaternion
                yaw_deg = float(first_wp[3]) if len(first_wp) > 3 else 0.0
                yaw_rad = np.radians(yaw_deg)
                x0_seg[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])  # Quaternion from yaw
                # Velocity and angular velocity start at zero
            
            # Solve each segment
            for seg_idx in range(len(durations)):
                if not self.running:
                    break
                
                duration = durations[seg_idx]
                start_wp = waypoints[seg_idx]
                end_wp = waypoints[seg_idx + 1]
                
                # Calculate number of time steps for this segment
                N = max(10, int(duration / dt))
                
                # Debug info
                start_yaw = float(start_wp[3]) if len(start_wp) > 3 else 0.0
                end_yaw = float(end_wp[3]) if len(end_wp) > 3 else 0.0
                print(f"Segment {seg_idx + 1}/{len(durations)}: {duration:.2f}s, {N} steps, "
                      f"from [{float(start_wp[0]):.2f}, {float(start_wp[1]):.2f}, {float(start_wp[2]):.2f}] yaw={start_yaw:.1f}° "
                      f"to [{float(end_wp[0]):.2f}, {float(end_wp[1]):.2f}, {float(end_wp[2]):.2f}] yaw={end_yaw:.1f}°")
                
                # For subsequent segments, use final state from previous segment
                # (x0_seg is already set from previous iteration)
                
                # Target state for this segment
                xg_seg = np.zeros(17)
                xg_seg[0:3] = [float(end_wp[0]), float(end_wp[1]), float(end_wp[2])]  # Target position
                # Convert yaw to quaternion
                yaw_deg = float(end_wp[3]) if len(end_wp) > 3 else 0.0
                yaw_rad = np.radians(yaw_deg)
                xg_seg[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])  # Quaternion from yaw
                # Target velocity and angular velocity are zero
                
                # Create models for this segment
                running = TVCRocketActionModel(dt, m, I, r_thrust,
                                             tvc_order="pitch_roll",
                                             x_goal=xg_seg, u_ref=uref,
                                             weights=weights,
                                             bounds=bounds)
                
                # Terminal model with weights from GUI (or fallback to defaults)
                tw = terminal_weights if terminal_weights else {
                    "p": 200.0, "v": 50.0, "R": 200.0, "w": 20.0, "u": 0.0, "du": 0.0
                }
                terminal = TVCRocketActionModel(dt, m, I, r_thrust,
                                              tvc_order="pitch_roll",
                                              x_goal=xg_seg, u_ref=uref,
                                              weights=tw,
                                              bounds=bounds)
                
                # Create problem for this segment
                problem = crocoddyl.ShootingProblem(x0_seg, [running]*N, terminal)
                solver = crocoddyl.SolverFDDP(problem)
                
                # Set callbacks
                logger = crocoddyl.CallbackLogger()
                # Create callback with access to completed segments for real-time updates
                callback = RealTimeCallback(self, seg_idx, all_xs.copy(), all_us.copy())
                # Use callback for all segments to show cumulative progress
                solver.setCallbacks([callback, logger])
                all_loggers.append(logger)
                
                # Initial guess
                xs_init = [x0_seg.copy() for _ in range(N+1)]
                us_init = [uref.copy() for _ in range(N)]
                
                # Solve this segment
                try:
                    solver.solve(xs_init, us_init, max_iter, False)
                    total_iters_method1 += solver.iter
                    print(f"  Segment {seg_idx + 1} solved: cost={solver.cost:.6e}, iter={solver.iter}")
                except Exception as e:
                    error_msg = f"Error solving segment {seg_idx + 1}: {str(e)}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                # Store results
                seg_xs = [np.array(x) for x in solver.xs]
                seg_us = [np.array(u) for u in solver.us]
                
                if len(seg_xs) == 0 or len(seg_us) == 0:
                    raise Exception(f"Segment {seg_idx + 1} produced empty trajectory")
                
                # Verify state continuity at connection point (for segments after the first)
                if seg_idx > 0 and len(all_xs) > 0:
                    prev_final = all_xs[-1][-1]  # Previous segment's final state
                    curr_initial = seg_xs[0]      # Current segment's initial state
                    state_diff = np.linalg.norm(prev_final - curr_initial)
                    if state_diff > 1e-6:  # Check if states match (allowing small numerical error)
                        print(f"  Warning: State discontinuity at segment {seg_idx + 1} connection: "
                              f"diff={state_diff:.2e}")
                        # Force continuity by using previous segment's final state
                        seg_xs[0] = prev_final.copy()
                
                self.all_xs.append(seg_xs)
                self.all_us.append(seg_us)
                all_xs.append(seg_xs)
                all_us.append(seg_us)
                
                # Update initial state for next segment (use final state of current segment)
                # This ensures state continuity: next segment starts exactly where current segment ends
                if seg_idx < len(durations) - 1:
                    x0_seg = seg_xs[-1].copy()  # Use final state as next segment's initial state
                    print(f"  Segment {seg_idx + 1} final state -> Segment {seg_idx + 2} initial state")
            
            if self.running:
                # Combine all segments
                combined_xs = []
                combined_us = []
                
                # Add all states and controls, connecting segments smoothly
                for i, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
                    if i == 0:
                        # First segment: include all states and controls
                        combined_xs.extend(seg_xs)
                        combined_us.extend(seg_us)
                    else:
                        # Subsequent segments: skip first state (duplicate of previous segment's last state)
                        # but keep all controls
                        combined_xs.extend(seg_xs[1:])  # Skip duplicate state
                        combined_us.extend(seg_us)  # Keep all controls
                
                # Validate combined trajectory
                if len(combined_xs) == 0 or len(combined_us) == 0:
                    raise Exception("Combined trajectory is empty")
                
                print(f"Combined trajectory: {len(combined_xs)} states, {len(combined_us)} controls")
                
                total_time_method1 = time.perf_counter() - t0_method1
                avg_time_per_iter = total_time_method1 / total_iters_method1 if total_iters_method1 > 0 else 0.0
                timing_info = {"total_time": total_time_method1, "total_iters": total_iters_method1,
                              "avg_time_per_iter": avg_time_per_iter, "method": "Method 1 (Custom calcDiff)"}
                self.finished.emit(combined_xs, combined_us, all_loggers, timing_info)
                
        except Exception as e:
            import traceback
            error_msg = f"Optimization error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            if self.running:
                self.error.emit(error_msg)
            print(error_msg)  # Also print to console for debugging
    
    def stop(self):
        """Stop optimization"""
        self.running = False


class MainWindow(QMainWindow):
    """Main window"""
    
    def __init__(self):
        super().__init__()
        self.opt_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle('TVC Rocket Trajectory Optimization')
        
        # Set window size
        window_width = 1400
        window_height = 900
        self.resize(window_width, window_height)
        
        # Remove any size restrictions to allow full maximization
        # QMainWindow should have maximize button by default, but ensure it's enabled
        # Don't set maximum size restrictions that would prevent maximization
        self.setMaximumSize(16777215, 16777215)  # Qt's maximum value, effectively unlimited
        
        # Center window on screen
        self.center_window()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: parameter settings
        left_panel = self.create_parameter_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel: display panel
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 2)
    
    def center_window(self):
        """Center the window on the screen"""
        try:
            # Get the QApplication instance
            app = QApplication.instance()
            if app is None:
                return
            
            # Get screen geometry
            try:
                # Try PyQt5 method
                screen = app.desktop().screenGeometry()
                screen_width = screen.width()
                screen_height = screen.height()
            except AttributeError:
                # Try PySide2 or newer PyQt5 method
                try:
                    screen = app.primaryScreen().geometry()
                    screen_width = screen.width()
                    screen_height = screen.height()
                except AttributeError:
                    # Fallback: use default screen size
                    screen_width = 1920
                    screen_height = 1080
            
            # Calculate center position
            window_width = self.width()
            window_height = self.height()
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            
            # Move window to center
            self.move(x, y)
        except Exception as e:
            # If centering fails, use default position
            print(f"Warning: Could not center window: {e}")
            self.move(100, 100)
        
    def create_parameter_panel(self):
        """Create parameter setting panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)  # Reduce spacing between widgets
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Title
        title = QLabel('Parameters')
        title.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(title)
        
        # Initial state - arrange in one row to save space
        init_group = QGroupBox('Initial State')
        init_layout = QGridLayout()
        init_layout.setSpacing(3)  # Reduce spacing
        
        self.x0_x = QDoubleSpinBox()
        self.x0_x.setRange(-100, 100)
        self.x0_x.setValue(0.0)
        self.x0_x.setDecimals(2)
        self.x0_x.setMaximumHeight(25)
        self.x0_x.setMaximumWidth(80)
        
        self.x0_y = QDoubleSpinBox()
        self.x0_y.setRange(-100, 100)
        self.x0_y.setValue(0.0)
        self.x0_y.setDecimals(2)
        self.x0_y.setMaximumHeight(25)
        self.x0_y.setMaximumWidth(80)
        
        self.x0_z = QDoubleSpinBox()
        self.x0_z.setRange(-100, 100)
        self.x0_z.setValue(0.0)
        self.x0_z.setDecimals(2)
        self.x0_z.setMaximumHeight(25)
        self.x0_z.setMaximumWidth(80)
        
        # Arrange in one row
        init_layout.addWidget(QLabel('X (m):'), 0, 0)
        init_layout.addWidget(self.x0_x, 0, 1)
        init_layout.addWidget(QLabel('Y (m):'), 0, 2)
        init_layout.addWidget(self.x0_y, 0, 3)
        init_layout.addWidget(QLabel('Z (m):'), 0, 4)
        init_layout.addWidget(self.x0_z, 0, 5)
        
        init_group.setLayout(init_layout)
        layout.addWidget(init_group)
        
        # Waypoints management
        waypoint_group = QGroupBox('Waypoints')
        waypoint_layout = QVBoxLayout()
        
        # Waypoint list widget
        if QT_AVAILABLE:
            try:
                from PyQt5.QtWidgets import QListWidget, QListWidgetItem
            except ImportError:
                from PySide2.QtWidgets import QListWidget, QListWidgetItem
        else:
            QListWidget = None
            QListWidgetItem = None
        
        self.waypoint_list = QListWidget()
        self.waypoint_list.setMaximumHeight(100)  # Reduce height
        waypoint_layout.addWidget(self.waypoint_list)
        
        # Buttons for waypoint management
        waypoint_btn_layout = QHBoxLayout()
        self.add_waypoint_btn = QPushButton('Add Waypoint')
        self.add_waypoint_btn.clicked.connect(self.add_waypoint)
        self.remove_waypoint_btn = QPushButton('Remove Selected')
        self.remove_waypoint_btn.clicked.connect(self.remove_waypoint)
        waypoint_btn_layout.addWidget(self.add_waypoint_btn)
        waypoint_btn_layout.addWidget(self.remove_waypoint_btn)
        waypoint_layout.addLayout(waypoint_btn_layout)
        
        # Current waypoint editor - arrange in 2 rows to save space
        current_wp_group = QGroupBox('Edit Waypoint')
        current_wp_layout = QGridLayout()
        current_wp_layout.setSpacing(3)  # Reduce spacing
        
        self.wp_x = QDoubleSpinBox()
        self.wp_x.setRange(-100, 100)
        self.wp_x.setValue(0.0)
        self.wp_x.setDecimals(2)
        self.wp_x.setMaximumHeight(25)
        self.wp_x.setMaximumWidth(80)
        
        self.wp_y = QDoubleSpinBox()
        self.wp_y.setRange(-100, 100)
        self.wp_y.setValue(0.0)
        self.wp_y.setDecimals(2)
        self.wp_y.setMaximumHeight(25)
        self.wp_y.setMaximumWidth(80)
        
        self.wp_z = QDoubleSpinBox()
        self.wp_z.setRange(-100, 100)
        self.wp_z.setValue(10.0)
        self.wp_z.setDecimals(2)
        self.wp_z.setMaximumHeight(25)
        self.wp_z.setMaximumWidth(80)
        
        self.wp_yaw = QDoubleSpinBox()
        self.wp_yaw.setRange(-180, 180)
        self.wp_yaw.setValue(0.0)
        self.wp_yaw.setDecimals(1)
        self.wp_yaw.setMaximumHeight(25)
        self.wp_yaw.setMaximumWidth(80)
        
        self.wp_time = QDoubleSpinBox()
        self.wp_time.setRange(0.0, 1000.0)
        self.wp_time.setValue(5.0)
        self.wp_time.setDecimals(2)
        self.wp_time.setMaximumHeight(25)
        self.wp_time.setMaximumWidth(80)
        
        # Two pairs per row (4 grid cols)
        current_wp_layout.addWidget(QLabel('X (m):'), 0, 0)
        current_wp_layout.addWidget(self.wp_x, 0, 1)
        current_wp_layout.addWidget(QLabel('Y (m):'), 0, 2)
        current_wp_layout.addWidget(self.wp_y, 0, 3)
        current_wp_layout.addWidget(QLabel('Z (m):'), 1, 0)
        current_wp_layout.addWidget(self.wp_z, 1, 1)
        current_wp_layout.addWidget(QLabel('Yaw (°):'), 1, 2)
        current_wp_layout.addWidget(self.wp_yaw, 1, 3)
        current_wp_layout.addWidget(QLabel('Arrival Time (s):'), 2, 0)
        current_wp_layout.addWidget(self.wp_time, 2, 1)
        
        self.update_waypoint_btn = QPushButton('Update Selected')
        self.update_waypoint_btn.clicked.connect(self.update_waypoint)
        self.update_waypoint_btn.setMaximumHeight(30)
        current_wp_layout.addWidget(self.update_waypoint_btn, 2, 2, 1, 2)
        
        current_wp_group.setLayout(current_wp_layout)
        waypoint_layout.addWidget(current_wp_group)
        
        # Connect waypoint list selection
        self.waypoint_list.itemSelectionChanged.connect(self.on_waypoint_selected)
        
        waypoint_group.setLayout(waypoint_layout)
        layout.addWidget(waypoint_group)
        
        # Initialize default waypoints: start (0,0,0,0°,0s), waypoint1 (0,0,10,0°,5s), waypoint2 (5,0,10,0°,10s)
        # Format: [x, y, z, yaw_deg, arrival_time]
        self.waypoints = [
            [0.0, 0.0, 0.0, 0.0, 0.0],      # Start at t=0, yaw=0°
            [0.0, 0.0, 10.0, 0.0, 5.0],      # Waypoint 1 at t=5s, yaw=0°
            [4.0, 0.0, 0.0, 0.0, 10.0]      # Waypoint 2 at t=10s, yaw=0°
        ]
        self.update_waypoint_list()
        
        # Optimization method selection
        method_group = QGroupBox('Optimization Method')
        method_layout = QHBoxLayout()
        method_layout.setSpacing(3)
        
        self.method_combo = QComboBox()
        self.method_combo.addItem('Method 1: Custom calcDiff (Slower, Numerical)')
        self.method_combo.addItem('Method 2: Pinocchio + FDDP (Penalty constraints)')
        self.method_combo.addItem('Method 3: Pinocchio + BoxFDDP (Native control bounds)')
        self.method_combo.addItem('Method 4: Acados (Native constraints)')
        self.method_combo.setCurrentIndex(0)  # Default to Method 1
        method_layout.addWidget(QLabel('Method:'))
        method_layout.addWidget(self.method_combo)
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        
        self.unified_checkbox = QCheckBox('Unified (merge all segments)')
        self.unified_checkbox.setToolTip('Merge all waypoint segments into one optimization problem (Method 2/3/4)')
        self.unified_checkbox.setChecked(False)
        self.unified_checkbox.setEnabled(False)  # Enabled only when Method 2/3 selected
        method_layout.addWidget(self.unified_checkbox)
        self.method_combo.currentIndexChanged.connect(self._update_unified_checkbox_state)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        self._update_unified_checkbox_state(self.method_combo.currentIndex())
        
        # Optimization parameters
        opt_group = QGroupBox('Optimization Parameters')
        opt_layout = QGridLayout()
        opt_layout.setSpacing(3)  # Reduce spacing
        
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 0.1)
        self.dt_spin.setValue(0.05)
        self.dt_spin.setSingleStep(0.01)
        self.dt_spin.setDecimals(3)
        self.dt_spin.setMaximumHeight(25)
        self.dt_spin.setMaximumWidth(100)
        
        self.N_spin = QSpinBox()
        self.N_spin.setRange(10, 500)
        self.N_spin.setValue(100)
        self.N_spin.setMaximumHeight(25)
        self.N_spin.setMaximumWidth(100)
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setValue(100)
        self.max_iter_spin.setMaximumHeight(25)
        self.max_iter_spin.setMaximumWidth(100)
        
        # Two pairs per row (4 grid cols)
        opt_layout.addWidget(QLabel('Time Step (s):'), 0, 0)
        opt_layout.addWidget(self.dt_spin, 0, 1)
        opt_layout.addWidget(QLabel('Time Steps:'), 0, 2)
        opt_layout.addWidget(self.N_spin, 0, 3)
        opt_layout.addWidget(QLabel('Max Iter:'), 1, 0)
        opt_layout.addWidget(self.max_iter_spin, 1, 1)
        
        opt_group.setLayout(opt_layout)
        
        # Tab widget for parameters
        params_tabs = QTabWidget()
        
        # Tab 1: Optimization Parameters
        opt_tab = QWidget()
        opt_tab_layout = QVBoxLayout()
        opt_tab_layout.addWidget(opt_group)
        opt_tab_layout.addStretch()
        opt_tab.setLayout(opt_tab_layout)
        params_tabs.addTab(opt_tab, 'Optimization')
        
        # Running cost weights
        cost_group = QGroupBox('Running Cost')
        cost_layout = QGridLayout()
        cost_layout.setSpacing(3)  # Reduce spacing
        
        self.w_p = QDoubleSpinBox()
        self.w_p.setRange(0, 1000)
        self.w_p.setValue(1.0)
        self.w_p.setDecimals(3)
        self.w_p.setMaximumHeight(25)
        self.w_p.setMaximumWidth(100)  # Reduce width
        
        self.w_v = QDoubleSpinBox()
        self.w_v.setRange(0, 1000)
        self.w_v.setValue(0.2)
        self.w_v.setDecimals(3)
        self.w_v.setMaximumHeight(25)
        self.w_v.setMaximumWidth(100)
        
        self.w_R = QDoubleSpinBox()
        self.w_R.setRange(0, 1000)
        self.w_R.setValue(0.5)
        self.w_R.setDecimals(3)
        self.w_R.setMaximumHeight(25)
        self.w_R.setMaximumWidth(100)
        
        self.w_yaw = QDoubleSpinBox()
        self.w_yaw.setRange(0, 1000)
        self.w_yaw.setValue(0.5)
        self.w_yaw.setDecimals(3)
        self.w_yaw.setMaximumHeight(25)
        self.w_yaw.setMaximumWidth(100)
        self.w_yaw.setToolTip('Separate weight for yaw (default same as Attitude R)')
        
        self.w_w = QDoubleSpinBox()
        self.w_w.setRange(0, 1000)
        self.w_w.setValue(0.1)
        self.w_w.setDecimals(3)
        self.w_w.setMaximumHeight(25)
        self.w_w.setMaximumWidth(100)
        
        self.w_u = QDoubleSpinBox()
        self.w_u.setRange(0, 1)
        self.w_u.setValue(0.01)
        self.w_u.setDecimals(3)
        self.w_u.setMaximumHeight(25)
        self.w_u.setMaximumWidth(100)
        
        self.w_du = QDoubleSpinBox()
        self.w_du.setRange(0, 1)
        self.w_du.setValue(0.1)  # Default: enable control rate penalty
        self.w_du.setDecimals(3)
        self.w_du.setMaximumHeight(25)
        self.w_du.setMaximumWidth(100)
        
        # Constraint penalty coefficients
        self.k_bound = QDoubleSpinBox()
        self.k_bound.setRange(0, 10000)
        self.k_bound.setValue(200.0)
        self.k_bound.setDecimals(1)
        self.k_bound.setMaximumHeight(25)
        self.k_bound.setMaximumWidth(100)
        
        self.k_state_bound = QDoubleSpinBox()
        self.k_state_bound.setRange(0, 10000)
        self.k_state_bound.setValue(20.0)  # Lower default to avoid constraint gradient dominating position cost
        self.k_state_bound.setDecimals(1)
        self.k_state_bound.setMaximumHeight(25)
        self.k_state_bound.setMaximumWidth(100)
        
        # Two pairs per row (4 grid cols)
        cost_layout.addWidget(QLabel('Position (p):'), 0, 0)
        cost_layout.addWidget(self.w_p, 0, 1)
        cost_layout.addWidget(QLabel('Velocity (v):'), 0, 2)
        cost_layout.addWidget(self.w_v, 0, 3)
        cost_layout.addWidget(QLabel('Attitude (R):'), 1, 0)
        cost_layout.addWidget(self.w_R, 1, 1)
        cost_layout.addWidget(QLabel('Yaw:'), 1, 2)
        cost_layout.addWidget(self.w_yaw, 1, 3)
        cost_layout.addWidget(QLabel('Ang Vel (w):'), 2, 0)
        cost_layout.addWidget(self.w_w, 2, 1)
        cost_layout.addWidget(QLabel('Control (u):'), 3, 0)
        cost_layout.addWidget(self.w_u, 3, 1)
        cost_layout.addWidget(QLabel('Ctrl Chg (du):'), 3, 2)
        cost_layout.addWidget(self.w_du, 3, 3)
        cost_layout.addWidget(QLabel('Ctrl Bound (k_bound):'), 4, 0)
        cost_layout.addWidget(self.k_bound, 4, 1)
        cost_layout.addWidget(QLabel('State Bound (k_sb):'), 4, 2)
        cost_layout.addWidget(self.k_state_bound, 4, 3)
        self.schedule_ref_checkbox = QCheckBox('On-time arrival (schedule_ref)')
        self.schedule_ref_checkbox.setChecked(True)
        self.schedule_ref_checkbox.setToolTip('Checked: time-interpolated ref for on-time arrival; unchecked: constant goal ref, may arrive early')
        cost_layout.addWidget(self.schedule_ref_checkbox, 5, 0, 1, 4)
        
        # Acados: first-order actuator dynamics (per-channel time constants)
        self.actuator_dynamics_checkbox = QCheckBox('Actuator dynamics (first-order) [Acados]')
        self.actuator_dynamics_checkbox.setChecked(False)
        self.actuator_dynamics_checkbox.setToolTip('When enabled: model actuator as tau*u_dot = u_cmd - u_actual per channel. Acados only.')
        cost_layout.addWidget(self.actuator_dynamics_checkbox, 6, 0, 1, 4)
        self.tau_pitch_spin = QDoubleSpinBox()
        self.tau_pitch_spin.setRange(0.001, 1.0)
        self.tau_pitch_spin.setValue(0.05)
        self.tau_pitch_spin.setDecimals(3)
        self.tau_pitch_spin.setSingleStep(0.01)
        self.tau_pitch_spin.setMaximumHeight(25)
        self.tau_pitch_spin.setMaximumWidth(80)
        self.tau_pitch_spin.setToolTip('Pitch channel time constant (s)')
        self.tau_roll_spin = QDoubleSpinBox()
        self.tau_roll_spin.setRange(0.001, 1.0)
        self.tau_roll_spin.setValue(0.05)
        self.tau_roll_spin.setDecimals(3)
        self.tau_roll_spin.setSingleStep(0.01)
        self.tau_roll_spin.setMaximumHeight(25)
        self.tau_roll_spin.setMaximumWidth(80)
        self.tau_roll_spin.setToolTip('Roll channel time constant (s)')
        self.tau_T_spin = QDoubleSpinBox()
        self.tau_T_spin.setRange(0.001, 1.0)
        self.tau_T_spin.setValue(0.05)
        self.tau_T_spin.setDecimals(3)
        self.tau_T_spin.setSingleStep(0.01)
        self.tau_T_spin.setMaximumHeight(25)
        self.tau_T_spin.setMaximumWidth(80)
        self.tau_T_spin.setToolTip('Thrust channel time constant (s)')
        self.tau_yaw_spin = QDoubleSpinBox()
        self.tau_yaw_spin.setRange(0.001, 1.0)
        self.tau_yaw_spin.setValue(0.05)
        self.tau_yaw_spin.setDecimals(3)
        self.tau_yaw_spin.setSingleStep(0.01)
        self.tau_yaw_spin.setMaximumHeight(25)
        self.tau_yaw_spin.setMaximumWidth(80)
        self.tau_yaw_spin.setToolTip('Yaw channel time constant (s)')
        cost_layout.addWidget(QLabel('tau pitch (s):'), 7, 0)
        cost_layout.addWidget(self.tau_pitch_spin, 7, 1)
        cost_layout.addWidget(QLabel('tau roll (s):'), 7, 2)
        cost_layout.addWidget(self.tau_roll_spin, 7, 3)
        cost_layout.addWidget(QLabel('tau T (s):'), 8, 0)
        cost_layout.addWidget(self.tau_T_spin, 8, 1)
        cost_layout.addWidget(QLabel('tau yaw (s):'), 8, 2)
        cost_layout.addWidget(self.tau_yaw_spin, 8, 3)
        
        cost_group.setLayout(cost_layout)
        
        # Terminal cost weights - applied at waypoint arrival
        terminal_cost_group = QGroupBox('Terminal Cost')
        terminal_cost_layout = QGridLayout()
        terminal_cost_layout.setSpacing(3)
        
        self.w_p_term = QDoubleSpinBox()
        self.w_p_term.setRange(0, 10000)
        self.w_p_term.setValue(200.0)
        self.w_p_term.setDecimals(1)
        self.w_p_term.setMaximumHeight(25)
        self.w_p_term.setMaximumWidth(100)
        
        self.w_v_term = QDoubleSpinBox()
        self.w_v_term.setRange(0, 10000)
        self.w_v_term.setValue(50.0)
        self.w_v_term.setDecimals(1)
        self.w_v_term.setMaximumHeight(25)
        self.w_v_term.setMaximumWidth(100)
        
        self.w_R_term = QDoubleSpinBox()
        self.w_R_term.setRange(0, 10000)
        self.w_R_term.setValue(200.0)
        self.w_R_term.setDecimals(1)
        self.w_R_term.setMaximumHeight(25)
        self.w_R_term.setMaximumWidth(100)
        
        self.w_yaw_term = QDoubleSpinBox()
        self.w_yaw_term.setRange(0, 10000)
        self.w_yaw_term.setValue(200.0)
        self.w_yaw_term.setDecimals(1)
        self.w_yaw_term.setMaximumHeight(25)
        self.w_yaw_term.setMaximumWidth(100)
        self.w_yaw_term.setToolTip('Terminal yaw weight')
        
        self.w_w_term = QDoubleSpinBox()
        self.w_w_term.setRange(0, 10000)
        self.w_w_term.setValue(20.0)
        self.w_w_term.setDecimals(1)
        self.w_w_term.setMaximumHeight(25)
        self.w_w_term.setMaximumWidth(100)
        
        self.w_u_term = QDoubleSpinBox()
        self.w_u_term.setRange(0, 10)
        self.w_u_term.setValue(0.0)
        self.w_u_term.setDecimals(2)
        self.w_u_term.setMaximumHeight(25)
        self.w_u_term.setMaximumWidth(100)
        
        self.w_du_term = QDoubleSpinBox()
        self.w_du_term.setRange(0, 10)
        self.w_du_term.setValue(0.0)
        self.w_du_term.setDecimals(2)
        self.w_du_term.setMaximumHeight(25)
        self.w_du_term.setMaximumWidth(100)
        
        terminal_cost_layout.addWidget(QLabel('Position (p):'), 0, 0)
        terminal_cost_layout.addWidget(self.w_p_term, 0, 1)
        terminal_cost_layout.addWidget(QLabel('Velocity (v):'), 0, 2)
        terminal_cost_layout.addWidget(self.w_v_term, 0, 3)
        terminal_cost_layout.addWidget(QLabel('Attitude (R):'), 1, 0)
        terminal_cost_layout.addWidget(self.w_R_term, 1, 1)
        terminal_cost_layout.addWidget(QLabel('Yaw:'), 1, 2)
        terminal_cost_layout.addWidget(self.w_yaw_term, 1, 3)
        terminal_cost_layout.addWidget(QLabel('Ang Vel (w):'), 2, 0)
        terminal_cost_layout.addWidget(self.w_w_term, 2, 1)
        terminal_cost_layout.addWidget(QLabel('Control (u):'), 3, 0)
        terminal_cost_layout.addWidget(self.w_u_term, 3, 1)
        terminal_cost_layout.addWidget(QLabel('Ctrl Chg (du):'), 3, 2)
        terminal_cost_layout.addWidget(self.w_du_term, 3, 3)
        terminal_cost_layout.addWidget(QLabel('Terminal Scale:'), 4, 0)
        self.terminal_scale_spin = QDoubleSpinBox()
        self.terminal_scale_spin.setRange(1, 10000)
        self.terminal_scale_spin.setValue(100.0)
        self.terminal_scale_spin.setDecimals(0)
        self.terminal_scale_spin.setMaximumHeight(25)
        self.terminal_scale_spin.setMaximumWidth(100)
        self.terminal_scale_spin.setToolTip('Scale up terminal cost weights for target achievement. Rule: >> N to dominate running cost. Acados recommends 100~1000')
        terminal_cost_layout.addWidget(self.terminal_scale_spin, 4, 1)
        self.terminal_constraint_checkbox = QCheckBox('Terminal position equality (p_N=p_g)')
        self.terminal_constraint_checkbox.setChecked(False)
        self.terminal_constraint_checkbox.setToolTip('Enforce exact terminal position. May cause infeasibility, use with caution')
        terminal_cost_layout.addWidget(self.terminal_constraint_checkbox, 4, 2, 1, 2)
        self.waypoint_terminal_checkbox = QCheckBox('Waypoint terminal cost (segment mode for multi-WP)')
        self.waypoint_terminal_checkbox.setChecked(True)
        self.waypoint_terminal_checkbox.setToolTip('Checked: terminal cost at each intermediate waypoint for arrival. Acados uses segment mode')
        terminal_cost_layout.addWidget(self.waypoint_terminal_checkbox, 5, 0, 1, 4)
        
        terminal_cost_group.setLayout(terminal_cost_layout)
        
        # Tab 2: Cost Weights
        cost_tab = QWidget()
        cost_tab_layout = QVBoxLayout()
        cost_tab_layout.addWidget(cost_group)
        cost_tab_layout.addWidget(terminal_cost_group)
        cost_tab_layout.addStretch()
        cost_tab.setLayout(cost_tab_layout)
        params_tabs.addTab(cost_tab, 'Cost Weights')
        
        # Control constraints
        bounds_group = QGroupBox('Control Constraints')
        bounds_layout = QGridLayout()
        
        self.th_p_max = QDoubleSpinBox()
        self.th_p_max.setRange(0, 90)  # Range in degrees
        self.th_p_max.setValue(10.0)  # 10 degrees
        self.th_p_max.setDecimals(1)
        self.th_p_max.setMaximumHeight(25)
        self.th_p_max.setMaximumWidth(100)
        
        self.th_r_max = QDoubleSpinBox()
        self.th_r_max.setRange(0, 90)  # Range in degrees
        self.th_r_max.setValue(10.0)  # 10 degrees
        self.th_r_max.setDecimals(1)
        self.th_r_max.setMaximumHeight(25)
        self.th_r_max.setMaximumWidth(100)
        
        self.T_max = QDoubleSpinBox()
        self.T_max.setRange(0, 100)
        self.T_max.setValue(25.0)
        self.T_max.setDecimals(2)
        self.T_max.setMaximumHeight(25)
        self.T_max.setMaximumWidth(100)
        
        self.tau_yaw_max = QDoubleSpinBox()
        self.tau_yaw_max.setRange(0, 10)
        self.tau_yaw_max.setValue(1.0)
        self.tau_yaw_max.setDecimals(2)
        self.tau_yaw_max.setMaximumHeight(25)
        self.tau_yaw_max.setMaximumWidth(100)
        
        bounds_layout.setSpacing(3)  # Reduce spacing
        # Two pairs per row (4 grid cols)
        bounds_layout.addWidget(QLabel('TVC Pitch (°):'), 0, 0)
        bounds_layout.addWidget(self.th_p_max, 0, 1)
        bounds_layout.addWidget(QLabel('TVC Roll (°):'), 0, 2)
        bounds_layout.addWidget(self.th_r_max, 0, 3)
        bounds_layout.addWidget(QLabel('Thrust (N):'), 1, 0)
        bounds_layout.addWidget(self.T_max, 1, 1)
        bounds_layout.addWidget(QLabel('Yaw (N·m):'), 1, 2)
        bounds_layout.addWidget(self.tau_yaw_max, 1, 3)
        
        bounds_group.setLayout(bounds_layout)
        
        # Tab 3: Control Constraints
        bounds_tab = QWidget()
        bounds_tab_layout = QVBoxLayout()
        bounds_tab_layout.addWidget(bounds_group)
        bounds_tab_layout.addStretch()
        bounds_tab.setLayout(bounds_tab_layout)
        params_tabs.addTab(bounds_tab, 'Control Constraints')
        
        # State constraints
        state_constraints_group = QGroupBox('State Constraints')
        state_constraints_layout = QGridLayout()
        state_constraints_layout.setSpacing(3)
        
        # Velocity constraints (horizontal and vertical)
        self.v_horizontal_max = QDoubleSpinBox()
        self.v_horizontal_max.setRange(0, 100)
        self.v_horizontal_max.setValue(1.0)
        self.v_horizontal_max.setDecimals(1)
        self.v_horizontal_max.setMaximumHeight(25)
        self.v_horizontal_max.setMaximumWidth(100)
        
        self.v_vertical_max = QDoubleSpinBox()
        self.v_vertical_max.setRange(0, 100)
        self.v_vertical_max.setValue(3.0)
        self.v_vertical_max.setDecimals(1)
        self.v_vertical_max.setMaximumHeight(25)
        self.v_vertical_max.setMaximumWidth(100)
        
        # Euler angle constraints (in degrees)
        self.roll_max = QDoubleSpinBox()
        self.roll_max.setRange(0, 180)
        self.roll_max.setValue(10.0)
        self.roll_max.setDecimals(1)
        self.roll_max.setMaximumHeight(25)
        self.roll_max.setMaximumWidth(100)
        
        self.pitch_max = QDoubleSpinBox()
        self.pitch_max.setRange(0, 180)
        self.pitch_max.setValue(10.0)
        self.pitch_max.setDecimals(1)
        self.pitch_max.setMaximumHeight(25)
        self.pitch_max.setMaximumWidth(100)
        
        self.yaw_max = QDoubleSpinBox()
        self.yaw_max.setRange(0, 180)
        self.yaw_max.setValue(180.0)
        self.yaw_max.setDecimals(1)
        self.yaw_max.setMaximumHeight(25)
        self.yaw_max.setMaximumWidth(100)
        
        # Angular velocity constraint
        self.w_max = QDoubleSpinBox()
        self.w_max.setRange(0, 10)
        self.w_max.setValue(2.0)
        self.w_max.setDecimals(2)
        self.w_max.setMaximumHeight(25)
        self.w_max.setMaximumWidth(100)
        
        # Two pairs per row (4 grid cols)
        state_constraints_layout.addWidget(QLabel('V_xy (m/s):'), 0, 0)
        state_constraints_layout.addWidget(self.v_horizontal_max, 0, 1)
        state_constraints_layout.addWidget(QLabel('V_z (m/s):'), 0, 2)
        state_constraints_layout.addWidget(self.v_vertical_max, 0, 3)
        state_constraints_layout.addWidget(QLabel('Roll (°):'), 1, 0)
        state_constraints_layout.addWidget(self.roll_max, 1, 1)
        state_constraints_layout.addWidget(QLabel('Pitch (°):'), 1, 2)
        state_constraints_layout.addWidget(self.pitch_max, 1, 3)
        state_constraints_layout.addWidget(QLabel('Yaw (°):'), 2, 0)
        state_constraints_layout.addWidget(self.yaw_max, 2, 1)
        state_constraints_layout.addWidget(QLabel('Ang Vel (rad/s):'), 2, 2)
        state_constraints_layout.addWidget(self.w_max, 2, 3)
        
        state_constraints_group.setLayout(state_constraints_layout)
        
        # Tab 4: State Constraints
        state_tab = QWidget()
        state_tab_layout = QVBoxLayout()
        state_tab_layout.addWidget(state_constraints_group)
        state_tab_layout.addStretch()
        state_tab.setLayout(state_tab_layout)
        params_tabs.addTab(state_tab, 'State Constraints')
        
        # Physical parameters
        physics_group = QGroupBox('Physical Parameters')
        physics_layout = QGridLayout()
        physics_layout.setSpacing(3)
        
        # Mass
        self.mass = QDoubleSpinBox()
        self.mass.setRange(0.01, 10.0)
        self.mass.setValue(0.6)
        self.mass.setDecimals(3)
        self.mass.setMaximumHeight(25)
        self.mass.setMaximumWidth(100)
        
        # Moment of inertia (diagonal components)
        self.Ixx = QDoubleSpinBox()
        self.Ixx.setRange(0.0001, 1.0)
        self.Ixx.setValue(0.02)
        self.Ixx.setDecimals(2)
        self.Ixx.setMaximumHeight(25)
        self.Ixx.setMaximumWidth(100)
        
        self.Iyy = QDoubleSpinBox()
        self.Iyy.setRange(0.0001, 1.0)
        self.Iyy.setValue(0.02)
        self.Iyy.setDecimals(2)
        self.Iyy.setMaximumHeight(25)
        self.Iyy.setMaximumWidth(100)
        
        self.Izz = QDoubleSpinBox()
        self.Izz.setRange(0.0001, 1.0)
        self.Izz.setValue(0.01)
        self.Izz.setDecimals(2)
        self.Izz.setMaximumHeight(25)
        self.Izz.setMaximumWidth(100)
        
        # Thrust position (r_thrust)
        self.r_thrust_x = QDoubleSpinBox()
        self.r_thrust_x.setRange(-1.0, 1.0)
        self.r_thrust_x.setValue(0.0)
        self.r_thrust_x.setDecimals(2)
        self.r_thrust_x.setMaximumHeight(25)
        self.r_thrust_x.setMaximumWidth(100)
        
        self.r_thrust_y = QDoubleSpinBox()
        self.r_thrust_y.setRange(-1.0, 1.0)
        self.r_thrust_y.setValue(0.0)
        self.r_thrust_y.setDecimals(2)
        self.r_thrust_y.setMaximumHeight(25)
        self.r_thrust_y.setMaximumWidth(100)
        
        self.r_thrust_z = QDoubleSpinBox()
        self.r_thrust_z.setRange(-1.0, 1.0)
        self.r_thrust_z.setValue(-0.2)
        self.r_thrust_z.setDecimals(2)
        self.r_thrust_z.setMaximumHeight(25)
        self.r_thrust_z.setMaximumWidth(100)
        
        # Two pairs per row (4 grid cols)
        physics_layout.addWidget(QLabel('Mass (kg):'), 0, 0)
        physics_layout.addWidget(self.mass, 0, 1)
        physics_layout.addWidget(QLabel('Ixx (kg·m²):'), 0, 2)
        physics_layout.addWidget(self.Ixx, 0, 3)
        physics_layout.addWidget(QLabel('Iyy (kg·m²):'), 1, 0)
        physics_layout.addWidget(self.Iyy, 1, 1)
        physics_layout.addWidget(QLabel('Izz (kg·m²):'), 1, 2)
        physics_layout.addWidget(self.Izz, 1, 3)
        physics_layout.addWidget(QLabel('Thrust X (m):'), 2, 0)
        physics_layout.addWidget(self.r_thrust_x, 2, 1)
        physics_layout.addWidget(QLabel('Thrust Y (m):'), 2, 2)
        physics_layout.addWidget(self.r_thrust_y, 2, 3)
        physics_layout.addWidget(QLabel('Thrust Z (m):'), 3, 0)
        physics_layout.addWidget(self.r_thrust_z, 3, 1)
        
        physics_group.setLayout(physics_layout)
        
        # Tab 5: Physical Parameters
        physics_tab = QWidget()
        physics_tab_layout = QVBoxLayout()
        physics_tab_layout.addWidget(physics_group)
        physics_tab_layout.addStretch()
        physics_tab.setLayout(physics_tab_layout)
        params_tabs.addTab(physics_tab, 'Physical')
        
        layout.addWidget(params_tabs)
        
        # Default parameters per method (state/control constraints equal across all methods)
        self.DEFAULT_PARAMS = {
            0: {  # Method 1: Custom calcDiff
                "w_p": 1.0, "w_v": 0.2, "w_R": 0.5, "w_yaw": 0.5, "w_w": 0.1,
                "w_u": 0.5, "w_du": 0.5,
                "k_bound": 200.0, "k_state_bound": 200.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 1.0, "v_vertical_max": 3.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 180.0, "w_max": 2.0,
            },
            1: {  # Method 2: FDDP - higher w_R and w_yaw to suppress yaw drift (Pinocchio cost structure)
                "w_p": 1.0, "w_v": 0.2, "w_R": 2.0, "w_yaw": 2.0, "w_w": 0.1,
                "w_u": 0.5, "w_du": 0.5,
                "k_bound": 200.0, "k_state_bound": 200.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 1.0, "v_vertical_max": 3.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
            2: {  # Method 3: BoxFDDP - higher w_R and w_yaw to suppress yaw drift
                "w_p": 1.0, "w_v": 0.2, "w_R": 2.0, "w_yaw": 2.0, "w_w": 0.1,
                "w_u": 0.5, "w_du": 0.5,
                "k_bound": 200.0, "k_state_bound": 200.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 1.0, "v_vertical_max": 3.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
            3: {  # Method 4: Acados - native constraints
                "w_p": 1.0, "w_v": 0.2, "w_R": 0.5, "w_yaw": 0.5, "w_w": 0.1,
                "actuator_dynamics": False, "actuator_tau": [0.05, 0.05, 0.05, 0.05],
                "w_u": 0.5, "w_du": 0.5, "schedule_ref": True,
                "terminal_scale": 100.0, "terminal_constraint": False, "waypoint_terminal_cost": True,
                "k_bound": 200.0, "k_state_bound": 20.0,
                "th_p_max": 10.0, "th_r_max": 10.0, "T_max": 25.0, "tau_yaw_max": 1.0,
                "v_horizontal_max": 2.5, "v_vertical_max": 2.0,
                "roll_max": 10.0, "pitch_max": 10.0, "yaw_max": 30.0, "w_max": 2.0,
            },
        }
        
        # Control buttons - arrange in one row
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton('Start Optimization')
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.run_btn.setMaximumHeight(35)  # Reduce button height
        self.run_btn.clicked.connect(self.start_optimization)
        button_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton('Stop Optimization')
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        self.stop_btn.setMaximumHeight(35)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_optimization)
        button_layout.addWidget(self.stop_btn)
        self.show_plot_checkbox = QCheckBox('Show full trajectory plot when done')
        self.show_plot_checkbox.setChecked(False)
        self.show_plot_checkbox.setToolTip('If checked, a detailed trajectory plot window will open after optimization completes')
        button_layout.addWidget(self.show_plot_checkbox)
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(20)  # Reduce height
        layout.addWidget(self.progress)
        
        # Status information
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)  # Reduce height
        self.status_text.setReadOnly(True)
        layout.addWidget(QLabel('Status:'))
        layout.addWidget(self.status_text)
        
        layout.addStretch()
        
        return panel
    
    def create_display_panel(self):
        """Create display panel - all states, controls and cost on one page"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create single canvas with all subplots
        self.fig = Figure(figsize=(20, 10.5))
        self.canvas = FigureCanvas(self.fig)
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.35, wspace=0.3)
        self.fig.suptitle('TVC Rocket Trajectory Optimization', 
                         fontsize=16, fontweight='bold', y=0.995)
        
        # First row: 3D trajectory and convergence curve
        # 1. 3D position trajectory (occupies 2 positions)
        self.ax_3d = self.fig.add_subplot(gs[0, 0:2], projection='3d')
        self.ax_3d.set_xlabel('X (m)', fontsize=10)
        self.ax_3d.set_ylabel('Y (m)', fontsize=10)
        self.ax_3d.set_zlabel('Z (m)', fontsize=10)
        self.ax_3d.set_title('3D Position Trajectory', fontsize=11, fontweight='bold')
        self.ax_3d.grid(True, alpha=0.3)
        
        # 2. Cost convergence curve (occupies 2 positions)
        self.ax_cost = self.fig.add_subplot(gs[0, 2:4])
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        
        # Second row: position states
        # 3. Position
        self.ax_pos = self.fig.add_subplot(gs[1, 0])
        self.ax_pos.set_xlabel('Time (s)', fontsize=9)
        self.ax_pos.set_ylabel('Position (m)', fontsize=9)
        self.ax_pos.set_title('Position', fontsize=10, fontweight='bold')
        self.ax_pos.grid(True, alpha=0.3)
        
        # 4. Velocity
        self.ax_vel = self.fig.add_subplot(gs[1, 1])
        self.ax_vel.set_xlabel('Time (s)', fontsize=9)
        self.ax_vel.set_ylabel('Velocity (m/s)', fontsize=9)
        self.ax_vel.set_title('Linear Velocity', fontsize=10, fontweight='bold')
        self.ax_vel.grid(True, alpha=0.3)
        
        # 5. Angular velocity
        self.ax_angvel = self.fig.add_subplot(gs[1, 2])
        self.ax_angvel.set_xlabel('Time (s)', fontsize=9)
        self.ax_angvel.set_ylabel('Angular Vel (rad/s)', fontsize=9)
        self.ax_angvel.set_title('Angular Velocity', fontsize=10, fontweight='bold')
        self.ax_angvel.grid(True, alpha=0.3)
        
        # 6. Euler angles
        self.ax_euler = self.fig.add_subplot(gs[1, 3])
        self.ax_euler.set_xlabel('Time (s)', fontsize=9)
        self.ax_euler.set_ylabel('Euler Angles (deg)', fontsize=9)
        self.ax_euler.set_title('Attitude (Euler)', fontsize=10, fontweight='bold')
        self.ax_euler.grid(True, alpha=0.3)
        
        # Third row: control inputs
        # 7. TVC Pitch angle
        self.ax_pitch = self.fig.add_subplot(gs[2, 0])
        self.ax_pitch.set_xlabel('Time (s)', fontsize=9)
        self.ax_pitch.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_pitch.set_title('TVC Pitch Angle', fontsize=10, fontweight='bold')
        self.ax_pitch.grid(True, alpha=0.3)
        
        # 8. TVC Roll angle
        self.ax_roll = self.fig.add_subplot(gs[2, 1])
        self.ax_roll.set_xlabel('Time (s)', fontsize=9)
        self.ax_roll.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_roll.set_title('TVC Roll Angle', fontsize=10, fontweight='bold')
        self.ax_roll.grid(True, alpha=0.3)
        
        # 9. Thrust
        self.ax_thrust = self.fig.add_subplot(gs[2, 2])
        self.ax_thrust.set_xlabel('Time (s)', fontsize=9)
        self.ax_thrust.set_ylabel('Thrust (N)', fontsize=9)
        self.ax_thrust.set_title('Thrust', fontsize=10, fontweight='bold')
        self.ax_thrust.grid(True, alpha=0.3)
        
        # 10. Yaw torque
        self.ax_yaw = self.fig.add_subplot(gs[2, 3])
        self.ax_yaw.set_xlabel('Time (s)', fontsize=9)
        self.ax_yaw.set_ylabel('Torque (N·m)', fontsize=9)
        self.ax_yaw.set_title('Yaw Torque', fontsize=10, fontweight='bold')
        self.ax_yaw.grid(True, alpha=0.3)
        
        layout.addWidget(self.canvas)
        
        # Data storage
        self.iterations = []
        self.costs = []
        self.stops = []
        self.current_xs = None
        self.current_us = None
        # Multi-segment cost tracking
        self.segment_costs = {}  # {segment_idx: [costs]}
        self.segment_iterations = {}  # {segment_idx: [iterations]}
        self.current_segment_idx = 0
        
        return panel
    
    def quat_to_euler(self, q):
        """Convert quaternion to Euler angles (ZYX order), q=[w,x,y,z]"""
        return quat_to_euler(q, format='wxyz')
    
    def yaw_to_quaternion(self, yaw_deg):
        """Convert yaw angle (degrees) to quaternion [w, x, y, z], roll=0, pitch=0"""
        return yaw_to_quaternion(yaw_deg)
    
    def _update_unified_checkbox_state(self, index):
        """Enable unified checkbox only for Method 2/3"""
        self.unified_checkbox.setEnabled(index >= 1)
        if index == 0:
            self.unified_checkbox.setChecked(False)
    
    def on_method_changed(self, index):
        """Load parameter defaults when optimization method changes"""
        params = self.DEFAULT_PARAMS.get(index, self.DEFAULT_PARAMS[0])
        self.w_p.setValue(params["w_p"])
        self.w_v.setValue(params["w_v"])
        self.w_R.setValue(params["w_R"])
        self.w_yaw.setValue(params.get("w_yaw", params["w_R"]))
        self.w_w.setValue(params["w_w"])
        self.w_u.setValue(params["w_u"])
        self.w_du.setValue(params["w_du"])
        self.k_bound.setValue(params["k_bound"])
        self.k_state_bound.setValue(params["k_state_bound"])
        self.th_p_max.setValue(params["th_p_max"])
        self.th_r_max.setValue(params["th_r_max"])
        self.T_max.setValue(params["T_max"])
        self.tau_yaw_max.setValue(params["tau_yaw_max"])
        self.v_horizontal_max.setValue(params["v_horizontal_max"])
        self.v_vertical_max.setValue(params["v_vertical_max"])
        self.roll_max.setValue(params["roll_max"])
        self.pitch_max.setValue(params["pitch_max"])
        self.yaw_max.setValue(params["yaw_max"])
        self.w_max.setValue(params["w_max"])
        if "schedule_ref" in params:
            self.schedule_ref_checkbox.setChecked(params["schedule_ref"])
        if "terminal_scale" in params:
            self.terminal_scale_spin.setValue(params["terminal_scale"])
        if "terminal_constraint" in params:
            self.terminal_constraint_checkbox.setChecked(params["terminal_constraint"])
        if "waypoint_terminal_cost" in params:
            self.waypoint_terminal_checkbox.setChecked(params["waypoint_terminal_cost"])
        if "actuator_dynamics" in params:
            self.actuator_dynamics_checkbox.setChecked(params["actuator_dynamics"])
        if "actuator_tau" in params:
            tau = params["actuator_tau"]
            if len(tau) >= 4:
                self.tau_pitch_spin.setValue(tau[0])
                self.tau_roll_spin.setValue(tau[1])
                self.tau_T_spin.setValue(tau[2])
                self.tau_yaw_spin.setValue(tau[3])
        method_names = ["Method 1 (Custom calcDiff)", "Method 2 (FDDP)", "Method 3 (BoxFDDP)", "Method 4 (Acados)"]
        self.status_text.append(f"Parameters loaded for {method_names[index] if index < len(method_names) else 'Unknown'}")
    
    def update_waypoint_list(self):
        """Update waypoint list display"""
        self.waypoint_list.clear()
        for i, wp in enumerate(self.waypoints):
            # Ensure waypoint has all required fields (for backward compatibility)
            # Format: [x, y, z, yaw_deg, time]
            if len(wp) < 5:
                # Old format: [x, y, z, time] -> add yaw=0
                if len(wp) == 4:
                    wp = [wp[0], wp[1], wp[2], 0.0, wp[3]]  # Insert yaw=0 before time
                else:
                    wp = list(wp) + [0.0] * (5 - len(wp))
            
            yaw = wp[3] if len(wp) > 3 else 0.0
            time = wp[4] if len(wp) > 4 else (wp[3] if len(wp) > 3 else 0.0)
            
            if i == 0:
                item_text = f"Start: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}] yaw={yaw:.1f}° @ t={time:.2f}s"
            else:
                item_text = f"WP {i}: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}] yaw={yaw:.1f}° @ t={time:.2f}s"
            try:
                from PyQt5.QtWidgets import QListWidgetItem
            except ImportError:
                from PySide2.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            self.waypoint_list.addItem(item)
    
    def add_waypoint(self):
        """Add a new waypoint"""
        new_wp = [self.wp_x.value(), self.wp_y.value(), self.wp_z.value(), 
                  self.wp_yaw.value(), self.wp_time.value()]
        self.waypoints.append(new_wp)
        self.update_waypoint_list()
        # Select the newly added waypoint
        self.waypoint_list.setCurrentRow(len(self.waypoints) - 1)
    
    def remove_waypoint(self):
        """Remove selected waypoint (cannot remove start point)"""
        current_row = self.waypoint_list.currentRow()
        if current_row >= 0 and current_row < len(self.waypoints):
            if current_row == 0:
                QMessageBox.warning(self, 'Warning', 'Cannot remove start point')
                return
            self.waypoints.pop(current_row)
            self.update_waypoint_list()
            # Select previous item if available
            if current_row > 0:
                self.waypoint_list.setCurrentRow(current_row - 1)
    
    def update_waypoint(self):
        """Update selected waypoint with current values"""
        current_row = self.waypoint_list.currentRow()
        if current_row >= 0 and current_row < len(self.waypoints):
            self.waypoints[current_row] = [self.wp_x.value(), self.wp_y.value(), self.wp_z.value(), 
                                          self.wp_yaw.value(), self.wp_time.value()]
            self.update_waypoint_list()
            self.waypoint_list.setCurrentRow(current_row)
    
    def on_waypoint_selected(self):
        """Handle waypoint selection"""
        current_row = self.waypoint_list.currentRow()
        if current_row >= 0 and current_row < len(self.waypoints):
            wp = self.waypoints[current_row]
            # Ensure waypoint has all required fields (for backward compatibility)
            # Format: [x, y, z, yaw_deg, time]
            if len(wp) < 5:
                # Old format: [x, y, z, time] -> add yaw=0
                if len(wp) == 4:
                    wp = [wp[0], wp[1], wp[2], 0.0, wp[3]]  # Insert yaw=0 before time
                else:
                    wp = list(wp) + [0.0] * (5 - len(wp))
            self.wp_x.setValue(wp[0])
            self.wp_y.setValue(wp[1])
            self.wp_z.setValue(wp[2])
            self.wp_yaw.setValue(wp[3] if len(wp) > 3 else 0.0)
            self.wp_time.setValue(wp[4] if len(wp) > 4 else (wp[3] if len(wp) > 3 else 0.0))
    
    def get_parameters(self):
        """Get optimization parameters"""
        # Initial state (use first waypoint as start) - kept for compatibility
        x0 = np.zeros(17)
        if len(self.waypoints) > 0:
            first_wp = self.waypoints[0]
            x0[0] = first_wp[0]
            x0[1] = first_wp[1]
            x0[2] = first_wp[2]
            # Convert yaw to quaternion
            yaw_deg = first_wp[3] if len(first_wp) > 3 else 0.0
            yaw_rad = np.radians(yaw_deg)
            x0[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])
        else:
            x0[0] = self.x0_x.value()
            x0[1] = self.x0_y.value()
            x0[2] = self.x0_z.value()
            x0[6:10] = np.array([1., 0., 0., 0.])  # Initial quaternion (default)
        
        # Target state (use last waypoint as goal) - kept for compatibility
        xg = np.zeros(17)
        if len(self.waypoints) > 0:
            last_wp = self.waypoints[-1]
            xg[0] = last_wp[0]
            xg[1] = last_wp[1]
            xg[2] = last_wp[2]
            # Convert yaw to quaternion
            yaw_deg = last_wp[3] if len(last_wp) > 3 else 0.0
            yaw_rad = np.radians(yaw_deg)
            xg[6:10] = np.array([np.cos(yaw_rad/2.0), 0.0, 0.0, np.sin(yaw_rad/2.0)])
        else:
            xg[0] = self.xg_x.value()
            xg[1] = self.xg_y.value()
            xg[2] = self.xg_z.value()
            xg[6:10] = np.array([1., 0., 0., 0.])  # Target quaternion (default)
        
        # Cost weights (running cost)
        weights = {
            "p": self.w_p.value(),
            "v": self.w_v.value(),
            "R": self.w_R.value(),
            "yaw": self.w_yaw.value(),
            "w": self.w_w.value(),
            "u": self.w_u.value(),
            "du": self.w_du.value(),
            "schedule_ref": self.schedule_ref_checkbox.isChecked(),
            "terminal_scale": self.terminal_scale_spin.value(),
            "terminal_constraint": self.terminal_constraint_checkbox.isChecked(),
            "waypoint_terminal_cost": self.waypoint_terminal_checkbox.isChecked(),
            "actuator_dynamics": self.actuator_dynamics_checkbox.isChecked(),
            "actuator_tau": [self.tau_pitch_spin.value(), self.tau_roll_spin.value(),
                            self.tau_T_spin.value(), self.tau_yaw_spin.value()]
        }
        
        # Terminal cost weights
        terminal_weights = {
            "p": self.w_p_term.value(),
            "v": self.w_v_term.value(),
            "R": self.w_R_term.value(),
            "yaw": self.w_yaw_term.value(),
            "w": self.w_w_term.value(),
            "u": self.w_u_term.value(),
            "du": self.w_du_term.value()
        }
        
        # Control constraints - convert degrees to radians for optimization
        th_p_max_rad = np.radians(self.th_p_max.value())
        th_r_max_rad = np.radians(self.th_r_max.value())
        tau_yaw_max_val = self.tau_yaw_max.value()
        bounds = {
            "th_p": (-th_p_max_rad, th_p_max_rad),
            "th_r": (-th_r_max_rad, th_r_max_rad),
            "T": (0.0, self.T_max.value()),
            "tau_yaw": (-tau_yaw_max_val, tau_yaw_max_val),
            "k_bound": self.k_bound.value(),  # Control constraint penalty coefficient
            # State constraints - convert degrees to radians
            "state_v_horizontal_max": self.v_horizontal_max.value(),
            "state_v_vertical_max": self.v_vertical_max.value(),
            "state_roll_max": np.radians(self.roll_max.value()),
            "state_pitch_max": np.radians(self.pitch_max.value()),
            "state_yaw_max": np.radians(self.yaw_max.value()),
            "state_w_max": self.w_max.value(),
            "state_k_state_bound": self.k_state_bound.value(),  # State constraint penalty coefficient
            "state_constraint_lxx_scale": 0.0,  # 0=Method1 style (recommended), state constraint not in Lxx
        }
        
        # Physical parameters - from GUI settings
        m = self.mass.value()
        I = np.diag([self.Ixx.value(), self.Iyy.value(), self.Izz.value()])
        r_thrust = np.array([self.r_thrust_x.value(), self.r_thrust_y.value(), self.r_thrust_z.value()])
        
        return {
            'dt': self.dt_spin.value(),
            'N': self.N_spin.value(),
            'max_iter': self.max_iter_spin.value(),
            'x0': x0,
            'xg': xg,
            'weights': weights,
            'terminal_weights': terminal_weights,
            'bounds': bounds,
            'm': m,
            'I': I,
            'r_thrust': r_thrust,
            'waypoints': self.waypoints.copy(),  # Include waypoints for plotting
            'method': self.method_combo.currentIndex(),  # 0=Method1, 1=Method2 FDDP, 2=Method3 BoxFDDP
            'unified': self.unified_checkbox.isChecked()  # Merge all segments (Method 2/3 only)
        }
    
    def start_optimization(self):
        """Start optimization"""
        if self.opt_thread and self.opt_thread.isRunning():
            QMessageBox.warning(self, 'Warning', 'Optimization in progress, please stop current optimization first')
            return
        
        # Validate waypoints and times
        if len(self.waypoints) < 2:
            QMessageBox.warning(self, 'Warning', 'Need at least 2 waypoints (start and at least one waypoint)')
            return
        
        # Ensure all waypoints have all required fields and validate time order
        # Format: [x, y, z, yaw_deg, time]
        for i, wp in enumerate(self.waypoints):
            if len(wp) == 4:
                # Old format: [x, y, z, time] -> convert to [x, y, z, yaw=0, time]
                self.waypoints[i] = [wp[0], wp[1], wp[2], 0.0, wp[3]]
            elif len(wp) < 5:
                self.waypoints[i] = list(wp) + [0.0] * (5 - len(wp))
        
        # Check time order (time is at index 4)
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i][4] >= self.waypoints[i+1][4]:
                QMessageBox.warning(self, 'Warning', 
                                  f'Waypoint {i+1} arrival time ({self.waypoints[i+1][4]:.2f}s) must be greater than waypoint {i} time ({self.waypoints[i][4]:.2f}s)')
                return
        
        # Reset display
        self.iterations = []
        self.costs = []
        self.stops = []
        # Reset segment tracking
        self.segment_costs = {}
        self.segment_iterations = {}
        self.current_segment_idx = 0
        
        # Clear all plots
        self.ax_3d.clear()
        self.ax_cost.clear()
        self.ax_pos.clear()
        self.ax_vel.clear()
        self.ax_angvel.clear()
        self.ax_euler.clear()
        self.ax_pitch.clear()
        self.ax_roll.clear()
        self.ax_thrust.clear()
        self.ax_yaw.clear()
        
        # Reset titles and labels
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
        # Get parameters
        params = self.get_parameters()
        
        # Create optimization thread
        self.opt_thread = OptimizationThread(params)
        self.opt_thread.iteration_update.connect(self.update_iteration)
        self.opt_thread.state_update.connect(self.update_state)
        self.opt_thread.finished.connect(self.optimization_finished)
        self.opt_thread.error.connect(self.optimization_error)
        
        # Update button state
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setMaximum(params['max_iter'])
        self.progress.setValue(0)
        
        # Start optimization
        self.status_text.append('Starting optimization...')
        if (params.get('method') == 3 and len(params.get('waypoints', [])) > 2 and
                params.get('unified') and params.get('weights', {}).get('waypoint_terminal_cost', True)):
            self.status_text.append('(Multi-WP + waypoint terminal cost: using segment mode)')
        self.opt_thread.start()
    
    def stop_optimization(self):
        """Stop optimization"""
        if self.opt_thread and self.opt_thread.isRunning():
            self.opt_thread.stop()
            self.opt_thread.wait()
            self.status_text.append('Optimization stopped')
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def update_iteration(self, iter_num, cost, stop, segment_idx):
        """Update iteration information"""
        self.iterations.append(iter_num)
        self.costs.append(cost)
        self.stops.append(stop)
        self.current_segment_idx = segment_idx
        
        # Track costs per segment
        if segment_idx not in self.segment_costs:
            self.segment_costs[segment_idx] = []
            self.segment_iterations[segment_idx] = []
        
        self.segment_costs[segment_idx].append(cost)
        # Calculate cumulative iteration number (total iterations so far)
        cumulative_iter = len(self.iterations) - 1
        self.segment_iterations[segment_idx].append(cumulative_iter)
        
        # Update progress bar
        self.progress.setValue(iter_num)
        
        # Update status text
        self.status_text.clear()
        self.status_text.append(f'Segment: {segment_idx + 1}')
        self.status_text.append(f'Iteration: {iter_num}')
        self.status_text.append(f'Cost: {cost:.6e}')
        self.status_text.append(f'Stop Condition: {stop:.6e}')
        
        # Update Cost curve with different colors for each segment
        self.ax_cost.clear()
        
        # Define colors for different segments
        colors = ['b', 'r', 'g', 'm', 'c', 'orange', 'purple', 'brown']
        
        # Plot each segment's cost with different color
        for seg_idx in sorted(self.segment_costs.keys()):
            if len(self.segment_costs[seg_idx]) > 0:
                color = colors[seg_idx % len(colors)]
                label = f'Segment {seg_idx + 1}'
                self.ax_cost.semilogy(self.segment_iterations[seg_idx], 
                                     self.segment_costs[seg_idx], 
                                     color=color, linewidth=2.5, 
                                     marker='o', markersize=3, label=label)
        
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        if len(self.segment_costs) > 1:
            self.ax_cost.legend(fontsize=8, loc='best')
        
        # Add current cost text
        if len(self.costs) > 0:
            final_cost = self.costs[-1]
            self.ax_cost.text(0.02, 0.98, f'Current Cost: {final_cost:.4e}', 
                            transform=self.ax_cost.transAxes, fontsize=9,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.canvas.draw()
    
    def update_state(self, xs, us):
        """Update all state and control displays"""
        self.current_xs = xs
        self.current_us = us
        
        if xs is None or len(xs) == 0:
            return
        
        dt = self.dt_spin.value()
        time_states = np.arange(len(xs)) * dt
        time_controls = np.arange(len(us)) * dt
        
        xs_array = np.array(xs)
        us_array = np.array(us)
        
        # Extract states
        positions = xs_array[:, 0:3]
        velocities = xs_array[:, 3:6]
        quaternions = xs_array[:, 6:10]
        angular_velocities = xs_array[:, 10:13]
        
        # Extract control inputs
        th_p = us_array[:, 0]
        th_r = us_array[:, 1]
        T = us_array[:, 2]
        tau_yaw = us_array[:, 3]
        
        # Convert quaternion to Euler angles
        euler_angles = np.array([self.quat_to_euler(q) for q in quaternions])
        
        # Get waypoints for plotting
        waypoints = self.waypoints if hasattr(self, 'waypoints') else None
        
        # 1. 3D position trajectory (per-segment coloring for multi-segment)
        self.ax_3d.clear()
        boundaries = [min(b, len(positions) - 1) for b in segment_boundaries_from_waypoints(waypoints or [], dt)]
        if boundaries:
            idx = 0
            for i, end_idx in enumerate(boundaries):
                if idx <= end_idx:
                    seg_pos = positions[idx:end_idx + 1]
                    c = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
                    self.ax_3d.plot(seg_pos[:, 0], seg_pos[:, 1], seg_pos[:, 2],
                                    color=c, linewidth=2.5, label=f'Segment {i + 1}')
                    if idx < end_idx:
                        self.ax_3d.scatter(seg_pos[-1, 0], seg_pos[-1, 1], seg_pos[-1, 2],
                                           color=c, s=60, marker='o', edgecolors='black', linewidths=1, zorder=5)
                idx = end_idx
            if idx < len(positions) - 1:
                seg_pos = positions[idx:]
                c = SEGMENT_COLORS[len(boundaries) % len(SEGMENT_COLORS)]
                self.ax_3d.plot(seg_pos[:, 0], seg_pos[:, 1], seg_pos[:, 2],
                                color=c, linewidth=2.5, label=f'Segment {len(boundaries) + 1}')
        else:
            self.ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                         'b-', linewidth=2, label='Trajectory')
        self.ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                          color='green', s=100, marker='o', label='Start')
        self.ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                          color='red', s=100, marker='*', label='End')
        # Plot waypoints with smaller markers and numbered labels
        if waypoints is not None and len(waypoints) > 0:
            for i, wp in enumerate(waypoints):
                if len(wp) >= 3:
                    if i == 0:
                        continue  # Start point already plotted
                    # Use smaller, clearer marker (triangle up)
                    self.ax_3d.scatter(wp[0], wp[1], wp[2], 
                                      color='orange', s=50, marker='^', 
                                      edgecolors='darkorange', linewidths=1.5, 
                                      label=f'WP {i}', zorder=5, alpha=0.8)
                    # Add text label with waypoint number
                    self.ax_3d.text(wp[0], wp[1], wp[2], f' {i}', 
                                   fontsize=9, color='darkorange', 
                                   fontweight='bold', zorder=6)
        
        # Calculate unified scale for all axes
        all_x = positions[:, 0].tolist()
        all_y = positions[:, 1].tolist()
        all_z = positions[:, 2].tolist()
        if waypoints is not None and len(waypoints) > 0:
            for wp in waypoints:
                if len(wp) >= 3:
                    all_x.append(wp[0])
                    all_y.append(wp[1])
                    all_z.append(wp[2])
        
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
        self.ax_3d.set_xlim([x_center - half_range, x_center + half_range])
        self.ax_3d.set_ylim([y_center - half_range, y_center + half_range])
        self.ax_3d.set_zlim([z_center - half_range, z_center + half_range])
        
        self.ax_3d.set_xlabel('X (m)', fontsize=10)
        self.ax_3d.set_ylabel('Y (m)', fontsize=10)
        self.ax_3d.set_zlabel('Z (m)', fontsize=10)
        self.ax_3d.set_title('3D Position Trajectory', fontsize=11, fontweight='bold')
        self.ax_3d.legend(fontsize=8)
        self.ax_3d.grid(True, alpha=0.3)
        
        # 2. Position
        self.ax_pos.clear()
        self.ax_pos.plot(time_states, positions[:, 0], 'r-', label='x', linewidth=2)
        self.ax_pos.plot(time_states, positions[:, 1], 'g-', label='y', linewidth=2)
        self.ax_pos.plot(time_states, positions[:, 2], 'b-', label='z', linewidth=2)
        # Show waypoint targets
        if waypoints is not None and len(waypoints) > 0:
            last_wp = waypoints[-1]
            self.ax_pos.axhline(y=last_wp[0], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
            self.ax_pos.axhline(y=last_wp[1], color='g', linestyle='--', alpha=0.5, linewidth=1.5)
            self.ax_pos.axhline(y=last_wp[2], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
        self.ax_pos.set_xlabel('Time (s)', fontsize=9)
        self.ax_pos.set_ylabel('Position (m)', fontsize=9)
        self.ax_pos.set_title('Position', fontsize=10, fontweight='bold')
        self.ax_pos.legend(fontsize=8, loc='best')
        self.ax_pos.grid(True, alpha=0.3)
        
        # 3. Velocity
        self.ax_vel.clear()
        self.ax_vel.plot(time_states, velocities[:, 0], 'r-', label='vx', linewidth=2)
        self.ax_vel.plot(time_states, velocities[:, 1], 'g-', label='vy', linewidth=2)
        self.ax_vel.plot(time_states, velocities[:, 2], 'b-', label='vz', linewidth=2)
        # Add velocity constraints (horizontal and vertical)
        v_horizontal = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        v_vertical = np.abs(velocities[:, 2])
        v_horizontal_max_val = self.v_horizontal_max.value()
        v_vertical_max_val = self.v_vertical_max.value()
        # self.ax_vel.plot(time_states, v_horizontal, 'purple', linestyle=':', linewidth=1.5, 
        #                 label=f'|v_h| (max={v_horizontal_max_val:.1f} m/s)', alpha=0.7)
        # self.ax_vel.plot(time_states, v_vertical, 'orange', linestyle=':', linewidth=1.5, 
        #                 label=f'|v_z| (max={v_vertical_max_val:.1f} m/s)', alpha=0.7)
        # Horizontal velocity constraint lines
        self.ax_vel.axhline(y=v_horizontal_max_val, color='purple', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label=f'Max V_h ({v_horizontal_max_val:.1f} m/s)')
        self.ax_vel.axhline(y=-v_horizontal_max_val, color='purple', linestyle='--', 
                           linewidth=1.5, alpha=0.7)
        # Vertical velocity constraint lines
        self.ax_vel.axhline(y=v_vertical_max_val, color='orange', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label=f'Max V_z ({v_vertical_max_val:.1f} m/s)')
        self.ax_vel.axhline(y=-v_vertical_max_val, color='orange', linestyle='--', 
                           linewidth=1.5, alpha=0.7)
        self.ax_vel.set_xlabel('Time (s)', fontsize=9)
        self.ax_vel.set_ylabel('Velocity (m/s)', fontsize=9)
        self.ax_vel.set_title('Linear Velocity', fontsize=10, fontweight='bold')
        self.ax_vel.legend(fontsize=7, loc='best')
        self.ax_vel.grid(True, alpha=0.3)
        
        # 4. Angular velocity
        self.ax_angvel.clear()
        self.ax_angvel.plot(time_states, angular_velocities[:, 0], 'r-', label='ωx', linewidth=2)
        self.ax_angvel.plot(time_states, angular_velocities[:, 1], 'g-', label='ωy', linewidth=2)
        self.ax_angvel.plot(time_states, angular_velocities[:, 2], 'b-', label='ωz', linewidth=2)
        # Add angular velocity magnitude constraint
        w_mag = np.linalg.norm(angular_velocities, axis=1)
        w_max_val = self.w_max.value()
        self.ax_angvel.plot(time_states, w_mag, 'purple', linestyle=':', linewidth=1.5, 
                           label=f'|ω| (max={w_max_val:.2f} rad/s)', alpha=0.7)
        self.ax_angvel.axhline(y=w_max_val, color='r', linestyle='--', 
                              linewidth=1.5, alpha=0.7, label=f'Max ({w_max_val:.2f} rad/s)')
        self.ax_angvel.axhline(y=-w_max_val, color='r', linestyle='--', 
                              linewidth=1.5, alpha=0.7)
        self.ax_angvel.set_xlabel('Time (s)', fontsize=9)
        self.ax_angvel.set_ylabel('Angular Vel (rad/s)', fontsize=9)
        self.ax_angvel.set_title('Angular Velocity', fontsize=10, fontweight='bold')
        self.ax_angvel.legend(fontsize=7, loc='best')
        self.ax_angvel.grid(True, alpha=0.3)
        
        # 5. Euler angles
        self.ax_euler.clear()
        euler_deg = np.degrees(euler_angles)
        self.ax_euler.plot(time_states, euler_deg[:, 0], 'r-', label='Roll', linewidth=2)
        self.ax_euler.plot(time_states, euler_deg[:, 1], 'g-', label='Pitch', linewidth=2)
        self.ax_euler.plot(time_states, euler_deg[:, 2], 'b-', label='Yaw', linewidth=2)
        # Add Euler angle constraints
        roll_max_deg = self.roll_max.value()
        pitch_max_deg = self.pitch_max.value()
        yaw_max_deg = self.yaw_max.value()
        self.ax_euler.axhline(y=roll_max_deg, color='r', linestyle='--', 
                             linewidth=1.5, alpha=0.7, label=f'Roll Max ({roll_max_deg:.1f}°)')
        self.ax_euler.axhline(y=-roll_max_deg, color='r', linestyle='--', 
                             linewidth=1.5, alpha=0.7)
        self.ax_euler.axhline(y=pitch_max_deg, color='g', linestyle='--', 
                             linewidth=1.5, alpha=0.7, label=f'Pitch Max ({pitch_max_deg:.1f}°)')
        self.ax_euler.axhline(y=-pitch_max_deg, color='g', linestyle='--', 
                             linewidth=1.5, alpha=0.7)
        # self.ax_euler.axhline(y=yaw_max_deg, color='b', linestyle='--', 
        #                      linewidth=1.5, alpha=0.7, label=f'Yaw Max ({yaw_max_deg:.1f}°)')
        # self.ax_euler.axhline(y=-yaw_max_deg, color='b', linestyle='--', 
        #                      linewidth=1.5, alpha=0.7)
        self.ax_euler.set_xlabel('Time (s)', fontsize=9)
        self.ax_euler.set_ylabel('Euler Angles (deg)', fontsize=9)
        self.ax_euler.set_title('Attitude (Euler)', fontsize=10, fontweight='bold')
        self.ax_euler.legend(fontsize=7, loc='best')
        self.ax_euler.grid(True, alpha=0.3)
        
        # 6. TVC Pitch angle
        self.ax_pitch.clear()
        th_p_deg = np.degrees(th_p)
        self.ax_pitch.plot(time_controls, th_p_deg, 'b-', linewidth=2, 
                          label='θ_pitch', marker='o', markersize=2)
        # Add constraint limits
        th_p_max_deg = self.th_p_max.value()
        self.ax_pitch.axhline(y=th_p_max_deg, color='r', linestyle='--', 
                             linewidth=1.5, alpha=0.7, label=f'Max ({th_p_max_deg:.1f}°)')
        self.ax_pitch.axhline(y=-th_p_max_deg, color='r', linestyle='--', 
                             linewidth=1.5, alpha=0.7, label=f'Min (-{th_p_max_deg:.1f}°)')
        self.ax_pitch.set_xlabel('Time (s)', fontsize=9)
        self.ax_pitch.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_pitch.set_title('TVC Pitch Angle', fontsize=10, fontweight='bold')
        self.ax_pitch.legend(fontsize=7, loc='best')
        self.ax_pitch.grid(True, alpha=0.3)
        
        # 7. TVC Roll angle
        self.ax_roll.clear()
        th_r_deg = np.degrees(th_r)
        self.ax_roll.plot(time_controls, th_r_deg, 'r-', linewidth=2, 
                         label='θ_roll', marker='o', markersize=2)
        # Add constraint limits
        th_r_max_deg = self.th_r_max.value()
        self.ax_roll.axhline(y=th_r_max_deg, color='b', linestyle='--', 
                            linewidth=1.5, alpha=0.7, label=f'Max ({th_r_max_deg:.1f}°)')
        self.ax_roll.axhline(y=-th_r_max_deg, color='b', linestyle='--', 
                            linewidth=1.5, alpha=0.7, label=f'Min (-{th_r_max_deg:.1f}°)')
        self.ax_roll.set_xlabel('Time (s)', fontsize=9)
        self.ax_roll.set_ylabel('Angle (deg)', fontsize=9)
        self.ax_roll.set_title('TVC Roll Angle', fontsize=10, fontweight='bold')
        self.ax_roll.legend(fontsize=7, loc='best')
        self.ax_roll.grid(True, alpha=0.3)
        
        # 8. Thrust
        self.ax_thrust.clear()
        self.ax_thrust.plot(time_controls, T, 'g-', linewidth=2, 
                           label='Thrust', marker='o', markersize=2)
        # Add constraint limits
        T_max_val = self.T_max.value()
        self.ax_thrust.axhline(y=T_max_val, color='r', linestyle='--', 
                              linewidth=1.5, alpha=0.7, label=f'Max ({T_max_val:.1f} N)')
        self.ax_thrust.axhline(y=0.0, color='r', linestyle='--', 
                              linewidth=1.5, alpha=0.7, label='Min (0 N)')
        self.ax_thrust.set_xlabel('Time (s)', fontsize=9)
        self.ax_thrust.set_ylabel('Thrust (N)', fontsize=9)
        self.ax_thrust.set_title('Thrust', fontsize=10, fontweight='bold')
        self.ax_thrust.legend(fontsize=7, loc='best')
        self.ax_thrust.grid(True, alpha=0.3)
        
        # 9. Yaw torque
        self.ax_yaw.clear()
        self.ax_yaw.plot(time_controls, tau_yaw, 'm-', linewidth=2, 
                        label='τ_yaw', marker='o', markersize=2)
        # Add constraint limits from GUI settings
        tau_yaw_max = self.tau_yaw_max.value()
        self.ax_yaw.axhline(y=tau_yaw_max, color='r', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label=f'Max ({tau_yaw_max:.2f} N·m)')
        self.ax_yaw.axhline(y=-tau_yaw_max, color='r', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label=f'Min (-{tau_yaw_max:.2f} N·m)')
        self.ax_yaw.set_xlabel('Time (s)', fontsize=9)
        self.ax_yaw.set_ylabel('Torque (N·m)', fontsize=9)
        self.ax_yaw.set_title('Yaw Torque', fontsize=10, fontweight='bold')
        self.ax_yaw.legend(fontsize=7, loc='best')
        self.ax_yaw.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def optimization_finished(self, xs, us, all_loggers, timing_info=None):
        """Optimization finished"""
        self.status_text.append('Optimization completed!')
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(self.progress.maximum())
        
        # Show timing: average time per iteration
        if timing_info:
            avg_ms = timing_info.get("avg_time_per_iter", 0) * 1000
            total_time = timing_info.get("total_time", 0)
            total_iters = timing_info.get("total_iters", 0)
            method_name = timing_info.get("method", "")
            self.status_text.append(f'{method_name}')
            self.status_text.append(f'Total time: {total_time:.3f}s, Total iterations: {total_iters}')
            self.status_text.append(f'Average time per iteration: {avg_ms:.2f} ms')
            print(f"[{method_name}] Total: {total_time:.3f}s, Iters: {total_iters}, Avg/iter: {avg_ms:.2f} ms")
        
        # Show final results from all segments
        if all_loggers and len(all_loggers) > 0:
            total_iterations = 0
            for i, logger in enumerate(all_loggers):
                if logger and len(logger.costs) > 0:
                    final_cost = logger.costs[-1]
                    total_iterations += len(logger.costs)
                    self.status_text.append(f'Segment {i+1} Final Cost: {final_cost:.6e}')
            if not timing_info:
                self.status_text.append(f'Total Iterations: {total_iterations}')
        
        # Update cost plot with all segments using different colors
        self.ax_cost.clear()
        
        # Define colors for different segments
        colors = ['b', 'r', 'g', 'm', 'c', 'orange', 'purple', 'brown']
        
        # Plot each segment's cost from loggers
        if all_loggers and len(all_loggers) > 0:
            cumulative_iter = 0
            for seg_idx, logger in enumerate(all_loggers):
                if logger and len(logger.costs) > 0:
                    color = colors[seg_idx % len(colors)]
                    label = f'Segment {seg_idx + 1}'
                    # Create iteration numbers for this segment
                    seg_iterations = np.arange(len(logger.costs)) + cumulative_iter
                    self.ax_cost.semilogy(seg_iterations, logger.costs, 
                                         color=color, linewidth=2.5, 
                                         marker='o', markersize=3, label=label)
                    cumulative_iter += len(logger.costs)
        
        self.ax_cost.set_xlabel('Iteration', fontsize=10)
        self.ax_cost.set_ylabel('Cost (log scale)', fontsize=10)
        self.ax_cost.set_title('Optimization Cost Convergence', fontsize=11, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3)
        self.ax_cost.legend(fontsize=8, loc='best')
        
        # Add final cost text
        if all_loggers and len(all_loggers) > 0:
            last_logger = all_loggers[-1]
            if last_logger and len(last_logger.costs) > 0:
                final_cost = last_logger.costs[-1]
                self.ax_cost.text(0.02, 0.98, f'Final Cost: {final_cost:.4e}', 
                                transform=self.ax_cost.transAxes, fontsize=9,
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.canvas.draw()
        
        # Update final state
        self.update_state(xs, us)
        
        # Show full trajectory plot only if checkbox is checked
        if self.show_plot_checkbox.isChecked():
            self.show_full_trajectory(xs, us, all_loggers)
    
    def optimization_error(self, error_msg):
        """Optimization error"""
        QMessageBox.critical(self, 'Error', f'Error during optimization:\n{error_msg}')
        self.status_text.append(f'Error: {error_msg}')
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def show_full_trajectory(self, xs, us, all_loggers=None):
        """Show full trajectory plot"""
        try:
            from tvc_traj_opt import plot_trajectory
            import matplotlib.pyplot as plt
            
            waypoints = self.waypoints if hasattr(self, 'waypoints') else None
            dt = self.dt_spin.value()
            fig = plot_trajectory(xs, us, dt, all_loggers=all_loggers, x_goal=None, waypoints=waypoints)
            plt.show()
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f'Cannot display full trajectory plot: {str(e)}')


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
