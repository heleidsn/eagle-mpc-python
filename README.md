# Eagle MPC Python - S500 Quadrotor & UAM Trajectory Planning

S500 quadrotor and S500 UAM (UAV with Arm) trajectory optimization project using Crocoddyl and Pinocchio.

## 📋 Project Overview

This project implements trajectory planning for the S500 quadrotor and S500 UAM (UAV with Arm) using the Crocoddyl optimal control library and Pinocchio robotics dynamics library. It supports generating optimized trajectories by defining waypoints and includes comprehensive visualization and a GUI interface.

## ✨ Key Features

- **Trajectory Optimization**: Crocoddyl DDP algorithm for trajectory optimization
- **S500 Quadrotor**: Full S500 quadrotor dynamics model
- **S500 UAM**: Quadrotor + 2-DOF arm with end-effector grasp constraints
- **Task Types**: Point-to-Point (start→target), Grasp (start→grasp point→target)
- **Thrust Constraints**: Automatic thrust upper and lower bounds
- **Visualization**: State trajectories, control inputs, 3D trajectory, cost convergence
- **GUI**: PyQt5 graphical interface with parameter tuning, save/load, interactive plots

## 🛠️ Requirements

### Python Version

- Python 3.7+

### Main Dependencies

- `crocoddyl`: Optimal control library
- `pinocchio`: Robotics dynamics library
- `numpy`: Numerical computing
- `matplotlib`: Visualization
- `pyyaml`: YAML configuration file parsing
- `example-robot-data`: Robot model data (optional, for examples)

### Installation

#### Linux/macOS (Recommended)

```bash
# Using conda environment (recommended)
conda create -n eagle_mpc python=3.10
conda activate eagle_mpc

# Install crocoddyl and pinocchio
conda install pinocchio -c conda-forge
conda install crocoddyl -c conda-forge
# Other dependencies (PyQt5 required for GUI)
pip install numpy matplotlib pyyaml pyqt5
```

#### Windows

**Note**: `crocoddyl` may not be directly installable via conda or pip on Windows, as conda-forge may not have pre-compiled packages for Windows platform.

**Option 1: Use WSL (Windows Subsystem for Linux) (Recommended)**

Follow the Linux installation steps in WSL:

```bash
# In WSL
conda create -n eagle_mpc python=3.10
conda activate eagle_mpc
conda install pinocchio -c conda-forge
conda install crocoddyl -c conda-forge
pip install numpy matplotlib pyyaml
```

**Option 2: Build from Source**

If you need to use it directly on Windows, you need to build `crocoddyl` from source:

1. Install required tools:
   - CMake (>= 3.10)
   - Visual Studio or MinGW-w64
   - Git

2. Clone and build crocoddyl:
```bash
git clone https://github.com/loco-3d/crocoddyl.git
cd crocoddyl
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cmake --install . --prefix <install_path>
```

3. Set Python path:
```bash
# Add crocoddyl Python bindings path to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;<install_path>\lib\python3.10\site-packages
```

**Option 3: Use Docker**

Use a Docker container that includes crocoddyl to run the project.

## 📁 Project Structure

```
eagle-mpc-python/
├── config/
│   └── yaml/
│       ├── multicopter/
│       │   └── s500.yaml              # S500 quadrotor configuration
│       ├── mpc/                       # MPC configuration
│       └── trajectories/             # Trajectory params (s500_uam_trajectory_params.json)
├── models/
│   ├── urdf/
│   │   ├── s500_simple.urdf           # S500 quadrotor URDF
│   │   └── s500_uam_simple.urdf       # S500 UAM (quadrotor + arm) URDF
│   └── sdf/                          # SDF models
├── scripts/
│   ├── s500_trajectory_planner.py     # S500 quadrotor trajectory planner
│   ├── example_s500_trajectory.py     # S500 examples
│   ├── s500_uam_trajectory_planner.py # S500 UAM trajectory planner
│   ├── s500_uam_trajectory_gui.py     # S500 UAM GUI
│   ├── example_s500_uam_trajectory.py # S500 UAM examples
│   └── crocoddyl_quad_trajectory_opt.py
└── results/                          # Optimization results output
```

## 🚀 Quick Start

### 1. S500 UAM GUI (Recommended)

```bash
python scripts/s500_uam_trajectory_gui.py
```

- **Task types**: Point-to-Point (start→target), Grasp (start→grasp point→target)
- **Waypoints**: Row 1 — Start x,y,z, Start j1,j2 (°); Row 2 — Target x,y,z, Target j1,j2 (°); Row 3 — Duration
- **Cost parameters**: State weight, Control weight, EE position weight
- **Visualization**: Trajectory tab (main plot), 3D Trajectory tab (3D plot)
- **Params save/load**: Fixed path `config/yaml/trajectories/s500_uam_trajectory_params.json`

### 2. S500 UAM Command-Line Example

```bash
python scripts/example_s500_uam_trajectory.py
```

Supports `grasp`, `catch`, and `simple` trajectory types.

### 3. S500 Quadrotor Example

```bash
python scripts/example_s500_trajectory.py
```

Provides Square, Figure-Eight, and other predefined trajectories.

### 4. S500 Quadrotor Basic Usage

```python
from s500_trajectory_planner import S500TrajectoryPlanner
import numpy as np

planner = S500TrajectoryPlanner()
waypoints = [
    np.array([0.0, 0.0, 0.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([0.0, 0.0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([2.0, 0.0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
]
durations = [2.0, 3.0]
planner.create_trajectory_problem(waypoints, durations, dt=0.01)
converged = planner.solve_trajectory(max_iter=100)
if converged:
    planner.plot_trajectory()
```

### 5. S500 UAM Command-Line Arguments

```bash
python scripts/s500_uam_trajectory_planner.py --simple --max-iter 150 --dt 0.02
```

## 📊 State Vector Format

### S500 Quadrotor (13-D)

```python
state = [
    x, y, z,              # Position (m)
    qx, qy, qz, qw,       # Quaternion orientation
    vx, vy, vz,           # Linear velocity (m/s)
    wx, wy, wz            # Angular velocity (rad/s)
]
```

### S500 UAM (17-D)

```python
state = [
    x, y, z,              # Base position (m)
    qx, qy, qz, qw,       # Base quaternion orientation
    j1, j2,               # Arm joint angles (rad)
    vx, vy, vz,           # Base linear velocity (m/s)
    wx, wy, wz,           # Base angular velocity (rad/s)
    j1_dot, j2_dot        # Arm joint angular velocity (rad/s)
]
```

Control: `[thrust_1..4, torque_j1, torque_j2]` (4 thrusters + 2 joint torques)

## 🎯 Features

### S500TrajectoryPlanner (Quadrotor)

- `create_trajectory_problem(waypoints, durations, dt, ...)`: Create trajectory optimization problem
- `solve_trajectory(max_iter, verbose)`: Solve optimization
- `plot_trajectory(save_path, show_waypoints)`: Plot results
- `save_trajectory(save_path)`: Save trajectory data

### S500UAMTrajectoryPlanner (Quadrotor + Arm)

- `create_trajectory_problem_simple(start_state, target_state, duration, ...)`: Simple mode (start→target)
- `create_trajectory_problem(start_state, grasp_position, target_state, durations, ...)`: Grasp mode (start→grasp→target)
- `get_plot_figure()`, `get_3d_plot_figure()`: Main and 3D figures for GUI embedding

### Key Parameters

- `waypoint_multiplier`: Waypoint weight multiplier (default 1000.0)
- `state_weight`, `control_weight`: State/control cost weights
- `grasp_ee_weight`: End-effector position weight in Grasp mode
- `dt`: Time step (default 0.02s)

## 📈 Output Results

After optimization:

1. **Main plot**: Base position, Base orientation (Euler), Joint angles (°), velocities, Base control, Arm control, Cost convergence
2. **3D plot**: Base and EE trajectories with equal axis scaling
3. **Data file** (`*.npz`): states, controls, ee_positions, cost, iterations

## 🔧 Configuration

### S500 Configuration File (`config/yaml/multicopter/s500.yaml`)

Contains key quadrotor parameters:

- Number of thrusters
- Thrust coefficient (cf)
- Moment coefficient (cm)
- Thrust upper and lower bounds
- Thruster positions and rotation directions

### URDF Model (`models/urdf/s500_simple.urdf`)

Contains robot physical parameters:

- Mass
- Inertia matrix
- Geometric structure

## 📝 Example Code

### S500 UAM Simple Trajectory

```python
from s500_uam_trajectory_planner import S500UAMTrajectoryPlanner, make_uam_state

planner = S500UAMTrajectoryPlanner()
start = make_uam_state(0, 0, 1.0, j1=-1.2, j2=-0.6)   # x,y,z, joint angles (rad)
target = make_uam_state(1.0, 0.5, 2.0, j1=-0.8, j2=-0.3)

planner.create_trajectory_problem_simple(
    start_state=start, target_state=target, duration=5.0, dt=0.02
)
converged = planner.solve_trajectory(max_iter=200)
planner.plot_trajectory(save_path='results/uam_traj.png')
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure crocoddyl and pinocchio are properly installed
   - **Windows Users**: If you encounter `ModuleNotFoundError: No module named 'crocoddyl'`, please refer to the Windows installation instructions above. Using WSL is recommended.
2. **Path Errors**: Check that configuration and URDF file paths are correct
3. **Convergence Issues**: Try increasing iteration count or adjusting weight parameters
4. **Thrust Constraints**: If trajectory is unreasonable, check if thrust constraints are too tight
5. **Conda Installation Failed (Windows)**: `crocoddyl` may not support Windows platform on conda-forge. Please use WSL or build from source.

## 📚 References

- [Crocoddyl Documentation](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/index.html)
- [Pinocchio Documentation](https://stack-of-tasks.github.io/pinocchio/)
- [S500 Quadrotor Specifications](https://github.com/PX4/PX4-Autopilot)

## 👤 Author

Lei He

## 📅 Changelog

- **2026-02-11**: S500 UAM support
  - S500 UAM trajectory planning (quadrotor + 2-DOF arm)
  - PyQt5 GUI: task selection, waypoints, cost parameters, tabbed plots (Trajectory + 3D)
  - Point-to-Point and Grasp task modes
  - Waypoint layout: Start/Target (x,y,z, j1,j2 in degrees), Duration
  - Fixed parameter save/load path
- **2026-01-15**: Initial version
  - Basic trajectory planning functionality
  - Adapted to new Crocoddyl API (ActuationModelFloatingBaseThrusters)
  - Added comprehensive visualization features

## TODO

- Add grasp support
- Add constrains support

## 📄 License

[Add license information here]
