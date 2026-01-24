# Eagle MPC Python - S500 Quadrotor Trajectory Planning

S500 quadrotor trajectory optimization project using Crocoddyl and Pinocchio.

## 📋 Project Overview

This project implements trajectory planning for the S500 quadrotor using the Crocoddyl optimal control library and Pinocchio robotics dynamics library. It supports generating optimized trajectories by defining waypoints and includes comprehensive visualization capabilities.

## ✨ Key Features

- **Trajectory Optimization**: Uses Crocoddyl's DDP algorithm for trajectory optimization
- **Multicopter Model**: Supports complete dynamics model for S500 quadrotor
- **Waypoint Planning**: Supports custom waypoint sequences for trajectory planning
- **Thrust Constraints**: Automatically applies thrust upper and lower bounds for thrusters
- **Visualization**: Complete state trajectory and control input visualization
- **Multiple Trajectory Types**: Supports square trajectories, figure-eight trajectories, and other predefined trajectories

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
# Install other dependencies
pip install numpy matplotlib pyyaml
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
│       │   └── s500.yaml          # S500 quadrotor configuration
│       ├── mpc/                   # MPC configuration files
│       └── trajectories/          # Trajectory configuration files
├── models/
│   ├── urdf/
│   │   └── s500_simple.urdf      # S500 URDF model
│   └── sdf/                      # SDF model files
├── scripts/
│   ├── s500_trajectory_planner.py    # Main trajectory planning class
│   ├── example_s500_trajectory.py    # Usage examples
│   └── crocoddyl_quad_trajectory_opt.py  # Crocoddyl examples
└── results/                      # Optimization results output directory
```

## 🚀 Quick Start

### 1. Basic Usage

Use the `S500TrajectoryPlanner` class for trajectory planning:

```python
from s500_trajectory_planner import S500TrajectoryPlanner
import numpy as np

# Create planner
planner = S500TrajectoryPlanner()

# Define waypoints
waypoints = [
    np.array([0.0, 0.0, 0.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Start point
    np.array([0.0, 0.0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Ascend
    np.array([2.0, 0.0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Forward
]

# Define duration for each segment
durations = [2.0, 3.0]

# Create optimization problem
planner.create_trajectory_problem(waypoints, durations, dt=0.01)

# Solve
converged = planner.solve_trajectory(max_iter=100)

# Visualize
if converged:
    planner.plot_trajectory()
```

### 2. Run Example Scripts

#### Interactive Trajectory Planning Example

```bash
python scripts/example_s500_trajectory.py
```

This script provides two predefined trajectories:

1. **Square Trajectory**: Square flight pattern at specified altitude
2. **Figure-Eight Trajectory**: Classic figure-eight flight pattern

#### Command Line Arguments

```bash
python scripts/s500_trajectory_planner.py \
    --s500-yaml config/yaml/multicopter/s500.yaml \
    --urdf models/urdf/s500_simple.urdf \
    --max-iter 100 \
    --dt 0.01 \
    --save-dir results/my_trajectory
```

## 📊 State Vector Format

The state vector is 13-dimensional with the following format:

```python
state = [
    x, y, z,              # Position (m)
    qx, qy, qz, qw,      # Quaternion orientation
    vx, vy, vz,          # Linear velocity (m/s)
    wx, wy, wz           # Angular velocity (rad/s)
]
```

## 🎯 Features

### S500TrajectoryPlanner Class

Main methods:

- `__init__(s500_yaml_path, urdf_path)`: Initialize planner
- `create_trajectory_problem(waypoints, durations, dt, ...)`: Create trajectory optimization problem
- `solve_trajectory(max_iter, verbose)`: Solve trajectory optimization
- `plot_trajectory(save_path, show_waypoints)`: Plot trajectory results
- `save_trajectory(save_path)`: Save trajectory data

### Key Parameters

- `waypoint_multiplier`: Waypoint weight enhancement multiplier (default 1000.0)
- `use_thrust_constraints`: Enable thrust constraints (default True)
- `dt`: Time step (default 0.01s)
- `max_iter`: Maximum iterations (default 100)

## 📈 Output Results

After optimization, the following files are generated:

1. **Visualization Plots** (`*.png`): Complete plots of all states and controls

   - Position, velocity, angular velocity trajectories
   - Orientation quaternion
   - Control inputs (thruster thrusts)
   - 3D trajectory plot
   - Cost convergence curve
2. **Data Files** (`*.npz`): Trajectory data in NumPy format

   - `states`: State trajectory
   - `controls`: Control input trajectory
   - `cost`: Final cost
   - `iterations`: Number of iterations

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

### Create Custom Trajectory

```python
from s500_trajectory_planner import S500TrajectoryPlanner
import numpy as np

planner = S500TrajectoryPlanner()

# Define waypoints
waypoints = []
durations = []

# Start point
start = np.zeros(13)
start[6] = 1.0  # qw = 1
waypoints.append(start)

# Add more waypoints...
# waypoints.append(...)
# durations.append(...)

# Create and solve
planner.create_trajectory_problem(waypoints, durations, dt=0.01)
converged = planner.solve_trajectory(max_iter=100)

if converged:
    planner.plot_trajectory()
    planner.save_trajectory('my_trajectory.npz')
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

- **2026-01-15**: Initial version
  - Basic trajectory planning functionality
  - Adapted to new Crocoddyl API (ActuationModelFloatingBaseThrusters)
  - Added comprehensive visualization features

## 📄 License

[Add license information here]
