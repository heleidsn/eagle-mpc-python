#!/usr/bin/env python3
"""
S500 UAM (UAV with Arm) Trajectory Planning Script
Using Crocoddyl and Pinocchio for trajectory optimization

Features:
- Load S500 UAM geometry from URDF (quadrotor + 2-DOF arm)
- Load Pinocchio model from URDF file
- Support end-effector (gripper_link) position constraints for grasping
- Perform trajectory optimization: start -> grasp point -> target

State: [x,y,z, qx,qy,qz,qw, j1,j2, vx,vy,vz, wx,wy,wz, j1_dot,j2_dot]  (q then v)
Control: [thrust_1, thrust_2, thrust_3, thrust_4, torque_j1, torque_j2]

Author: Lei He
Date: 2026-02-11
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import time
import pinocchio as pin
import crocoddyl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def compute_ee_kinematics_along_trajectory(
    states: np.ndarray,
    robot_model,
    data,
    ee_frame_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EE frame in world: position, linear velocity, RPY (rad), angular velocity (world)."""
    n = len(states)
    nq = robot_model.nq
    ee_pos = np.zeros((n, 3))
    ee_v = np.zeros((n, 3))
    ee_rpy = np.zeros((n, 3))
    ee_w = np.zeros((n, 3))
    rf = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    for i, x in enumerate(states):
        q = np.asarray(x[:nq], dtype=float).flatten()
        v = np.asarray(x[nq:], dtype=float).flatten()
        pin.forwardKinematics(robot_model, data, q, v)
        pin.updateFramePlacements(robot_model, data)
        oMf = data.oMf[ee_frame_id]
        ee_pos[i] = oMf.translation
        ee_rpy[i] = pin.rpy.matrixToRpy(oMf.rotation)
        vel = pin.getFrameVelocity(robot_model, data, ee_frame_id, rf)
        ee_v[i] = np.array(vel.linear).flatten()
        ee_w[i] = np.array(vel.angular).flatten()
    return ee_pos, ee_v, ee_rpy, ee_w


class S500UAMTrajectoryPlanner:
    """S500 UAM (UAV with Arm) Trajectory Planner"""

    # End-effector frame name in URDF
    EE_FRAME_NAME = "gripper_link"

    def __init__(self, s500_yaml_path: str = None, urdf_path: str = None):
        """
        Initialize S500 UAM trajectory planner

        Args:
            s500_yaml_path: Path to S500 configuration YAML file
            urdf_path: Path to S500 UAM URDF model file
        """
        if s500_yaml_path is None:
            s500_yaml_path = str(Path(__file__).parent.parent / 'config' / 'yaml' / 'multicopter' / 's500.yaml')
        if urdf_path is None:
            urdf_path = str(Path(__file__).parent.parent / 'models' / 'urdf' / 's500_uam_simple.urdf')

        self.s500_yaml_path = s500_yaml_path
        self.urdf_path = urdf_path

        self.s500_config = None
        self.robot_model = None
        self.robot_data = None
        self.state = None
        self.actuation = None
        self.problem = None
        self.solver = None
        self.dt = None
        self.ee_frame_id = None
        self._cost_logger = None

        self.load_s500_config()
        self.load_pinocchio_model()
        self.create_actuation_model()

    def load_s500_config(self):
        """Load S500 configuration from YAML file"""
        try:
            with open(self.s500_yaml_path, 'r') as f:
                self.s500_config = yaml.safe_load(f)
            print(f"✓ Loaded S500 configuration: {self.s500_yaml_path}")
            platform = self.s500_config['platform']
            print(f"  - Rotors: {platform['n_rotors']}, cf: {platform['cf']}, max_thrust: {platform['max_thrust']} N")
        except Exception as e:
            print(f"✗ Failed to load config: {e}")
            raise

    def load_pinocchio_model(self):
        """Load Pinocchio model from URDF file"""
        try:
            print(f"Loading URDF: {self.urdf_path}")
            self.robot_model = pin.buildModelFromUrdf(self.urdf_path, pin.JointModelFreeFlyer())
            self.robot_data = self.robot_model.createData()
            self.state = crocoddyl.StateMultibody(self.robot_model)

            self.ee_frame_id = self.robot_model.getFrameId(self.EE_FRAME_NAME)
            if self.ee_frame_id == -1:
                raise ValueError(f"Frame '{self.EE_FRAME_NAME}' not found in model")

            print(f"✓ Loaded Pinocchio model")
            print(f"  - nq: {self.robot_model.nq}, nv: {self.robot_model.nv}, ndx: {self.state.ndx}")
            print(f"  - EE frame '{self.EE_FRAME_NAME}' id: {self.ee_frame_id}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise

    def create_actuation_model(self):
        """Create actuation model (thrusters + arm joint torques)"""
        try:
            platform = self.s500_config['platform']
            cf = platform['cf']
            cm = platform['cm']
            rotors = platform['$rotors']
            min_thrust = platform['min_thrust']
            max_thrust = platform['max_thrust']

            thruster_list = []
            for i, rotor in enumerate(rotors):
                pos = np.array(rotor['translation'])
                spin_dir = rotor['spin_direction'][0]
                M = pin.SE3(np.eye(3), pos)
                ctorque = abs(spin_dir) * cm / cf
                thruster_type = crocoddyl.ThrusterType.CCW if spin_dir < 0 else crocoddyl.ThrusterType.CW
                thruster = crocoddyl.Thruster(M, ctorque, thruster_type, min_thrust, max_thrust)
                thruster_list.append(thruster)

            self.actuation = crocoddyl.ActuationModelFloatingBaseThrusters(self.state, thruster_list)
            print(f"✓ Actuation: nu={self.actuation.nu} (thrusters + arm torques)")
        except Exception as e:
            print(f"✗ Failed to create actuation: {e}")
            raise

    def create_cost_model(self,
                         target_state: np.ndarray = None,
                         grasp_position: np.ndarray = None,
                         control_weight: float = 1e-5,
                         state_weight: float = 1,
                         ee_position_weight: float = 0,
                         is_terminal: bool = False,
                         is_waypoint: bool = False,
                         waypoint_multiplier: float = 10000.0) -> crocoddyl.CostModelSum:
        """
        Create cost model

        Args:
            target_state: Target full state [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz,j1,j2,j1_dot,j2_dot]
            grasp_position: Target end-effector position [x,y,z] (world frame)
            control_weight: Control regularization weight
            state_weight: State tracking weight
            ee_position_weight: End-effector position tracking weight
            is_terminal: Terminal cost
            is_waypoint: Waypoint cost (enhanced weight)
            waypoint_multiplier: Weight multiplier for waypoints
        """
        control_dim = self.actuation.nu
        cost_model = crocoddyl.CostModelSum(self.state, control_dim)

        # Default target state (full state: q then v)
        nq, nv = self.robot_model.nq, self.robot_model.nv
        if target_state is None:
            target_state = np.zeros(nq + nv)
            target_state[2] = 1.0   # z
            target_state[6] = 1.0  # qw

        effective_state_weight = float(state_weight)
        effective_control_weight = float(control_weight)
        effective_ee_weight = float(ee_position_weight)
        if is_waypoint:
            effective_state_weight *= waypoint_multiplier
            effective_control_weight *= waypoint_multiplier
            effective_ee_weight *= waypoint_multiplier

        # State cost
        state_activation = crocoddyl.ActivationModelQuad(self.state.ndx)
        state_residual = crocoddyl.ResidualModelState(self.state, target_state, control_dim)
        cost_model.addCost("state_reg",
                          crocoddyl.CostModelResidual(self.state, state_activation, state_residual),
                          effective_state_weight)

        # End-effector position cost (grasp point)
        if grasp_position is not None and ee_position_weight > 0:
            ee_residual = crocoddyl.ResidualModelFrameTranslation(
                self.state, self.ee_frame_id, grasp_position, control_dim
            )
            cost_model.addCost("ee_translation",
                              crocoddyl.CostModelResidual(self.state, ee_residual),
                              effective_ee_weight)

        # Control cost
        if not is_terminal:
            mass = self.robot_model.inertias[1].mass
            hover_thrust = mass * 9.81 / 4
            control_ref = np.array([hover_thrust] * 4 + [0.0] * (control_dim - 4))
            control_activation = crocoddyl.ActivationModelQuad(self.actuation.nu)
            control_residual = crocoddyl.ResidualModelControl(self.state, control_ref)
            cost_model.addCost("control_reg",
                              crocoddyl.CostModelResidual(self.state, control_activation, control_residual),
                              effective_control_weight)

        return cost_model

    def create_trajectory_problem(self,
                                 start_state: np.ndarray,
                                 grasp_position: np.ndarray,
                                 target_state: np.ndarray,
                                 durations: List[float],
                                 dt: float = 0.02,
                                 grasp_ee_weight: float = 5000.0,
                                 waypoint_multiplier: float = 1000.0,
                                 state_weight: float = 1.0,
                                 control_weight: float = 1e-5,
                                 use_thrust_constraints: bool = True) -> None:
        """
        Create trajectory optimization problem: start -> grasp -> target

        Args:
            start_state: Initial state [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz,j1,j2,j1_dot,j2_dot]
            grasp_position: End-effector target position at grasp [x,y,z]
            target_state: Final target state
            durations: [duration_to_grasp, duration_to_target]
            dt: Time step
            grasp_ee_weight: Weight for EE position at grasp waypoint
            waypoint_multiplier: Waypoint weight multiplier
            use_thrust_constraints: Apply thrust limits
        """
        if len(durations) != 2:
            raise ValueError("durations must have 2 elements: [to_grasp, to_target]")

        self.dt = dt

        # Initial state: full state (q,v) for ShootingProblem
        x0 = np.array(start_state, dtype=float).copy()
        if len(x0) != self.robot_model.nq + self.robot_model.nv:
            raise ValueError(f"start_state must have {self.robot_model.nq + self.robot_model.nv} elements (q+v), got {len(x0)}")

        self._waypoint_times = [0.0]
        self._waypoint_positions = [start_state[:3]]
        self._waypoint_labels = ["Start"]

        running_models = []

        # Segment 1: Start -> Grasp
        n_steps_1 = max(1, int(durations[0] / dt))
        self._waypoint_times.append(durations[0])
        self._waypoint_positions.append(grasp_position)
        self._waypoint_labels.append("Grasp")

        # Grasp waypoint cost: reach EE position + state
        grasp_cost = self.create_cost_model(
            target_state=target_state,
            grasp_position=grasp_position,
            ee_position_weight=grasp_ee_weight,
            state_weight=state_weight,
            control_weight=control_weight,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier
        )
        grasp_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, grasp_cost
        )
        grasp_int = crocoddyl.IntegratedActionModelEuler(grasp_diff, dt)
        running_models.append(grasp_int)

        normal_cost_1 = self.create_cost_model(
            target_state=target_state,
            grasp_position=grasp_position,
            ee_position_weight=grasp_ee_weight * 0.1,
            state_weight=state_weight,
            control_weight=control_weight
        )
        normal_diff_1 = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, normal_cost_1
        )
        normal_int_1 = crocoddyl.IntegratedActionModelEuler(normal_diff_1, dt)

        for _ in range(n_steps_1 - 1):
            running_models.append(normal_int_1)

        # Segment 2: Grasp -> Target
        n_steps_2 = max(1, int(durations[1] / dt))
        self._waypoint_times.append(durations[0] + durations[1])
        self._waypoint_positions.append(target_state[:3])
        self._waypoint_labels.append("Target")

        target_waypoint_cost = self.create_cost_model(
            target_state=target_state,
            state_weight=state_weight,
            control_weight=control_weight,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier
        )
        target_waypoint_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, target_waypoint_cost
        )
        target_waypoint_int = crocoddyl.IntegratedActionModelEuler(target_waypoint_diff, dt)
        running_models.append(target_waypoint_int)

        normal_cost_2 = self.create_cost_model(
            target_state=target_state,
            state_weight=state_weight,
            control_weight=control_weight
        )
        normal_diff_2 = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, normal_cost_2
        )
        normal_int_2 = crocoddyl.IntegratedActionModelEuler(normal_diff_2, dt)

        for _ in range(n_steps_2 - 1):
            running_models.append(normal_int_2)

        # Thrust constraints
        if use_thrust_constraints:
            platform = self.s500_config['platform']
            u_lb = np.array([platform['min_thrust']] * 4 + [-2.0] * 2)
            u_ub = np.array([platform['max_thrust']] * 4 + [2.0] * 2)
            for m in running_models:
                m.u_lb = u_lb
                m.u_ub = u_ub

        # Terminal model
        terminal_cost = self.create_cost_model(
            target_state=target_state,
            state_weight=10.0 * state_weight,
            control_weight=control_weight,
            is_terminal=True,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier
        )
        terminal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, terminal_cost
        )
        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)

        self.problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
        total_time = sum(durations)
        print(f"✓ Created trajectory problem: {len(running_models)} nodes, {total_time:.2f}s total")

    def create_trajectory_problem_simple(self,
                                        start_state: np.ndarray,
                                        target_state: np.ndarray,
                                        duration: float,
                                        dt: float = 0.02,
                                        waypoint_multiplier: float = 1000.0,
                                        state_weight: float = 1.0,
                                        control_weight: float = 1e-5,
                                        use_thrust_constraints: bool = True) -> None:
        """
        Create trajectory optimization problem: start -> target (no grasp point)

        Simpler case for better convergence when only initial and target states are needed.

        Args:
            start_state: Initial state [x,y,z,qx,qy,qz,qw,j1,j2,vx,vy,vz,wx,wy,wz,j1_dot,j2_dot]
            target_state: Final target state
            duration: Total trajectory duration (seconds)
            dt: Time step
            waypoint_multiplier: Waypoint weight multiplier
            use_thrust_constraints: Apply thrust limits
        """
        self.dt = dt

        x0 = np.array(start_state, dtype=float).copy()
        if len(x0) != self.robot_model.nq + self.robot_model.nv:
            raise ValueError(f"start_state must have {self.robot_model.nq + self.robot_model.nv} elements (q+v), got {len(x0)}")

        self._waypoint_times = [0.0, duration]
        self._waypoint_positions = [start_state[:3], target_state[:3]]
        self._waypoint_labels = ["Start", "Target"]

        n_steps = max(1, int(duration / dt))
        running_models = []

        # First step: waypoint with enhanced weight
        waypoint_cost = self.create_cost_model(
            target_state=target_state,
            grasp_position=None,
            state_weight=state_weight,
            control_weight=control_weight,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier
        )
        waypoint_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, waypoint_cost
        )
        waypoint_int = crocoddyl.IntegratedActionModelEuler(waypoint_diff, dt)
        running_models.append(waypoint_int)

        # Remaining steps
        normal_cost = self.create_cost_model(
            target_state=target_state,
            state_weight=state_weight,
            control_weight=control_weight
        )
        normal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, normal_cost
        )
        normal_int = crocoddyl.IntegratedActionModelEuler(normal_diff, dt)
        for _ in range(n_steps - 1):
            running_models.append(normal_int)

        if use_thrust_constraints:
            platform = self.s500_config['platform']
            u_lb = np.array([platform['min_thrust']] * 4 + [-2.0] * 2)
            u_ub = np.array([platform['max_thrust']] * 4 + [2.0] * 2)
            for m in running_models:
                m.u_lb = u_lb
                m.u_ub = u_ub

        terminal_cost = self.create_cost_model(
            target_state=target_state,
            state_weight=10.0 * state_weight,
            control_weight=control_weight,
            is_terminal=True,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier
        )
        terminal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, terminal_cost
        )
        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)

        self.problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
        print(f"✓ Created simple trajectory problem: {len(running_models)} nodes, {duration:.2f}s total")

    def create_trajectory_problem_waypoints(self,
                                          waypoints: List[np.ndarray],
                                          durations: List[float],
                                          dt: float = 0.02,
                                          waypoint_multiplier: float = 1000.0,
                                          state_weight: float = 1.0,
                                          control_weight: float = 1e-5,
                                          use_thrust_constraints: bool = True) -> None:
        """
        Create trajectory optimization problem with multiple waypoints.
        waypoints: list of full states [x,y,z,qx,qy,qz,qw,j1,j2,vx,vy,vz,wx,wy,wz,j1_dot,j2_dot]
        durations: duration of each segment (len = len(waypoints) - 1)
        """
        if len(waypoints) != len(durations) + 1:
            raise ValueError("Number of waypoints should be one more than number of durations")
        self.dt = dt
        x0 = np.array(waypoints[0], dtype=float).copy()
        if len(x0) != self.robot_model.nq + self.robot_model.nv:
            raise ValueError(f"waypoint must have {self.robot_model.nq + self.robot_model.nv} elements")

        self._waypoint_times = [0.0]
        self._waypoint_positions = [waypoints[0][:3]]
        self._waypoint_labels = ["Start"] + [f"WP{i+1}" for i in range(len(waypoints) - 2)] + ["Target"]

        running_models = []
        current_time = 0.0

        for i, duration in enumerate(durations):
            target_state = waypoints[i + 1]
            current_time += duration
            self._waypoint_times.append(current_time)
            self._waypoint_positions.append(target_state[:3])

            n_steps = max(1, int(duration / dt))
            waypoint_cost = self.create_cost_model(
                target_state=target_state,
                state_weight=state_weight,
                control_weight=control_weight,
                is_waypoint=True,
                waypoint_multiplier=waypoint_multiplier
            )
            waypoint_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, waypoint_cost
            )
            waypoint_int = crocoddyl.IntegratedActionModelEuler(waypoint_diff, dt)
            running_models.append(waypoint_int)

            normal_cost = self.create_cost_model(
                target_state=target_state,
                state_weight=state_weight,
                control_weight=control_weight
            )
            normal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, normal_cost
            )
            normal_int = crocoddyl.IntegratedActionModelEuler(normal_diff, dt)
            for _ in range(n_steps - 1):
                running_models.append(normal_int)

        if use_thrust_constraints:
            platform = self.s500_config['platform']
            u_lb = np.array([platform['min_thrust']] * 4 + [-2.0] * 2)
            u_ub = np.array([platform['max_thrust']] * 4 + [2.0] * 2)
            for m in running_models:
                m.u_lb = u_lb
                m.u_ub = u_ub

        terminal_target = waypoints[-1]
        terminal_cost = self.create_cost_model(
            target_state=terminal_target,
            state_weight=10.0 * state_weight,
            control_weight=control_weight,
            is_terminal=True,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier
        )
        terminal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, terminal_cost
        )
        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)

        self.problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
        total_time = sum(durations)
        print(f"✓ Created waypoint trajectory: {len(waypoints)} waypoints, {len(running_models)} nodes, {total_time:.2f}s total")

    def solve_trajectory(self, max_iter: int = 150, verbose: bool = True) -> bool:
        """Solve trajectory optimization"""
        if self.problem is None:
            raise RuntimeError("Create trajectory problem first")

        self.solver = crocoddyl.SolverBoxDDP(self.problem)
        self.solver.convergence_init = 1e-12
        self.solver.convergence_stop = 1e-12

        self._cost_logger = crocoddyl.CallbackLogger()
        callbacks = [self._cost_logger]
        if verbose:
            callbacks.append(crocoddyl.CallbackVerbose())
        self.solver.setCallbacks(callbacks)

        print("Solving trajectory optimization...")
        start_time = time.time()
        converged = self.solver.solve([], [], max_iter)
        elapsed = (time.time() - start_time) * 1000
        print(f"✓ Done: {elapsed:.1f} ms, converged={converged}, cost={self.solver.cost:.6f}")
        return converged

    def get_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get optimized states and controls"""
        if self.solver is None:
            raise RuntimeError("Solve trajectory first")
        return self.solver.xs, self.solver.us

    def get_ee_positions(self) -> np.ndarray:
        """Compute end-effector positions along trajectory"""
        if self.solver is None:
            raise RuntimeError("Solve trajectory first")
        xs = self.solver.xs
        ee_positions = []
        for x in xs:
            q = x[:self.robot_model.nq]
            v = x[self.robot_model.nq:]
            pin.forwardKinematics(self.robot_model, self.robot_data, q, v)
            pin.updateFramePlacements(self.robot_model, self.robot_data)
            pos = self.robot_data.oMf[self.ee_frame_id].translation.copy()
            ee_positions.append(pos)
        return np.array(ee_positions)

    def _identify_waypoint_indices(self) -> List[int]:
        if not hasattr(self, '_waypoint_times') or self.dt is None:
            return []
        return [int(t / self.dt) for t in self._waypoint_times]

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [qx,qy,qz,qw] to Euler angles (roll, pitch, yaw) in rad."""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.column_stack([roll, pitch, yaw])

    def _build_plot_figure(self, title: str = "S500 UAM Trajectory", show_waypoints: bool = True,
                           fig=None, timing_info: Optional[dict] = None):
        """4x4 grid: base / EE kinematics, arm & controls, trajectories & solver stats."""
        cache = getattr(self, '_plot_cache', None)
        if cache:
            states = np.array(cache['xs'])
            controls = np.array(cache['us'])
            dt = cache.get('dt', 0.02)
            waypoint_times = cache.get('waypoint_times', [])
            waypoint_indices = [int(t / dt) for t in waypoint_times] if show_waypoints and waypoint_times else []
            cost_logger = cache.get('cost_logger')
        elif self.solver is not None:
            states = np.array(self.solver.xs)
            controls = np.array(self.solver.us)
            dt = self.dt or 0.02
            waypoint_indices = self._identify_waypoint_indices() if show_waypoints else []
            cost_logger = getattr(self, '_cost_logger', None)
        else:
            return None
        time_states = np.arange(len(states)) * dt
        time_controls = np.arange(len(controls)) * dt

        nq, nv = self.robot_model.nq, self.robot_model.nv
        positions = states[:, :3]
        quat = states[:, 3:7]
        euler = self._quat_to_euler(quat)
        arm_angles_deg = np.degrees(states[:, 7:9]) if nq >= 9 else np.zeros((len(states), 2))
        velocities = states[:, nq:nq + 3]
        angular_vel = states[:, nq + 3:nq + 6]
        arm_vel = states[:, nq + 6:nq + 8] if nv >= 8 else np.zeros((len(states), 2))

        ee_pos, ee_v, ee_rpy, ee_w = compute_ee_kinematics_along_trajectory(
            states, self.robot_model, self.robot_data, self.ee_frame_id
        )

        def add_waypoint_lines(ax):
            if show_waypoints and waypoint_indices:
                for idx in waypoint_indices:
                    if idx < len(time_states):
                        ax.axvline(x=time_states[idx], color='orange', linestyle='--', alpha=0.5)

        if fig is None:
            fig = plt.figure(figsize=(20, 16))
        else:
            fig.clear()
        fig.suptitle(title, fontsize=12, y=0.98)
        gs = fig.add_gridspec(4, 4, hspace=0.42, wspace=0.32,
                              left=0.05, right=0.98, top=0.93, bottom=0.05)
        tinfo = {'fontsize': 9, 'labelpad': 2}

        # Row 0: Base — position, linear vel, orientation, angular vel
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(time_states, positions[:, 0], 'r-', label='x')
        ax00.plot(time_states, positions[:, 1], 'g-', label='y')
        ax00.plot(time_states, positions[:, 2], 'b-', label='z')
        add_waypoint_lines(ax00)
        ax00.set_xlabel('Time (s)', **tinfo)
        ax00.set_ylabel('Position (m)', **tinfo)
        ax00.set_title('Base Position', fontsize=9)
        ax00.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax01 = fig.add_subplot(gs[0, 1])
        ax01.plot(time_states, velocities[:, 0], 'r-', label='vx')
        ax01.plot(time_states, velocities[:, 1], 'g-', label='vy')
        ax01.plot(time_states, velocities[:, 2], 'b-', label='vz')
        add_waypoint_lines(ax01)
        ax01.set_xlabel('Time (s)', **tinfo)
        ax01.set_ylabel('Velocity (m/s)', **tinfo)
        ax01.set_title('Base Linear Velocity', fontsize=9)
        ax01.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax02 = fig.add_subplot(gs[0, 2])
        ax02.plot(time_states, np.degrees(euler[:, 0]), 'r-', label='roll')
        ax02.plot(time_states, np.degrees(euler[:, 1]), 'g-', label='pitch')
        ax02.plot(time_states, np.degrees(euler[:, 2]), 'b-', label='yaw')
        add_waypoint_lines(ax02)
        ax02.set_xlabel('Time (s)', **tinfo)
        ax02.set_ylabel('Angle (°)', **tinfo)
        ax02.set_title('Base Orientation (Euler)', fontsize=9)
        ax02.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax03 = fig.add_subplot(gs[0, 3])
        ax03.plot(time_states, angular_vel[:, 0], 'r-', label='ωx')
        ax03.plot(time_states, angular_vel[:, 1], 'g-', label='ωy')
        ax03.plot(time_states, angular_vel[:, 2], 'b-', label='ωz')
        add_waypoint_lines(ax03)
        ax03.set_xlabel('Time (s)', **tinfo)
        ax03.set_ylabel('Angular vel (rad/s)', **tinfo)
        ax03.set_title('Base Angular Velocity', fontsize=9)
        ax03.legend(loc='upper right', fontsize=7, framealpha=0.9)

        # Row 1: EE — position, linear vel, orientation (RPY), angular vel
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.plot(time_states, ee_pos[:, 0], 'r-', label='x')
        ax10.plot(time_states, ee_pos[:, 1], 'g-', label='y')
        ax10.plot(time_states, ee_pos[:, 2], 'b-', label='z')
        add_waypoint_lines(ax10)
        ax10.set_xlabel('Time (s)', **tinfo)
        ax10.set_ylabel('Position (m)', **tinfo)
        ax10.set_title('EE Position', fontsize=9)
        ax10.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax11 = fig.add_subplot(gs[1, 1])
        ax11.plot(time_states, ee_v[:, 0], 'r-', label='vx')
        ax11.plot(time_states, ee_v[:, 1], 'g-', label='vy')
        ax11.plot(time_states, ee_v[:, 2], 'b-', label='vz')
        add_waypoint_lines(ax11)
        ax11.set_xlabel('Time (s)', **tinfo)
        ax11.set_ylabel('Velocity (m/s)', **tinfo)
        ax11.set_title('EE Linear Velocity', fontsize=9)
        ax11.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax12 = fig.add_subplot(gs[1, 2])
        ax12.plot(time_states, np.degrees(ee_rpy[:, 0]), 'r-', label='roll')
        ax12.plot(time_states, np.degrees(ee_rpy[:, 1]), 'g-', label='pitch')
        ax12.plot(time_states, np.degrees(ee_rpy[:, 2]), 'b-', label='yaw')
        add_waypoint_lines(ax12)
        ax12.set_xlabel('Time (s)', **tinfo)
        ax12.set_ylabel('Angle (°)', **tinfo)
        ax12.set_title('EE Orientation (RPY)', fontsize=9)
        ax12.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax13 = fig.add_subplot(gs[1, 3])
        ax13.plot(time_states, ee_w[:, 0], 'r-', label='ωx')
        ax13.plot(time_states, ee_w[:, 1], 'g-', label='ωy')
        ax13.plot(time_states, ee_w[:, 2], 'b-', label='ωz')
        add_waypoint_lines(ax13)
        ax13.set_xlabel('Time (s)', **tinfo)
        ax13.set_ylabel('Angular vel (rad/s)', **tinfo)
        ax13.set_title('EE Angular Velocity', fontsize=9)
        ax13.legend(loc='upper right', fontsize=7, framealpha=0.9)

        # Row 2: Arm joints, arm joint rates, base control, arm control
        ax20 = fig.add_subplot(gs[2, 0])
        ax20.plot(time_states, arm_angles_deg[:, 0], 'r-', label='j1')
        ax20.plot(time_states, arm_angles_deg[:, 1], 'g-', label='j2')
        add_waypoint_lines(ax20)
        ax20.set_xlabel('Time (s)', **tinfo)
        ax20.set_ylabel('Angle (°)', **tinfo)
        ax20.set_title('Arm Joint Angles', fontsize=9)
        ax20.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax21 = fig.add_subplot(gs[2, 1])
        ax21.plot(time_states, arm_vel[:, 0], 'r-', label='j1_dot')
        ax21.plot(time_states, arm_vel[:, 1], 'g-', label='j2_dot')
        add_waypoint_lines(ax21)
        ax21.set_xlabel('Time (s)', **tinfo)
        ax21.set_ylabel('Angular vel (rad/s)', **tinfo)
        ax21.set_title('Arm Joint Angular Velocity', fontsize=9)
        ax21.legend(loc='upper right', fontsize=7, framealpha=0.9)

        colors = ['r', 'g', 'b', 'orange']
        ax22 = fig.add_subplot(gs[2, 2])
        for i in range(min(4, controls.shape[1])):
            ax22.plot(time_controls, controls[:, i], color=colors[i], label=f'T{i+1}')
        ax22.set_xlabel('Time (s)', **tinfo)
        ax22.set_ylabel('Thrust (N)', **tinfo)
        ax22.set_title('Base Control (Thrusters)', fontsize=9)
        ax22.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax23 = fig.add_subplot(gs[2, 3])
        if controls.shape[1] >= 6:
            ax23.plot(time_controls, controls[:, 4], 'r-', label='τ1')
            ax23.plot(time_controls, controls[:, 5], 'g-', label='τ2')
        ax23.set_xlabel('Time (s)', **tinfo)
        ax23.set_ylabel('Torque (N·m)', **tinfo)
        ax23.set_title('Arm Control (Joint Torques)', fontsize=9)
        ax23.legend(loc='upper right', fontsize=7, framealpha=0.9)

        # Row 3: Horizontal (XY), Vertical (XZ), Cost, Time per iteration
        ax30 = fig.add_subplot(gs[3, 0])
        ax30.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, label='Base')
        ax30.plot(ee_pos[:, 0], ee_pos[:, 1], 'm--', linewidth=1.2, label='EE')
        ax30.plot(positions[0, 0], positions[0, 1], 'go', markersize=6, label='Start')
        ax30.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=6, label='End')
        ax30.set_xlabel('X (m)', **tinfo)
        ax30.set_ylabel('Y (m)', **tinfo)
        ax30.set_title('Horizontal trajectory (XY)', fontsize=9)
        ax30.axis('equal')
        ax30.grid(True, alpha=0.3)
        ax30.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax31 = fig.add_subplot(gs[3, 1])
        ax31.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=1.5, label='Base')
        ax31.plot(ee_pos[:, 0], ee_pos[:, 2], 'm--', linewidth=1.2, label='EE')
        ax31.plot(positions[0, 0], positions[0, 2], 'go', markersize=6, label='Start')
        ax31.plot(positions[-1, 0], positions[-1, 2], 'rs', markersize=6, label='End')
        ax31.set_xlabel('X (m)', **tinfo)
        ax31.set_ylabel('Z (m)', **tinfo)
        ax31.set_title('Vertical profile (XZ)', fontsize=9)
        ax31.grid(True, alpha=0.3)
        ax31.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax32 = fig.add_subplot(gs[3, 2])
        if cost_logger is not None and hasattr(cost_logger, 'costs') and len(cost_logger.costs) > 0:
            ax32.semilogy(cost_logger.costs, 'b-', linewidth=2)
        else:
            ax32.text(0.5, 0.5, 'No cost data', ha='center', va='center', transform=ax32.transAxes)
        ax32.set_xlabel('Iteration', **tinfo)
        ax32.set_ylabel('Cost', **tinfo)
        ax32.set_title('Cost convergence', fontsize=9)
        ax32.grid(True, alpha=0.3)

        ax33 = fig.add_subplot(gs[3, 3])
        if timing_info and timing_info.get('n_iter', 0) and timing_info.get('n_iter', 0) > 0:
            n_it = int(timing_info['n_iter'])
            avg_ms = float(timing_info.get('avg_ms_per_iter', 0))
            iters = np.arange(1, n_it + 1)
            ax33.plot(iters, np.full(n_it, avg_ms), 'g-', linewidth=2, label=f'Avg {avg_ms:.2f} ms/iter')
            ax33.fill_between(iters, 0, np.full(n_it, avg_ms), alpha=0.15, color='g')
            ax33.set_xlabel('Iteration', **tinfo)
            ax33.set_ylabel('Time per iter (ms)', **tinfo)
            ax33.set_title('Solver time / iteration', fontsize=9)
            ax33.legend(loc='upper right', fontsize=7)
            ax33.grid(True, alpha=0.3)
        else:
            ax33.text(0.5, 0.5, 'Timing N/A', ha='center', va='center', transform=ax33.transAxes)
            ax33.set_title('Solver time / iteration', fontsize=9)

        all_axes = [ax00, ax01, ax02, ax03, ax10, ax11, ax12, ax13, ax20, ax21, ax22, ax23, ax30, ax31, ax32, ax33]
        for ax in all_axes:
            ax.tick_params(axis='both', labelsize=8)
        return fig

    def _build_3d_plot_figure(self, fig=None):
        """Build 3D trajectory figure only."""
        cache = getattr(self, '_plot_cache', None)
        if cache:
            states = np.array(cache['xs'])
            positions = states[:, :3]
            ee_positions = cache.get('ee_positions')
            if ee_positions is None:
                ee_positions = positions
            waypoint_positions = cache.get('waypoint_positions', [])
        elif self.solver is not None:
            states = np.array(self.solver.xs)
            positions = states[:, :3]
            ee_positions = self.get_ee_positions()
            waypoint_positions = getattr(self, '_waypoint_positions', [])
        else:
            return None
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
        else:
            fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Base')
        ee_arr = np.array(ee_positions)
        if len(ee_arr.shape) == 2 and ee_arr.shape[0] == len(positions):
            ax.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2], 'm--', linewidth=1.5, label='EE')
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='g', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='r', s=100, label='End')
        for wp in waypoint_positions:
            ax.scatter(wp[0], wp[1], wp[2], color='orange', s=150, marker='*')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend(loc='upper right', fontsize=8)
        # 三轴使用相同尺度
        all_pts = positions.copy()
        if len(np.array(ee_positions).shape) == 2:
            all_pts = np.vstack([all_pts, np.array(ee_positions)])
        if len(waypoint_positions) > 0:
            all_pts = np.vstack([all_pts, np.array(waypoint_positions)])
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5
        if max_range < 0.1:
            max_range = 0.5
        x_mid, y_mid, z_mid = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        return fig

    def plot_trajectory(self, save_path: Optional[str] = None, show_waypoints: bool = True):
        """Plot trajectory results (main + 3D in separate figures)"""
        if self.solver is None:
            print("✗ Solve trajectory first")
            return
        fig_main = self._build_plot_figure(show_waypoints=show_waypoints)
        fig_3d = self._build_3d_plot_figure()
        if save_path:
            if fig_main:
                fig_main.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Plot saved: {save_path}")
            if fig_3d:
                path_3d = save_path.replace('.png', '_3d.png') if save_path.endswith('.png') else str(save_path) + '_3d.png'
                fig_3d.savefig(path_3d, dpi=300, bbox_inches='tight')
                print(f"✓ 3D plot saved: {path_3d}")
        if fig_main or fig_3d:
            plt.show()

    def get_plot_figure(self, title: str = "S500 UAM Trajectory", show_waypoints: bool = True, fig=None,
                        timing_info: Optional[dict] = None):
        """Return main plot figure. If fig provided, plot into it for interactivity."""
        return self._build_plot_figure(title=title, show_waypoints=show_waypoints, fig=fig,
                                       timing_info=timing_info)

    def get_3d_plot_figure(self, fig=None):
        """Return 3D trajectory figure. If fig provided, plot into it for interactivity."""
        return self._build_3d_plot_figure(fig=fig)

    def save_trajectory(self, save_path: str):
        """Save trajectory data"""
        if self.solver is None:
            print("✗ Solve trajectory first")
            return
        states, controls = self.get_trajectory()
        ee_positions = self.get_ee_positions()
        np.savez(save_path,
                 states=np.array(states),
                 controls=np.array(controls),
                 ee_positions=ee_positions,
                 cost=self.solver.cost,
                 iterations=self.solver.iter,
                 s500_config=self.s500_config)
        print(f"✓ Data saved: {save_path}")


def make_uam_state(x, y, z, j1=0, j2=0, yaw=0):
    """Full state: [x,y,z, qx,qy,qz,qw, j1,j2, vx,vy,vz, wx,wy,wz, j1_dot,j2_dot].
    yaw: rotation around world z-axis (rad). For yaw-only: q = [0, 0, sin(yaw/2), cos(yaw/2)]."""
    s = np.zeros(17)
    s[0], s[1], s[2] = x, y, z
    half = yaw / 2
    s[3], s[4] = 0.0, 0.0
    s[5] = np.sin(half)
    s[6] = np.cos(half)
    s[7], s[8] = j1, j2
    return s


def create_uam_grasp_waypoints() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Create waypoints for UAM grasp trajectory: start -> grasp -> target"""
    start_state = make_uam_state(0, 0, 1.0, j1=-1.2, j2=-0.6)
    grasp_position = np.array([0.5, 0.0, 0.7])
    target_state = make_uam_state(1.0, 0.5, 1.2, j1=-0.8, j2=-0.3)
    durations = [3.0, 3.0]
    return start_state, grasp_position, target_state, durations


def create_uam_simple_waypoints() -> Tuple[np.ndarray, np.ndarray, float]:
    """Create waypoints for simple trajectory: start -> target only"""
    start_state = make_uam_state(0, 0, 1.0, j1=-1.2, j2=-0.6)
    target_state = make_uam_state(1.0, 0.5, 2.0, j1=-0.8, j2=-0.3)
    duration = 5.0
    return start_state, target_state, duration


def main():
    parser = argparse.ArgumentParser(description='S500 UAM Trajectory Planning')
    parser.add_argument('--s500-yaml', type=str, help='S500 config YAML')
    parser.add_argument('--urdf', type=str, help='S500 UAM URDF')
    parser.add_argument('--simple', action='store_true', help='Simple mode: start -> target only (no grasp point)')
    parser.add_argument('--max-iter', type=int, default=150)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--save-dir', type=str, help='Results directory')
    parser.add_argument('--no-thrust-constraints', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    if args.simple:
        print("S500 UAM Trajectory Planning (Simple: Start → Target)")
    else:
        print("S500 UAM Trajectory Planning (Start → Grasp → Target)")
    print("=" * 70)

    planner = S500UAMTrajectoryPlanner(s500_yaml_path=args.s500_yaml, urdf_path=args.urdf)

    if args.simple:
        start_state, target_state, duration = create_uam_simple_waypoints()
        print(f"\nWaypoints:")
        print(f"  Start:  {start_state[:3]}, arm=[{start_state[7]:.2f}, {start_state[8]:.2f}]")
        print(f"  Target: {target_state[:3]}, arm=[{target_state[7]:.2f}, {target_state[8]:.2f}]")
        print(f"  Duration: {duration} s")
        planner.create_trajectory_problem_simple(
            start_state, target_state, duration,
            dt=args.dt, use_thrust_constraints=not args.no_thrust_constraints
        )
        save_name = 's500_uam_simple_trajectory'
    else:
        start_state, grasp_position, target_state, durations = create_uam_grasp_waypoints()
        print(f"\nWaypoints:")
        print(f"  Start:     {start_state[:3]}, arm=[{start_state[7]:.2f}, {start_state[8]:.2f}]")
        print(f"  Grasp EE:  {grasp_position}")
        print(f"  Target:    {target_state[:3]}, arm=[{target_state[7]:.2f}, {target_state[8]:.2f}]")
        print(f"  Durations: {durations} s")
        planner.create_trajectory_problem(
            start_state, grasp_position, target_state, durations,
            dt=args.dt, use_thrust_constraints=not args.no_thrust_constraints
        )
        save_name = 's500_uam_grasp_trajectory'

    converged = planner.solve_trajectory(max_iter=args.max_iter, verbose=True)

    save_dir = args.save_dir or str(Path(__file__).parent.parent / 'results' / 's500_uam_trajectory_optimization')
    os.makedirs(save_dir, exist_ok=True)
    planner.plot_trajectory(save_path=os.path.join(save_dir, f'{save_name}.png'))
    planner.save_trajectory(os.path.join(save_dir, f'{save_name}.npz'))

    if not converged:
        print("Optimization did not converge. Try --simple mode or increase --max-iter.")


if __name__ == "__main__":
    main()
