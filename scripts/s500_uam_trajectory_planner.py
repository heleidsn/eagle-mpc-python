#!/usr/bin/env python3
"""
S500 UAM (UAV with Arm) Trajectory Planning Script
Using Crocoddyl and Pinocchio for trajectory optimization

Features:
- Load S500 UAM geometry from URDF (quadrotor + 2-DOF arm)
- Load Pinocchio model from URDF file
- Support end-effector (gripper_link) position constraints for grasping
- Perform trajectory optimization: start -> grasp point -> target

State: [x,y,z, qx,qy,qz,qw, j1,j2, vx,vy,vz, wx,wy,wz, j1_dot,j2_dot]  (q then v).
  Base (vx,vy,vz) and (wx,wy,wz) are in the floating-base *body* frame (Pinocchio free-flyer);
  plots convert them to world frame for display.
Control: [thrust_1, thrust_2, thrust_3, thrust_4, torque_j1, torque_j2]

Author: Lei He
Date: 2026-02-11
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def quat_xyzw_batch_to_R(quat: np.ndarray) -> np.ndarray:
    """
    Rotation matrix R_wb from body to world for each row.
    quat: (N, 4) with columns [qx, qy, qz, qw] (Pinocchio / state layout).
    Returns R with shape (N, 3, 3) such that v_world = einsum('nij,nj->ni', R, v_body).
    """
    quat = np.asarray(quat, dtype=float)
    if quat.ndim == 1:
        quat = quat.reshape(1, -1)
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    nn = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    nn = np.maximum(nn, 1e-12)
    qx, qy, qz, qw = qx / nn, qy / nn, qz / nn, qw / nn
    R = np.empty((len(quat), 3, 3), dtype=float)
    R[:, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2.0 * (qx * qy - qw * qz)
    R[:, 0, 2] = 2.0 * (qx * qz + qw * qy)
    R[:, 1, 0] = 2.0 * (qx * qy + qw * qz)
    R[:, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2.0 * (qy * qz - qw * qx)
    R[:, 2, 0] = 2.0 * (qx * qz - qw * qy)
    R[:, 2, 1] = 2.0 * (qy * qz + qw * qx)
    R[:, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    return R


def base_lin_ang_world_from_robot_state(simX: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Base linear and angular velocity in world frame from state rows.

    Pinocchio free-flyer tangent: v[0:3] linear in body, v[3:6] angular in body
    (same as s500_uam_acados_model q_dot convention).
    State layout: x = [q(9), v(8)] with q = [x,y,z,qx,qy,qz,qw,j1,j2].
    """
    X = np.asarray(simX, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < 15:
        raise ValueError(f"state needs ≥15 columns for base twist, got {X.shape[1]}")
    quat = X[:, 3:7]
    vb = X[:, 9:12]
    wb = X[:, 12:15]
    R = quat_xyzw_batch_to_R(quat)
    vw = np.einsum("nij,nj->ni", R, vb)
    ww = np.einsum("nij,nj->ni", R, wb)
    return vw, ww


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
        self._use_actuator_first_order = False
        self._tau_cmd = None
        self._plot_cache = None

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

    def _build_thruster_list(self) -> List[crocoddyl.Thruster]:
        """Thruster objects from S500 YAML (same geometry for any Pinocchio copy of this robot)."""
        platform = self.s500_config['platform']
        cf = platform['cf']
        cm = platform['cm']
        rotors = platform['$rotors']
        min_thrust = platform['min_thrust']
        max_thrust = platform['max_thrust']
        thruster_list = []
        for rotor in rotors:
            pos = np.array(rotor['translation'])
            spin_dir = rotor['spin_direction'][0]
            M = pin.SE3(np.eye(3), pos)
            ctorque = abs(spin_dir) * cm / cf
            thruster_type = crocoddyl.ThrusterType.CCW if spin_dir < 0 else crocoddyl.ThrusterType.CW
            thruster = crocoddyl.Thruster(M, ctorque, thruster_type, min_thrust, max_thrust)
            thruster_list.append(thruster)
        return thruster_list

    def thruster_actuation_for_model(self, robot_model: pin.Model):
        """Crocoddyl floating-base thruster actuation bound to a (possibly copied) ``robot_model``."""
        state = crocoddyl.StateMultibody(robot_model)
        thruster_list = self._build_thruster_list()
        actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, thruster_list)
        return state, actuation

    def create_actuation_model(self):
        """Create actuation model (thrusters + arm joint torques)"""
        try:
            self.actuation = crocoddyl.ActuationModelFloatingBaseThrusters(
                self.state, self._build_thruster_list()
            )
            print(f"✓ Actuation: nu={self.actuation.nu} (thrusters + arm torques)")
        except Exception as e:
            print(f"✗ Failed to create actuation: {e}")
            raise

    def align_state_ee_to_world_point(
        self, x_robot: np.ndarray, p_des_world: np.ndarray
    ) -> np.ndarray:
        """
        Translate floating-base origin q[0:3] so the EE frame matches p_des_world.
        Quaternion, arm joints, and velocities unchanged (rigid shift of the whole system).
        """
        x = np.asarray(x_robot, dtype=float).flatten().copy()
        p_des = np.asarray(p_des_world, dtype=float).reshape(3)
        nq, nv = self.robot_model.nq, self.robot_model.nv
        data = self.robot_model.createData()
        q = x[:nq].copy()
        v = x[nq : nq + nv].copy()
        pin.forwardKinematics(self.robot_model, data, q, v)
        pin.updateFramePlacements(self.robot_model, data)
        ee = np.array(data.oMf[self.ee_frame_id].translation, dtype=float).flatten()
        delta = p_des - ee
        x[0] += float(delta[0])
        x[1] += float(delta[1])
        x[2] += float(delta[2])
        return x

    def create_cost_model(self,
                         target_state: np.ndarray = None,
                         grasp_position: np.ndarray = None,
                         grasp_orientation_rpy: Optional[np.ndarray] = None,
                         control_weight: float = 1e-5,
                         state_weight: float = 1,
                         ee_position_weight: float = 0,
                         ee_rotation_weight: float = 0.0,
                         ee_frame_velocity_weight: float = 0.0,
                         ee_frame_velocity_pitch_rate_weight: float = 0.0,
                         ee_velocity_ref_lin: Optional[np.ndarray] = None,
                         ee_velocity_ref_ang: Optional[np.ndarray] = None,
                         is_terminal: bool = False,
                         is_waypoint: bool = False,
                         waypoint_multiplier: float = 10.0,
                         include_state_reg: bool = True) -> crocoddyl.CostModelSum:
        """
        Create cost model

        Args:
            target_state: Reference full state for state_reg (q then v). Ignored for state_reg if
                include_state_reg is False; still used as fallback when include_state_reg is True
                and target_state is None (internal default pose).
            grasp_position: Target end-effector position [x,y,z] (world frame)
            grasp_orientation_rpy: If set with ee_rotation_weight>0, world RPY (rad, Pinocchio ZYX)
                for ``ResidualModelFramePlacement`` together with grasp_position. Otherwise only
                translation cost is used.
            control_weight: Control regularization weight
            state_weight: State tracking weight
            ee_position_weight: End-effector position tracking weight (or translation part of SE3)
            ee_rotation_weight: Rotation part weight for SE3 placement (0 => translation-only)
            ee_frame_velocity_weight: If >0, add ``ResidualModelFrameVelocity`` (LOCAL_WORLD_ALIGNED)
                penalizing deviation from reference spatial velocity; default ref is zero when refs are None.
            ee_frame_velocity_pitch_rate_weight: Weight on the angular-velocity component about **world Y**
                (the middle entry of the 3D angular part in LOCAL_WORLD_ALIGNED, i.e. pitch rate for Z-up).
                Default 0 leaves pitch angular rate unconstrained in the velocity cost; set >0 to penalize it.
            ee_velocity_ref_lin: Desired EE linear velocity in world (m/s), shape (3,); default zero.
            ee_velocity_ref_ang: Desired EE angular velocity in world (rad/s), shape (3,); default zero.
            is_terminal: Terminal cost
            is_waypoint: Waypoint cost (enhanced weight)
            waypoint_multiplier: Weight multiplier for waypoints
            include_state_reg: If False, omit full-state tracking (e.g. grasp approach segment where
                only EE position is specified, not a desired pose at grasp).
        """
        control_dim = self.actuation.nu
        cost_model = crocoddyl.CostModelSum(self.state, control_dim)

        nq, nv = self.robot_model.nq, self.robot_model.nv
        if target_state is None:
            target_state = np.zeros(nq + nv)
            target_state[2] = 1.0   # z
            target_state[6] = 1.0  # qw

        effective_state_weight = float(state_weight)
        effective_control_weight = float(control_weight)
        effective_ee_weight = float(ee_position_weight)
        effective_ee_rot_w = float(ee_rotation_weight)
        effective_ee_vel_w = float(ee_frame_velocity_weight)
        effective_ee_vel_pitch_w = float(ee_frame_velocity_pitch_rate_weight)
        if is_waypoint:
            effective_state_weight *= waypoint_multiplier
            effective_control_weight *= waypoint_multiplier
            effective_ee_weight *= waypoint_multiplier
            effective_ee_rot_w *= float(waypoint_multiplier)
            effective_ee_vel_w *= float(waypoint_multiplier)
            effective_ee_vel_pitch_w *= float(waypoint_multiplier)

        # State cost (optional: off for segments with only task-space targets, e.g. EE at grasp)
        if include_state_reg and effective_state_weight > 0:
            state_activation = crocoddyl.ActivationModelQuad(self.state.ndx)
            state_residual = crocoddyl.ResidualModelState(self.state, target_state, control_dim)
            cost_model.addCost("state_reg",
                              crocoddyl.CostModelResidual(self.state, state_activation, state_residual),
                              effective_state_weight)

        # End-effector task: SE3 placement or translation-only
        if grasp_position is not None and ee_position_weight > 0:
            p_des = np.asarray(grasp_position, dtype=float).reshape(3)
            use_se3 = (
                grasp_orientation_rpy is not None
                and float(ee_rotation_weight) > 0.0
            )
            if use_se3:
                rpy = np.asarray(grasp_orientation_rpy, dtype=float).reshape(3)
                R = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
                T_des = pin.SE3(R, p_des)
                w6 = np.array(
                    [effective_ee_weight] * 3 + [effective_ee_rot_w] * 3,
                    dtype=np.float64,
                )
                pose_act = crocoddyl.ActivationModelWeightedQuad(w6)
                ee_res = crocoddyl.ResidualModelFramePlacement(
                    self.state, self.ee_frame_id, T_des, control_dim
                )
                cost_model.addCost(
                    "ee_placement",
                    crocoddyl.CostModelResidual(self.state, pose_act, ee_res),
                    1.0,
                )
            else:
                ee_residual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, self.ee_frame_id, p_des, control_dim
                )
                cost_model.addCost(
                    "ee_translation",
                    crocoddyl.CostModelResidual(self.state, ee_residual),
                    effective_ee_weight,
                )

        # EE spatial velocity (e.g. stop at pose waypoint): ref default zero in LOCAL_WORLD_ALIGNED
        if effective_ee_vel_w > 0.0:
            v_lin = (
                np.zeros(3, dtype=np.float64)
                if ee_velocity_ref_lin is None
                else np.asarray(ee_velocity_ref_lin, dtype=np.float64).reshape(3)
            )
            v_ang = (
                np.zeros(3, dtype=np.float64)
                if ee_velocity_ref_ang is None
                else np.asarray(ee_velocity_ref_ang, dtype=np.float64).reshape(3)
            )
            vel_motion_ref = pin.Motion(v_lin, v_ang)
            rf = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            # Residual order: [v_x, v_y, v_z, ω_x, ω_y, ω_z] in LOCAL_WORLD_ALIGNED; ω_y ≈ pitch rate (Z-up).
            w6 = np.array(
                [
                    effective_ee_vel_w,
                    effective_ee_vel_w,
                    effective_ee_vel_w,
                    effective_ee_vel_w,
                    effective_ee_vel_pitch_w,
                    effective_ee_vel_w,
                ],
                dtype=np.float64,
            )
            vel_act = crocoddyl.ActivationModelWeightedQuad(w6)
            vel_res = None
            if hasattr(crocoddyl, "ResidualModelFrameVelocityTpl"):
                try:
                    vel_res = crocoddyl.ResidualModelFrameVelocityTpl(
                        self.state, self.ee_frame_id, vel_motion_ref, rf, control_dim
                    )
                except Exception:
                    vel_res = None
            if vel_res is None:
                vel_res = crocoddyl.ResidualModelFrameVelocity(
                    self.state, self.ee_frame_id, vel_motion_ref, rf, control_dim
                )
            cost_model.addCost(
                "ee_velocity",
                crocoddyl.CostModelResidual(self.state, vel_act, vel_res),
                1.0,
            )

        # Control cost
        if not is_terminal:
            mass = self.robot_model.inertias[1].mass
            hover_thrust = mass * 9.81 / 4
            control_ref = np.array([hover_thrust] * 4 + [0.0] * (control_dim - 4))
            # u = [T1..T4, τ_j1, τ_j2]: penalize arm torques 100× more than thrust (smoother arm u).
            w_u = np.ones(control_dim, dtype=np.float64)
            if control_dim >= 6:
                w_u[4] = 100.0
                w_u[5] = 100.0
            control_activation = crocoddyl.ActivationModelWeightedQuad(w_u)
            control_residual = crocoddyl.ResidualModelControl(self.state, control_ref)
            cost_model.addCost("control_reg",
                              crocoddyl.CostModelResidual(self.state, control_activation, control_residual),
                              effective_control_weight)

        return cost_model

    def create_trajectory_problem_grasp(self,
                                 start_state: np.ndarray,
                                 grasp_position: np.ndarray,
                                 target_state: np.ndarray,
                                 durations: List[float],
                                 dt: float = 0.02,
                                 grasp_ee_weight: float = 500.0,
                                 waypoint_multiplier: float = 500.0,
                                 state_weight: float = 1.0,
                                 control_weight: float = 1.0,
                                 use_thrust_constraints: bool = True,
                                 use_actuator_first_order: bool = False,
                                 tau_cmd: Optional[np.ndarray] = None) -> None:
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

        Per-segment first-step costs (including grasp EE) use waypoint_multiplier × N_segment;
        terminal state cost uses an extra factor N_total (sum of both segments' node counts).
        """
        if len(durations) != 2:
            raise ValueError("durations must have 2 elements: [to_grasp, to_target]")

        self.dt = dt
        self._use_actuator_first_order = bool(use_actuator_first_order)
        self._tau_cmd = None if tau_cmd is None else np.asarray(tau_cmd, dtype=float).reshape(-1)
        self._plot_cache = None

        # Initial state: full state (q,v) for ShootingProblem
        x0 = np.array(start_state, dtype=float).copy()
        if len(x0) != self.robot_model.nq + self.robot_model.nv:
            raise ValueError(f"start_state must have {self.robot_model.nq + self.robot_model.nv} elements (q+v), got {len(x0)}")

        self._waypoint_times = [0.0]
        self._waypoint_positions = [start_state[:3]]
        self._waypoint_labels = ["Start"]
        self._waypoint_ee_positions = [self.get_ee_position_from_state(start_state)]

        running_models = []

        # Segment 1: Start -> Grasp
        n_steps_1 = max(1, int(durations[0] / dt))
        scale_seg1 = float(max(1, n_steps_1))
        self._waypoint_times.append(durations[0])
        self._waypoint_positions.append(grasp_position)
        self._waypoint_labels.append("Grasp")
        self._waypoint_ee_positions.append(np.asarray(grasp_position, dtype=float).reshape(-1)[:3])

        # Grasp waypoint: only EE pose is specified at grasp (no full target state there)
        grasp_cost = self.create_cost_model(
            target_state=target_state,
            grasp_position=grasp_position,
            ee_position_weight=grasp_ee_weight,
            state_weight=state_weight,
            control_weight=control_weight,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier * scale_seg1,
            include_state_reg=False,
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
            control_weight=control_weight,
            include_state_reg=False,
        )
        normal_diff_1 = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, normal_cost_1
        )
        normal_int_1 = crocoddyl.IntegratedActionModelEuler(normal_diff_1, dt)

        for _ in range(n_steps_1 - 1):
            running_models.append(normal_int_1)

        # Segment 2: Grasp -> Target
        n_steps_2 = max(1, int(durations[1] / dt))
        n_total = n_steps_1 + n_steps_2
        terminal_scale = float(max(1, n_total))
        scale_seg2 = float(max(1, n_steps_2))
        self._waypoint_times.append(durations[0] + durations[1])
        self._waypoint_positions.append(target_state[:3])
        self._waypoint_labels.append("Target")
        self._waypoint_ee_positions.append(self.get_ee_position_from_state(target_state))

        target_waypoint_cost = self.create_cost_model(
            target_state=target_state,
            state_weight=state_weight,
            control_weight=control_weight,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier * scale_seg2
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
            state_weight=10.0 * state_weight * terminal_scale,
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
        print(f"✓ Created grasp trajectory: {len(running_models)} nodes, terminal scale N={n_total:.0f}, {total_time:.2f}s total")

    create_trajectory_problem = create_trajectory_problem_grasp

    def create_trajectory_problem_waypoints(self,
                                          waypoints: List[np.ndarray],
                                          durations: List[float],
                                          dt: float = 0.02,
                                          waypoint_multiplier: float = 5000.0,
                                          state_weight: float = 1.0,
                                          control_weight: float = 1e-5,
                                          use_thrust_constraints: bool = True,
                                          use_actuator_first_order: bool = False,
                                          tau_cmd: Optional[np.ndarray] = None) -> None:
        """
        Create trajectory optimization problem with multiple waypoints.
        waypoints: list of full states [x,y,z,qx,qy,qz,qw,j1,j2,vx,vy,vz,wx,wy,wz,j1_dot,j2_dot]
        durations: duration of each segment (len = len(waypoints) - 1)

        Waypoint multiplier is applied only to the *boundary knot* corresponding to each
        intermediate waypoint state. Concretely, we apply the boosted (is_waypoint) cost on
        the first running model of each segment, whose evaluated state equals the segment's
        start waypoint (i.e., the previous segment's end). The final target state is enforced
        by the terminal cost only, avoiding double weighting against the same state.
        Terminal state cost is scaled by total node count N_total for running vs terminal balance.
        """
        if len(waypoints) != len(durations) + 1:
            raise ValueError("Number of waypoints should be one more than number of durations")
        self.dt = dt
        self._use_actuator_first_order = bool(use_actuator_first_order)
        self._tau_cmd = None if tau_cmd is None else np.asarray(tau_cmd, dtype=float).reshape(-1)
        self._plot_cache = None
        x0 = np.array(waypoints[0], dtype=float).copy()
        if len(x0) != self.robot_model.nq + self.robot_model.nv:
            raise ValueError(f"waypoint must have {self.robot_model.nq + self.robot_model.nv} elements")

        segment_n_steps = [max(1, int(d / dt)) for d in durations]
        n_total = int(sum(segment_n_steps))
        terminal_scale = float(max(1, n_total))

        self._waypoint_times = [0.0]
        self._waypoint_positions = [waypoints[0][:3]]
        self._waypoint_labels = ["Start"] + [f"WP{i+1}" for i in range(len(waypoints) - 2)] + ["Target"]
        self._waypoint_ee_positions = [self.get_ee_position_from_state(wp) for wp in waypoints]

        running_models = []
        current_time = 0.0

        for i, duration in enumerate(durations):
            start_state = waypoints[i]
            target_state = waypoints[i + 1]
            current_time += duration
            self._waypoint_times.append(current_time)
            self._waypoint_positions.append(target_state[:3])

            n_steps = segment_n_steps[i]
            # First running model evaluates the current knot state, which at segment start
            # equals the corresponding waypoint_i state. So apply the waypoint multiplier here.
            waypoint_cost = self.create_cost_model(
                target_state=start_state,
                state_weight=state_weight,
                control_weight=control_weight,
                is_waypoint=True,
                waypoint_multiplier=waypoint_multiplier,
            )
            waypoint_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, waypoint_cost
            )
            waypoint_int = crocoddyl.IntegratedActionModelEuler(waypoint_diff, dt)
            running_models.append(waypoint_int)

            # Remaining running models track the segment target (end waypoint) with normal weights.
            normal_cost = self.create_cost_model(
                target_state=target_state,
                state_weight=state_weight,
                control_weight=control_weight,
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
            state_weight=waypoint_multiplier * state_weight,
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
        print(f"✓ Created waypoint trajectory: {len(waypoints)} waypoints, {len(running_models)} nodes, terminal scale N={n_total:.0f}, {total_time:.2f}s total")

    def create_trajectory_problem_mixed_waypoints(
        self,
        modes: List[str],
        resolved_states: List[np.ndarray],
        ee_targets: List[Optional[np.ndarray]],
        durations: List[float],
        dt: float = 0.02,
        waypoint_multiplier: float = 5000.0,
        state_weight: float = 1.0,
        control_weight: float = 1e-5,
        ee_knot_weight: float = 5000.0,
        ee_knot_state_reg_weight: float = 0.0,
        ee_pose_rpy_world: Optional[List[Optional[np.ndarray]]] = None,
        ee_knot_rotation_weight: float = 0.0,
        ee_knot_velocity_weight: float = 200.0,
        ee_knot_velocity_pitch_weight: float = 0.0,
        use_thrust_constraints: bool = True,
        use_actuator_first_order: bool = False,
        tau_cmd: Optional[np.ndarray] = None,
    ) -> None:
        """
        Multi-waypoint problem where each knot may be a full-state (base) constraint or an
        EE task cost (translation-only or SE3 placement), similar to YAML stages mixing
        state_reg and translation_ee costs.

        modes[i]: "base" | "ee_pose" | "ee_pos" — cost at the first running node of segment i.
        resolved_states[i]: nominal full state at knot i (EE rows: base translated so EE matches target xyz).
        ee_targets[i]: world [x,y,z] when modes[i] is ee_pose or ee_pos, else None.
        ee_pose_rpy_world[i]: EE orientation (rad, Pinocchio ZYX roll-pitch-yaw) when modes[i]=="ee_pose";
            used with ee_knot_rotation_weight > 0 for ``ResidualModelFramePlacement``; otherwise translation-only.
        ee_pos: A,B,C in the table are j1/j2/yaw seeds (deg), not EE orientation.
        ee_knot_velocity_weight: For ``ee_pose`` knots only, weight on ``ResidualModelFrameVelocity``
            with reference spatial velocity zero (LOCAL_WORLD_ALIGNED). Set 0 to disable.
        ee_knot_velocity_pitch_weight: Weight on angular velocity about world Y (pitch rate) in that
            residual; 0 leaves pitch rate free (default). Use same scale as ee_knot_velocity_weight to
            penalize pitch rate like other angular axes.
        durations: len N-1 segment lengths [s].
        """
        n = len(modes)
        if n != len(resolved_states) or n != len(ee_targets) or len(durations) != n - 1:
            raise ValueError("modes, resolved_states, ee_targets lengths must match; len(durations)==N-1")
        if ee_pose_rpy_world is None:
            ee_pose_rpy_world = [None] * n
        elif len(ee_pose_rpy_world) != n:
            raise ValueError("ee_pose_rpy_world must have same length as modes")
        self.dt = dt
        self._use_actuator_first_order = bool(use_actuator_first_order)
        self._tau_cmd = None if tau_cmd is None else np.asarray(tau_cmd, dtype=float).reshape(-1)
        self._plot_cache = None

        x0 = np.array(resolved_states[0], dtype=float).copy()
        nvq = self.robot_model.nq + self.robot_model.nv
        if len(x0) != nvq:
            raise ValueError(f"resolved state must have {nvq} elements")

        segment_n_steps = [max(1, int(d / dt)) for d in durations]
        n_total = int(sum(segment_n_steps))

        self._waypoint_times = [0.0]
        self._waypoint_positions = [x0[:3]]
        self._waypoint_labels = ["Start"]
        self._waypoint_ee_positions = [self.get_ee_position_from_state(x0)]
        for i in range(1, n):
            self._waypoint_times.append(self._waypoint_times[-1] + float(durations[i - 1]))
            si = np.asarray(resolved_states[i], dtype=float).flatten()
            self._waypoint_positions.append(si[:3])
            is_last = i == n - 1
            mi = str(modes[i]).lower()
            if mi in ("ee_pose", "ee_pos") and ee_targets[i] is not None:
                self._waypoint_ee_positions.append(np.asarray(ee_targets[i], dtype=float).reshape(3).copy())
                if mi == "ee_pose":
                    self._waypoint_labels.append(
                        "Target (EE pose)" if is_last else f"WP{i}(EE pose)"
                    )
                else:
                    self._waypoint_labels.append("Target (EE)" if is_last else f"WP{i}(EE)")
            else:
                self._waypoint_ee_positions.append(self.get_ee_position_from_state(si))
                self._waypoint_labels.append("Target" if is_last else f"WP{i}")

        running_models = []
        for i, duration in enumerate(durations):
            start_state = np.asarray(resolved_states[i], dtype=float).flatten()
            target_state = np.asarray(resolved_states[i + 1], dtype=float).flatten()
            mode_i = str(modes[i]).lower()

            if mode_i in ("ee_pose", "ee_pos") and ee_targets[i] is not None:
                ee_w = float(ee_knot_weight)
                sr_w = float(ee_knot_state_reg_weight)
                rpy_des = None
                rot_w = 0.0
                vel_w = 0.0
                vel_pitch_w = 0.0
                if mode_i == "ee_pose":
                    slot = ee_pose_rpy_world[i]
                    if slot is not None and float(ee_knot_rotation_weight) > 0.0:
                        rpy_des = np.asarray(slot, dtype=float).reshape(3)
                        rot_w = float(ee_knot_rotation_weight)
                    vel_w = float(ee_knot_velocity_weight)
                    vel_pitch_w = float(ee_knot_velocity_pitch_weight)
                waypoint_cost = self.create_cost_model(
                    target_state=start_state,
                    grasp_position=np.asarray(ee_targets[i], dtype=float).reshape(3),
                    grasp_orientation_rpy=rpy_des,
                    state_weight=max(sr_w, 1e-12),
                    control_weight=control_weight,
                    ee_position_weight=ee_w,
                    ee_rotation_weight=rot_w,
                    ee_frame_velocity_weight=vel_w,
                    ee_frame_velocity_pitch_rate_weight=vel_pitch_w,
                    is_waypoint=True,
                    waypoint_multiplier=waypoint_multiplier,
                    include_state_reg=(sr_w > 0),
                )
            else:
                waypoint_cost = self.create_cost_model(
                    target_state=start_state,
                    state_weight=state_weight,
                    control_weight=control_weight,
                    is_waypoint=True,
                    waypoint_multiplier=waypoint_multiplier,
                )
            waypoint_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, waypoint_cost
            )
            waypoint_int = crocoddyl.IntegratedActionModelEuler(waypoint_diff, dt)
            running_models.append(waypoint_int)

            n_steps = segment_n_steps[i]
            normal_cost = self.create_cost_model(
                target_state=target_state,
                state_weight=state_weight,
                control_weight=control_weight,
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

        terminal_target = np.asarray(resolved_states[-1], dtype=float).flatten()
        terminal_cost = self.create_cost_model(
            target_state=terminal_target,
            state_weight=waypoint_multiplier * state_weight,
            control_weight=control_weight,
            is_terminal=True,
            is_waypoint=True,
            waypoint_multiplier=waypoint_multiplier,
        )
        terminal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, terminal_cost
        )
        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)

        self.problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
        total_time = sum(durations)
        n_ee = sum(1 for m in modes if str(m).lower() in ("ee_pose", "ee_pos"))
        print(
            f"✓ Created mixed waypoint trajectory: {n} knots ({n_ee} EE), "
            f"{len(running_models)} nodes, terminal scale N={n_total:.0f}, {total_time:.2f}s total"
        )

    def solve_trajectory(self, max_iter: int = 150, verbose: bool = True) -> bool:
        """Solve trajectory optimization"""
        if self.problem is None:
            raise RuntimeError("Create trajectory problem first")

        self.solver = crocoddyl.SolverBoxFDDP(self.problem)
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
        self._refresh_plot_cache()
        return converged

    def _effective_tau_cmd(self) -> np.ndarray:
        nu = int(self.actuation.nu)
        default_tau = np.array([0.06, 0.06, 0.06, 0.06, 0.05, 0.05], dtype=float)
        if self._tau_cmd is None:
            tau = default_tau
        else:
            tau = np.asarray(self._tau_cmd, dtype=float).reshape(-1)
        if tau.size != nu:
            if tau.size < nu:
                pad = np.full(nu - tau.size, float(tau[-1] if tau.size > 0 else 0.05), dtype=float)
                tau = np.concatenate([tau, pad])
            else:
                tau = tau[:nu]
        return np.maximum(tau, 1e-4)

    def _rollout_with_actuator_first_order(self, xs_cmd: List[np.ndarray], us_cmd: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self.dt is None:
            return xs_cmd, us_cmd
        if not us_cmd:
            return xs_cmd, us_cmd
        nu = int(self.actuation.nu)
        tau = self._effective_tau_cmd()

        x = np.asarray(xs_cmd[0], dtype=float).flatten().copy()
        u_act = np.asarray(us_cmd[0], dtype=float).flatten().copy()
        if u_act.size != nu:
            u_act = np.zeros(nu, dtype=float)

        zero_cost = crocoddyl.CostModelSum(self.state, nu)
        diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, zero_cost)
        int_model = crocoddyl.IntegratedActionModelEuler(diff, float(self.dt))
        data = int_model.createData()

        xs_roll = [x.copy()]
        us_act = []
        alpha = float(self.dt) / tau
        alpha = np.clip(alpha, 0.0, 1.0)
        for k in range(len(us_cmd)):
            u_cmd = np.asarray(us_cmd[k], dtype=float).flatten()
            if u_cmd.size != nu:
                u_cmd = np.zeros(nu, dtype=float)
            u_act = u_act + alpha * (u_cmd - u_act)
            us_act.append(u_act.copy())
            int_model.calc(data, x, u_act)
            x = np.asarray(data.xnext, dtype=float).flatten().copy()
            xs_roll.append(x.copy())
        return xs_roll, us_act

    def _refresh_plot_cache(self) -> None:
        if self.solver is None:
            self._plot_cache = None
            return
        xs_cmd = [np.asarray(x, dtype=float).copy() for x in self.solver.xs]
        us_cmd = [np.asarray(u, dtype=float).copy() for u in self.solver.us]
        if self._use_actuator_first_order:
            xs_plot, us_plot = self._rollout_with_actuator_first_order(xs_cmd, us_cmd)
        else:
            xs_plot, us_plot = xs_cmd, us_cmd
        try:
            ee_positions = []
            for x in xs_plot:
                ee_positions.append(self.get_ee_position_from_state(x))
            ee_positions = np.asarray(ee_positions, dtype=float)
        except Exception:
            ee_positions = np.zeros((len(xs_plot), 3), dtype=float)
        self._plot_cache = {
            "xs": xs_plot,
            "us": us_plot,
            "dt": self.dt or 0.02,
            "waypoint_times": list(getattr(self, "_waypoint_times", [])),
            "waypoint_positions": [np.asarray(p, dtype=float).copy() for p in getattr(self, "_waypoint_positions", [])],
            "waypoint_labels": list(getattr(self, "_waypoint_labels", [])),
            "waypoint_ee_positions": [np.asarray(p, dtype=float).copy() for p in getattr(self, "_waypoint_ee_positions", [])],
            "cost_logger": getattr(self, "_cost_logger", None),
            "ee_positions": ee_positions,
            "use_actuator_first_order": bool(self._use_actuator_first_order),
            "tau_cmd": self._effective_tau_cmd().copy(),
        }

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

    def get_ee_position_from_state(self, state: np.ndarray) -> np.ndarray:
        """Compute EE position for a single full state vector (q,v)."""
        x = np.asarray(state, dtype=float).reshape(-1)
        nq, nv = self.robot_model.nq, self.robot_model.nv
        if x.size != nq + nv:
            raise ValueError(f"state must have {nq + nv} elements (q+v), got {x.size}")
        q = x[:nq]
        v = x[nq:]
        pin.forwardKinematics(self.robot_model, self.robot_data, q, v)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        return self.robot_data.oMf[self.ee_frame_id].translation.copy()

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
            waypoint_positions = cache.get('waypoint_positions', []) or []
            waypoint_labels = cache.get('waypoint_labels', []) or []
            waypoint_ee_positions = cache.get('waypoint_ee_positions', []) or []
            cost_logger = cache.get('cost_logger')
        elif self.solver is not None:
            states = np.array(self.solver.xs)
            controls = np.array(self.solver.us)
            dt = self.dt or 0.02
            waypoint_indices = self._identify_waypoint_indices() if show_waypoints else []
            waypoint_positions = getattr(self, '_waypoint_positions', []) or []
            waypoint_labels = getattr(self, '_waypoint_labels', []) or []
            waypoint_ee_positions = getattr(self, '_waypoint_ee_positions', []) or []
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
        v_lin_w, w_ang_w = base_lin_ang_world_from_robot_state(states)
        arm_vel = states[:, nq + 6:nq + 8] if nv >= 8 else np.zeros((len(states), 2))

        ee_pos, ee_v, ee_rpy, ee_w = compute_ee_kinematics_along_trajectory(
            states, self.robot_model, self.robot_data, self.ee_frame_id
        )

        def add_waypoint_lines(
            ax,
            show_text: bool = False,
            include_target_vec: bool = False,
            target_vecs: Optional[List[np.ndarray]] = None,
        ):
            """
            Add vertical dashed lines at waypoint times.
            Optionally add a compact label for each waypoint on top of the subplot.
            """
            if not (show_waypoints and waypoint_indices):
                return

            # Labels can get cluttered quickly; only show when reasonably small.
            show_text = bool(show_text) and (len(waypoint_indices) <= 8)
            y_top = ax.get_ylim()[1] if show_text else None
            if target_vecs is None:
                target_vecs = waypoint_positions

            for k, idx in enumerate(waypoint_indices):
                if idx >= len(time_states):
                    continue
                x = time_states[idx]
                ax.axvline(
                    x=x, color='darkorange', linestyle='--',
                    alpha=0.7, linewidth=1.0, zorder=1,
                )
                if show_text and k < len(waypoint_labels) and y_top is not None:
                    label = str(waypoint_labels[k])
                    if include_target_vec and k < len(target_vecs):
                        tgt = np.asarray(target_vecs[k], dtype=float).reshape(-1)
                        if tgt.size >= 3:
                            label = f"{label} tgt=[{tgt[0]:.2f},{tgt[1]:.2f},{tgt[2]:.2f}]"
                    ax.text(
                        x, y_top, label,
                        rotation=90, va='top', ha='right',
                        fontsize=7, color='darkorange',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0),
                        zorder=3,
                    )

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
        add_waypoint_lines(ax00, show_text=True, include_target_vec=True)
        if show_waypoints and waypoint_indices:
            # Mark the *expected* base position at each waypoint boundary time.
            for k, idx in enumerate(waypoint_indices):
                if idx >= len(time_states):
                    continue
                if k >= len(waypoint_positions):
                    continue
                tgt = np.asarray(waypoint_positions[k], dtype=float).reshape(-1)
                if tgt.size < 3:
                    continue
                ax00.scatter(time_states[idx], tgt[0], color='r', s=28, marker='o',
                             zorder=4, edgecolors='k', linewidths=0.3)
                ax00.scatter(time_states[idx], tgt[1], color='g', s=28, marker='o',
                             zorder=4, edgecolors='k', linewidths=0.3)
                ax00.scatter(time_states[idx], tgt[2], color='b', s=28, marker='o',
                             zorder=4, edgecolors='k', linewidths=0.3)
        ax00.set_xlabel('Time (s)', **tinfo)
        ax00.set_ylabel('Position (m)', **tinfo)
        ax00.set_title('Base Position', fontsize=9)
        h0, l0 = ax00.get_legend_handles_labels()
        if show_waypoints and waypoint_indices:
            h0.append(Line2D([0], [0], color='darkorange', linestyle='--', alpha=0.7, linewidth=1.0))
            l0.append('Waypoint')
        ax00.legend(h0, l0, loc='upper right', fontsize=7, framealpha=0.9)

        ax01 = fig.add_subplot(gs[0, 1])
        ax01.plot(time_states, v_lin_w[:, 0], 'r-', label='vx')
        ax01.plot(time_states, v_lin_w[:, 1], 'g-', label='vy')
        ax01.plot(time_states, v_lin_w[:, 2], 'b-', label='vz')
        add_waypoint_lines(ax01)
        ax01.set_xlabel('Time (s)', **tinfo)
        ax01.set_ylabel('Velocity (m/s)', **tinfo)
        ax01.set_title('Base linear vel. (world)', fontsize=9)
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
        ax03.plot(time_states, np.degrees(w_ang_w[:, 0]), 'r-', label='ωx')
        ax03.plot(time_states, np.degrees(w_ang_w[:, 1]), 'g-', label='ωy')
        ax03.plot(time_states, np.degrees(w_ang_w[:, 2]), 'b-', label='ωz')
        add_waypoint_lines(ax03)
        ax03.set_xlabel('Time (s)', **tinfo)
        ax03.set_ylabel('Angular vel (deg/s)', **tinfo)
        ax03.set_title('Base angular vel. (world)', fontsize=9)
        ax03.legend(loc='upper right', fontsize=7, framealpha=0.9)

        # Row 1: EE — position, linear vel, orientation (RPY), angular vel
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.plot(time_states, ee_pos[:, 0], 'r-', label='x')
        ax10.plot(time_states, ee_pos[:, 1], 'g-', label='y')
        ax10.plot(time_states, ee_pos[:, 2], 'b-', label='z')
        add_waypoint_lines(ax10, show_text=True, include_target_vec=True, target_vecs=waypoint_ee_positions)
        ax10.set_xlabel('Time (s)', **tinfo)
        ax10.set_ylabel('Position (m)', **tinfo)
        ax10.set_title('EE Position', fontsize=9)
        if show_waypoints and waypoint_indices:
            # Mark the *expected* EE position at each waypoint boundary time.
            for k, idx in enumerate(waypoint_indices):
                if idx >= len(time_states):
                    continue
                if k >= len(waypoint_ee_positions):
                    continue
                tgt = np.asarray(waypoint_ee_positions[k], dtype=float).reshape(-1)
                if tgt.size < 3:
                    continue
                ax10.scatter(
                    time_states[idx], tgt[0], color='r', s=22, marker='o',
                    zorder=4, edgecolors='k', linewidths=0.3
                )
                ax10.scatter(
                    time_states[idx], tgt[1], color='g', s=22, marker='o',
                    zorder=4, edgecolors='k', linewidths=0.3
                )
                ax10.scatter(
                    time_states[idx], tgt[2], color='b', s=22, marker='o',
                    zorder=4, edgecolors='k', linewidths=0.3
                )
        ax10.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax11 = fig.add_subplot(gs[1, 1])
        ax11.plot(time_states, ee_v[:, 0], 'r-', label='vx')
        ax11.plot(time_states, ee_v[:, 1], 'g-', label='vy')
        ax11.plot(time_states, ee_v[:, 2], 'b-', label='vz')
        add_waypoint_lines(ax11)
        ax11.set_xlabel('Time (s)', **tinfo)
        ax11.set_ylabel('Velocity (m/s)', **tinfo)
        ax11.set_title('EE linear vel. (world)', fontsize=9)
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
        ax13.plot(time_states, np.degrees(ee_w[:, 0]), 'r-', label='ωx')
        ax13.plot(time_states, np.degrees(ee_w[:, 1]), 'g-', label='ωy')
        ax13.plot(time_states, np.degrees(ee_w[:, 2]), 'b-', label='ωz')
        add_waypoint_lines(ax13)
        ax13.set_xlabel('Time (s)', **tinfo)
        ax13.set_ylabel('Angular vel (deg/s)', **tinfo)
        ax13.set_title('EE angular vel. (world)', fontsize=9)
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
        ax21.plot(time_states, np.degrees(arm_vel[:, 0]), 'r-', label='j1_dot')
        ax21.plot(time_states, np.degrees(arm_vel[:, 1]), 'g-', label='j2_dot')
        add_waypoint_lines(ax21)
        ax21.set_xlabel('Time (s)', **tinfo)
        ax21.set_ylabel('Joint rate (deg/s)', **tinfo)
        ax21.set_title('Arm joint angular velocity', fontsize=9)
        ax21.legend(loc='upper right', fontsize=7, framealpha=0.9)

        colors = ['r', 'g', 'b', 'orange']
        ax22 = fig.add_subplot(gs[2, 2])
        for i in range(min(4, controls.shape[1])):
            ax22.plot(time_controls, controls[:, i], color=colors[i], label=f'T{i+1}')
        add_waypoint_lines(ax22)
        ax22.set_xlabel('Time (s)', **tinfo)
        ax22.set_ylabel('Thrust (N)', **tinfo)
        ax22.set_title('Base Control (Thrusters)', fontsize=9)
        ax22.legend(loc='upper right', fontsize=7, framealpha=0.9)

        ax23 = fig.add_subplot(gs[2, 3])
        if controls.shape[1] >= 6:
            ax23.plot(time_controls, controls[:, 4], 'r-', label='τ1')
            ax23.plot(time_controls, controls[:, 5], 'g-', label='τ2')
        add_waypoint_lines(ax23)
        ax23.set_xlabel('Time (s)', **tinfo)
        ax23.set_ylabel('Torque (N·m)', **tinfo)
        ax23.set_title('Arm Control (Joint Torques)', fontsize=9)
        ax23.legend(loc='upper right', fontsize=7, framealpha=0.9)

        # Row 3: Horizontal (XY), Vertical (XZ), Cost, Time per iteration
        ax30 = fig.add_subplot(gs[3, 0])
        ax30.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, label='Base')
        ax30.plot(ee_pos[:, 0], ee_pos[:, 1], 'm--', linewidth=1.2, label='EE')
        if show_waypoints and len(waypoint_positions) > 0:
            wps = np.asarray(waypoint_positions, dtype=float).reshape(-1, 3)
            nwp = wps.shape[0]
            wl = (
                list(waypoint_labels)
                if waypoint_labels and len(waypoint_labels) == nwp
                else [f"P{i}" for i in range(nwp)]
            )
            for k in range(nwp):
                if k == 0:
                    mk, sz, cl = 'o', 64, 'tab:green'
                elif k == nwp - 1:
                    mk, sz, cl = 's', 64, 'tab:red'
                else:
                    mk, sz, cl = '*', 110, 'darkorange'
                ax30.scatter(
                    wps[k, 0], wps[k, 1], c=cl, s=sz, marker=mk, zorder=6,
                    edgecolors='black', linewidths=0.6,
                )
                ax30.annotate(
                    wl[k], (wps[k, 0], wps[k, 1]), xytext=(4, 4),
                    textcoords='offset points', fontsize=7, zorder=7,
                )
        else:
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
        if show_waypoints and len(waypoint_positions) > 0:
            wps = np.asarray(waypoint_positions, dtype=float).reshape(-1, 3)
            nwp = wps.shape[0]
            wl = (
                list(waypoint_labels)
                if waypoint_labels and len(waypoint_labels) == nwp
                else [f"P{i}" for i in range(nwp)]
            )
            for k in range(nwp):
                if k == 0:
                    mk, sz, cl = 'o', 64, 'tab:green'
                elif k == nwp - 1:
                    mk, sz, cl = 's', 64, 'tab:red'
                else:
                    mk, sz, cl = '*', 110, 'darkorange'
                ax31.scatter(
                    wps[k, 0], wps[k, 2], c=cl, s=sz, marker=mk, zorder=6,
                    edgecolors='black', linewidths=0.6,
                )
                ax31.annotate(
                    wl[k], (wps[k, 0], wps[k, 2]), xytext=(4, 4),
                    textcoords='offset points', fontsize=7, zorder=7,
                )
        else:
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
            waypoint_positions = cache.get('waypoint_positions', []) or []
            waypoint_labels = cache.get('waypoint_labels', []) or []
        elif self.solver is not None:
            states = np.array(self.solver.xs)
            positions = states[:, :3]
            ee_positions = self.get_ee_positions()
            waypoint_positions = getattr(self, '_waypoint_positions', []) or []
            waypoint_labels = getattr(self, '_waypoint_labels', []) or []
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
        if len(waypoint_positions) > 0:
            wps = np.asarray(waypoint_positions, dtype=float).reshape(-1, 3)
            nwp = wps.shape[0]
            wl = (
                list(waypoint_labels)
                if waypoint_labels and len(waypoint_labels) == nwp
                else [f"P{i}" for i in range(nwp)]
            )
            for k in range(nwp):
                wp = wps[k]
                if k == 0:
                    mk, sz, cl = 'o', 90, 'tab:green'
                elif k == nwp - 1:
                    mk, sz, cl = 's', 90, 'tab:red'
                else:
                    mk, sz, cl = '*', 130, 'darkorange'
                ax.scatter(
                    wp[0], wp[1], wp[2], color=cl, s=sz, marker=mk, edgecolors='k',
                    linewidths=0.5, zorder=5,
                )
                ax.text(float(wp[0]), float(wp[1]), float(wp[2]), f"  {wl[k]}", fontsize=8, zorder=6)
        else:
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='g', s=100, label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='r', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend(loc='upper right', fontsize=8)
        # Use the same scale for all three axes.
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


def rotation_world_R_body_tool_z_along(
    direction_world: np.ndarray,
    world_up: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rotation ``R`` (3x3) with columns = body x,y,z expressed in world, such that body +Z aligns
    with ``direction_world`` (normalized). Use with Pinocchio ``matrixToRpy`` for EE table entries.

    Typical use: make gripper ``+Z`` (check ``gripper_link`` in RViz/Meshcat) point from the EE
    toward the grasp target. If your approach axis is ``-Z``, negate ``direction_world`` or use
    ``flip=True`` in :func:`rpy_rad_tool_z_toward_point`.
    """
    d = np.asarray(direction_world, dtype=float).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        raise ValueError("direction_world norm too small")
    ez = d / n
    up = np.array([0.0, 0.0, 1.0], dtype=float) if world_up is None else np.asarray(world_up, dtype=float).reshape(3)
    un = float(np.linalg.norm(up))
    if un < 1e-9:
        raise ValueError("world_up norm too small")
    up = up / un
    if abs(float(np.dot(ez, up))) > 0.999:
        up = np.array([1.0, 0.0, 0.0], dtype=float)
    ex = np.cross(up, ez)
    exn = float(np.linalg.norm(ex))
    if exn < 1e-9:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        ex = np.cross(up, ez)
        exn = float(np.linalg.norm(ex))
    ex = ex / max(exn, 1e-12)
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez], axis=1)


def rpy_rad_tool_z_toward_point(
    p_ee: np.ndarray,
    p_target: np.ndarray,
    *,
    world_up: Optional[np.ndarray] = None,
    flip: bool = False,
) -> np.ndarray:
    """
    Roll-pitch-yaw (rad, Pinocchio ZYX / ``rpyToMatrix`` consistent) so that tool +Z points from
    ``p_ee`` toward ``p_target``. Only the direction ``(p_target - p_ee)`` matters for attitude.

    Args:
        p_ee: EE position in world (m), e.g. row ``x,y,z`` or FK sample along approach.
        p_target: Target point in world (m).
        flip: If True, align with ``-(p_target - p_ee)`` (e.g. ``-Z`` approach axis).
        world_up: Reference up vector to resolve roll/yaw ambiguity (default ``[0,0,1]``).
    """
    d = np.asarray(p_target, dtype=float).reshape(3) - np.asarray(p_ee, dtype=float).reshape(3)
    if flip:
        d = -d
    R = rotation_world_R_body_tool_z_along(d, world_up=world_up)
    return pin.rpy.matrixToRpy(R)


def rpy_deg_tool_z_toward_point(
    p_ee: np.ndarray,
    p_target: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Same as :func:`rpy_rad_tool_z_toward_point` but returns degrees for GUI waypoint columns."""
    return np.degrees(rpy_rad_tool_z_toward_point(p_ee, p_target, **kwargs))


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
    start_state = make_uam_state(-1.5, 0, 1.5, j1=0.0, j2=0.0)
    grasp_position = np.array([0.0, 0.0, 0.5])
    target_state = make_uam_state(1.5, 0.0, 1.5, j1=0.0, j2=0.0)
    durations = [5.0, 5.0]
    return start_state, grasp_position, target_state, durations


def create_uam_simple_waypoints() -> Tuple[List[np.ndarray], List[float]]:
    """Default multi-waypoint demo (no grasp): full states and one duration per segment."""
    waypoints = [
        make_uam_state(-1.5, 0.0, 1.5, j1=0.0, j2=0.0),
        make_uam_state(0.0, 0.0, 1.2, j1=0.0, j2=0.0),
        make_uam_state(1.5, 0.0, 1.5, j1=0.0, j2=0.0),
        # make_uam_state(4.5, 1.5, 3.0, j1=1.0, j2=0.8),
    ]
    # durations = [2.0, 2.0, 2.0]
    durations = [5.0, 5.0]
    return waypoints, durations


def main():
    parser = argparse.ArgumentParser(description='S500 UAM Trajectory Planning')
    parser.add_argument('--s500-yaml', type=str, help='S500 config YAML')
    parser.add_argument('--urdf', type=str, help='S500 UAM URDF')
    parser.add_argument(
        '--simple',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Waypoint mode: built-in multi-waypoint demo (no grasp); use --no-simple for grasp trajectory (default: on)',
    )
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--save-dir', type=str, help='Results directory')
    parser.add_argument(
        '--thrust-constraints',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable rotor thrust limits in optimization (default: on); pass --no-thrust-constraints to disable',
    )
    args = parser.parse_args()

    print("=" * 70)
    if args.simple:
        print("S500 UAM Trajectory Planning (Multi-waypoint demo)")
    else:
        print("S500 UAM Trajectory Planning (Start → Grasp → Target)")
    print("=" * 70)

    planner = S500UAMTrajectoryPlanner(s500_yaml_path=args.s500_yaml, urdf_path=args.urdf)

    if args.simple:
        waypoints, durations = create_uam_simple_waypoints()
        print(f"\nWaypoints ({len(waypoints)} points, {len(durations)} segments):")
        labels = (
            ["Start"]
            + [f"WP{i}" for i in range(1, len(waypoints) - 1)]
            + ["Target"]
        )
        for lab, w in zip(labels, waypoints):
            print(f"  {lab:8s} {w[:3]}, arm=[{w[7]:.2f}, {w[8]:.2f}]")
        print(f"  Durations: {durations} s (total {sum(durations):.2f} s)")
        planner.create_trajectory_problem_waypoints(
            waypoints=waypoints,
            durations=durations,
            dt=args.dt,
            use_thrust_constraints=args.thrust_constraints,
        )
        save_name = 's500_uam_simple_trajectory'
    else:
        start_state, grasp_position, target_state, durations = create_uam_grasp_waypoints()
        print(f"\nWaypoints:")
        print(f"  Start:     {start_state[:3]}, arm=[{start_state[7]:.2f}, {start_state[8]:.2f}]")
        print(f"  Grasp EE:  {grasp_position}")
        print(f"  Target:    {target_state[:3]}, arm=[{target_state[7]:.2f}, {target_state[8]:.2f}]")
        print(f"  Durations: {durations} s")
        planner.create_trajectory_problem_grasp(
            start_state, grasp_position, target_state, durations,
            dt=args.dt, use_thrust_constraints=args.thrust_constraints
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
