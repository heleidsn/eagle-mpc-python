#!/usr/bin/env python3
"""平面 2R 臂零空间 IK（机体系 EE 位置目标）— 用于悬停扰动后 EE 位置补偿。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares


def _arm_joint_slice(robot_model: pin.Model) -> slice:
    n_arm = int(robot_model.nq) - 7
    if n_arm < 1:
        raise ValueError(f"expected nq>=8 (base+arm), got nq={robot_model.nq}")
    return slice(7, 7 + n_arm)


def arm_joint_limits(
    robot_model: pin.Model,
    j_angle_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """臂关节上下界 [lo, hi]，各 (n_arm,)。"""
    sl = _arm_joint_slice(robot_model)
    lo = np.asarray(robot_model.lowerPositionLimit[sl], dtype=float).copy()
    hi = np.asarray(robot_model.upperPositionLimit[sl], dtype=float).copy()
    if j_angle_max is not None:
        m = float(j_angle_max)
        lo = np.maximum(lo, -m)
        hi = np.minimum(hi, m)
    return lo, hi


def ee_position_world(
    robot_model: pin.Model,
    data: pin.Data,
    ee_frame_id: int,
    q: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(robot_model.nq)
    pin.forwardKinematics(robot_model, data, q)
    pin.updateFramePlacements(robot_model, data)
    return np.asarray(data.oMf[int(ee_frame_id)].translation, dtype=float).copy()


def ee_position_body_from_joints(
    robot_model: pin.Model,
    data: pin.Data,
    ee_frame_id: int,
    j: np.ndarray,
    quat_xyzw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """EE 在机体系下的位置（base 在原点；姿态默认水平）。"""
    q = np.zeros(robot_model.nq, dtype=float)
    if quat_xyzw is not None:
        q[3:7] = np.asarray(quat_xyzw, dtype=float).reshape(4)
    else:
        q[6] = 1.0
    sl = _arm_joint_slice(robot_model)
    q[sl] = np.asarray(j, dtype=float).reshape(sl.stop - sl.start)
    pin.forwardKinematics(robot_model, data, q)
    pin.updateFramePlacements(robot_model, data)
    oMb = data.oMf[robot_model.getFrameId("base_link")].inverse()
    oMe = data.oMf[int(ee_frame_id)]
    return np.asarray((oMb * oMe).translation, dtype=float).copy()


def solve_arm_ik_plane2r(
    robot_model: pin.Model,
    ee_frame_id: int,
    target_local: np.ndarray,
    j0: Optional[np.ndarray] = None,
    *,
    j_angle_max: Optional[float] = 2.0,
    data: Optional[pin.Data] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """求 j 使 ee_body(j) ≈ target_local（3,）。

    平面臂 y 不可达时最小二乘只压 x-z 残差。返回 (j*, residual_local)。
    """
    if data is None:
        data = robot_model.createData()
    target_local = np.asarray(target_local, dtype=float).reshape(3)
    sl = _arm_joint_slice(robot_model)
    n_arm = sl.stop - sl.start
    lo, hi = arm_joint_limits(robot_model, j_angle_max=j_angle_max)
    if j0 is None:
        j0 = np.zeros(n_arm, dtype=float)
    j0 = np.clip(np.asarray(j0, dtype=float).reshape(n_arm), lo, hi)

    def residual(j):
        return ee_position_body_from_joints(
            robot_model, data, ee_frame_id, j
        ) - target_local

    sol = least_squares(
        residual,
        j0,
        bounds=(lo, hi),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    j_star = np.clip(sol.x, lo, hi)
    return j_star, residual(j_star)
