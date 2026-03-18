#!/usr/bin/env python3
"""
S500 UAM dynamics model for acados OCP.
Uses Pinocchio (from URDF) + CasADi for symbolic dynamics.
Thrust-to-wrench mapping from s500 config.

Dynamics source: URDF via pinocchio.buildModelFromUrdf().
Alternative: urdf2casadi (mahaarbo/urdf2casadi) for FK; Pinocchio+CasADi
is used here for full rigid-body dynamics (ABA) with automatic differentiation.
"""

import numpy as np
import yaml
from pathlib import Path

try:
    from pinocchio import casadi as cpin
    import pinocchio as pin
    import casadi as ca
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    ca = None


def _quat_to_R(quat):
    """Quaternion [qx,qy,qz,qw] to rotation matrix (CasADi)."""
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    return ca.vertcat(
        ca.horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)),
        ca.horzcat(2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)),
        ca.horzcat(2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2))
    )


def _quat_prod(p, q):
    """Quaternion product p * q (CasADi)."""
    return ca.vertcat(
        p[3]*q[0] + p[0]*q[3] + p[1]*q[2] - p[2]*q[1],
        p[3]*q[1] - p[0]*q[2] + p[1]*q[3] + p[2]*q[0],
        p[3]*q[2] + p[0]*q[1] - p[1]*q[0] + p[2]*q[3],
        p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2]
    )


def load_s500_config():
    """Load S500 platform config."""
    path = Path(__file__).parent.parent / 'config' / 'yaml' / 'multicopter' / 's500.yaml'
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def thrust_to_tau_base(thrusts, rotors_cfg, cm_cf):
    """
    Map thrusts [T1,T2,T3,T4] to base wrench (force,moment) in body frame.
    Force: thrust along body -z (upward in world when level)
    Moment: from lever arms and reaction torques.
    """
    T1, T2, T3, T4 = thrusts[0], thrusts[1], thrusts[2], thrusts[3]
    rotors = rotors_cfg['$rotors']
    Fz = T1 + T2 + T3 + T4  # total thrust (body z)
    Mx, My, Mz = 0.0, 0.0, 0.0
    for i, r in enumerate(rotors):
        pos = np.array(r['translation'])
        spin = r['spin_direction'][0]
        T = thrusts[i]
        # r x F with F=(0,0,T) in body: (-pos[1]*T, pos[0]*T, 0)
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T  # reaction torque
    return np.array([0, 0, Fz, Mx, My, Mz])


def build_acados_model(urdf_path=None, s500_yaml_path=None):
    """
    Build AcadosModel for S500 UAM with dynamics from URDF.
    State: [x,y,z, qx,qy,qz,qw, j1,j2, vx,vy,vz, wx,wy,wz, j1_dot,j2_dot]
    Control: [thrust1, thrust2, thrust3, thrust4, tau1, tau2]
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError("Pinocchio and CasADi required. Install: conda install pinocchio casadi -c conda-forge")

    from acados_template import AcadosModel

    base = Path(__file__).parent.parent
    if urdf_path is None:
        urdf_path = str(base / 'models' / 'urdf' / 's500_uam_simple.urdf')
    if s500_yaml_path is None:
        s500_yaml_path = str(base / 'config' / 'yaml' / 'multicopter' / 's500.yaml')

    # Load robot model
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = model.createData()
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    cfg = load_s500_config()
    platform = cfg['platform']
    cm_cf = platform['cm'] / platform['cf']
    rotors = platform['$rotors']

    nq, nv = model.nq, model.nv
    n_thrust = 4
    n_arm = 2
    nu = n_thrust + n_arm

    # State: q (7+2), v (6+2)
    q = ca.SX.sym('q', nq)
    v = ca.SX.sym('v', nv)
    u = ca.SX.sym('u', nu)

    # Control: thrusts + arm torques
    thrusts = u[:n_thrust]
    arm_tau = u[n_thrust:]

    # Build thrust-to-wrench for base (6 DOF)
    Fz = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
    Mx = My = Mz = 0.0
    for i, r in enumerate(rotors):
        pos = r['translation']
        spin = r['spin_direction'][0]
        T = thrusts[i]
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T

    tau_base = ca.vertcat(0, 0, Fz, Mx, My, Mz)
    tau_arm = arm_tau
    tau = ca.vertcat(tau_base, tau_arm)

    # Forward dynamics: a = ABA(q, v, tau)
    a = cpin.aba(cmodel, cdata, q, v, tau)

    # q_dot: Pinocchio freeflyer v = [v_linear_body, v_angular_body, j1_dot, j2_dot]
    # pos_dot = R @ v_linear, quat_dot = 0.5 * quat_prod(q, [omega;0]), joint_dot = v_joint
    pos = q[:3]
    quat = q[3:7]   # qx,qy,qz,qw
    v_lin = v[:3]
    v_ang = v[3:6]
    v_joint = v[6:8]
    R = _quat_to_R(quat)
    pos_dot = ca.mtimes(R, v_lin)
    quat_dot = 0.5 * _quat_prod(quat, ca.vertcat(v_ang[0], v_ang[1], v_ang[2], 0))
    jq_dot = v_joint
    q_dot = ca.vertcat(pos_dot, quat_dot, jq_dot)

    x = ca.vertcat(q, v)
    x_dot = ca.vertcat(q_dot, a)

    f_expl = ca.Function('f_expl', [x, u], [x_dot])

    acados_model = AcadosModel()
    acados_model.name = 's500_uam'
    acados_model.x = x
    acados_model.u = u
    acados_model.xdot = ca.SX.sym('xdot', x.rows())
    acados_model.f_impl_expr = acados_model.xdot - x_dot
    acados_model.f_expl_expr = x_dot

    # EE position available for path constraints (grasp mode)
    # con_h_expr can be set when using path constraints; omitted for simple OCP
    return acados_model, model, nq, nv, nu
