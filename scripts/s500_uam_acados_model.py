#!/usr/bin/env python3
"""
S500 UAM dynamics models for acados OCP.
Uses Pinocchio (from URDF) + CasADi for symbolic dynamics.
Thrust-to-wrench mapping from s500 config.

Includes:
  - Direct model: state (q,v), control = rotor thrusts + arm torques.
  - First-order actuator layer: control = high-level rate/thrust/joint commands; state augmented with u_act.
  - Cascade actuator chain: filtered command z plus u_act (two first-order stages).

Dynamics source: URDF via pinocchio.buildModelFromUrdf().
"""

from __future__ import annotations

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
    pin = None  # type: ignore


def _quat_to_R(quat):
    """Quaternion [qx,qy,qz,qw] to rotation matrix (CasADi)."""
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        ca.horzcat(2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)),
        ca.horzcat(2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)),
    )


def _quat_prod(p, q):
    """Quaternion product p * q (CasADi)."""
    return ca.vertcat(
        p[3] * q[0] + p[0] * q[3] + p[1] * q[2] - p[2] * q[1],
        p[3] * q[1] - p[0] * q[2] + p[1] * q[3] + p[2] * q[0],
        p[3] * q[2] + p[0] * q[1] - p[1] * q[0] + p[2] * q[3],
        p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2],
    )


def load_s500_config():
    """Load S500 platform config."""
    path = Path(__file__).parent.parent / "config" / "yaml" / "multicopter" / "s500.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def thrust_to_tau_base(thrusts, rotors_cfg, cm_cf):
    """
    Map thrusts [T1,T2,T3,T4] to base wrench (force,moment) in body frame.
    Force: thrust along body -z (upward in world when level)
    Moment: from lever arms and reaction torques.
    """
    T1, T2, T3, T4 = thrusts[0], thrusts[1], thrusts[2], thrusts[3]
    rotors = rotors_cfg["$rotors"]
    Fz = T1 + T2 + T3 + T4  # total thrust (body z)
    Mx, My, Mz = 0.0, 0.0, 0.0
    for i, r in enumerate(rotors):
        pos = np.array(r["translation"])
        spin = r["spin_direction"][0]
        T = thrusts[i]
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T  # reaction torque
    return np.array([0, 0, Fz, Mx, My, Mz])


def _allocation_matrix_inv(rotors: list, cm_cf: float) -> np.ndarray:
    """
    Construct A ∈ R^{4x4} such that A @ T = [Fz, Mx, My, Mz]^T (consistent with build_acados_model).
    """
    rows = []
    for i, r in enumerate(rotors):
        pos = r["translation"]
        spin = r["spin_direction"][0]
        col = np.array(
            [
                1.0,
                float(-pos[1]),
                float(pos[0]),
                float(spin * cm_cf),
            ],
            dtype=float,
        )
        rows.append(col)
    A = np.column_stack(rows)
    return np.linalg.inv(A)


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
        urdf_path = str(base / "models" / "urdf" / "s500_uam_simple.urdf")
    if s500_yaml_path is None:
        s500_yaml_path = str(base / "config" / "yaml" / "multicopter" / "s500.yaml")

    # Load robot model
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = model.createData()
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    cfg = load_s500_config()
    platform = cfg["platform"]
    cm_cf = platform["cm"] / platform["cf"]
    rotors = platform["$rotors"]

    nq, nv = model.nq, model.nv
    n_thrust = 4
    n_arm = 2
    nu = n_thrust + n_arm

    # State: q (7+2), v (6+2)
    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv)
    u = ca.SX.sym("u", nu)

    # Control: thrusts + arm torques
    thrusts = u[:n_thrust]
    arm_tau = u[n_thrust:]

    # Build thrust-to-wrench for base (6 DOF)
    Fz = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
    Mx = My = Mz = 0.0
    for i, r in enumerate(rotors):
        pos = r["translation"]
        spin = r["spin_direction"][0]
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
    pos = q[:3]
    quat = q[3:7]  # qx,qy,qz,qw
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

    f_expl = ca.Function("f_expl", [x, u], [x_dot])

    acados_model = AcadosModel()
    acados_model.name = "s500_uam"
    acados_model.x = x
    acados_model.u = u
    acados_model.xdot = ca.SX.sym("xdot", x.rows())
    acados_model.f_impl_expr = acados_model.xdot - x_dot
    acados_model.f_expl_expr = x_dot

    # EE position available for path constraints (grasp mode)
    # con_h_expr can be set when using path constraints; omitted for simple OCP
    return acados_model, model, nq, nv, nu


def build_acados_model_actuator_first_order(
    urdf_path=None,
    s500_yaml_path=None,
    k_omega: float | np.ndarray = 0.15,
    k_p_arm: float | np.ndarray = 8.0,
    k_d_arm: float | np.ndarray = 0.8,
    tau_act: np.ndarray | None = None,
):
    """
    First-order actuator dynamics between high-level commands and low-level actuator outputs.

    MPC control u_cmd ∈ R^6:
      [ωx_cmd, ωy_cmd, ωz_cmd]  body-frame angular-rate commands [rad/s]
      [T_total_cmd]             total thrust command [N]
      [θ1_cmd, θ2_cmd]          arm joint angle commands [rad]

    Actuator state u_act ∈ R^6: [T1..T4, τ1, τ2]. Full state x = [q; v; u_act], nx = 23.

    Returns AcadosModel with control u_cmd(6) and state [q; v; u_act] (23).
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError("Pinocchio + CasADi + acados_template required")

    from acados_template import AcadosModel

    base = Path(__file__).parent.parent
    if urdf_path is None:
        urdf_path = str(base / "models" / "urdf" / "s500_uam_simple.urdf")
    if s500_yaml_path is None:
        s500_yaml_path = str(base / "config" / "yaml" / "multicopter" / "s500.yaml")

    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    cfg = load_s500_config()
    platform = cfg["platform"]
    cm_cf = platform["cm"] / platform["cf"]
    rotors = platform["$rotors"]

    A_inv_np = _allocation_matrix_inv(rotors, cm_cf)
    A_inv = ca.DM(A_inv_np)

    nq, nv = model.nq, model.nv
    n_thrust, n_arm = 4, 2
    nu_phys = n_thrust + n_arm
    nu_cmd = 6

    if tau_act is None:
        tau_act = np.array([0.04, 0.04, 0.04, 0.04, 0.05, 0.05], dtype=float)
    tau_act = np.asarray(tau_act, dtype=float).reshape(-1)
    if tau_act.size == 1:
        tau_act = np.full(6, float(tau_act[0]))
    assert tau_act.shape[0] == 6

    k_omega = np.asarray(k_omega, dtype=float).reshape(-1)
    if k_omega.size == 1:
        k_omega = np.full(3, float(k_omega[0]))
    k_p_arm = np.asarray(k_p_arm, dtype=float).reshape(-1)
    if k_p_arm.size == 1:
        k_p_arm = np.full(2, float(k_p_arm[0]))
    k_d_arm = np.asarray(k_d_arm, dtype=float).reshape(-1)
    if k_d_arm.size == 1:
        k_d_arm = np.full(2, float(k_d_arm[0]))

    k_omega_c = ca.DM(k_omega)
    k_p_c = ca.DM(k_p_arm)
    k_d_c = ca.DM(k_d_arm)
    tau_c = ca.DM(tau_act)

    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv)
    u_act = ca.SX.sym("u_act", nu_phys)
    u_cmd = ca.SX.sym("u_cmd", nu_cmd)

    omega_cmd = u_cmd[0:3]
    T_total_cmd = u_cmd[3]
    theta_cmd = u_cmd[4:6]

    omega = v[3:6]
    q_j = q[7:9]
    v_j = v[6:8]

    M_des = k_omega_c * (omega_cmd - omega)
    b_wrench = ca.vertcat(T_total_cmd, M_des[0], M_des[1], M_des[2])
    T_target = ca.mtimes(A_inv, b_wrench)

    min_thrust = platform["min_thrust"]
    max_thrust = platform["max_thrust"]
    T_clipped = ca.vertcat(*[ca.fmin(ca.fmax(T_target[i], min_thrust), max_thrust) for i in range(4)])

    tau_arm_target = k_p_c * (theta_cmd - q_j) - k_d_c * v_j
    tau_arm_target = ca.vertcat(
        ca.fmin(ca.fmax(tau_arm_target[0], -2.0), 2.0),
        ca.fmin(ca.fmax(tau_arm_target[1], -2.0), 2.0),
    )

    u_target = ca.vertcat(T_clipped, tau_arm_target)

    thrusts = u_act[0:4]
    arm_tau = u_act[4:6]

    Fz = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
    Mx = My = Mz = 0.0
    for i, r in enumerate(rotors):
        pos = r["translation"]
        spin = r["spin_direction"][0]
        T = thrusts[i]
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T

    tau_base = ca.vertcat(0, 0, Fz, Mx, My, Mz)
    tau = ca.vertcat(tau_base, arm_tau)
    cpin.forwardKinematics(cmodel, cdata, q)
    a = cpin.aba(cmodel, cdata, q, v, tau)

    quat = q[3:7]
    quat_n = ca.norm_2(quat)
    quat_u = quat / ca.fmax(quat_n, 1e-9)
    v_lin = v[:3]
    v_ang = v[3:6]
    v_joint = v[6:8]
    R = _quat_to_R(quat_u)
    pos_dot = ca.mtimes(R, v_lin)
    quat_dot = 0.5 * _quat_prod(quat_u, ca.vertcat(v_ang[0], v_ang[1], v_ang[2], 0))
    q_dot = ca.vertcat(pos_dot, quat_dot, v_joint)

    x_r = ca.vertcat(q, v)
    x_dot_r = ca.vertcat(q_dot, a)

    u_act_dot = (u_target - u_act) / tau_c
    x = ca.vertcat(x_r, u_act)
    x_dot = ca.vertcat(x_dot_r, u_act_dot)

    acados_model = AcadosModel()
    acados_model.name = "s500_uam_act1"
    acados_model.x = x
    acados_model.u = u_cmd
    acados_model.xdot = ca.SX.sym("xdot", x.rows())
    acados_model.f_impl_expr = acados_model.xdot - x_dot
    acados_model.f_expl_expr = x_dot

    meta = {
        "nu_cmd": nu_cmd,
        "nu_phys": nu_phys,
        "nx_robot": nq + nv,
        "nx": int(x.size1()),
        "k_omega": k_omega,
        "k_p_arm": k_p_arm,
        "k_d_arm": k_d_arm,
        "tau_act": tau_act,
    }
    return acados_model, model, nq, nv, nu_cmd, meta


def pack_initial_state_with_actuators(
    x_robot_17: np.ndarray,
    u_act0: np.ndarray | None = None,
    pin_model=None,
    min_thrust: float = 0.085,
    max_thrust: float = 10.34,
) -> np.ndarray:
    """Combine the 17-dim robot state with the 6-dim initial actuator outputs into a 23-dim state."""
    x_r = np.asarray(x_robot_17, dtype=float).flatten()
    if u_act0 is None:
        if pin_model is None:
            raise ValueError("u_act0 or pin_model required")
        m = float(sum(inertia.mass for inertia in pin_model.inertias))
        t_each = np.clip(m * 9.81 / 4.0, min_thrust, max_thrust)
        u_act0 = np.concatenate([np.full(4, t_each), np.zeros(2)])
    u_act0 = np.asarray(u_act0, dtype=float).flatten()
    assert u_act0.shape[0] == 6
    return np.concatenate([x_r, u_act0])


def nominal_command_hover(
    pin_model: pin.Model,
    x_robot_17: np.ndarray,
    min_thrust: float,
    max_thrust: float,
) -> np.ndarray:
    """High-level approximate hover command: ω_cmd=0, T_total=mg, θ_cmd=current joint angles."""
    x_r = np.asarray(x_robot_17, dtype=float).flatten()
    m = float(sum(inertia.mass for inertia in pin_model.inertias))
    T_tot = float(np.clip(m * 9.81, 4 * min_thrust, 4 * max_thrust))
    j1, j2 = x_r[7], x_r[8]
    return np.array([0.0, 0.0, 0.0, T_tot, j1, j2], dtype=float)


def build_acados_model_cascade_actuator(
    urdf_path=None,
    s500_yaml_path=None,
    k_omega: float | np.ndarray = 0.15,
    k_p_arm: float | np.ndarray = 8.0,
    k_d_arm: float | np.ndarray = 0.8,
    tau_cmd: np.ndarray | None = None,
    tau_act: np.ndarray | None = None,
):
    """
    Two-layer first-order actuator chain: filtered command z, then u_act.

    u_cmd in R^6 (same semantics as actuator_first_order). State x = [q; v; z(6); u_act(6)], nx = 29.
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError("Pinocchio + CasADi + acados_template required")

    from acados_template import AcadosModel

    base = Path(__file__).parent.parent
    if urdf_path is None:
        urdf_path = str(base / "models" / "urdf" / "s500_uam_simple.urdf")
    if s500_yaml_path is None:
        s500_yaml_path = str(base / "config" / "yaml" / "multicopter" / "s500.yaml")

    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    cfg = load_s500_config()
    platform = cfg["platform"]
    cm_cf = platform["cm"] / platform["cf"]
    rotors = platform["$rotors"]

    A_inv_np = _allocation_matrix_inv(rotors, cm_cf)
    A_inv = ca.DM(A_inv_np)

    nq, nv = model.nq, model.nv
    n_thrust, n_arm = 4, 2
    nu_phys = n_thrust + n_arm
    nu_cmd = 6

    if tau_cmd is None:
        tau_cmd = np.array([0.03, 0.03, 0.08, 0.06, 0.05, 0.05], dtype=float)
    tau_cmd = np.asarray(tau_cmd, dtype=float).reshape(-1)
    if tau_cmd.size == 1:
        tau_cmd = np.full(6, float(tau_cmd[0]))
    assert tau_cmd.shape[0] == 6

    if tau_act is None:
        tau_act = np.array([0.04, 0.04, 0.04, 0.04, 0.05, 0.05], dtype=float)
    tau_act = np.asarray(tau_act, dtype=float).reshape(-1)
    if tau_act.size == 1:
        tau_act = np.full(6, float(tau_act[0]))
    assert tau_act.shape[0] == 6

    k_omega = np.asarray(k_omega, dtype=float).reshape(-1)
    if k_omega.size == 1:
        k_omega = np.full(3, float(k_omega[0]))
    k_p_arm = np.asarray(k_p_arm, dtype=float).reshape(-1)
    if k_p_arm.size == 1:
        k_p_arm = np.full(2, float(k_p_arm[0]))
    k_d_arm = np.asarray(k_d_arm, dtype=float).reshape(-1)
    if k_d_arm.size == 1:
        k_d_arm = np.full(2, float(k_d_arm[0]))

    k_omega_c = ca.DM(k_omega)
    k_p_c = ca.DM(k_p_arm)
    k_d_c = ca.DM(k_d_arm)
    tau_cmd_c = ca.DM(tau_cmd)
    tau_act_c = ca.DM(tau_act)

    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv)
    z_f = ca.SX.sym("z_cmd", nu_cmd)
    u_act = ca.SX.sym("u_act", nu_phys)
    u_cmd = ca.SX.sym("u_cmd", nu_cmd)

    z_dot = (u_cmd - z_f) / tau_cmd_c

    omega_ref = z_f[0:3]
    T_total_ref = z_f[3]
    theta_ref = z_f[4:6]

    omega = v[3:6]
    q_j = q[7:9]
    v_j = v[6:8]

    M_des = k_omega_c * (omega_ref - omega)
    b_wrench = ca.vertcat(T_total_ref, M_des[0], M_des[1], M_des[2])
    T_target = ca.mtimes(A_inv, b_wrench)

    min_thrust = platform["min_thrust"]
    max_thrust = platform["max_thrust"]
    T_clipped = ca.vertcat(*[ca.fmin(ca.fmax(T_target[i], min_thrust), max_thrust) for i in range(4)])

    tau_arm_target = k_p_c * (theta_ref - q_j) - k_d_c * v_j
    tau_arm_target = ca.vertcat(
        ca.fmin(ca.fmax(tau_arm_target[0], -2.0), 2.0),
        ca.fmin(ca.fmax(tau_arm_target[1], -2.0), 2.0),
    )

    u_target = ca.vertcat(T_clipped, tau_arm_target)

    thrusts = u_act[0:4]
    arm_tau = u_act[4:6]

    Fz = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
    Mx = My = Mz = 0.0
    for i, r in enumerate(rotors):
        pos = r["translation"]
        spin = r["spin_direction"][0]
        T = thrusts[i]
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T

    tau_base = ca.vertcat(0, 0, Fz, Mx, My, Mz)
    tau = ca.vertcat(tau_base, arm_tau)
    cpin.forwardKinematics(cmodel, cdata, q)
    a = cpin.aba(cmodel, cdata, q, v, tau)

    quat = q[3:7]
    quat_n = ca.norm_2(quat)
    quat_u = quat / ca.fmax(quat_n, 1e-9)
    v_lin = v[:3]
    v_ang = v[3:6]
    v_joint = v[6:8]
    R = _quat_to_R(quat_u)
    pos_dot = ca.mtimes(R, v_lin)
    quat_dot = 0.5 * _quat_prod(quat_u, ca.vertcat(v_ang[0], v_ang[1], v_ang[2], 0))
    q_dot = ca.vertcat(pos_dot, quat_dot, v_joint)

    x_r = ca.vertcat(q, v)
    x_dot_r = ca.vertcat(q_dot, a)

    u_act_dot = (u_target - u_act) / tau_act_c
    x = ca.vertcat(x_r, z_f, u_act)
    x_dot = ca.vertcat(x_dot_r, z_dot, u_act_dot)

    acados_model = AcadosModel()
    acados_model.name = "s500_uam_cascade_act"
    acados_model.x = x
    acados_model.u = u_cmd
    acados_model.xdot = ca.SX.sym("xdot", x.size1())
    acados_model.f_impl_expr = acados_model.xdot - x_dot
    acados_model.f_expl_expr = x_dot

    meta = {
        "nu_cmd": nu_cmd,
        "nu_phys": nu_phys,
        "nx_robot": nq + nv,
        "nx": int(x.size1()),
        "nz_cmd": nu_cmd,
        "k_omega": k_omega,
        "k_p_arm": k_p_arm,
        "k_d_arm": k_d_arm,
        "tau_cmd": tau_cmd,
        "tau_act": tau_act,
    }
    return acados_model, model, nq, nv, nu_cmd, meta


def pack_initial_state_cascade(
    x_robot_17: np.ndarray,
    pin_model: pin.Model,
    z0: np.ndarray | None = None,
    u_act0: np.ndarray | None = None,
    min_thrust: float = 0.085,
    max_thrust: float = 10.34,
) -> np.ndarray:
    """17-dim robot state + initial filtered command z0 + initial actuator state u_act0 -> 29-dim full state."""
    xr = np.asarray(x_robot_17, dtype=float).flatten()
    assert xr.shape[0] == 17
    if z0 is None:
        z0 = nominal_command_hover(pin_model, xr, min_thrust, max_thrust)
    z0 = np.asarray(z0, dtype=float).flatten()
    assert z0.shape[0] == 6
    if u_act0 is None:
        tmp = pack_initial_state_with_actuators(xr, None, pin_model=pin_model, min_thrust=min_thrust, max_thrust=max_thrust)
        u_act0 = tmp[17:23]
    u_act0 = np.asarray(u_act0, dtype=float).flatten()
    assert u_act0.shape[0] == 6
    return np.concatenate([xr, z0, u_act0])
