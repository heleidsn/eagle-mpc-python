#!/usr/bin/env python3
"""
Acados model for a two-layer first-order actuator chain (used in the trajectory optimization GUI).

Optimization variable u_cmd in R^6 (consistent with the actuator_layer semantics):
  [ωx_cmd, ωy_cmd, ωz_cmd]  body angular-rate commands [rad/s]
  [T_total_cmd]             total thrust command [N]
  [θ1_cmd, θ2_cmd]          joint angle commands [rad]

Intermediate state z in R^6: first-order lag tracking of u_cmd in each channel
  dz/dt = (u_cmd - z) ./ τ_cmd
  τ_cmd can be set per channel: typically ωx/ωy (roll/pitch rates) are faster and ωz (yaw) is slower;
  thrust and joint commands have their own independent time constants.

Lower-level actuator state u_act in R^6 (actual rotor thrust + joint torques), same as s500_uam_acados_actuator_layer:
  du_act/dt = (u_target - u_act) ./ τ_act
  u_target is computed from the filtered z and (q, v) via torque allocation / joint PD.

Full state x = [q; v; z; u_act]; dimension 9+8+6+6 = 29.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from pinocchio import casadi as cpin
    import pinocchio as pin
    import casadi as ca
    from acados_template import AcadosModel

    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    ca = None

from s500_uam_acados_model import _quat_prod, _quat_to_R, load_s500_config
from s500_uam_acados_actuator_layer import _allocation_matrix_inv


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
    Returns AcadosModel: u = u_cmd(6), x = [q; v; z(6); u_act(6)].

    tau_cmd : (6,) filtering time constants for high-level commands [τ_ωx, τ_ωy, τ_ωz, τ_T, τ_θ1, τ_θ2]
    tau_act : (6,) first-order lag time constants for the lower-level T1..4 and τ1, τ2 (consistent with actuator_layer defaults)
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError("Pinocchio + CasADi + acados_template required")

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
        # ωx, ωy are faster; ωz is slower; thrust and joint commands are medium
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

    pos = q[:3]
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
    from s500_uam_acados_actuator_layer import nominal_command_hover, pack_initial_state_with_actuators

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
