#!/usr/bin/env python3
"""
Add first-order actuator dynamics between the "high-level commands" and the "low-level actuator outputs".

MPC control input u_cmd ∈ R^6 (same dimension as the direct thrust model, but different semantics):
  [ωx_cmd, ωy_cmd, ωz_cmd]  body-frame angular-rate commands [rad/s]
  [T_total_cmd]             total thrust command [N] (allocated to the four rotors)
  [θ1_cmd, θ2_cmd]          robotic arm joint angle commands [rad]

Actuator state u_act ∈ R^6 (consistent with the existing model; values applied to the body):
  [T1, T2, T3, T4, τ1, τ2]

First-order lag:
  du_act/dt = (u_target - u_act) ./ τ

where u_target is computed from u_cmd and the current (q, v):
  - Angular rate: use the PD torque target M_des = Kω ⊙ (ω_cmd - ω_body), then allocate along with T_total to obtain T1..T4
  - Arm: τ_target = Kp ⊙ (θ_cmd - q_j) - Kd ⊙ q̇_j

Full state x = [q; v; u_act]; dimension nx = (nq+nv) + 6 = 17 + 6 = 23.
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


def _allocation_matrix_inv(rotors: list, cm_cf: float) -> np.ndarray:
    """
    Construct A ∈ R^{4x4} such that A @ T = [Fz, Mx, My, Mz]^T (consistent with the torque definition in build_acados_model).
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


def build_acados_model_actuator_first_order(
    urdf_path=None,
    s500_yaml_path=None,
    k_omega: float | np.ndarray = 0.15,
    k_p_arm: float | np.ndarray = 8.0,
    k_d_arm: float | np.ndarray = 0.8,
    tau_act: np.ndarray | None = None,
):
    """
    Return AcadosModel with control u_cmd(6) and state [q; v; u_act] (23).

    Parameters
    ----------
    k_omega : float or (3,)
        body angular-rate tracking gain; M_des = k_omega * (ω_cmd - ω) (per-component)
    k_p_arm, k_d_arm : float or (2,)
        joint PD gains
    tau_act : (6,) first-order time constants τ_i, du/dt = (u_target - u)/τ_i
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

    # x = [q(9); v(8); u_act(6)]
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
    # Update kinematics before ABA to avoid NaNs due to uninitialized intermediate quantities inside CasADi/Pinocchio
    cpin.forwardKinematics(cmodel, cdata, q)
    a = cpin.aba(cmodel, cdata, q, v, tau)

    pos = q[:3]
    quat = q[3:7]
    # IRK internal points may move the quaternion off the unit sphere; apply approximate normalization to mitigate numerical issues in R and ABA
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
