#!/usr/bin/env python3
"""
闭环仿真扰动模型 + 扰动 plant 包装 —— sim_disturbance.py

为 Sim Tracking 的闭环仿真（Acados full-state, direct 控制）提供一组可在
**指定时间段**开启的扰动，并统一注入到 plant 动力学，用于验证 ros_tracking
里的 L1 自适应（scripts/l1_adaptive.py）对这些扰动的在线估计与补偿能力。

支持的扰动（均可单独设开启时间窗 [t0, t1]，t1<=0 表示一直持续到结束）：

1) 外界定常扰动 const：
   - 力 F=[fx,fy,fz]（N，世界系），力矩 M=[mx,my,mz]（N·m，世界系）。
   - 注入时力矩转机体系：M_body = Rᵀ·M_world（与 Pinocchio free-flyer 广义力约定一致）。
2) 外界变化扰动 varying：
   - 世界系力的逐轴正弦 A_i·sin(2π f t + φ)（N）。
3) 总推力估计偏差 thrust bias：
   - 控制器“以为”自己指令的总推力 T_cmd，但 plant 实际产生
       T_real = scale · T_cmd + bias
     （scale≠1 模拟推力系数标定误差，bias 模拟常值偏置）。
     这相当于沿机体 z 轴的一个附加比力，L1 会把它并入集总扰动 σ̂。
4) 桨叶空气阻力 drag（模仿 Gazebo gazebo_motor_model 转子阻力）：
       F_drag = -Cd · (Σ|ω_i|) · V_perp ,  Σ|ω_i| = sqrt(n · T / kf)
       V_perp = (v_world - wind) - ((v_world - wind)·b3) b3
   b3 为机体 z 轴在世界系方向；可设常值风 wind。
5) 状态估计误差 est error：
   - 给控制器/估计器看到的状态加高斯噪声（位置/姿态/线速度/角速度），
     **不**改变真实 plant 状态（即“量测含噪”）。

注入方式（与 MPC 模型严格一致，避免 model mismatch）
────────────────────────────────────────────────
外力/力矩通过广义外力 τ_ext=[w_body(6); 0(臂)] 注入：ABA 对 τ 线性，
   a(τ+τ_ext) = a(τ) + M(q)^{-1} τ_ext
故 plant 的 ẋ = f_model(x,u_real) + [0_{nq}; M(q)^{-1} τ_ext]，其中
f_model 复用 MPC 的 CasADi 显式动力学，M^{-1} 用 pinocchio 在运行时算。
推力偏置直接改 u（四路推力），自然经 f_model 生效。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None  # type: ignore

from s500_uam_closed_loop_plant import rk4_step

GRAVITY = 9.81


def _quat_to_R_np(quat_xyzw: np.ndarray) -> np.ndarray:
    """四元数 [qx,qy,qz,qw] -> 旋转矩阵 R（世界<-机体）。"""
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


@dataclass
class DisturbanceParams:
    """闭环仿真扰动配置（plant 侧，仅影响仿真，不改 MPC 名义模型）。"""

    # 1) 外界定常扰动
    const_enable: bool = False
    const_t0: float = 0.0
    const_t1: float = 0.0  # <=0 表示到仿真结束
    const_fx: float = 0.0
    const_fy: float = 0.0
    const_fz: float = 0.0
    const_mx: float = 0.0
    const_my: float = 0.0
    const_mz: float = 0.0
    # 定常力/力矩的定义坐标系：False=世界系（默认），True=机体系。
    # 机体系沿 b3 的力天然 matched（纯推力可补）；世界系竖直力在机体倾斜时
    # 会出现垂直 b3 的 unmatched 分量，只能靠倾转补偿——故此开关用于对照实验。
    const_body_frame: bool = False

    # 2) 外界变化扰动（逐轴正弦，力 + 力矩，坐标系可切换）
    var_enable: bool = False
    var_t0: float = 0.0
    var_t1: float = 0.0
    var_amp_x: float = 0.0
    var_amp_y: float = 0.0
    var_amp_z: float = 0.0
    var_amp_mx: float = 0.0
    var_amp_my: float = 0.0
    var_amp_mz: float = 0.0
    var_freq: float = 0.5  # Hz
    var_phase_deg: float = 0.0
    var_body_frame: bool = False

    # 2b) EE-link 外界定常扰动（力/力矩，坐标系=EE系或世界系，仅 s500_uam）
    ee_const_enable: bool = False
    ee_const_t0: float = 0.0
    ee_const_t1: float = 0.0
    ee_const_fx: float = 0.0
    ee_const_fy: float = 0.0
    ee_const_fz: float = 0.0
    ee_const_mx: float = 0.0
    ee_const_my: float = 0.0
    ee_const_mz: float = 0.0
    ee_const_body_frame: bool = False  # True=EE 局部系（随 EE 旋转），False=世界系

    # 2c) EE-link 外界变化扰动（逐轴正弦力/力矩）
    ee_var_enable: bool = False
    ee_var_t0: float = 0.0
    ee_var_t1: float = 0.0
    ee_var_amp_x: float = 0.0
    ee_var_amp_y: float = 0.0
    ee_var_amp_z: float = 0.0
    ee_var_amp_mx: float = 0.0
    ee_var_amp_my: float = 0.0
    ee_var_amp_mz: float = 0.0
    ee_var_freq: float = 0.5
    ee_var_phase_deg: float = 0.0
    ee_var_body_frame: bool = False

    # 2d) 模拟负载（刚性附着 EE 的物体，默认 400g 可乐罐圆柱；仅 s500_uam）
    #   plant 用"含负载惯量的增广 Pinocchio 模型"积分（重力/科氏/惯性耦合全精确），
    #   MPC 仍用名义模型 → 模型失配=真实扰动，适合抓取后用动量观测器估计再补偿。
    load_enable: bool = False
    load_t0: float = 0.0       # 抓取时刻（之前 plant 用名义模型，之后用增广模型）
    load_t1: float = 0.0       # <=0 持续到结束
    load_mass: float = 0.4     # kg
    load_ixx: float = 5.5e-4   # kg·m²（可乐罐横向，r≈33mm/h≈115mm）
    load_iyy: float = 5.5e-4
    load_izz: float = 2.18e-4  # 轴向
    load_com_x: float = 0.0    # 负载质心相对 EE 系偏置 [m]
    load_com_y: float = 0.0
    load_com_z: float = 0.0

    # 3) 总推力估计偏差： T_real = scale·T_cmd + bias
    thrust_enable: bool = False
    thrust_t0: float = 0.0
    thrust_t1: float = 0.0
    thrust_scale: float = 1.0
    thrust_bias: float = 0.0  # N（总推力常值偏置，均分到各转子）

    # 4) 桨叶空气阻力（Gazebo 风格）
    drag_enable: bool = False
    drag_t0: float = 0.0
    drag_t1: float = 0.0
    drag_cd: float = 1.0e-4        # rotor_drag_coefficient (Gazebo rotorS 量级)
    drag_kf: float = 8.54858e-06   # motor_constant kf
    drag_wind_x: float = 0.0
    drag_wind_y: float = 0.0
    drag_wind_z: float = 0.0

    # 5) 状态估计误差（量测噪声，仅影响控制器/L1 看到的状态）
    est_enable: bool = False
    est_t0: float = 0.0
    est_t1: float = 0.0
    est_pos_std: float = 0.0       # m
    est_att_std: float = 0.0       # rad（小角度扰动）
    est_vel_std: float = 0.0       # m/s（机体系线速度）
    est_omega_std: float = 0.0     # rad/s（机体系角速度）
    est_seed: int = 0

    def any_plant_disturbance(self) -> bool:
        """是否存在改变 plant 真实动力学的扰动（不含状态估计误差）。"""
        return bool(
            self.const_enable or self.var_enable or self.thrust_enable
            or self.drag_enable or self.ee_const_enable or self.ee_var_enable
            or self.load_enable
        )

    def any_ee_disturbance(self) -> bool:
        """是否存在 EE-link 力旋量扰动（需 J^T 注入；不含负载增广模型）。"""
        return bool(self.ee_const_enable or self.ee_var_enable)

    def any_enabled(self) -> bool:
        return bool(self.any_plant_disturbance() or self.est_enable)

    @staticmethod
    def from_dict(d: Optional[dict]) -> "DisturbanceParams":
        p = DisturbanceParams()
        if not d:
            return p
        for k, v in d.items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)
        return p


def _active(t: float, enable: bool, t0: float, t1: float) -> bool:
    if not enable:
        return False
    if t < float(t0) - 1e-12:
        return False
    if float(t1) > 0.0 and t > float(t1) + 1e-12:
        return False
    return True


def external_force_torque(
    params: DisturbanceParams,
    t: float,
    R: np.ndarray,
    v_world: np.ndarray,
    thrust_total: float,
    n_rotors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (F_world, M_body, M_world)：const+varying+drag 合成的外力（世界系，N）、
    注入 plant 的外力矩（机体系，N·m）与等效世界系外力矩（N·m，用于日志/绘图）。

    thrust_total 用于桨叶阻力的 Σ|ω| 估计；推力偏置等效力不在此处计入。
  定常力矩在世界系定义，每步按当前姿态转为 M_body = Rᵀ·M_world。
    """
    F_world = np.zeros(3, dtype=float)
    M_body = np.zeros(3, dtype=float)
    M_world = np.zeros(3, dtype=float)
    R = np.asarray(R, dtype=float)
    b3 = R[:, 2]

    if _active(t, params.const_enable, params.const_t0, params.const_t1):
        F_const = np.array([params.const_fx, params.const_fy, params.const_fz], dtype=float)
        M_const = np.array([params.const_mx, params.const_my, params.const_mz], dtype=float)
        if params.const_body_frame:
            # 机体系定义：随姿态旋转到世界系（力矩同理转世界系记录）。
            F_world += R @ F_const
            M_w = R @ M_const
        else:
            F_world += F_const
            M_w = M_const
        M_world += M_w
        M_body += R.T @ M_w

    if _active(t, params.var_enable, params.var_t0, params.var_t1):
        ph = math.radians(params.var_phase_deg)
        s = math.sin(2.0 * math.pi * params.var_freq * t + ph)
        F_var = np.array(
            [params.var_amp_x * s, params.var_amp_y * s, params.var_amp_z * s], dtype=float
        )
        M_var = np.array(
            [params.var_amp_mx * s, params.var_amp_my * s, params.var_amp_mz * s], dtype=float
        )
        if params.var_body_frame:
            F_world += R @ F_var
            M_w_var = R @ M_var
        else:
            F_world += F_var
            M_w_var = M_var
        M_world += M_w_var
        M_body += R.T @ M_w_var

    if _active(t, params.drag_enable, params.drag_t0, params.drag_t1):
        wind = np.array(
            [params.drag_wind_x, params.drag_wind_y, params.drag_wind_z], dtype=float
        )
        v_rel = np.asarray(v_world, dtype=float).reshape(3) - wind
        v_perp = v_rel - float(np.dot(v_rel, b3)) * b3
        kf = max(float(params.drag_kf), 1e-12)
        n_rot = max(int(n_rotors), 1)
        sum_omega = math.sqrt(max(float(thrust_total), 0.0) * n_rot / kf)
        F_world += -float(params.drag_cd) * sum_omega * v_perp

    return F_world, M_body, M_world


def ee_wrench_world(
    params: DisturbanceParams, t: float, R_ee: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """EE-link 外界扰动（定常 + 变化）合成的力旋量，返回世界系 (F_world, M_world)。

    定义坐标系：body=EE 局部系（随 EE 姿态旋转，用 R_ee 转世界）；world=世界系直给。
    施力点取 EE 帧原点（注入侧用 LOCAL_WORLD_ALIGNED 雅可比折算到广义力）。
    """
    F_w = np.zeros(3, dtype=float)
    M_w = np.zeros(3, dtype=float)
    R_ee = np.asarray(R_ee, dtype=float)

    if _active(t, params.ee_const_enable, params.ee_const_t0, params.ee_const_t1):
        F = np.array([params.ee_const_fx, params.ee_const_fy, params.ee_const_fz], dtype=float)
        M = np.array([params.ee_const_mx, params.ee_const_my, params.ee_const_mz], dtype=float)
        if params.ee_const_body_frame:
            F_w += R_ee @ F
            M_w += R_ee @ M
        else:
            F_w += F
            M_w += M

    if _active(t, params.ee_var_enable, params.ee_var_t0, params.ee_var_t1):
        ph = math.radians(params.ee_var_phase_deg)
        s = math.sin(2.0 * math.pi * params.ee_var_freq * t + ph)
        F = np.array(
            [params.ee_var_amp_x * s, params.ee_var_amp_y * s, params.ee_var_amp_z * s],
            dtype=float,
        )
        M = np.array(
            [params.ee_var_amp_mx * s, params.ee_var_amp_my * s, params.ee_var_amp_mz * s],
            dtype=float,
        )
        if params.ee_var_body_frame:
            F_w += R_ee @ F
            M_w += R_ee @ M
        else:
            F_w += F
            M_w += M

    return F_w, M_w


def disturbance_base_equiv_world(
    params: DisturbanceParams,
    t: float,
    q: np.ndarray,
    R: np.ndarray,
    v_world: np.ndarray,
    thrust_total: float,
    n_rotors: int,
    pin_model,
    data,
    ee_frame_id: int,
    nv: int,
    n_arm: int,
    gravity: float = GRAVITY,
    return_arm: bool = False,
):
    """base 等效世界系扰动 (F_world, M_world)：base wrench + EE(J^T 折算到 base) +
    负载重力等效。**不含**推力估计偏差（调用方按需叠加 dT·b3）。

    与动量观测器估计的 base 6D wrench 同口径，便于真值/估计对照与 oracle 补偿。
    EE/负载需 n_arm>0；负载项为准静态重力等效（忽略运动惯性，plant 侧由增广模型精确）。

    ``return_arm=True`` 时额外返回臂关节空间扰动力矩 ``tau_arm`` (n_arm,)：来自 EE 力旋量
    与负载重力经 J^T 折算的关节分量（= 整段 J^T·w 的第 6: 维）。base 6D wrench 只表示
    浮动基所受合力/合力矩，**无法**表达"力作用在 EE 上同时压弯机械臂关节"这一效应——
    要在 whole-body 增广里补偿 EE 侧向力，必须把这一关节力矩也喂给 MPC（见调用方）。
    """
    q = np.asarray(q, dtype=float)[: pin_model.nq]
    R = np.asarray(R, dtype=float)
    F_ext, M_body_ext, _M_world_ext = external_force_torque(
        params, t, R, v_world, thrust_total, n_rotors
    )
    tau = np.zeros(int(nv), dtype=float)
    tau[:3] = R.T @ F_ext
    tau[3:6] = M_body_ext
    J = None
    if n_arm > 0 and params.any_ee_disturbance():
        pin.framesForwardKinematics(pin_model, data, q)
        R_ee = np.asarray(data.oMf[int(ee_frame_id)].rotation, dtype=float)
        F_ee_w, M_ee_w = ee_wrench_world(params, t, R_ee)
        if np.any(np.abs(F_ee_w) > 1e-12) or np.any(np.abs(M_ee_w) > 1e-12):
            J = pin.computeFrameJacobian(
                pin_model, data, q, int(ee_frame_id), pin.LOCAL_WORLD_ALIGNED
            )
            tau += J.T @ np.concatenate([F_ee_w, M_ee_w])
    F_w = R @ tau[:3]
    M_w = R @ tau[3:6]
    # 臂关节扰动力矩（关节空间）：EE 力旋量经 J^T 的第 6: 维；负载部分在下方累加。
    tau_arm = (
        np.asarray(tau[6 : 6 + int(n_arm)], dtype=float).copy()
        if (return_arm and n_arm > 0)
        else np.zeros(int(n_arm) if n_arm > 0 else 0, dtype=float)
    )
    if (
        n_arm > 0 and params.load_enable
        and _active(t, True, params.load_t0, params.load_t1)
    ):
        pin.framesForwardKinematics(pin_model, data, q)
        oMf = data.oMf[int(ee_frame_id)]
        com_off = np.array(
            [params.load_com_x, params.load_com_y, params.load_com_z], dtype=float
        )
        com_off_w = np.asarray(oMf.rotation, dtype=float) @ com_off
        p_com_w = np.asarray(oMf.translation, dtype=float) + com_off_w
        F_load_w = float(params.load_mass) * np.array([0.0, 0.0, -gravity], dtype=float)
        F_w = F_w + F_load_w
        M_w = M_w + np.cross(p_com_w - q[:3], F_load_w)
        if return_arm:
            if J is None:
                pin.framesForwardKinematics(pin_model, data, q)
                J = pin.computeFrameJacobian(
                    pin_model, data, q, int(ee_frame_id), pin.LOCAL_WORLD_ALIGNED
                )
            # 负载力作用在 COM：等价于 EE 帧原点处力 F + 力矩 (com_off×F)（LWA 力旋量）。
            w_load_ee = np.concatenate([F_load_w, np.cross(com_off_w, F_load_w)])
            tau_arm = tau_arm + (J.T @ w_load_ee)[6 : 6 + int(n_arm)]
    if return_arm:
        return F_w, M_w, tau_arm
    return F_w, M_w


def make_load_inertia(params: DisturbanceParams):
    """构造负载的 Pinocchio 惯量（质量 + 质心偏置 + 对角主惯量，COM 系下）。"""
    if pin is None:
        raise ImportError("pinocchio is required for load model")
    com = np.array([params.load_com_x, params.load_com_y, params.load_com_z], dtype=float)
    I3 = np.diag([
        max(float(params.load_ixx), 0.0),
        max(float(params.load_iyy), 0.0),
        max(float(params.load_izz), 0.0),
    ]).astype(float)
    return pin.Inertia(float(params.load_mass), com, I3)


def augment_pin_model_with_load(
    pin_model, ee_frame_id: int, params: DisturbanceParams
):
    """返回把负载惯量刚性附着到 EE 帧后的 Pinocchio 模型副本（plant 用）。

    负载惯量加到 EE 帧所属关节上、放在 EE 帧位姿处 → 重力/科氏/惯性耦合在积分中全精确。
    """
    if pin is None:
        raise ImportError("pinocchio is required for load model")
    model = pin_model.copy()
    frame = model.frames[int(ee_frame_id)]
    jid = int(getattr(frame, "parentJoint", getattr(frame, "parent", 0)))
    iMf = frame.placement  # EE 帧相对父关节的位姿
    model.appendBodyToJoint(jid, make_load_inertia(params), iMf)
    return model


def apply_thrust_bias(
    params: DisturbanceParams, t: float, u: np.ndarray, n_rotors: int
) -> Tuple[np.ndarray, float]:
    """对控制量的四路推力施加估计偏差，返回 (u_real, dT_total)。

    dT_total = T_real_total - T_cmd_total（用于诊断/真值力日志）。
    """
    u_real = np.asarray(u, dtype=float).flatten().copy()
    if not _active(t, params.thrust_enable, params.thrust_t0, params.thrust_t1):
        return u_real, 0.0
    n_rot = max(int(n_rotors), 1)
    t_cmd = float(np.sum(u_real[:n_rot]))
    u_real[:n_rot] = float(params.thrust_scale) * u_real[:n_rot] + float(params.thrust_bias) / n_rot
    t_real = float(np.sum(u_real[:n_rot]))
    return u_real, (t_real - t_cmd)


def corrupt_state_estimate(
    params: DisturbanceParams,
    t: float,
    x: np.ndarray,
    nq: int,
    nv: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """对 plant 真实状态加量测噪声，得到控制器/L1 看到的状态估计 x_est。"""
    x_est = np.asarray(x, dtype=float).flatten().copy()
    if not _active(t, params.est_enable, params.est_t0, params.est_t1):
        return x_est
    if params.est_pos_std > 0.0:
        x_est[0:3] += rng.normal(0.0, params.est_pos_std, size=3)
    if params.est_att_std > 0.0:
        dtheta = rng.normal(0.0, params.est_att_std, size=3)
        ang = float(np.linalg.norm(dtheta))
        if ang > 1e-12:
            axis = dtheta / ang
            dq = np.array(
                [
                    axis[0] * math.sin(ang / 2),
                    axis[1] * math.sin(ang / 2),
                    axis[2] * math.sin(ang / 2),
                    math.cos(ang / 2),
                ],
                dtype=float,
            )
            q = x_est[3:7]
            qn = _quat_mul_xyzw(q, dq)
            x_est[3:7] = qn / max(float(np.linalg.norm(qn)), 1e-12)
    if params.est_vel_std > 0.0:
        x_est[nq : nq + 3] += rng.normal(0.0, params.est_vel_std, size=3)
    if params.est_omega_std > 0.0:
        x_est[nq + 3 : nq + 6] += rng.normal(0.0, params.est_omega_std, size=3)
    return x_est


class DisturbedRK4Plant:
    """RK4 plant：在 MPC 的 CasADi 显式动力学上叠加广义外力 τ_ext（ZOH/步内常值）。

    ẋ = f_model(x,u) + [0_{nq}; M(q)^{-1} τ_ext]，τ_ext 为完整 nv 维广义外力
    （base 6 维 wrench + 经 J^T 折算的 EE 力旋量 + 臂关节外力）。

    模拟负载：可设第二套动力学 f_load（含负载惯量的增广模型）与抓取时间窗
    [load_t0, load_t1]；窗内用 f_load 积分（M^{-1} 也用增广模型），否则用名义 f。
    每步由 on_pre_step(t,·) 缓存当前时刻以选择动力学。
    """

    def __init__(
        self,
        f_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
        pin_model,
        sim_dt: float,
        nq: int,
        nv: int,
        nu: int,
        f_load: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        load_model=None,
        load_t0: float = 0.0,
        load_t1: float = 0.0,
    ):
        if pin is None:
            raise ImportError("pinocchio is required for DisturbedRK4Plant")
        self._f = f_fun
        self._model = pin_model
        self._data = pin_model.createData()
        self._dt = float(sim_dt)
        self.nq = int(nq)
        self.nv = int(nv)
        self.nu = int(nu)
        self._tau_ext = np.zeros(self.nv, dtype=float)
        self._has_ext = False
        # 负载增广动力学（可选）。
        self._f_load = f_load
        self._load_model = load_model
        self._load_data = load_model.createData() if load_model is not None else None
        self._load_t0 = float(load_t0)
        self._load_t1 = float(load_t1)
        self._t = 0.0

    def set_tau_ext_body(self, w_body6: np.ndarray) -> None:
        """设定本步 base 外力（机体系 6 维 wrench [F; M]，其余分量置 0）。"""
        tau = np.zeros(self.nv, dtype=float)
        tau[:6] = np.asarray(w_body6, dtype=float).reshape(6)
        self.set_tau_ext_full(tau)

    def set_tau_ext_full(self, tau_nv: np.ndarray) -> None:
        """设定本步完整广义外力 τ_ext（nv 维，含 base wrench + EE 折算 + 臂关节）。"""
        tau = np.asarray(tau_nv, dtype=float).flatten()
        out = np.zeros(self.nv, dtype=float)
        out[: min(self.nv, tau.size)] = tau[: min(self.nv, tau.size)]
        self._tau_ext = out
        self._has_ext = bool(np.any(np.abs(out) > 1e-12))

    def _load_active(self) -> bool:
        if self._f_load is None:
            return False
        if self._t < self._load_t0 - 1e-12:
            return False
        if self._load_t1 > 0.0 and self._t > self._load_t1 + 1e-12:
            return False
        return True

    def on_pre_step(self, t: float, step_index: int) -> None:
        self._t = float(t)

    def _f_total(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        use_load = self._load_active()
        f = self._f_load if use_load else self._f
        model = self._load_model if use_load else self._model
        data = self._load_data if use_load else self._data
        xdot = np.asarray(f(x, u), dtype=float).flatten()
        if self._has_ext:
            q = np.asarray(x, dtype=float).flatten()[: self.nq]
            Minv = pin.computeMinverse(model, data, q)
            xdot[self.nq : self.nq + self.nv] += Minv @ self._tau_ext
        return xdot

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return rk4_step(self._f_total, x, u, self._dt)


def build_rotor_allocation(rotors_cfg, cm_cf: float) -> np.ndarray:
    """构建 4x4 推力分配矩阵 A：[Fz, Mx, My, Mz]^T = A · [T1..T4]^T（机体系）。"""
    A = np.zeros((4, 4), dtype=float)
    for i, r in enumerate(rotors_cfg):
        pos = r["translation"]
        spin = r["spin_direction"][0]
        A[0, i] = 1.0
        A[1, i] = -pos[1]
        A[2, i] = pos[0]
        A[3, i] = spin * cm_cf
    return A
