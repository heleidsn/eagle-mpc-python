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

    # 2) 外界变化扰动（世界系力，逐轴正弦）
    var_enable: bool = False
    var_t0: float = 0.0
    var_t1: float = 0.0
    var_amp_x: float = 0.0
    var_amp_y: float = 0.0
    var_amp_z: float = 0.0
    var_freq: float = 0.5  # Hz
    var_phase_deg: float = 0.0

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
            self.const_enable or self.var_enable or self.thrust_enable or self.drag_enable
        )

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
        F_world += np.array(
            [params.var_amp_x * s, params.var_amp_y * s, params.var_amp_z * s], dtype=float
        )

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

    ẋ = f_model(x,u) + [0_{nq}; M(q)^{-1} τ_ext]，τ_ext=[w_body(6); 0(臂)]。
    每个仿真步开始用 set_tau_ext_body() 设定本步外力（机体系 wrench）。
    """

    def __init__(
        self,
        f_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
        pin_model,
        sim_dt: float,
        nq: int,
        nv: int,
        nu: int,
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

    def set_tau_ext_body(self, w_body6: np.ndarray) -> None:
        """设定本步外力（机体系 6 维 wrench [F; M]，臂关节外力置 0）。"""
        tau = np.zeros(self.nv, dtype=float)
        tau[:6] = np.asarray(w_body6, dtype=float).reshape(6)
        self._tau_ext = tau
        self._has_ext = bool(np.any(np.abs(tau) > 1e-12))

    def on_pre_step(self, t: float, step_index: int) -> None:
        return None

    def _f_total(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xdot = np.asarray(self._f(x, u), dtype=float).flatten()
        if self._has_ext:
            q = np.asarray(x, dtype=float).flatten()[: self.nq]
            Minv = pin.computeMinverse(self._model, self._data, q)
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
