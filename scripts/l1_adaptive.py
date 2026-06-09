#!/usr/bin/env python3
"""
L1 自适应增广控制器（disturbance estimator + 补偿器）—— l1_adaptive.py

设计目标
────────
为现有 baseline 控制器（MPC tracking / PX4 / geometric）提供一个可插拔的
L1 自适应增广层，最终控制量形如：

        u = u_baseline + u_ac

其中 u_ac 来自对“未建模力/扰动”的在线估计与补偿。典型用途：
  • 估计并补偿空气阻力（drag）造成的稳态跟踪误差；
  • 估计并补偿模型变化（如抓取物体后整机质量/受力变化）。

工作原理（平动通道上的标准 L1 三件套）
────────────────────────────────────────
把世界系平动速度 v 的动力学写成

        v̇ = a_model + σ                                   (1)

  - a_model: 名义模型预测的“比力加速度”，即在无扰动假设下，当前
             baseline 指令（推力 + 姿态）应当产生的世界系加速度
             a_model = (T_base / m) * b3 - g * e3
  - σ      : 集总扰动（drag、质量误差、外力等）在世界系下的等效加速度（3 维）

1) 状态预测器（companion model，A_s 为 Hurwitz）：

        v̂̇ = a_model + σ̂ - a_s * (v̂ - v)                  (2)

2) 分段常数自适应律（piecewise-constant，L1 标准形式，仅依赖 a_s 与 dt）：

   令 ṽ = v̂ - v，对 A_m = -a_s·I 的标准增益：

        k_pc = a_s · e^{-a_s·dt} / (1 - e^{-a_s·dt})

   这里采用 **增量（积分）形式** 的更新，使集总扰动估计在直流处无偏
   （绝对式 σ̂ = -k_pc·ṽ 在有限采样周期下存在 e^{-a_s·dt} 的稳态偏差，
   不利于精确估计质量变化等慢变扰动）：

        σ̂ ← σ̂ - k_pc · ṽ                                  (3)

   稳态时 ṽ→0 ⇒ σ̂→σ（真实集总扰动），兼具平滑与快速。

3) 低通滤波（L1 control law，u_ac = C(s)·(-σ̂)）：

        a_ac = LPF(-σ̂)                                    (4)

   xy / z 通道允许设置不同截止频率，便于单独调节水平与竖直补偿带宽。

补偿量 a_ac（世界系加速度，3 维）如何映射回控制通道由调用方负责，
本模块只输出 a_ac 与扰动估计 σ̂。对推力+体角速度接口的典型映射见
run_tracking_controller.py 中的 `_l1_augment_bodyrate_thrust`。

该模块为纯 numpy 实现，不依赖 ROS，便于离线单元测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

GRAVITY = 9.81


@dataclass
class L1Params:
    """L1 自适应增广参数。"""

    enabled: bool = False
    # 状态预测器收敛速率 a_s（A_s = -a_s·I，越大越快收敛，噪声敏感度也越高）。
    as_gain: float = 8.0
    # 补偿低通滤波截止频率（rad/s）。水平 / 竖直可分开设置。
    wc_xy: float = 6.0
    wc_z: float = 6.0
    # 体角速度修正增益：将 a_ac 的横向分量转换为指向期望比力方向的体角速度。
    tilt_gain: float = 3.0
    # 安全限幅：补偿加速度幅值上限（m/s^2），防止估计发散导致大幅指令。
    max_accel_xy: float = 6.0
    max_accel_z: float = 6.0
    # 扰动估计幅值上限（m/s^2），用于裁剪 σ̂，提升鲁棒性。
    max_sigma: float = 25.0
    # 补偿低通滤波的 xy/z 通道所在坐标系：
    #   "body"（默认）：wc_z 作用于机体 z（matched，纯推力可瞬时补）、wc_xy 作用于
    #     机体 xy（unmatched，需倾转），与四旋翼欠驱动的补偿能力对齐；
    #   "world"：xy/z 按世界系水平/竖直划分（旧行为）。
    # 需调用方在 step() 传入机体→世界旋转矩阵 R_bw；未提供则回退世界系。
    frame: str = "body"

    # ── 位置误差积分增广（消除 L1 扰动补偿残差导致的稳态位置误差）─────────
    # 纯速度预测器的 L1 只能让 plant≈名义；若 σ̂ 因带宽/滞后有残差，baseline
    # 仍会留下稳态位置误差 e_p≈d_res/kp。这里并联一个对跟踪位置误差的积分
    # （以及可选比例）通道：a_ac += -k_pos_i·∫e_p - k_pos_p·e_p。
    use_pos_feedback: bool = False
    k_pos_i_xy: float = 0.6      # 水平位置误差积分增益 (1/s^2)
    k_pos_i_z: float = 0.8       # 竖直位置误差积分增益 (1/s^2)
    k_pos_p_xy: float = 0.0      # 水平位置误差比例增益 (1/s^2)，默认 0（避免与 baseline 冲突）
    k_pos_p_z: float = 0.0       # 竖直位置误差比例增益 (1/s^2)
    max_pos_integral_xy: float = 1.5  # 积分量 ∫e_p 的 anti-windup 上限 (m·s)
    max_pos_integral_z: float = 1.5

    def sanitize(self) -> "L1Params":
        self.as_gain = float(max(1e-3, self.as_gain))
        self.wc_xy = float(max(0.0, self.wc_xy))
        self.wc_z = float(max(0.0, self.wc_z))
        self.tilt_gain = float(max(0.0, self.tilt_gain))
        self.max_accel_xy = float(max(0.0, self.max_accel_xy))
        self.max_accel_z = float(max(0.0, self.max_accel_z))
        self.max_sigma = float(max(0.0, self.max_sigma))
        self.k_pos_i_xy = float(max(0.0, self.k_pos_i_xy))
        self.k_pos_i_z = float(max(0.0, self.k_pos_i_z))
        self.k_pos_p_xy = float(max(0.0, self.k_pos_p_xy))
        self.k_pos_p_z = float(max(0.0, self.k_pos_p_z))
        self.max_pos_integral_xy = float(max(0.0, self.max_pos_integral_xy))
        self.max_pos_integral_z = float(max(0.0, self.max_pos_integral_z))
        self.frame = "body" if str(self.frame).strip().lower() == "body" else "world"
        return self


class L1AdaptiveAugmentation:
    """
    平动通道 L1 自适应扰动估计 + 补偿器。

    用法（每个控制周期一次）：
        a_ac = l1.step(dt, v_world, a_applied_world)
    其中：
        dt              : 控制周期 (s)
        v_world         : 当前世界系平动速度 (3,)
        a_applied_world : 上一周期 **baseline** 名义比力加速度 a_nom（不含 u_ac），
                          a_nom = (T_b/m)*b3 - g*e3，用于预测器(2)
    返回：
        a_ac            : 世界系补偿加速度 (3,)，调用方据此修正推力/姿态。

    属性：
        sigma_hat       : 当前集总扰动估计（世界系加速度, m/s^2）
        a_ac            : 当前补偿加速度（世界系, m/s^2）
    """

    def __init__(self, params: Optional[L1Params] = None):
        self.params: L1Params = (params or L1Params()).sanitize()
        self.v_hat: Optional[np.ndarray] = None
        self.sigma_hat = np.zeros(3, dtype=float)
        self.a_ac = np.zeros(3, dtype=float)
        # L1 扰动补偿分量（不含位置反馈，世界系），用于诊断。
        self.a_l1 = np.zeros(3, dtype=float)
        # body 系滤波时的 LPF 内部状态（机体系），输出再旋回世界系存入 a_l1。
        self._a_l1_body = np.zeros(3, dtype=float)
        # 跟踪位置误差积分（世界系, m·s）与位置反馈补偿分量。
        self.pos_integral = np.zeros(3, dtype=float)
        self.a_pos = np.zeros(3, dtype=float)
        self._initialized = False

    # ──────────────────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return bool(self.params.enabled)

    def set_enabled(self, flag: bool) -> None:
        self.params.enabled = bool(flag)
        if not flag:
            # 关闭时清空补偿与预测器，下次开启重新 seed，避免残留旧估计。
            self.reset()

    def update_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        self.params.sanitize()

    def reset(self, v_world: Optional[np.ndarray] = None) -> None:
        """重置内部状态。可选地用当前速度 seed 预测器。"""
        self.sigma_hat = np.zeros(3, dtype=float)
        self.a_ac = np.zeros(3, dtype=float)
        self.a_l1 = np.zeros(3, dtype=float)
        self._a_l1_body = np.zeros(3, dtype=float)
        self.pos_integral = np.zeros(3, dtype=float)
        self.a_pos = np.zeros(3, dtype=float)
        if v_world is not None:
            self.v_hat = np.asarray(v_world, dtype=float).copy()
            self._initialized = True
        else:
            self.v_hat = None
            self._initialized = False

    def reseed_predictor(self, v_world: np.ndarray) -> None:
        """软重置：仅把预测器速度 v̂ 重新种到当前速度，**保留**已收敛的扰动
        估计 σ̂、补偿 a_ac/a_l1 与位置积分 pos_integral。

        用于 regulation→tracking 等模式切换：扰动（如推力不匹配、负载）在切换
        前后不变，硬 reset 会把补偿瞬间清零，导致系统重新掉落并让 L1 从头收敛
        （表现为起步掉高 / 轨迹被带偏）。速度连续，故只需对齐 v̂ 消除一次预测
        误差尖峰，其余状态原样保留即可实现无缝过渡。
        """
        self.v_hat = np.asarray(v_world, dtype=float).reshape(3).copy()
        self._initialized = True

    # ──────────────────────────────────────────────────────────────────────
    def step(
        self,
        dt: float,
        v_world: np.ndarray,
        a_applied_world: np.ndarray,
        pos_err_world: Optional[np.ndarray] = None,
        R_bw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        推进一个控制周期，返回世界系补偿加速度 a_ac (3,)。

        参数：
            pos_err_world : 跟踪位置误差 e_p = p - p_ref（世界系, 3,）。
                            提供且 use_pos_feedback=True 时，启用位置误差积分通道，
                            消除 L1 扰动补偿残差导致的稳态位置误差。
            R_bw          : 机体→世界旋转矩阵 (3,3)。frame="body" 时把补偿 LPF 的
                            xy/z 通道放到机体系；未提供则回退世界系滤波。
        """
        p = self.params
        v = np.asarray(v_world, dtype=float).reshape(3)
        a_applied = np.asarray(a_applied_world, dtype=float).reshape(3)

        if not p.enabled or dt <= 0.0:
            return self.a_ac.copy()

        # 首拍（或重置后）：用当前测量 seed 预测器，本拍不产生补偿。
        if not self._initialized or self.v_hat is None:
            self.v_hat = v.copy()
            self.sigma_hat = np.zeros(3, dtype=float)
            self.a_ac = np.zeros(3, dtype=float)
            self.a_l1 = np.zeros(3, dtype=float)
            self._a_l1_body = np.zeros(3, dtype=float)
            self.a_pos = np.zeros(3, dtype=float)
            self.pos_integral = np.zeros(3, dtype=float)
            self._initialized = True
            return self.a_ac.copy()

        a_s = p.as_gain
        v_tilde = self.v_hat - v  # 预测误差 ṽ

        # ── (3) 分段常数自适应律（增量/积分形式，DC 无偏） ────────────────
        # k_pc = a_s * e^{-a_s·dt} / (1 - e^{-a_s·dt}) > 0
        e = math.exp(-a_s * dt)
        denom = 1.0 - e
        if denom < 1e-9:
            denom = 1e-9
        k_pc = a_s * e / denom
        self.sigma_hat = self.sigma_hat - k_pc * v_tilde
        if p.max_sigma > 0.0:
            self.sigma_hat = np.clip(self.sigma_hat, -p.max_sigma, p.max_sigma)

        # ── (2) 预测器积分（前向欧拉） ─────────────────────────────────────
        v_hat_dot = a_applied + self.sigma_hat - a_s * v_tilde
        self.v_hat = self.v_hat + dt * v_hat_dot

        # ── (4) 低通滤波得到 L1 扰动补偿（u_ac = C(s)·(-σ̂)） ──────────────
        alpha_xy = 1.0 - math.exp(-p.wc_xy * dt) if p.wc_xy > 0.0 else 0.0
        alpha_z = 1.0 - math.exp(-p.wc_z * dt) if p.wc_z > 0.0 else 0.0
        self._lpf_compensation(-self.sigma_hat, alpha_xy, alpha_z, R_bw)

        # ── (5) 位置误差积分通道 + 合成总补偿（与 oracle 共用同一管线）──────
        return self._compose_a_ac(dt, pos_err_world)

    # ──────────────────────────────────────────────────────────────────────
    def _lpf_compensation(
        self,
        target_world: np.ndarray,
        alpha_xy: float,
        alpha_z: float,
        R_bw: Optional[np.ndarray],
    ) -> None:
        """一阶低通生成 a_l1（世界系）。frame="body" 且给定 R_bw 时，在机体系做
        xy/z 分通道滤波（wc_z 对齐机体 z=matched 推力轴），再旋回世界系。"""
        target_world = np.asarray(target_world, dtype=float).reshape(3)
        use_body = (self.params.frame == "body") and (R_bw is not None)
        if use_body:
            R = np.asarray(R_bw, dtype=float).reshape(3, 3)
            target_b = R.T @ target_world
            self._a_l1_body[0] += alpha_xy * (target_b[0] - self._a_l1_body[0])
            self._a_l1_body[1] += alpha_xy * (target_b[1] - self._a_l1_body[1])
            self._a_l1_body[2] += alpha_z * (target_b[2] - self._a_l1_body[2])
            self.a_l1 = R @ self._a_l1_body
        else:
            self.a_l1[0] += alpha_xy * (target_world[0] - self.a_l1[0])
            self.a_l1[1] += alpha_xy * (target_world[1] - self.a_l1[1])
            self.a_l1[2] += alpha_z * (target_world[2] - self.a_l1[2])
            # 同步机体状态，便于 world↔body 切换时无跳变。
            if R_bw is not None:
                self._a_l1_body = np.asarray(R_bw, dtype=float).reshape(3, 3).T @ self.a_l1
            else:
                self._a_l1_body = self.a_l1.copy()

    # ──────────────────────────────────────────────────────────────────────
    def step_oracle(
        self,
        dt: float,
        sigma_true_world: np.ndarray,
        pos_err_world: Optional[np.ndarray] = None,
        R_bw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Oracle 补偿：假设可**精确测量**扰动（绕过预测器与 LPF）。

        把 σ̂ 直接置为扰动真值（世界系加速度），补偿 a_l1 = -σ̂ 当拍生效（无估计
        误差、无估计/滤波滞后）；其余补偿管线（dFz/tilt、CTBR、位置误差通道、限幅）
        与自适应 step() 完全一致。用于把「估计质量」与「补偿环节质量」解耦评估。

        参数：
            sigma_true_world : 扰动真值（世界系加速度, m/s^2）= F_dist_world / m。
        """
        p = self.params
        if not p.enabled or dt <= 0.0:
            return self.a_ac.copy()
        sigma = np.asarray(sigma_true_world, dtype=float).reshape(3)
        self.sigma_hat = sigma.copy()
        if p.max_sigma > 0.0:
            self.sigma_hat = np.clip(self.sigma_hat, -p.max_sigma, p.max_sigma)
        # 完美测量：补偿直接等于 -σ̂，不经预测器/低通（frame 无关，瞬时无滤波）。
        self.a_l1 = -self.sigma_hat.copy()
        if R_bw is not None:
            self._a_l1_body = np.asarray(R_bw, dtype=float).reshape(3, 3).T @ self.a_l1
        else:
            self._a_l1_body = self.a_l1.copy()
        self._initialized = True
        return self._compose_a_ac(dt, pos_err_world)

    # ──────────────────────────────────────────────────────────────────────
    def _compose_a_ac(
        self, dt: float, pos_err_world: Optional[np.ndarray]
    ) -> np.ndarray:
        """位置误差积分通道（并联）+ 合成总补偿 a_ac = a_l1 + a_pos 并限幅。

        self.a_l1 须由调用方先行写入（自适应：LPF(-σ̂)；oracle：-σ̂）。
        """
        p = self.params
        self.a_pos = np.zeros(3, dtype=float)
        if p.use_pos_feedback and pos_err_world is not None:
            e_p = np.asarray(pos_err_world, dtype=float).reshape(3)
            # conditional integration（anti-windup）：补偿未饱和时才积分。
            sat_xy = (p.max_accel_xy > 0.0) and (
                np.hypot(self.a_ac[0], self.a_ac[1]) >= p.max_accel_xy - 1e-6
            )
            sat_z = (p.max_accel_z > 0.0) and (abs(self.a_ac[2]) >= p.max_accel_z - 1e-6)
            if not sat_xy:
                self.pos_integral[0] += dt * e_p[0]
                self.pos_integral[1] += dt * e_p[1]
            if not sat_z:
                self.pos_integral[2] += dt * e_p[2]
            if p.max_pos_integral_xy > 0.0:
                self.pos_integral[0] = float(
                    np.clip(self.pos_integral[0], -p.max_pos_integral_xy, p.max_pos_integral_xy)
                )
                self.pos_integral[1] = float(
                    np.clip(self.pos_integral[1], -p.max_pos_integral_xy, p.max_pos_integral_xy)
                )
            if p.max_pos_integral_z > 0.0:
                self.pos_integral[2] = float(
                    np.clip(self.pos_integral[2], -p.max_pos_integral_z, p.max_pos_integral_z)
                )
            # a_pos = -k_i·∫e_p - k_p·e_p（e_p = p - p_ref，正误差需负向补偿）
            self.a_pos[0] = -p.k_pos_i_xy * self.pos_integral[0] - p.k_pos_p_xy * e_p[0]
            self.a_pos[1] = -p.k_pos_i_xy * self.pos_integral[1] - p.k_pos_p_xy * e_p[1]
            self.a_pos[2] = -p.k_pos_i_z * self.pos_integral[2] - p.k_pos_p_z * e_p[2]

        # ── 合成总补偿并做安全限幅 ─────────────────────────────────────────
        self.a_ac = self.a_l1 + self.a_pos
        if p.max_accel_xy > 0.0:
            self.a_ac[0] = float(np.clip(self.a_ac[0], -p.max_accel_xy, p.max_accel_xy))
            self.a_ac[1] = float(np.clip(self.a_ac[1], -p.max_accel_xy, p.max_accel_xy))
        if p.max_accel_z > 0.0:
            self.a_ac[2] = float(np.clip(self.a_ac[2], -p.max_accel_z, p.max_accel_z))

        return self.a_ac.copy()

    # ──────────────────────────────────────────────────────────────────────
    def disturbance_force_world(self, mass: float) -> np.ndarray:
        """集总扰动等效力（世界系, N）= m·σ̂。"""
        return float(mass) * self.sigma_hat.copy()

    def estimated_added_mass(self, mass: float) -> float:
        """
        由竖直方向扰动粗略反推“附加质量”估计（kg）。

        若实际质量比名义大 Δm（如抓取物体），则真实重力 -(m+Δm)g 大于名义
        建模的 -m·g，表现为竖直向下的扰动加速度 σ_z<0，故
            F_dist_z = m·σ_z ≈ -Δm·g  ⇒  Δm ≈ -F_dist_z / g。
        仅作竖直载荷变化的近似指示，不区分气动竖直力。
        """
        if GRAVITY <= 0.0:
            return 0.0
        return float(-mass * self.sigma_hat[2] / GRAVITY)
