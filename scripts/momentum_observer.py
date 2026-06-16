"""广义动量观测器（generalized momentum / wrench observer）。

用于 aerial manipulator 抓取负载后的外部广义力估计：把“抓到物体”等效为作用在
系统上的外部广义力 τ_ext（nv 维），其 base 前 6 维即传递到浮动基的机体系 wrench
（= J_ee,base^T · F_load），正是 disturbance-aware MPC 的 base wrench 参数所需。

理论（De Luca / Haddadin 动量观测器）
------------------------------------
带外力的运动方程：  M(q) v̇ + C(q,v) v + g(q) = τ + τ_ext
广义动量 p = M(q) v，利用 Ṁ = C + C^T：
    ṗ = τ + τ_ext + (C^T v - g)
构造一阶残差观测器：
    r = K_O ( p - p0 - ∫₀ᵗ (τ + C^T v - g + r) dτ )
  ⟹ ṙ = K_O (τ_ext - r)
即 r 是 τ_ext 经带宽 K_O 的一阶低通，常值外力下无静差、收敛时间 ≈ 3/K_O。
抓取是阶跃常值扰动 → 调高 K_O（或抓取瞬间 reset + 临时高增益）即可“很快”收敛。

坐标系
------
pinocchio free-flyer 的 v[:6] 为机体系空间速度 [线;角]，其对偶广义力 r[:6] 为机体
系 wrench [F_b; M_b]。本模块提供 base_wrench_world() 用 R_bw 旋到世界系，便于直接
喂给 MPC 的世界系 p_dist_w 参数。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None  # type: ignore


class MomentumWrenchObserver:
    """全阶广义动量观测器，估计外部广义力 τ_ext (nv,)。

    用法（每控制拍）：
        obs.step(q, v, tau_applied, dt)
        F_w, M_w = obs.base_wrench_world(R_bw)   # base 等效世界系 6D wrench
    其中 ``tau_applied`` 为**名义**施加广义力（不含 τ_ext）：base 机体系旋翼 wrench
    [0,0,Fz, Mx,My,Mz] 叠加臂关节力矩，与动力学模型 aba() 的输入一致。
    """

    def __init__(
        self,
        pin_model,
        nq: int,
        nv: int,
        k_force: float = 20.0,
        k_torque: float = 20.0,
        k_arm: float = 20.0,
    ) -> None:
        if pin is None:
            raise ImportError("pinocchio is required for MomentumWrenchObserver")
        self._model = pin_model
        self._data = pin_model.createData()
        self.nq = int(nq)
        self.nv = int(nv)
        self.set_gains(k_force, k_torque, k_arm)
        self.enabled = True
        self.reset()

    # ──────────────────────────────────────────────────────────────────────
    def set_gains(self, k_force: float, k_torque: float, k_arm: float) -> None:
        """设观测器带宽（rad/s），按通道展开为对角增益向量。"""
        K = np.zeros(self.nv, dtype=float)
        K[:3] = max(float(k_force), 0.0)
        K[3:6] = max(float(k_torque), 0.0)
        if self.nv > 6:
            K[6:] = max(float(k_arm), 0.0)
        self.K = K

    def set_enabled(self, flag: bool) -> None:
        self.enabled = bool(flag)
        if not flag:
            self.reset()

    def reset(self, q: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None) -> None:
        """重置观测器。给定 (q,v) 时用当前动量 seed p0（抓取事件触发软重置常用）。"""
        self.r = np.zeros(self.nv, dtype=float)
        self.integral = np.zeros(self.nv, dtype=float)
        self.p0: Optional[np.ndarray] = None
        self._init = False
        if q is not None and v is not None:
            self._seed_p0(np.asarray(q, dtype=float), np.asarray(v, dtype=float))

    def _seed_p0(self, q: np.ndarray, v: np.ndarray) -> None:
        M = self._mass_matrix(q)
        self.p0 = M @ v
        self._init = True

    # ──────────────────────────────────────────────────────────────────────
    def _mass_matrix(self, q: np.ndarray) -> np.ndarray:
        M = pin.crba(self._model, self._data, q)  # 仅上三角有效
        return np.triu(M) + np.triu(M, 1).T

    def step(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau_applied: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """推进一拍，返回外部广义力残差 r (nv,)。dt<=0 或未启用时原样返回上拍 r。"""
        if not self.enabled or dt <= 0.0:
            return self.r.copy()
        q = np.asarray(q, dtype=float).reshape(self.nq)
        v = np.asarray(v, dtype=float).reshape(self.nv)
        tau = np.asarray(tau_applied, dtype=float).reshape(self.nv)

        M = self._mass_matrix(q)
        p = M @ v

        # 首拍：仅 seed p0，不产出残差（避免初始动量被误判为外力）。
        if not self._init or self.p0 is None:
            self.p0 = p
            self.integral = np.zeros(self.nv, dtype=float)
            self.r = np.zeros(self.nv, dtype=float)
            self._init = True
            return self.r.copy()

        # β' = C^T v - g（修正非线性项）。
        C = pin.computeCoriolisMatrix(self._model, self._data, q, v)
        g = pin.computeGeneralizedGravity(self._model, self._data, q)
        beta = C.T @ v - g

        # 积分被积项 = τ + β' + r（上拍 r），再算本拍 r。
        self.integral = self.integral + (tau + beta + self.r) * dt
        self.r = self.K * (p - self.p0 - self.integral)
        return self.r.copy()

    # ──────────────────────────────────────────────────────────────────────
    def base_wrench_body(self) -> np.ndarray:
        """base 机体系外部 wrench [F_b(3); M_b(3)] = r[:6]。"""
        return self.r[:6].copy()

    def base_wrench_world(self, R_bw: np.ndarray) -> np.ndarray:
        """base 世界系外部 wrench [F_w(3); M_w(3)]，R_bw 为机体→世界旋转。"""
        R = np.asarray(R_bw, dtype=float).reshape(3, 3)
        w = self.r[:6]
        out = np.zeros(6, dtype=float)
        out[:3] = R @ w[:3]
        out[3:6] = R @ w[3:6]
        return out

    def arm_torque_ext(self) -> np.ndarray:
        """臂关节外部力矩残差 r[6:]（base 等效方案下不喂 MPC，仅供诊断）。"""
        return self.r[6:].copy() if self.nv > 6 else np.zeros(0, dtype=float)
