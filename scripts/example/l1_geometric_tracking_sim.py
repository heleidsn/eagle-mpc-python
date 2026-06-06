#!/usr/bin/env python3
"""
Geometric tracking + L1 自适应增广 — 带固定扰动的闭环仿真对比。

Plant（平动 + 简化姿态）：
    v̇ = (T/m_true)·b3 - g·e3 + σ_true(t, v)
    σ_true = 线性阻力 + 常值风扰 +（可选）质量阶跃引起的等效加速度

Baseline：与 run_tracking_controller 相同的几何控制器（SE3）。
L1：scripts/l1_adaptive.py，预测器输入为 **baseline** 名义加速度 a_nom（不含 u_ac）。

对比四种工况：
    1) baseline only
    2) baseline + L1
    3) baseline + 扰动
    4) baseline + L1 + 扰动

运行：
    python l1_geometric_tracking_sim.py
    python l1_geometric_tracking_sim.py --no-show --save results/l1_compare.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from l1_adaptive import L1AdaptiveAugmentation, L1Params

GRAVITY = 9.81


# ─────────────────────────────────────────────────────────────────────────────
# 参考轨迹 & 几何控制器
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Ref3D:
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    yaw: float


def ref_circle(t: float, omega: float = 0.35, radius: float = 2.0, z0: float = 1.5) -> Ref3D:
    w = omega
    c, s = math.cos(w * t), math.sin(w * t)
    p = np.array([radius * c, radius * s, z0], dtype=float)
    v = np.array([-radius * w * s, radius * w * c, 0.0], dtype=float)
    a = np.array([-radius * w * w * c, -radius * w * w * s, 0.0], dtype=float)
    yaw = math.atan2(v[1], v[0] + 1e-9)
    return Ref3D(p=p, v=v, a=a, yaw=yaw)


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [x,y,z,w]."""
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    return q / max(np.linalg.norm(q), 1e-12)


def R_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def vee_so3(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def geometric_baseline(
    p: np.ndarray,
    v: np.ndarray,
    R: np.ndarray,
    omega: np.ndarray,
    ref: Ref3D,
    mass: float,
    *,
    kp_pos: float = 4.0,
    kd_vel: float = 2.5,
    kR: float = 4.0,
    kOmega: float = 0.35,
    max_tilt_deg: float = 35.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (thrust_N, body_rate_cmd, a_des) — 与 ROS geometric 一致。
    """
    e_p = p - ref.p
    e_v = v - ref.v
    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    a_des = ref.a - kp_pos * e_p - kd_vel * e_v + GRAVITY * e3

    a_xy = np.linalg.norm(a_des[:2])
    a_z = max(1e-3, float(a_des[2]))
    tilt = math.atan2(a_xy, a_z)
    tilt_max = math.radians(max(1.0, max_tilt_deg))
    if tilt > tilt_max and a_xy > 1e-6:
        scale = math.tan(tilt_max) * a_z / a_xy
        a_des[0] *= scale
        a_des[1] *= scale

    if np.linalg.norm(a_des) < 1e-6:
        a_des = GRAVITY * e3
    b3_des = a_des / np.linalg.norm(a_des)
    b1_yaw = np.array([math.cos(ref.yaw), math.sin(ref.yaw), 0.0], dtype=float)
    b2_des = np.cross(b3_des, b1_yaw)
    n_b2 = np.linalg.norm(b2_des)
    if n_b2 < 1e-6:
        b2_des = np.array([-math.sin(ref.yaw), math.cos(ref.yaw), 0.0], dtype=float)
    else:
        b2_des /= n_b2
    b1_des = np.cross(b2_des, b3_des)
    R_des = np.column_stack([b1_des, b2_des, b3_des])

    e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
    e_R = vee_so3(e_R_mat)
    rate_cmd = -kR * e_R - kOmega * omega

    thrust_N = float(mass * np.dot(a_des, R[:, 2]))
    return thrust_N, rate_cmd, a_des, R_des


def l1_augment_ctbr(
    thrust_N: float,
    rate_cmd: np.ndarray,
    R: np.ndarray,
    a_ac: np.ndarray,
    mass: float,
    max_thrust: float,
    tilt_gain: float,
    max_rate: float,
) -> Tuple[float, np.ndarray]:
    """将 a_ac 映射到 thrust + body_rate（与 ROS 节点相同逻辑）。"""
    a_ac = np.asarray(a_ac, dtype=float).reshape(3)
    if not np.any(np.abs(a_ac) > 1e-9):
        return thrust_N, rate_cmd

    b3 = R[:, 2]
    thrust_N = float(thrust_N + mass * np.dot(a_ac, b3))
    thrust_N = float(np.clip(thrust_N, 0.0, max_thrust))

    a_tilt = a_ac.copy()
    a_tilt[2] += GRAVITY
    n_des = np.linalg.norm(a_tilt)
    if n_des < 1e-6:
        return thrust_N, rate_cmd
    b3_des = a_tilt / n_des
    d_rate = tilt_gain * np.cross(b3, b3_des)
    rate_cmd = np.clip(rate_cmd + d_rate, -max_rate, max_rate)
    return thrust_N, rate_cmd


def baseline_accel_world(thrust_N: float, R: np.ndarray, mass: float) -> np.ndarray:
    return (thrust_N / mass) * R[:, 2] - np.array([0.0, 0.0, GRAVITY], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 扰动模型
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DisturbanceConfig:
    k_drag: float = 0.35          # N·s/m per axis (F_drag = -k_drag * v)
    a_const: Tuple[float, float, float] = (0.25, -0.15, 0.0)  # m/s² 常值风
    delta_m: float = 0.0          # kg，真实质量 = m_nom + delta_m（t>=t_step 时）
    t_mass_step: float = 999.0    # 质量阶跃时刻


def true_mass(t: float, m_nom: float, cfg: DisturbanceConfig) -> float:
    return max(m_nom + (cfg.delta_m if t >= cfg.t_mass_step else 0.0), 1e-3)


def drag_const_accel(v: np.ndarray, m_true: float, cfg: DisturbanceConfig) -> np.ndarray:
    """气动阻力 + 常值风（作用在真实质量上）。"""
    sigma_drag = -cfg.k_drag * np.asarray(v, dtype=float) / m_true
    sigma_const = np.array(cfg.a_const, dtype=float)
    return sigma_drag + sigma_const


def lumped_sigma(
    thrust_N: float, b3: np.ndarray, v: np.ndarray, m_nom: float, m_true: float, cfg: DisturbanceConfig
) -> np.ndarray:
    """
    集总扰动（L1 视角）：σ = 真实加速度 - 名义加速度
      = (T/m_true - T/m_nom)·b3 + drag + const
    与预测器 v̇ = (T/m_nom)·b3 - g·e3 + σ 一致。
    """
    sigma_mass = thrust_N * (1.0 / m_true - 1.0 / m_nom) * np.asarray(b3, dtype=float)
    return sigma_mass + drag_const_accel(v, m_true, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 仿真主循环
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SimLog:
    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    p_err: np.ndarray
    sigma_hat: np.ndarray
    sigma_true: np.ndarray
    a_ac: np.ndarray
    thrust: np.ndarray


def simulate_case(
    *,
    use_l1: bool,
    use_disturbance: bool,
    t_end: float,
    dt: float,
    m_nom: float,
    dist_cfg: DisturbanceConfig,
    l1_params: L1Params,
    ref_fn: Callable[[float], Ref3D],
    att_tau: float = 0.08,
    ideal_attitude: bool = True,
    max_thrust: float = 40.0,
    max_rate: float = math.radians(120),
) -> SimLog:
    n = int(round(t_end / dt))
    t_arr = np.arange(n + 1) * dt

    ref0 = ref_fn(0.0)
    p = ref0.p.copy()
    v = ref0.v.copy()
    yaw0 = ref0.yaw
    R = np.array(
        [
            [math.cos(yaw0), -math.sin(yaw0), 0.0],
            [math.sin(yaw0), math.cos(yaw0), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    omega = np.zeros(3)

    l1 = L1AdaptiveAugmentation(l1_params)
    if use_l1:
        l1.set_enabled(True)
        l1.reset(v.copy())
    else:
        l1.set_enabled(False)

    dcfg = dist_cfg if use_disturbance else DisturbanceConfig(
        k_drag=0.0, a_const=(0.0, 0.0, 0.0), delta_m=0.0
    )

    p_log = np.zeros((n + 1, 3))
    v_log = np.zeros((n + 1, 3))
    pe_log = np.zeros((n + 1, 3))
    sh_log = np.zeros((n + 1, 3))
    st_log = np.zeros((n + 1, 3))
    ac_log = np.zeros((n + 1, 3))
    th_log = np.zeros(n + 1)

    # 预测器输入 = 上一拍“名义施加加速度”u_cmd =(T/m_nom)·b3 - g·e3（含 a_ac）。
    u_cmd_prev = np.zeros(3)
    e3 = np.array([0.0, 0.0, GRAVITY], dtype=float)

    for k in range(n + 1):
        t = float(t_arr[k])
        ref = ref_fn(t)
        p_log[k] = p
        v_log[k] = v
        pe_log[k] = ref.p - p
        sh_log[k] = l1.sigma_hat.copy()
        ac_log[k] = l1.a_ac.copy()

        if k >= n:
            break

        # L1 更新：预测器输入为上一拍 u_cmd（含 a_ac），并送入跟踪位置误差。
        a_ac = np.zeros(3)
        if use_l1:
            pos_err = p - ref.p
            a_ac = l1.step(dt, v, u_cmd_prev, pos_err_world=pos_err)

        thrust_b, rate_b, a_des, R_des = geometric_baseline(
            p, v, R, omega, ref, m_nom
        )

        if ideal_attitude:
            # 理想内环：姿态瞬间对齐含 L1 补偿的总期望加速度方向。
            a_des_total = a_des + a_ac
            nrm = float(np.linalg.norm(a_des_total))
            if nrm < 1e-6:
                a_des_total = e3.copy()
                nrm = GRAVITY
            b3_des = a_des_total / nrm
            b1_yaw = np.array([math.cos(ref.yaw), math.sin(ref.yaw), 0.0])
            b2 = np.cross(b3_des, b1_yaw)
            nb2 = np.linalg.norm(b2)
            b2 = b2 / nb2 if nb2 > 1e-6 else np.array([-math.sin(ref.yaw), math.cos(ref.yaw), 0.0])
            b1 = np.cross(b2, b3_des)
            R = np.column_stack([b1, b2, b3_des])
            thrust = float(m_nom * nrm)
            omega = rate_b.copy()
        else:
            thrust, rate_cmd = l1_augment_ctbr(
                thrust_b, rate_b, R, a_ac, m_nom, max_thrust, l1_params.tilt_gain, max_rate
            )
            omega = omega + (dt / max(att_tau, 1e-3)) * (rate_cmd - omega)
            omega = np.clip(omega, -max_rate, max_rate)
            dR = np.array(
                [
                    [0.0, -omega[2], omega[1]],
                    [omega[2], 0.0, -omega[0]],
                    [-omega[1], omega[0], 0.0],
                ],
                dtype=float,
            )
            R = R @ (np.eye(3) + dt * dR)
            u, _, vt = np.linalg.svd(R)
            R = u @ vt

        b3 = R[:, 2]
        th_log[k] = thrust

        # 名义施加加速度（L1 预测器下一拍输入）：含 a_ac，用 m_nom。
        u_cmd_prev = (thrust / m_nom) * b3 - e3

        m_true = true_mass(t, m_nom, dcfg)
        st_log[k] = lumped_sigma(thrust, b3, v, m_nom, m_true, dcfg)

        # 真实 plant：用真实质量 + 阻力/常值风（质量失配自然成为 σ 的一部分）。
        a_trans = (thrust / m_true) * b3 - e3 + drag_const_accel(v, m_true, dcfg)
        v = v + dt * a_trans
        p = p + dt * v

    return SimLog(
        t=t_arr,
        p=p_log,
        v=v_log,
        p_err=pe_log,
        sigma_hat=sh_log,
        sigma_true=st_log,
        a_ac=ac_log,
        thrust=th_log,
    )


def rmse_pos(log: SimLog) -> float:
    return float(np.sqrt(np.mean(np.sum(log.p_err**2, axis=1))))


def plot_comparison(
    cases: Dict[str, SimLog],
    dist_cfg: DisturbanceConfig,
    save: Optional[str],
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        "L1 geometric tracking — disturbance comparison\n"
        f"drag k={dist_cfg.k_drag:.2f} N·s/m, "
        f"const a={dist_cfg.a_const}, "
        f"Δm={dist_cfg.delta_m:.2f} kg @ t={dist_cfg.t_mass_step:.1f}s"
    )

    colors = {
        "baseline+dist": "C3",
        "L1(vel)+dist": "C1",
        "L1(vel+posI)+dist": "C0",
        "baseline(no dist)": "0.5",
    }

    ax = axes[0, 0]
    any_log = next(iter(cases.values()))
    ref_p = np.array([ref_circle(float(t)).p for t in any_log.t])
    ax.plot(ref_p[:, 0], ref_p[:, 1], "k--", lw=1.0, alpha=0.5, label="ref xy")
    for name, log in cases.items():
        ax.plot(log.p[:, 0], log.p[:, 1], lw=1.1, label=name, color=colors.get(name, None))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Trajectory (top view)")

    ax = axes[0, 1]
    for name, log in cases.items():
        err = np.linalg.norm(log.p_err, axis=1)
        ax.plot(log.t, err, lw=1.1, label=f"{name} RMSE={rmse_pos(log):.3f}m", color=colors.get(name, None))
    ax.set_xlabel("t [s]")
    ax.set_ylabel("||e_p|| [m]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Position tracking error")

    ax = axes[1, 0]
    log = cases.get("L1(vel+posI)+dist") or cases.get("L1(vel)+dist")
    if log is not None:
        ax.plot(log.t, log.sigma_true[:, 0], "k--", alpha=0.6, label=r"$\sigma_{true,x}$")
        ax.plot(log.t, log.sigma_true[:, 1], "k:", alpha=0.6, label=r"$\sigma_{true,y}$")
        ax.plot(log.t, log.sigma_hat[:, 0], "C1", lw=1.0, label=r"$\hat\sigma_x$ (L1)")
        ax.plot(log.t, log.sigma_hat[:, 1], "C3", lw=1.0, label=r"$\hat\sigma_y$ (L1)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("m/s²")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Disturbance estimate vs truth (disturbed + L1)")

    ax = axes[1, 1]
    if log is not None:
        ax.plot(log.t, log.sigma_true[:, 2], "k--", label=r"$\sigma_{true,z}$")
        ax.plot(log.t, log.sigma_hat[:, 2], "C1", label=r"$\hat\sigma_z$")
        ax.plot(log.t, log.a_ac[:, 2], "C2", alpha=0.8, label=r"$a_{ac,z}$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("m/s²")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Vertical channel (z)")

    ax = axes[2, 0]
    for name, log in cases.items():
        ax.plot(log.t, log.p_err[:, 2], lw=1.0, label=name, color=colors.get(name, None))
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_z [m]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Altitude error")

    ax = axes[2, 1]
    ax.axis("off")
    lines = ["RMSE position error [m]:", ""]
    for name, log in cases.items():
        lines.append(f"  {name:22s} {rmse_pos(log):.4f}")
    lines.append("")
    lines.append("predictor input = (T/m_nom)*b3 - g*e3 + a_ac")
    lines.append("=> sigma_hat -> true sigma (full compensation)")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)

    plt.tight_layout()
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="L1 vs baseline geometric tracking with disturbances")
    parser.add_argument("--t-end", type=float, default=25.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--m-nom", type=float, default=1.772)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--omega", type=float, default=0.25)
    parser.add_argument("--k-drag", type=float, default=0.35)
    parser.add_argument("--a-const", type=float, nargs=3, default=[0.25, -0.15, 0.0])
    parser.add_argument("--delta-m", type=float, default=0.25, help="payload mass added at t-step")
    parser.add_argument("--t-mass-step", type=float, default=12.0)
    parser.add_argument("--l1-as", type=float, default=8.0)
    parser.add_argument("--l1-wc", type=float, default=6.0)
    parser.add_argument("--l1-kpi", type=float, default=0.7, help="position-error integral gain")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    dist_cfg = DisturbanceConfig(
        k_drag=args.k_drag,
        a_const=tuple(args.a_const),
        delta_m=args.delta_m,
        t_mass_step=args.t_mass_step,
    )
    def make_l1(use_pos: bool) -> L1Params:
        return L1Params(
            enabled=True,
            as_gain=args.l1_as,
            wc_xy=args.l1_wc,
            wc_z=args.l1_wc,
            tilt_gain=3.0,
            use_pos_feedback=use_pos,
            k_pos_i_xy=args.l1_kpi,
            k_pos_i_z=args.l1_kpi,
        )

    def ref_fn(t: float) -> Ref3D:
        return ref_circle(t, omega=args.omega, radius=args.radius)

    common = dict(
        t_end=args.t_end,
        dt=args.dt,
        m_nom=args.m_nom,
        dist_cfg=dist_cfg,
        ref_fn=ref_fn,
    )

    print("Running comparison cases (with fixed disturbance)...")
    cases = {
        "baseline+dist": simulate_case(
            use_l1=False, use_disturbance=True, l1_params=make_l1(False), **common
        ),
        "L1(vel)+dist": simulate_case(
            use_l1=True, use_disturbance=True, l1_params=make_l1(False), **common
        ),
        "L1(vel+posI)+dist": simulate_case(
            use_l1=True, use_disturbance=True, l1_params=make_l1(True), **common
        ),
        "baseline(no dist)": simulate_case(
            use_l1=False, use_disturbance=False, l1_params=make_l1(False), **common
        ),
    }

    print("\n--- RMSE position error ---")
    for name, log in cases.items():
        print(f"  {name:22s} {rmse_pos(log):.4f} m")
    # 稳态（后 30%）位置误差，凸显 posI 对稳态误差的消除
    print("\n--- steady-state mean |e_p| (last 30%) ---")
    for name, log in cases.items():
        n = len(log.t)
        s = int(0.7 * n)
        ess = float(np.mean(np.linalg.norm(log.p_err[s:], axis=1)))
        print(f"  {name:22s} {ess:.4f} m")

    save = args.save.strip()
    if not save:
        root = Path(__file__).resolve().parents[2]
        save = str(root / "results" / "l1_geometric_tracking_compare.png")

    plot_comparison(cases, dist_cfg, save=save, show=not args.no_show)


if __name__ == "__main__":
    main()
