#!/usr/bin/env python3
"""
Acados 悬停 + 一阶内环延迟 —— 复现 ROS 中「纯 L1 有静差、开 posI 归零」。

与 l1_acados_hover_mass_jump_sim 的区别
────────────────────────────────────
离线理想仿真：a_ac 当拍完整作用到 plant，预测器输入也用同一 a_ac → 稳态误差可归零。

ROS / PX4 CTBR：竖直 a_ac 经推力较快实现；水平 a_ac 经 bodyrate→姿态→比力，
存在带宽/相位滞后。预测器仍假定 u_cmd = a_nom + a_ac（整段 a_ac 已施加），
plant 实际只得到部分/延迟的水平补偿 → 纯 L1 留下稳态位置误差。

本脚本用 lumped 一阶模型近似内环：
    τ·ȧ_xy,eff + a_xy,eff = a_ac,xy          （水平补偿实现滞后）
    a_ac,z  → 仍当拍注入总推力（与 ROS 推力通道一致）

并固定「预测器输入 = 指令 a_ac」（predictor_sees_commanded=True，模拟 ROS）。

对比四组：
  1) MPC only
  2) MPC + L1 理想（τ=0，无内环延迟）
  3) MPC + L1 + 内环延迟，无位置积分（ROS 纯 L1）
  4) MPC + L1 + 内环延迟 + 位置积分（ROS + l1_pos_fb）

扰动：恒定水平风加速度 + 可选质量突变（竖直）。

运行（eagle_mpc 环境）：
    python l1_acados_inner_loop_delay_sim.py
    python l1_acados_inner_loop_delay_sim.py --no-show --tau-inner 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import pinocchio as pin

from l1_adaptive import L1AdaptiveAugmentation, L1Params
from s500_uam_acados_realtime_mpc import AcadosFullStateRealtimeMPC
from l1_acados_hover_mass_jump_sim import (
    GRAVITY,
    R_from_quat_xyzw,
    hover_state,
    make_heavy_model,
    plant_step,
    thrusts_to_tau,
)

REPO = Path(__file__).resolve().parents[2]


class FirstOrderXYInnerLoop:
    """
    水平 a_ac 实现模型（模拟 PX4 bodyrate→姿态→比力）：
        τ·ȧ_eff + a_eff = k_achieve·a_cmd
    稳态 a_eff = k_achieve·a_cmd（k<1 表示内环无法 100% 实现指令比力）。
    预测器若假定 a_cmd 已全部施加，则直流处留下 (1-k)·σ 残差 → 静差。
    """

    def __init__(self, tau: float, k_achieve: float = 0.72):
        self.tau = float(max(tau, 1e-6))
        self.k_achieve = float(np.clip(k_achieve, 0.05, 1.0))
        self.a_xy_eff = np.zeros(2, dtype=float)

    def reset(self) -> None:
        self.a_xy_eff[:] = 0.0

    def step(self, a_ac_xy: np.ndarray, dt: float) -> np.ndarray:
        cmd = self.k_achieve * np.asarray(a_ac_xy, dtype=float).reshape(2)
        alpha = float(np.clip(dt / self.tau, 0.0, 1.0))
        self.a_xy_eff = (1.0 - alpha) * self.a_xy_eff + alpha * cmd
        return self.a_xy_eff.copy()


def plant_step_with_extras(
    model: pin.Model,
    data,
    x: np.ndarray,
    u_thrust: np.ndarray,
    dt: float,
    rotors,
    cm_cf: float,
    a_wind_world: np.ndarray,
    f_xy_world: np.ndarray,
) -> np.ndarray:
    """plant_step + 风扰 + 水平 lumped 内环补偿力。"""
    nq, nv = model.nq, model.nv
    q = np.asarray(x[:nq], dtype=float).copy()
    v = np.asarray(x[nq:], dtype=float).copy()

    tau = thrusts_to_tau(u_thrust, rotors, cm_cf)
    R = R_from_quat_xyzw(q[3:7])
    m = float(pin.computeTotalMass(model))

    # 水平 lumped 内环补偿 + 恒定风（世界系加速度 → 机体系力）
    F_world = np.zeros(3, dtype=float)
    fxy = np.asarray(f_xy_world, dtype=float).reshape(3)
    F_world += fxy
    if np.any(np.abs(a_wind_world) > 1e-12):
        F_world += m * np.asarray(a_wind_world, dtype=float).reshape(3)
    if np.linalg.norm(F_world) > 1e-12:
        tau[:3] += R.T @ F_world

    a = pin.aba(model, data, q, v, tau)
    v_next = v + dt * a
    q_next = pin.integrate(model, q, dt * v_next)
    return np.concatenate([q_next, v_next])


def simulate(
    *,
    label: str,
    use_l1: bool,
    l1_params: L1Params,
    mpc: AcadosFullStateRealtimeMPC,
    m_nom: float,
    rotors,
    cm_cf: float,
    t_end: float,
    control_dt: float,
    sim_dt: float,
    z0: float,
    delta_m: float,
    t_step: float,
    com_offset: np.ndarray,
    a_wind: np.ndarray,
    wind_start: float,
    tau_inner: float,
    k_inner: float,
    predictor_sees_commanded: bool,
    freeze_mpc_after: float,
    hover_only_baseline: bool,
) -> dict:
    nq, nv = mpc.nq, mpc.nv
    x_hover = hover_state(z0, nq, nv)
    t_plan = np.array([0.0, 2.0 * t_end], dtype=float)
    x_plan = np.vstack([x_hover, x_hover])

    model_nom = pin.Model(mpc.robot_model)
    data_nom = model_nom.createData()
    model_heavy = make_heavy_model(model_nom, delta_m, com_offset)
    data_heavy = model_heavy.createData()

    l1 = L1AdaptiveAugmentation(l1_params)
    l1.set_enabled(use_l1)
    if use_l1:
        l1.reset(np.zeros(3))

    inner = FirstOrderXYInnerLoop(tau_inner, k_achieve=k_inner)
    mpc.reset_warm_start()
    mpc.warmup(x_hover, t_plan, x_plan, iters=5)

    n = int(round(t_end / control_dt))
    n_inner = max(1, int(round(control_dt / sim_dt)))
    e3g = np.array([0.0, 0.0, GRAVITY])
    a_wind = np.asarray(a_wind, dtype=float).reshape(3)

    x = x_hover.copy()
    u_cmd_prev = np.zeros(3)
    u_mpc_frozen: Optional[np.ndarray] = None

    t_log = np.zeros(n + 1)
    p_log = np.zeros((n + 1, 3))
    ez_log = np.zeros(n + 1)
    exy_log = np.zeros(n + 1)
    aac_log = np.zeros((n + 1, 3))
    axy_eff_log = np.zeros((n + 1, 2))

    for k in range(n + 1):
        t = k * control_dt
        q = x[:nq]
        v = x[nq:]
        R = R_from_quat_xyzw(q[3:7])
        b3 = R[:, 2]
        v_world = R @ v[:3]

        t_log[k] = t
        p_log[k] = q[:3].copy()
        ez_log[k] = q[2] - z0
        exy_log[k] = float(np.hypot(q[0], q[1]))
        aac_log[k] = l1.a_ac.copy()
        axy_eff_log[k] = inner.a_xy_eff.copy()

        if k >= n:
            break

        if hover_only_baseline:
            # 仅竖直悬停推力，无 MPC 水平纠偏 → 风扰靠 L1 水平通道补偿
            u_thrust = np.ones(4, dtype=float) * (m_nom * GRAVITY / 4.0)
        else:
            freeze_now = freeze_mpc_after > 0.0 and t >= freeze_mpc_after
            if freeze_now and u_mpc_frozen is not None:
                u_thrust = u_mpc_frozen.copy()
            else:
                u_mpc, _, _ = mpc.solve_step(x, t, t_plan, x_plan)
                u_thrust = np.asarray(u_mpc[:4], dtype=float).copy()
                if freeze_now and u_mpc_frozen is None:
                    u_mpc_frozen = u_thrust.copy()

        a_ac = np.zeros(3)
        if use_l1:
            pos_err = q[:3] - x_hover[:3]
            a_ac = l1.step(control_dt, v_world, u_cmd_prev, pos_err_world=pos_err)

            # 竖直：当拍推力（ROS 推力通道；悬停时 ≈ m·a_ac,z）
            dF = float(m_nom * np.dot(a_ac, b3))
            u_thrust = np.clip(u_thrust + dF / 4.0, mpc.min_thrust, mpc.max_thrust)

        sumT = float(np.sum(u_thrust))
        a_nom = (sumT / m_nom) * b3 - e3g

        # 水平：仅经一阶内环实现（ROS bodyrate→姿态），不重复计入推力
        a_xy_eff = inner.step(a_ac[:2], control_dt) if use_l1 else np.zeros(2)
        if tau_inner < 1e-9:
            a_xy_eff = a_ac[:2].copy()
        # 预测器“以为”的施加加速度（ROS：u_cmd = a_nom + 整段 a_ac）
        a_predicted = a_nom + a_ac if use_l1 else a_nom
        # plant 实际得到的水平比力（滞后）
        a_realized = np.array([a_xy_eff[0], a_xy_eff[1], a_ac[2]], dtype=float)

        if predictor_sees_commanded and use_l1:
            u_cmd_prev = a_predicted
        elif use_l1:
            u_cmd_prev = a_nom + a_realized
        else:
            u_cmd_prev = a_nom

        f_xy_world = np.array([m_nom * a_xy_eff[0], m_nom * a_xy_eff[1], 0.0], dtype=float)

        model = model_heavy if t >= t_step else model_nom
        data = data_heavy if t >= t_step else data_nom
        a_wind_now = a_wind if t >= wind_start else np.zeros(3)
        for _ in range(n_inner):
            x = plant_step_with_extras(
                model, data, x, u_thrust, sim_dt, rotors, cm_cf, a_wind_now, f_xy_world
            )

    return {
        "label": label,
        "t": t_log,
        "p": p_log,
        "ez": ez_log,
        "exy": exy_log,
        "a_ac": aac_log,
        "a_xy_eff": axy_eff_log,
    }


def steady_norm(log, key, frac=0.25, t_start: float = 0.0):
    t = log["t"]
    s = int(np.searchsorted(t, t_start, side="left"))
    s = max(s, int((1 - frac) * len(t)))
    return float(np.mean(np.abs(log[key][s:])))


def main() -> None:
    ap = argparse.ArgumentParser(description="Acados hover + inner-loop delay vs posI")
    ap.add_argument("--t-end", type=float, default=18.0)
    ap.add_argument("--z0", type=float, default=1.5)
    ap.add_argument("--control-dt", type=float, default=0.02)
    ap.add_argument("--sim-dt", type=float, default=0.005)
    ap.add_argument("--dt-mpc", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--delta-m", type=float, default=0.0)
    ap.add_argument("--t-step", type=float, default=6.0)
    ap.add_argument("--wind-xy", type=float, nargs=2, default=[0.5, 0.38])
    ap.add_argument("--wind-start", type=float, default=6.0, help="apply wind after [s]")
    ap.add_argument("--tau-inner", type=float, default=0.3,
                    help="inner-loop time constant for horizontal a_ac [s]")
    ap.add_argument("--k-inner", type=float, default=0.7,
                    help="steady-state gain of horizontal inner loop (0..1)")
    ap.add_argument("--w-pos-scale", type=float, default=0.2,
                    help="scale MPC w_pos (<1 exposes L1 inner-loop mismatch)")
    ap.add_argument("--l1-max-accel-xy", type=float, default=0.32,
                    help="cap on L1 horizontal accel (m/s^2); mimics limited CTBR authority")
    ap.add_argument(
        "--freeze-mpc-after", type=float, default=0.0,
        help="freeze MPC baseline thrust after this time [s]; 0=never freeze",
    )
    ap.add_argument(
        "--hover-only", action="store_true",
        help="constant hover thrust only (no MPC); isolates L1 XY channel",
    )
    ap.add_argument("--l1-as", type=float, default=8.0)
    ap.add_argument("--l1-wc", type=float, default=6.0)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

    urdf = str(REPO / "models" / "urdf" / "s500_simple.urdf")
    print(f"Building acados MPC (tau_inner={args.tau_inner}s, k_inner={args.k_inner})...")
    w_pos = 6.0 * float(args.w_pos_scale)
    mpc = AcadosFullStateRealtimeMPC(
        urdf_path=urdf,
        dt_mpc=args.dt_mpc,
        horizon=args.horizon,
        w_state_track=10.0,
        w_terminal_track=50.0,
        w_pos=w_pos,
        w_att=2.0,
        w_vel=1.0,
        w_omega=1.0,
        w_control=1e-2,
        w_u_thrust=1.0,
        solver_mode="sqp",
        integrator_type="ERK",
        max_iter=10,
    )
    m_nom = float(sum(mpc.robot_model.inertias[i].mass for i in range(1, len(mpc.robot_model.inertias))))
    plat = mpc.s500_config["platform"]
    rotors = plat["$rotors"]
    cm_cf = plat["cm"] / plat["cf"]
    a_wind = np.array([args.wind_xy[0], args.wind_xy[1], 0.0])

    common = dict(
        mpc=mpc,
        m_nom=m_nom,
        rotors=rotors,
        cm_cf=cm_cf,
        t_end=args.t_end,
        control_dt=args.control_dt,
        sim_dt=args.sim_dt,
        z0=args.z0,
        delta_m=args.delta_m,
        t_step=args.t_step,
        com_offset=np.zeros(3),
        a_wind=a_wind,
        predictor_sees_commanded=True,
        freeze_mpc_after=float(args.freeze_mpc_after),
        wind_start=float(args.wind_start),
        hover_only_baseline=bool(args.hover_only),
    )

    max_xy = float(args.l1_max_accel_xy)
    l1_base = L1Params(
        enabled=True, as_gain=args.l1_as, wc_xy=args.l1_wc, wc_z=args.l1_wc,
        tilt_gain=3.0, max_accel_z=8.0, max_accel_xy=max_xy,
        use_pos_feedback=False,
    )
    l1_pos = L1Params(
        enabled=True, as_gain=args.l1_as, wc_xy=args.l1_wc, wc_z=args.l1_wc,
        tilt_gain=3.0, max_accel_z=8.0,
        # 允许 posI 在 L1 通道饱和后仍叠加（ROS 中建议略放宽或分项限幅）
        max_accel_xy=max(max_xy, 1.2),
        use_pos_feedback=True,
        k_pos_i_xy=1.0, k_pos_i_z=0.8,
        k_pos_p_xy=0.4,
        max_pos_integral_xy=2.5, max_pos_integral_z=1.5,
    )

    if args.hover_only:
        cases = [
            ("hover only (no L1)", False, l1_base, 0.0),
            ("L1 ideal (τ=0)", True, l1_base, 0.0),
            ("L1 + delay (ROS-like)", True, l1_base, args.tau_inner),
            ("L1 + delay + posI", True, l1_pos, args.tau_inner),
        ]
    else:
        cases = [
            ("MPC only", False, l1_base, 0.0),
            ("L1 ideal (τ=0)", True, l1_base, 0.0),
            ("L1 + delay (ROS-like)", True, l1_base, args.tau_inner),
            ("L1 + delay + posI", True, l1_pos, args.tau_inner),
        ]

    logs = []
    print("\n  case                      | e_xy(ss) | e_z(ss)")
    print("  --------------------------+----------+---------")
    for label, use_l1, lp, tau in cases:
        print(f"  Running {label}...")
        log = simulate(
            label=label,
            use_l1=use_l1,
            l1_params=lp if use_l1 else L1Params(enabled=False),
            tau_inner=float(tau),
            k_inner=float(args.k_inner if tau > 1e-9 else 1.0),
            **common,
        )
        logs.append(log)
        t_ss = max(float(args.wind_start), float(args.freeze_mpc_after)) + 3.0
        print(
            f"  {label:25s} | {steady_norm(log, 'exy', t_start=t_ss):7.4f}  | "
            f"{steady_norm(log, 'ez', t_start=t_ss):7.4f}"
        )

    save = args.save.strip() or str(REPO / "results" / "l1_acados_inner_loop_delay.png")
    plot(logs, args, save, show=not args.no_show)


def plot(logs, args, save, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Acados hover — inner loop τ={args.tau_inner}s, k={args.k_inner}, "
        f"wind={args.wind_xy}, Δm={args.delta_m}kg @ t={args.t_step}s"
    )

    colors = ["C3", "C2", "C0", "C1"]
    for log, c in zip(logs, colors):
        ax[0, 0].plot(log["t"], log["exy"], color=c, lw=1.2, label=log["label"])
        ax[0, 1].plot(log["t"], log["ez"], color=c, lw=1.2, label=log["label"])
    ax[0, 0].set_ylabel("|e_xy| [m]")
    ax[0, 1].set_ylabel("e_z [m]")
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].legend(fontsize=7)
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 0].set_title("Horizontal tracking error")
    ax[0, 1].set_title("Altitude error")

    # 延迟案例：指令 a_ac_xy vs 实际实现
    delayed = logs[2]
    ax[1, 0].plot(delayed["t"], delayed["a_ac"][:, 0], "C0--", lw=1, label="a_ac_x cmd")
    ax[1, 0].plot(delayed["t"], delayed["a_xy_eff"][:, 0], "C0-", lw=1.2, label="a_xy,eff")
    ax[1, 0].set_ylabel("m/s²")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].set_title("L1+delay: commanded vs realized a_x")

    ax[1, 1].axis("off")
    txt = (
        "Model: predictor u_cmd = a_nom + a_ac (full, like ROS);\n"
        "plant XY: τ·ȧ_eff + a_eff = k·a_ac,xy (k<1 → DC gain error);\n"
        "Z: thrust from a_ac·b3 (instant).\n\n"
        "Expect: delay+k<1 → steady e_xy; posI → e_xy→0."
    )
    ax[1, 1].text(0.05, 0.5, txt, fontsize=10, va="center", family="monospace")

    for a in ax.flat:
        if a is not ax[1, 1]:
            a.set_xlabel("t [s]")

    plt.tight_layout()
    out = Path(save)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
