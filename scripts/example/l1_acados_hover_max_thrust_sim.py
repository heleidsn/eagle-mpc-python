#!/usr/bin/env python3
"""
Acados 悬停 + max_thrust 估计错误 —— 纯速度型 L1 能否消除竖直静差？

场景（对应 ROS + PX4 推力转换）
──────────────────────────────
MPC 输出推力 T_cmd [N]
  → 归一化   thrust_norm = T_cmd / max_thrust_assumed   （控制器假定的最大推力）
  → PX4 实际 T_actual    = thrust_norm · max_thrust_real （真实最大推力）
  ⇒ T_actual = (max_thrust_real / max_thrust_assumed) · T_cmd = k · T_cmd

例：real=30 N, assumed=50 N → k=0.6（每单位指令推力实际只产生 60%）。
MPC 不知道 k，按名义推力悬停 → 实际推力不足 → 掉高 → 稳态高度静差（MPC 无积分）。

核心问题
────────
只用「速度型 L1」（速度预测器 + 扰动估计，**不含**位置积分 posI）能否消除？

理论（竖直 1D 稳态）
────────────────────
  σ_true = a_real − a_nom = (k−1)·T_cmd/m         （预测器只看指令 T_cmd）
  速度型 L1 稳态：σ̂→σ_true，a_ac,z=−σ̂ ⇒ ΔF_ac=(1−k)·T_cmd
  代入 T_cmd=T_base+ΔF_ac  ⇒  k·T_cmd = T_base  ⇒  T_actual = T_base
  即「实际推力 = MPC 名义悬停推力」，而 MPC 名义下零静差 ⇒ 系统回到 z0。

结论：能消除（只要不饱和）。L1 补偿推力与真实推力走同一条 PX4 转换通道，
      增益误差被闭环吸收。这与水平通道（a_ac 经 bodyrate→姿态、有延迟/装不满）
      本质不同——水平那条路才需要 posI 兜底。

运行（eagle_mpc 环境）：
    python l1_acados_hover_max_thrust_sim.py
    python l1_acados_hover_max_thrust_sim.py --max-thrust-real 30 --max-thrust-assumed 50
    python l1_acados_hover_max_thrust_sim.py --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from l1_adaptive import L1Params
from s500_uam_acados_realtime_mpc import AcadosFullStateRealtimeMPC
from l1_acados_hover_mass_jump_sim import simulate

REPO = Path(__file__).resolve().parents[2]
GRAVITY = 9.81


def tail(arr, t, frac=0.25):
    s = int((1 - frac) * len(t))
    return arr[s:]


def main() -> None:
    ap = argparse.ArgumentParser(description="Acados hover + max_thrust mismatch (velocity-L1)")
    ap.add_argument("--t-end", type=float, default=12.0)
    ap.add_argument("--z0", type=float, default=1.5)
    ap.add_argument("--max-thrust-real", type=float, default=30.0,
                    help="true total max thrust [N] (what motors actually deliver)")
    ap.add_argument("--max-thrust-assumed", type=float, default=50.0,
                    help="max thrust [N] assumed by controller normalization")
    ap.add_argument("--control-dt", type=float, default=0.02)
    ap.add_argument("--sim-dt", type=float, default=0.005)
    ap.add_argument("--dt-mpc", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--l1-as", type=float, default=8.0)
    ap.add_argument("--l1-wc", type=float, default=6.0)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

    k = float(args.max_thrust_real) / float(args.max_thrust_assumed)

    urdf = str(REPO / "models" / "urdf" / "s500_simple.urdf")
    print(f"Building acados hover MPC (s500, dt_mpc={args.dt_mpc}, N={args.horizon})...")
    mpc = AcadosFullStateRealtimeMPC(
        urdf_path=urdf,
        dt_mpc=args.dt_mpc,
        horizon=args.horizon,
        w_state_track=10.0,
        w_terminal_track=50.0,
        w_pos=6.0,
        w_att=2.0,
        w_vel=1.0,
        w_omega=1.0,
        w_control=1e-2,
        w_u_thrust=1.0,
        solver_mode="sqp",
        integrator_type="ERK",
        max_iter=10,
    )
    m_nom = float(
        sum(mpc.robot_model.inertias[i].mass for i in range(1, len(mpc.robot_model.inertias)))
    )
    plat = mpc.s500_config["platform"]
    rotors = plat["$rotors"]
    cm_cf = plat["cm"] / plat["cf"]
    hover_N = m_nom * GRAVITY
    print(f"  m_nom = {m_nom:.3f} kg,  hover thrust = {hover_N:.2f} N")
    print(f"  max_thrust real={args.max_thrust_real:.1f} N, assumed={args.max_thrust_assumed:.1f} N"
          f"  ->  k = {k:.3f}")
    print(f"  cmd thrust needed to hover = hover/k = {hover_N / k:.2f} N "
          f"(per-rotor {hover_N / k / 4:.2f} N, motor max {mpc.max_thrust:.2f} N)")

    common = dict(
        delta_m=0.0,        # 不改质量，纯推力增益（max_thrust）误差
        t_step=1e9,
        t_end=args.t_end,
        control_dt=args.control_dt,
        sim_dt=args.sim_dt,
        z0=args.z0,
        mpc=mpc,
        m_nom=m_nom,
        rotors=rotors,
        cm_cf=cm_cf,
        com_offset=np.zeros(3),
        thrust_gain=k,
    )

    l1_vel = L1Params(
        enabled=True, as_gain=args.l1_as, wc_xy=args.l1_wc, wc_z=args.l1_wc,
        tilt_gain=3.0, max_accel_z=8.0, max_accel_xy=6.0,
        use_pos_feedback=False,                         # 纯速度型 L1
    )
    l1_pos = L1Params(
        enabled=True, as_gain=args.l1_as, wc_xy=args.l1_wc, wc_z=args.l1_wc,
        tilt_gain=3.0, max_accel_z=8.0, max_accel_xy=6.0,
        use_pos_feedback=True, k_pos_i_xy=0.6, k_pos_i_z=0.8,
        max_pos_integral_xy=1.5, max_pos_integral_z=1.5,
    )

    print("\nRunning: MPC only...")
    mpc_only = simulate(use_l1=False, l1_params=L1Params(enabled=False), **common)
    print("Running: MPC + velocity-only L1...")
    mpc_vel = simulate(use_l1=True, l1_params=l1_vel, **common)
    print("Running: MPC + L1 + posI...")
    mpc_pos = simulate(use_l1=True, l1_params=l1_pos, **common)

    t = mpc_only["t"]
    print("\n=== steady-state (last 25%) altitude error |e_z| [m] ===")
    print(f"  MPC only              : {np.mean(np.abs(tail(mpc_only['ez'], t))):.5f}")
    print(f"  MPC + velocity-only L1: {np.mean(np.abs(tail(mpc_vel['ez'], t))):.5f}")
    print(f"  MPC + L1 + posI       : {np.mean(np.abs(tail(mpc_pos['ez'], t))):.5f}")
    print("\n=== vertical disturbance estimate (velocity-only L1, last 25%) ===")
    sig_z = np.mean(tail(mpc_vel["sigma_hat"][:, 2], t))
    sig_expected = (k - 1.0) * (hover_N / k) / m_nom  # (k-1)·T_cmd/m, T_cmd=hover/k
    print(f"  sigma_hat_z (ss) : {sig_z:.4f} m/s^2  (expected (k-1)*T_cmd/m = {sig_expected:.4f})")

    save = args.save.strip()
    if not save:
        save = str(REPO / "results" / "l1_acados_hover_max_thrust.png")
    plot(mpc_only, mpc_vel, mpc_pos, args, m_nom, k, hover_N, save, show=not args.no_show)


def plot(mpc_only, mpc_vel, mpc_pos, args, m_nom, k, hover_N, save, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = mpc_only["t"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Acados hover — max_thrust mismatch (real={args.max_thrust_real:.0f}N, "
        f"assumed={args.max_thrust_assumed:.0f}N, k={k:.2f})   m_nom={m_nom:.3f} kg"
    )

    a = ax[0, 0]
    a.axhline(args.z0, color="k", ls="--", lw=1.0, alpha=0.6, label="ref z")
    a.plot(t, mpc_only["z"], "C3", lw=1.3, label="MPC only")
    a.plot(t, mpc_vel["z"], "C0", lw=1.3, label="MPC + velocity-L1")
    a.plot(t, mpc_pos["z"], "C2", lw=1.0, ls="--", label="MPC + L1 + posI")
    a.set_xlabel("t [s]"); a.set_ylabel("altitude z [m]")
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.set_title("Hover altitude")

    a = ax[0, 1]
    a.axhline(0, color="k", lw=0.5, ls=":")
    a.plot(t, mpc_only["ez"], "C3", lw=1.3, label="MPC only")
    a.plot(t, mpc_vel["ez"], "C0", lw=1.3, label="MPC + velocity-L1")
    a.plot(t, mpc_pos["ez"], "C2", lw=1.0, ls="--", label="MPC + L1 + posI")
    a.set_xlabel("t [s]"); a.set_ylabel("e_z [m]")
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.set_title("Altitude error")

    a = ax[1, 0]
    a.plot(t, mpc_vel["sigma_true"][:, 2], "k--", lw=1.2, label=r"$\sigma_{true,z}$")
    a.plot(t, mpc_vel["sigma_hat"][:, 2], "C1", lw=1.3, label=r"$\hat\sigma_z$ (vel-L1)")
    a.plot(t, mpc_vel["a_ac"][:, 2], "C2", lw=1.0, alpha=0.8, label=r"$a_{ac,z}$")
    a.set_xlabel("t [s]"); a.set_ylabel("m/s²")
    a.legend(fontsize=8); a.grid(True, alpha=0.3)
    a.set_title("Vertical disturbance estimate (velocity-only L1)")

    a = ax[1, 1]
    a.axhline(hover_N, color="k", ls="--", lw=1.0, alpha=0.7, label="hover mg")
    a.plot(t, mpc_only["thrust"], "C3", lw=1.2, label="T_cmd (MPC only)")
    a.plot(t, mpc_vel["thrust"], "C0", lw=1.2, label="T_cmd (vel-L1)")
    a.plot(t, k * mpc_vel["thrust"], "C0", lw=1.0, ls=":", label="T_actual = k·T_cmd (vel-L1)")
    a.set_xlabel("t [s]"); a.set_ylabel("thrust [N]")
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.set_title("Commanded vs actual total thrust")

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
