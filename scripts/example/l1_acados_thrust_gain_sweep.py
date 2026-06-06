#!/usr/bin/env python3
"""
Acados 悬停 + 执行器推力增益误差 k —— L1 鲁棒性验证（扫描实验）。

动机
────
真实 CTBR/PX4 链路里，归一化推力 → 实际力 的映射并不精确：
    实际推力 = k · 指令推力   （k≠1：max_thrust_total 标错、推力曲线、电池下垂…）
MPC 与 L1 预测器都**只知道指令推力**（不知道 k），因此 k 的偏差会被并入
L1 的集总扰动 σ。本实验扫描 k，验证：

  • MPC only：常值推力增益误差 → 稳态高度误差（无积分，补不掉）；
  • MPC + L1：把 (k-1) 当扰动估计并补偿 → 稳态误差是否仍归零；
  • k 偏离 1 多大时 L1 开始失稳/出现明显残差。

复用 l1_acados_hover_mass_jump_sim.simulate()（执行器增益经 thrust_gain 注入）。

运行（需 eagle_mpc 环境）：
    python l1_acados_thrust_gain_sweep.py
    python l1_acados_thrust_gain_sweep.py --no-show --gains 0.7 0.85 1.0 1.15 1.3
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


def steady_metrics(log, frac=0.25):
    t = log["t"]
    s = int((1 - frac) * len(t))
    ez = float(np.mean(np.abs(log["ez"][s:])))
    ez_std = float(np.std(log["ez"][s:]))
    zmax = float(np.max(np.abs(log["z"][s:] - (log["z"][0]))))
    diverged = bool(
        not np.all(np.isfinite(log["z"]))
        or np.max(np.abs(log["ez"])) > 5.0
    )
    return ez, ez_std, diverged


def main() -> None:
    ap = argparse.ArgumentParser(description="Acados hover thrust-gain sweep (L1 robustness)")
    ap.add_argument("--t-end", type=float, default=14.0)
    ap.add_argument("--z0", type=float, default=1.5)
    ap.add_argument("--control-dt", type=float, default=0.02)
    ap.add_argument("--sim-dt", type=float, default=0.005)
    ap.add_argument("--dt-mpc", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--l1-as", type=float, default=8.0)
    ap.add_argument("--l1-wc", type=float, default=6.0)
    ap.add_argument(
        "--gains", type=float, nargs="+",
        default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        help="actuator thrust gains k to sweep",
    )
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

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
    print(f"  m_nom = {m_nom:.3f} kg")

    common = dict(
        delta_m=0.0,          # 隔离推力增益误差，无质量突变
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
    )
    l1p = L1Params(
        enabled=True, as_gain=args.l1_as, wc_xy=args.l1_wc, wc_z=args.l1_wc,
        tilt_gain=3.0, max_accel_z=8.0, max_accel_xy=6.0,
    )

    rows = []
    logs_l1 = {}
    print("\n  k     | MPC-only e_z(ss) | MPC+L1 e_z(ss) | L1 diverged?")
    print("  ------+------------------+----------------+-------------")
    for k in args.gains:
        mpc_only = simulate(use_l1=False, l1_params=L1Params(enabled=False),
                            thrust_gain=float(k), **common)
        mpc_l1 = simulate(use_l1=True, l1_params=l1p, thrust_gain=float(k), **common)
        ez_b, _, _ = steady_metrics(mpc_only)
        ez_l, ez_l_std, div_l = steady_metrics(mpc_l1)
        rows.append((k, ez_b, ez_l, ez_l_std, div_l))
        logs_l1[k] = mpc_l1
        print(f"  {k:4.2f}  |     {ez_b:7.4f}      |    {ez_l:7.4f}     |   {'YES' if div_l else 'no'}")

    save = args.save.strip()
    if not save:
        save = str(REPO / "results" / "l1_acados_thrust_gain_sweep.png")
    plot(rows, logs_l1, args, m_nom, save, show=not args.no_show)


def plot(rows, logs_l1, args, m_nom, save, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r[0] for r in rows]
    ez_b = [r[1] for r in rows]
    ez_l = [r[2] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Acados hover — actuator thrust-gain error robustness   (m_nom={m_nom:.3f} kg)"
    )

    a = ax[0]
    a.plot(ks, ez_b, "C3-o", lw=1.6, label="MPC only")
    a.plot(ks, ez_l, "C0-o", lw=1.6, label="MPC + L1")
    a.axvline(1.0, color="gray", ls=":", lw=1.0)
    a.set_xlabel("actuator thrust gain  k  (actual = k · commanded)")
    a.set_ylabel("steady-state |e_z|  [m]")
    a.set_yscale("log")
    a.legend(fontsize=9)
    a.grid(True, which="both", alpha=0.3)
    a.set_title("Steady-state altitude error vs k")

    a = ax[1]
    for k, log in sorted(logs_l1.items()):
        a.plot(log["t"], log["ez"], lw=1.1, label=f"k={k:.2f}")
    a.axhline(0, color="k", lw=0.5, ls=":")
    a.set_xlabel("t [s]")
    a.set_ylabel("e_z [m]  (MPC + L1)")
    a.legend(fontsize=7, ncol=2)
    a.grid(True, alpha=0.3)
    a.set_title("MPC+L1 altitude-error transient per k")

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
