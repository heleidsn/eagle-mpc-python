#!/usr/bin/env python3
"""
Acados 悬停 NMPC + 质量突变 + L1 自适应增广 —— 闭环仿真验证。

目的
────
验证「L1 与 MPC 的配合」：在 acados 全状态 MPC 悬停的基础上，于 t_step 时刻
让 plant 的真实质量突然增加（模拟抓取物体），检验

  1) L1 能否正确估计出附加质量 Δm（est_added_mass ≈ Δm）；
  2) 加 L1 后悬停高度误差能否回到 0（MPC 单独时因模型质量错误留下稳态误差）。

为什么 MPC 单独不行
──────────────────
MPC 用名义质量 m_nom 求解，悬停推力 ≈ m_nom·g。质量突变后真实质量 m_true>m_nom，
该推力不足 → 掉高。MPC 无积分作用，只能在某个稳态 z 偏差处用反馈把推力提到平衡，
留下稳态高度误差 e_z ≈ Δ/ (位置增益)。

L1 如何修复
───────────
把"质量误差"视为竖直方向集总扰动：
    σ_z = ΣT·(1/m_true − 1/m_nom)            （世界系加速度）
L1 速度预测器估计 σ̂_z，补偿加速度 a_ac = −LPF(σ̂)，再映射成总推力增量
    ΔF = m_nom · (a_ac · b3)                  （均分到 4 个旋翼）
悬停补偿稳态满足 ΣT = m_true·g，可推出
    σ_z = −g·Δm/m_nom   ⇒   Δm = −m_nom·σ̂_z/g
即 L1AdaptiveAugmentation.estimated_added_mass() 在悬停下精确成立。

预测器输入（关键）
─────────────────
a_nom = (ΣT_total/m_nom)·b3 − g·e3，其中 ΣT_total 含 L1 注入的推力增量；
否则 σ̂ 只会收敛到真实扰动的一半（欠补偿）。

运行（需 eagle_mpc 环境：pinocchio + acados）：
    python l1_acados_hover_mass_jump_sim.py
    python l1_acados_hover_mass_jump_sim.py --no-show --delta-m 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import pinocchio as pin

from l1_adaptive import L1AdaptiveAugmentation, L1Params
from s500_uam_acados_realtime_mpc import AcadosFullStateRealtimeMPC

REPO = Path(__file__).resolve().parents[2]
GRAVITY = 9.81


def R_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = max((x * x + y * y + z * z + w * w) ** 0.5, 1e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def thrusts_to_tau(u_thrust: np.ndarray, rotors, cm_cf: float) -> np.ndarray:
    """4 旋翼推力 → free-flyer 广义力 [Fx,Fy,Fz, Mx,My,Mz]（机体系），与 acados 模型一致。"""
    Fz = float(np.sum(u_thrust[:4]))
    Mx = My = Mz = 0.0
    for i, r in enumerate(rotors):
        pos = r["translation"]
        spin = r["spin_direction"][0]
        T = float(u_thrust[i])
        Mx += -pos[1] * T
        My += pos[0] * T
        Mz += spin * cm_cf * T
    return np.array([0.0, 0.0, Fz, Mx, My, Mz], dtype=float)


def make_heavy_model(model: pin.Model, delta_m: float, com_offset: np.ndarray) -> pin.Model:
    """返回 base_link 质量增加 Δm 的 plant 模型拷贝（模拟刚性抓取物体）。"""
    heavy = pin.Model(model)
    if delta_m > 1e-9:
        # free-flyer 的 base body 惯性在 inertias[1]（0 为 universe）。
        com = np.asarray(com_offset, dtype=float).reshape(3)
        # 小球等效惯性（半径 0.03 m），主要关心质量项。
        r = 0.03
        I = 2.0 / 5.0 * delta_m * r * r
        heavy.inertias[1] = heavy.inertias[1] + pin.Inertia(
            float(delta_m), com, np.diag([I, I, I])
        )
    return heavy


def plant_step(
    model: pin.Model,
    data,
    x: np.ndarray,
    u_thrust: np.ndarray,
    dt: float,
    rotors,
    cm_cf: float,
) -> np.ndarray:
    """半隐式欧拉积分一拍 pinocchio 全动力学（free-flyer + 4 旋翼）。"""
    nq, nv = model.nq, model.nv
    q = np.asarray(x[:nq], dtype=float).copy()
    v = np.asarray(x[nq:], dtype=float).copy()
    tau = thrusts_to_tau(u_thrust, rotors, cm_cf)
    a = pin.aba(model, data, q, v, tau)
    v_next = v + dt * a
    q_next = pin.integrate(model, q, dt * v_next)
    return np.concatenate([q_next, v_next])


def hover_state(z0: float, nq: int, nv: int) -> np.ndarray:
    x = np.zeros(nq + nv, dtype=float)
    x[2] = z0
    x[6] = 1.0  # qw
    return x


def simulate(
    *,
    use_l1: bool,
    delta_m: float,
    t_step: float,
    t_end: float,
    control_dt: float,
    sim_dt: float,
    z0: float,
    l1_params: L1Params,
    mpc: AcadosFullStateRealtimeMPC,
    m_nom: float,
    rotors,
    cm_cf: float,
    com_offset: np.ndarray,
    thrust_gain: float = 1.0,
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
    if use_l1:
        l1.set_enabled(True)
        l1.reset(np.zeros(3))
    else:
        l1.set_enabled(False)

    mpc.reset_warm_start()
    mpc.warmup(x_hover, t_plan, x_plan, iters=5)

    n = int(round(t_end / control_dt))
    n_inner = max(1, int(round(control_dt / sim_dt)))
    e3g = np.array([0.0, 0.0, GRAVITY])

    x = x_hover.copy()
    u_cmd_prev = np.zeros(3)

    t_log = np.zeros(n + 1)
    z_log = np.zeros(n + 1)
    ez_log = np.zeros(n + 1)
    exy_log = np.zeros(n + 1)
    sh_log = np.zeros((n + 1, 3))
    st_log = np.zeros((n + 1, 3))
    mass_log = np.zeros(n + 1)
    thrust_log = np.zeros(n + 1)
    aac_log = np.zeros((n + 1, 3))

    for k in range(n + 1):
        t = k * control_dt
        q = x[:nq]
        v = x[nq:]
        R = R_from_quat_xyzw(q[3:7])
        b3 = R[:, 2]
        v_world = R @ v[:3]
        m_true = m_nom + (delta_m if t >= t_step else 0.0)

        t_log[k] = t
        z_log[k] = q[2]
        ez_log[k] = q[2] - z0
        exy_log[k] = float(np.hypot(q[0], q[1]))
        sh_log[k] = l1.sigma_hat.copy()
        aac_log[k] = l1.a_ac.copy()
        mass_log[k] = l1.estimated_added_mass(m_nom) if use_l1 else 0.0

        if k >= n:
            break

        # ── baseline: acados MPC（名义模型）──────────────────────────────
        u_mpc, _x_next, status = mpc.solve_step(x, t, t_plan, x_plan)
        u_thrust = np.asarray(u_mpc[:4], dtype=float).copy()

        # ── L1 更新（预测器输入 = 上一拍含补偿的名义加速度）──────────────
        a_ac = np.zeros(3)
        if use_l1:
            pos_err = q[:3] - x_hover[:3]
            a_ac = l1.step(control_dt, v_world, u_cmd_prev, pos_err_world=pos_err)
            # 推力注入：ΔF = m_nom·(a_ac·b3)，均分到 4 旋翼（纯竖直/总推力修正）。
            dF = float(m_nom * np.dot(a_ac, b3))
            u_thrust = np.clip(
                u_thrust + dF / 4.0, mpc.min_thrust, mpc.max_thrust
            )

        # 名义施加加速度（下一拍 L1 预测器输入）：基于**指令**推力 sumT，
        # 估计器并不知道执行器增益 thrust_gain（k 的偏差自动并入扰动 σ）。
        sumT = float(np.sum(u_thrust))
        u_cmd_prev = (sumT / m_nom) * b3 - e3g
        thrust_log[k] = sumT
        # 真实集总扰动（含执行器增益误差 + 质量误差）：
        #   σ = a_real - a_nom = (k·ΣT/m_true - ΣT/m_nom)·b3
        st_log[k] = sumT * (thrust_gain / m_true - 1.0 / m_nom) * b3

        # ── plant：执行器增益 k 作用在指令推力上（CTBR/推力映射不准）──────
        u_thrust_actual = float(thrust_gain) * u_thrust
        model = model_heavy if t >= t_step else model_nom
        data = data_heavy if t >= t_step else data_nom
        for _ in range(n_inner):
            x = plant_step(model, data, x, u_thrust_actual, sim_dt, rotors, cm_cf)

    # 末点（k=n 未解 MPC）补齐，避免绘图虚线掉到 0。
    st_log[n] = st_log[n - 1]
    thrust_log[n] = thrust_log[n - 1]

    return {
        "t": t_log,
        "z": z_log,
        "ez": ez_log,
        "exy": exy_log,
        "sigma_hat": sh_log,
        "sigma_true": st_log,
        "est_mass": mass_log,
        "thrust": thrust_log,
        "a_ac": aac_log,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Acados hover + mass jump + L1 verification")
    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--t-step", type=float, default=8.0, help="mass-jump time [s]")
    ap.add_argument("--delta-m", type=float, default=0.3, help="added payload mass [kg]")
    ap.add_argument("--z0", type=float, default=1.5, help="hover altitude [m]")
    ap.add_argument("--control-dt", type=float, default=0.02)
    ap.add_argument("--sim-dt", type=float, default=0.005)
    ap.add_argument("--dt-mpc", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--l1-as", type=float, default=8.0)
    ap.add_argument("--l1-wc", type=float, default=6.0)
    ap.add_argument("--com-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
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
    m_nom = float(sum(mpc.robot_model.inertias[i].mass for i in range(1, len(mpc.robot_model.inertias))))
    plat = mpc.s500_config["platform"]
    rotors = plat["$rotors"]
    cm_cf = plat["cm"] / plat["cf"]
    print(f"  m_nom = {m_nom:.3f} kg,  Δm = {args.delta_m:.3f} kg (at t={args.t_step}s)")

    com_offset = np.asarray(args.com_offset, dtype=float)
    common = dict(
        delta_m=args.delta_m,
        t_step=args.t_step,
        t_end=args.t_end,
        control_dt=args.control_dt,
        sim_dt=args.sim_dt,
        z0=args.z0,
        mpc=mpc,
        m_nom=m_nom,
        rotors=rotors,
        cm_cf=cm_cf,
        com_offset=com_offset,
    )
    l1p = L1Params(
        enabled=True, as_gain=args.l1_as, wc_xy=args.l1_wc, wc_z=args.l1_wc,
        tilt_gain=3.0, max_accel_z=8.0, max_accel_xy=6.0,
    )

    print("Running: MPC only (with mass jump)...")
    mpc_only = simulate(use_l1=False, l1_params=L1Params(enabled=False), **common)
    print("Running: MPC + L1 (with mass jump)...")
    mpc_l1 = simulate(use_l1=True, l1_params=l1p, **common)

    def tail(arr, t, frac=0.25):
        s = int((1 - frac) * len(t))
        return arr[s:]

    t = mpc_only["t"]
    print("\n=== steady-state (last 25%) altitude error e_z [m] ===")
    print(f"  MPC only     : {np.mean(np.abs(tail(mpc_only['ez'], t))):.4f}")
    print(f"  MPC + L1     : {np.mean(np.abs(tail(mpc_l1['ez'], t))):.4f}")
    print("\n=== mass estimate (last 25% mean) ===")
    print(f"  true Δm          : {args.delta_m:.4f} kg")
    print(f"  L1 est_added_mass: {np.mean(tail(mpc_l1['est_mass'], t)):.4f} kg")
    print(f"  L1 sigma_z (ss)  : {np.mean(tail(mpc_l1['sigma_hat'][:,2], t)):.4f} m/s^2"
          f"  (expected {-GRAVITY*args.delta_m/m_nom:.4f})")

    save = args.save.strip()
    if not save:
        save = str(REPO / "results" / "l1_acados_hover_mass_jump.png")
    plot(mpc_only, mpc_l1, args, m_nom, save, show=not args.no_show)


def plot(mpc_only, mpc_l1, args, m_nom, save, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = mpc_only["t"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Acados hover + mass jump + L1   (m_nom={m_nom:.3f} kg, Δm={args.delta_m:.2f} kg @ t={args.t_step}s)"
    )

    a = ax[0, 0]
    a.axhline(args.z0, color="k", ls="--", lw=1.0, alpha=0.6, label="ref z")
    a.axvline(args.t_step, color="gray", ls=":", lw=1.0)
    a.plot(t, mpc_only["z"], "C3", lw=1.2, label="MPC only")
    a.plot(t, mpc_l1["z"], "C0", lw=1.2, label="MPC + L1")
    a.set_xlabel("t [s]"); a.set_ylabel("altitude z [m]")
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.set_title("Hover altitude")

    a = ax[0, 1]
    a.axhline(0, color="k", lw=0.5, ls=":")
    a.axvline(args.t_step, color="gray", ls=":", lw=1.0)
    a.plot(t, mpc_only["ez"], "C3", lw=1.2, label="MPC only")
    a.plot(t, mpc_l1["ez"], "C0", lw=1.2, label="MPC + L1")
    a.set_xlabel("t [s]"); a.set_ylabel("e_z [m]")
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.set_title("Altitude error")

    a = ax[1, 0]
    a.axvline(args.t_step, color="gray", ls=":", lw=1.0)
    a.plot(t, mpc_l1["sigma_true"][:, 2], "k--", lw=1.2, label=r"$\sigma_{true,z}$")
    a.plot(t, mpc_l1["sigma_hat"][:, 2], "C1", lw=1.2, label=r"$\hat\sigma_z$ (L1)")
    a.plot(t, mpc_l1["a_ac"][:, 2], "C2", lw=1.0, alpha=0.8, label=r"$a_{ac,z}$")
    a.set_xlabel("t [s]"); a.set_ylabel("m/s²")
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.set_title("Vertical disturbance estimate")

    a = ax[1, 1]
    a.axhline(args.delta_m, color="k", ls="--", lw=1.2, alpha=0.7, label="true Δm")
    a.axvline(args.t_step, color="gray", ls=":", lw=1.0)
    a.plot(t, mpc_l1["est_mass"], "C0", lw=1.4, label="L1 est_added_mass")
    a.set_xlabel("t [s]"); a.set_ylabel("Δm [kg]")
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.set_title("Estimated added mass")

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
