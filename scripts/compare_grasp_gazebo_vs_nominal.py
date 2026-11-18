#!/usr/bin/env python3
"""Compare Gazebo tracking log vs nominal Acados EE closed-loop on grasp trajectory."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from s500_uam_acados_ee_realtime_mpc import AcadosEECentricRealtimeMPC  # noqa: E402
from s500_uam_acados_model import build_acados_model  # noqa: E402
from s500_uam_closed_loop_plant import CasadiRK4Plant, mpc_inner_stride  # noqa: E402
from s500_uam_trajectory_planner import compute_ee_kinematics_along_trajectory  # noqa: E402

GAZEBO_CSV = _REPO / "tracking_results/2__grasp__acados_ee_pose.csv"
TRAJ_NPZ = _REPO / "research/wbc_hover_underactuated/trajectory/traj_s500uam_grasp.npz"
URDF = _REPO / "models/urdf/s500_uam_simple.urdf"
OUT_DIR = _REPO / "tracking_results/analysis_plots"
CACHE_NPZ = OUT_DIR / "grasp_compare_cache.npz"


def _set_equal_aspect_3d(ax, *point_arrays, pad: float = 0.05) -> None:
    """Equal x/y/z scale so 3D trajectories are not visually stretched."""
    chunks = []
    for arr in point_arrays:
        a = np.asarray(arr, dtype=float).reshape(-1, 3)
        if a.size:
            chunks.append(a)
    if not chunks:
        return
    pts = np.vstack(chunks)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * float(np.max(maxs - mins))
    half = max(half * (1.0 + pad), 0.25)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _load_gazebo_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    t = np.array([float(r["time"]) for r in rows], dtype=float)
    pos = np.array([[float(r["px"]), float(r["py"]), float(r["pz"])] for r in rows])
    quat = np.array(
        [[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows]
    )
    pref = np.array(
        [[float(r["ref_px"]), float(r["ref_py"]), float(r["ref_pz"])] for r in rows]
    )
    return {"t": t, "pos": pos, "quat": quat, "pref": pref, "rows": rows}


def _build_refs(t_plan: np.ndarray, x_plan: np.ndarray, pin_model: pin.Model, eid: int):
    ee_pos, _, ee_rpy, _ = compute_ee_kinematics_along_trajectory(
        x_plan, pin_model, pin_model.createData(), eid
    )
    nq = pin_model.nq
    nv = pin_model.nv
    p_base = x_plan[:, 0:3]
    q = x_plan[:, 3:7]
    yaw_base = np.arctan2(
        2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
        1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2),
    )
    joint = x_plan[:, 7 : 7 + (nq - 7)]
    joint_vel = x_plan[:, nq + 6 : nq + 6 + (nq - 7)]
    return {
        "t_ref_ee": t_plan.copy(),
        "p_ref_ee": ee_pos.copy(),
        "yaw_ref_ee": ee_rpy[:, 2].copy(),
        "roll_ref_ee": ee_rpy[:, 0].copy(),
        "pitch_ref_ee": ee_rpy[:, 1].copy(),
        "p_base_ref": p_base.copy(),
        "yaw_base_ref": yaw_base.copy(),
        "joint_ref": joint.copy(),
        "joint_vel_ref": joint_vel.copy(),
        "p_ref_ee_plan": ee_pos.copy(),
    }


def run_nominal_closed_loop(
    x0: np.ndarray,
    t_plan: np.ndarray,
    x_plan: np.ndarray,
    refs: dict,
    *,
    T_sim: float,
    sim_dt: float = 0.001,
    control_dt: float = 0.02,
    dt_mpc: float = 0.05,
    horizon: int = 25,
) -> dict:
    mpc = AcadosEECentricRealtimeMPC(
        urdf_path=str(URDF),
        dt_mpc=dt_mpc,
        horizon=horizon,
        w_ee_pos=500.0,
        w_ee_yaw=200.0,
        w_ee_rot_rp=1.0,
        w_state_track=2.0,
        w_joint=0.2,
        w_vel=0.1,
        w_omega=0.1,
        w_joint_vel=0.05,
        w_state_reg=0.05,
        w_control=1e-4,
        w_terminal_scale=3.0,
        solver_mode="rti",
    )
    am, pin_model, nq, nv, nu = build_acados_model(str(URDF))
    import casadi as ca

    f = ca.Function("f", [am.x, am.u], [am.f_expl_expr])
    plant = CasadiRK4Plant(lambda x, u: np.array(f(x, u)).flatten(), sim_dt, nu)
    eid = mpc.ee_frame_id
    pdata = pin_model.createData()

    t0 = float(t_plan[0])
    mpc.warmup(
        x0,
        refs["t_ref_ee"],
        refs["p_ref_ee"],
        refs["yaw_ref_ee"],
        iters=8,
    )

    stride = mpc_inner_stride(sim_dt, control_dt)
    n_steps = int(round(T_sim / sim_dt))
    x = np.asarray(x0, dtype=float).flatten().copy()
    u_hold = mpc.u_hover.copy()

    ts, xs, us, ee_pos = [], [], [], []
    t_el = 0.0

    for k in range(n_steps + 1):
        ts.append(t_el)
        xs.append(x.copy())
        pin.forwardKinematics(pin_model, pdata, x[:nq])
        pin.updateFramePlacements(pin_model, pdata)
        ee_pos.append(pdata.oMf[eid].translation.copy())

        if k >= n_steps:
            break

        if k % stride == 0:
            t_query = t0 + t_el
            u_opt, x_next, status = mpc.solve_step(
                x,
                t_query,
                refs["t_ref_ee"],
                refs["p_ref_ee"],
                refs["yaw_ref_ee"],
                refs["roll_ref_ee"],
                refs["pitch_ref_ee"],
                t_plan=t_plan,
                x_plan=x_plan,
            )
            if u_opt is not None:
                u_hold = u_opt

        x = plant.step(x, u_hold)
        t_el += sim_dt

    return {
        "t": np.asarray(ts),
        "x": np.asarray(xs),
        "u": np.asarray(us),
        "ee": np.asarray(ee_pos),
        "nq": nq,
        "pin_model": pin_model,
        "eid": eid,
    }


def _interp_plan(tq: float, t_plan: np.ndarray, arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.array([np.interp(tq, t_plan, arr)])
    out = np.zeros(arr.shape[1], dtype=float)
    for j in range(arr.shape[1]):
        out[j] = np.interp(tq, t_plan, arr[:, j])
    return out


def _ee_fk(pin_model, eid, x_full, nq):
    q = np.asarray(x_full[:nq], dtype=float)
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q)
    pin.updateFramePlacements(pin_model, data)
    return data.oMf[eid].translation.copy()


def make_plots(gz: dict, sim_plan0: dict, sim_gz0: dict, refs: dict, t_plan, x_plan):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pin_model = sim_plan0["pin_model"]
    eid = sim_plan0["eid"]
    nq = sim_plan0["nq"]
    t0 = float(t_plan[0])

    # Align time: gazebo t starts ~0, sim t starts 0
    t_gz = gz["t"]
    t_sim = sim_plan0["t"]
    T = min(t_gz[-1], t_sim[-1], float(t_plan[-1] - t0))

    # Reference base / EE along plan
    n_s = 300
    t_ref = np.linspace(0, T, n_s)
    pref = np.array([_interp_plan(t0 + ti, t_plan, x_plan[:, 0:3]) for ti in t_ref])
    ee_ref = np.array(
        [
            _ee_fk(pin_model, eid, _interp_plan(t0 + ti, t_plan, x_plan), nq)
            for ti in t_ref
        ]
    )

    pos_gz = gz["pos"]
    pos_sim = sim_plan0["x"][:, 0:3]
    pos_sim_gz0 = sim_gz0["x"][:, 0:3]
    ee_sim = sim_plan0["ee"]
    ee_sim_gz0 = sim_gz0["ee"]

    def _sample(t_src, arr, t_dst):
        out = np.zeros((len(t_dst), arr.shape[1]), dtype=float)
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(t_dst, t_src, arr[:, j])
        return out

    tg = t_gz[t_gz <= T]
    ts = t_sim[t_sim <= T]
    pg = _sample(t_gz, pos_gz, tg)
    ps = _sample(t_sim, pos_sim, tg)
    psg = _sample(t_sim, pos_sim_gz0, tg)
    pr = _sample(t_ref + 0.0, pref, tg)  # t_ref is 0..T
    pr[:, :] = np.array([_interp_plan(t0 + ti, t_plan, x_plan[:, 0:3]) for ti in tg])

    err_gz = np.linalg.norm(pg - pr, axis=1)
    err_sim = np.linalg.norm(ps - pr, axis=1)
    err_sim_gz0 = np.linalg.norm(psg - pr, axis=1)

    ee_ref_g = np.array(
        [_ee_fk(pin_model, eid, _interp_plan(t0 + ti, t_plan, x_plan), nq) for ti in tg]
    )
    ees = _sample(t_sim, ee_sim, tg)
    eegz0 = _sample(t_sim, ee_sim_gz0, tg)
    err_ee_sim = np.linalg.norm(ees - ee_ref_g, axis=1)
    err_ee_gz0 = np.linalg.norm(eegz0 - ee_ref_g, axis=1)

    # --- Single figure: base + EE ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.2, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.plot(pr[:, 0], pr[:, 1], pr[:, 2], "k--", lw=1.8, label="plan base")
    ax3d.plot(ee_ref_g[:, 0], ee_ref_g[:, 1], ee_ref_g[:, 2], "k-.", lw=1.8, label="plan EE")
    ax3d.plot(ps[:, 0], ps[:, 1], ps[:, 2], "C0-", lw=1.4, label="sim base")
    ax3d.plot(ees[:, 0], ees[:, 1], ees[:, 2], color="C0", ls=":", lw=1.6, label="sim EE")
    ax3d.plot(pg[:, 0], pg[:, 1], pg[:, 2], "C3-", lw=1.4, label="Gazebo base")
    ax3d.scatter(*ps[0], c="C0", s=36, marker="o", zorder=5)
    ax3d.scatter(*pg[0], c="C3", s=36, marker="o", zorder=5)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("Base + EE 3D — grasp / acados_ee_pose\n(Gazebo log: no arm joints, EE omitted)")
    ax3d.legend(fontsize=7, loc="upper left")
    _set_equal_aspect_3d(ax3d, pr, ps, pg, ee_ref_g, ees)

    ax_berr = fig.add_subplot(gs[0, 1])
    ax_berr.plot(
        tg, err_gz * 1000, "C3-", label=f"Gazebo RMSE={np.sqrt(np.mean(err_gz**2))*1e3:.1f} mm"
    )
    ax_berr.plot(
        tg, err_sim * 1000, "C0-", label=f"sim plan0 RMSE={np.sqrt(np.mean(err_sim**2))*1e3:.1f} mm"
    )
    ax_berr.plot(
        tg,
        err_sim_gz0 * 1000,
        "C1--",
        label=f"sim gz0 RMSE={np.sqrt(np.mean(err_sim_gz0**2))*1e3:.1f} mm",
    )
    ax_berr.axvspan(0, 1, color="gray", alpha=0.12)
    ax_berr.set_xlabel("t [s]")
    ax_berr.set_ylabel("|p_base - p_plan| [mm]")
    ax_berr.set_title("Base error")
    ax_berr.legend(fontsize=7)
    ax_berr.grid(True, alpha=0.3)

    ax_eerr = fig.add_subplot(gs[1, 1])
    ax_eerr.plot(
        tg,
        err_ee_sim * 1000,
        "C0-",
        label=f"sim plan0 RMSE={np.sqrt(np.mean(err_ee_sim**2))*1e3:.1f} mm",
    )
    ax_eerr.plot(
        tg,
        err_ee_gz0 * 1000,
        "C1--",
        label=f"sim gz0 RMSE={np.sqrt(np.mean(err_ee_gz0**2))*1e3:.1f} mm",
    )
    ax_eerr.axvspan(0, 1, color="gray", alpha=0.12)
    ax_eerr.set_xlabel("t [s]")
    ax_eerr.set_ylabel("|p_ee - p_ee_plan| [mm]")
    ax_eerr.set_title("EE error (nominal sim)")
    ax_eerr.legend(fontsize=7)
    ax_eerr.grid(True, alpha=0.3)

    fig.tight_layout()
    p_main = OUT_DIR / "grasp_gazebo_vs_nominal.png"
    fig.savefig(p_main, dpi=150)
    plt.close(fig)

    # XY top view (base + EE on one axes)
    fig_xy, ax_xy = plt.subplots(figsize=(8, 6))
    ax_xy.plot(pr[:, 0], pr[:, 1], "k--", lw=1.4, label="plan base")
    ax_xy.plot(ee_ref_g[:, 0], ee_ref_g[:, 1], "k-.", lw=1.4, label="plan EE")
    ax_xy.plot(ps[:, 0], ps[:, 1], "C0-", lw=1.2, label="sim base")
    ax_xy.plot(ees[:, 0], ees[:, 1], color="C0", ls=":", lw=1.4, label="sim EE")
    ax_xy.plot(pg[:, 0], pg[:, 1], "C3-", lw=1.2, label="Gazebo base")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Top view (XY) — base + EE")
    ax_xy.axis("equal")
    ax_xy.legend(fontsize=8)
    ax_xy.grid(True, alpha=0.3)
    fig_xy.tight_layout()
    p_xy = OUT_DIR / "grasp_gazebo_vs_nominal_xy.png"
    fig_xy.savefig(p_xy, dpi=150)
    plt.close(fig_xy)

    # Summary text
    summary = OUT_DIR / "grasp_gazebo_vs_nominal_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("grasp acados_ee_pose: Gazebo vs nominal closed-loop\n")
        f.write(f"Gazebo CSV: {GAZEBO_CSV.name}\n")
        f.write(f"Trajectory: {TRAJ_NPZ.name}\n\n")
        for label, err in [
            ("Gazebo base (full)", err_gz),
            ("Sim x0=plan base", err_sim),
            ("Sim x0=Gazebo base", err_sim_gz0),
            ("Sim x0=plan EE", err_ee_sim),
            ("Sim x0=Gazebo EE", err_ee_gz0),
        ]:
            f.write(
                f"{label}: mean={err.mean()*1e3:.2f} mm, "
                f"rmse={np.sqrt(np.mean(err**2))*1e3:.2f} mm, max={err.max()*1e3:.2f} mm\n"
            )
        mask = tg >= 2.0
        f.write("\nAfter t>=2s:\n")
        for label, err in [
            ("Gazebo base", err_gz[mask]),
            ("Sim plan0 base", err_sim[mask]),
            ("Sim plan0 EE", err_ee_sim[mask]),
        ]:
            f.write(
                f"  {label}: rmse={np.sqrt(np.mean(err**2))*1e3:.2f} mm\n"
            )

    return p_main, p_xy, summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Replot from cache (skip MPC closed-loop; requires grasp_compare_cache.npz)",
    )
    args = parser.parse_args()

    if not GAZEBO_CSV.is_file():
        raise SystemExit(f"Missing Gazebo log: {GAZEBO_CSV}")
    if not TRAJ_NPZ.is_file():
        raise SystemExit(f"Missing trajectory: {TRAJ_NPZ}")

    gz = _load_gazebo_csv(GAZEBO_CSV)
    d = np.load(TRAJ_NPZ)
    t_plan = np.asarray(d["t_plan"], dtype=float).flatten()
    x_plan = np.asarray(d["x_plan"], dtype=float)
    T_sim = min(float(t_plan[-1] - t_plan[0]), float(gz["t"][-1]))

    pin_model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    eid = pin_model.getFrameId("gripper_link")
    refs = _build_refs(t_plan, x_plan, pin_model, eid)

    if args.plots_only:
        if not CACHE_NPZ.is_file():
            raise SystemExit(f"No cache at {CACHE_NPZ}; run without --plots-only first.")
        c = np.load(CACHE_NPZ, allow_pickle=True)
        sim_plan0 = {
            "t": c["sim_t"],
            "x": c["sim_x"],
            "ee": c["sim_ee"],
            "nq": int(c["nq"]),
            "pin_model": pin_model,
            "eid": eid,
        }
        sim_gz0 = {
            "t": c["sim_gz0_t"],
            "x": c["sim_gz0_x"],
            "ee": c["sim_gz0_ee"],
            "nq": int(c["nq"]),
            "pin_model": pin_model,
            "eid": eid,
        }
    else:
        x_plan0 = x_plan[0].copy()
        x_gz0 = x_plan0.copy()
        x_gz0[0:3] = gz["pos"][0]
        x_gz0[3:7] = gz["quat"][0]

        print(f"Running nominal sim T={T_sim:.2f}s from x_plan[0]...")
        sim_plan0 = run_nominal_closed_loop(
            x_plan0, t_plan, x_plan, refs, T_sim=T_sim, control_dt=0.02
        )
        print(f"Running nominal sim from Gazebo initial pose...")
        sim_gz0 = run_nominal_closed_loop(
            x_gz0, t_plan, x_plan, refs, T_sim=T_sim, control_dt=0.02
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            CACHE_NPZ,
            sim_t=sim_plan0["t"],
            sim_x=sim_plan0["x"],
            sim_ee=sim_plan0["ee"],
            sim_gz0_t=sim_gz0["t"],
            sim_gz0_x=sim_gz0["x"],
            sim_gz0_ee=sim_gz0["ee"],
            nq=sim_plan0["nq"],
        )

    paths = make_plots(gz, sim_plan0, sim_gz0, refs, t_plan, x_plan)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
