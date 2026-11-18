#!/usr/bin/env python3
"""
Double-integrator plant + PVA trajectory tracking demo.

Plant (per axis, decoupled 2D):  p_dot = v,  v_dot = u  (u = acceleration command)

PVA control law (feedforward acceleration + PD on tracking errors):
    u = a_ref + Kp * (p_ref - p) + Kd * (v_ref - v)

Reference: analytically differentiable trajectory p_ref(t), v_ref(t), a_ref(t).

Run:
    python double_integrator_pva_tracking.py
    python double_integrator_pva_tracking.py --trajectory circle --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class TrajectoryName(str, Enum):
    LISSAJOUS = "lissajous"
    CIRCLE = "circle"
    LINE = "line"


@dataclass
class TrajectorySample:
    """Reference position, velocity, acceleration at time t."""

    p: np.ndarray  # shape (2,)
    v: np.ndarray
    a: np.ndarray


def trajectory_lissajous(t: np.ndarray, omega: float = 0.4) -> TrajectorySample:
    """Smooth figure-eight style path in the plane."""
    w = omega
    p = np.stack(
        [
            np.sin(w * t),
            0.6 * np.sin(2.0 * w * t),
        ],
        axis=-1,
    )
    v = np.stack(
        [
            w * np.cos(w * t),
            0.6 * 2.0 * w * np.cos(2.0 * w * t),
        ],
        axis=-1,
    )
    a = np.stack(
        [
            -(w**2) * np.sin(w * t),
            -0.6 * (2.0 * w) ** 2 * np.sin(2.0 * w * t),
        ],
        axis=-1,
    )
    return TrajectorySample(p=p, v=v, a=a)


def trajectory_circle(t: np.ndarray, omega: float = 0.35, radius: float = 2.0) -> TrajectorySample:
    """Uniform circular motion: p = R [cos(wt), sin(wt)]."""
    w = omega
    c = np.cos(w * t)
    s = np.sin(w * t)
    p = radius * np.stack([c, s], axis=-1)
    v = radius * w * np.stack([-s, c], axis=-1)
    a = -(w**2) * p
    return TrajectorySample(p=p, v=v, a=a)


def trajectory_line(t: np.ndarray) -> TrajectorySample:
    """Rest-to-rest move along x using a smooth step (polynomial blend)."""
    T = float(np.max(t)) if t.size else 1.0
    T = max(T, 1e-6)
    tau = np.clip(t / T, 0.0, 1.0)
    # s(tau) = 6 tau^5 - 15 tau^4 + 10 tau^3  (zero vel/acc at ends)
    s = 6.0 * tau**5 - 15.0 * tau**4 + 10.0 * tau**3
    ds = (30.0 * tau**4 - 60.0 * tau**3 + 30.0 * tau**2) / T
    dds = (120.0 * tau**3 - 180.0 * tau**2 + 60.0 * tau) / (T**2)
    length = 3.0
    p = np.stack([length * s, np.zeros_like(t)], axis=-1)
    v = np.stack([length * ds, np.zeros_like(t)], axis=-1)
    a = np.stack([length * dds, np.zeros_like(t)], axis=-1)
    return TrajectorySample(p=p, v=v, a=a)


def sample_reference(name: TrajectoryName, t: np.ndarray) -> TrajectorySample:
    if name == TrajectoryName.LISSAJOUS:
        return trajectory_lissajous(t)
    if name == TrajectoryName.CIRCLE:
        return trajectory_circle(t)
    if name == TrajectoryName.LINE:
        return trajectory_line(t)
    raise ValueError(f"unknown trajectory {name!r}")


def double_integrator_step(
    p: np.ndarray, v: np.ndarray, u: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit Euler for p_dot = v, v_dot = u."""
    v_next = v + dt * u
    p_next = p + dt * v_next
    return p_next, v_next


def pva_control(
    p: np.ndarray,
    v: np.ndarray,
    ref: TrajectorySample,
    kp: float,
    kd: float,
    u_max: Optional[float],
) -> np.ndarray:
    """Acceleration command: feedforward + PD on (p,v) errors."""
    e_p = ref.p - p
    e_v = ref.v - v
    u = ref.a * 1 + kp * e_p + kd * e_v
    if u_max is not None and u_max > 0:
        n = np.linalg.norm(u)
        if n > u_max:
            u = u * (u_max / n)
    return u


def simulate(
    traj: TrajectoryName,
    t_end: float,
    dt: float,
    kp: float,
    kd: float,
    u_max: Optional[float],
    p0: np.ndarray,
    v0: np.ndarray,
) -> dict:
    n_steps = int(np.round(t_end / dt))
    t = np.arange(n_steps + 1, dtype=float) * dt
    ref = sample_reference(traj, t)

    p = np.zeros((n_steps + 1, 2))
    v = np.zeros_like(p)
    u = np.zeros((n_steps, 2))
    p[0] = p0
    v[0] = v0

    for k in range(n_steps):
        rk = TrajectorySample(p=ref.p[k], v=ref.v[k], a=ref.a[k])
        u[k] = pva_control(p[k], v[k], rk, kp, kd, u_max)
        p[k + 1], v[k + 1] = double_integrator_step(p[k], v[k], u[k], dt)

    return {
        "t": t,
        "p": p,
        "v": v,
        "u": u,
        "p_ref": ref.p,
        "v_ref": ref.v,
        "a_ref": ref.a,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Double integrator + PVA tracking demo")
    parser.add_argument(
        "--trajectory",
        type=str,
        default=TrajectoryName.LISSAJOUS.value,
        choices=[e.value for e in TrajectoryName],
    )
    parser.add_argument("--t-end", type=float, default=30.0, help="simulation horizon [s]")
    parser.add_argument("--dt", type=float, default=0.01, help="integrator step [s]")
    parser.add_argument("--kp", type=float, default=20, help="position error gain")
    parser.add_argument("--kd", type=float, default=1, help="velocity error gain")
    parser.add_argument("--u-max", type=float, default=3.0, help="acceleration limit (<=0 disables)")
    parser.add_argument("--no-show", action="store_true", help="save PNG only, do not open window")
    parser.add_argument("--save", type=str, default="", help="optional path to save figure")
    args = parser.parse_args()

    traj = TrajectoryName(args.trajectory)
    ref0 = sample_reference(traj, np.array([0.0]))
    p0 = np.asarray(ref0.p[0], dtype=float)
    v0 = np.asarray(ref0.v[0], dtype=float)
    u_max: Optional[float] = args.u_max if args.u_max > 0 else None

    log = simulate(
        traj=traj,
        t_end=args.t_end,
        dt=args.dt,
        kp=args.kp,
        kd=args.kd,
        u_max=u_max,
        p0=p0,
        v0=v0,
    )

    t = log["t"]
    p, v = log["p"], log["v"]
    p_ref, v_ref, a_ref = log["p_ref"], log["v_ref"], log["a_ref"]
    u_cmd = log["u"]
    t_u = t[:-1]
    a_ref_u = a_ref[:-1]  # same length as u_cmd (reference at step k)

    e_p = p_ref - p
    e_v = v_ref - v

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    umax_title = f"{args.u_max}" if args.u_max > 0 else "none"
    fig.suptitle(
        f"Double integrator PVA tracking — {traj.value} "
        f"(Kp={args.kp}, Kd={args.kd}, u_max={umax_title})"
    )

    ax = axes[0, 0]
    ax.plot(p_ref[:, 0], p_ref[:, 1], "k--", lw=1.0, label="reference")
    ax.plot(p[:, 0], p[:, 1], lw=1.2, label="closed-loop")
    ax.scatter([p0[0]], [p0[1]], c="C1", s=36, zorder=5, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, e_p[:, 0], lw=1.0, label=r"$e_x$")
    ax.plot(t, e_p[:, 1], lw=1.0, label=r"$e_y$")
    ax.plot(t, np.linalg.norm(e_p, axis=1), "k--", lw=0.9, alpha=0.6, label=r"$\|e_p\|$")
    ax.axhline(0.0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("position error [m]")
    ax.set_xlabel("t [s]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, e_v[:, 0], lw=1.0, label=r"$e_{vx}$")
    ax.plot(t, e_v[:, 1], lw=1.0, label=r"$e_{vy}$")
    ax.plot(t, np.linalg.norm(e_v, axis=1), "k--", lw=0.9, alpha=0.6, label=r"$\|e_v\|$")
    ax.axhline(0.0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("velocity error [m/s]")
    ax.set_xlabel("t [s]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_u, u_cmd[:, 0], lw=1.0, label=r"$u_x$ (cmd)")
    ax.plot(t_u, u_cmd[:, 1], lw=1.0, label=r"$u_y$ (cmd)")
    ax.plot(t_u, a_ref_u[:, 0], "k--", alpha=0.55, lw=0.9, label=r"$a_{x,ref}$")
    ax.plot(t_u, a_ref_u[:, 1], "k:", alpha=0.55, lw=0.9, label=r"$a_{y,ref}$")
    ax.axhline(0.0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("acceleration [m/s²]")
    ax.set_xlabel("t [s]")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t, p_ref[:, 0], "k--", alpha=0.5, label="x_ref")
    ax.plot(t, p[:, 0], lw=1.0, label="x")
    ax.plot(t, p_ref[:, 1], "k:", alpha=0.5, label="y_ref")
    ax.plot(t, p[:, 1], lw=1.0, label="y")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("position [m]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t, v_ref[:, 0], "k--", alpha=0.5, label="vx_ref")
    ax.plot(t, v[:, 0], lw=1.0, label="vx")
    ax.plot(t, v_ref[:, 1], "k:", alpha=0.5, label="vy_ref")
    ax.plot(t, v[:, 1], lw=1.0, label="vy")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("velocity [m/s]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = args.save.strip()
    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved figure: {out}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    rmse = float(np.sqrt(np.mean(np.sum((p_ref - p) ** 2, axis=1))))
    print(f"Position RMSE (2-norm over time): {rmse:.4f} m")


if __name__ == "__main__":
    main()
