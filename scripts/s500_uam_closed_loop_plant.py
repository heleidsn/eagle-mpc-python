#!/usr/bin/env python3
"""
Common **plant dynamics** interface for closed-loop tracking:
single-step integration ``x_{k+1} = Φ(x_k, u_k)`` shared by

- Crocoddyl ``IntegratedActionModelEuler`` (Pinocchio multibody + actuation),
- Acados / CasADi explicit model with RK4.

MPC solvers stay per-algorithm; only the simulation roll-out is unified here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import crocoddyl
except ImportError:  # pragma: no cover
    crocoddyl = None  # type: ignore


def rk4_step(
    f_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One RK4 step for ``ẋ = f(x,u)`` with ZOH on ``u``."""
    x = np.asarray(x, dtype=float).flatten()
    u = np.asarray(u, dtype=float).flatten()
    k1 = np.array(f_fun(x, u)).flatten()
    k2 = np.array(f_fun(x + 0.5 * dt * k1, u)).flatten()
    k3 = np.array(f_fun(x + 0.5 * dt * k2, u)).flatten()
    k4 = np.array(f_fun(x + dt * k3, u)).flatten()
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def crocoddyl_euler_step(
    integrated: "crocoddyl.IntegratedActionModelEuler",
    data,
    x: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """One explicit Euler step of a Crocoddyl integrated action model."""
    x = np.asarray(x, dtype=float).flatten()
    u = np.asarray(u, dtype=float).flatten()
    integrated.calc(data, x, u)
    return np.array(data.xnext, dtype=float).flatten().copy()


class CrocoddylEulerPlant:
    """Plant from ``crocoddyl.IntegratedActionModelEuler`` (same as MPC dynamics when model matches)."""

    __slots__ = ("_inte", "_data", "nu")

    def __init__(
        self,
        integrated: "crocoddyl.IntegratedActionModelEuler",
        data=None,
    ):
        if crocoddyl is None:
            raise ImportError("crocoddyl is required for CrocoddylEulerPlant")
        self._inte = integrated
        self._data = data if data is not None else integrated.createData()
        self.nu = int(integrated.nu)

    def on_pre_step(self, t: float, step_index: int) -> None:
        """Optional hook (e.g. time-varying inertia); default no-op."""
        return None

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return crocoddyl_euler_step(self._inte, self._data, x, u)


class CasadiRK4Plant:
    """Explicit dynamics via CasADi ``f_fun(x,u) -> xdot`` and fixed RK4 sub-steps of length ``sim_dt``."""

    __slots__ = ("_f", "_dt", "nu")

    def __init__(
        self,
        f_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
        sim_dt: float,
        nu: int,
    ):
        self._f = f_fun
        self._dt = float(sim_dt)
        self.nu = int(nu)

    def on_pre_step(self, t: float, step_index: int) -> None:
        return None

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return rk4_step(self._f, x, u, self._dt)


class PayloadSchedulePlant:
    """Runs ``apply_once()`` on the first ``on_pre_step`` with ``t >= t_trigger`` (sim-only load, etc.)."""

    __slots__ = ("_inner", "_t_trig", "_apply_once", "_done", "nu")

    def __init__(
        self,
        inner: CrocoddylEulerPlant,
        t_trigger: float,
        apply_once: Callable[[], None],
    ):
        self._inner = inner
        self._t_trig = float(t_trigger)
        self._apply_once = apply_once
        self._done = False
        self.nu = inner.nu

    @property
    def schedule_applied(self) -> bool:
        return self._done

    def on_pre_step(self, t: float, step_index: int) -> None:
        if not self._done and t + 1e-12 >= self._t_trig:
            self._apply_once()
            self._done = True

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._inner.step(x, u)


def mpc_inner_stride(control_dt: float, sim_dt: float) -> int:
    """Number of simulation sub-steps per MPC update (ZOH), ≥1."""
    sim_dt = float(sim_dt)
    control_dt = float(control_dt)
    if sim_dt <= 0 or control_dt <= 0:
        raise ValueError("sim_dt and control_dt must be positive")
    n_inner = max(1, int(round(control_dt / sim_dt)))
    if abs(n_inner * sim_dt - control_dt) > 0.1 * sim_dt:
        n_inner = max(1, int(np.ceil(control_dt / sim_dt)))
    return n_inner
