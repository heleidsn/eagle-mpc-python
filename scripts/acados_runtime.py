"""Shared acados runtime setup for this repo.

After rebuilding acados, ``libacados.so`` is loaded via ctypes without a proper
RPATH in the GUI process. Preload dependent .so files from ``ACADOS_SOURCE_DIR/lib``
*before* ``acados_template`` loads ``libacados.so`` (fixes undefined symbol
``LINSYS_SOLVER_NAME`` from libosqp, and similar qpOASES/hpipm issues).
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_acados_source_dir() -> str | None:
    """Set ``ACADOS_SOURCE_DIR`` when unset; return resolved acados root or None."""
    env = os.environ.get("ACADOS_SOURCE_DIR")
    if env and os.path.isdir(os.path.join(env, "lib")):
        return env
    here = Path(__file__).resolve().parent
    for root in (
        here.parent.parent / "acados",          # .../src/acados
        here.parent.parent.parent / "acados",   # .../catkin_eagle_mpc/acados
    ):
        if (root / "lib").is_dir() and (root / "interfaces").is_dir():
            os.environ["ACADOS_SOURCE_DIR"] = str(root)
            return str(root)
    return None


def preload_acados_shared_libs() -> None:
    """Load acados dependency chain via absolute paths (Linux only)."""
    if os.name == "nt":
        return
    try:
        from ctypes import CDLL
    except ImportError:
        return
    root = ensure_acados_source_dir()
    if not root:
        return
    libdir = os.path.join(root, "lib")
    if not os.path.isdir(libdir):
        return
    # Order: blasfeo <- hpipm; qp solvers; then libacados consumers resolve symbols.
    for name in (
        "libblasfeo.so.0",
        "libqdldl.so",
        "libosqp.so",
        "libdaqp.so",
        "libqpOASES_e.so",
        "libhpipm.so",
    ):
        path = os.path.join(libdir, name)
        if os.path.isfile(path):
            CDLL(path)
