#!/usr/bin/env python3
"""PlotJuggler-friendly MPC debug topics under ``/suite_mpc/mpc_debug/``.

Acados and Crocoddyl backends publish the same *unified* cost channels
(``cost/total``, ``cost/state_track``, ``cost/state_reg``, ``cost/control``,
``cost/terminal``) plus backend-specific *detail* scalars under
``cost/detail/<name>``.

Solver diagnostics (units in topic names):
``solver/wall_solve_time_ms``, ``solver/solving_time_s`` (same wall time),
``solver/cpu_solve_time_ms`` (acados internal CPU only),
``solver/sqp_iter``, ``solver/qp_iter``, ``solver/status``.

IO: ``io/u``, ``io/x_next``, ``io/x_now`` as ``Float64MultiArray``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import rospy
    from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, String

    _ROS_OK = True
except ImportError:
    _ROS_OK = False

NS = "/suite_mpc/mpc_debug"

UNIFIED_COST_KEYS = ("total", "state_track", "state_reg", "control", "terminal")

ACADOS_DETAIL_KEYS = (
    "pos",
    "att",
    "joint",
    "vel",
    "omega",
    "joint_vel",
    "u_thrust",
    "u_torque",
    "terminal",
)


def _sanitize_key(s: str) -> str:
    return str(s).replace("/", "_").replace("-", "_")


def pack_acados_cost(
    terms: Dict[str, float], total: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Map acados NONLINEAR_LS breakdown to unified + detail dicts."""
    terms = {k: float(v) for k, v in (terms or {}).items()}
    state_track = sum(
        terms.get(k, 0.0)
        for k in ("pos", "att", "joint", "vel", "omega", "joint_vel")
    )
    control = float(terms.get("u_thrust", 0.0)) + float(terms.get("u_torque", 0.0))
    terminal = float(terms.get("terminal", 0.0))
    cost_total = float(total) if np.isfinite(total) else float(sum(terms.values()))
    unified = {
        "total": cost_total,
        "state_track": state_track,
        "state_reg": 0.0,
        "control": control,
        "terminal": terminal,
    }
    detail = {k: float(terms.get(k, 0.0)) for k in ACADOS_DETAIL_KEYS}
    return unified, detail


def pack_croc_cost(
    terms: Dict[str, float],
    groups: Dict[str, float],
    total: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Map Crocoddyl ``_extract_solver_cost_terms`` output to unified + detail."""
    terms = {k: float(v) for k, v in (terms or {}).items()}
    groups = {k: float(v) for k, v in (groups or {}).items()}

    def _sum_name(substr: str) -> float:
        return float(sum(v for k, v in terms.items() if substr in k))

    state_track = _sum_name("x_track")
    state_reg = _sum_name("x_reg")
    control = _sum_name("u_reg") + float(
        sum(v for k, v in groups.items() if "action" in k)
    )
    terminal = float(sum(v for k, v in terms.items() if k.startswith("terminal/")))
    cost_total = float(total) if np.isfinite(total) else float(sum(terms.values()))
    unified = {
        "total": cost_total,
        "state_track": state_track,
        "state_reg": state_reg,
        "control": control,
        "terminal": terminal,
    }
    detail: Dict[str, float] = {}
    for k, v in terms.items():
        detail[f"croc_{_sanitize_key(k)}"] = float(v)
    for k, v in groups.items():
        detail[f"group_{_sanitize_key(k)}"] = float(v)
    return unified, detail


class MpcRosDebugPublisher:
    """Lazy ROS publishers for MPC debug scalars and IO vectors."""

    def __init__(self, *, min_interval_s: float = 0.0) -> None:
        if not _ROS_OK:
            raise ImportError("rospy not available")
        self._min_interval_s = float(min_interval_s)
        self._last_pub_t = 0.0
        self._f64_pubs: Dict[str, rospy.Publisher] = {}
        self._str_pubs: Dict[str, rospy.Publisher] = {}
        self._arr_pubs: Dict[str, rospy.Publisher] = {}
        self._detail_keys: set = set()

    def _f64_pub(self, rel: str) -> rospy.Publisher:
        key = f"{NS}/{rel}"
        if key not in self._f64_pubs:
            self._f64_pubs[key] = rospy.Publisher(key, Float64, queue_size=10)
        return self._f64_pubs[key]

    def _str_pub(self, rel: str) -> rospy.Publisher:
        key = f"{NS}/{rel}"
        if key not in self._str_pubs:
            self._str_pubs[key] = rospy.Publisher(key, String, queue_size=5)
        return self._str_pubs[key]

    def _arr_pub(self, rel: str) -> rospy.Publisher:
        key = f"{NS}/{rel}"
        if key not in self._arr_pubs:
            self._arr_pubs[key] = rospy.Publisher(
                key, Float64MultiArray, queue_size=10
            )
        return self._arr_pubs[key]

    @staticmethod
    def _publish_f64(pub: rospy.Publisher, value: float) -> None:
        msg = Float64()
        msg.data = float(value) if np.isfinite(value) else float("nan")
        pub.publish(msg)

    @staticmethod
    def _publish_array(
        pub: rospy.Publisher, data, labels: Optional[list] = None
    ) -> None:
        arr = np.asarray(data, dtype=float).flatten()
        msg = Float64MultiArray()
        msg.data = [float(x) for x in arr]
        if labels:
            msg.layout.dim = [
                MultiArrayDimension(label=str(lbl), size=1, stride=1)
                for lbl in labels[: arr.size]
            ]
        pub.publish(msg)

    @staticmethod
    def _u_labels(nu: int, n_arm: int) -> list:
        labels = [f"thrust_{i}" for i in range(min(4, nu))]
        if nu > 4:
            labels += [f"tau_{i}" for i in range(nu - 4)]
        elif n_arm > 0 and nu == 4 + n_arm:
            labels += [f"tau_{i}" for i in range(n_arm)]
        return labels

    def publish(
        self,
        *,
        backend: str,
        phase: str,
        solve_time_ms: float,
        cpu_time_ms: float,
        sqp_iter: int,
        qp_iter: int,
        status: int,
        cost_unified: Dict[str, float],
        cost_detail: Dict[str, float],
        u_cmd,
        x_next,
        x_now,
        nu: int = 4,
        n_arm: int = 0,
    ) -> None:
        now = rospy.Time.now().to_sec() if rospy.core is not None else 0.0
        if self._min_interval_s > 0.0 and (now - self._last_pub_t) < self._min_interval_s:
            return
        self._last_pub_t = now

        self._publish_f64(
            self._f64_pub("solver/wall_solve_time_ms"), solve_time_ms
        )
        self._publish_f64(
            self._f64_pub("solver/solving_time_s"),
            float(solve_time_ms) / 1000.0 if np.isfinite(solve_time_ms) else float("nan"),
        )
        self._publish_f64(self._f64_pub("solver/cpu_solve_time_ms"), cpu_time_ms)
        self._publish_f64(self._f64_pub("solver/sqp_iter"), float(sqp_iter))
        self._publish_f64(self._f64_pub("solver/qp_iter"), float(qp_iter))
        self._publish_f64(self._f64_pub("solver/status"), float(status))

        for k in UNIFIED_COST_KEYS:
            if k == "total":
                rel = "cost/total"
            else:
                rel = f"cost/{k}"
            self._publish_f64(
                self._f64_pub(rel), float(cost_unified.get(k, float("nan")))
            )

        for name, val in (cost_detail or {}).items():
            safe = _sanitize_key(name)
            rel = f"cost/detail/{safe}"
            if safe not in self._detail_keys:
                self._detail_keys.add(safe)
            self._publish_f64(self._f64_pub(rel), float(val))

        phase_code = 1.0 if str(phase).lower().startswith("reg") else 0.0
        self._publish_f64(self._f64_pub("meta/phase_code"), phase_code)
        self._str_pub("meta/backend").publish(String(data=str(backend)))
        self._str_pub("meta/phase").publish(String(data=str(phase)))

        u_labels = self._u_labels(int(nu), int(n_arm))
        if u_cmd is not None:
            self._publish_array(self._arr_pub("io/u"), u_cmd, u_labels)
        if x_next is not None:
            self._publish_array(self._arr_pub("io/x_next"), x_next)
        if x_now is not None:
            self._publish_array(self._arr_pub("io/x_now"), x_now)
