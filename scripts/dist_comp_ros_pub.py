#!/usr/bin/env python3
"""ROS topics for disturbance estimation & compensation state.

Published under ``/suite_mpc/dist_comp/`` for PlotJuggler / GUI / logging:

JSON (full snapshot):
  ``state`` — all fields as one JSON string

Estimation (world frame):
  ``estimation/force_world``     Float64MultiArray [Fx, Fy, Fz]  (N)
  ``estimation/wrench_world``    Float64MultiArray [Fx..Mz]     (N, N·m)
  ``estimation/accel_world``     Float64MultiArray [ax, ay, az] (m/s²)
  ``estimation/force_norm``      Float64

Compensation:
  ``compensation/accel_bolt_on``   Float64MultiArray [ax, ay, az] applied via CTBR
  ``compensation/wrench_in_model``  Float64MultiArray [Fx..Mz] fed into acados OCP
  ``compensation/accel_norm``       Float64

Flags (0/1):
  ``flags/dist_enabled``, ``flags/comp_enabled``,
  ``flags/bolt_on_active``, ``flags/in_model_active``

Meta:
  ``mode``   String — adaptive | oracle | drag_ff
  ``inject`` String — bolt_on | in_model
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

try:
    import rospy
    from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, String

    _ROS_OK = True
except ImportError:
    _ROS_OK = False

NS = "/suite_mpc/dist_comp"


def _fvec(x, n: int) -> list:
    arr = np.asarray(x, dtype=float).reshape(-1)
    out = [0.0] * n
    for i in range(min(n, arr.size)):
        out[i] = float(arr[i])
    return out


class DistCompRosPublisher:
    """Lazy ROS publishers for disturbance estimation / compensation debug."""

    def __init__(self, *, min_interval_s: float = 0.02) -> None:
        if not _ROS_OK:
            raise ImportError("rospy not available")
        self._min_interval_s = float(min_interval_s)
        self._last_pub_t = 0.0
        self._f64: Dict[str, rospy.Publisher] = {}
        self._str: Dict[str, rospy.Publisher] = {}
        self._arr: Dict[str, rospy.Publisher] = {}

    def _pub_f64(self, rel: str) -> rospy.Publisher:
        key = f"{NS}/{rel}"
        if key not in self._f64:
            self._f64[key] = rospy.Publisher(key, Float64, queue_size=10)
        return self._f64[key]

    def _pub_str(self, rel: str) -> rospy.Publisher:
        key = f"{NS}/{rel}"
        if key not in self._str:
            self._str[key] = rospy.Publisher(key, String, queue_size=5)
        return self._str[key]

    def _pub_arr(self, rel: str) -> rospy.Publisher:
        key = f"{NS}/{rel}"
        if key not in self._arr:
            self._arr[key] = rospy.Publisher(key, Float64MultiArray, queue_size=10)
        return self._arr[key]

    @staticmethod
    def _send_f64(pub: rospy.Publisher, value: float) -> None:
        msg = Float64()
        v = float(value)
        msg.data = v if np.isfinite(v) else float("nan")
        pub.publish(msg)

    @staticmethod
    def _send_arr(pub: rospy.Publisher, data, labels: Optional[list] = None) -> None:
        arr = np.asarray(data, dtype=float).flatten()
        msg = Float64MultiArray()
        msg.data = [float(x) if np.isfinite(x) else float("nan") for x in arr]
        if labels:
            msg.layout.dim = [
                MultiArrayDimension(label=str(lbl), size=1, stride=1)
                for lbl in labels[: arr.size]
            ]
        pub.publish(msg)

    def publish(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return
        now = rospy.Time.now().to_sec()
        if self._min_interval_s > 0.0 and (now - self._last_pub_t) < self._min_interval_s:
            return
        self._last_pub_t = now

        est = snapshot.get("estimation") or {}
        comp = snapshot.get("compensation") or {}
        flags = snapshot.get("flags") or {}

        self._pub_str("state").publish(String(data=json.dumps(snapshot)))
        self._pub_str("mode").publish(String(data=str(snapshot.get("l1_mode", ""))))
        self._pub_str("estimation/mode").publish(
            String(data=str(snapshot.get("l1_mode", "")))
        )
        self._pub_str("inject").publish(String(data=str(snapshot.get("l1_inject", ""))))
        self._pub_str("estimation/source").publish(
            String(data=str(est.get("source", "none")))
        )

        f3 = _fvec(est.get("force_world", [0, 0, 0]), 3)
        w6 = _fvec(est.get("wrench_world", [0] * 6), 6)
        a3 = _fvec(est.get("accel_world", [0, 0, 0]), 3)
        ac3 = _fvec(comp.get("accel_bolt_on", [0, 0, 0]), 3)
        wm6 = _fvec(comp.get("wrench_in_model", [0] * 6), 6)

        self._send_arr(
            self._pub_arr("estimation/force_world"), f3, ["Fx", "Fy", "Fz"]
        )
        self._send_arr(
            self._pub_arr("estimation/wrench_world"), w6,
            ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        )
        self._send_arr(
            self._pub_arr("estimation/accel_world"), a3, ["ax", "ay", "az"]
        )
        self._send_arr(
            self._pub_arr("compensation/accel_bolt_on"), ac3, ["ax", "ay", "az"]
        )
        self._send_arr(
            self._pub_arr("compensation/wrench_in_model"), wm6,
            ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        )

        self._send_f64(self._pub_f64("estimation/force_norm"), float(est.get("force_norm", 0.0)))
        self._send_f64(
            self._pub_f64("compensation/accel_norm"), float(comp.get("accel_norm", 0.0))
        )

        for key in ("dist_enabled", "comp_enabled", "bolt_on_active", "in_model_active"):
            self._send_f64(
                self._pub_f64(f"flags/{key}"),
                1.0 if bool(flags.get(key, False)) else 0.0,
            )
