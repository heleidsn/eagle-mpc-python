#!/usr/bin/env python3
"""Publish RViz disturbance markers from /suite_mpc/disturbance_config (rosparam)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from gazebo_disturbance_helper import (  # noqa: E402
    DISTURBANCE_CMD_TOPIC,
    DISTURBANCE_CONFIG_PARAM,
    DISTURBANCE_MARKER_TOPIC,
    delete_disturbance_markers,
    markers_for_disturbance_cfg,
)

_INACTIVE = {"active": False}


def _cfg_from_param() -> dict | None:
    if not rospy.has_param(DISTURBANCE_CONFIG_PARAM):
        return None
    raw = rospy.get_param(DISTURBANCE_CONFIG_PARAM)
    if not isinstance(raw, dict) or not raw.get("active", False):
        return None
    return {
        "model": str(raw.get("model", "s500_uam")),
        "link": str(raw.get("link", "base_link")),
        "frame": str(raw.get("frame", "world")),
        "force": list(raw.get("force", [0.0, 0.0, 0.0])),
        "torque": list(raw.get("torque", [0.0, 0.0, 0.0])),
    }


class GazeboDisturbanceVizNode:
    def __init__(self) -> None:
        self._active_cfg: dict | None = None
        self._was_active = False
        self._warned_pose = False
        self._hz = float(rospy.get_param("~publish_hz", 10.0))
        # ADD marker 寿命略大于发布周期；停止发布后自动消失
        self._marker_lifetime = max(0.25, 1.5 / max(self._hz, 1.0))

        self._marker_pub = rospy.Publisher(
            DISTURBANCE_MARKER_TOPIC, MarkerArray, queue_size=2, latch=False
        )
        rospy.Subscriber(DISTURBANCE_CMD_TOPIC, String, self._cmd_cb, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / max(self._hz, 1.0)), self._on_timer)

        # 清除上次会话残留的 active:true，避免一启动就有箭头
        try:
            rospy.set_param(DISTURBANCE_CONFIG_PARAM, dict(_INACTIVE))
        except Exception:
            pass
        self._publish_clear()
        rospy.loginfo(
            f"[dist_viz] {DISTURBANCE_CONFIG_PARAM} -> {DISTURBANCE_MARKER_TOPIC} "
            f"@ {self._hz:g}Hz, lifetime={self._marker_lifetime:.2f}s"
        )

    def _publish_clear(self) -> None:
        self._marker_pub.publish(delete_disturbance_markers())
        self._active_cfg = None
        self._was_active = False
        self._warned_pose = False

    def _cmd_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data or "{}")
        except json.JSONDecodeError as e:
            rospy.logwarn(f"[dist_viz] bad JSON: {e}")
            return
        try:
            rospy.set_param(DISTURBANCE_CONFIG_PARAM, data)
        except Exception as e:
            rospy.logwarn(f"[dist_viz] set_param failed: {e}")
        self._apply_param_state()

    def _apply_param_state(self) -> None:
        cfg = _cfg_from_param()
        if cfg is None:
            if self._was_active:
                self._publish_clear()
                rospy.loginfo("[dist_viz] disturbance cleared")
            return
        self._was_active = True
        self._active_cfg = cfg
        self._warned_pose = False
        self._publish_active(cfg)

    def _publish_active(self, cfg: dict) -> None:
        ok, msg, arr = markers_for_disturbance_cfg(
            cfg, marker_lifetime=self._marker_lifetime
        )
        if ok and arr.markers:
            self._marker_pub.publish(arr)
            self._warned_pose = False
        elif not ok and not self._warned_pose:
            rospy.logwarn(f"[dist_viz] markers failed: {msg}")
            self._warned_pose = True

    def _on_timer(self, _evt) -> None:
        cfg = _cfg_from_param()
        if cfg is None:
            if self._was_active:
                self._publish_clear()
                rospy.loginfo("[dist_viz] disturbance cleared")
            return
        if self._active_cfg != cfg:
            self._active_cfg = cfg
            self._warned_pose = False
            rospy.loginfo(f"[dist_viz] active: {cfg}")
        self._was_active = True
        self._publish_active(cfg)


def main() -> None:
    rospy.init_node("gazebo_disturbance_viz")
    GazeboDisturbanceVizNode()
    rospy.spin()


if __name__ == "__main__":
    main()
