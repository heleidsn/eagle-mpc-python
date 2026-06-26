#!/usr/bin/env python3
"""Unpause Gazebo after wall-clock delay (works when /clock is silent while paused).

With ``use_sim_time`` and Gazebo ``paused:=true``, ``/clock`` often has **no
messages**.  Any ``rospy.init_node()`` then blocks forever waiting for sim time.

Strategy:

  1. **Phase 1 (no rospy):** wall-clock sleep, then unpause via ``gz world -p 0``
     (direct Gazebo API) with ``rosservice`` fallback.
  2. **Phase 2 (rospy):** after physics runs, ``/clock`` is published — init rospy
     and best-effort switch arm controllers + prime joint setpoints.

Launch passes ``UNPAUSE_DELAY_SEC`` etc. via ``<env>`` so phase 1 needs no rosparam.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time


def _log(msg: str) -> None:
    print(f"[unpause_arm] {msg}", flush=True)


def _wall_sleep(sec: float) -> None:
    if sec > 0.0:
        time.sleep(sec)


def _run(cmd: list[str], *, timeout: float = 10.0) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(timeout),
        )
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0, out.strip()
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout:.1f}s"
    except FileNotFoundError:
        return False, f"command not found: {cmd[0]}"
    except Exception as e:
        return False, str(e)


def _unpause_gazebo_no_ros() -> bool:
    """Unpause without rospy (safe when /clock is absent)."""
    ok, out = _run(["gz", "world", "-p", "0"], timeout=5.0)
    if ok:
        _log("gz world -p 0 OK (unpause)")
        return True
    _log(f"gz world -p 0 failed ({out}); trying rosservice ...")

    ok, out = _run(
        ["timeout", "5", "rosservice", "call", "/gazebo/unpause_physics", "{}"],
        timeout=8.0,
    )
    if ok:
        _log("rosservice /gazebo/unpause_physics OK")
        return True

    _log(f"rosservice unpause failed: {out}")
    return False


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def _phase1_wall_clock_unpause() -> bool:
    if not _env_bool("UNPAUSE_AUTO", True):
        _log("UNPAUSE_AUTO=false, exiting.")
        return False

    delay = max(0.5, _env_float("UNPAUSE_DELAY_SEC", 8.0))
    _log(
        f"Phase-1: wait {delay:.1f}s wall clock "
        "(no rospy — /clock may be empty while paused) ..."
    )
    _wall_sleep(delay)

    for attempt in range(1, 16):
        if _unpause_gazebo_no_ros():
            return True
        _log(f"unpause retry {attempt}/15 in 0.5s ...")
        _wall_sleep(0.5)

    _log("ERROR: could not unpause Gazebo")
    return False


def _wait_for_clock(timeout_wall: float = 15.0) -> bool:
    """Block until /clock appears (needs rospy + sim time running)."""
    import rospy
    from rosgraph_msgs.msg import Clock

    deadline = time.monotonic() + float(timeout_wall)
    while time.monotonic() < deadline:
        try:
            rospy.wait_for_message("/clock", Clock, timeout=0.5)
            return True
        except Exception:
            if rospy.is_shutdown():
                return False
    return False


def _phase2_rospy_arm_prep() -> None:
    import rospy
    from controller_manager_msgs.srv import ListControllers, SwitchController
    from std_msgs.msg import Float64
    from std_srvs.srv import Empty

    rospy.init_node("gazebo_unpause_after_arm_ready")

    ns = os.environ.get("UNPAUSE_CONTROLLER_NS", "arm_controller").strip().strip("/")
    post_hold = _env_float("UNPAUSE_POST_HOLD_SEC", 1.5)
    switch_timeout = _env_float("UNPAUSE_SWITCH_TIMEOUT_SEC", 4.0)
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4"]
    joints = [
        (j, _env_float(f"UNPAUSE_{j.upper()}", 0.0)) for j in joint_names
    ]
    pos_ctrl = [f"{j}_controller" for j in joint_names]

    if not _wait_for_clock(15.0):
        rospy.logwarn("[unpause_arm] /clock not seen after unpause; arm prep may be flaky.")

    def _list_states() -> dict[str, str]:
        svc = rospy.ServiceProxy(
            f"/{ns}/controller_manager/list_controllers", ListControllers
        )
        return {c.name: c.state for c in svc().controller}

    def _switch(names: list[str]) -> bool:
        try:
            states = _list_states()
        except Exception as e:
            rospy.logwarn(f"[unpause_arm] list_controllers: {e}")
            return False
        to_start = [n for n in names if states.get(n) != "running"]
        if not to_start:
            return True
        switch = rospy.ServiceProxy(
            f"/{ns}/controller_manager/switch_controller", SwitchController
        )
        result: dict = {"ok": False, "err": None}

        def _call():
            try:
                resp = switch(to_start, [], 1, False, 0.0)
                result["ok"] = bool(getattr(resp, "ok", 0))
            except Exception as e:
                result["err"] = str(e)

        th = threading.Thread(target=_call, daemon=True)
        th.start()
        th.join(timeout=max(0.5, switch_timeout))
        if th.is_alive():
            rospy.logwarn("[unpause_arm] switch_controller timed out")
            return False
        if result["err"]:
            rospy.logwarn(f"[unpause_arm] switch_controller: {result['err']}")
        elif result["ok"]:
            rospy.loginfo(f"[unpause_arm] Started: {', '.join(to_start)}")
        return bool(result["ok"])

    # Ensure unpause via ROS too (idempotent).
    try:
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=3.0)
        rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
    except Exception:
        pass

    _switch(pos_ctrl)

    if post_hold > 0.0:
        pubs = []
        for jname, pos in joints:
            topic = f"/{ns}/{jname}_controller/command"
            pubs.append((rospy.Publisher(topic, Float64, queue_size=1, latch=True), pos))
        _wall_sleep(0.2)
        rate = rospy.Rate(50)
        t_end = rospy.Time.now() + rospy.Duration(post_hold)
        while not rospy.is_shutdown() and rospy.Time.now() < t_end:
            for pub, pos in pubs:
                pub.publish(Float64(data=float(pos)))
            rate.sleep()
        for pub, pos in pubs:
            pub.publish(Float64(data=float(pos)))
        rospy.loginfo(f"[unpause_arm] Post-unpause joint hold ({post_hold:.1f}s) done.")

    rospy.loginfo("[unpause_arm] Done.")


def main() -> None:
    if not _phase1_wall_clock_unpause():
        sys.exit(1)
    # Let Gazebo start publishing /clock before rospy init.
    _wall_sleep(0.3)
    try:
        _phase2_rospy_arm_prep()
    except Exception as e:
        _log(f"Phase-2 rospy arm prep failed (sim may still be running): {e}")


if __name__ == "__main__":
    main()
