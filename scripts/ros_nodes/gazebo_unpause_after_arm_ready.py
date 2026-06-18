#!/usr/bin/env python3
"""Unpause Gazebo after s500_uam arm controllers are ready.

When ``paused:=true``, spawn is safe but controllers may stay ``loaded`` (not
``running``) until physics runs — ``controller_manager/spawner`` can block on
``switch_controller`` while the world is paused.

This node:

  1. Waits until joint controllers appear in controller_manager (loaded).
  2. Tries to start them (switch_controller), with a short timeout while paused.
  3. Latches initial joint position commands.
  4. Calls /gazebo/unpause_physics.
  5. Re-primes joint commands briefly after unpause so position loops engage.
"""

from __future__ import annotations

import threading

import rospy
from controller_manager_msgs.srv import ListControllers, SwitchController
from std_msgs.msg import Float64
from std_srvs.srv import Empty

# controller_manager_msgs/SwitchController: STRICT=2, BEST_EFFORT=1
_SWITCH_BEST_EFFORT = 1


def _list_controller_states(ns: str) -> dict[str, str]:
    svc = rospy.ServiceProxy(f"/{ns}/controller_manager/list_controllers", ListControllers)
    resp = svc()
    return {c.name: c.state for c in resp.controller}


def _wait_for_controllers_loaded(ns: str, names: list[str], timeout: float) -> bool:
    svc_name = f"/{ns}/controller_manager/list_controllers"
    rospy.wait_for_service(svc_name, timeout=timeout)
    want = set(names)
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            states = _list_controller_states(ns)
            if want.issubset(states.keys()):
                rospy.loginfo(
                    f"[unpause_arm] Controllers loaded: "
                    + ", ".join(f"{k}={states[k]}" for k in sorted(want))
                )
                return True
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[unpause_arm] list_controllers: {e}")
        rate.sleep()
    return False


def _switch_controllers(ns: str, names: list[str], timeout: float) -> bool:
    """Start controllers; return False if switch call does not finish in time."""
    try:
        states = _list_controller_states(ns)
    except Exception as e:
        rospy.logwarn(f"[unpause_arm] list before switch: {e}")
        states = {}
    to_start = [n for n in names if states.get(n) != "running"]
    if not to_start:
        return True

    switch = rospy.ServiceProxy(
        f"/{ns}/controller_manager/switch_controller", SwitchController
    )
    rospy.wait_for_service(f"/{ns}/controller_manager/switch_controller", timeout=5.0)

    result = {"ok": False, "err": None}

    def _call():
        try:
            resp = switch(to_start, [], _SWITCH_BEST_EFFORT, False, 0.0)
            result["ok"] = bool(getattr(resp, "ok", 0))
        except Exception as e:
            result["err"] = str(e)

    th = threading.Thread(target=_call, daemon=True)
    th.start()
    th.join(timeout=max(0.5, float(timeout)))
    if th.is_alive():
        rospy.logwarn(
            f"[unpause_arm] switch_controller timed out after {timeout:.1f}s "
            f"(common while paused); will unpause and retry."
        )
        return False
    if result["err"]:
        rospy.logwarn(f"[unpause_arm] switch_controller error: {result['err']}")
    elif result["ok"]:
        rospy.loginfo(f"[unpause_arm] Started controllers: {', '.join(to_start)}")
    else:
        rospy.logwarn(f"[unpause_arm] switch_controller returned not-ok for {to_start}")
    return bool(result["ok"])


def _wait_for_controllers_running(ns: str, names: list[str], timeout: float) -> bool:
    want = set(names)
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            states = _list_controller_states(ns)
            running = {n for n, st in states.items() if st == "running"}
            if want.issubset(running):
                return True
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[unpause_arm] list_controllers: {e}")
        rate.sleep()
    return False


def _prime_joint_commands(ns: str, joints: list[tuple[str, float]], hold_sec: float) -> None:
    pubs = []
    for jname, pos in joints:
        topic = f"/{ns}/{jname}_controller/command"
        pub = rospy.Publisher(topic, Float64, queue_size=1, latch=True)
        pubs.append((pub, float(pos)))
    rospy.sleep(0.2)
    if hold_sec <= 0.0:
        for pub, pos in pubs:
            pub.publish(Float64(data=pos))
        return
    rate = rospy.Rate(50)
    t_end = rospy.Time.now() + rospy.Duration(hold_sec)
    while not rospy.is_shutdown() and rospy.Time.now() < t_end:
        for pub, pos in pubs:
            pub.publish(Float64(data=pos))
        rate.sleep()
    for pub, pos in pubs:
        pub.publish(Float64(data=pos))


def _unpause_physics() -> bool:
    rospy.wait_for_service("/gazebo/unpause_physics", timeout=15.0)
    rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
    return True


def main() -> None:
    rospy.init_node("gazebo_unpause_after_arm_ready")

    if not bool(rospy.get_param("~auto_unpause", True)):
        rospy.loginfo("[unpause_arm] auto_unpause=false, exiting.")
        return

    ns = str(rospy.get_param("~controller_ns", "arm_controller")).strip().strip("/")
    hold_sec = float(rospy.get_param("~hold_sec", 1.0))
    post_hold_sec = float(rospy.get_param("~post_unpause_hold_sec", 1.0))
    wait_sec = float(rospy.get_param("~controller_wait_sec", 90.0))
    switch_timeout = float(rospy.get_param("~switch_timeout_sec", 4.0))
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4"]
    joints = [
        (jname, float(rospy.get_param(f"~{jname}", 0.0))) for jname in joint_names
    ]
    pos_ctrl_names = [f"{j}_controller" for j in joint_names]
    all_ctrl_names = pos_ctrl_names + ["joint_state_controller"]

    rospy.loginfo(
        f"[unpause_arm] Waiting for arm controllers in /{ns} "
        f"(joint targets: {[p for _, p in joints]})..."
    )
    if not _wait_for_controllers_loaded(ns, all_ctrl_names, wait_sec):
        rospy.logerr("[unpause_arm] Timed out waiting for controllers to load; not unpausing.")
        return

    # While paused: try switch (may time out) and latch joint setpoints.
    _switch_controllers(ns, pos_ctrl_names, switch_timeout)
    rospy.loginfo(f"[unpause_arm] Priming joint commands ({hold_sec:.1f}s, paused)...")
    _prime_joint_commands(ns, joints, hold_sec)

    try:
        _unpause_physics()
        rospy.loginfo("[unpause_arm] /gazebo/unpause_physics OK — simulation running.")
    except Exception as e:
        rospy.logerr(f"[unpause_arm] unpause failed: {e}")
        return

    # After unpause: switch again if needed, wait for running, then hold joints.
    _switch_controllers(ns, pos_ctrl_names, switch_timeout)
    if not _wait_for_controllers_running(ns, pos_ctrl_names, timeout=8.0):
        rospy.logwarn(
            "[unpause_arm] Position controllers not all 'running' after unpause; "
            "continuing with joint command priming."
        )
    if post_hold_sec > 0.0:
        rospy.loginfo(f"[unpause_arm] Post-unpause joint hold ({post_hold_sec:.1f}s)...")
        _prime_joint_commands(ns, joints, post_hold_sec)
    rospy.loginfo("[unpause_arm] Done.")


if __name__ == "__main__":
    main()
