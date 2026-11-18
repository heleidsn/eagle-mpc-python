#!/usr/bin/env python3
"""PX4 SITL：关掉摇杆输入，使解锁 / OFFBOARD / 起飞不依赖 QGroundControl。

PX4 1.16 SITL 的 rcS 把 COM_RC_IN_MODE 设成 1（仅摇杆）。没有 QGC 虚拟摇杆时，
手动控制丢失会阻止在 POSCTL 解锁，并按 NAV_RCL_ACT 把 OFFBOARD 拉走。

COM_RC_IN_MODE=4 表示忽略摇杆。COM_RCL_EXCEPT 的 bit2 让 OFFBOARD 忽略 RC 丢失。
"""

from __future__ import annotations

# PX4 commander_params: COM_RC_IN_MODE value 4 = stick input disabled
_RC_IN_MODE_STICK_DISABLED = 4
# COM_RCL_EXCEPT bit 2 = ignore RC loss in OFFBOARD
_RCL_EXCEPT_OFFBOARD = 4


def ensure_px4_offboard_without_stick(timeout_s: float = 8.0):
    """把 PX4 设成不需要摇杆即可解锁并保持 OFFBOARD。

    返回 (ok, message)。参数服务尚未就绪时抛异常，调用方可以重试。
    """
    import rospy
    from mavros_msgs.msg import ParamValue
    from mavros_msgs.srv import ParamGet, ParamSet

    rospy.wait_for_service("/mavros/param/get", timeout=float(timeout_s))
    rospy.wait_for_service("/mavros/param/set", timeout=float(timeout_s))
    pget = rospy.ServiceProxy("/mavros/param/get", ParamGet)
    pset = rospy.ServiceProxy("/mavros/param/set", ParamSet)

    def _set_int(name: str, value: int) -> bool:
        resp = pset(param_id=name, value=ParamValue(integer=int(value), real=0.0))
        return bool(getattr(resp, "success", False))

    def _get_int(name: str):
        resp = pget(param_id=name)
        if not bool(getattr(resp, "success", False)):
            return None
        return int(resp.value.integer)

    ok_mode = _set_int("COM_RC_IN_MODE", _RC_IN_MODE_STICK_DISABLED)
    current = _get_int("COM_RCL_EXCEPT")
    except_val = _RCL_EXCEPT_OFFBOARD if current is None else (int(current) | _RCL_EXCEPT_OFFBOARD)
    ok_except = _set_int("COM_RCL_EXCEPT", except_val)
    msg = (
        f"COM_RC_IN_MODE={_RC_IN_MODE_STICK_DISABLED} {'OK' if ok_mode else 'FAIL'}, "
        f"COM_RCL_EXCEPT={except_val} {'OK' if ok_except else 'FAIL'}"
    )
    return bool(ok_mode and ok_except), msg


def main() -> None:
    import time

    import rospy
    from mavros_msgs.msg import State

    rospy.init_node("px4_sitl_disable_rc")
    if not bool(rospy.get_param("~enable", True)):
        rospy.loginfo("[px4_sitl_rc] disabled (~enable:=false)")
        return

    timeout_s = float(rospy.get_param("~timeout_s", 90.0))
    connected = {"v": False}

    def _state_cb(msg: State) -> None:
        connected["v"] = bool(msg.connected)

    rospy.Subscriber("/mavros/state", State, _state_cb, queue_size=1)
    deadline = time.time() + timeout_s
    rospy.loginfo("[px4_sitl_rc] waiting for MAVROS, then disabling stick input")
    while not rospy.is_shutdown() and time.time() < deadline:
        if connected["v"]:
            try:
                ok, msg = ensure_px4_offboard_without_stick(timeout_s=5.0)
                rospy.loginfo(f"[px4_sitl_rc] {msg}")
                if ok:
                    rospy.loginfo(
                        "[px4_sitl_rc] QGC joystick is not required for arm / OFFBOARD / takeoff"
                    )
                    return
            except Exception as exc:
                rospy.logwarn_throttle(5.0, f"[px4_sitl_rc] param set not ready: {exc}")
        rospy.sleep(1.0)
    rospy.logerr("[px4_sitl_rc] timed out; arming may still require a QGC joystick")


if __name__ == "__main__":
    main()
