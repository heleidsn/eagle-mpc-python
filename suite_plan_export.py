"""
suite_plan_export.py — 将 GUI plan_bundle 写成跨脚本可读的 npz 文件。

唯一依赖：numpy + pathlib（无 ROS、无 eagle_mpc、无 utils）。
可在 GUI 进程和 ROS 节点进程中安全导入。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def export_suite_plan_npz(
    path: str | os.PathLike,
    plan_bundle: dict[str, Any],
    *,
    dt_plan_fallback_s: float | None = None,
) -> Path:
    """
    将 GUI 的 full-state plan_bundle 序列化为 .npz 文件。

    plan_bundle 必须包含：kind, t_plan, x_plan；可选 u_plan。
    返回写入的绝对路径。
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    pb = plan_bundle
    kind = str(pb.get("kind", ""))
    t_plan = np.asarray(pb["t_plan"], dtype=float).reshape(-1)
    x_plan = np.asarray(pb["x_plan"], dtype=float)

    u_raw = pb.get("u_plan")
    if u_raw is None:
        u_plan = np.zeros((0, 0), dtype=float)
    else:
        u_plan = np.asarray(u_raw, dtype=float)
        if u_plan.ndim != 2:
            u_plan = np.zeros((0, 0), dtype=float)

    if t_plan.size >= 2:
        dt_ms = int(max(1, round(float(np.median(np.diff(t_plan))) * 1000.0)))
    elif dt_plan_fallback_s is not None:
        dt_ms = int(max(1, round(float(dt_plan_fallback_s) * 1000.0)))
    else:
        dt_ms = 50

    np.savez(
        out,
        kind=np.array(kind, dtype=object),
        t_plan=t_plan,
        x_plan=x_plan,
        u_plan=u_plan,
        dt_traj_opt_ms=np.array(dt_ms, dtype=np.int32),
    )
    return out
