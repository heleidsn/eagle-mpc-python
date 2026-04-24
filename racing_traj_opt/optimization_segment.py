import argparse
import csv
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from segment_planner import SegmentPlanner


def load_track(track_yaml: Path) -> dict:
    with track_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    initial = cfg.get("initial", {})
    end = cfg.get("end", {})
    return {
        "gates": cfg.get("gates", []),
        "init_pos": initial.get("position", [0.0, 0.0, 1.0]),
        "init_vel": initial.get("velocity", [0.0, 0.0, 0.0]),
        "end_pos": end.get("position"),
        "end_vel": end.get("velocity", [0.0, 0.0, 0.0]),
    }


def _load_quad_from_urdf_with_pinocchio(urdf_path: Path):
    try:
        import pinocchio as pin
    except Exception:
        return None
    if not hasattr(pin, "buildModelFromUrdf"):
        return None
    model = pin.buildModelFromUrdf(str(urdf_path))
    mass = float(np.sum([i.mass for i in model.inertias]))
    inertia = model.inertias[1].inertia if model.njoints > 1 else np.eye(3) * 0.01
    return mass, np.array(inertia)


def _load_quad_from_urdf_xml(urdf_path: Path):
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    base_link = root.find("./link[@name='base_link']")
    if base_link is None:
        raise ValueError("URDF does not contain link 'base_link'")
    inertial = base_link.find("inertial")
    if inertial is None:
        raise ValueError("URDF base_link has no inertial tag")
    mass = float(inertial.find("mass").attrib["value"])
    inertia_node = inertial.find("inertia")
    ixx = float(inertia_node.attrib.get("ixx", 0.01))
    iyy = float(inertia_node.attrib.get("iyy", 0.01))
    izz = float(inertia_node.attrib.get("izz", 0.02))
    return mass, np.diag([ixx, iyy, izz])


def load_quad_from_urdf(urdf_path: Path) -> dict:
    urdf_data = _load_quad_from_urdf_with_pinocchio(urdf_path)
    if urdf_data is None:
        urdf_data = _load_quad_from_urdf_xml(urdf_path)
    mass, inertia = urdf_data
    g = 9.81
    # conservative bounds derived from mass (can be tuned later)
    a_max_xy = 12.0
    a_max_z = 16.0
    return {
        "mass": mass,
        "inertia": inertia,
        "g": g,
        "a_max_xy": a_max_xy,
        "a_max_z": a_max_z,
    }


def save_result_csv(path: Path, t: np.ndarray, p: np.ndarray, v: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "p_x", "p_y", "p_z", "v_x", "v_y", "v_z"])
        for i in range(len(t)):
            w.writerow([float(t[i]), float(p[i, 0]), float(p[i, 1]), float(p[i, 2]), float(v[i, 0]), float(v[i, 1]), float(v[i, 2])])


def main():
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    parser = argparse.ArgumentParser(description="Segment-wise time-optimal quad trajectory")
    parser.add_argument("--track", type=str, default=str(this_dir / "track.yaml"))
    parser.add_argument("--urdf", type=str, default=str(repo_root / "models" / "urdf" / "s500_simple.urdf"))
    parser.add_argument("--nodes-per-segment", type=int, default=20)
    parser.add_argument("--vel-guess", type=float, default=3.0)
    args = parser.parse_args()

    track = load_track(Path(args.track))
    quad = load_quad_from_urdf(Path(args.urdf))

    planner = SegmentPlanner(
        quad=quad,
        track=track,
        options={
            "nodes_per_segment": args.nodes_per_segment,
            "vel_guess": args.vel_guess,
            "accel_penalty": 0.01,
            "solver_options": {
                "ipopt": {
                    "max_iter": 2000,
                    "tol": 1e-6,
                    "acceptable_tol": 1e-4,
                    "acceptable_iter": 20,
                    "hessian_approximation": "limited-memory",
                }
            },
        },
    )
    planner.setup()
    planner.solve()
    t, p, v = planner.extract_position_velocity_trajectory()

    results_dir = repo_root / "trajectory"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = results_dir / f"result_segment_{ts}.csv"
    latest = results_dir / "result_segment_latest.csv"
    save_result_csv(timestamped, t, p, v)
    save_result_csv(latest, t, p, v)

    print(f"Loaded URDF: {args.urdf}")
    print(f"Mass from URDF: {quad['mass']:.3f} kg")
    print(f"Saved segment result: {timestamped}")
    print(f"Updated latest file: {latest}")

    print("Optimization finished. Use `python3 racing_traj_opt/plot_segment_result.py` to plot results.")


if __name__ == "__main__":
    main()