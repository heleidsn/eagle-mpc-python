import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
import joblib

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
        "init_att": initial.get("attitude", [1.0, 0.0, 0.0, 0.0]),
        "init_omega": initial.get("omega", [0.0, 0.0, 0.0]),
        "end_pos": end.get("position"),
        "end_vel": end.get("velocity", [0.0, 0.0, 0.0]),
        "end_att": end.get("attitude", None),
        "end_omega": end.get("omega", None),
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
        "T_min": 0.0,
        "T_max": 12.0,
        "a_max_xy": a_max_xy,
        "a_max_z": a_max_z,
        "k_yaw": 0.01,
        "rotor_pos": [
            [0.171, 0.171, 0.0],
            [-0.171, 0.171, 0.0],
            [-0.171, -0.171, 0.0],
            [0.171, -0.171, 0.0],
        ],
        "rotor_yaw_dirs": [1.0, -1.0, 1.0, -1.0],
    }


def load_quad_from_yaml(quad_yaml: Path) -> dict:
    with quad_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    mass = float(cfg.get("mass", 0.85))
    inertia = np.asarray(cfg.get("inertia", np.diag([0.001, 0.001, 0.0017])), dtype=float)
    arm_length = float(cfg.get("arm_length", 0.15))
    thrust_min = float(cfg.get("thrust_min", 0.0))
    twr_max = float(cfg.get("TWR_max", 3.3))
    torque_coeff = float(cfg.get("torque_coeff", 0.01))

    g = 9.81
    # Per-rotor max thrust inferred from total thrust-to-weight ratio.
    thrust_max = (mass * g * twr_max) / 4.0
    # X-configuration rotor placement from center-to-motor arm length.
    xy = arm_length / np.sqrt(2.0)

    return {
        "mass": mass,
        "inertia": inertia,
        "g": g,
        "T_min": thrust_min,
        "T_max": thrust_max,
        "a_max_xy": 12.0,
        "a_max_z": 16.0,
        "k_yaw": torque_coeff,
        "rotor_pos": [
            [xy, xy, 0.0],
            [-xy, xy, 0.0],
            [-xy, -xy, 0.0],
            [xy, -xy, 0.0],
        ],
        "rotor_yaw_dirs": [1.0, -1.0, 1.0, -1.0],
    }


def save_result_csv(path: Path, t: np.ndarray, p: np.ndarray, v: np.ndarray, q: np.ndarray, w: np.ndarray, u: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "p_x",
                "p_y",
                "p_z",
                "v_x",
                "v_y",
                "v_z",
                "q_w",
                "q_x",
                "q_y",
                "q_z",
                "omega_x",
                "omega_y",
                "omega_z",
                "u_1",
                "u_2",
                "u_3",
                "u_4",
            ]
        )
        for i in range(len(t)):
            writer.writerow(
                [
                    float(t[i]),
                    float(p[i, 0]),
                    float(p[i, 1]),
                    float(p[i, 2]),
                    float(v[i, 0]),
                    float(v[i, 1]),
                    float(v[i, 2]),
                    float(q[i, 0]),
                    float(q[i, 1]),
                    float(q[i, 2]),
                    float(q[i, 3]),
                    float(w[i, 0]),
                    float(w[i, 1]),
                    float(w[i, 2]),
                    float(u[i, 0]),
                    float(u[i, 1]),
                    float(u[i, 2]),
                    float(u[i, 3]),
                ]
            )


def main():
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    planner_fixed = {
        "dynamics_model": "full",  # set "simple" to use simplified point-mass model
        "vel_guess": 3.0,
        "omega_max_deg": 150.0,      # set 0 to disable
        "tilt_max_deg": 70.0,       # set 0 to disable
        "fixed_yaw_deg": None,      # set None to disable
    }
    parser = argparse.ArgumentParser(description="Segment-wise time-optimal quad trajectory")
    parser.add_argument("--track", type=str, default=str(this_dir / "track.yaml"))
    parser.add_argument(
        "--dynamics-source",
        type=str,
        default="urdf",
        choices=["urdf", "yaml"],
        help="Dynamics parameter source: urdf or quad yaml",
    )
    parser.add_argument("--urdf", type=str, default=str(repo_root / "models" / "urdf" / "s500_simple.urdf"))
    parser.add_argument(
        "--quad-yaml",
        type=str,
        default=str(this_dir / "quad.yaml"),
        help="Path to quad dynamics yaml (used when --dynamics-source yaml)",
    )
    parser.add_argument("--nodes-per-segment", type=int, default=40)
    parser.add_argument(
        "--planning-mode",
        type=str,
        default="segmented",
        choices=["segmented", "monolithic"],
        help="Planning mode: segmented multiple-shooting or one monolithic NLP",
    )
    parser.add_argument(
        "--nodes-total",
        type=int,
        default=None,
        help="Total nodes for monolithic mode (default: nodes_per_segment * num_waypoints)",
    )
    parser.add_argument(
        "--waypoint-tolerance",
        type=float,
        default=0.25,
        help="Waypoint pass tolerance [m] for monolithic mode",
    )
    parser.add_argument(
        "--waypoint-penalty",
        type=float,
        default=2000.0,
        help="Soft-constraint penalty on waypoint tolerance violation (monolithic mode)",
    )
    parser.add_argument("--use-error-model", default=False, help="Enable NN-predicted gate error constraints")
    parser.add_argument(
        "--error-model-path",
        type=str,
        default=str(repo_root / "tracking_results" / "tracking_error_nn_model.joblib"),
        help="Path to trained NN error model",
    )
    parser.add_argument("--error-bound", type=float, default=0.10, help="Max allowed predicted |e_p| at gate [m]")
    parser.add_argument(
        "--error-penalty",
        type=float,
        default=5000.0,
        help="Soft-constraint penalty weight for NN predicted gate error",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(this_dir),
        help="Directory to save CSV/plot outputs (default: racing_traj_opt directory)",
    )
    parser.add_argument(
        "--no-show-plot",
        action="store_true",
        help="Disable interactive plot window after optimization",
    )
    args = parser.parse_args()

    track = load_track(Path(args.track))
    if args.dynamics_source == "urdf":
        quad = load_quad_from_urdf(Path(args.urdf))
    else:
        quad = load_quad_from_yaml(Path(args.quad_yaml))

    error_model_obj = None
    if args.use_error_model:
        model_path = Path(args.error_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Error model file not found: {model_path}")
        error_model_obj = joblib.load(model_path)

    planner = SegmentPlanner(
        quad=quad,
        track=track,
        options={
            "planning_mode": args.planning_mode,
            "nodes_per_segment": args.nodes_per_segment,
            "nodes_total": args.nodes_total if args.nodes_total is not None else args.nodes_per_segment * max(len(track["gates"]) + (1 if track.get("end_pos") is not None else 0), 1),
            "waypoint_tolerance": args.waypoint_tolerance,
            "waypoint_penalty": args.waypoint_penalty,
            "vel_guess": planner_fixed["vel_guess"],
            "omega_max": np.deg2rad(planner_fixed["omega_max_deg"]),
            "tilt_max": np.deg2rad(planner_fixed["tilt_max_deg"]),
            "fixed_yaw": None if planner_fixed["fixed_yaw_deg"] is None else np.deg2rad(planner_fixed["fixed_yaw_deg"]),
            "dynamics_model": planner_fixed["dynamics_model"],
            "accel_penalty": 0.01,
            "error_model": error_model_obj,
            "error_bound": args.error_bound,
            "error_penalty": args.error_penalty,
            "solver_options": {
                "ipopt": {
                    "max_iter": 200,
                    "tol": 1e-6,
                    "acceptable_tol": 1e-4,
                    "acceptable_iter": 20,
                    "hessian_approximation": "limited-memory",
                }
            },
        },
    )
    planner.setup()
    # Always allow partial solution export so plotting/output runs even when IPOPT hits iteration limit.
    planner.solve(allow_partial=True)
    t, p, v, q, w, u = planner.extract_full_trajectory()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = results_dir / f"result_segment_{ts}.csv"
    save_result_csv(timestamped, t, p, v, q, w, u)
    plot_path = results_dir / f"result_segment_{ts}.png"

    plot_cmd = [
        sys.executable,
        str(this_dir / "plot_segment_result.py"),
        "--plot-mode",
        "optimization",
        "--result-csv",
        str(timestamped),
        "--track",
        str(args.track),
        "--save-path",
        str(plot_path),
        "--disable-tracking",
        "--meta-dynamics-model",
        str(planner_fixed["dynamics_model"]),
        "--meta-planning-mode",
        str(args.planning_mode),
        "--meta-dynamics-source",
        str(args.dynamics_source),
        "--meta-solver-status",
        str(getattr(planner, "solve_status", "unknown")),
        "--meta-timestamp",
        ts,
        "--meta-thrust-min",
        str(quad.get("T_min", 0.0)),
        "--meta-thrust-max",
        str(quad.get("T_max", 0.0)),
    ]
    if not args.no_show_plot:
        plot_cmd.append("--show")
    try:
        subprocess.run(plot_cmd, check=True)
        plot_status = f"Saved plot: {plot_path}"
    except Exception as exc:
        plot_status = f"[WARN] Auto-plot failed: {exc}"

    if args.dynamics_source == "urdf":
        print(f"Dynamics source: URDF ({args.urdf})")
    else:
        print(f"Dynamics source: YAML ({args.quad_yaml})")
    print(f"Mass: {quad['mass']:.3f} kg")
    print(f"Planning mode: {args.planning_mode}")
    if args.planning_mode == "monolithic":
        print(f"Waypoint tolerance: {args.waypoint_tolerance:.3f} m")
        print(f"Waypoint penalty: {args.waypoint_penalty:.1f}")
    print(f"Dynamics model: {planner_fixed['dynamics_model']}")
    if planner_fixed["omega_max_deg"] > 0:
        print(f"Omega max: {planner_fixed['omega_max_deg']:.1f} deg/s")
    else:
        print("Omega max: disabled")
    if planner_fixed["tilt_max_deg"] > 0:
        print(f"Tilt max: {planner_fixed['tilt_max_deg']:.1f} deg")
    else:
        print("Tilt max: disabled")
    if planner_fixed["fixed_yaw_deg"] is None:
        print("Fixed yaw: disabled (free yaw)")
    else:
        print(f"Fixed yaw: {planner_fixed['fixed_yaw_deg']:.1f} deg")
    if args.use_error_model:
        print(f"Enabled NN error constraint: |e_p_pred| <= {args.error_bound:.3f} m")
        print(f"Soft penalty weight: {args.error_penalty:.1f}")
        print(f"Error model: {args.error_model_path}")
    print(f"Solver status: {getattr(planner, 'solve_status', 'unknown')}")
    print("Partial-save mode: always enabled")
    print(f"Saved segment result: {timestamped}")
    print(plot_status)
    print("Optimization finished.")


if __name__ == "__main__":
    main()