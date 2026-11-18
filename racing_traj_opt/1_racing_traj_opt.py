import csv
from pathlib import Path
from re import T
import subprocess
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
import joblib

from segment_planner import SegmentPlanner


def _safe_name_token(name: str) -> str:
    token = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in str(name).strip())
    return token.strip("_") or "trajectory"


def load_error_model_joblib(path: Path):
    """Load sklearn/joblib error models across NumPy 1.x/2.x pickle differences.

    ``model_3.joblib`` was produced in an environment whose pickle references
    ``numpy._core`` and a newer random bit-generator state. The planner only
    needs the fitted scaler and MLP weights for forward prediction, not the
    RNG state, so the fallback below safely stubs RNG reconstruction.
    """
    try:
        return joblib.load(path)
    except (ModuleNotFoundError, ValueError) as first_err:
        import sys as _sys

        try:
            import numpy.core as _np_core

            _sys.modules.setdefault("numpy._core", _np_core)
            for name in (
                "multiarray",
                "numeric",
                "fromnumeric",
                "umath",
                "_multiarray_umath",
                "records",
                "numerictypes",
            ):
                try:
                    mod = __import__(f"numpy.core.{name}", fromlist=["*"])
                    _sys.modules.setdefault(f"numpy._core.{name}", mod)
                except Exception:
                    pass

            class _DummyRng:
                def __init__(self, *args, **kwargs):
                    pass

                def __setstate__(self, state):
                    pass

                def __getstate__(self):
                    return None

            import numpy.random._pickle as _nrp

            _nrp.__bit_generator_ctor = lambda *args, **kwargs: _DummyRng()
            _nrp.__generator_ctor = lambda *args, **kwargs: _DummyRng()
            _nrp.__randomstate_ctor = lambda *args, **kwargs: _DummyRng()
            return joblib.load(path)
        except Exception as second_err:
            raise RuntimeError(
                f"Failed to load error model {path}.\n"
                f"Initial error: {first_err}\n"
                f"Compatibility fallback error: {second_err}"
            ) from second_err


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


def save_gate_events_csv(path: Path, gate_events: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "index", "t", "x", "y", "z"])
        for ev in gate_events:
            writer.writerow(
                [
                    str(ev["kind"]),
                    int(ev["index"]),
                    float(ev["t"]),
                    float(ev["x"]),
                    float(ev["y"]),
                    float(ev["z"]),
                ]
            )


def main():
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    # -------------------------------------------------------------------------
    # User settings: edit here and run directly (no argparse needed).
    # -------------------------------------------------------------------------
    cfg = {
        # Paths
        "trajectory_name": "racing_s500_new",
        
        "track": str(this_dir / "track.yaml"),
        "dynamics_source": "urdf",  # "urdf" or "yaml"
        "urdf": str(repo_root / "models" / "urdf" / "s500_simple.urdf"),
        "quad_yaml": str(this_dir / "quad.yaml"),
        # Results are saved under: <output_dir>/<trajectory_name>/
        #   no error model: traj_planned.csv / traj_planned.png
        #   with NN model : traj_planned_nn.csv / traj_planned_nn.png
        "output_dir": str(this_dir / "results"),
        "show_plot": True,

        # Planner mode
        "planning_mode": "segmented",  # "segmented" or "monolithic"
        "nodes_per_segment": 40,
        "nodes_total": None,  # only used in monolithic mode
        "waypoint_tolerance": 0.25,
        "waypoint_penalty": 2000.0,

        # Dynamics/planner fixed params
        "dynamics_model": "full",  # set "simple" to use point-mass model
        "vel_guess": 3.0,
        "omega_max_deg": 150.0,  # set 0 to disable
        "tilt_max_deg": 70.0,  # set 0 to disable
        "fixed_yaw_deg": None,  # set None to disable
        "accel_penalty": 0.01,

        # Error model options
        "use_error_model": True,
        "error_model_type": "nn",  # "nn" or "linear_speed"
        "error_model_path": str(this_dir / "error_model" / "model_3.joblib"),
        "linear_error_k": 0.05,
        "error_bound": 0.10,
        "error_penalty": 5000.0,

        # IPOPT options
        "ipopt_max_iter": 1000,
        "ipopt_tol": 1e-6,
        "ipopt_acceptable_tol": 1e-4,
        "ipopt_acceptable_iter": 20,
    }

    track = load_track(Path(cfg["track"]))
    if cfg["dynamics_source"] == "urdf":
        quad = load_quad_from_urdf(Path(cfg["urdf"]))
    else:
        quad = load_quad_from_yaml(Path(cfg["quad_yaml"]))

    error_model_obj = None
    if cfg["use_error_model"] and cfg["error_model_type"] == "nn":
        model_path = Path(cfg["error_model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Error model file not found: {model_path}")
        error_model_obj = load_error_model_joblib(model_path)
        feature_names = error_model_obj.get("feature_names_per_frame") if isinstance(error_model_obj, dict) else None
        model_name = error_model_obj.get("model_name") if isinstance(error_model_obj, dict) else None
        if feature_names:
            print(
                f"Loaded error model {model_name or 'legacy'} with "
                f"{len(feature_names)} planner features: {feature_names}"
            )

    planner = SegmentPlanner(
        quad=quad,
        track=track,
        options={
            "planning_mode": cfg["planning_mode"],
            "nodes_per_segment": cfg["nodes_per_segment"],
            "nodes_total": cfg["nodes_total"] if cfg["nodes_total"] is not None else cfg["nodes_per_segment"] * max(len(track["gates"]) + (1 if track.get("end_pos") is not None else 0), 1),
            "waypoint_tolerance": cfg["waypoint_tolerance"],
            "waypoint_penalty": cfg["waypoint_penalty"],
            "vel_guess": cfg["vel_guess"],
            "omega_max": np.deg2rad(cfg["omega_max_deg"]),
            "tilt_max": np.deg2rad(cfg["tilt_max_deg"]),
            "fixed_yaw": None if cfg["fixed_yaw_deg"] is None else np.deg2rad(cfg["fixed_yaw_deg"]),
            "dynamics_model": cfg["dynamics_model"],
            "accel_penalty": cfg["accel_penalty"],
            "use_error_model": cfg["use_error_model"],
            "error_model": error_model_obj,
            "error_model_type": cfg["error_model_type"],
            "linear_error_k": cfg["linear_error_k"],
            "error_bound": cfg["error_bound"],
            "error_penalty": cfg["error_penalty"],
            "solver_options": {
                "ipopt": {
                    "max_iter": cfg["ipopt_max_iter"],
                    "tol": cfg["ipopt_tol"],
                    "acceptable_tol": cfg["ipopt_acceptable_tol"],
                    "acceptable_iter": cfg["ipopt_acceptable_iter"],
                    "hessian_approximation": "limited-memory",
                }
            },
        },
    )
    planner.setup()
    # Always allow partial solution export so plotting/output runs even when IPOPT hits iteration limit.
    planner.solve(allow_partial=True)
    t, p, v, q, w, u = planner.extract_full_trajectory()

    trajectory_name = _safe_name_token(cfg.get("trajectory_name") or Path(cfg["track"]).stem)
    results_dir = Path(cfg["output_dir"]) / trajectory_name
    results_dir.mkdir(parents=True, exist_ok=True)
    result_stem = "traj_planned_nn" if cfg["use_error_model"] else "traj_planned"
    result_csv = results_dir / f"{result_stem}.csv"
    save_result_csv(result_csv, t, p, v, q, w, u)
    gate_stem = "traj_gates_nn" if cfg["use_error_model"] else "traj_gates"
    gate_csv = results_dir / f"{gate_stem}.csv"
    save_gate_events_csv(gate_csv, planner.extract_gate_events())
    plot_path = results_dir / f"{result_stem}.png"
    copied_error_model_path = None
    if cfg["use_error_model"] and cfg["error_model_type"] == "nn":
        src_model_path = Path(cfg["error_model_path"])
        copied_error_model_path = results_dir / src_model_path.name
        if src_model_path.exists():
            shutil.copy2(src_model_path, copied_error_model_path)

    plot_cmd = [
        sys.executable,
        str(this_dir / "plot_segment_result.py"),
        "--plot-mode",
        "optimization",
        "--result-csv",
        str(result_csv),
        "--track",
        str(cfg["track"]),
        "--save-path",
        str(plot_path),
        "--disable-tracking",
        "--meta-dynamics-model",
        str(cfg["dynamics_model"]),
        "--meta-planning-mode",
        str(cfg["planning_mode"]),
        "--meta-dynamics-source",
        str(cfg["dynamics_source"]),
        "--meta-solver-status",
        str(getattr(planner, "solve_status", "unknown")),
        "--meta-timestamp",
        trajectory_name,
        "--meta-thrust-min",
        str(quad.get("T_min", 0.0)),
        "--meta-thrust-max",
        str(quad.get("T_max", 0.0)),
    ]
    if cfg["show_plot"]:
        plot_cmd.append("--show")
    try:
        subprocess.run(plot_cmd, check=True)
        plot_status = f"Saved plot: {plot_path}"
    except Exception as exc:
        plot_status = f"[WARN] Auto-plot failed: {exc}"

    if cfg["dynamics_source"] == "urdf":
        print(f"Dynamics source: URDF ({cfg['urdf']})")
    else:
        print(f"Dynamics source: YAML ({cfg['quad_yaml']})")
    print(f"Mass: {quad['mass']:.3f} kg")
    print(f"Planning mode: {cfg['planning_mode']}")
    if cfg["planning_mode"] == "monolithic":
        print(f"Waypoint tolerance: {cfg['waypoint_tolerance']:.3f} m")
        print(f"Waypoint penalty: {cfg['waypoint_penalty']:.1f}")
    print(f"Dynamics model: {cfg['dynamics_model']}")
    if cfg["omega_max_deg"] > 0:
        print(f"Omega max: {cfg['omega_max_deg']:.1f} deg/s")
    else:
        print("Omega max: disabled")
    if cfg["tilt_max_deg"] > 0:
        print(f"Tilt max: {cfg['tilt_max_deg']:.1f} deg")
    else:
        print("Tilt max: disabled")
    if cfg["fixed_yaw_deg"] is None:
        print("Fixed yaw: disabled (free yaw)")
    else:
        print(f"Fixed yaw: {cfg['fixed_yaw_deg']:.1f} deg")
    if cfg["use_error_model"]:
        print(
            f"Enabled error constraint ({cfg['error_model_type']}): "
            f"|e_p_pred| <= {cfg['error_bound']:.3f} m"
        )
        print(f"Soft penalty weight: {cfg['error_penalty']:.1f}")
        if cfg["error_model_type"] == "nn":
            print(f"Error model: {cfg['error_model_path']}")
        else:
            print(f"Linear speed coefficient k: {cfg['linear_error_k']:.4f}")
    print(f"Solver status: {getattr(planner, 'solve_status', 'unknown')}")
    print("Partial-save mode: always enabled")
    print(f"Trajectory name: {trajectory_name}")
    print(f"Saved segment result: {result_csv}")
    print(f"Saved gate events: {gate_csv}")
    if copied_error_model_path is not None:
        print(f"Copied error model: {copied_error_model_path}")
    print(plot_status)
    print("Optimization finished.")


if __name__ == "__main__":
    main()