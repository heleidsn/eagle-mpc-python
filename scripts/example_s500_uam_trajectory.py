#!/usr/bin/env python3
"""
S500 UAM Trajectory Planning Example

Demonstrates trajectory optimization for S500 UAM (UAV with Arm):
- Start state: initial position, orientation, arm configuration
- Grasp point: end-effector (gripper_link) target position
- Target state: final position and configuration

Usage:
    python example_s500_uam_trajectory.py
    python example_s500_uam_trajectory.py --max-iter 200 --dt 0.01
"""

import numpy as np
import os
from pathlib import Path
from s500_uam_trajectory_planner import S500UAMTrajectoryPlanner, make_uam_state


def create_grasp_trajectory():
    """Create trajectory: start -> grasp -> target"""
    start_state = make_uam_state(0, 0, 1.0, j1=-1.2, j2=-0.6)
    grasp_position = np.array([0.5, 0.0, 0.7])
    target_state = make_uam_state(1.0, 0.5, 1.2, j1=-0.8, j2=-0.3)
    durations = [3.0, 3.0]
    return start_state, grasp_position, target_state, durations


def create_catch_trajectory():
    """Create trajectory similar to s500_uam_catch: approach from side"""
    start_state = make_uam_state(-1.5, 0, 1.5, j1=-1.2, j2=-0.6)
    grasp_position = np.array([0.0, 0.0, 0.82])
    target_state = make_uam_state(1.5, 0, 1.5, j1=0, j2=-0.6)
    durations = [4.0, 4.0]
    return start_state, grasp_position, target_state, durations


def create_simple_trajectory():
    """Create simple trajectory: start -> target only (no grasp point, easier to converge)"""
    start_state = make_uam_state(0, 0, 1.0, j1=-1.2, j2=-0.6)
    target_state = make_uam_state(1.0, 0.5, 1.2, j1=-0.8, j2=-0.3)
    duration = 5.0
    return start_state, target_state, duration


def main():
    print("=" * 60)
    print("S500 UAM Trajectory Planning Example")
    print("Trajectory: Start → Grasp (EE position) → Target")
    print("=" * 60)

    try:
        planner = S500UAMTrajectoryPlanner()

        print("\nSelect trajectory:")
        print("  1. Grasp trajectory (start -> grasp -> target)")
        print("  2. Catch trajectory")
        print("  3. Simple trajectory (start -> target only, easier to converge)")
        choice = input("Choice (1/2/3) [3]: ").strip() or "3"

        if choice == "2":
            start_state, grasp_position, target_state, durations = create_catch_trajectory()
            trajectory_name = "catch"
            print(f"\nTrajectory: {trajectory_name}")
            print(f"  Start:     pos={start_state[:3]}, arm=[{start_state[7]:.2f}, {start_state[8]:.2f}]")
            print(f"  Grasp EE:  {grasp_position}")
            print(f"  Target:    pos={target_state[:3]}, arm=[{target_state[7]:.2f}, {target_state[8]:.2f}]")
            print(f"  Durations: {durations} s")
            planner.create_trajectory_problem(
                start_state, grasp_position, target_state, durations,
                dt=0.02, grasp_ee_weight=5000.0
            )
        elif choice == "1":
            start_state, grasp_position, target_state, durations = create_grasp_trajectory()
            trajectory_name = "grasp"
            print(f"\nTrajectory: {trajectory_name}")
            print(f"  Start:     pos={start_state[:3]}, arm=[{start_state[7]:.2f}, {start_state[8]:.2f}]")
            print(f"  Grasp EE:  {grasp_position}")
            print(f"  Target:    pos={target_state[:3]}, arm=[{target_state[7]:.2f}, {target_state[8]:.2f}]")
            print(f"  Durations: {durations} s")
            planner.create_trajectory_problem(
                start_state, grasp_position, target_state, durations,
                dt=0.02, grasp_ee_weight=5000.0
            )
        else:
            start_state, target_state, duration = create_simple_trajectory()
            trajectory_name = "simple"
            print(f"\nTrajectory: {trajectory_name} (start -> target only)")
            print(f"  Start:  pos={start_state[:3]}, arm=[{start_state[7]:.2f}, {start_state[8]:.2f}]")
            print(f"  Target: pos={target_state[:3]}, arm=[{target_state[7]:.2f}, {target_state[8]:.2f}]")
            print(f"  Duration: {duration} s")
            planner.create_trajectory_problem_simple(
                start_state, target_state, duration, dt=0.02
            )

        print("\nOptimizing...")
        converged = planner.solve_trajectory(max_iter=200, verbose=True)

        results_dir = Path(__file__).parent.parent / 'results' / 's500_uam_trajectory_optimization'
        os.makedirs(results_dir, exist_ok=True)
        plot_path = results_dir / f's500_uam_{trajectory_name}_trajectory.png'
        data_path = results_dir / f's500_uam_{trajectory_name}_trajectory.npz'

        planner.plot_trajectory(save_path=str(plot_path))
        planner.save_trajectory(str(data_path))

        print(f"\n✓ Results saved:")
        print(f"  - Plot: {plot_path}")
        print(f"  - Data: {data_path}")
        if not converged:
            print("\n✗ Optimization did not converge")
            print("  Try: option 3 (simple) for easier convergence, or increase --max-iter")

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
