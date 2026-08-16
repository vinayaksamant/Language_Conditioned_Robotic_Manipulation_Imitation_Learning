from __future__ import annotations

import argparse
from pathlib import Path

from robot_manipulation_pi0.demos import validate_demo_directory
from robot_manipulation_pi0.sim import PickPlaceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate collected oracle demonstration files.")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("data/demos/pick_place_oracle"),
        help="Directory containing manifest.json and episode .npz files.",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=None,
        help="Fail if the directory does not contain this many successful episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_episodes is not None and args.expected_episodes <= 0:
        raise ValueError("--expected-episodes must be positive.")

    config = PickPlaceConfig(robot_name="franka_panda", object_names=("cube",), workspace_size=0.5)
    summary = validate_demo_directory(args.demo_dir, expected_action_dimension=config.action_dimension)
    if args.expected_episodes is not None and summary.episode_count != args.expected_episodes:
        raise SystemExit(
            f"Expected {args.expected_episodes} episodes, found {summary.episode_count} in {args.demo_dir}."
        )

    print(f"Demo directory: {summary.directory}")
    print(f"Episodes: {summary.episode_count}")
    print(f"Total transitions: {summary.total_steps}")
    print(f"Action dimension: {summary.action_dimension}")
    print(f"qpos dimension: {summary.qpos_dimension}")
    print(f"qvel dimension: {summary.qvel_dimension}")
    print(f"Episode length: min={summary.min_steps}, mean={summary.mean_steps:.1f}, max={summary.max_steps}")
    print(f"Failed seeds recorded in manifest: {len(summary.failed_seeds)}")


if __name__ == "__main__":
    main()
