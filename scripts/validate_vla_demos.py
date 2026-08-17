from __future__ import annotations

import argparse
from pathlib import Path

from robot_manipulation_pi0.vla import validate_vla_demo_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a collected multi-view VLA demonstration directory.")
    parser.add_argument("--demo-dir", type=Path, default=Path("data/demos/vla_multi_object_v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_vla_demo_directory(args.demo_dir)
    print(f"Episodes: {summary.episode_count}")
    print(f"Transitions: {summary.total_steps}")
    print(f"Average steps: {summary.mean_steps:.1f}")
    print(f"Objects: {dict(summary.object_counts)}")
    print(f"Targets: {dict(summary.target_counts)}")
    print(f"Cameras: {', '.join(summary.camera_keys)}")
    print(f"Robot state dimension: {summary.state_dimension}")
    print(f"Action dimension: {summary.action_dimension}")
    print("Privileged coordinate check: passed")


if __name__ == "__main__":
    main()
