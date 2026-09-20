from __future__ import annotations

import argparse
from pathlib import Path

from robot_manipulation_pi0.vla import (
    LeRobotConversionResult,
    RawVLAEpisode,
    convert_vla_to_lerobot,
    existing_lerobot_conversion,
    plan_lerobot_conversion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the local LeRobot dataset required for pi0 fine-tuning, if absent."
    )
    parser.add_argument("--demo-dir", type=Path, default=Path("data/demos/vla_centered_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/lerobot/vla_centered_v4"))
    parser.add_argument("--repo-id", default="local/robot_manipulation_pi0_vla")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Store camera streams as videos instead of individual images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing = existing_lerobot_conversion(
        args.output_dir,
        expected_repo_id=args.repo_id,
    )
    if existing is not None:
        _print_result("Dataset already prepared", existing)
        return
    if not (args.demo_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"No raw demonstrations were found at {args.demo_dir}. A VLA dataset cannot be "
            "synthesized without demonstrations. Record human/teleoperated demonstrations in "
            "the raw VLA schema, or point --demo-dir at an existing demonstration dataset."
        )

    plan = plan_lerobot_conversion(
        args.demo_dir,
        split_seed=args.split_seed,
        use_videos=args.videos,
    )
    print(
        f"Preparing {len(plan.episodes)} episodes / {plan.summary.total_steps} frames "
        f"at {plan.summary.fps:g} Hz"
    )
    print(
        f"Splits: train={len(plan.splits.train)} val={len(plan.splits.val)} "
        f"test={len(plan.splits.test)}"
    )
    result = convert_vla_to_lerobot(
        plan,
        args.output_dir,
        args.repo_id,
        image_writer_threads=args.image_writer_threads,
        on_episode=_print_progress,
    )
    _print_result("Dataset prepared", result)


def _print_progress(current: int, total: int, episode: RawVLAEpisode) -> None:
    print(f"[{current:03d}/{total:03d}] {episode.instruction} ({episode.steps} frames)")


def _print_result(label: str, result: LeRobotConversionResult) -> None:
    print(f"{label}: {result.output_directory}")
    print(f"Episodes: {result.episode_count}")
    print(f"Frames: {result.frame_count}")
    print(f"LeRobot repo ID: {result.repo_id}")


if __name__ == "__main__":
    main()
