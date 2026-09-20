from __future__ import annotations

import argparse
from pathlib import Path

from robot_manipulation_pi0.vla import (
    RawVLAEpisode,
    convert_vla_to_lerobot,
    plan_lerobot_conversion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, split, and convert raw MuJoCo VLA demonstrations to LeRobot v3 format."
    )
    parser.add_argument("--demo-dir", type=Path, default=Path("data/demos/vla_centered_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/lerobot/vla_centered_v4"))
    parser.add_argument("--repo-id", default="local/robot_manipulation_pi0_vla")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Encode camera streams as videos. Image storage is the dependency-light default.",
    )
    parser.add_argument(
        "--allow-legacy-scene",
        action="store_true",
        help="Permit conversion of unversioned v1 data that does not match the current simulator scene.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and print the split plan without requiring LeRobot or writing data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fractions = {
        "train": args.train_fraction,
        "val": args.val_fraction,
        "test": args.test_fraction,
    }
    plan = plan_lerobot_conversion(
        args.demo_dir,
        split_seed=args.split_seed,
        split_fractions=fractions,
        use_videos=args.videos,
        allow_legacy_scene=args.allow_legacy_scene,
    )
    print(f"Validated episodes: {len(plan.episodes)}")
    print(f"Validated frames: {plan.summary.total_steps}")
    print(f"Scene: {plan.summary.scene_ids[0]}")
    print(f"Frame rate: {plan.summary.fps:g} Hz")
    print(
        "Splits: "
        f"train={len(plan.splits.train)} val={len(plan.splits.val)} test={len(plan.splits.test)}"
    )
    print(f"Storage: {'video' if plan.use_videos else 'image'}")
    if args.plan_only:
        print("Plan only: no files were written.")
        return

    result = convert_vla_to_lerobot(
        plan,
        args.output_dir,
        args.repo_id,
        image_writer_threads=args.image_writer_threads,
        on_episode=_print_progress,
    )
    print(f"Converted episodes: {result.episode_count}")
    print(f"Converted frames: {result.frame_count}")
    print(f"LeRobot repo ID: {result.repo_id}")
    print(f"Output directory: {result.output_directory}")
    print(f"Project metadata: {result.output_directory / 'robot_manipulation_pi0.json'}")


def _print_progress(current: int, total: int, episode: RawVLAEpisode) -> None:
    print(
        f"Converting episode {current}/{total}: seed={episode.seed} "
        f"task={episode.object_key}->{episode.target_key} frames={episode.steps}"
    )


if __name__ == "__main__":
    main()
