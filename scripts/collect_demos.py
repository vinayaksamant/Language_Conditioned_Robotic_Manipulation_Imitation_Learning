from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_manipulation_pi0.demos import DEMONSTRATION_SCHEMA_VERSION, collect_episode, save_episode
from robot_manipulation_pi0.sim import ACTION_MODE, PickPlaceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect successful scripted-oracle Panda pick-place demonstrations.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of successful demonstrations to save.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed to try.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum seeds to try before failing.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum simulation steps per rollout.")
    parser.add_argument("--workspace-size", type=float, default=0.5, help="Workspace size used by the environment reset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/demos/pick_place_oracle"),
        help="Directory where successful demonstration .npz files and manifest.json are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    max_attempts = args.max_attempts if args.max_attempts is not None else args.episodes * 3
    if max_attempts < args.episodes:
        raise ValueError("--max-attempts must be at least --episodes.")

    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=args.workspace_size,
        max_steps=args.max_steps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, object]] = []
    failures: list[int] = []
    seed = args.seed_start
    attempts = 0
    while len(saved) < args.episodes and attempts < max_attempts:
        episode = collect_episode(seed, config)
        if episode.success:
            filename = f"episode_{len(saved):06d}_seed_{seed:06d}.npz"
            save_episode(args.output_dir / filename, episode)
            saved.append({"seed": seed, "file": filename, "steps": episode.steps})
        else:
            failures.append(seed)
        seed += 1
        attempts += 1

    manifest = {
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "action_mode": ACTION_MODE,
        "task": "pick_place_single_object",
        "robot": config.robot_name,
        "successful_episodes": len(saved),
        "attempted_episodes": attempts,
        "seed_start": args.seed_start,
        "max_steps": config.max_steps,
        "workspace_size": config.workspace_size,
        "episodes": saved,
        "failed_seeds": failures,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved successful episodes: {len(saved)}")
    print(f"Attempts: {attempts}")
    print(f"Output directory: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    if failures:
        print("Failed seeds:", ", ".join(str(seed_value) for seed_value in failures))
    if len(saved) < args.episodes:
        raise SystemExit(f"Only collected {len(saved)} successful episodes out of requested {args.episodes}.")


if __name__ == "__main__":
    main()
