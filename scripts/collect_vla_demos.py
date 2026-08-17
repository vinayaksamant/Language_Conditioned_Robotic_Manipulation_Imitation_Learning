from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from robot_manipulation_pi0.sim import ACTION_MODE, MultiObjectPickPlaceEnvironment, PickPlaceConfig
from robot_manipulation_pi0.vla import (
    VLA_CAMERA_SPECS,
    VLA_DATASET_SCHEMA_VERSION,
    CameraSpec,
    MujocoCameraRig,
    collect_vla_episode,
    sample_task,
    save_vla_episode,
)


OBJECT_CHOICES = ("mixed", "red_cube", "blue_cylinder")
TARGET_CHOICES = ("mixed", "green_plate", "yellow_plate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect language-conditioned multi-view VLA demonstrations.")
    parser.add_argument("--episodes", type=int, default=10, help="Successful episodes to save.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--object", choices=OBJECT_CHOICES, default="mixed")
    parser.add_argument("--target", choices=TARGET_CHOICES, default="green_plate")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--record-every",
        type=int,
        default=4,
        help="Record every Nth 40 Hz simulator control step; 4 produces a 10 Hz camera dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/demos/vla_multi_object_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must be positive.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.record_every <= 0:
        raise ValueError("--record-every must be positive.")
    max_attempts = args.max_attempts if args.max_attempts is not None else args.episodes * 2
    if max_attempts < args.episodes:
        raise ValueError("--max-attempts must be at least --episodes.")

    object_keys = ("red_cube", "blue_cylinder") if args.object == "mixed" else (args.object,)
    target_keys = ("green_plate", "yellow_plate") if args.target == "mixed" else (args.target,)
    task_pairs = tuple((object_key, target_key) for object_key in object_keys for target_key in target_keys)
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("red_cube", "blue_cylinder"),
        workspace_size=0.5,
        max_steps=args.max_steps,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    camera_specs = tuple(
        CameraSpec(spec.key, spec.camera_name, width=args.width, height=args.height)
        for spec in VLA_CAMERA_SPECS
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    seed = args.seed_start
    attempts = 0
    with MujocoCameraRig(environment.model, camera_specs) as cameras:
        while len(saved) < args.episodes and attempts < max_attempts:
            object_key, target_key = task_pairs[len(saved) % len(task_pairs)]
            task = sample_task(object_key, np.random.default_rng(seed), target_key=target_key)
            print(
                f"Collecting attempt {attempts + 1}/{max_attempts}: "
                f"seed={seed} object={object_key} target={target_key}"
            )
            episode = collect_vla_episode(
                seed,
                config,
                task,
                environment,
                cameras,
                record_every=args.record_every,
            )
            if episode.success:
                filename = (
                    f"episode_{len(saved):06d}_seed_{seed:06d}_{object_key}_{target_key}.npz"
                )
                episode_path = args.output_dir / filename
                save_vla_episode(episode_path, episode)
                saved.append(
                    {
                        "seed": seed,
                        "file": filename,
                        "steps": episode.steps,
                        "object_key": object_key,
                        "target_key": target_key,
                        "instruction": task.instruction,
                    }
                )
                counts[object_key] += 1
                target_counts[target_key] += 1
                size_mb = episode_path.stat().st_size / (1024 * 1024)
                print(
                    f"  saved episode {len(saved)}/{args.episodes}: {episode.steps} frames, "
                    f"{episode.simulation_steps} simulation steps, {size_mb:.1f} MB"
                )
            else:
                failures.append({"seed": seed, "object_key": object_key, "target_key": target_key})
                print(f"  failed: seed={seed} object={object_key} target={target_key}")
            seed += 1
            attempts += 1

    manifest = {
        "schema_version": VLA_DATASET_SCHEMA_VERSION,
        "action_mode": ACTION_MODE,
        "dataset_type": "language_conditioned_multi_object_pick_place",
        "robot": config.robot_name,
        "camera_keys": [spec.key for spec in camera_specs],
        "successful_episodes": len(saved),
        "attempted_episodes": attempts,
        "object_counts": dict(sorted(counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "seed_start": args.seed_start,
        "max_steps": config.max_steps,
        "record_every": args.record_every,
        "fps": 1.0 / (float(environment.model.opt.timestep) * config.substeps * args.record_every),
        "episodes": saved,
        "failures": failures,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved successful episodes: {len(saved)}")
    print(f"Object counts: {dict(sorted(counts.items()))}")
    print(f"Target counts: {dict(sorted(target_counts.items()))}")
    print(f"Attempts: {attempts}")
    print(f"Output directory: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    if len(saved) < args.episodes:
        raise SystemExit(f"Only collected {len(saved)} successful episodes out of {args.episodes} requested.")


if __name__ == "__main__":
    main()
