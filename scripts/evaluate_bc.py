from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_manipulation_pi0.learning import BehaviorCloningPolicy, load_demonstration_dataset
from robot_manipulation_pi0.learning.dataset import observation_to_feature
from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a behavior cloning checkpoint on demonstration data.")
    parser.add_argument("--demo-dir", type=Path, default=Path("data/demos/pick_place_oracle"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rollout-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {args.checkpoint}. "
            "Run scripts/train_bc.py first with the same --output path."
        )
    dataset = load_demonstration_dataset(args.demo_dir)
    policy = BehaviorCloningPolicy.load(args.checkpoint, device=args.device)
    observations = dataset.observations[:, : policy.observation_mean.shape[0]]
    predictions = policy.predict(observations)
    errors = predictions - dataset.actions
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    max_abs_error = float(np.max(np.abs(errors)))

    print(f"Transitions: {len(dataset.actions)}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Max absolute error: {max_abs_error:.6f}")

    if args.rollout_episodes > 0:
        successes, distances, failures = evaluate_rollouts(
            policy,
            episodes=args.rollout_episodes,
            seed_start=args.seed_start,
            max_steps=args.max_steps,
        )
        print(f"Closed-loop episodes: {args.rollout_episodes}")
        print(f"Closed-loop successes: {successes}")
        print(f"Closed-loop success rate: {successes / args.rollout_episodes:.2%}")
        print(f"Average final planar distance: {np.mean(distances):.4f} m")
        if failures:
            print("Failed rollout seeds:", ", ".join(failures))


def evaluate_rollouts(
    policy: BehaviorCloningPolicy,
    episodes: int,
    seed_start: int,
    max_steps: int,
) -> tuple[int, list[float], list[str]]:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=max_steps,
    )
    successes = 0
    distances: list[float] = []
    failures: list[str] = []

    for seed in range(seed_start, seed_start + episodes):
        environment = PickPlaceEnvironment(config)
        stage_tracker = ScriptedOraclePolicy(environment)
        observation = environment.reset(seed=seed)

        for _ in range(max_steps):
            stage_tracker.act(observation)
            feature = observation_to_feature(observation, stage_tracker.stage_index)
            if feature.shape[0] != policy.observation_mean.shape[0]:
                raise ValueError(
                    f"Checkpoint expects {policy.observation_mean.shape[0]} observation features, "
                    f"but the current environment provides {feature.shape[0]}. Retrain the checkpoint."
                )
            action = policy.predict(feature[None, :])[0]
            result = environment.step(action)
            observation = result.observation
            if result.terminated or result.truncated:
                break

        successes += int(result.info["success"])
        object_position = np.asarray(observation["object_pos"], dtype=float)
        target_position = np.asarray(observation["target_pos"], dtype=float)
        distances.append(float(np.linalg.norm((object_position - target_position)[:2])))
        if not result.info["success"]:
            failures.append(f"{seed}({stage_tracker.stage})")

    return successes, distances, failures


if __name__ == "__main__":
    main()
