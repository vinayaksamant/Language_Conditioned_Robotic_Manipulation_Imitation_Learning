from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    truncated: bool
    steps: int
    final_distance: float
    held: bool


def run_episode(seed: int, config: PickPlaceConfig) -> EpisodeResult:
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)
    observation = environment.reset(seed=seed)
    result = None

    for _ in range(config.max_steps):
        result = environment.step(policy.act(observation))
        observation = result.observation
        if result.terminated or result.truncated:
            break

    if result is None:
        raise RuntimeError("Oracle episode did not execute any simulation steps.")

    object_position = np.asarray(observation["object_pos"], dtype=float)
    target_position = np.asarray(observation["target_pos"], dtype=float)
    return EpisodeResult(
        seed=seed,
        success=bool(result.info["success"]),
        truncated=bool(result.truncated),
        steps=int(observation["step_count"]),
        final_distance=float(np.linalg.norm((object_position - target_position)[:2])),
        held=bool(observation["held"]),
    )


def evaluate(seed_start: int, episodes: int, config: PickPlaceConfig) -> list[EpisodeResult]:
    return [run_episode(seed, config) for seed in range(seed_start, seed_start + episodes)]


def print_summary(results: list[EpisodeResult]) -> None:
    successes = [result for result in results if result.success]
    failures = [result for result in results if not result.success]
    success_rate = len(successes) / len(results) if results else 0.0
    average_steps = np.mean([result.steps for result in successes]) if successes else float("nan")
    average_distance = np.mean([result.final_distance for result in results]) if results else float("nan")

    print(f"Episodes: {len(results)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average successful steps: {average_steps:.1f}")
    print(f"Average final planar distance: {average_distance:.4f} m")

    if failures:
        failed_seeds = ", ".join(str(result.seed) for result in failures)
        print(f"Failed seeds: {failed_seeds}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the scripted oracle on seeded Panda pick-place rollouts.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of seeded episodes to evaluate.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed to evaluate.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum simulation steps per episode.")
    parser.add_argument("--workspace-size", type=float, default=0.5, help="Workspace size used by the environment reset.")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=None,
        help="Exit with an error if the success rate is below this fraction, for example 0.8.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.min_success_rate is not None and not 0.0 <= args.min_success_rate <= 1.0:
        raise ValueError("--min-success-rate must be between 0 and 1.")

    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=args.workspace_size,
        max_steps=args.max_steps,
    )
    results = evaluate(args.seed_start, args.episodes, config)
    print_summary(results)

    if args.min_success_rate is not None:
        success_rate = sum(result.success for result in results) / len(results)
        if success_rate < args.min_success_rate:
            raise SystemExit(
                f"Oracle success rate {success_rate:.2%} is below required {args.min_success_rate:.2%}."
            )


if __name__ == "__main__":
    main()
