from __future__ import annotations

import argparse
import time

import mujoco.viewer

from robot_manipulation_pi0.demos import collect_episode
from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


class ViewerClosed(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the exact oracle rollout used by collect_demos.py without saving data."
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.04, help="Delay between simulation frames in seconds.")
    parser.add_argument("--hold-seconds", type=float, default=1.5, help="Pause after each episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=args.max_steps,
    )
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)

    with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.55
        viewer.cam.lookat[:] = (0.50, 0.02, 0.32)

        for episode_index in range(args.episodes):
            seed = args.seed_start + episode_index
            last_stage: str | None = None

            def show_step(env, oracle, observation, action, result) -> None:
                del env, action, result
                nonlocal last_stage
                if not viewer.is_running():
                    raise ViewerClosed
                if oracle.stage != last_stage:
                    print(
                        f"seed={seed} step={observation['step_count']} stage={oracle.stage} "
                        f"contacts={observation['finger_contacts']} held={observation['held']}"
                    )
                    last_stage = oracle.stage
                viewer.sync()
                time.sleep(args.sleep)

            try:
                episode = collect_episode(
                    seed,
                    config,
                    environment=environment,
                    policy=policy,
                    step_callback=show_step,
                )
            except ViewerClosed:
                break

            status = "success" if episode.success else "timeout"
            print(f"seed={seed} {status} steps={episode.steps}")
            end_time = time.time() + args.hold_seconds
            while viewer.is_running() and time.time() < end_time:
                viewer.sync()
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
