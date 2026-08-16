from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer

from robot_manipulation_pi0.learning import BehaviorCloningPolicy
from robot_manipulation_pi0.learning.dataset import observation_to_feature
from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained behavior cloning policy in MuJoCo.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/bc_policy.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sleep", type=float, default=0.03, help="Delay between viewer frames in seconds.")
    parser.add_argument("--hold-seconds", type=float, default=2.0, help="Pause before reset after success or timeout.")
    parser.add_argument("--repeat", action="store_true", help="Keep resetting and replaying after each rollout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = BehaviorCloningPolicy.load(args.checkpoint, device=args.device)
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=args.max_steps,
    )
    environment = PickPlaceEnvironment(config)
    observation = environment.reset(seed=args.seed)
    stage_tracker = ScriptedOraclePolicy(environment)
    rollout_index = 0

    with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = (0.45, 0.0, 0.35)

        while viewer.is_running():
            stage_tracker.act(observation)
            feature = observation_to_feature(observation, stage_tracker.stage_index)
            if feature.shape[0] != policy.observation_mean.shape[0]:
                raise ValueError(
                    f"Checkpoint expects {policy.observation_mean.shape[0]} observation features, "
                    f"but the current environment provides {feature.shape[0]}. Retrain the checkpoint."
                )
            feature = feature[None, :]
            action = policy.predict(feature)[0]
            result = environment.step(action)
            observation = result.observation
            viewer.sync()
            time.sleep(args.sleep)

            if result.terminated or result.truncated:
                status = "success" if result.terminated else "timeout"
                print(
                    f"Rollout {rollout_index}: {status} "
                    f"steps={observation['step_count']} held={observation['held']} "
                    f"contacts={observation['finger_contacts']} stage={stage_tracker.stage}"
                )
                end_time = time.time() + args.hold_seconds
                while viewer.is_running() and time.time() < end_time:
                    viewer.sync()
                    time.sleep(args.sleep)
                if not args.repeat:
                    break
                rollout_index += 1
                observation = environment.reset(seed=args.seed + rollout_index)
                stage_tracker.reset()


if __name__ == "__main__":
    main()
