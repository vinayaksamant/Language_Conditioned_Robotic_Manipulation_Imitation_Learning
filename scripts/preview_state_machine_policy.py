from __future__ import annotations

import argparse
import time

import mujoco.viewer

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the deterministic pick-place state machine in MuJoCo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.03, help="Delay between viewer frames in seconds.")
    parser.add_argument("--hold-seconds", type=float, default=2.0, help="Pause before reset after success or timeout.")
    parser.add_argument("--repeat", action="store_true", help="Keep resetting and replaying after each rollout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=args.max_steps,
    )
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)
    observation = environment.reset(seed=args.seed)
    rollout_index = 0

    with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = (0.45, 0.0, 0.35)

        while viewer.is_running():
            result = environment.step(policy.act(observation))
            observation = result.observation
            viewer.sync()
            time.sleep(args.sleep)

            if result.terminated or result.truncated:
                status = "success" if result.terminated else "timeout"
                print(
                    f"Rollout {rollout_index}: {status} "
                    f"steps={observation['step_count']} held={observation['held']} "
                    f"contacts={observation['finger_contacts']} stage={policy.stage}"
                )
                end_time = time.time() + args.hold_seconds
                while viewer.is_running() and time.time() < end_time:
                    viewer.sync()
                    time.sleep(args.sleep)
                if not args.repeat:
                    break
                rollout_index += 1
                policy.reset()
                observation = environment.reset(seed=args.seed + rollout_index)


if __name__ == "__main__":
    main()
