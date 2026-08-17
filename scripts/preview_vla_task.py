from __future__ import annotations

import argparse
import time

import mujoco.viewer

from robot_manipulation_pi0.sim import MultiObjectPickPlaceEnvironment, PickPlaceConfig, ScriptedOraclePolicy
from robot_manipulation_pi0.vla import (
    VLA_ORACLE_JOINT_STEP_LIMIT,
    resolve_task_instruction,
    task_for_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a typed multi-object language task using the scripted demonstration expert."
    )
    parser.add_argument(
        "--instruction",
        default=task_for_object("red_cube").instruction,
        help="For example: 'Pick the red block and put it on the green plate.'",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.025)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must be positive.")
    if args.sleep < 0.0 or args.hold_seconds < 0.0:
        raise ValueError("--sleep and --hold-seconds cannot be negative.")

    task = resolve_task_instruction(args.instruction)
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("red_cube", "blue_cylinder"),
        workspace_size=0.5,
        max_steps=args.max_steps,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(
        environment,
        joint_command_step_limit=VLA_ORACLE_JOINT_STEP_LIMIT,
    )

    print("Controller: scripted VLA demonstration expert (not a trained pi0 policy)")
    print(f"Instruction: {task.instruction}")
    print(f"Resolved object: {task.object_key}")
    print(f"Resolved target: {task.target_key}")

    with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.55
        viewer.cam.lookat[:] = (0.50, 0.02, 0.32)

        for episode_index in range(args.episodes):
            if not viewer.is_running():
                break
            seed = args.seed + episode_index
            observation = environment.reset(
                seed=seed,
                object_key=task.object_key,
                target_key=task.target_key,
            )
            policy.reset()
            last_stage: str | None = None
            result = None

            for _ in range(config.max_steps):
                if not viewer.is_running():
                    return
                action = policy.act(observation)
                if policy.stage != last_stage:
                    print(
                        f"seed={seed} step={observation['step_count']} stage={policy.stage} "
                        f"held={observation['held']} contacts={observation['finger_contacts']}"
                    )
                    last_stage = policy.stage
                result = environment.step(action)
                observation = result.observation
                viewer.sync()
                time.sleep(args.sleep)
                if result.terminated or result.truncated:
                    break

            if result is None:
                raise RuntimeError("VLA preview did not execute any simulation steps.")
            status = "success" if result.info["success"] else "timeout"
            print(
                f"seed={seed} {status} steps={observation['step_count']} "
                f"selected={task.object_key} target={task.target_key}"
            )
            end_time = time.time() + args.hold_seconds
            while viewer.is_running() and time.time() < end_time:
                viewer.sync()
                time.sleep(min(args.sleep or 0.01, 0.05))


if __name__ == "__main__":
    main()
