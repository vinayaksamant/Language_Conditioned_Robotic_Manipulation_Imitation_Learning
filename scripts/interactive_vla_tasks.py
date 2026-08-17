from __future__ import annotations

import argparse
import time

import mujoco.viewer

from robot_manipulation_pi0.sim import MultiObjectPickPlaceEnvironment, PickPlaceConfig, ScriptedOraclePolicy
from robot_manipulation_pi0.vla import VLA_ORACLE_JOINT_STEP_LIMIT, resolve_task_instruction


HELP_TEXT = """Commands:
  Pick the red cube and place it on the green plate.
  Pick the red cube and place it on the yellow plate.
  Place the red cube on a surface other than the green plate.
  Pick the blue cylinder and place it on the green plate.
  Pick the taller object and place it on the yellow plate.
  reset  - randomize the complete scene
  help   - show these examples
  quit   - close the simulator
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep MuJoCo open and execute multiple scripted language tasks in one persistent scene."
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.025)
    parser.add_argument("--hold-seconds", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.sleep < 0.0 or args.hold_seconds < 0.0:
        raise ValueError("--sleep and --hold-seconds cannot be negative.")

    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("red_cube", "blue_cylinder"),
        workspace_size=0.5,
        max_steps=args.max_steps,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    environment.reset(seed=args.seed)
    policy = ScriptedOraclePolicy(
        environment,
        joint_command_step_limit=VLA_ORACLE_JOINT_STEP_LIMIT,
    )
    current_seed = args.seed

    print("Controller: persistent scripted expert (not a trained pi0 policy)")
    print(HELP_TEXT)
    with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.55
        viewer.cam.lookat[:] = (0.50, 0.02, 0.32)
        viewer.sync()

        while viewer.is_running():
            try:
                instruction = input("instruction> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not instruction:
                continue
            command = instruction.lower()
            if command in {"quit", "exit", "q"}:
                break
            if command in {"help", "?"}:
                print(HELP_TEXT)
                continue
            if command == "reset":
                current_seed += 1
                environment.reset(seed=current_seed)
                policy.reset()
                viewer.sync()
                print(f"Scene reset with seed {current_seed}.")
                continue

            try:
                task = resolve_task_instruction(instruction, require_explicit_target=True)
                observation = environment.begin_task(task.object_key, task.target_key)
            except ValueError as error:
                print(f"Cannot execute instruction: {error}")
                continue

            policy.reset()
            print(f"Executing: object={task.object_key} target={task.target_key}")
            result = None
            last_stage: str | None = None
            for _ in range(config.max_steps):
                if not viewer.is_running():
                    return
                action = policy.act(observation)
                if policy.stage != last_stage:
                    print(
                        f"  step={observation['step_count']:3d} stage={policy.stage:13s} "
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
                raise RuntimeError("Interactive task did not execute any simulation steps.")
            status = "success" if result.info["success"] else "timeout"
            print(
                f"Finished: {status}, steps={observation['step_count']}, "
                f"object={task.object_key}, target={task.target_key}"
            )
            if result.info["success"]:
                print("Returning robot to home position...")
                for _ in range(config.max_steps):
                    if environment.arm_is_home():
                        break
                    environment.step(environment.home_action())
                    viewer.sync()
                    time.sleep(args.sleep)
                if not environment.arm_is_home():
                    print("Warning: robot did not fully reach home; use 'reset' before continuing.")
            end_time = time.time() + args.hold_seconds
            while viewer.is_running() and time.time() < end_time:
                viewer.sync()
                time.sleep(min(args.sleep or 0.01, 0.05))


if __name__ == "__main__":
    main()
