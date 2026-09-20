from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer

from robot_manipulation_pi0.sim import MultiObjectPickPlaceEnvironment, PickPlaceConfig
from robot_manipulation_pi0.vla import (
    VLA_CAMERA_SPECS,
    CameraSpec,
    MujocoCameraRig,
    capture_vla_observation,
)
from robot_manipulation_pi0.vla.small_vla import SmallVLAPolicy


HELP_TEXT = """Commands:
  Pick the red cube and place it on the green plate.
  Pick the red cube and place it on the yellow plate.
  Pick the blue cylinder and place it on the green plate.
  Pick the taller object and place it on the yellow plate.
  reset  - randomize the complete scene
  help   - show these examples
  quit   - close the simulator
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute repeated camera-based language tasks with the compact VLA policy."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/small_vla_strict_v7.pt"),
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=0,
        help="Policy decisions per task; zero selects a checkpoint-appropriate limit.",
    )
    parser.add_argument(
        "--control-repeat",
        type=int,
        default=0,
        help="Simulator steps per prediction; zero uses the checkpoint training horizon.",
    )
    parser.add_argument("--sleep", type=float, default=0.025)
    parser.add_argument(
        "--minimum-task-confidence",
        type=float,
        default=0.55,
        help="Reject uncertain learned language grounding instead of executing a random task.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_policy_steps < 0 or args.control_repeat < 0:
        raise ValueError("Policy and control step counts cannot be negative.")
    if args.sleep < 0.0:
        raise ValueError("--sleep cannot be negative.")
    if not 0.0 <= args.minimum_task_confidence <= 1.0:
        raise ValueError("--minimum-task-confidence must be between zero and one.")
    policy = SmallVLAPolicy.load(args.checkpoint, device=args.device)
    control_repeat = args.control_repeat or policy.control_repeat
    if control_repeat != policy.control_repeat:
        raise ValueError(
            f"Checkpoint requires --control-repeat {policy.control_repeat}, got {control_repeat}."
        )
    max_policy_steps = args.max_policy_steps or (200 if control_repeat > 1 else 700)
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("red_cube", "blue_cylinder"),
        workspace_size=0.5,
        max_steps=max_policy_steps * control_repeat,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    environment.reset(seed=args.seed)
    specs_by_key = {spec.key: spec for spec in VLA_CAMERA_SPECS}
    camera_specs = tuple(
        CameraSpec(key, specs_by_key[key].camera_name) for key in policy.camera_keys
    )
    current_seed = args.seed

    print(f"Controller: compact VLA on {policy.device}")
    print(f"Strict VLA inference: {policy.strict_vla_inference}")
    print(f"Privileged rollout feedback: {policy.uses_privileged_feedback}")
    print(f"Policy cameras: {', '.join(policy.camera_keys)}")
    print(f"Control horizon: {control_repeat} simulator steps per prediction")
    print(HELP_TEXT)
    with MujocoCameraRig(environment.model, camera_specs) as cameras:
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
                    viewer.sync()
                    print(f"Scene reset with seed {current_seed}.")
                    continue
                policy.reset()
                try:
                    task = policy.ground_instruction(instruction)
                except (RuntimeError, ValueError) as error:
                    print(f"Cannot execute instruction: {error}")
                    continue
                if task.learned and task.confidence < args.minimum_task_confidence:
                    print(
                        "Cannot execute instruction: learned task grounding confidence "
                        f"{task.confidence:.3f} is below {args.minimum_task_confidence:.3f}."
                    )
                    continue
                environment.begin_task(task.object_key, task.target_key)

                grounding = "learned" if task.learned else "rule-based legacy"
                print(
                    f"Grounding ({grounding}): object={task.object_key} "
                    f"target={task.target_key} confidence={task.confidence:.3f}"
                )
                result = None
                last_stage: str | None = None
                for policy_step in range(max_policy_steps):
                    observation = capture_vla_observation(
                        environment.model,
                        environment.data,
                        cameras,
                        instruction,
                    )
                    action, predicted_stage = policy.predict(observation)
                    if predicted_stage != last_stage:
                        print(f"  policy_step={policy_step:3d} predicted_stage={predicted_stage}")
                        last_stage = predicted_stage
                    for _ in range(control_repeat):
                        result = environment.step(action)
                        if policy.uses_privileged_feedback:
                            policy.update_feedback(
                                held=bool(result.info["held"]),
                                finger_contacts=int(result.info["finger_contacts"]),
                                has_lifted_object=bool(result.info["has_lifted_object"]),
                                has_released_after_lift=bool(
                                    result.info["has_released_after_lift"]
                                ),
                            )
                        viewer.sync()
                        if args.sleep:
                            time.sleep(args.sleep)
                        if result.terminated or result.truncated or not viewer.is_running():
                            break
                    if result.terminated or result.truncated or not viewer.is_running():
                        break

                if result is None:
                    print("No policy action was executed.")
                    continue
                status = "success" if result.info["success"] else "timeout"
                print(
                    f"Finished: {status}, simulation_steps={result.observation['step_count']}, "
                    f"predicted_stage={last_stage}"
                )
                if result.info["success"]:
                    print("Returning robot to home position...")
                    for _ in range(config.max_steps):
                        if environment.arm_is_home() or not viewer.is_running():
                            break
                        environment.step(environment.home_action())
                        viewer.sync()
                        if args.sleep:
                            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
