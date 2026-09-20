from __future__ import annotations

import argparse
import time

import mujoco.viewer

from robot_manipulation_pi0.sim import MultiObjectPickPlaceEnvironment, PickPlaceConfig
from robot_manipulation_pi0.vla import (
    PI0_DATASET_CONTROL_REPEAT,
    VLA_CAMERA_SPECS,
    CameraSpec,
    LeRobotPi0Policy,
    MujocoCameraRig,
    capture_vla_observation,
)


HELP_TEXT = """Commands:
  Enter a natural-language manipulation instruction to send it directly to pi0.
  reset  - randomize the complete scene
  help   - show command help
  quit   - close the simulator
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep MuJoCo open and execute repeated tasks with a fine-tuned pi0 policy."
    )
    parser.add_argument("--checkpoint", required=True, help="Local pretrained_model directory or HF model ID.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-policy-steps", type=int, default=350)
    parser.add_argument(
        "--execute-actions",
        type=int,
        default=5,
        help="Actions from each predicted chunk to execute before capturing new images.",
    )
    parser.add_argument(
        "--control-repeat",
        type=int,
        default=PI0_DATASET_CONTROL_REPEAT,
        help="40 Hz MuJoCo steps per 20 Hz dataset action.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--sleep", type=float, default=0.025)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_values = (
        args.max_policy_steps,
        args.execute_actions,
        args.control_repeat,
        args.width,
        args.height,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("Policy, control, and camera values must be positive.")
    if args.sleep < 0.0:
        raise ValueError("--sleep cannot be negative.")

    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("red_cube", "blue_cylinder"),
        workspace_size=0.5,
        max_steps=args.max_policy_steps * args.control_repeat,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    environment.reset(seed=args.seed)
    camera_specs = tuple(
        CameraSpec(spec.key, spec.camera_name, width=args.width, height=args.height)
        for spec in VLA_CAMERA_SPECS
    )

    print("Loading pi0 checkpoint. The first load can take several minutes.")
    policy = LeRobotPi0Policy(args.checkpoint, device=args.device)
    current_seed = args.seed
    print("Controller: trained pi0 using camera images, robot state, and language")
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
                    policy.reset()
                    viewer.sync()
                    print(f"Scene reset with seed {current_seed}.")
                    continue

                environment.restart_rollout_tracking()
                policy.reset()
                print(f"Executing with learned pi0 policy: {instruction}")
                result = None
                policy_steps = 0
                while viewer.is_running() and policy_steps < args.max_policy_steps:
                    observation = capture_vla_observation(
                        environment.model,
                        environment.data,
                        cameras,
                        instruction,
                    )
                    inference_start = time.perf_counter()
                    action_chunk = policy.predict_action_chunk(observation)
                    inference_seconds = time.perf_counter() - inference_start
                    if action_chunk.shape[1] != config.action_dimension:
                        raise RuntimeError(
                            f"Checkpoint action dimension is {action_chunk.shape[1]}, "
                            f"but the environment requires {config.action_dimension}."
                        )

                    actions_to_execute = min(args.execute_actions, len(action_chunk))
                    print(
                        f"  replan={policy_steps:3d} chunk={len(action_chunk)} "
                        f"inference={inference_seconds:.2f}s"
                    )
                    for action in action_chunk[:actions_to_execute].numpy():
                        for _ in range(args.control_repeat):
                            result = environment.step(action)
                            viewer.sync()
                            if args.sleep:
                                time.sleep(args.sleep)
                            if result.truncated or not viewer.is_running():
                                break
                        policy_steps += 1
                        if result is not None and result.truncated:
                            break
                    if result is not None and result.truncated:
                        break

                if result is None:
                    print("No policy action was executed.")
                    continue
                print(
                    f"Finished learned-policy rollout: policy_steps={policy_steps}, "
                    f"simulation_steps={result.observation['step_count']}"
                )


if __name__ == "__main__":
    main()
