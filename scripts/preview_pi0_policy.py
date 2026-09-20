from __future__ import annotations

import argparse
import time

import mujoco.viewer

from robot_manipulation_pi0.sim import MultiObjectPickPlaceEnvironment, PickPlaceConfig
from robot_manipulation_pi0.vla import (
    VLA_CAMERA_SPECS,
    CameraSpec,
    LeRobotPi0Policy,
    MujocoCameraRig,
    capture_vla_observation,
    resolve_task_instruction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fine-tuned LeRobot pi0 checkpoint in the multi-object MuJoCo scene."
    )
    parser.add_argument("--checkpoint", required=True, help="Local pretrained_model directory or HF model ID.")
    parser.add_argument(
        "--instruction",
        default="Pick the red cube and place it on the green plate.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-policy-steps", type=int, default=150)
    parser.add_argument(
        "--execute-actions",
        type=int,
        default=5,
        help="Actions from each predicted chunk to execute before replanning.",
    )
    parser.add_argument(
        "--control-repeat",
        type=int,
        default=4,
        help="40 Hz MuJoCo steps per 10 Hz dataset action.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--sleep", type=float, default=0.025)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_policy_steps <= 0 or args.execute_actions <= 0 or args.control_repeat <= 0:
        raise ValueError("Policy steps, executed chunk actions, and control repeat must be positive.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Camera width and height must be positive.")
    if args.sleep < 0.0:
        raise ValueError("--sleep cannot be negative.")

    task = resolve_task_instruction(args.instruction, require_explicit_target=True)
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("red_cube", "blue_cylinder"),
        workspace_size=0.5,
        max_steps=args.max_policy_steps * args.control_repeat,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    environment.reset(seed=args.seed, object_key=task.object_key, target_key=task.target_key)
    camera_specs = tuple(
        CameraSpec(spec.key, spec.camera_name, width=args.width, height=args.height)
        for spec in VLA_CAMERA_SPECS
    )

    print("Loading pi0 checkpoint. The first load can take several minutes.")
    policy = LeRobotPi0Policy(args.checkpoint, device=args.device)
    policy.reset()
    print(f"Instruction: {args.instruction}")
    print(f"Evaluation label: object={task.object_key} target={task.target_key}")
    print(
        f"Control: dataset rate={40 / args.control_repeat:g} Hz, "
        f"replan every {args.execute_actions} action(s)"
    )

    result = None
    policy_steps = 0
    with MujocoCameraRig(environment.model, camera_specs) as cameras:
        with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.55
            viewer.cam.lookat[:] = (0.50, 0.02, 0.32)
            viewer.sync()

            while viewer.is_running() and policy_steps < args.max_policy_steps:
                observation = capture_vla_observation(
                    environment.model,
                    environment.data,
                    cameras,
                    args.instruction,
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
                    f"Replan at policy step {policy_steps}: chunk={len(action_chunk)} "
                    f"inference={inference_seconds:.2f}s"
                )
                for action in action_chunk[:actions_to_execute].numpy():
                    for _ in range(args.control_repeat):
                        result = environment.step(action)
                        viewer.sync()
                        if args.sleep:
                            time.sleep(args.sleep)
                        if result.terminated or result.truncated or not viewer.is_running():
                            break
                    policy_steps += 1
                    if result is not None and (result.terminated or result.truncated):
                        break
                if result is not None and (result.terminated or result.truncated):
                    break

    if result is None:
        raise RuntimeError("No pi0 action was executed.")
    status = "success" if result.info["success"] else "timeout"
    print(
        f"Rollout: {status} policy_steps={policy_steps} simulation_steps="
        f"{result.observation['step_count']} held={result.info['held']}"
    )


if __name__ == "__main__":
    main()
