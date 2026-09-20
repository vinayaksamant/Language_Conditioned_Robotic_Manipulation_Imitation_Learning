from __future__ import annotations

import argparse
from collections import Counter

from robot_manipulation_pi0.sim import (
    MultiObjectPickPlaceEnvironment,
    PickPlaceConfig,
    VLA_SCENE_OBJECTS,
    VLA_SCENE_TARGETS,
)
from robot_manipulation_pi0.vla import (
    VLA_CAMERA_SPECS,
    CameraSpec,
    LeRobotPi0Policy,
    MujocoCameraRig,
    capture_vla_observation,
    task_for_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned LeRobot pi0 checkpoint on unseen MuJoCo seeds."
    )
    parser.add_argument("--checkpoint", required=True, help="Local pretrained_model directory or HF model ID.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--max-policy-steps", type=int, default=150)
    parser.add_argument("--execute-actions", type=int, default=5)
    parser.add_argument("--control-repeat", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_values = (
        args.episodes,
        args.max_policy_steps,
        args.execute_actions,
        args.control_repeat,
        args.width,
        args.height,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("Episode, control, and camera values must be positive.")

    task_pairs = tuple(
        (scene_object.key, scene_target.key)
        for scene_object in VLA_SCENE_OBJECTS
        for scene_target in VLA_SCENE_TARGETS
    )
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=tuple(scene_object.key for scene_object in VLA_SCENE_OBJECTS),
        workspace_size=0.5,
        max_steps=args.max_policy_steps * args.control_repeat,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    camera_specs = tuple(
        CameraSpec(spec.key, spec.camera_name, width=args.width, height=args.height)
        for spec in VLA_CAMERA_SPECS
    )

    print("Loading pi0 checkpoint. The first load can take several minutes.")
    policy = LeRobotPi0Policy(args.checkpoint, device=args.device)
    successes: Counter[str] = Counter()
    attempts: Counter[str] = Counter()

    with MujocoCameraRig(environment.model, camera_specs) as cameras:
        for episode_index in range(args.episodes):
            object_key, target_key = task_pairs[episode_index % len(task_pairs)]
            task = task_for_object(object_key, variant=episode_index, target_key=target_key)
            seed = args.seed_start + episode_index
            environment.reset(seed=seed, object_key=object_key, target_key=target_key)
            policy.reset()
            result = None
            policy_steps = 0

            while policy_steps < args.max_policy_steps:
                observation = capture_vla_observation(
                    environment.model,
                    environment.data,
                    cameras,
                    task.instruction,
                )
                action_chunk = policy.predict_action_chunk(observation)
                if action_chunk.shape[1] != config.action_dimension:
                    raise RuntimeError(
                        f"Checkpoint action dimension is {action_chunk.shape[1]}, "
                        f"but the environment requires {config.action_dimension}."
                    )
                actions_to_execute = min(args.execute_actions, len(action_chunk))
                for action in action_chunk[:actions_to_execute].numpy():
                    for _ in range(args.control_repeat):
                        result = environment.step(action)
                        if result.terminated or result.truncated:
                            break
                    policy_steps += 1
                    if result is not None and (result.terminated or result.truncated):
                        break
                if result is not None and (result.terminated or result.truncated):
                    break

            if result is None:
                raise RuntimeError("No pi0 action was executed.")
            task_label = task.canonical_label
            success = bool(result.info["success"])
            attempts[task_label] += 1
            successes[task_label] += int(success)
            status = "success" if success else "timeout"
            print(
                f"Episode {episode_index + 1}/{args.episodes}: {status} seed={seed} "
                f"task={task_label} policy_steps={policy_steps} "
                f"simulation_steps={result.observation['step_count']}"
            )

    total_successes = sum(successes.values())
    print(f"Overall: {total_successes}/{args.episodes} ({100.0 * total_successes / args.episodes:.1f}%)")
    for task_label in sorted(attempts):
        task_successes = successes[task_label]
        task_attempts = attempts[task_label]
        print(
            f"{task_label}: {task_successes}/{task_attempts} "
            f"({100.0 * task_successes / task_attempts:.1f}%)"
        )


if __name__ == "__main__":
    main()
