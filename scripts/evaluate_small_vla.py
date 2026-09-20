from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from robot_manipulation_pi0.sim import (
    MultiObjectPickPlaceEnvironment,
    PickPlaceConfig,
    VLA_SCENE_OBJECTS,
    VLA_SCENE_TARGETS,
)
from robot_manipulation_pi0.vla import (
    VLA_CAMERA_SPECS,
    CameraSpec,
    MujocoCameraRig,
    capture_vla_observation,
    task_for_object,
)
from robot_manipulation_pi0.vla.small_vla import SmallVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a compact VLA checkpoint on unseen MuJoCo scene seeds."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/small_vla_strict_v7.pt"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=0,
        help="Policy decisions per episode; zero selects a checkpoint-appropriate limit.",
    )
    parser.add_argument(
        "--control-repeat",
        type=int,
        default=0,
        help="Simulator steps per prediction; zero uses the checkpoint training horizon.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_policy_steps < 0 or args.control_repeat < 0:
        raise ValueError("Episodes must be positive and control values cannot be negative.")
    policy = SmallVLAPolicy.load(args.checkpoint, device=args.device)
    control_repeat = args.control_repeat or policy.control_repeat
    if control_repeat != policy.control_repeat:
        raise ValueError(
            f"Checkpoint requires --control-repeat {policy.control_repeat}, got {control_repeat}."
        )
    max_policy_steps = args.max_policy_steps or (200 if control_repeat > 1 else 700)
    task_pairs = tuple(
        (scene_object.key, scene_target.key)
        for scene_object in VLA_SCENE_OBJECTS
        for scene_target in VLA_SCENE_TARGETS
    )
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=tuple(scene_object.key for scene_object in VLA_SCENE_OBJECTS),
        workspace_size=0.5,
        max_steps=max_policy_steps * control_repeat,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    specs_by_key = {spec.key: spec for spec in VLA_CAMERA_SPECS}
    camera_specs = tuple(
        CameraSpec(key, specs_by_key[key].camera_name) for key in policy.camera_keys
    )
    successes: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    grounding_successes: Counter[str] = Counter()

    mode = "strict" if policy.strict_vla_inference else "legacy hybrid"
    print(f"Inference mode: {mode}")
    print(f"Privileged rollout feedback: {policy.uses_privileged_feedback}")

    with MujocoCameraRig(environment.model, camera_specs) as cameras:
        for episode_index in range(args.episodes):
            object_key, target_key = task_pairs[episode_index % len(task_pairs)]
            task = task_for_object(object_key, variant=episode_index, target_key=target_key)
            seed = args.seed_start + episode_index
            environment.reset(seed=seed, object_key=object_key, target_key=target_key)
            initial_object_positions = {
                key: value.copy() for key, value in environment.scene_object_positions.items()
            }
            policy.reset()
            result = None
            final_stage = "unknown"
            for policy_step in range(max_policy_steps):
                observation = capture_vla_observation(
                    environment.model,
                    environment.data,
                    cameras,
                    task.instruction,
                )
                action, final_stage = policy.predict(observation)
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
                    if result.terminated or result.truncated:
                        break
                if result.terminated or result.truncated:
                    break
            if result is None:
                raise RuntimeError("Small VLA evaluation executed no simulation action.")
            label = task.canonical_label
            success = bool(result.info["success"])
            task_prediction = policy.task_prediction
            if task_prediction is None:
                raise RuntimeError("Small VLA policy did not produce a task prediction.")
            grounded_correctly = (
                task_prediction.object_key == object_key
                and task_prediction.target_key == target_key
            )
            attempts[label] += 1
            successes[label] += int(success)
            grounding_successes[label] += int(grounded_correctly)
            status = "success" if success else "timeout"
            outcome = _describe_physical_outcome(environment, initial_object_positions)
            print(
                f"Episode {episode_index + 1}/{args.episodes}: {status} seed={seed} "
                f"task={label} grounded={task_prediction.canonical_label} "
                f"confidence={task_prediction.confidence:.3f} outcome={outcome} stage={final_stage} "
                f"steps={result.observation['step_count']}"
            )

    total_successes = sum(successes.values())
    print(f"Overall: {total_successes}/{args.episodes} ({100.0 * total_successes / args.episodes:.1f}%)")
    total_grounded = sum(grounding_successes.values())
    print(
        f"Language grounding: {total_grounded}/{args.episodes} "
        f"({100.0 * total_grounded / args.episodes:.1f}%)"
    )
    for label in sorted(attempts):
        print(
            f"{label}: {successes[label]}/{attempts[label]} "
            f"({100.0 * successes[label] / attempts[label]:.1f}%)"
        )


def _describe_physical_outcome(
    environment: MultiObjectPickPlaceEnvironment,
    initial_positions: dict[str, np.ndarray],
) -> str:
    final_positions = environment.scene_object_positions
    movement = {
        key: float(np.linalg.norm(final_positions[key] - initial_positions[key]))
        for key in final_positions
    }
    moved_object = max(movement, key=movement.__getitem__)
    if movement[moved_object] < 0.02:
        return "none"
    target_positions = environment.scene_target_positions
    nearest_target = min(
        target_positions,
        key=lambda key: float(
            np.linalg.norm(final_positions[moved_object][:2] - target_positions[key][:2])
        ),
    )
    return f"{moved_object}->{nearest_target}"


if __name__ == "__main__":
    main()
