from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_manipulation_pi0.sim import (
    MultiObjectPickPlaceEnvironment,
    PickPlaceConfig,
    ScriptedOraclePolicy,
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
        description="Compare a small VLA rollout with the oracle on the same physical state."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/small_vla_strict_v7.pt"),
    )
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--object",
        choices=tuple(scene_object.key for scene_object in VLA_SCENE_OBJECTS),
        default="red_cube",
    )
    parser.add_argument(
        "--target",
        choices=tuple(scene_target.key for scene_target in VLA_SCENE_TARGETS),
        default="green_plate",
    )
    parser.add_argument("--max-policy-steps", type=int, default=700)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_policy_steps <= 0 or args.print_every <= 0:
        raise ValueError("Step counts must be positive.")
    policy = SmallVLAPolicy.load(args.checkpoint, device=args.device)
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=tuple(scene_object.key for scene_object in VLA_SCENE_OBJECTS),
        workspace_size=0.5,
        max_steps=args.max_policy_steps * policy.control_repeat,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    task = task_for_object(args.object, variant=0, target_key=args.target)
    privileged_observation = environment.reset(
        seed=args.seed,
        object_key=task.object_key,
        target_key=task.target_key,
    )
    oracle = ScriptedOraclePolicy(environment, joint_command_step_limit=0.02)
    policy.reset()
    specs_by_key = {spec.key: spec for spec in VLA_CAMERA_SPECS}
    camera_specs = tuple(
        CameraSpec(key, specs_by_key[key].camera_name) for key in policy.camera_keys
    )
    last_policy_stage: str | None = None
    result = None

    print(
        "step policy_stage  oracle_stage  held aligned  ee_object_xy  "
        "object_target_xy  action_mae nearest_target"
    )
    with MujocoCameraRig(environment.model, camera_specs) as cameras:
        for policy_step in range(args.max_policy_steps):
            observation = capture_vla_observation(
                environment.model,
                environment.data,
                cameras,
                task.instruction,
            )
            policy_action, policy_stage = policy.predict(observation)
            oracle_action = np.asarray(oracle.act(privileged_observation), dtype=np.float32)
            ee_position = np.asarray(privileged_observation["ee_pos"], dtype=float)
            object_position = np.asarray(privileged_observation["object_pos"], dtype=float)
            xy_error = float(np.linalg.norm(ee_position[:2] - object_position[:2]))
            target_position = np.asarray(privileged_observation["target_pos"], dtype=float)
            object_target_error = float(
                np.linalg.norm(object_position[:2] - target_position[:2])
            )
            target_positions = environment.scene_target_positions
            nearest_target = min(
                target_positions,
                key=lambda key: float(
                    np.linalg.norm(object_position[:2] - target_positions[key][:2])
                ),
            )
            action_mae = float(np.mean(np.abs(policy_action - oracle_action)))
            if policy_step % args.print_every == 0 or policy_stage != last_policy_stage:
                print(
                    f"{policy_step:4d} {policy_stage:13s} {oracle.stage:13s} "
                    f"{str(bool(privileged_observation['held'])):5s} "
                    f"{str(policy.target_aligned):7s} {xy_error:12.4f} "
                    f"{object_target_error:16.4f} {action_mae:11.4f} {nearest_target}"
                )
            last_policy_stage = policy_stage

            for _ in range(policy.control_repeat):
                result = environment.step(policy_action)
                privileged_observation = result.observation
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
        raise RuntimeError("Diagnostic rollout executed no actions.")
    status = "success" if result.info["success"] else "timeout"
    print(
        f"Final: {status}; simulation_steps={result.observation['step_count']}; "
        f"policy_stage={last_policy_stage}; oracle_stage={oracle.stage}; "
        f"held={result.info['held']}; contacts={result.info['finger_contacts']}"
    )


if __name__ == "__main__":
    main()
