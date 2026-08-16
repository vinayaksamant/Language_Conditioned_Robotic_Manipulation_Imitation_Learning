from __future__ import annotations

import argparse

import numpy as np

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print state-machine rollout positions for debugging.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=args.max_steps,
    )

    for seed in args.seeds:
        environment = PickPlaceEnvironment(config)
        policy = ScriptedOraclePolicy(environment)
        observation = environment.reset(seed=seed)
        last_stage: str | None = None

        print(f"\nSEED {seed}")
        print(
            "step stage         held contacts  ee_xyz                 object_xyz             "
            "target_xyz             ee_obj  obj_target_xy"
        )

        for _ in range(config.max_steps):
            result = environment.step(policy.act(observation))
            observation = result.observation
            stage = policy._stage
            ee_position = np.asarray(observation["ee_pos"], dtype=float)
            object_position = np.asarray(observation["object_pos"], dtype=float)
            target_position = np.asarray(observation["target_pos"], dtype=float)
            ee_object_distance = float(np.linalg.norm(ee_position - object_position))
            object_target_distance = float(np.linalg.norm((object_position - target_position)[:2]))

            should_print = (
                stage != last_stage
                or int(observation["step_count"]) % args.print_every == 0
                or result.terminated
                or result.truncated
            )
            if should_print:
                print(
                    f"{int(observation['step_count']):4d} "
                    f"{stage:13s} "
                    f"{str(observation['held']):5s} "
                    f"{int(observation['finger_contacts']):8d} "
                    f"[{ee_position[0]: .3f} {ee_position[1]: .3f} {ee_position[2]: .3f}] "
                    f"[{object_position[0]: .3f} {object_position[1]: .3f} {object_position[2]: .3f}] "
                    f"[{target_position[0]: .3f} {target_position[1]: .3f} {target_position[2]: .3f}] "
                    f"{ee_object_distance: .4f} "
                    f"{object_target_distance: .4f}"
                )
                last_stage = stage

            if result.terminated or result.truncated:
                status = "success" if result.terminated else "timeout"
                print(f"final {status}: {dict(result.info)}")
                break


if __name__ == "__main__":
    main()
