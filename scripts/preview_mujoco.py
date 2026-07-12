from __future__ import annotations

import mujoco.viewer

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


def main() -> None:
    config = PickPlaceConfig(robot_name="franka_panda", object_names=("cube",), workspace_size=0.5)
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)
    observation = environment.reset(seed=0)
    with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
        while viewer.is_running():
            action = policy.act(observation)
            result = environment.step(action)
            observation = result.observation
            viewer.sync()
            if result.terminated or result.truncated:
                observation = environment.reset(seed=0)


if __name__ == "__main__":
    main()
