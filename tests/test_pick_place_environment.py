import pytest

pytest.importorskip("mujoco")

import numpy as np

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


def test_pick_place_environment_step() -> None:
    config = PickPlaceConfig(robot_name="franka_panda", object_names=("cube",), workspace_size=0.5)
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)
    observation = environment.reset(seed=0)
    action = policy.act(observation)
    result = environment.step(action)
    assert len(action) == config.action_dimension
    assert "ee_pos" in result.observation
    assert "object_pos" in result.observation
    assert "target_pos" in result.observation


def test_target_zone_stays_fixed_during_simulation() -> None:
    config = PickPlaceConfig(robot_name="franka_panda", object_names=("cube",), workspace_size=0.5)
    environment = PickPlaceEnvironment(config)
    observation = environment.reset(seed=0)
    initial_target = np.asarray(observation["target_pos"], dtype=float)

    for _ in range(25):
        result = environment.step(np.zeros(config.action_dimension, dtype=float))

    final_target = np.asarray(result.observation["target_pos"], dtype=float)
    np.testing.assert_allclose(final_target, initial_target, atol=1e-8)


def test_scripted_oracle_completes_easy_pick_place_rollout() -> None:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=150,
    )
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)
    observation = environment.reset(seed=0)

    for _ in range(config.max_steps):
        result = environment.step(policy.act(observation))
        observation = result.observation
        if result.terminated or result.truncated:
            break

    assert result.terminated
    assert result.info["success"]
