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


def test_object_at_target_is_not_success_without_pick_sequence() -> None:
    config = PickPlaceConfig(robot_name="franka_panda", object_names=("cube",), workspace_size=0.5)
    environment = PickPlaceEnvironment(config)
    observation = environment.reset(seed=0)
    target_position = np.asarray(observation["target_pos"], dtype=float)
    object_position = target_position.copy()
    object_position[2] = np.asarray(observation["object_pos"], dtype=float)[2]

    environment._set_free_joint_pose(environment._object_joint_id, object_position)
    result = environment.step(np.zeros(config.action_dimension, dtype=float))

    assert not result.terminated
    assert not result.info["success"]
    assert not result.info["has_held_object"]
    assert not result.info["has_lifted_object"]
    assert not result.info["has_released_after_lift"]
    assert not result.info["has_retreated_after_release"]


def test_reset_keeps_object_and_target_separated() -> None:
    config = PickPlaceConfig(robot_name="franka_panda", object_names=("cube",), workspace_size=0.5)
    environment = PickPlaceEnvironment(config)

    for seed in range(10):
        observation = environment.reset(seed=seed)
        object_position = np.asarray(observation["object_pos"], dtype=float)
        target_position = np.asarray(observation["target_pos"], dtype=float)
        planar_distance = np.linalg.norm((object_position - target_position)[:2])
        assert planar_distance >= config.min_initial_object_target_distance


def test_scripted_oracle_completes_easy_pick_place_rollout() -> None:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=450,
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
    assert result.info["has_held_object"]
    assert result.info["has_lifted_object"]
    assert result.info["has_released_after_lift"]
    assert result.info["has_retreated_after_release"]
    assert not result.info["held"]
    assert result.info["finger_contacts"] == 0


def test_scripted_oracle_only_marks_held_with_both_finger_contacts() -> None:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=200,
    )
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)
    observation = environment.reset(seed=0)

    for _ in range(config.max_steps):
        result = environment.step(policy.act(observation))
        observation = result.observation
        if observation["held"]:
            break

    assert observation["held"]
    assert observation["finger_contacts"] == 2
