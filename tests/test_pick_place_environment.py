import pytest

pytest.importorskip("mujoco")

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy


def test_pick_place_environment_step() -> None:
    config = PickPlaceConfig(robot_name="franka_panda", object_names=("cube",), workspace_size=0.5)
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)
    observation = environment.reset(seed=0)
    action = policy.act(observation)
    result = environment.step(action)
    assert "ee_pos" in result.observation
    assert "object_pos" in result.observation
    assert "target_pos" in result.observation
