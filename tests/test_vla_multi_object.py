import json

import mujoco
import numpy as np
import pytest

from robot_manipulation_pi0.sim import (
    MultiObjectPickPlaceEnvironment,
    PickPlaceConfig,
    ScriptedOraclePolicy,
)
from robot_manipulation_pi0.vla import (
    VLA_CAMERA_SPECS,
    VLA_ORACLE_JOINT_STEP_LIMIT,
    CameraSpec,
    collect_vla_episode,
    resolve_task_instruction,
    save_vla_episode,
    task_for_object,
    validate_vla_episode_file,
)


class FakeCameraRig:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.specs = tuple(
            CameraSpec(spec.key, spec.camera_name, width=16, height=16)
            for spec in VLA_CAMERA_SPECS
        )

    def capture(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        del data
        return {
            spec.key: np.full((spec.height, spec.width, 3), index * 40, dtype=np.uint8)
            for index, spec in enumerate(self.specs)
        }


def test_language_instruction_resolves_selected_object() -> None:
    red_task = resolve_task_instruction("Pick the red block and put it on the yellow plate.")
    blue_task = resolve_task_instruction("Move the tall blue cylinder to the green plate.")
    assert red_task.object_key == "red_cube"
    assert red_task.target_key == "yellow_plate"
    assert blue_task.object_key == "blue_cylinder"
    assert blue_task.target_key == "green_plate"
    assert (
        resolve_task_instruction("Place the red cube on a surface other than the green plate.").target_key
        == "yellow_plate"
    )
    with pytest.raises(ValueError, match="identify exactly one"):
        resolve_task_instruction("Move an object to the plate.")
    with pytest.raises(ValueError, match="must name one destination"):
        resolve_task_instruction("Pick the red cube.", require_explicit_target=True)


def test_multi_object_scene_has_three_vla_cameras_and_two_objects() -> None:
    environment = _environment()
    observation = environment.reset(seed=0, object_key="blue_cylinder")

    assert observation["active_object_key"] == "blue_cylinder"
    assert observation["active_target_key"] == "green_plate"
    assert set(observation["scene_object_pos"]) == {"red_cube", "blue_cylinder"}
    assert set(observation["scene_target_pos"]) == {"green_plate", "yellow_plate"}
    assert observation["scene_object_pos"]["blue_cylinder"][2] > observation["scene_object_pos"]["red_cube"][2]
    for spec in VLA_CAMERA_SPECS:
        camera_id = mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_CAMERA, spec.camera_name)
        assert camera_id >= 0


@pytest.mark.parametrize(
    ("object_key", "target_key"),
    [
        ("red_cube", "green_plate"),
        ("red_cube", "yellow_plate"),
        ("blue_cylinder", "green_plate"),
        ("blue_cylinder", "yellow_plate"),
    ],
)
def test_smooth_vla_oracle_physically_places_selected_object(
    object_key: str,
    target_key: str,
) -> None:
    environment = _environment()
    policy = ScriptedOraclePolicy(
        environment,
        joint_command_step_limit=VLA_ORACLE_JOINT_STEP_LIMIT,
    )
    observation = environment.reset(seed=0, object_key=object_key, target_key=target_key)

    for _ in range(environment.config.max_steps):
        result = environment.step(policy.act(observation))
        observation = result.observation
        if result.terminated or result.truncated:
            break

    assert result.info["success"]
    assert result.info["active_object_key"] == object_key
    assert result.info["active_target_key"] == target_key
    assert result.info["has_held_object"]
    assert result.info["has_lifted_object"]
    assert result.info["has_released_after_lift"]


def test_second_task_preserves_scene_and_moves_same_object_to_another_target() -> None:
    environment = _environment()
    policy = ScriptedOraclePolicy(
        environment,
        joint_command_step_limit=VLA_ORACLE_JOINT_STEP_LIMIT,
    )
    observation = environment.reset(seed=100, object_key="red_cube", target_key="green_plate")
    first_result = _run_policy(environment, policy, observation)
    first_placement = np.asarray(first_result.observation["object_pos"], dtype=float).copy()
    _return_home(environment)

    observation = environment.begin_task("red_cube", "yellow_plate")
    policy.reset()
    second_result = _run_policy(environment, policy, observation)
    second_placement = np.asarray(second_result.observation["object_pos"], dtype=float)

    assert first_result.info["success"]
    assert second_result.info["success"]
    assert np.linalg.norm((second_placement - first_placement)[:2]) > 0.1
    assert second_result.info["active_target_key"] == "yellow_plate"


def test_saved_vla_episode_has_aligned_multiview_data_without_coordinates(tmp_path) -> None:
    environment = _environment()
    cameras = FakeCameraRig(environment.model)
    task = task_for_object("red_cube", target_key="yellow_plate")
    episode = collect_vla_episode(
        seed=0,
        config=environment.config,
        task=task,
        environment=environment,
        cameras=cameras,  # type: ignore[arg-type]
        record_every=8,
    )
    path = tmp_path / "episode.npz"
    save_vla_episode(path, episode)
    metadata = validate_vla_episode_file(path, ("top", "side", "wrist"))

    assert metadata["task"]["instruction"] == task.instruction
    assert metadata["record_every"] == 8
    assert episode.success
    with np.load(path, allow_pickle=False) as data:
        assert data["observation.state"].shape[0] == data["action"].shape[0]
        assert data["observation.images.side"].shape[0] == data["action"].shape[0]
        assert not any("object_pos" in key or "target_pos" in key for key in data.files)
        json.loads(str(data["metadata"]))


def _environment() -> MultiObjectPickPlaceEnvironment:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("red_cube", "blue_cylinder"),
        workspace_size=0.5,
        max_steps=500,
    )
    return MultiObjectPickPlaceEnvironment(config)


def _run_policy(
    environment: MultiObjectPickPlaceEnvironment,
    policy: ScriptedOraclePolicy,
    observation: dict[str, object],
):
    for _ in range(environment.config.max_steps):
        result = environment.step(policy.act(observation))
        observation = dict(result.observation)
        if result.terminated or result.truncated:
            return result
    raise AssertionError("Policy did not terminate or truncate.")


def _return_home(environment: MultiObjectPickPlaceEnvironment) -> None:
    for _ in range(environment.config.max_steps):
        if environment.arm_is_home():
            return
        environment.step(environment.home_action())
    raise AssertionError("Robot did not return home.")
