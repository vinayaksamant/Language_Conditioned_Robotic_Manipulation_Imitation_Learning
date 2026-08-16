import mujoco
import numpy as np

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment
from robot_manipulation_pi0.task_spec import DEFAULT_TASK_INSTRUCTION
from robot_manipulation_pi0.vla import (
    DEFAULT_CAMERA_SPECS,
    ROBOT_STATE_NAMES,
    MujocoCameraRig,
    capture_vla_observation,
)


class FakeCameraSource:
    def capture(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        del data
        return {
            "front": np.zeros((32, 32, 3), dtype=np.uint8),
            "wrist": np.full((32, 32, 3), 127, dtype=np.uint8),
        }


def test_model_contains_front_and_wrist_cameras() -> None:
    environment = _environment()

    for spec in DEFAULT_CAMERA_SPECS:
        camera_id = mujoco.mj_name2id(
            environment.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            spec.camera_name,
        )
        assert camera_id >= 0


def test_vla_observation_contains_robot_state_images_and_language() -> None:
    environment = _environment()
    environment.reset(seed=0)

    observation = capture_vla_observation(
        environment.model,
        environment.data,
        FakeCameraSource(),
        DEFAULT_TASK_INSTRUCTION.instruction,
    )
    values = observation.as_dict()

    assert observation.state.shape == (len(ROBOT_STATE_NAMES),)
    assert observation.state.dtype == np.float32
    assert values["task"] == DEFAULT_TASK_INSTRUCTION.instruction
    assert values["observation.images.front"].shape == (32, 32, 3)
    assert values["observation.images.wrist"].shape == (32, 32, 3)
    assert "object_pos" not in values
    assert "target_pos" not in values


def test_camera_rig_validates_named_cameras_without_rendering() -> None:
    environment = _environment()

    with MujocoCameraRig(environment.model) as cameras:
        assert tuple(spec.key for spec in cameras.specs) == ("front", "wrist")


def _environment() -> PickPlaceEnvironment:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
    )
    return PickPlaceEnvironment(config)
