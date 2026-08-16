from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import mujoco
import numpy as np


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
ROBOT_STATE_NAMES = ARM_JOINT_NAMES + ("gripper_width",)


class CameraSource(Protocol):
    def capture(self, data: mujoco.MjData) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class VLAObservation:
    state: np.ndarray
    images: Mapping[str, np.ndarray]
    instruction: str

    def as_dict(self) -> dict[str, np.ndarray | str]:
        observation: dict[str, np.ndarray | str] = {
            "observation.state": self.state.copy(),
            "task": self.instruction,
        }
        for camera_key, image in self.images.items():
            observation[f"observation.images.{camera_key}"] = image.copy()
        return observation


def robot_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    arm_positions = [_joint_position(model, data, name) for name in ARM_JOINT_NAMES]
    finger_positions = [_joint_position(model, data, name) for name in FINGER_JOINT_NAMES]
    gripper_width = float(np.mean(finger_positions))
    return np.asarray([*arm_positions, gripper_width], dtype=np.float32)


def capture_vla_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cameras: CameraSource,
    instruction: str,
) -> VLAObservation:
    if not instruction.strip():
        raise ValueError("VLA task instruction cannot be empty.")
    return VLAObservation(
        state=robot_state(model, data),
        images=cameras.capture(data),
        instruction=instruction,
    )


def _joint_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint {name!r} does not exist in the model.")
    return float(data.qpos[int(model.jnt_qposadr[joint_id])])
