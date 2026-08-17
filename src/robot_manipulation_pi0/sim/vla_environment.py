from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import mujoco
import numpy as np

from .config import PickPlaceConfig
from .environment import PickPlaceEnvironment


VLA_MODEL_FILE = "franka_emika_panda/vla_pick_place_scene.xml"
VLA_HOME_ARM_CONFIGURATION = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
    dtype=float,
)


@dataclass(frozen=True)
class SceneObject:
    key: str
    display_name: str
    body_name: str
    joint_name: str
    geom_name: str
    half_height: float

    @property
    def resting_height(self) -> float:
        return 0.22 + self.half_height


@dataclass(frozen=True)
class SceneTarget:
    key: str
    display_name: str
    body_name: str


VLA_SCENE_OBJECTS = (
    SceneObject(
        key="red_cube",
        display_name="red cube",
        body_name="object",
        joint_name="object_free",
        geom_name="cube",
        half_height=0.025,
    ),
    SceneObject(
        key="blue_cylinder",
        display_name="tall blue cylinder",
        body_name="blue_cylinder",
        joint_name="blue_cylinder_free",
        geom_name="blue_cylinder_geom",
        half_height=0.035,
    ),
)
VLA_SCENE_OBJECT_BY_KEY = {scene_object.key: scene_object for scene_object in VLA_SCENE_OBJECTS}
VLA_SCENE_TARGETS = (
    SceneTarget(key="green_plate", display_name="green plate", body_name="target"),
    SceneTarget(key="yellow_plate", display_name="yellow plate", body_name="yellow_target"),
)
VLA_SCENE_TARGET_BY_KEY = {scene_target.key: scene_target for scene_target in VLA_SCENE_TARGETS}


class MultiObjectPickPlaceEnvironment(PickPlaceEnvironment):
    """Physical pick-place scene with one language-selected object and one distractor."""

    def __init__(self, config: PickPlaceConfig) -> None:
        super().__init__(replace(config, model_file=VLA_MODEL_FILE))
        self._scene_object_ids = {
            scene_object.key: (
                self._required_id(mujoco.mjtObj.mjOBJ_BODY, scene_object.body_name),
                self._required_id(mujoco.mjtObj.mjOBJ_JOINT, scene_object.joint_name),
                self._required_id(mujoco.mjtObj.mjOBJ_GEOM, scene_object.geom_name),
            )
            for scene_object in VLA_SCENE_OBJECTS
        }
        self._scene_target_ids = {}
        for scene_target in VLA_SCENE_TARGETS:
            body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, scene_target.body_name)
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                raise ValueError(f"VLA target {scene_target.body_name!r} must be a MuJoCo mocap body.")
            self._scene_target_ids[scene_target.key] = (body_id, mocap_id)
        self._active_object_key = VLA_SCENE_OBJECTS[0].key
        self._active_target_key = VLA_SCENE_TARGETS[0].key
        self._set_active_object(self._active_object_key)
        self._set_active_target(self._active_target_key)

    @property
    def active_object_key(self) -> str:
        return self._active_object_key

    @property
    def active_target_key(self) -> str:
        return self._active_target_key

    @property
    def scene_object_positions(self) -> Mapping[str, np.ndarray]:
        return {
            key: self._body_position(ids[0])
            for key, ids in self._scene_object_ids.items()
        }

    @property
    def scene_target_positions(self) -> Mapping[str, np.ndarray]:
        return {
            key: self._body_position(ids[0])
            for key, ids in self._scene_target_ids.items()
        }

    def reset(
        self,
        seed: int | None = None,
        object_key: str = "red_cube",
        target_key: str = "green_plate",
    ) -> Mapping[str, Any]:
        self._validate_task_keys(object_key, target_key)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._reset_task_tracking()
        self._set_arm_configuration(
            VLA_HOME_ARM_CONFIGURATION
        )
        self._set_finger_configuration(self._open_finger_width)
        self._set_active_object(object_key)
        self._set_active_target(target_key)

        positions = self._sample_scene_positions()
        for scene_object in VLA_SCENE_OBJECTS:
            _, joint_id, _ = self._scene_object_ids[scene_object.key]
            yaw = float(self._rng.uniform(-np.pi, np.pi))
            self._set_free_joint_pose_with_yaw(joint_id, positions[scene_object.key], yaw)

        for scene_target in VLA_SCENE_TARGETS:
            self._set_scene_target_pose(scene_target.key, positions[scene_target.key])

        active_spec = VLA_SCENE_OBJECT_BY_KEY[self._active_object_key]
        self._object_table_height = active_spec.resting_height
        self._target_position = positions[self._active_target_key].copy()
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def begin_task(self, object_key: str, target_key: str) -> Mapping[str, Any]:
        """Start another task without resetting the physical scene."""
        self._validate_task_keys(object_key, target_key)
        self._reset_task_tracking()
        self._set_active_object(object_key)
        self._set_active_target(target_key)
        self._object_table_height = VLA_SCENE_OBJECT_BY_KEY[object_key].resting_height
        self._target_position = self._body_position(self._target_body_id)
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def home_action(self, joint_step_limit: float = 0.02) -> np.ndarray:
        if joint_step_limit <= 0.0:
            raise ValueError("joint_step_limit must be positive.")
        current = np.asarray(
            [self.data.qpos[int(self.model.jnt_qposadr[joint_id])] for joint_id in self._arm_joint_ids],
            dtype=float,
        )
        delta = VLA_HOME_ARM_CONFIGURATION - current
        peak_delta = float(np.max(np.abs(delta)))
        target = VLA_HOME_ARM_CONFIGURATION
        if peak_delta > joint_step_limit:
            target = current + delta * (joint_step_limit / peak_delta)
        return self.arm_positions_to_action(target, gripper_command=-1.0)

    def arm_is_home(self, tolerance: float = 0.015) -> bool:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        current = np.asarray(
            [self.data.qpos[int(self.model.jnt_qposadr[joint_id])] for joint_id in self._arm_joint_ids],
            dtype=float,
        )
        return bool(np.max(np.abs(current - VLA_HOME_ARM_CONFIGURATION)) <= tolerance)

    def _set_active_object(self, object_key: str) -> None:
        self._active_object_key = object_key
        body_id, joint_id, geom_id = self._scene_object_ids[object_key]
        self._object_body_id = body_id
        self._object_joint_id = joint_id
        self._cube_geom_id = geom_id

    def _set_active_target(self, target_key: str) -> None:
        self._active_target_key = target_key
        self._target_body_id, self._target_mocap_id = self._scene_target_ids[target_key]

    def _reset_task_tracking(self) -> None:
        self._step_count = 0
        self._held = False
        self._has_held_object = False
        self._has_lifted_object = False
        self._has_released_after_lift = False
        self._has_retreated_after_release = False
        self._grasp_contact_count = 0

    def _validate_task_keys(self, object_key: str, target_key: str) -> None:
        if object_key not in VLA_SCENE_OBJECT_BY_KEY:
            choices = ", ".join(VLA_SCENE_OBJECT_BY_KEY)
            raise ValueError(f"Unknown VLA object {object_key!r}. Expected one of: {choices}.")
        if target_key not in VLA_SCENE_TARGET_BY_KEY:
            choices = ", ".join(VLA_SCENE_TARGET_BY_KEY)
            raise ValueError(f"Unknown VLA target {target_key!r}. Expected one of: {choices}.")

    def _sample_scene_positions(self) -> dict[str, np.ndarray]:
        slots = np.array(
            [
                [0.43, -0.05],
                [0.59, -0.05],
                [0.43, 0.11],
                [0.59, 0.11],
            ],
            dtype=float,
        )
        for _ in range(100):
            selected = slots[self._rng.permutation(len(slots))]
            selected += self._rng.uniform(-0.005, 0.005, size=selected.shape)
            red_xy, blue_xy, green_xy, yellow_xy = selected
            if abs(red_xy[0] - blue_xy[0]) < 0.14:
                continue
            if np.linalg.norm(red_xy - blue_xy) < 0.14:
                continue
            return {
                "red_cube": np.array([*red_xy, VLA_SCENE_OBJECT_BY_KEY["red_cube"].resting_height]),
                "blue_cylinder": np.array(
                    [*blue_xy, VLA_SCENE_OBJECT_BY_KEY["blue_cylinder"].resting_height]
                ),
                "green_plate": np.array([*green_xy, 0.222], dtype=float),
                "yellow_plate": np.array([*yellow_xy, 0.222], dtype=float),
            }

        raise RuntimeError("Could not sample separated positions from the VLA table slots.")

    def _set_free_joint_pose_with_yaw(self, joint_id: int, position: np.ndarray, yaw: float) -> None:
        address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
            dtype=float,
        )
        dof_address = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _set_scene_target_pose(self, target_key: str, position: np.ndarray) -> None:
        _, mocap_id = self._scene_target_ids[target_key]
        self.data.mocap_pos[mocap_id] = np.asarray(position, dtype=float)
        self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _observation(self) -> Mapping[str, Any]:
        observation = dict(super()._observation())
        observation["active_object_key"] = self._active_object_key
        observation["active_target_key"] = self._active_target_key
        observation["scene_object_pos"] = self.scene_object_positions
        observation["scene_target_pos"] = self.scene_target_positions
        return observation

    def _info(self) -> Mapping[str, Any]:
        info = dict(super()._info())
        info["active_object_key"] = self._active_object_key
        info["active_target_key"] = self._active_target_key
        return info

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object {name!r} is missing from the VLA scene.")
        return object_id
