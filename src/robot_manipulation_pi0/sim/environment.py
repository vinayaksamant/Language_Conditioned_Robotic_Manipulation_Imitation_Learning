from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping

import mujoco
import numpy as np

from .config import PickPlaceConfig


@dataclass(frozen=True)
class StepResult:
    observation: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


class PickPlaceEnvironment:
    def __init__(self, config: PickPlaceConfig) -> None:
        self.config = config
        model_path = resources.files(__package__).joinpath("assets", config.model_file)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng()
        self._step_count = 0
        self._held = False
        self._target_position = np.zeros(3, dtype=float)
        self._arm_joint_names = (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        )
        self._arm_joint_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._arm_joint_names
        )
        self._arm_dof_indices = tuple(int(self.model.jnt_dofadr[joint_id]) for joint_id in self._arm_joint_ids)
        self._object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self._target_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        self._object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self._gripper_ctrl_index = 7

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._held = False
        self._set_arm_configuration(np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float))
        self._set_finger_configuration(0.04)
        half = self.config.workspace_size / 2.0
        workspace_center = np.array([0.5, 0.0], dtype=float)
        object_position = np.array(
            [
                workspace_center[0] + self._rng.uniform(-half * 0.55, half * 0.55),
                workspace_center[1] + self._rng.uniform(-half * 0.55, half * 0.55),
                0.245,
            ],
            dtype=float,
        )
        target_position = np.array(
            [
                workspace_center[0] + self._rng.uniform(-half * 0.55, half * 0.55),
                workspace_center[1] + self._rng.uniform(-half * 0.55, half * 0.55),
                0.222,
            ],
            dtype=float,
        )
        self._set_free_joint_pose(self._object_joint_id, object_position)
        self._set_free_joint_pose(self._target_joint_id, target_position)
        self._target_position = target_position.copy()
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def step(self, action: Any) -> StepResult:
        vector = np.asarray(action, dtype=float).reshape(-1)
        if vector.size < self.config.action_dimension:
            padded = np.zeros(self.config.action_dimension, dtype=float)
            padded[: vector.size] = vector
            vector = padded
        arm_dof = len(self._arm_joint_ids)
        arm_command = np.clip(vector[:arm_dof], -1.0, 1.0) * self.config.arm_velocity_scale
        gripper_command = float(np.clip(vector[arm_dof], -1.0, 1.0))
        control_dt = self.model.opt.timestep * self.config.substeps
        joint_positions = np.array(
            [self.data.qpos[int(self.model.jnt_qposadr[joint_id])] for joint_id in self._arm_joint_ids],
            dtype=float,
        )
        ctrl_min = self.model.actuator_ctrlrange[:arm_dof, 0]
        ctrl_max = self.model.actuator_ctrlrange[:arm_dof, 1]
        self.data.ctrl[:arm_dof] = np.clip(joint_positions + arm_command * control_dt, ctrl_min, ctrl_max)
        self.data.ctrl[self._gripper_ctrl_index] = 0.0 if gripper_command >= 0.0 else 0.04
        if self._held:
            self._attach_object_to_hand()
        for _ in range(self.config.substeps):
            mujoco.mj_step(self.model, self.data)
            self._pin_target()
        ee_position = self._site_position(self._ee_site_id)
        object_position = self._body_position(self._object_body_id)
        if not self._held and gripper_command >= 0.5 and np.linalg.norm(ee_position - object_position) <= self.config.grasp_radius:
            self._held = True
        if self._held and gripper_command <= -0.5:
            self._held = False
        if self._held:
            self._attach_object_to_hand()
        self._step_count += 1
        terminated = self._success()
        truncated = self._step_count >= self.config.max_steps
        reward = 1.0 if terminated else -float(np.linalg.norm(self._body_position(self._object_body_id) - self._body_position(self._target_body_id)))
        return StepResult(self._observation(), reward, terminated, truncated, self._info())

    def _set_arm_configuration(self, positions: np.ndarray) -> None:
        for joint_id, value in zip(self._arm_joint_ids, positions, strict=True):
            address = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[address] = value
            dof_address = int(self.model.jnt_dofadr[joint_id])
            self.data.qvel[dof_address] = 0.0
        self.data.ctrl[: len(self._arm_joint_ids)] = positions

    def _set_finger_configuration(self, width: float) -> None:
        for name in ("finger_joint1", "finger_joint2"):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            address = int(self.model.jnt_qposadr[joint_id])
            dof_address = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[address] = width
            self.data.qvel[dof_address] = 0.0
        self.data.ctrl[self._gripper_ctrl_index] = width

    def _set_free_joint_pose(self, joint_id: int, position: np.ndarray) -> None:
        address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        dof_address = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _pin_target(self) -> None:
        self._set_free_joint_pose(self._target_joint_id, self._target_position)
        mujoco.mj_forward(self.model, self.data)

    def _attach_object_to_hand(self) -> None:
        ee_position = self._site_position(self._ee_site_id)
        self._set_free_joint_pose(self._object_joint_id, ee_position + np.array([0.0, 0.0, -0.015], dtype=float))
        mujoco.mj_forward(self.model, self.data)

    def _site_position(self, site_id: int) -> np.ndarray:
        return np.array(self.data.site_xpos[site_id], dtype=float)

    def _body_position(self, body_id: int) -> np.ndarray:
        return np.array(self.data.xpos[body_id], dtype=float)

    def _success(self) -> bool:
        distance = np.linalg.norm(self._body_position(self._object_body_id) - self._body_position(self._target_body_id))
        return bool(distance <= self.config.target_radius)

    def _observation(self) -> Mapping[str, Any]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ee_pos": self._site_position(self._ee_site_id),
            "object_pos": self._body_position(self._object_body_id),
            "target_pos": self._body_position(self._target_body_id),
            "held": self._held,
            "step_count": self._step_count,
        }

    def _info(self) -> Mapping[str, Any]:
        return {
            "success": self._success(),
            "held": self._held,
        }
