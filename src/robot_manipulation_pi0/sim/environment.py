from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping

import mujoco
import numpy as np

from .config import PickPlaceConfig


ACTION_MODE = "normalized_joint_position_v1"


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
        self._has_held_object = False
        self._has_lifted_object = False
        self._has_released_after_lift = False
        self._has_retreated_after_release = False
        self._grasp_contact_count = 0
        self._object_table_height = 0.245
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
        self._object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_id = int(self.model.body_mocapid[self._target_body_id])
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self._finger_pad_geom_ids = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad"),
        )
        self._gripper_ctrl_index = 7
        self._open_finger_width = 0.04
        self._grasp_finger_width = 0.018

        if self._target_mocap_id < 0:
            raise ValueError("The target body must be a MuJoCo mocap body.")

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._held = False
        self._has_held_object = False
        self._has_lifted_object = False
        self._has_released_after_lift = False
        self._has_retreated_after_release = False
        self._grasp_contact_count = 0
        self._set_arm_configuration(np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float))
        self._set_finger_configuration(self._open_finger_width)
        half = self.config.workspace_size / 2.0
        workspace_center = np.array([0.5, 0.03], dtype=float)
        reset_span = half * 0.35
        object_position = np.array(
            [
                workspace_center[0] + self._rng.uniform(-reset_span, reset_span),
                workspace_center[1] + self._rng.uniform(-reset_span, reset_span),
                0.245,
            ],
            dtype=float,
        )
        self._object_table_height = float(object_position[2])
        target_position = self._sample_target_position(workspace_center, reset_span, object_position)
        self._set_free_joint_pose(self._object_joint_id, object_position)
        self._set_target_pose(target_position)
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def step(self, action: Any) -> StepResult:
        vector = np.asarray(action, dtype=float).reshape(-1)
        if vector.size < self.config.action_dimension:
            padded = np.zeros(self.config.action_dimension, dtype=float)
            padded[: vector.size] = vector
            vector = padded
        arm_dof = len(self._arm_joint_ids)
        arm_command = np.clip(vector[:arm_dof], -1.0, 1.0)
        gripper_command = float(np.clip(vector[arm_dof], -1.0, 1.0))
        ctrl_min = self.model.actuator_ctrlrange[:arm_dof, 0]
        ctrl_max = self.model.actuator_ctrlrange[:arm_dof, 1]
        self.data.ctrl[:arm_dof] = ctrl_min + (arm_command + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        self.data.ctrl[self._gripper_ctrl_index] = self._gripper_width_from_command(gripper_command)

        was_held = self._held
        for _ in range(self.config.substeps):
            self.data.qfrc_applied[:] = 0.0
            arm_indices = np.asarray(self._arm_dof_indices, dtype=int)
            self.data.qfrc_applied[arm_indices] = self.data.qfrc_bias[arm_indices]
            mujoco.mj_step(self.model, self.data)

        contact_count = self._finger_contact_count()
        if gripper_command >= 0.5 and contact_count == len(self._finger_pad_geom_ids):
            self._grasp_contact_count += 1
        else:
            self._grasp_contact_count = 0
        self._held = bool(
            gripper_command >= 0.5
            and contact_count == len(self._finger_pad_geom_ids)
            and (was_held or self._grasp_contact_count >= self.config.grasp_contact_steps)
        )
        if self._held:
            self._has_held_object = True
        object_position = self._body_position(self._object_body_id)
        if self._held and object_position[2] >= self._object_table_height + self.config.object_lift_height:
            self._has_lifted_object = True
        if was_held and not self._held and gripper_command <= -0.5 and self._has_lifted_object:
            self._has_released_after_lift = True
        if self._has_released_after_lift and not self._held:
            retreat_height = self._object_table_height + 0.08
            if self._site_position(self._ee_site_id)[2] >= retreat_height:
                self._has_retreated_after_release = True
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

    def _gripper_width_from_command(self, command: float) -> float:
        normalized = (float(np.clip(command, -1.0, 1.0)) + 1.0) / 2.0
        return float(self._open_finger_width + normalized * (self._grasp_finger_width - self._open_finger_width))

    def arm_positions_to_action(self, joint_positions: np.ndarray, gripper_command: float) -> np.ndarray:
        positions = np.asarray(joint_positions, dtype=float)
        arm_dof = len(self._arm_joint_ids)
        if positions.shape != (arm_dof,):
            raise ValueError(f"Expected {arm_dof} arm joint positions, got shape {positions.shape}.")
        ctrl_min = self.model.actuator_ctrlrange[:arm_dof, 0]
        ctrl_max = self.model.actuator_ctrlrange[:arm_dof, 1]
        normalized = 2.0 * (positions - ctrl_min) / (ctrl_max - ctrl_min) - 1.0
        action = np.zeros(self.config.action_dimension, dtype=float)
        action[:arm_dof] = np.clip(normalized, -1.0, 1.0)
        action[arm_dof] = float(np.clip(gripper_command, -1.0, 1.0))
        return action

    def _sample_target_position(
        self,
        workspace_center: np.ndarray,
        reset_span: float,
        object_position: np.ndarray,
    ) -> np.ndarray:
        for _ in range(100):
            target_position = np.array(
                [
                    workspace_center[0] + self._rng.uniform(-reset_span, reset_span),
                    workspace_center[1] + self._rng.uniform(-reset_span, reset_span),
                    0.222,
                ],
                dtype=float,
            )
            planar_distance = np.linalg.norm((target_position - object_position)[:2])
            if planar_distance >= self.config.min_initial_object_target_distance:
                return target_position

        direction = np.array([1.0, 0.0], dtype=float)
        target_xy = object_position[:2] + direction * self.config.min_initial_object_target_distance
        lower = workspace_center - reset_span
        upper = workspace_center + reset_span
        return np.array(
            [
                np.clip(target_xy[0], lower[0], upper[0]),
                np.clip(target_xy[1], lower[1], upper[1]),
                0.222,
            ],
            dtype=float,
        )

    def _set_free_joint_pose(self, joint_id: int, position: np.ndarray) -> None:
        address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        dof_address = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _set_target_pose(self, position: np.ndarray) -> None:
        self._target_position = np.asarray(position, dtype=float).copy()
        self.data.mocap_pos[self._target_mocap_id] = self._target_position
        self.data.mocap_quat[self._target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _finger_contact_count(self) -> int:
        contacting_pads: set[int] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._cube_geom_id not in pair:
                continue
            for pad_id in self._finger_pad_geom_ids:
                if pad_id in pair:
                    contacting_pads.add(pad_id)
        return len(contacting_pads)

    def _site_position(self, site_id: int) -> np.ndarray:
        return np.array(self.data.site_xpos[site_id], dtype=float)

    def _body_position(self, body_id: int) -> np.ndarray:
        return np.array(self.data.xpos[body_id], dtype=float)

    def _success(self) -> bool:
        object_position = self._body_position(self._object_body_id)
        target_position = self._body_position(self._target_body_id)
        planar_distance = np.linalg.norm((object_position - target_position)[:2])
        resting_on_table = abs(object_position[2] - self._object_table_height) <= 0.012
        return bool(
            planar_distance <= self.config.target_radius
            and resting_on_table
            and self._has_held_object
            and self._has_lifted_object
            and self._has_released_after_lift
            and self._has_retreated_after_release
            and not self._held
        )

    def _observation(self) -> Mapping[str, Any]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ee_pos": self._site_position(self._ee_site_id),
            "object_pos": self._body_position(self._object_body_id),
            "target_pos": self._body_position(self._target_body_id),
            "held": self._held,
            "finger_contacts": self._finger_contact_count(),
            "has_held_object": self._has_held_object,
            "has_lifted_object": self._has_lifted_object,
            "has_released_after_lift": self._has_released_after_lift,
            "has_retreated_after_release": self._has_retreated_after_release,
            "step_count": self._step_count,
        }

    def _info(self) -> Mapping[str, Any]:
        return {
            "success": self._success(),
            "held": self._held,
            "finger_contacts": self._finger_contact_count(),
            "has_held_object": self._has_held_object,
            "has_lifted_object": self._has_lifted_object,
            "has_released_after_lift": self._has_released_after_lift,
            "has_retreated_after_release": self._has_retreated_after_release,
        }
