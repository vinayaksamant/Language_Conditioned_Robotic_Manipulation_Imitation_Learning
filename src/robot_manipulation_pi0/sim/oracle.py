from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from .environment import PickPlaceEnvironment


@dataclass
class ScriptedOraclePolicy:
    environment: PickPlaceEnvironment
    position_gain: float = 4.0
    damping: float = 0.05
    hover_height: float = 0.12
    place_height: float = 0.04

    def act(self, observation: Mapping[str, Any]) -> list[float]:
        ee_position = np.asarray(observation["ee_pos"], dtype=float)
        object_position = np.asarray(observation["object_pos"], dtype=float)
        target_position = np.asarray(observation["target_pos"], dtype=float)
        held = bool(observation["held"])
        target_point, gripper_command = self._target_and_gripper(
            ee_position,
            object_position,
            target_position,
            held,
        )
        action = self._cartesian_action(ee_position, target_point)
        action[len(self.environment._arm_dof_indices)] = gripper_command
        return action.tolist()

    def _target_and_gripper(
        self,
        ee_position: np.ndarray,
        object_position: np.ndarray,
        target_position: np.ndarray,
        held: bool,
    ) -> tuple[np.ndarray, float]:
        if held:
            target_hover = target_position + np.array([0.0, 0.0, self.hover_height], dtype=float)
            horizontal_error = np.linalg.norm((ee_position - target_hover)[:2])
            if horizontal_error > 0.035:
                return target_hover, 1.0
            place_point = target_position + np.array([0.0, 0.0, self.place_height], dtype=float)
            if np.linalg.norm(ee_position - place_point) > 0.035:
                return place_point, 1.0
            return place_point, -1.0

        object_hover = object_position + np.array([0.0, 0.0, self.hover_height], dtype=float)
        horizontal_error = np.linalg.norm((ee_position - object_hover)[:2])
        if horizontal_error > 0.035:
            return object_hover, -1.0

        grasp_point = object_position + np.array([0.0, 0.0, 0.02], dtype=float)
        if np.linalg.norm(ee_position - grasp_point) > self.environment.config.grasp_radius * 0.75:
            return grasp_point, -1.0
        return grasp_point, 1.0

    def _cartesian_action(self, ee_position: np.ndarray, target_point: np.ndarray) -> np.ndarray:
        cartesian_error = target_point - ee_position
        jacobian = np.zeros((3, self.environment.model.nv), dtype=float)
        angular_jacobian = np.zeros((3, self.environment.model.nv), dtype=float)
        mujoco.mj_jacSite(self.environment.model, self.environment.data, jacobian, angular_jacobian, self.environment._ee_site_id)
        arm_jacobian = jacobian[:, self.environment._arm_dof_indices]
        regularizer = (self.damping**2) * np.eye(arm_jacobian.shape[0])
        joint_velocity = arm_jacobian.T @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + regularizer,
            self.position_gain * cartesian_error,
        )
        action = np.zeros(self.environment.config.action_dimension, dtype=float)
        arm_dof = len(self.environment._arm_dof_indices)
        action[:arm_dof] = np.clip(joint_velocity, -1.0, 1.0)
        return action
