from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from .environment import PickPlaceEnvironment


@dataclass
class ScriptedOraclePolicy:
    environment: PickPlaceEnvironment
    position_gain: float = 2.0

    def act(self, observation: Mapping[str, Any]) -> list[float]:
        ee_position = np.asarray(observation["ee_pos"], dtype=float)
        object_position = np.asarray(observation["object_pos"], dtype=float)
        target_position = np.asarray(observation["target_pos"], dtype=float)
        held = bool(observation["held"])
        target_point = target_position + np.array([0.0, 0.0, 0.08], dtype=float) if held else object_position + np.array([0.0, 0.0, 0.08], dtype=float)
        cartesian_error = target_point - ee_position
        jacobian = np.zeros((3, self.environment.model.nv), dtype=float)
        angular_jacobian = np.zeros((3, self.environment.model.nv), dtype=float)
        mujoco.mj_jacSite(self.environment.model, self.environment.data, jacobian, angular_jacobian, self.environment._ee_site_id)
        arm_jacobian = jacobian[:, self.environment._arm_dof_indices]
        joint_velocity = np.linalg.pinv(arm_jacobian) @ (self.position_gain * cartesian_error)
        action = np.zeros(self.environment.config.action_dimension, dtype=float)
        arm_dof = len(self.environment._arm_dof_indices)
        action[:arm_dof] = np.clip(joint_velocity, -1.0, 1.0)
        if held:
            action[arm_dof] = -1.0 if np.linalg.norm(target_position - ee_position) < self.environment.config.grasp_radius * 1.5 else 1.0
        else:
            action[arm_dof] = 1.0 if np.linalg.norm(object_position - ee_position) < self.environment.config.grasp_radius else -1.0
        return action.tolist()
