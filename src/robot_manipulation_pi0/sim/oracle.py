from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import mujoco
import numpy as np

from .environment import PickPlaceEnvironment


STAGE_NAMES = (
    "approach",
    "descend",
    "close",
    "lift",
    "transfer",
    "descend_place",
    "release",
    "retreat",
)
STAGE_TO_INDEX = {stage: index for index, stage in enumerate(STAGE_NAMES)}


@dataclass
class ScriptedOraclePolicy:
    environment: PickPlaceEnvironment
    hover_height: float = 0.13
    transport_height: float = 0.17
    place_offset: float = 0.004
    position_tolerance: float = 0.008
    transfer_tolerance: float = 0.025
    grasp_tolerance: float = 0.006
    release_steps: int = 8
    ik_damping: float = 0.03
    ik_step_limit: float = 0.10
    ik_max_iterations: int = 100
    _stage: str = field(default="approach", init=False)
    _release_count: int = field(default=0, init=False)
    _last_step_count: int = field(default=-1, init=False)
    _grasp_xy: np.ndarray | None = field(default=None, init=False)
    _desired_orientation: np.ndarray | None = field(default=None, init=False)
    _cached_stage: str | None = field(default=None, init=False)
    _cached_target_point: np.ndarray | None = field(default=None, init=False)
    _cached_joint_target: np.ndarray | None = field(default=None, init=False)

    def act(self, observation: Mapping[str, Any]) -> list[float]:
        step_count = int(observation.get("step_count", 0))
        if step_count == 0 or step_count < self._last_step_count:
            self.reset()
        self._last_step_count = step_count
        if self._desired_orientation is None:
            self._desired_orientation = self._site_orientation(self.environment.data)

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
        joint_target = self._joint_target_for_stage(target_point)
        return self.environment.arm_positions_to_action(joint_target, gripper_command).tolist()

    def reset(self) -> None:
        self._stage = "approach"
        self._release_count = 0
        self._last_step_count = -1
        self._grasp_xy = None
        self._desired_orientation = None
        self._clear_joint_target()

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def stage_index(self) -> int:
        return STAGE_TO_INDEX[self._stage]

    def _target_and_gripper(
        self,
        ee_position: np.ndarray,
        object_position: np.ndarray,
        target_position: np.ndarray,
        held: bool,
    ) -> tuple[np.ndarray, float]:
        if self._grasp_xy is None:
            self._grasp_xy = object_position[:2].copy()

        table_height = self.environment._object_table_height
        object_hover = np.array(
            [self._grasp_xy[0], self._grasp_xy[1], table_height + self.hover_height],
            dtype=float,
        )
        object_grasp = np.array(
            [self._grasp_xy[0], self._grasp_xy[1], table_height],
            dtype=float,
        )
        transport_z = table_height + self.transport_height
        lift_point = np.array([self._grasp_xy[0], self._grasp_xy[1], transport_z], dtype=float)
        place_hover = np.array([target_position[0], target_position[1], transport_z], dtype=float)
        place_point = np.array(
            [target_position[0], target_position[1], table_height + self.place_offset],
            dtype=float,
        )

        if self._stage == "approach":
            if self._is_close(ee_position, object_hover, self.position_tolerance):
                self._set_stage("descend")
            else:
                return object_hover, -1.0

        if self._stage == "descend":
            if self._is_close(ee_position, object_grasp, self.grasp_tolerance):
                self._set_stage("close")
            else:
                return object_grasp, -1.0

        if self._stage == "close":
            if held:
                self._set_stage("lift")
            else:
                return object_grasp, 1.0

        if self._stage == "lift":
            if held and ee_position[2] >= transport_z - self.position_tolerance:
                self._set_stage("transfer")
            else:
                return lift_point, 1.0

        if self._stage == "transfer":
            if held and self._is_close(ee_position, place_hover, self.transfer_tolerance):
                self._set_stage("descend_place")
            else:
                return place_hover, 1.0

        if self._stage == "descend_place":
            if held and self._is_close(ee_position, place_point, self.grasp_tolerance):
                self._set_stage("release")
                self._release_count = 0
            else:
                return place_point, 1.0

        if self._stage == "release":
            self._release_count += 1
            if not held and self._release_count >= self.release_steps:
                self._set_stage("retreat")
            else:
                return place_point, -1.0

        if self._stage == "retreat":
            return place_hover, -1.0

        self._set_stage("approach")
        return object_hover, -1.0

    @staticmethod
    def _is_close(current: np.ndarray, target: np.ndarray, tolerance: float) -> bool:
        return bool(np.linalg.norm(current - target) <= tolerance)

    def _set_stage(self, stage: str) -> None:
        if stage != self._stage:
            self._stage = stage
            self._clear_joint_target()

    def _clear_joint_target(self) -> None:
        self._cached_stage = None
        self._cached_target_point = None
        self._cached_joint_target = None

    def _joint_target_for_stage(self, target_point: np.ndarray) -> np.ndarray:
        if (
            self._cached_joint_target is None
            or self._cached_stage != self._stage
            or self._cached_target_point is None
            or not np.allclose(self._cached_target_point, target_point, atol=1e-8)
        ):
            self._cached_joint_target = self._solve_inverse_kinematics(target_point)
            self._cached_target_point = target_point.copy()
            self._cached_stage = self._stage
        return self._cached_joint_target.copy()

    def _solve_inverse_kinematics(self, target_point: np.ndarray) -> np.ndarray:
        model = self.environment.model
        ik_data = mujoco.MjData(model)
        ik_data.qpos[:] = self.environment.data.qpos
        ik_data.qvel[:] = 0.0
        mujoco.mj_forward(model, ik_data)

        joint_ids = np.asarray(self.environment._arm_joint_ids, dtype=int)
        qpos_indices = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
        dof_indices = np.asarray(self.environment._arm_dof_indices, dtype=int)
        joint_lower = model.jnt_range[joint_ids, 0] + 1e-3
        joint_upper = model.jnt_range[joint_ids, 1] - 1e-3

        for _ in range(self.ik_max_iterations):
            position_error = target_point - ik_data.site_xpos[self.environment._ee_site_id]
            orientation_error = self._orientation_error(ik_data)
            if np.linalg.norm(position_error) <= 1e-5 and np.linalg.norm(orientation_error) <= 1e-5:
                break

            jacobian_position = np.zeros((3, model.nv), dtype=float)
            jacobian_rotation = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacSite(
                model,
                ik_data,
                jacobian_position,
                jacobian_rotation,
                self.environment._ee_site_id,
            )
            jacobian = np.vstack(
                [
                    jacobian_position[:, dof_indices],
                    jacobian_rotation[:, dof_indices],
                ]
            )
            error = np.concatenate([position_error, orientation_error])
            regularizer = (self.ik_damping**2) * np.eye(jacobian.shape[0])
            joint_delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + regularizer,
                error,
            )
            peak_delta = float(np.max(np.abs(joint_delta)))
            if peak_delta > self.ik_step_limit:
                joint_delta *= self.ik_step_limit / peak_delta
            ik_data.qpos[qpos_indices] = np.clip(
                ik_data.qpos[qpos_indices] + joint_delta,
                joint_lower,
                joint_upper,
            )
            mujoco.mj_forward(model, ik_data)

        return np.asarray(ik_data.qpos[qpos_indices], dtype=float).copy()

    def _site_orientation(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(data.site_xmat[self.environment._ee_site_id], dtype=float).reshape(3, 3)

    def _orientation_error(self, data: mujoco.MjData) -> np.ndarray:
        if self._desired_orientation is None:
            return np.zeros(3, dtype=float)
        current = self._site_orientation(data)
        desired = self._desired_orientation
        return 0.5 * sum(np.cross(current[:, index], desired[:, index]) for index in range(3))
