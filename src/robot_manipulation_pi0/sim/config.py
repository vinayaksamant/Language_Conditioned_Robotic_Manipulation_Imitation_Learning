from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PickPlaceConfig:
    robot_name: str
    object_names: Sequence[str]
    workspace_size: float
    action_dimension: int = 8
    observation_dimension: int = 32
    success_distance: float = 0.02
    max_steps: int = 100
    arm_velocity_scale: float = 1.5
    grasp_radius: float = 0.06
    target_radius: float = 0.05
    object_lift_height: float = 0.04
    substeps: int = 5
    model_file: str = "franka_emika_panda/pick_place_scene.xml"
