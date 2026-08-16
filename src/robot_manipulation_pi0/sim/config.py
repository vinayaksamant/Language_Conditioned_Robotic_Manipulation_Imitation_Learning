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
    target_radius: float = 0.035
    min_initial_object_target_distance: float = 0.12
    object_lift_height: float = 0.06
    grasp_contact_steps: int = 3
    substeps: int = 5
    model_file: str = "franka_emika_panda/pick_place_scene.xml"
