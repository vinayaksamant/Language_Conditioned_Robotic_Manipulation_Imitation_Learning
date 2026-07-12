from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BenchmarkConfig:
    """Defines the first pick-and-place benchmark contract."""

    task_name: str
    robot_name: str
    object_names: Sequence[str]
    action_dimension: int
    observation_dimension: int
    success_threshold: float


DEFAULT_BENCHMARK = BenchmarkConfig(
    task_name="pick_place_single_object",
    robot_name="franka_panda",
    object_names=("cube",),
    action_dimension=8,
    observation_dimension=32,
    success_threshold=1.0,
)
