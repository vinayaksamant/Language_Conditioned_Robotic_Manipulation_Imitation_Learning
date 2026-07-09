from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TaskInstruction:
    """Natural-language instruction paired with the manipulation task."""

    instruction: str
    canonical_label: str
    paraphrases: Sequence[str]


DEFAULT_TASK_INSTRUCTION = TaskInstruction(
    instruction="Pick up the cube and place it in the target zone.",
    canonical_label="pick_cube_to_zone",
    paraphrases=(
        "Grasp the cube and move it to the goal area.",
        "Pick the cube from the table and place it at the target.",
    ),
)
