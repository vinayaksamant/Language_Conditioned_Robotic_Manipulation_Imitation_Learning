from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from robot_manipulation_pi0.sim import VLA_SCENE_OBJECTS, VLA_SCENE_TARGETS


@dataclass(frozen=True)
class LanguagePickPlaceTask:
    instruction: str
    object_key: str
    target_key: str
    canonical_label: str


TASK_TEMPLATES: Mapping[str, Sequence[str]] = {
    "red_cube": (
        "Pick up the red cube and place it on the {target}.",
        "Pick the red block and put it on the {target}.",
        "Move the red cube onto the {target}.",
        "Grasp the red block and place it on the {target}.",
        "Pick the cube and place it on the {target}.",
    ),
    "blue_cylinder": (
        "Pick up the tall blue cylinder and place it on the {target}.",
        "Pick the blue cylinder and put it on the {target}.",
        "Move the tall blue object onto the {target}.",
        "Grasp the blue cylindrical object and place it on the {target}.",
        "Pick the cylinder and place it on the {target}.",
        "Pick the taller object and place it on the {target}.",
    ),
}

_OBJECT_ALIASES = {
    "red_cube": ("red", "cube", "red block"),
    "blue_cylinder": ("blue", "cylinder", "cylindrical", "tall object", "taller object", "tall blue"),
}
_VALID_OBJECT_KEYS = {scene_object.key for scene_object in VLA_SCENE_OBJECTS}
_TARGET_NAMES = {scene_target.key: scene_target.display_name for scene_target in VLA_SCENE_TARGETS}
_TARGET_ALIASES = {
    "green_plate": ("green plate", "green target"),
    "yellow_plate": ("yellow plate", "yellow target"),
}
TASK_INSTRUCTIONS: Mapping[str, Sequence[str]] = {
    object_key: tuple(template.format(target=_TARGET_NAMES["green_plate"]) for template in templates)
    for object_key, templates in TASK_TEMPLATES.items()
}


def task_for_object(
    object_key: str,
    variant: int = 0,
    target_key: str = "green_plate",
) -> LanguagePickPlaceTask:
    if object_key not in TASK_TEMPLATES or object_key not in _VALID_OBJECT_KEYS:
        choices = ", ".join(sorted(TASK_TEMPLATES))
        raise ValueError(f"Unknown task object {object_key!r}. Expected one of: {choices}.")
    if target_key not in _TARGET_NAMES:
        choices = ", ".join(sorted(_TARGET_NAMES))
        raise ValueError(f"Unknown task target {target_key!r}. Expected one of: {choices}.")
    templates = TASK_TEMPLATES[object_key]
    instruction = templates[variant % len(templates)].format(target=_TARGET_NAMES[target_key])
    return LanguagePickPlaceTask(
        instruction=instruction,
        object_key=object_key,
        target_key=target_key,
        canonical_label=f"pick_{object_key}_to_{target_key}",
    )


def sample_task(
    object_key: str,
    rng: np.random.Generator,
    target_key: str = "green_plate",
) -> LanguagePickPlaceTask:
    templates = TASK_TEMPLATES.get(object_key)
    if templates is None:
        return task_for_object(object_key, target_key=target_key)
    return task_for_object(
        object_key,
        int(rng.integers(0, len(templates))),
        target_key=target_key,
    )


def resolve_task_instruction(
    instruction: str,
    require_explicit_target: bool = False,
) -> LanguagePickPlaceTask:
    normalized = re.sub(r"[^a-z0-9]+", " ", instruction.lower()).strip()
    if not normalized:
        raise ValueError("Task instruction cannot be empty.")

    matches = {
        object_key
        for object_key, aliases in _OBJECT_ALIASES.items()
        if any(_contains_phrase(normalized, alias) for alias in aliases)
    }
    if "red" in normalized.split():
        matches.discard("blue_cylinder")
    if "blue" in normalized.split():
        matches.discard("red_cube")

    if len(matches) != 1:
        examples = "; ".join(task_for_object(key).instruction for key in sorted(TASK_TEMPLATES))
        raise ValueError(
            f"Could not identify exactly one supported object in {instruction!r}. Examples: {examples}"
        )
    object_key = matches.pop()
    if _contains_phrase(normalized, "other than the green plate") or _contains_phrase(
        normalized, "other than green plate"
    ):
        target_matches = {"yellow_plate"}
    elif _contains_phrase(normalized, "other than the yellow plate") or _contains_phrase(
        normalized, "other than yellow plate"
    ):
        target_matches = {"green_plate"}
    else:
        target_matches = {
            target_key
            for target_key, aliases in _TARGET_ALIASES.items()
            if any(_contains_phrase(normalized, alias) for alias in aliases)
        }
    if len(target_matches) > 1:
        raise ValueError(f"Instruction names more than one destination: {instruction!r}.")
    if not target_matches:
        if require_explicit_target:
            choices = ", ".join(_TARGET_NAMES.values())
            raise ValueError(f"Instruction must name one destination ({choices}): {instruction!r}.")
        target_key = "green_plate"
    else:
        target_key = target_matches.pop()
    return LanguagePickPlaceTask(
        instruction=instruction.strip(),
        object_key=object_key,
        target_key=target_key,
        canonical_label=f"pick_{object_key}_to_{target_key}",
    )


def _contains_phrase(normalized_instruction: str, phrase: str) -> bool:
    normalized_phrase = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
    return f" {normalized_phrase} " in f" {normalized_instruction} "
