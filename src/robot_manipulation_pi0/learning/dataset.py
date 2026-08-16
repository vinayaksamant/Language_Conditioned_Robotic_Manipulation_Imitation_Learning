from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_manipulation_pi0.demos import validate_demo_directory
from robot_manipulation_pi0.sim.oracle import STAGE_NAMES


@dataclass(frozen=True)
class DemonstrationDataset:
    observations: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplit:
    train: DemonstrationDataset
    validation: DemonstrationDataset


def load_demonstration_dataset(directory: Path) -> DemonstrationDataset:
    validate_demo_directory(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []

    for episode_index, entry in enumerate(manifest["episodes"]):
        episode_path = directory / entry["file"]
        with np.load(episode_path, allow_pickle=False) as data:
            features = _make_features(data)
            episode_actions = np.asarray(data["actions"], dtype=np.float32)
        if features.shape[0] != episode_actions.shape[0]:
            raise ValueError(f"Feature/action length mismatch in {episode_path}")
        observations.append(features)
        actions.append(episode_actions)
        episode_ids.append(np.full(features.shape[0], episode_index, dtype=np.int32))

    feature_names = (
        "qpos",
        "qvel",
        "ee_pos",
        "object_pos",
        "target_pos",
        "held",
        "finger_contacts",
        "has_held_object",
        "has_lifted_object",
        "has_released_after_lift",
        "has_retreated_after_release",
    ) + tuple(f"stage_{stage}" for stage in STAGE_NAMES)
    return DemonstrationDataset(
        observations=np.concatenate(observations, axis=0).astype(np.float32),
        actions=np.concatenate(actions, axis=0).astype(np.float32),
        episode_ids=np.concatenate(episode_ids, axis=0),
        feature_names=feature_names,
    )


def split_dataset(
    dataset: DemonstrationDataset,
    validation_fraction: float,
    seed: int,
) -> DatasetSplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    unique_episodes = np.unique(dataset.episode_ids)
    if unique_episodes.size < 2:
        raise ValueError("At least two episodes are required for a train/validation split.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_episodes)
    validation_count = max(1, int(round(unique_episodes.size * validation_fraction)))
    validation_episodes = set(int(value) for value in shuffled[:validation_count])
    validation_mask = np.array([int(value) in validation_episodes for value in dataset.episode_ids])
    train_mask = ~validation_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Train/validation split produced an empty subset.")

    return DatasetSplit(
        train=_subset(dataset, train_mask),
        validation=_subset(dataset, validation_mask),
    )


def _make_features(data: np.lib.npyio.NpzFile) -> np.ndarray:
    held = np.asarray(data["obs_held"], dtype=np.float32).reshape(-1, 1)
    steps = held.shape[0]
    finger_contacts = np.asarray(data["obs_finger_contacts"], dtype=np.float32).reshape(-1, 1) / 2.0
    has_held_object = _optional_flag(data, "obs_has_held_object", steps)
    has_lifted_object = _optional_flag(data, "obs_has_lifted_object", steps)
    has_released_after_lift = _optional_flag(data, "obs_has_released_after_lift", steps)
    has_retreated_after_release = _optional_flag(data, "obs_has_retreated_after_release", steps)
    oracle_stage = _optional_stage_one_hot(data, steps)
    return np.concatenate(
        [
            np.asarray(data["obs_qpos"], dtype=np.float32),
            np.asarray(data["obs_qvel"], dtype=np.float32),
            np.asarray(data["obs_ee_pos"], dtype=np.float32),
            np.asarray(data["obs_object_pos"], dtype=np.float32),
            np.asarray(data["obs_target_pos"], dtype=np.float32),
            held,
            finger_contacts,
            has_held_object,
            has_lifted_object,
            has_released_after_lift,
            has_retreated_after_release,
            oracle_stage,
        ],
        axis=1,
    )


def _optional_flag(data: np.lib.npyio.NpzFile, key: str, steps: int) -> np.ndarray:
    if key not in data.files:
        return np.zeros((steps, 1), dtype=np.float32)
    return np.asarray(data[key], dtype=np.float32).reshape(-1, 1)


def _optional_stage_one_hot(data: np.lib.npyio.NpzFile, steps: int) -> np.ndarray:
    one_hot = np.zeros((steps, len(STAGE_NAMES)), dtype=np.float32)
    if "obs_oracle_stage_index" not in data.files:
        return one_hot
    stage_indices = np.asarray(data["obs_oracle_stage_index"], dtype=np.int64).reshape(-1)
    if stage_indices.shape[0] != steps:
        raise ValueError("obs_oracle_stage_index length does not match episode steps.")
    valid = (0 <= stage_indices) & (stage_indices < len(STAGE_NAMES))
    one_hot[np.arange(steps)[valid], stage_indices[valid]] = 1.0
    return one_hot


def stage_one_hot(stage_index: int | None) -> np.ndarray:
    one_hot = np.zeros(len(STAGE_NAMES), dtype=np.float32)
    if stage_index is not None:
        one_hot[int(stage_index)] = 1.0
    return one_hot


def observation_to_feature(observation: Mapping[str, Any], stage_index: int | None = None) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(observation["qpos"], dtype=np.float32),
            np.asarray(observation["qvel"], dtype=np.float32),
            np.asarray(observation["ee_pos"], dtype=np.float32),
            np.asarray(observation["object_pos"], dtype=np.float32),
            np.asarray(observation["target_pos"], dtype=np.float32),
            np.asarray([float(observation["held"])], dtype=np.float32),
            np.asarray([float(observation.get("finger_contacts", 0)) / 2.0], dtype=np.float32),
            np.asarray([float(observation.get("has_held_object", False))], dtype=np.float32),
            np.asarray([float(observation.get("has_lifted_object", False))], dtype=np.float32),
            np.asarray([float(observation.get("has_released_after_lift", False))], dtype=np.float32),
            np.asarray([float(observation.get("has_retreated_after_release", False))], dtype=np.float32),
            stage_one_hot(stage_index),
        ],
        axis=0,
    )


def _subset(dataset: DemonstrationDataset, mask: np.ndarray) -> DemonstrationDataset:
    return DemonstrationDataset(
        observations=dataset.observations[mask],
        actions=dataset.actions[mask],
        episode_ids=dataset.episode_ids[mask],
        feature_names=dataset.feature_names,
    )
