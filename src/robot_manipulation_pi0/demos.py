from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from robot_manipulation_pi0.sim import ACTION_MODE, PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy, StepResult
from robot_manipulation_pi0.task_spec import DEFAULT_TASK_INSTRUCTION, TaskInstruction


DEMONSTRATION_SCHEMA_VERSION = 2
StepCallback = Callable[
    [PickPlaceEnvironment, ScriptedOraclePolicy, Mapping[str, Any], list[float], StepResult],
    None,
]


@dataclass(frozen=True)
class DemonstrationEpisode:
    seed: int
    success: bool
    steps: int
    instruction: TaskInstruction
    observations: Mapping[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


@dataclass(frozen=True)
class DemonstrationSummary:
    directory: Path
    episode_count: int
    total_steps: int
    action_dimension: int
    qpos_dimension: int
    qvel_dimension: int
    min_steps: int
    max_steps: int
    mean_steps: float
    failed_seeds: tuple[int, ...]


def collect_episode(
    seed: int,
    config: PickPlaceConfig,
    instruction: TaskInstruction = DEFAULT_TASK_INSTRUCTION,
    environment: PickPlaceEnvironment | None = None,
    policy: ScriptedOraclePolicy | None = None,
    step_callback: StepCallback | None = None,
) -> DemonstrationEpisode:
    environment = environment or PickPlaceEnvironment(config)
    policy = policy or ScriptedOraclePolicy(environment)
    if policy.environment is not environment:
        raise ValueError("The policy and demonstration environment must be the same instance.")
    policy.reset()
    observation = environment.reset(seed=seed)

    observations: dict[str, list[Any]] = {
        "qpos": [],
        "qvel": [],
        "ee_pos": [],
        "object_pos": [],
        "target_pos": [],
        "held": [],
        "finger_contacts": [],
        "has_held_object": [],
        "has_lifted_object": [],
        "has_released_after_lift": [],
        "has_retreated_after_release": [],
        "oracle_stage_index": [],
        "step_count": [],
    }
    actions: list[list[float]] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    success = False

    for _ in range(config.max_steps):
        action = policy.act(observation)
        _append_observation(observations, observation)
        observations["oracle_stage_index"].append(int(policy.stage_index))
        result = environment.step(action)
        actions.append(action)
        rewards.append(float(result.reward))
        terminated.append(bool(result.terminated))
        truncated.append(bool(result.truncated))
        observation = result.observation
        success = bool(result.info["success"])
        if step_callback is not None:
            step_callback(environment, policy, observation, action, result)
        if result.terminated or result.truncated:
            break

    return DemonstrationEpisode(
        seed=seed,
        success=success,
        steps=len(actions),
        instruction=instruction,
        observations={key: np.asarray(value) for key, value in observations.items()},
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
    )


def save_episode(path: Path, episode: DemonstrationEpisode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "action_mode": ACTION_MODE,
        "seed": episode.seed,
        "success": episode.success,
        "steps": episode.steps,
        "instruction": asdict(episode.instruction),
    }
    np.savez_compressed(
        path,
        metadata=json.dumps(metadata),
        actions=episode.actions,
        rewards=episode.rewards,
        terminated=episode.terminated,
        truncated=episode.truncated,
        **{f"obs_{key}": value for key, value in episode.observations.items()},
    )


def validate_demo_directory(directory: Path, expected_action_dimension: int | None = None) -> DemonstrationSummary:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_current_schema(manifest, manifest_path)
    episode_entries = manifest.get("episodes", [])
    if not isinstance(episode_entries, list) or not episode_entries:
        raise ValueError(f"Manifest contains no episodes: {manifest_path}")

    steps: list[int] = []
    action_dimension: int | None = None
    qpos_dimension: int | None = None
    qvel_dimension: int | None = None

    for entry in episode_entries:
        if not isinstance(entry, dict) or "file" not in entry:
            raise ValueError(f"Malformed episode entry in manifest: {entry!r}")
        episode_path = directory / str(entry["file"])
        metadata, shapes = validate_episode_file(episode_path, expected_action_dimension)
        _require_current_schema(metadata, episode_path)
        if not bool(metadata["success"]):
            raise ValueError(f"Episode is marked unsuccessful: {episode_path}")
        if int(entry["seed"]) != int(metadata["seed"]):
            raise ValueError(f"Manifest seed does not match episode metadata: {episode_path}")
        if int(entry["steps"]) != int(metadata["steps"]):
            raise ValueError(f"Manifest step count does not match episode metadata: {episode_path}")

        steps.append(int(metadata["steps"]))
        action_dimension = _consistent_dimension(action_dimension, shapes["actions"][1], "action")
        qpos_dimension = _consistent_dimension(qpos_dimension, shapes["obs_qpos"][1], "qpos")
        qvel_dimension = _consistent_dimension(qvel_dimension, shapes["obs_qvel"][1], "qvel")

    failed_seeds = tuple(int(seed) for seed in manifest.get("failed_seeds", []))
    return DemonstrationSummary(
        directory=directory,
        episode_count=len(episode_entries),
        total_steps=sum(steps),
        action_dimension=int(action_dimension),
        qpos_dimension=int(qpos_dimension),
        qvel_dimension=int(qvel_dimension),
        min_steps=min(steps),
        max_steps=max(steps),
        mean_steps=float(np.mean(steps)),
        failed_seeds=failed_seeds,
    )


def validate_episode_file(
    path: Path,
    expected_action_dimension: int | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    required_keys = {
        "metadata",
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "obs_qpos",
        "obs_qvel",
        "obs_ee_pos",
        "obs_object_pos",
        "obs_target_pos",
        "obs_held",
        "obs_finger_contacts",
        "obs_has_held_object",
        "obs_has_lifted_object",
        "obs_has_released_after_lift",
        "obs_has_retreated_after_release",
        "obs_oracle_stage_index",
        "obs_step_count",
    }
    if not path.exists():
        raise FileNotFoundError(f"Missing episode file: {path}")

    with np.load(path, allow_pickle=False) as data:
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Episode is missing keys {missing}: {path}")

        metadata = json.loads(str(data["metadata"]))
        _require_current_schema(metadata, path)
        steps = int(metadata["steps"])
        if steps <= 0:
            raise ValueError(f"Episode has no steps: {path}")

        shapes = {key: tuple(data[key].shape) for key in required_keys if key != "metadata"}
        actions = data["actions"]
        action_dimension = actions.shape[1] if actions.ndim == 2 else None
        if action_dimension is None:
            raise ValueError(f"Actions must be a 2D array: {path}")
        if expected_action_dimension is not None and action_dimension != expected_action_dimension:
            raise ValueError(
                f"Expected action dimension {expected_action_dimension}, got {action_dimension}: {path}"
            )
        for key in required_keys:
            if key == "metadata":
                continue
            if data[key].shape[0] != steps:
                raise ValueError(f"{key} length does not match metadata steps in {path}")

        _require_shape(data["obs_ee_pos"], (steps, 3), "obs_ee_pos", path)
        _require_shape(data["obs_object_pos"], (steps, 3), "obs_object_pos", path)
        _require_shape(data["obs_target_pos"], (steps, 3), "obs_target_pos", path)
        _require_shape(data["rewards"], (steps,), "rewards", path)
        _require_shape(data["terminated"], (steps,), "terminated", path)
        _require_shape(data["truncated"], (steps,), "truncated", path)

        for key in ("actions", "rewards", "obs_qpos", "obs_qvel", "obs_ee_pos", "obs_object_pos", "obs_target_pos"):
            if not np.isfinite(data[key]).all():
                raise ValueError(f"{key} contains NaN or infinite values: {path}")

        if not bool(data["terminated"][-1]):
            raise ValueError(f"Successful demo should end with terminated=True: {path}")
        if bool(data["truncated"][-1]):
            raise ValueError(f"Successful demo should not end with truncated=True: {path}")

    return metadata, shapes


def _append_observation(target: dict[str, list[Any]], observation: Mapping[str, Any]) -> None:
    target["qpos"].append(np.asarray(observation["qpos"], dtype=np.float32))
    target["qvel"].append(np.asarray(observation["qvel"], dtype=np.float32))
    target["ee_pos"].append(np.asarray(observation["ee_pos"], dtype=np.float32))
    target["object_pos"].append(np.asarray(observation["object_pos"], dtype=np.float32))
    target["target_pos"].append(np.asarray(observation["target_pos"], dtype=np.float32))
    target["held"].append(bool(observation["held"]))
    target["finger_contacts"].append(int(observation["finger_contacts"]))
    target["has_held_object"].append(bool(observation["has_held_object"]))
    target["has_lifted_object"].append(bool(observation["has_lifted_object"]))
    target["has_released_after_lift"].append(bool(observation["has_released_after_lift"]))
    target["has_retreated_after_release"].append(bool(observation["has_retreated_after_release"]))
    target["step_count"].append(int(observation["step_count"]))


def _consistent_dimension(previous: int | None, current: int, name: str) -> int:
    if previous is not None and previous != current:
        raise ValueError(f"Inconsistent {name} dimension: expected {previous}, got {current}")
    return current


def _require_shape(array: np.ndarray, expected: tuple[int, ...], key: str, path: Path) -> None:
    if array.shape != expected:
        raise ValueError(f"{key} shape mismatch in {path}: expected {expected}, got {array.shape}")


def _require_current_schema(metadata: Mapping[str, Any], path: Path) -> None:
    schema_version = metadata.get("schema_version")
    action_mode = metadata.get("action_mode")
    if schema_version != DEMONSTRATION_SCHEMA_VERSION or action_mode != ACTION_MODE:
        raise ValueError(
            f"Incompatible demonstration data in {path}: expected schema "
            f"{DEMONSTRATION_SCHEMA_VERSION} with action_mode={ACTION_MODE!r}, got "
            f"schema={schema_version!r}, action_mode={action_mode!r}. Recollect the demonstrations."
        )
