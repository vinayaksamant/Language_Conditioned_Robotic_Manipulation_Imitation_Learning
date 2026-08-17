from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_manipulation_pi0.sim import (
    ACTION_MODE,
    MultiObjectPickPlaceEnvironment,
    PickPlaceConfig,
    ScriptedOraclePolicy,
)
from robot_manipulation_pi0.sim.oracle import STAGE_NAMES

from .camera import CameraSpec, MujocoCameraRig
from .observation import ROBOT_STATE_NAMES, capture_vla_observation
from .tasks import LanguagePickPlaceTask


VLA_DATASET_SCHEMA_VERSION = 1
VLA_ORACLE_JOINT_STEP_LIMIT = 0.02


@dataclass(frozen=True)
class VLAEpisode:
    seed: int
    task: LanguagePickPlaceTask
    success: bool
    states: np.ndarray
    images: Mapping[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    oracle_stage_index: np.ndarray
    camera_specs: tuple[CameraSpec, ...]
    fps: float
    simulation_steps: int
    record_every: int

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class VLADatasetSummary:
    directory: Path
    episode_count: int
    total_steps: int
    mean_steps: float
    object_counts: Mapping[str, int]
    target_counts: Mapping[str, int]
    camera_keys: tuple[str, ...]
    state_dimension: int
    action_dimension: int


def collect_vla_episode(
    seed: int,
    config: PickPlaceConfig,
    task: LanguagePickPlaceTask,
    environment: MultiObjectPickPlaceEnvironment,
    cameras: MujocoCameraRig,
    policy: ScriptedOraclePolicy | None = None,
    record_every: int = 4,
) -> VLAEpisode:
    if record_every <= 0:
        raise ValueError("record_every must be positive.")
    policy = policy or ScriptedOraclePolicy(
        environment,
        joint_command_step_limit=VLA_ORACLE_JOINT_STEP_LIMIT,
    )
    if policy.environment is not environment:
        raise ValueError("The policy and VLA environment must be the same instance.")
    if cameras.model is not environment.model:
        raise ValueError("The camera rig and VLA environment must use the same MuJoCo model.")

    policy.reset()
    oracle_observation = environment.reset(
        seed=seed,
        object_key=task.object_key,
        target_key=task.target_key,
    )
    states: list[np.ndarray] = []
    image_frames: dict[str, list[np.ndarray]] = {spec.key: [] for spec in cameras.specs}
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    stages: list[int] = []
    success = False
    simulation_steps = 0

    for simulation_step in range(config.max_steps):
        action = np.asarray(policy.act(oracle_observation), dtype=np.float32)
        should_record = simulation_step % record_every == 0
        if should_record:
            policy_observation = capture_vla_observation(
                environment.model,
                environment.data,
                cameras,
                task.instruction,
            )
            states.append(policy_observation.state)
            for camera_key, image in policy_observation.images.items():
                image_frames[camera_key].append(image)
            actions.append(action)
            stages.append(policy.stage_index)

        result = environment.step(action)
        simulation_steps += 1
        if should_record:
            rewards.append(float(result.reward))
            terminated.append(bool(result.terminated))
            truncated.append(bool(result.truncated))
        oracle_observation = result.observation
        success = bool(result.info["success"])
        if result.terminated or result.truncated:
            if not should_record and terminated:
                rewards[-1] = float(result.reward)
                terminated[-1] = bool(result.terminated)
                truncated[-1] = bool(result.truncated)
            break

    if not actions:
        raise RuntimeError("VLA episode did not execute any actions.")

    control_period = float(environment.model.opt.timestep) * config.substeps * record_every
    return VLAEpisode(
        seed=seed,
        task=task,
        success=success,
        states=np.stack(states).astype(np.float32, copy=False),
        images={key: np.stack(frames) for key, frames in image_frames.items()},
        actions=np.stack(actions).astype(np.float32, copy=False),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
        oracle_stage_index=np.asarray(stages, dtype=np.int64),
        camera_specs=cameras.specs,
        fps=1.0 / control_period,
        simulation_steps=simulation_steps,
        record_every=record_every,
    )


def save_vla_episode(path: Path, episode: VLAEpisode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": VLA_DATASET_SCHEMA_VERSION,
        "action_mode": ACTION_MODE,
        "seed": episode.seed,
        "success": episode.success,
        "steps": episode.steps,
        "simulation_steps": episode.simulation_steps,
        "record_every": episode.record_every,
        "fps": episode.fps,
        "task": asdict(episode.task),
        "state_names": list(ROBOT_STATE_NAMES),
        "oracle_stage_names": list(STAGE_NAMES),
        "cameras": [asdict(spec) for spec in episode.camera_specs],
    }
    payload: dict[str, np.ndarray | str] = {
        "metadata": json.dumps(metadata),
        "observation.state": episode.states,
        "action": episode.actions,
        "reward": episode.rewards,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "oracle_stage_index": episode.oracle_stage_index,
    }
    for camera_key, frames in episode.images.items():
        payload[f"observation.images.{camera_key}"] = frames
    np.savez_compressed(path, **payload)


def validate_vla_episode_file(
    path: Path,
    expected_camera_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing VLA episode: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            raise ValueError(f"VLA episode is missing metadata: {path}")
        metadata = json.loads(str(data["metadata"]))
        _require_current_schema(metadata, path)
        steps = int(metadata.get("steps", 0))
        if steps <= 0:
            raise ValueError(f"VLA episode has no steps: {path}")

        camera_keys = tuple(camera["key"] for camera in metadata.get("cameras", []))
        if not camera_keys:
            raise ValueError(f"VLA episode has no camera metadata: {path}")
        if expected_camera_keys is not None and camera_keys != tuple(expected_camera_keys):
            raise ValueError(
                f"VLA camera keys differ in {path}: expected {tuple(expected_camera_keys)}, got {camera_keys}."
            )

        required = {
            "observation.state",
            "action",
            "reward",
            "terminated",
            "truncated",
            "oracle_stage_index",
            *(f"observation.images.{key}" for key in camera_keys),
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"VLA episode is missing keys {', '.join(sorted(missing))}: {path}")
        forbidden = ("object_pos", "target_pos", "scene_object_pos")
        leaked = [key for key in data.files if any(name in key for name in forbidden)]
        if leaked:
            raise ValueError(f"VLA policy data contains privileged coordinates {leaked}: {path}")

        for key in required:
            if data[key].shape[0] != steps:
                raise ValueError(f"{key} length does not match metadata steps in {path}")
        state = data["observation.state"]
        action = data["action"]
        if state.shape != (steps, len(ROBOT_STATE_NAMES)):
            raise ValueError(f"Robot state shape is {state.shape}, expected {(steps, len(ROBOT_STATE_NAMES))}: {path}")
        if action.ndim != 2:
            raise ValueError(f"Actions must be a 2D array: {path}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"VLA state or action contains NaN or infinite values: {path}")

        camera_metadata = {camera["key"]: camera for camera in metadata["cameras"]}
        for camera_key in camera_keys:
            frames = data[f"observation.images.{camera_key}"]
            camera = camera_metadata[camera_key]
            expected_shape = (steps, int(camera["height"]), int(camera["width"]), 3)
            if frames.shape != expected_shape:
                raise ValueError(
                    f"Camera {camera_key!r} shape is {frames.shape}, expected {expected_shape}: {path}"
                )
            if frames.dtype != np.uint8:
                raise ValueError(f"Camera {camera_key!r} must contain uint8 RGB frames: {path}")

        if bool(metadata.get("success")):
            if not bool(data["terminated"][-1]) or bool(data["truncated"][-1]):
                raise ValueError(f"Successful VLA episode has invalid terminal flags: {path}")
        task = metadata.get("task", {})
        if not str(task.get("instruction", "")).strip():
            raise ValueError(f"VLA episode has an empty language instruction: {path}")

    return metadata


def validate_vla_demo_directory(directory: Path) -> VLADatasetSummary:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing VLA manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_current_schema(manifest, manifest_path)
    entries = manifest.get("episodes", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"VLA manifest contains no episodes: {manifest_path}")

    expected_camera_keys = tuple(manifest.get("camera_keys", []))
    steps: list[int] = []
    object_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    state_dimension: int | None = None
    action_dimension: int | None = None
    for entry in entries:
        episode_path = directory / str(entry["file"])
        metadata = validate_vla_episode_file(episode_path, expected_camera_keys)
        if not bool(metadata["success"]):
            raise ValueError(f"Manifest includes an unsuccessful VLA episode: {episode_path}")
        if int(entry["seed"]) != int(metadata["seed"]):
            raise ValueError(f"Manifest seed does not match VLA episode: {episode_path}")
        if int(entry["steps"]) != int(metadata["steps"]):
            raise ValueError(f"Manifest steps do not match VLA episode: {episode_path}")
        steps.append(int(metadata["steps"]))
        object_counts[str(metadata["task"]["object_key"])] += 1
        target_counts[str(metadata["task"]["target_key"])] += 1
        with np.load(episode_path, allow_pickle=False) as data:
            current_state_dimension = int(data["observation.state"].shape[1])
            current_action_dimension = int(data["action"].shape[1])
        state_dimension = _consistent_dimension(state_dimension, current_state_dimension, "state")
        action_dimension = _consistent_dimension(action_dimension, current_action_dimension, "action")

    return VLADatasetSummary(
        directory=directory,
        episode_count=len(entries),
        total_steps=sum(steps),
        mean_steps=float(np.mean(steps)),
        object_counts=dict(sorted(object_counts.items())),
        target_counts=dict(sorted(target_counts.items())),
        camera_keys=expected_camera_keys,
        state_dimension=int(state_dimension),
        action_dimension=int(action_dimension),
    )


def _require_current_schema(metadata: Mapping[str, Any], path: Path) -> None:
    schema = metadata.get("schema_version")
    action_mode = metadata.get("action_mode")
    if schema != VLA_DATASET_SCHEMA_VERSION or action_mode != ACTION_MODE:
        raise ValueError(
            f"Incompatible VLA data in {path}: expected schema {VLA_DATASET_SCHEMA_VERSION} "
            f"with action_mode={ACTION_MODE!r}, got schema={schema!r}, action_mode={action_mode!r}."
        )


def _consistent_dimension(previous: int | None, current: int, name: str) -> int:
    if previous is not None and previous != current:
        raise ValueError(f"Inconsistent VLA {name} dimension: expected {previous}, got {current}.")
    return current
