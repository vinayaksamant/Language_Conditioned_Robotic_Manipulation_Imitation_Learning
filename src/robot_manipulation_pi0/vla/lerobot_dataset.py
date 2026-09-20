from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from robot_manipulation_pi0.sim import VLA_SCENE_ID

from .dataset import (
    VLA_ACTION_NAMES,
    VLA_LEGACY_SCENE_ID,
    VLADatasetSummary,
    scene_id_from_metadata,
    validate_vla_demo_directory,
)
from .observation import ROBOT_STATE_NAMES


LEROBOT_FORMAT_VERSION = 1
SUPPORTED_LEROBOT_MINOR = (0, 6)
DEFAULT_SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
PI0_CAMERA_RENAME_MAP = {
    "observation.images.top": "observation.images.base_0_rgb",
    "observation.images.side": "observation.images.left_wrist_0_rgb",
    "observation.images.wrist": "observation.images.right_wrist_0_rgb",
}


@dataclass(frozen=True)
class RawVLAEpisode:
    index: int
    path: Path
    seed: int
    steps: int
    instruction: str
    object_key: str
    target_key: str
    schema_version: int
    scene_id: str

    @property
    def stratum(self) -> str:
        return f"{self.object_key}:{self.target_key}"


@dataclass(frozen=True)
class EpisodeSplits:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }


@dataclass(frozen=True)
class LeRobotConversionPlan:
    source_directory: Path
    summary: VLADatasetSummary
    episodes: tuple[RawVLAEpisode, ...]
    splits: EpisodeSplits
    split_seed: int
    features: Mapping[str, Mapping[str, Any]]
    use_videos: bool


@dataclass(frozen=True)
class LeRobotConversionResult:
    output_directory: Path
    repo_id: str
    episode_count: int
    frame_count: int
    splits: EpisodeSplits


def plan_lerobot_conversion(
    source_directory: Path,
    *,
    split_seed: int = 0,
    split_fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
    use_videos: bool = False,
    allow_legacy_scene: bool = False,
) -> LeRobotConversionPlan:
    source_directory = Path(source_directory)
    summary = validate_vla_demo_directory(source_directory)
    episodes = _load_raw_episodes(source_directory)
    if len(episodes) != summary.episode_count:
        raise ValueError(
            f"Manifest describes {len(episodes)} episodes but validation found {summary.episode_count}."
        )
    if len(summary.scene_ids) != 1:
        raise ValueError(f"Cannot convert a dataset containing multiple scenes: {summary.scene_ids}")
    scene_id = summary.scene_ids[0]
    if scene_id != VLA_SCENE_ID and not allow_legacy_scene:
        raise ValueError(
            f"Dataset scene {scene_id!r} does not match the current simulator scene {VLA_SCENE_ID!r}. "
            "Collect a new v2 dataset, or pass allow_legacy_scene=True only for an old-scene experiment."
        )
    if scene_id == VLA_LEGACY_SCENE_ID and any(episode.target_key != "green_plate" for episode in episodes):
        raise ValueError("Unversioned legacy data contains unexpected non-green targets and cannot be identified safely.")
    expected_camera_keys = tuple(
        feature_key.removeprefix("observation.images.") for feature_key in PI0_CAMERA_RENAME_MAP
    )
    if summary.camera_keys != expected_camera_keys:
        raise ValueError(
            f"pi0 conversion requires cameras {expected_camera_keys}, got {summary.camera_keys}."
        )
    if summary.state_dimension != len(ROBOT_STATE_NAMES):
        raise ValueError(
            f"pi0 conversion requires {len(ROBOT_STATE_NAMES)} robot-state values, "
            f"got {summary.state_dimension}."
        )
    if summary.action_dimension != len(VLA_ACTION_NAMES):
        raise ValueError(
            f"pi0 conversion requires {len(VLA_ACTION_NAMES)} action values, "
            f"got {summary.action_dimension}."
        )

    splits = make_stratified_episode_splits(
        episodes,
        split_seed=split_seed,
        split_fractions=split_fractions,
    )
    features = _build_features(episodes[0], summary.camera_keys, use_videos=use_videos)
    return LeRobotConversionPlan(
        source_directory=source_directory,
        summary=summary,
        episodes=episodes,
        splits=splits,
        split_seed=split_seed,
        features=features,
        use_videos=use_videos,
    )


def make_stratified_episode_splits(
    episodes: Sequence[RawVLAEpisode],
    *,
    split_seed: int = 0,
    split_fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
) -> EpisodeSplits:
    fractions = _validated_split_fractions(split_fractions)
    if not episodes:
        raise ValueError("Cannot split an empty episode collection.")
    indices = [episode.index for episode in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError("Episode indices must be unique before splitting.")

    split_names = tuple(fractions)
    capacities = _apportion(len(episodes), tuple(fractions.values()))
    remaining = dict(zip(split_names, capacities, strict=True))
    assignments: dict[str, list[int]] = {name: [] for name in split_names}
    groups: dict[str, list[RawVLAEpisode]] = defaultdict(list)
    for episode in episodes:
        groups[episode.stratum].append(episode)

    rng = np.random.default_rng(split_seed)
    for stratum in sorted(groups):
        group = groups[stratum]
        shuffled = [group[index] for index in rng.permutation(len(group))]
        group_counts = Counter({name: 0 for name in split_names})
        tie_order = list(np.asarray(split_names)[rng.permutation(len(split_names))])
        tie_rank = {name: rank for rank, name in enumerate(tie_order)}
        for position, episode in enumerate(shuffled, start=1):
            candidates = [name for name in split_names if remaining[name] > 0]
            if not candidates:
                raise RuntimeError("Split capacity was exhausted before every episode was assigned.")
            selected = max(
                candidates,
                key=lambda name: (
                    position * fractions[name] - group_counts[name],
                    remaining[name],
                    -tie_rank[name],
                ),
            )
            assignments[selected].append(episode.index)
            group_counts[selected] += 1
            remaining[selected] -= 1

    if any(remaining.values()):
        raise RuntimeError(f"Split assignment left unused capacities: {remaining}")
    covered = set().union(*(set(values) for values in assignments.values()))
    if covered != set(indices) or sum(len(values) for values in assignments.values()) != len(indices):
        raise RuntimeError("Episode split does not provide exactly-once coverage.")

    return EpisodeSplits(
        train=tuple(sorted(assignments["train"])),
        val=tuple(sorted(assignments["val"])),
        test=tuple(sorted(assignments["test"])),
    )


def convert_vla_to_lerobot(
    plan: LeRobotConversionPlan,
    output_directory: Path,
    repo_id: str,
    *,
    image_writer_threads: int = 4,
    on_episode: Callable[[int, int, RawVLAEpisode], None] | None = None,
    dataset_class: type | None = None,
) -> LeRobotConversionResult:
    output_directory = Path(output_directory)
    _validate_repo_id(repo_id)
    if image_writer_threads < 0:
        raise ValueError("image_writer_threads cannot be negative.")
    if output_directory.exists():
        raise FileExistsError(f"LeRobot output already exists: {output_directory}")
    if plan.use_videos and shutil.which("ffmpeg") is None:
        raise RuntimeError("Video conversion requires ffmpeg. Omit --videos to use image storage.")
    fps = int(round(plan.summary.fps))
    if not np.isclose(float(fps), plan.summary.fps):
        raise ValueError(f"LeRobot requires an integer frame rate, got {plan.summary.fps} Hz.")

    dataset_class = dataset_class or _load_lerobot_dataset_class()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = output_directory.parent / f".{output_directory.name}.incomplete"
    if temporary_directory.exists():
        raise FileExistsError(
            f"Incomplete conversion directory already exists: {temporary_directory}. "
            "Inspect or remove it before retrying."
        )

    dataset = None
    try:
        dataset = dataset_class.create(
            repo_id=repo_id,
            fps=fps,
            features=dict(plan.features),
            root=temporary_directory,
            robot_type="franka_panda",
            use_videos=plan.use_videos,
            image_writer_processes=0,
            image_writer_threads=image_writer_threads,
        )
        for converted_index, episode in enumerate(plan.episodes):
            if on_episode is not None:
                on_episode(converted_index + 1, len(plan.episodes), episode)
            with np.load(episode.path, allow_pickle=False) as data:
                for frame_index in range(episode.steps):
                    frame: dict[str, Any] = {
                        "observation.state": data["observation.state"][frame_index].astype(
                            np.float32, copy=True
                        ),
                        "action": data["action"][frame_index].astype(np.float32, copy=True),
                        "task": episode.instruction,
                    }
                    for camera_key in plan.summary.camera_keys:
                        feature_key = f"observation.images.{camera_key}"
                        frame[feature_key] = data[feature_key][frame_index].astype(np.uint8, copy=True)
                    dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=plan.use_videos)
        dataset.finalize()
        _write_conversion_metadata(temporary_directory, plan, repo_id)
        temporary_directory.replace(output_directory)
    except Exception:
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                pass
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    return LeRobotConversionResult(
        output_directory=output_directory,
        repo_id=repo_id,
        episode_count=len(plan.episodes),
        frame_count=plan.summary.total_steps,
        splits=plan.splits,
    )


def read_conversion_metadata(directory: Path) -> dict[str, Any]:
    path = Path(directory) / "robot_manipulation_pi0.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing project conversion metadata: {path}. Run scripts/convert_vla_to_lerobot.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def split_indices_from_metadata(metadata: Mapping[str, Any], split: str) -> list[int]:
    splits = metadata.get("splits")
    if not isinstance(splits, Mapping) or split not in splits:
        choices = ", ".join(sorted(splits)) if isinstance(splits, Mapping) else "none"
        raise ValueError(f"Unknown dataset split {split!r}. Available splits: {choices}.")
    indices = [int(index) for index in splits[split]]
    if not indices:
        raise ValueError(f"Dataset split {split!r} is empty.")
    return indices


def _load_raw_episodes(source_directory: Path) -> tuple[RawVLAEpisode, ...]:
    manifest_path = source_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes: list[RawVLAEpisode] = []
    source_root = source_directory.resolve()
    seen_paths: set[Path] = set()
    for index, entry in enumerate(manifest["episodes"]):
        path = (source_directory / str(entry["file"])).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError(f"Episode path escapes the dataset directory: {entry['file']!r}")
        if path in seen_paths:
            raise ValueError(f"Manifest references an episode more than once: {entry['file']!r}")
        seen_paths.add(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        task = metadata["task"]
        episodes.append(
            RawVLAEpisode(
                index=index,
                path=path,
                seed=int(metadata["seed"]),
                steps=int(metadata["steps"]),
                instruction=str(task["instruction"]),
                object_key=str(task["object_key"]),
                target_key=str(task.get("target_key", "green_plate")),
                schema_version=int(metadata["schema_version"]),
                scene_id=scene_id_from_metadata(metadata),
            )
        )
    return tuple(episodes)


def _build_features(
    episode: RawVLAEpisode,
    camera_keys: Sequence[str],
    *,
    use_videos: bool,
) -> dict[str, dict[str, Any]]:
    with np.load(episode.path, allow_pickle=False) as data:
        state_dimension = int(data["observation.state"].shape[1])
        action_dimension = int(data["action"].shape[1])
        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dimension,),
                "names": list(ROBOT_STATE_NAMES[:state_dimension]),
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dimension,),
                "names": list(VLA_ACTION_NAMES[:action_dimension]),
            },
        }
        image_dtype = "video" if use_videos else "image"
        for camera_key in camera_keys:
            frames = data[f"observation.images.{camera_key}"]
            height, width, channels = (int(value) for value in frames.shape[1:])
            features[f"observation.images.{camera_key}"] = {
                "dtype": image_dtype,
                "shape": (channels, height, width),
                "names": ["channels", "height", "width"],
            }
    return features


def _validated_split_fractions(split_fractions: Mapping[str, float]) -> dict[str, float]:
    expected = ("train", "val", "test")
    if set(split_fractions) != set(expected):
        raise ValueError(f"Split fractions must contain exactly {expected}.")
    fractions = {name: float(split_fractions[name]) for name in expected}
    if any(value < 0.0 for value in fractions.values()):
        raise ValueError("Split fractions cannot be negative.")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(fractions.values())}.")
    return fractions


def _apportion(total: int, fractions: Sequence[float]) -> tuple[int, ...]:
    raw = np.asarray(fractions, dtype=float) * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    order = sorted(range(len(fractions)), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(int(value) for value in counts)


def _validate_repo_id(repo_id: str) -> None:
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError("LeRobot --repo-id must have the form 'owner/dataset_name'.")


def _load_lerobot_dataset_class() -> type:
    try:
        version = importlib_metadata.version("lerobot")
    except importlib_metadata.PackageNotFoundError as error:
        raise ModuleNotFoundError(
            "LeRobot is not installed. Create the pi0 environment with: pip install -e '.[pi0]'"
        ) from error
    parsed = _major_minor(version)
    if parsed != SUPPORTED_LEROBOT_MINOR:
        expected = ".".join(str(value) for value in SUPPORTED_LEROBOT_MINOR)
        raise RuntimeError(f"This converter targets LeRobot {expected}.x, but {version} is installed.")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def _major_minor(version: str) -> tuple[int, int]:
    numeric = version.split("+", maxsplit=1)[0].split("-", maxsplit=1)[0]
    pieces = numeric.split(".")
    if len(pieces) < 2 or not pieces[0].isdigit() or not pieces[1].isdigit():
        raise RuntimeError(f"Cannot parse LeRobot version {version!r}.")
    return int(pieces[0]), int(pieces[1])


def _write_conversion_metadata(
    directory: Path,
    plan: LeRobotConversionPlan,
    repo_id: str,
) -> None:
    strata = Counter(episode.stratum for episode in plan.episodes)
    payload = {
        "format_version": LEROBOT_FORMAT_VERSION,
        "repo_id": repo_id,
        "source_directory": str(plan.source_directory.resolve()),
        "source_schema_versions": list(plan.summary.schema_versions),
        "scene_id": plan.summary.scene_ids[0],
        "current_scene_compatible": plan.summary.scene_ids == (VLA_SCENE_ID,),
        "fps": plan.summary.fps,
        "episode_count": len(plan.episodes),
        "frame_count": plan.summary.total_steps,
        "camera_keys": list(plan.summary.camera_keys),
        "camera_rename_map": PI0_CAMERA_RENAME_MAP,
        "use_videos": plan.use_videos,
        "split_seed": plan.split_seed,
        "splits": plan.splits.as_dict(),
        "strata": dict(sorted(strata.items())),
        "episodes": [
            {
                "index": episode.index,
                "source_file": episode.path.name,
                "seed": episode.seed,
                "steps": episode.steps,
                "instruction": episode.instruction,
                "object_key": episode.object_key,
                "target_key": episode.target_key,
            }
            for episode in plan.episodes
        ],
    }
    (directory / "robot_manipulation_pi0.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
