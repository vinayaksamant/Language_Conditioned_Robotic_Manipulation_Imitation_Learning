import json
from pathlib import Path

import numpy as np
import pytest

from robot_manipulation_pi0.sim import ACTION_MODE, VLA_SCENE_ID
from robot_manipulation_pi0.vla import (
    PI0_CAMERA_RENAME_MAP,
    VLA_CAMERA_SPECS,
    VLA_DATASET_SCHEMA_VERSION,
    CameraSpec,
    LanguagePickPlaceTask,
    Pi0TrainingRequest,
    RawVLAEpisode,
    VLAEpisode,
    build_pi0_training_command,
    convert_vla_to_lerobot,
    existing_lerobot_conversion,
    make_stratified_episode_splits,
    plan_lerobot_conversion,
    resolve_pi0_checkpoint,
    save_vla_episode,
)


def test_stratified_split_has_exact_coverage_and_balanced_tasks(tmp_path: Path) -> None:
    episodes = []
    index = 0
    for object_key in ("red_cube", "blue_cylinder"):
        for target_key in ("green_plate", "yellow_plate"):
            for local_index in range(25):
                episodes.append(
                    RawVLAEpisode(
                        index=index,
                        path=tmp_path / f"{index}.npz",
                        seed=index,
                        steps=10,
                        instruction=f"Move {object_key} to {target_key}",
                        object_key=object_key,
                        target_key=target_key,
                        schema_version=VLA_DATASET_SCHEMA_VERSION,
                        scene_id=VLA_SCENE_ID,
                    )
                )
                index += 1

    splits = make_stratified_episode_splits(episodes, split_seed=7)
    assert (len(splits.train), len(splits.val), len(splits.test)) == (80, 10, 10)
    all_indices = [*splits.train, *splits.val, *splits.test]
    assert len(all_indices) == len(set(all_indices)) == 100
    assert set(all_indices) == set(range(100))

    by_index = {episode.index: episode for episode in episodes}
    for split in (splits.train, splits.val, splits.test):
        represented = {by_index[index].stratum for index in split}
        assert represented == {
            "red_cube:green_plate",
            "red_cube:yellow_plate",
            "blue_cylinder:green_plate",
            "blue_cylinder:yellow_plate",
        }


def test_raw_v2_conversion_writes_lerobot_frames_and_project_metadata(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    _write_raw_dataset(raw_directory, episode_count=4)
    plan = plan_lerobot_conversion(raw_directory, split_seed=3)

    output_directory = tmp_path / "lerobot"
    FakeLeRobotDataset.instances.clear()
    result = convert_vla_to_lerobot(
        plan,
        output_directory,
        "local/test_dataset",
        dataset_class=FakeLeRobotDataset,
    )

    assert result.episode_count == 4
    assert result.frame_count == 12
    assert output_directory.exists()
    metadata = json.loads(
        (output_directory / "robot_manipulation_pi0.json").read_text(encoding="utf-8")
    )
    assert metadata["scene_id"] == VLA_SCENE_ID
    assert metadata["camera_rename_map"] == PI0_CAMERA_RENAME_MAP
    assert metadata["episodes"][0]["seed"] == 0
    assert metadata["episodes"][0]["source_file"] == "episode_000000.npz"
    writer = FakeLeRobotDataset.instances[-1]
    assert len(writer.episodes) == 4
    assert writer.episodes[0][0]["task"].startswith("Pick")
    assert writer.episodes[0][0]["observation.images.top"].shape == (16, 16, 3)
    assert writer.features["observation.images.top"]["shape"] == (3, 16, 16)


def test_pi0_training_command_uses_train_split_and_camera_mapping(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "lerobot"
    (dataset_directory / "meta").mkdir(parents=True)
    (dataset_directory / "meta" / "info.json").write_text("{}", encoding="utf-8")
    project_metadata = {
        "repo_id": "local/test_dataset",
        "scene_id": VLA_SCENE_ID,
        "camera_rename_map": PI0_CAMERA_RENAME_MAP,
        "splits": {"train": [0, 2, 3], "val": [1], "test": [4]},
    }
    (dataset_directory / "robot_manipulation_pi0.json").write_text(
        json.dumps(project_metadata), encoding="utf-8"
    )
    request = Pi0TrainingRequest(
        dataset_directory=dataset_directory,
        output_directory=tmp_path / "output",
        steps=100,
    )

    command = build_pi0_training_command(request)
    assert "--dataset.episodes=[0,2,3]" in command
    assert "--policy.path=lerobot/pi0_base" in command
    assert "--policy.train_expert_only=true" in command
    rename_argument = next(argument for argument in command if argument.startswith("--rename_map="))
    assert json.loads(rename_argument.split("=", maxsplit=1)[1]) == PI0_CAMERA_RENAME_MAP


def test_existing_lerobot_conversion_is_reused_after_validation(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "lerobot"
    (dataset_directory / "meta").mkdir(parents=True)
    (dataset_directory / "meta" / "info.json").write_text("{}", encoding="utf-8")
    metadata = {
        "format_version": 1,
        "repo_id": "local/test_dataset",
        "scene_id": VLA_SCENE_ID,
        "episode_count": 4,
        "frame_count": 12,
        "splits": {"train": [0, 1], "val": [2], "test": [3]},
    }
    (dataset_directory / "robot_manipulation_pi0.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    result = existing_lerobot_conversion(
        dataset_directory,
        expected_repo_id="local/test_dataset",
    )

    assert result is not None
    assert result.episode_count == 4
    assert result.frame_count == 12
    assert result.splits.train == (0, 1)


def test_resolve_pi0_checkpoint_finds_last_pretrained_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"type": "pi0"}), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    assert resolve_pi0_checkpoint(tmp_path / "run") == checkpoint.resolve()


def test_conversion_rejects_duplicate_manifest_episode(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    _write_raw_dataset(raw_directory, episode_count=2)
    manifest_path = raw_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episodes"][1] = dict(manifest["episodes"][0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="more than once"):
        plan_lerobot_conversion(raw_directory)


class FakeLeRobotDataset:
    instances: list["FakeLeRobotDataset"] = []

    def __init__(self, root: Path, features: dict[str, dict[str, object]]) -> None:
        self.root = root
        self.features = features
        self.current: list[dict[str, object]] = []
        self.episodes: list[list[dict[str, object]]] = []
        self.finalized = False
        self.root.mkdir(parents=True)
        self.instances.append(self)

    @classmethod
    def create(cls, **kwargs):
        return cls(Path(kwargs["root"]), kwargs["features"])

    def add_frame(self, frame: dict[str, object]) -> None:
        self.current.append(frame)

    def save_episode(self, parallel_encoding: bool = True) -> None:
        del parallel_encoding
        self.episodes.append(self.current)
        self.current = []

    def finalize(self) -> None:
        if self.finalized:
            return
        (self.root / "meta").mkdir(exist_ok=True)
        (self.root / "meta" / "info.json").write_text("{}", encoding="utf-8")
        self.finalized = True


def _write_raw_dataset(directory: Path, episode_count: int) -> None:
    directory.mkdir(parents=True)
    entries = []
    camera_specs = tuple(
        CameraSpec(spec.key, spec.camera_name, width=16, height=16) for spec in VLA_CAMERA_SPECS
    )
    for index in range(episode_count):
        object_key = "red_cube" if index % 2 == 0 else "blue_cylinder"
        target_key = "green_plate" if index % 4 < 2 else "yellow_plate"
        task = LanguagePickPlaceTask(
            instruction=f"Pick {object_key} and place it on the {target_key}.",
            object_key=object_key,
            target_key=target_key,
            canonical_label=f"pick_{object_key}_to_{target_key}",
        )
        episode = VLAEpisode(
            seed=index,
            task=task,
            success=True,
            states=np.zeros((3, 8), dtype=np.float32),
            images={
                spec.key: np.full((3, 16, 16, 3), index, dtype=np.uint8)
                for spec in camera_specs
            },
            actions=np.zeros((3, 8), dtype=np.float32),
            rewards=np.array([-1.0, -0.5, 1.0], dtype=np.float32),
            terminated=np.array([False, False, True]),
            truncated=np.zeros(3, dtype=bool),
            oracle_stage_index=np.array([0, 3, 8], dtype=np.int64),
            camera_specs=camera_specs,
            fps=10.0,
            simulation_steps=12,
            record_every=4,
        )
        filename = f"episode_{index:06d}.npz"
        save_vla_episode(directory / filename, episode)
        entries.append(
            {
                "seed": index,
                "file": filename,
                "steps": episode.steps,
                "object_key": object_key,
                "target_key": target_key,
                "instruction": task.instruction,
            }
        )
    manifest = {
        "schema_version": VLA_DATASET_SCHEMA_VERSION,
        "scene_id": VLA_SCENE_ID,
        "action_mode": ACTION_MODE,
        "camera_keys": [spec.key for spec in camera_specs],
        "fps": 10.0,
        "episodes": entries,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
