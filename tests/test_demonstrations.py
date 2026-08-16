import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_manipulation_pi0.demos import (
    DEMONSTRATION_SCHEMA_VERSION,
    collect_episode,
    save_episode,
    validate_demo_directory,
    validate_episode_file,
)
from robot_manipulation_pi0.sim import ACTION_MODE, PickPlaceConfig


def test_collect_episode_returns_aligned_successful_trajectory() -> None:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=450,
    )
    episode = collect_episode(seed=0, config=config)

    assert episode.success
    assert episode.steps == len(episode.actions)
    assert episode.actions.shape == (episode.steps, config.action_dimension)
    assert episode.rewards.shape == (episode.steps,)
    assert episode.terminated.shape == (episode.steps,)
    assert episode.truncated.shape == (episode.steps,)
    assert episode.observations["qpos"].shape[0] == episode.steps
    assert episode.observations["qvel"].shape[0] == episode.steps
    assert episode.observations["ee_pos"].shape == (episode.steps, 3)
    assert episode.observations["object_pos"].shape == (episode.steps, 3)
    assert episode.observations["target_pos"].shape == (episode.steps, 3)


def test_collect_episode_step_callback_observes_exact_saved_rollout() -> None:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=300,
    )
    observed_steps: list[int] = []

    def observe_step(environment, policy, observation, action, result) -> None:
        del environment, policy, action, result
        observed_steps.append(int(observation["step_count"]))

    episode = collect_episode(seed=0, config=config, step_callback=observe_step)

    assert episode.success
    assert observed_steps == list(range(1, episode.steps + 1))


def test_save_episode_writes_npz_schema(tmp_path) -> None:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=450,
    )
    episode = collect_episode(seed=0, config=config)
    output_path = tmp_path / "episode_seed_000000.npz"

    save_episode(output_path, episode)

    with np.load(output_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        assert metadata["seed"] == 0
        assert metadata["schema_version"] == DEMONSTRATION_SCHEMA_VERSION
        assert metadata["action_mode"] == ACTION_MODE
        assert metadata["success"] is True
        assert metadata["instruction"]["canonical_label"] == "pick_cube_to_zone"
        assert data["actions"].shape == (episode.steps, config.action_dimension)
        assert data["rewards"].shape == (episode.steps,)
        assert data["obs_qpos"].shape[0] == episode.steps
        assert data["obs_object_pos"].shape == (episode.steps, 3)


def test_validate_episode_file_accepts_saved_episode(tmp_path) -> None:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=450,
    )
    episode = collect_episode(seed=0, config=config)
    output_path = tmp_path / "episode_seed_000000.npz"
    save_episode(output_path, episode)

    metadata, shapes = validate_episode_file(output_path, expected_action_dimension=config.action_dimension)

    assert metadata["success"] is True
    assert shapes["actions"] == (episode.steps, config.action_dimension)
    assert shapes["obs_ee_pos"] == (episode.steps, 3)


def test_validate_demo_directory_accepts_manifest_dataset(tmp_path) -> None:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=450,
    )
    episodes = []
    for index, seed in enumerate((0, 1)):
        episode = collect_episode(seed=seed, config=config)
        filename = f"episode_{index:06d}_seed_{seed:06d}.npz"
        save_episode(tmp_path / filename, episode)
        episodes.append({"seed": seed, "file": filename, "steps": episode.steps})

    manifest = {
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "action_mode": ACTION_MODE,
        "task": "pick_place_single_object",
        "robot": config.robot_name,
        "successful_episodes": len(episodes),
        "attempted_episodes": len(episodes),
        "seed_start": 0,
        "max_steps": config.max_steps,
        "workspace_size": config.workspace_size,
        "episodes": episodes,
        "failed_seeds": [],
    }
    Path(tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = validate_demo_directory(tmp_path, expected_action_dimension=config.action_dimension)

    assert summary.episode_count == 2
    assert summary.total_steps == sum(entry["steps"] for entry in episodes)
    assert summary.action_dimension == config.action_dimension
