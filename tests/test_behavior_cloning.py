import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")

from robot_manipulation_pi0.demos import DEMONSTRATION_SCHEMA_VERSION, collect_episode, save_episode
from robot_manipulation_pi0.learning import BehaviorCloningPolicy, TrainingConfig, load_demonstration_dataset
from robot_manipulation_pi0.learning.bc import train_behavior_cloning
from robot_manipulation_pi0.learning.dataset import observation_to_feature, split_dataset
from robot_manipulation_pi0.sim import ACTION_MODE, PickPlaceConfig


def test_load_demonstration_dataset_and_split(tmp_path) -> None:
    config = _write_demo_dataset(tmp_path)

    dataset = load_demonstration_dataset(tmp_path)
    split = split_dataset(dataset, validation_fraction=0.5, seed=0)

    assert dataset.observations.ndim == 2
    assert dataset.actions.ndim == 2
    assert dataset.actions.shape[1] == config.action_dimension
    assert split.train.observations.shape[0] > 0
    assert split.validation.observations.shape[0] > 0


def test_train_behavior_cloning_saves_loadable_policy(tmp_path) -> None:
    config = _write_demo_dataset(tmp_path)
    dataset = load_demonstration_dataset(tmp_path)
    training_config = TrainingConfig(
        hidden_size=32,
        epochs=3,
        batch_size=32,
        learning_rate=1e-3,
        validation_fraction=0.5,
        seed=0,
        device="cpu",
    )

    policy, metrics = train_behavior_cloning(dataset, training_config)
    predictions = policy.predict(dataset.observations[:5])

    assert predictions.shape == (5, config.action_dimension)
    assert np.isfinite(predictions).all()
    assert metrics["train_loss"] >= 0.0
    assert metrics["validation_loss"] >= 0.0

    checkpoint_path = tmp_path / "bc_policy.pt"
    policy.save(checkpoint_path, training_config, metrics)
    loaded_policy = BehaviorCloningPolicy.load(checkpoint_path, device="cpu")
    loaded_predictions = loaded_policy.predict(dataset.observations[:5])

    np.testing.assert_allclose(loaded_predictions, predictions, atol=1e-6)


def test_observation_to_feature_matches_dataset_layout(tmp_path) -> None:
    config = _write_demo_dataset(tmp_path)
    dataset = load_demonstration_dataset(tmp_path)

    with np.load(tmp_path / "episode_000000_seed_000000.npz", allow_pickle=False) as data:
        observation = {
            "qpos": data["obs_qpos"][0],
            "qvel": data["obs_qvel"][0],
            "ee_pos": data["obs_ee_pos"][0],
            "object_pos": data["obs_object_pos"][0],
            "target_pos": data["obs_target_pos"][0],
            "held": bool(data["obs_held"][0]),
            "finger_contacts": int(data["obs_finger_contacts"][0]),
            "has_held_object": bool(data["obs_has_held_object"][0]),
            "has_lifted_object": bool(data["obs_has_lifted_object"][0]),
            "has_released_after_lift": bool(data["obs_has_released_after_lift"][0]),
            "has_retreated_after_release": bool(data["obs_has_retreated_after_release"][0]),
        }
        stage_index = int(data["obs_oracle_stage_index"][0])
    feature = observation_to_feature(observation, stage_index)

    assert feature.shape == (dataset.observations.shape[1],)
    np.testing.assert_allclose(feature, dataset.observations[0])


def _write_demo_dataset(directory) -> PickPlaceConfig:
    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=450,
    )
    episodes = []
    for index, seed in enumerate((0, 1, 2)):
        episode = collect_episode(seed=seed, config=config)
        filename = f"episode_{index:06d}_seed_{seed:06d}.npz"
        save_episode(directory / filename, episode)
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
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return config
