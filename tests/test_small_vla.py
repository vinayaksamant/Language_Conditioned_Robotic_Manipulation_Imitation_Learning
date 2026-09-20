from pathlib import Path

import numpy as np
import torch

from robot_manipulation_pi0.sim import (
    ACTION_MODE,
    VLA_SCENE_ID,
    MultiObjectPickPlaceEnvironment,
    PickPlaceConfig,
)
from robot_manipulation_pi0.sim.oracle import STAGE_NAMES
from robot_manipulation_pi0.vla import VLAObservation
from robot_manipulation_pi0.vla.small_vla import (
    SMALL_VLA_CHECKPOINT_VERSION,
    FRANKA_ARM_CONTROL_MAX,
    FRANKA_ARM_CONTROL_MIN,
    JOINT_DELTA_ACTION_REPRESENTATION,
    STATE_TRANSITION_ACTION_REPRESENTATION,
    TRANSITION_SMALL_VLA_CHECKPOINT_VERSION,
    SmallVLAEpisodeRecord,
    SmallVLAModel,
    SmallVLAModelConfig,
    SmallVLAPolicy,
    Vocabulary,
    _cuda_architecture_supported,
    _decode_model_actions,
    _encode_model_actions,
    _warm_start_small_vla,
    _wrist_target_is_aligned,
    split_small_vla_episodes,
    tokenize_instruction,
)


def test_vocabulary_tokenizes_language_and_handles_unknown_words() -> None:
    vocabulary = Vocabulary.build(
        [
            "Pick the red cube and place it on the green plate.",
            "Move the blue cylinder to the yellow plate.",
        ]
    )
    encoded = vocabulary.encode("Pick the unseen object.", max_tokens=8)

    assert tokenize_instruction("Red cube, please!") == ("red", "cube", "please")
    assert encoded.shape == (8,)
    assert encoded[0] != 0
    assert encoded[2] == 1
    assert np.all(encoded[4:] == 0)


def test_cuda_architecture_check_accepts_compatible_pascal_binary() -> None:
    assert _cuda_architecture_supported((6, 1), ("sm_60", "sm_70"))
    assert not _cuda_architecture_supported((6, 1), ("sm_75", "sm_80"))


def test_small_vla_model_returns_action_and_stage_logits() -> None:
    config = SmallVLAModelConfig(
        camera_keys=("top", "side"),
        image_size=32,
        vocabulary_size=12,
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
    )
    model = SmallVLAModel(config)
    actions, stages = model(
        torch.zeros(3, 2, 3, 32, 32),
        torch.zeros(3, 8),
        torch.tensor([[2, 3, 0], [4, 5, 6], [2, 0, 0]]),
        task_indices=torch.tensor([0, 1, 3]),
    )

    assert actions.shape == (3, 8)
    assert stages.shape == (3, len(STAGE_NAMES))


def test_small_vla_defaults_to_direct_oracle_joint_commands() -> None:
    config = SmallVLAModelConfig()

    assert config.action_representation == JOINT_DELTA_ACTION_REPRESENTATION
    assert config.joint_delta_scale == 0.02
    assert config.control_repeat == 1
    assert config.visual_grid_size == 4
    assert config.task_conditioned_actions
    assert config.task_count == 4


def test_small_vla_training_can_select_ground_truth_stage_expert() -> None:
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=4,
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
        task_conditioned_actions=False,
    )
    model = SmallVLAModel(config)
    with torch.no_grad():
        model.action_head.weight.zero_()
        model.action_head.bias.copy_(
            torch.arange(len(STAGE_NAMES), dtype=torch.float32).repeat_interleave(8)
        )
    actions, _ = model(
        torch.zeros(2, 1, 3, 32, 32),
        torch.zeros(2, 8),
        torch.tensor([[2, 0], [3, 0]]),
        stage_indices=torch.tensor([2, 5]),
    )

    np.testing.assert_allclose(actions.detach().numpy()[0], 2.0)
    np.testing.assert_allclose(actions.detach().numpy()[1], 5.0)


def test_small_vla_routes_actions_to_language_task_experts() -> None:
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=4,
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
    )
    model = SmallVLAModel(config)
    with torch.no_grad():
        model.action_head.weight.zero_()
        biases = torch.zeros(config.task_count, len(STAGE_NAMES), 8)
        for task_index in range(config.task_count):
            biases[task_index] = float(task_index)
        model.action_head.bias.copy_(biases.flatten())

    actions, _ = model(
        torch.zeros(2, 1, 3, 32, 32),
        torch.zeros(2, 8),
        torch.tensor([[2, 0], [3, 0]]),
        stage_indices=torch.tensor([0, 0]),
        task_indices=torch.tensor([0, 3]),
    )

    np.testing.assert_allclose(actions.detach().numpy()[0], 0.0)
    np.testing.assert_allclose(actions.detach().numpy()[1], 3.0)


def test_small_vla_learns_task_routing_without_external_task_index() -> None:
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=4,
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
    )
    model = SmallVLAModel(config)
    with torch.no_grad():
        model.task_head.weight.zero_()
        model.task_head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))
        model.action_head.weight.zero_()
        biases = torch.zeros(config.task_count, len(STAGE_NAMES), 8)
        for task_index in range(config.task_count):
            biases[task_index] = float(task_index)
        model.action_head.bias.copy_(biases.flatten())

    actions, _, tasks = model.forward_with_aux(
        torch.zeros(2, 1, 3, 32, 32),
        torch.zeros(2, 8),
        torch.tensor([[2, 0], [3, 0]]),
        stage_indices=torch.tensor([0, 0]),
    )

    assert tasks is not None
    np.testing.assert_allclose(actions.detach().numpy(), 3.0)
    np.testing.assert_array_equal(tasks.argmax(dim=1).numpy(), np.array([3, 3]))


def test_strict_vla_policy_grounds_language_and_rejects_privileged_feedback() -> None:
    vocabulary = Vocabulary.build(["Pick the blue cylinder and place it on the yellow plate."])
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=len(vocabulary),
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
    )
    model = SmallVLAModel(config)
    with torch.no_grad():
        model.task_head.weight.zero_()
        model.task_head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))
    policy = SmallVLAPolicy(
        model=model,
        vocabulary=vocabulary,
        state_mean=np.zeros(8, dtype=np.float32),
        state_std=np.ones(8, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        device="cpu",
    )

    task = policy.ground_instruction(
        "Pick the blue cylinder and place it on the yellow plate."
    )

    assert (task.object_key, task.target_key) == ("blue_cylinder", "yellow_plate")
    assert task.learned
    assert task.confidence > 0.99
    with np.testing.assert_raises_regex(RuntimeError, "does not accept simulator"):
        policy.update_feedback(held=True)


def test_wrist_target_alignment_requires_requested_plate_at_gripper_center() -> None:
    centered_green = np.zeros((256, 256, 3), dtype=np.uint8)
    centered_green[5:45, 108:148] = np.array([20, 220, 40], dtype=np.uint8)
    offset_green = np.roll(centered_green, shift=90, axis=1)
    centered_yellow = np.zeros((256, 256, 3), dtype=np.uint8)
    centered_yellow[5:45, 108:148] = np.array([230, 210, 20], dtype=np.uint8)

    assert _wrist_target_is_aligned(centered_green, "green_plate", placement=False)
    assert not _wrist_target_is_aligned(offset_green, "green_plate", placement=False)
    assert _wrist_target_is_aligned(centered_yellow, "yellow_plate", placement=False)
    assert not _wrist_target_is_aligned(centered_yellow, "green_plate", placement=False)


def test_small_vla_policy_stage_filter_requires_two_ordered_predictions() -> None:
    vocabulary = Vocabulary.build(["Pick the red cube."])
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=len(vocabulary),
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
    )
    model = SmallVLAModel(config)
    with torch.no_grad():
        model.stage_head.weight.zero_()
        model.stage_head.bias.zero_()
        model.stage_head.bias[1] = 10.0
    policy = SmallVLAPolicy(
        model=model,
        vocabulary=vocabulary,
        state_mean=np.zeros(8, dtype=np.float32),
        state_std=np.ones(8, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        device="cpu",
    )
    observation = VLAObservation(
        state=np.zeros(8, dtype=np.float32),
        images={"top": np.zeros((32, 32, 3), dtype=np.uint8)},
        instruction="Pick the red cube and place it on the green plate.",
    )

    _, first_stage = policy.predict(observation)
    _, second_stage = policy.predict(observation)
    _, third_stage = policy.predict(observation)

    assert first_stage == "approach"
    assert second_stage == "approach"
    assert third_stage == "descend"


def test_small_vla_aligned_contact_loss_advances_to_release() -> None:
    policy = _stage_test_policy()
    policy._set_stage(STAGE_NAMES.index("descend_place"))
    policy._held = True
    policy._has_held_object = True
    policy._target_aligned = True

    policy.update_feedback(held=False, has_lifted_object=True)

    assert policy._stage_index == STAGE_NAMES.index("release")


def test_small_vla_release_has_deterministic_retreat_transition() -> None:
    policy = _stage_test_policy()
    policy._set_stage(STAGE_NAMES.index("release"))

    for _ in range(7):
        policy._update_stage(predicted_stage=STAGE_NAMES.index("release"))
    assert policy._stage_index == STAGE_NAMES.index("release")

    policy._update_stage(predicted_stage=STAGE_NAMES.index("release"))
    assert policy._stage_index == STAGE_NAMES.index("retreat")


def test_small_vla_warm_starts_from_previous_camera_scene(tmp_path: Path) -> None:
    source_vocabulary = Vocabulary(("<pad>", "<unk>", "old"))
    target_vocabulary = Vocabulary(("<pad>", "<unk>", "new"))
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=3,
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
    )
    source_model = SmallVLAModel(config)
    target_model = SmallVLAModel(config)
    path = tmp_path / "previous_scene.pt"
    torch.save(
        {
            "checkpoint_version": 5,
            "action_mode": ACTION_MODE,
            "scene_id": "franka_panda_two_object_two_target_calibrated_cameras_v3",
            "model_state_dict": source_model.state_dict(),
            "model_config": config.__dict__,
            "vocabulary": list(source_vocabulary.tokens),
        },
        path,
    )

    _warm_start_small_vla(target_model, target_vocabulary, path)

    np.testing.assert_allclose(
        target_model.language_embedding.weight.detach().numpy()[:2],
        source_model.language_embedding.weight.detach().numpy()[:2],
    )


def test_joint_delta_action_representation_round_trips_environment_actions() -> None:
    config = SmallVLAModelConfig(
        action_representation=JOINT_DELTA_ACTION_REPRESENTATION,
        joint_delta_scale=0.02,
        control_repeat=1,
    )
    states = np.asarray(
        [[0.1, -0.2, 0.3, -1.5, 0.2, 1.4, -0.7, 0.04]],
        dtype=np.float32,
    )
    desired_delta = np.asarray(
        [[0.01, -0.02, 0.005, -0.01, 0.0, 0.02, -0.015]],
        dtype=np.float32,
    )
    control_min = np.asarray(FRANKA_ARM_CONTROL_MIN, dtype=np.float32)
    control_max = np.asarray(FRANKA_ARM_CONTROL_MAX, dtype=np.float32)
    arm_targets = states[:, :7] + desired_delta
    actions = np.empty((1, 8), dtype=np.float32)
    actions[:, :7] = 2.0 * (arm_targets - control_min) / (control_max - control_min) - 1.0
    actions[:, 7] = 1.0

    encoded = _encode_model_actions(actions, states, config)
    decoded = _decode_model_actions(encoded, states, config)

    np.testing.assert_allclose(encoded[:, :7], desired_delta / config.joint_delta_scale, atol=2e-5)
    np.testing.assert_allclose(decoded, actions, atol=1e-5)


def test_state_transition_representation_uses_next_recorded_state() -> None:
    config = SmallVLAModelConfig(
        action_representation=STATE_TRANSITION_ACTION_REPRESENTATION,
        joint_delta_scale=0.09,
        control_repeat=4,
    )
    states = np.zeros((3, 8), dtype=np.float32)
    states[1, :7] = 0.045
    states[2, :7] = 0.09
    actions = np.zeros((3, 8), dtype=np.float32)
    actions[:, 7] = np.asarray([-1.0, 1.0, -1.0])

    encoded = _encode_model_actions(actions, states, config)

    np.testing.assert_allclose(encoded[0, :7], 0.5)
    np.testing.assert_allclose(encoded[1, :7], 0.5)
    np.testing.assert_allclose(encoded[2, :7], 0.0)
    np.testing.assert_allclose(encoded[:, 7], actions[:, 7])


def test_small_vla_control_ranges_match_mujoco_model() -> None:
    environment = MultiObjectPickPlaceEnvironment(
        PickPlaceConfig(
            robot_name="franka_panda",
            object_names=("red_cube", "blue_cylinder"),
            workspace_size=0.5,
        )
    )

    np.testing.assert_allclose(environment.model.actuator_ctrlrange[:7, 0], FRANKA_ARM_CONTROL_MIN)
    np.testing.assert_allclose(environment.model.actuator_ctrlrange[:7, 1], FRANKA_ARM_CONTROL_MAX)


def test_small_vla_split_is_episode_level_and_stratified(tmp_path: Path) -> None:
    records = []
    index = 0
    for object_key in ("red_cube", "blue_cylinder"):
        for target_key in ("green_plate", "yellow_plate"):
            for _ in range(5):
                records.append(
                    SmallVLAEpisodeRecord(
                        index=index,
                        path=tmp_path / f"episode_{index}.npz",
                        seed=index,
                        steps=3,
                        instruction=f"Move {object_key} to {target_key}",
                        object_key=object_key,
                        target_key=target_key,
                    )
                )
                index += 1

    train, validation = split_small_vla_episodes(records, validation_fraction=0.2, seed=7)

    assert len(train) == 16
    assert len(validation) == 4
    assert not {record.index for record in train}.intersection(record.index for record in validation)
    assert {record.stratum for record in validation} == {
        "red_cube:green_plate",
        "red_cube:yellow_plate",
        "blue_cylinder:green_plate",
        "blue_cylinder:yellow_plate",
    }


def test_paired_small_vla_split_keeps_scene_seeds_together(tmp_path: Path) -> None:
    records = []
    index = 0
    for scene_seed in range(5):
        for object_key in ("red_cube", "blue_cylinder"):
            for target_key in ("green_plate", "yellow_plate"):
                records.append(
                    SmallVLAEpisodeRecord(
                        index=index,
                        path=tmp_path / f"episode_{index}.npz",
                        seed=scene_seed,
                        steps=3,
                        instruction=f"Move {object_key} to {target_key}",
                        object_key=object_key,
                        target_key=target_key,
                    )
                )
                index += 1

    train, validation = split_small_vla_episodes(
        records,
        validation_fraction=0.2,
        seed=7,
        group_by_seed=True,
    )

    train_seeds = {record.seed for record in train}
    validation_seeds = {record.seed for record in validation}
    assert len(train) == 16
    assert len(validation) == 4
    assert train_seeds.isdisjoint(validation_seeds)


def test_small_vla_checkpoint_loads_and_predicts(tmp_path: Path) -> None:
    vocabulary = Vocabulary.build(["Pick the red cube and place it on the green plate."])
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=len(vocabulary),
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
    )
    model = SmallVLAModel(config)
    path = tmp_path / "small_vla.pt"
    torch.save(
        {
            "checkpoint_version": SMALL_VLA_CHECKPOINT_VERSION,
            "action_mode": ACTION_MODE,
            "scene_id": VLA_SCENE_ID,
            "model_state_dict": model.state_dict(),
            "model_config": config.__dict__,
            "training_config": {},
            "vocabulary": list(vocabulary.tokens),
            "state_mean": np.zeros(8, dtype=np.float32),
            "state_std": np.ones(8, dtype=np.float32),
            "action_mean": np.zeros(8, dtype=np.float32),
            "action_std": np.ones(8, dtype=np.float32),
            "history": [],
            "best_epoch": 1,
        },
        path,
    )
    policy = SmallVLAPolicy.load(path, device="cpu")
    action, stage = policy.predict(
        VLAObservation(
            state=np.zeros(8, dtype=np.float32),
            images={"top": np.zeros((64, 64, 3), dtype=np.uint8)},
            instruction="Pick the red cube and place it on the green plate.",
        )
    )

    assert action.shape == (8,)
    assert np.isfinite(action).all()
    assert isinstance(stage, str)


def test_small_vla_loads_v3_two_by_two_visual_encoder(tmp_path: Path) -> None:
    vocabulary = Vocabulary.build(["Pick the red cube and place it on the green plate."])
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=len(vocabulary),
        visual_dimension=16,
        visual_grid_size=2,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
        action_representation=STATE_TRANSITION_ACTION_REPRESENTATION,
        control_repeat=2,
        task_conditioned_actions=False,
        learned_task_routing=False,
        strict_vla_inference=False,
    )
    model = SmallVLAModel(config)
    legacy_model_config = dict(config.__dict__)
    legacy_model_config.pop("visual_grid_size")
    path = tmp_path / "small_vla_v3.pt"
    torch.save(
        {
            "checkpoint_version": TRANSITION_SMALL_VLA_CHECKPOINT_VERSION,
            "action_mode": ACTION_MODE,
            "scene_id": VLA_SCENE_ID,
            "model_state_dict": model.state_dict(),
            "model_config": legacy_model_config,
            "training_config": {},
            "vocabulary": list(vocabulary.tokens),
            "state_mean": np.zeros(8, dtype=np.float32),
            "state_std": np.ones(8, dtype=np.float32),
            "action_mean": np.zeros(8, dtype=np.float32),
            "action_std": np.ones(8, dtype=np.float32),
            "history": [],
            "best_epoch": 1,
        },
        path,
    )

    policy = SmallVLAPolicy.load(path, device="cpu")

    assert policy.model.config.visual_grid_size == 2


def _stage_test_policy() -> SmallVLAPolicy:
    vocabulary = Vocabulary.build(["Pick the red cube and place it on the green plate."])
    config = SmallVLAModelConfig(
        camera_keys=("top",),
        image_size=32,
        vocabulary_size=len(vocabulary),
        visual_dimension=16,
        language_dimension=8,
        state_feature_dimension=12,
        hidden_size=24,
        dropout=0.0,
        strict_vla_inference=False,
    )
    return SmallVLAPolicy(
        model=SmallVLAModel(config),
        vocabulary=vocabulary,
        state_mean=np.zeros(8, dtype=np.float32),
        state_std=np.ones(8, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        device="cpu",
    )
