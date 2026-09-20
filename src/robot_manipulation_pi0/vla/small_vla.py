from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from robot_manipulation_pi0.sim import (
    ACTION_MODE,
    VLA_SCENE_ID,
    VLA_SCENE_OBJECTS,
    VLA_SCENE_TARGETS,
)
from robot_manipulation_pi0.sim.oracle import STAGE_NAMES

from .dataset import VLA_DATASET_SCHEMA_VERSION
from .observation import ROBOT_STATE_NAMES, VLAObservation
from .tasks import resolve_task_instruction


SMALL_VLA_CHECKPOINT_VERSION = 7
LEGACY_SMALL_VLA_CHECKPOINT_VERSION = 1
DELTA_SMALL_VLA_CHECKPOINT_VERSION = 2
TRANSITION_SMALL_VLA_CHECKPOINT_VERSION = 3
DIRECT_ACTION_SMALL_VLA_CHECKPOINT_VERSION = 4
TASK_EXPERT_SMALL_VLA_CHECKPOINT_VERSION = 5
CENTERED_CAMERA_SMALL_VLA_CHECKPOINT_VERSION = 6
ABSOLUTE_ACTION_REPRESENTATION = "absolute_normalized_v1"
JOINT_DELTA_ACTION_REPRESENTATION = "joint_delta_v1"
STATE_TRANSITION_ACTION_REPRESENTATION = "state_transition_delta_v1"
FRANKA_ARM_CONTROL_MIN = (
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
)
FRANKA_ARM_CONTROL_MAX = (
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973,
)
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
VLA_TASK_KEYS = tuple(
    (scene_object.key, scene_target.key)
    for scene_object in VLA_SCENE_OBJECTS
    for scene_target in VLA_SCENE_TARGETS
)
VLA_TASK_TO_INDEX = {task_key: index for index, task_key in enumerate(VLA_TASK_KEYS)}
_OPEN_GRIPPER_STAGES = {"approach", "descend", "release", "retreat"}
_CLOSE_RETRY_STEPS = 40
_RELEASE_STEPS = 8
_WARM_START_SCENE_IDS = {
    VLA_SCENE_ID,
    "franka_panda_two_object_two_target_calibrated_cameras_v3",
}


@dataclass(frozen=True)
class SmallVLAEpisodeRecord:
    index: int
    path: Path
    seed: int
    steps: int
    instruction: str
    object_key: str
    target_key: str

    @property
    def stratum(self) -> str:
        return f"{self.object_key}:{self.target_key}"


@dataclass(frozen=True)
class SmallVLAModelConfig:
    camera_keys: tuple[str, ...] = ("top", "side", "wrist")
    image_size: int = 96
    state_dimension: int = len(ROBOT_STATE_NAMES)
    action_dimension: int = 8
    vocabulary_size: int = 2
    max_tokens: int = 24
    visual_dimension: int = 64
    visual_grid_size: int = 4
    language_dimension: int = 48
    state_feature_dimension: int = 64
    hidden_size: int = 256
    dropout: float = 0.1
    action_representation: str = JOINT_DELTA_ACTION_REPRESENTATION
    joint_delta_scale: float = 0.02
    control_repeat: int = 1
    stage_conditioned_actions: bool = True
    task_count: int = len(VLA_TASK_KEYS)
    task_conditioned_actions: bool = True
    learned_task_routing: bool = True
    strict_vla_inference: bool = True

    def __post_init__(self) -> None:
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("Small VLA camera keys must be non-empty and unique.")
        integer_values = (
            self.image_size,
            self.state_dimension,
            self.action_dimension,
            self.vocabulary_size,
            self.max_tokens,
            self.visual_dimension,
            self.visual_grid_size,
            self.language_dimension,
            self.state_feature_dimension,
            self.hidden_size,
            self.task_count,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("Small VLA model dimensions must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Small VLA dropout must be in [0, 1).")
        if self.action_representation not in {
            ABSOLUTE_ACTION_REPRESENTATION,
            JOINT_DELTA_ACTION_REPRESENTATION,
            STATE_TRANSITION_ACTION_REPRESENTATION,
        }:
            raise ValueError(f"Unsupported small VLA action representation: {self.action_representation}")
        if self.joint_delta_scale <= 0.0:
            raise ValueError("Small VLA joint delta scale must be positive.")
        if self.control_repeat <= 0:
            raise ValueError("Small VLA control repeat must be positive.")


@dataclass(frozen=True)
class SmallVLATrainingConfig:
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    stage_loss_weight: float = 0.1
    task_loss_weight: float = 0.2
    seed: int = 0
    device: str = "auto"
    episodes_per_task: int | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Training epochs and batch size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Learning rate must be positive and weight decay cannot be negative.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("Validation fraction must be between zero and one.")
        if self.stage_loss_weight < 0.0:
            raise ValueError("Stage loss weight cannot be negative.")
        if self.task_loss_weight < 0.0:
            raise ValueError("Task loss weight cannot be negative.")
        if self.episodes_per_task is not None and self.episodes_per_task < 2:
            raise ValueError("Episodes per task must be at least two when specified.")


@dataclass(frozen=True)
class SmallVLATrainingResult:
    checkpoint: Path
    train_episodes: int
    validation_episodes: int
    train_transitions: int
    validation_transitions: int
    best_epoch: int
    best_validation_loss: float
    best_validation_action_mae: float
    best_validation_task_accuracy: float
    history: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class SmallVLATaskPrediction:
    object_key: str
    target_key: str
    confidence: float
    learned: bool

    @property
    def canonical_label(self) -> str:
        return f"pick_{self.object_key}_to_{self.target_key}"


class Vocabulary:
    def __init__(self, tokens: Sequence[str]) -> None:
        if len(tokens) < 2 or tokens[0] != PAD_TOKEN or tokens[1] != UNKNOWN_TOKEN:
            raise ValueError("Vocabulary must start with pad and unknown tokens.")
        if len(tokens) != len(set(tokens)):
            raise ValueError("Vocabulary tokens must be unique.")
        self.tokens = tuple(tokens)
        self._indices = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def build(cls, instructions: Sequence[str]) -> "Vocabulary":
        counts = Counter(token for instruction in instructions for token in tokenize_instruction(instruction))
        tokens = (PAD_TOKEN, UNKNOWN_TOKEN, *sorted(counts))
        return cls(tokens)

    def encode(self, instruction: str, max_tokens: int) -> np.ndarray:
        if max_tokens <= 0:
            raise ValueError("Maximum token count must be positive.")
        encoded = np.zeros(max_tokens, dtype=np.int64)
        unknown_index = self._indices[UNKNOWN_TOKEN]
        for position, token in enumerate(tokenize_instruction(instruction)[:max_tokens]):
            encoded[position] = self._indices.get(token, unknown_index)
        return encoded


class SmallVLAModel(nn.Module):
    def __init__(self, config: SmallVLAModelConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = nn.Sequential(
            _conv_block(3, 16, kernel_size=5),
            _conv_block(16, 32),
            _conv_block(32, 48),
            _conv_block(48, 64),
            nn.AdaptiveAvgPool2d((config.visual_grid_size, config.visual_grid_size)),
            nn.Flatten(),
            nn.Linear(
                64 * config.visual_grid_size * config.visual_grid_size,
                config.visual_dimension,
            ),
            nn.ReLU(),
        )
        self.camera_embedding = nn.Parameter(
            torch.zeros(len(config.camera_keys), config.visual_dimension)
        )
        nn.init.normal_(self.camera_embedding, std=0.02)
        self.language_embedding = nn.Embedding(
            config.vocabulary_size,
            config.language_dimension,
            padding_idx=0,
        )
        self.language_projection = nn.Sequential(
            nn.Linear(config.language_dimension, config.language_dimension),
            nn.ReLU(),
        )
        self.task_head = (
            nn.Linear(config.language_dimension, config.task_count)
            if config.learned_task_routing
            else None
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dimension, config.state_feature_dimension),
            nn.ReLU(),
            nn.Linear(config.state_feature_dimension, config.state_feature_dimension),
            nn.ReLU(),
        )
        fusion_dimension = (
            len(config.camera_keys) * config.visual_dimension
            + config.language_dimension
            + config.state_feature_dimension
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dimension, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        action_outputs = config.action_dimension
        if config.stage_conditioned_actions:
            action_outputs *= len(STAGE_NAMES)
        if config.task_conditioned_actions:
            action_outputs *= config.task_count
        self.action_head = nn.Linear(config.hidden_size, action_outputs)
        self.stage_head = nn.Linear(config.hidden_size, len(STAGE_NAMES))

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        stage_indices: torch.Tensor | None = None,
        task_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, stage_logits, _ = self.forward_with_aux(
            images,
            state,
            language_tokens,
            stage_indices=stage_indices,
            task_indices=task_indices,
        )
        return action, stage_logits

    def forward_with_aux(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        stage_indices: torch.Tensor | None = None,
        task_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if images.ndim != 5 or images.shape[1] != len(self.config.camera_keys):
            raise ValueError(
                "Images must have shape [batch, cameras, channels, height, width] "
                f"with {len(self.config.camera_keys)} cameras, got {tuple(images.shape)}."
            )
        batch_size, camera_count = images.shape[:2]
        visual = self.visual_encoder(images.flatten(0, 1))
        visual = visual.reshape(batch_size, camera_count, self.config.visual_dimension)
        visual = visual + self.camera_embedding.unsqueeze(0)
        visual = visual.flatten(1)

        language = self.encode_language(language_tokens)
        task_logits = self.task_head(language) if self.task_head is not None else None
        state_features = self.state_encoder(state)
        fused = self.fusion(torch.cat((visual, language, state_features), dim=1))
        stage_logits = self.stage_head(fused)
        action_outputs = self.action_head(fused)
        if not self.config.stage_conditioned_actions and not self.config.task_conditioned_actions:
            return action_outputs, stage_logits, task_logits
        selected_stages = stage_logits.argmax(dim=1) if stage_indices is None else stage_indices
        if selected_stages.shape != (batch_size,):
            raise ValueError(f"Stage indices must have shape {(batch_size,)}, got {selected_stages.shape}.")
        batch_indices = torch.arange(batch_size, device=selected_stages.device)
        if self.config.task_conditioned_actions:
            if task_indices is None:
                if task_logits is None:
                    raise ValueError("Task indices are required when learned task routing is disabled.")
                task_indices = task_logits.argmax(dim=1)
            if task_indices.shape != (batch_size,):
                raise ValueError(
                    f"Task indices must have shape {(batch_size,)}, got {tuple(task_indices.shape)}."
                )
            if torch.any(task_indices < 0) or torch.any(task_indices >= self.config.task_count):
                raise ValueError("Task index falls outside the configured task expert range.")
            if self.config.stage_conditioned_actions:
                action_candidates = action_outputs.reshape(
                    batch_size,
                    self.config.task_count,
                    len(STAGE_NAMES),
                    self.config.action_dimension,
                )
                return (
                    action_candidates[batch_indices, task_indices, selected_stages],
                    stage_logits,
                    task_logits,
                )
            action_candidates = action_outputs.reshape(
                batch_size,
                self.config.task_count,
                self.config.action_dimension,
            )
            return action_candidates[batch_indices, task_indices], stage_logits, task_logits

        action_candidates = action_outputs.reshape(
            batch_size,
            len(STAGE_NAMES),
            self.config.action_dimension,
        )
        return action_candidates[batch_indices, selected_stages], stage_logits, task_logits

    def encode_language(self, language_tokens: torch.Tensor) -> torch.Tensor:
        if language_tokens.ndim != 2:
            raise ValueError(
                "Language tokens must have shape [batch, tokens], "
                f"got {tuple(language_tokens.shape)}."
            )
        token_mask = language_tokens.ne(0).unsqueeze(-1)
        embedded_tokens = self.language_embedding(language_tokens)
        language = (embedded_tokens * token_mask).sum(dim=1)
        token_count = token_mask.sum(dim=1).clamp_min(1)
        return self.language_projection(language / token_count)

    def predict_task_logits(self, language_tokens: torch.Tensor) -> torch.Tensor:
        if self.task_head is None:
            raise RuntimeError("This checkpoint does not contain a learned language task head.")
        return self.task_head(self.encode_language(language_tokens))


@dataclass
class SmallVLAPolicy:
    model: SmallVLAModel
    vocabulary: Vocabulary
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    device: str
    _stage_index: int = field(default=0, init=False, repr=False)
    _next_stage_predictions: int = field(default=0, init=False, repr=False)
    _stage_steps: int = field(default=0, init=False, repr=False)
    _held: bool = field(default=False, init=False, repr=False)
    _finger_contacts: int = field(default=0, init=False, repr=False)
    _has_held_object: bool = field(default=False, init=False, repr=False)
    _has_lifted_object: bool = field(default=False, init=False, repr=False)
    _has_released_after_lift: bool = field(default=False, init=False, repr=False)
    _target_aligned: bool = field(default=False, init=False, repr=False)
    _task_index: int | None = field(default=None, init=False, repr=False)
    _task_confidence: float = field(default=0.0, init=False, repr=False)

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return self.model.config.camera_keys

    @property
    def control_repeat(self) -> int:
        return self.model.config.control_repeat

    @property
    def strict_vla_inference(self) -> bool:
        return self.model.config.strict_vla_inference

    @property
    def learned_task_routing(self) -> bool:
        return self.model.config.learned_task_routing

    @property
    def uses_privileged_feedback(self) -> bool:
        return not self.strict_vla_inference

    @property
    def task_prediction(self) -> SmallVLATaskPrediction | None:
        if self._task_index is None:
            return None
        object_key, target_key = VLA_TASK_KEYS[self._task_index]
        return SmallVLATaskPrediction(
            object_key=object_key,
            target_key=target_key,
            confidence=self._task_confidence,
            learned=self.model.config.learned_task_routing,
        )

    def reset(self) -> None:
        self._stage_index = 0
        self._next_stage_predictions = 0
        self._stage_steps = 0
        self._held = False
        self._finger_contacts = 0
        self._has_held_object = False
        self._has_lifted_object = False
        self._has_released_after_lift = False
        self._target_aligned = False
        self._task_index = None
        self._task_confidence = 0.0

    def update_feedback(
        self,
        held: bool,
        finger_contacts: int = 0,
        has_lifted_object: bool = False,
        has_released_after_lift: bool = False,
    ) -> None:
        if self.strict_vla_inference:
            raise RuntimeError(
                "Strict VLA inference does not accept simulator contact or lift feedback."
            )
        if finger_contacts < 0:
            raise ValueError("Finger contact count cannot be negative.")
        was_held = self._held
        self._held = bool(held)
        self._finger_contacts = int(finger_contacts)
        self._has_held_object = self._has_held_object or self._held
        self._has_lifted_object = self._has_lifted_object or bool(has_lifted_object)
        self._has_released_after_lift = self._has_released_after_lift or bool(
            has_released_after_lift
        )
        release_index = STAGE_NAMES.index("release")
        if self._has_released_after_lift and self._stage_index < release_index:
            self._set_stage(release_index)
            return
        if was_held and not self._held and self._stage_index in {
            STAGE_NAMES.index("lift"),
            STAGE_NAMES.index("transfer"),
            STAGE_NAMES.index("descend_place"),
        }:
            if (
                self._stage_index == STAGE_NAMES.index("descend_place")
                and self._target_aligned
            ):
                self._set_stage(release_index)
            else:
                self._set_stage(STAGE_NAMES.index("approach"))

    @property
    def target_aligned(self) -> bool:
        return self._target_aligned

    def ground_instruction(self, instruction: str) -> SmallVLATaskPrediction:
        if not instruction.strip():
            raise ValueError("Task instruction cannot be empty.")
        if self.model.config.learned_task_routing:
            tokens = self.vocabulary.encode(instruction, self.model.config.max_tokens)
            self.model.eval()
            with torch.inference_mode():
                logits = self.model.predict_task_logits(
                    torch.as_tensor(
                        tokens[None, :],
                        dtype=torch.long,
                        device=self.device,
                    )
                )
                probabilities = functional.softmax(logits, dim=1)[0]
            self._task_index = int(probabilities.argmax().item())
            self._task_confidence = float(probabilities[self._task_index].item())
        else:
            task = resolve_task_instruction(instruction, require_explicit_target=True)
            self._task_index = VLA_TASK_TO_INDEX[(task.object_key, task.target_key)]
            self._task_confidence = 1.0
        prediction = self.task_prediction
        if prediction is None:
            raise RuntimeError("Task grounding did not produce a task prediction.")
        return prediction

    def predict(self, observation: VLAObservation) -> tuple[np.ndarray, str]:
        missing = set(self.camera_keys).difference(observation.images)
        if missing:
            raise ValueError(f"Small VLA observation is missing cameras: {', '.join(sorted(missing))}.")
        state = (observation.state.astype(np.float32) - self.state_mean) / self.state_std
        tokens = self.vocabulary.encode(observation.instruction, self.model.config.max_tokens)
        if self._task_index is None:
            self.ground_instruction(observation.instruction)
        if self._task_index is None:
            raise RuntimeError("Task must be grounded before action prediction.")
        if self.strict_vla_inference:
            self._target_aligned = False
        else:
            _, target_key = VLA_TASK_KEYS[self._task_index]
            self._target_aligned = _wrist_target_is_aligned(
                observation.images.get("wrist"),
                target_key,
                placement=self._stage_index >= STAGE_NAMES.index("descend_place"),
            )
        image_arrays = [observation.images[key] for key in self.camera_keys]
        images = _prepare_image_tensor(
            np.stack(image_arrays, axis=0)[None, ...],
            self.model.config.image_size,
            torch.device(self.device),
            augment=False,
        )
        self.model.eval()
        selected_stage = self._stage_index
        with torch.inference_mode():
            normalized_action, stage_logits, task_logits = self.model.forward_with_aux(
                images,
                torch.as_tensor(state[None, :], dtype=torch.float32, device=self.device),
                torch.as_tensor(tokens[None, :], dtype=torch.long, device=self.device),
                stage_indices=torch.as_tensor(
                    [selected_stage],
                    dtype=torch.long,
                    device=self.device,
                ),
                task_indices=(
                    None
                    if self.model.config.learned_task_routing
                    else torch.as_tensor(
                        [self._task_index], dtype=torch.long, device=self.device
                    )
                    if self.model.config.task_conditioned_actions
                    else None
                ),
            )
        if task_logits is not None:
            task_probabilities = functional.softmax(task_logits, dim=1)[0]
            predicted_task_index = int(task_probabilities.argmax().item())
            if predicted_task_index != self._task_index:
                raise RuntimeError("Learned task prediction changed within a rollout.")
            self._task_confidence = float(task_probabilities[predicted_task_index].item())
        model_action = normalized_action[0].cpu().numpy() * self.action_std + self.action_mean
        action = _decode_model_actions(
            model_action[None, :],
            observation.state[None, :],
            self.model.config,
        )[0]
        if not self.strict_vla_inference:
            action[7] = -1.0 if STAGE_NAMES[selected_stage] in _OPEN_GRIPPER_STAGES else 1.0
        predicted_stage = int(stage_logits[0].argmax().item())
        if self.strict_vla_inference:
            self._update_stage_from_predictions(predicted_stage)
        else:
            self._update_stage(predicted_stage)
        return action, STAGE_NAMES[selected_stage]

    def _update_stage_from_predictions(self, predicted_stage: int) -> None:
        self._stage_steps += 1
        if self._stage_index >= len(STAGE_NAMES) - 1:
            return
        if predicted_stage != self._stage_index + 1:
            self._next_stage_predictions = 0
            return
        self._next_stage_predictions += 1
        if self._next_stage_predictions >= 2:
            self._set_stage(self._stage_index + 1)

    def _update_stage(self, predicted_stage: int) -> None:
        self._stage_steps += 1
        close_index = STAGE_NAMES.index("close")
        if self._stage_index == close_index and not self._held:
            self._next_stage_predictions = 0
            if self._stage_steps >= _CLOSE_RETRY_STEPS:
                self._set_stage(STAGE_NAMES.index("approach"))
            return
        release_index = STAGE_NAMES.index("release")
        if self._stage_index == release_index:
            self._next_stage_predictions = 0
            if not self._held and self._stage_steps >= _RELEASE_STEPS:
                self._set_stage(STAGE_NAMES.index("retreat"))
            return
        if self._stage_index in {
            STAGE_NAMES.index("lift"),
            STAGE_NAMES.index("transfer"),
            STAGE_NAMES.index("descend_place"),
        } and self._has_held_object and not self._held:
            if (
                self._stage_index == STAGE_NAMES.index("descend_place")
                and self._target_aligned
            ):
                self._set_stage(release_index)
            else:
                self._set_stage(STAGE_NAMES.index("approach"))
            return
        if self._stage_index >= len(STAGE_NAMES) - 1:
            return
        if self._stage_index == STAGE_NAMES.index("transfer") and not self._target_aligned:
            self._next_stage_predictions = 0
            return
        if self._stage_index == STAGE_NAMES.index("descend_place") and not self._target_aligned:
            self._next_stage_predictions = 0
            return
        if predicted_stage != self._stage_index + 1:
            self._next_stage_predictions = 0
            return
        self._next_stage_predictions += 1
        if self._next_stage_predictions >= 2:
            self._set_stage(self._stage_index + 1)

    def _set_stage(self, stage_index: int) -> None:
        self._stage_index = stage_index
        self._next_stage_predictions = 0
        self._stage_steps = 0

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "SmallVLAPolicy":
        resolved_device = resolve_torch_device(device)
        checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
        checkpoint_version = int(checkpoint.get("checkpoint_version", -1))
        if checkpoint_version not in {
            LEGACY_SMALL_VLA_CHECKPOINT_VERSION,
            DELTA_SMALL_VLA_CHECKPOINT_VERSION,
            TRANSITION_SMALL_VLA_CHECKPOINT_VERSION,
            DIRECT_ACTION_SMALL_VLA_CHECKPOINT_VERSION,
            TASK_EXPERT_SMALL_VLA_CHECKPOINT_VERSION,
            CENTERED_CAMERA_SMALL_VLA_CHECKPOINT_VERSION,
            SMALL_VLA_CHECKPOINT_VERSION,
        }:
            raise ValueError(f"Unsupported small VLA checkpoint version in {path}.")
        if checkpoint.get("action_mode") != ACTION_MODE:
            raise ValueError(f"Small VLA checkpoint action mode does not match the simulator: {path}.")
        if checkpoint.get("scene_id") != VLA_SCENE_ID:
            raise ValueError(f"Small VLA checkpoint scene does not match the simulator: {path}.")
        model_config_values = dict(checkpoint["model_config"])
        if checkpoint_version <= TRANSITION_SMALL_VLA_CHECKPOINT_VERSION:
            model_config_values["visual_grid_size"] = 2
        if checkpoint_version <= DIRECT_ACTION_SMALL_VLA_CHECKPOINT_VERSION:
            model_config_values["task_count"] = len(VLA_TASK_KEYS)
            model_config_values["task_conditioned_actions"] = False
        if checkpoint_version <= CENTERED_CAMERA_SMALL_VLA_CHECKPOINT_VERSION:
            model_config_values["learned_task_routing"] = False
            model_config_values["strict_vla_inference"] = False
        if checkpoint_version == LEGACY_SMALL_VLA_CHECKPOINT_VERSION:
            model_config_values["action_representation"] = ABSOLUTE_ACTION_REPRESENTATION
            model_config_values["stage_conditioned_actions"] = False
        if checkpoint_version in {
            LEGACY_SMALL_VLA_CHECKPOINT_VERSION,
            DELTA_SMALL_VLA_CHECKPOINT_VERSION,
        }:
            model_config_values["control_repeat"] = 1
        model_config = SmallVLAModelConfig(**model_config_values)
        model = SmallVLAModel(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            vocabulary=Vocabulary(checkpoint["vocabulary"]),
            state_mean=np.asarray(checkpoint["state_mean"], dtype=np.float32),
            state_std=np.asarray(checkpoint["state_std"], dtype=np.float32),
            action_mean=np.asarray(checkpoint["action_mean"], dtype=np.float32),
            action_std=np.asarray(checkpoint["action_std"], dtype=np.float32),
            device=resolved_device,
        )


def tokenize_instruction(instruction: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(instruction.lower()))


def _wrist_target_is_aligned(
    image: np.ndarray | None,
    target_key: str,
    placement: bool,
) -> bool:
    if image is None:
        return True
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Wrist image must have shape [height, width, 3], got {image.shape}.")
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    if target_key == "green_plate":
        mask = (green > 90) & (green > red * 9 // 5) & (green > blue * 7 // 5)
    elif target_key == "yellow_plate":
        mask = (red > 140) & (green > 110) & (blue < 70) & (red < green * 3 // 2)
    else:
        raise ValueError(f"Unsupported VLA target for wrist alignment: {target_key!r}")

    rows, columns = np.nonzero(mask)
    minimum_pixels = max(20, round(80 * image.shape[0] * image.shape[1] / (256 * 256)))
    if columns.size < minimum_pixels:
        return False
    horizontal_offset = float(columns.mean() / max(image.shape[1] - 1, 1) - 0.5)
    vertical_offset = float(rows.mean() / max(image.shape[0] - 1, 1) - 0.5)
    vertical_limit = -0.12 if placement else -0.30
    return abs(horizontal_offset) <= 0.15 and vertical_offset <= vertical_limit


def load_small_vla_episode_records(directory: Path) -> tuple[SmallVLAEpisodeRecord, ...]:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing VLA manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != VLA_DATASET_SCHEMA_VERSION:
        raise ValueError(f"Small VLA training requires schema {VLA_DATASET_SCHEMA_VERSION} data.")
    if manifest.get("scene_id") != VLA_SCENE_ID or manifest.get("action_mode") != ACTION_MODE:
        raise ValueError("Small VLA dataset scene or action mode does not match the current simulator.")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"VLA manifest contains no episodes: {manifest_path}")

    source_root = directory.resolve()
    records: list[SmallVLAEpisodeRecord] = []
    seen_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        path = (directory / str(entry["file"])).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError(f"Episode path escapes the dataset directory: {entry['file']!r}")
        if path in seen_paths:
            raise ValueError(f"Manifest references an episode more than once: {entry['file']!r}")
        if not path.exists():
            raise FileNotFoundError(f"Missing VLA episode: {path}")
        seen_paths.add(path)
        records.append(
            SmallVLAEpisodeRecord(
                index=index,
                path=path,
                seed=int(entry["seed"]),
                steps=int(entry["steps"]),
                instruction=str(entry["instruction"]),
                object_key=str(entry["object_key"]),
                target_key=str(entry.get("target_key", "green_plate")),
            )
        )
    return tuple(records)


def split_small_vla_episodes(
    records: Sequence[SmallVLAEpisodeRecord],
    validation_fraction: float,
    seed: int,
    group_by_seed: bool = False,
) -> tuple[tuple[SmallVLAEpisodeRecord, ...], tuple[SmallVLAEpisodeRecord, ...]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must be between zero and one.")
    if not records:
        raise ValueError("Cannot split an empty VLA dataset.")
    if group_by_seed:
        return _split_small_vla_episodes_by_seed(records, validation_fraction, seed)
    groups: dict[str, list[SmallVLAEpisodeRecord]] = defaultdict(list)
    for record in records:
        groups[record.stratum].append(record)
    rng = np.random.default_rng(seed)
    train: list[SmallVLAEpisodeRecord] = []
    validation: list[SmallVLAEpisodeRecord] = []
    for stratum in sorted(groups):
        group = groups[stratum]
        if len(group) < 2:
            raise ValueError(f"Task group {stratum!r} needs at least two episodes for a split.")
        shuffled = [group[index] for index in rng.permutation(len(group))]
        validation_count = min(len(group) - 1, max(1, round(len(group) * validation_fraction)))
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])
    return tuple(sorted(train, key=lambda record: record.index)), tuple(
        sorted(validation, key=lambda record: record.index)
    )


def _split_small_vla_episodes_by_seed(
    records: Sequence[SmallVLAEpisodeRecord],
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[SmallVLAEpisodeRecord, ...], tuple[SmallVLAEpisodeRecord, ...]]:
    seed_groups: dict[int, list[SmallVLAEpisodeRecord]] = defaultdict(list)
    expected_strata = {record.stratum for record in records}
    for record in records:
        seed_groups[record.seed].append(record)
    if len(seed_groups) < 2:
        raise ValueError("Paired VLA data needs at least two scene seeds for a split.")
    for scene_seed, group in seed_groups.items():
        strata = {record.stratum for record in group}
        if strata != expected_strata or len(group) != len(expected_strata):
            raise ValueError(
                f"Paired scene seed {scene_seed} does not contain exactly one episode per task."
            )
    rng = np.random.default_rng(seed)
    shuffled_seeds = [int(value) for value in rng.permutation(tuple(seed_groups))]
    validation_count = min(
        len(shuffled_seeds) - 1,
        max(1, round(len(shuffled_seeds) * validation_fraction)),
    )
    validation_seeds = set(shuffled_seeds[:validation_count])
    train = [record for record in records if record.seed not in validation_seeds]
    validation = [record for record in records if record.seed in validation_seeds]
    return tuple(sorted(train, key=lambda record: record.index)), tuple(
        sorted(validation, key=lambda record: record.index)
    )


def train_small_vla(
    demo_directory: Path,
    checkpoint_path: Path,
    model_config: SmallVLAModelConfig,
    training_config: SmallVLATrainingConfig,
    initial_checkpoint: Path | None = None,
) -> SmallVLATrainingResult:
    if checkpoint_path.exists():
        raise FileExistsError(f"Small VLA checkpoint already exists: {checkpoint_path}")
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    device = resolve_torch_device(training_config.device)
    records = load_small_vla_episode_records(demo_directory)
    paired_task_seeds = _uses_paired_task_seeds(demo_directory)
    if training_config.episodes_per_task is not None:
        records = _limit_episodes_per_task(
            records,
            training_config.episodes_per_task,
            training_config.seed,
            group_by_seed=paired_task_seeds,
        )
    _validate_control_horizon(demo_directory, model_config)
    train_records, validation_records = split_small_vla_episodes(
        records,
        training_config.validation_fraction,
        training_config.seed,
        group_by_seed=paired_task_seeds,
    )
    train_transitions = sum(_usable_transition_count(record, model_config) for record in train_records)
    validation_transitions = sum(
        _usable_transition_count(record, model_config) for record in validation_records
    )
    print(
        f"Dataset split: {len(train_records)} train episodes ({train_transitions} transitions), "
        f"{len(validation_records)} validation episodes ({validation_transitions} transitions)"
    )
    vocabulary = Vocabulary.build([record.instruction for record in train_records])
    model_config = SmallVLAModelConfig(
        **{
            **asdict(model_config),
            "vocabulary_size": len(vocabulary),
        }
    )
    print("Computing normalization statistics from the training episodes...")
    statistics = _dataset_statistics(train_records, model_config)
    model = SmallVLAModel(model_config).to(device)
    if initial_checkpoint is not None:
        _warm_start_small_vla(model, vocabulary, initial_checkpoint)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    stage_weights = torch.as_tensor(statistics["stage_weights"], dtype=torch.float32, device=device)
    rng = np.random.default_rng(training_config.seed)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_validation_action_mae = float("inf")
    best_validation_task_accuracy = 0.0
    best_epoch = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, training_config.epochs + 1):
        print(f"Starting epoch {epoch:03d}/{training_config.epochs}...")
        train_metrics = _run_epoch(
            model,
            train_records,
            vocabulary,
            statistics,
            model_config,
            training_config,
            device,
            stage_weights,
            optimizer=optimizer,
            rng=rng,
        )
        validation_metrics = _run_epoch(
            model,
            validation_records,
            vocabulary,
            statistics,
            model_config,
            training_config,
            device,
            stage_weights,
            optimizer=None,
            rng=rng,
        )
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_action_mae": train_metrics["action_mae"],
            "train_stage_accuracy": train_metrics["stage_accuracy"],
            "train_task_accuracy": train_metrics["task_accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_action_mae": validation_metrics["action_mae"],
            "validation_stage_accuracy": validation_metrics["stage_accuracy"],
            "validation_task_accuracy": validation_metrics["task_accuracy"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:03d}/{training_config.epochs}: "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"val_mae={validation_metrics['action_mae']:.5f} "
            f"val_stage={100.0 * validation_metrics['stage_accuracy']:.1f}% "
            f"val_task={100.0 * validation_metrics['task_accuracy']:.1f}%"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_validation_action_mae = validation_metrics["action_mae"]
            best_validation_task_accuracy = validation_metrics["task_accuracy"]
            best_epoch = epoch
            _save_small_vla_checkpoint(
                checkpoint_path,
                model,
                model_config,
                training_config,
                vocabulary,
                statistics,
                history,
                best_epoch,
            )

    return SmallVLATrainingResult(
        checkpoint=checkpoint_path,
        train_episodes=len(train_records),
        validation_episodes=len(validation_records),
        train_transitions=train_transitions,
        validation_transitions=validation_transitions,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_validation_action_mae=best_validation_action_mae,
        best_validation_task_accuracy=best_validation_task_accuracy,
        history=tuple(history),
    )


def resolve_torch_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Small VLA device must be 'auto', 'cpu', or 'cuda'.")
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        required_architecture = f"sm_{capability[0]}{capability[1]}"
        supported_architectures = set(torch.cuda.get_arch_list())
        if not supported_architectures or _cuda_architecture_supported(
            capability,
            supported_architectures,
        ):
            return "cuda"
        if requested == "cuda":
            supported = ", ".join(sorted(supported_architectures))
            raise RuntimeError(
                f"This PyTorch build lacks {required_architecture} kernels; it contains {supported}."
            )
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but no compatible CUDA device is available.")
    return "cpu"


def _cuda_architecture_supported(
    capability: tuple[int, int],
    architectures: Sequence[str],
) -> bool:
    for architecture in architectures:
        match = re.fullmatch(r"(?:sm|compute)_(\d+)", architecture)
        if match is None:
            continue
        encoded = int(match.group(1))
        architecture_capability = (encoded // 10, encoded % 10)
        if (
            architecture_capability[0] == capability[0]
            and architecture_capability[1] <= capability[1]
        ):
            return True
    return False


def _limit_episodes_per_task(
    records: Sequence[SmallVLAEpisodeRecord],
    episodes_per_task: int,
    seed: int,
    group_by_seed: bool = False,
) -> tuple[SmallVLAEpisodeRecord, ...]:
    if group_by_seed:
        seed_values = sorted({record.seed for record in records})
        if len(seed_values) < episodes_per_task:
            raise ValueError(
                f"Paired dataset has {len(seed_values)} scene seeds, fewer than {episodes_per_task}."
            )
        rng = np.random.default_rng(seed)
        selected_seeds = set(rng.choice(seed_values, size=episodes_per_task, replace=False).tolist())
        return tuple(record for record in records if record.seed in selected_seeds)
    groups: dict[str, list[SmallVLAEpisodeRecord]] = defaultdict(list)
    for record in records:
        groups[record.stratum].append(record)
    rng = np.random.default_rng(seed)
    selected: list[SmallVLAEpisodeRecord] = []
    for stratum in sorted(groups):
        group = groups[stratum]
        if len(group) < episodes_per_task:
            raise ValueError(
                f"Task group {stratum!r} has {len(group)} episodes, fewer than {episodes_per_task}."
            )
        indices = rng.permutation(len(group))[:episodes_per_task]
        selected.extend(group[index] for index in indices)
    return tuple(sorted(selected, key=lambda record: record.index))


def _validate_control_horizon(
    demo_directory: Path,
    model_config: SmallVLAModelConfig,
) -> None:
    if model_config.action_representation != STATE_TRANSITION_ACTION_REPRESENTATION:
        return
    manifest_path = Path(demo_directory) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_every = int(manifest.get("record_every", 0))
    if record_every != model_config.control_repeat:
        raise ValueError(
            "State-transition actions must use the dataset recording horizon: "
            f"checkpoint control_repeat={model_config.control_repeat}, dataset record_every={record_every}."
        )


def _uses_paired_task_seeds(demo_directory: Path) -> bool:
    manifest_path = Path(demo_directory) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(manifest.get("paired_task_seeds", False))


def _usable_transition_count(
    record: SmallVLAEpisodeRecord,
    model_config: SmallVLAModelConfig,
) -> int:
    if model_config.action_representation == STATE_TRANSITION_ACTION_REPRESENTATION:
        if record.steps < 2:
            raise ValueError(f"State-transition episode needs at least two frames: {record.path}")
        return record.steps - 1
    return record.steps


def _conv_block(input_channels: int, output_channels: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride=2, padding=kernel_size // 2),
        nn.GroupNorm(4, output_channels),
        nn.ReLU(),
    )


def _dataset_statistics(
    records: Sequence[SmallVLAEpisodeRecord],
    model_config: SmallVLAModelConfig,
) -> dict[str, np.ndarray]:
    state_sum = np.zeros(model_config.state_dimension, dtype=np.float64)
    state_square_sum = np.zeros(model_config.state_dimension, dtype=np.float64)
    action_sum = np.zeros(model_config.action_dimension, dtype=np.float64)
    action_square_sum = np.zeros(model_config.action_dimension, dtype=np.float64)
    stage_counts = np.zeros(len(STAGE_NAMES), dtype=np.int64)
    count = 0
    for record in records:
        with np.load(record.path, allow_pickle=False) as data:
            states = np.asarray(data["observation.state"], dtype=np.float64)
            actions = np.asarray(data["action"], dtype=np.float64)
            stages = np.asarray(data["oracle_stage_index"], dtype=np.int64)
        if states.shape != (record.steps, model_config.state_dimension):
            raise ValueError(f"Unexpected state shape in {record.path}: {states.shape}")
        if actions.shape != (record.steps, model_config.action_dimension):
            raise ValueError(f"Unexpected action shape in {record.path}: {actions.shape}")
        if np.any(stages < 0) or np.any(stages >= len(STAGE_NAMES)):
            raise ValueError(f"Invalid oracle stage index in {record.path}.")
        model_actions = _encode_model_actions(actions, states, model_config)
        usable_count = _usable_transition_count(record, model_config)
        usable_states = states[:usable_count]
        usable_actions = model_actions[:usable_count]
        usable_stages = stages[:usable_count]
        state_sum += usable_states.sum(axis=0)
        state_square_sum += np.square(usable_states).sum(axis=0)
        action_sum += usable_actions.sum(axis=0)
        action_square_sum += np.square(usable_actions).sum(axis=0)
        stage_counts += np.bincount(usable_stages, minlength=len(STAGE_NAMES))
        count += usable_count
    if count <= 0 or np.any(stage_counts == 0):
        raise ValueError("Small VLA training data must contain every oracle stage.")
    state_mean, state_std = _mean_and_std(state_sum, state_square_sum, count, minimum_std=1e-4)
    action_mean, action_std = _mean_and_std(action_sum, action_square_sum, count, minimum_std=1e-3)
    stage_weights = count / (len(STAGE_NAMES) * stage_counts.astype(np.float64))
    stage_weights = np.minimum(stage_weights, 5.0)
    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "stage_weights": stage_weights.astype(np.float32),
    }


def _mean_and_std(
    values_sum: np.ndarray,
    square_sum: np.ndarray,
    count: int,
    minimum_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = values_sum / count
    variance = np.maximum(square_sum / count - np.square(mean), 0.0)
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), minimum_std).astype(np.float32)


def _run_epoch(
    model: SmallVLAModel,
    records: Sequence[SmallVLAEpisodeRecord],
    vocabulary: Vocabulary,
    statistics: Mapping[str, np.ndarray],
    model_config: SmallVLAModelConfig,
    training_config: SmallVLATrainingConfig,
    device: str,
    stage_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    ordered_records = list(records)
    if training:
        rng.shuffle(ordered_records)
    total_loss = 0.0
    total_action_error = 0.0
    correct_stages = 0
    correct_tasks = 0
    sample_count = 0
    device_object = torch.device(device)

    for record in ordered_records:
        with np.load(record.path, allow_pickle=False) as data:
            states = np.asarray(data["observation.state"], dtype=np.float32)
            actions = np.asarray(data["action"], dtype=np.float32)
            stages = np.asarray(data["oracle_stage_index"], dtype=np.int64)
            camera_frames = np.stack(
                [np.asarray(data[f"observation.images.{key}"], dtype=np.uint8) for key in model_config.camera_keys],
                axis=1,
            )
        model_actions = _encode_model_actions(actions, states, model_config)
        usable_count = _usable_transition_count(record, model_config)
        indices = np.arange(usable_count)
        if training:
            rng.shuffle(indices)
        encoded_instruction = vocabulary.encode(record.instruction, model_config.max_tokens)
        task_index = VLA_TASK_TO_INDEX[(record.object_key, record.target_key)]
        for start in range(0, usable_count, training_config.batch_size):
            batch_indices = indices[start : start + training_config.batch_size]
            state_batch = (states[batch_indices] - statistics["state_mean"]) / statistics["state_std"]
            action_batch = (
                model_actions[batch_indices] - statistics["action_mean"]
            ) / statistics["action_std"]
            stage_batch = stages[batch_indices]
            image_batch = _prepare_image_tensor(
                camera_frames[batch_indices],
                model_config.image_size,
                device_object,
                augment=training,
            )
            state_tensor = torch.as_tensor(state_batch, dtype=torch.float32, device=device_object)
            action_tensor = torch.as_tensor(action_batch, dtype=torch.float32, device=device_object)
            stage_tensor = torch.as_tensor(stage_batch, dtype=torch.long, device=device_object)
            token_tensor = torch.as_tensor(
                np.repeat(encoded_instruction[None, :], len(batch_indices), axis=0),
                dtype=torch.long,
                device=device_object,
            )
            task_tensor = torch.full(
                (len(batch_indices),),
                task_index,
                dtype=torch.long,
                device=device_object,
            )

            with torch.set_grad_enabled(training):
                predicted_action, predicted_stage, predicted_task = model.forward_with_aux(
                    image_batch,
                    state_tensor,
                    token_tensor,
                    stage_indices=stage_tensor if training else None,
                    task_indices=(
                        task_tensor
                        if model_config.task_conditioned_actions
                        and (training or not model_config.learned_task_routing)
                        else None
                    ),
                )
                per_sample_action_loss = functional.mse_loss(
                    predicted_action,
                    action_tensor,
                    reduction="none",
                ).mean(dim=1)
                action_loss = (per_sample_action_loss * stage_weights[stage_tensor]).mean()
                stage_loss = functional.cross_entropy(
                    predicted_stage,
                    stage_tensor,
                    weight=stage_weights,
                )
                if predicted_task is None:
                    task_loss = torch.zeros((), device=device_object)
                    predicted_task_indices = task_tensor
                else:
                    task_loss = functional.cross_entropy(predicted_task, task_tensor)
                    predicted_task_indices = predicted_task.argmax(dim=1)
                loss = (
                    action_loss
                    + training_config.stage_loss_weight * stage_loss
                    + training_config.task_loss_weight * task_loss
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            batch_size = len(batch_indices)
            predicted_model_actions = (
                predicted_action.detach().cpu().numpy() * statistics["action_std"]
                + statistics["action_mean"]
            )
            if model_config.action_representation == STATE_TRANSITION_ACTION_REPRESENTATION:
                predicted_metric = np.clip(predicted_model_actions, -1.0, 1.0)
                target_metric = model_actions[batch_indices]
            else:
                predicted_metric = _decode_model_actions(
                    predicted_model_actions,
                    states[batch_indices],
                    model_config,
                )
                target_metric = actions[batch_indices]
            total_action_error += float(np.abs(predicted_metric - target_metric).sum())
            total_loss += float(loss.detach().item()) * batch_size
            correct_stages += int((predicted_stage.argmax(dim=1) == stage_tensor).sum().item())
            correct_tasks += int((predicted_task_indices == task_tensor).sum().item())
            sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Small VLA epoch processed no samples.")
    return {
        "loss": total_loss / sample_count,
        "action_mae": total_action_error / (sample_count * model_config.action_dimension),
        "stage_accuracy": correct_stages / sample_count,
        "task_accuracy": correct_tasks / sample_count,
    }


def _prepare_image_tensor(
    images: np.ndarray,
    image_size: int,
    device: torch.device,
    augment: bool,
) -> torch.Tensor:
    if images.ndim != 5 or images.shape[-1] != 3:
        raise ValueError(f"Expected RGB image batch [B, C, H, W, 3], got {images.shape}.")
    batch_size, camera_count = images.shape[:2]
    tensor = torch.as_tensor(images.copy(), dtype=torch.float32, device=device)
    tensor = tensor.permute(0, 1, 4, 2, 3).reshape(
        batch_size * camera_count,
        3,
        images.shape[2],
        images.shape[3],
    )
    tensor = functional.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    tensor = tensor.reshape(batch_size, camera_count, 3, image_size, image_size) / 255.0
    if augment:
        brightness = torch.empty(batch_size, 1, 1, 1, 1, device=device).uniform_(0.9, 1.1)
        contrast = torch.empty(batch_size, 1, 1, 1, 1, device=device).uniform_(0.9, 1.1)
        mean = tensor.mean(dim=(-1, -2), keepdim=True)
        tensor = ((tensor - mean) * contrast + mean) * brightness
        tensor = tensor.clamp(0.0, 1.0)
    return (tensor - 0.5) / 0.5


def _encode_model_actions(
    environment_actions: np.ndarray,
    states: np.ndarray,
    model_config: SmallVLAModelConfig,
) -> np.ndarray:
    actions = np.asarray(environment_actions)
    robot_states = np.asarray(states)
    if actions.ndim != 2 or actions.shape[1] != model_config.action_dimension:
        raise ValueError(f"Unexpected environment action shape: {actions.shape}")
    if robot_states.shape != (actions.shape[0], model_config.state_dimension):
        raise ValueError(f"State shape {robot_states.shape} does not match actions {actions.shape}.")
    if model_config.action_representation == ABSOLUTE_ACTION_REPRESENTATION:
        return actions.copy()

    if model_config.action_representation == STATE_TRANSITION_ACTION_REPRESENTATION:
        encoded = actions.copy()
        encoded[:-1, :7] = (
            robot_states[1:, :7] - robot_states[:-1, :7]
        ) / model_config.joint_delta_scale
        encoded[-1, :7] = 0.0
        encoded[:, :7] = np.clip(encoded[:, :7], -1.0, 1.0)
        return encoded

    control_min = np.asarray(FRANKA_ARM_CONTROL_MIN, dtype=actions.dtype)
    control_max = np.asarray(FRANKA_ARM_CONTROL_MAX, dtype=actions.dtype)
    arm_targets = control_min + (actions[:, :7] + 1.0) * 0.5 * (control_max - control_min)
    encoded = actions.copy()
    encoded[:, :7] = (arm_targets - robot_states[:, :7]) / model_config.joint_delta_scale
    encoded[:, :7] = np.clip(encoded[:, :7], -1.0, 1.0)
    return encoded


def _decode_model_actions(
    model_actions: np.ndarray,
    states: np.ndarray,
    model_config: SmallVLAModelConfig,
) -> np.ndarray:
    actions = np.asarray(model_actions)
    robot_states = np.asarray(states)
    if actions.ndim != 2 or actions.shape[1] != model_config.action_dimension:
        raise ValueError(f"Unexpected model action shape: {actions.shape}")
    if robot_states.shape != (actions.shape[0], model_config.state_dimension):
        raise ValueError(f"State shape {robot_states.shape} does not match actions {actions.shape}.")
    if model_config.action_representation == ABSOLUTE_ACTION_REPRESENTATION:
        return np.clip(actions, -1.0, 1.0)

    control_min = np.asarray(FRANKA_ARM_CONTROL_MIN, dtype=actions.dtype)
    control_max = np.asarray(FRANKA_ARM_CONTROL_MAX, dtype=actions.dtype)
    bounded_deltas = np.clip(actions[:, :7], -1.0, 1.0)
    arm_targets = robot_states[:, :7] + bounded_deltas * model_config.joint_delta_scale
    decoded = actions.copy()
    decoded[:, :7] = 2.0 * (arm_targets - control_min) / (control_max - control_min) - 1.0
    return np.clip(decoded, -1.0, 1.0)


def _warm_start_small_vla(
    model: SmallVLAModel,
    vocabulary: Vocabulary,
    checkpoint_path: Path,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial small VLA checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_scene = str(checkpoint.get("scene_id", ""))
    if source_scene not in _WARM_START_SCENE_IDS or checkpoint.get("action_mode") != ACTION_MODE:
        raise ValueError("Initial checkpoint does not match a supported VLA scene or action mode.")
    if source_scene != VLA_SCENE_ID:
        print(
            f"Warm-starting across camera calibrations: {source_scene} -> {VLA_SCENE_ID}. "
            "All layers will be fine-tuned on the new images."
        )

    source_config = dict(checkpoint["model_config"])
    target_config = model.config
    required_matches = {
        "camera_keys": tuple(target_config.camera_keys),
        "image_size": target_config.image_size,
        "state_dimension": target_config.state_dimension,
        "action_dimension": target_config.action_dimension,
        "hidden_size": target_config.hidden_size,
        "visual_grid_size": target_config.visual_grid_size,
        "action_representation": target_config.action_representation,
        "joint_delta_scale": target_config.joint_delta_scale,
    }
    for key, expected in required_matches.items():
        actual = tuple(source_config[key]) if key == "camera_keys" else source_config[key]
        if actual != expected:
            raise ValueError(
                f"Initial checkpoint {key} is {actual!r}, but the new model requires {expected!r}."
            )

    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    loaded_tensors = 0
    for key, source_value in source_state.items():
        if key.startswith("action_head.") or key == "language_embedding.weight":
            continue
        if key in target_state and target_state[key].shape == source_value.shape:
            target_state[key] = source_value
            loaded_tensors += 1

    source_tokens = tuple(checkpoint.get("vocabulary", ()))
    source_token_indices = {token: index for index, token in enumerate(source_tokens)}
    common_tokens = [token for token in vocabulary.tokens if token in source_token_indices]
    if "language_embedding.weight" in source_state:
        target_embeddings = target_state["language_embedding.weight"]
        source_embeddings = source_state["language_embedding.weight"]
        if target_embeddings.shape[1] != source_embeddings.shape[1]:
            raise ValueError("Initial checkpoint language embedding dimension does not match.")
        for target_index, token in enumerate(vocabulary.tokens):
            source_index = source_token_indices.get(token)
            if source_index is not None:
                target_embeddings[target_index] = source_embeddings[source_index]
        target_state["language_embedding.weight"] = target_embeddings
        loaded_tensors += 1
        print(
            f"Reused {len(common_tokens)}/{len(vocabulary.tokens)} language token embeddings."
        )

    source_task_conditioned = bool(source_config.get("task_conditioned_actions", False))
    if source_state["action_head.weight"].shape == target_state["action_head.weight"].shape:
        target_state["action_head.weight"] = source_state["action_head.weight"]
        target_state["action_head.bias"] = source_state["action_head.bias"]
        loaded_tensors += 2
    elif (
        not source_task_conditioned
        and target_config.task_conditioned_actions
        and source_state["action_head.weight"].shape[0] * target_config.task_count
        == target_state["action_head.weight"].shape[0]
    ):
        target_state["action_head.weight"] = source_state["action_head.weight"].repeat(
            target_config.task_count,
            1,
        )
        target_state["action_head.bias"] = source_state["action_head.bias"].repeat(
            target_config.task_count
        )
        loaded_tensors += 2
        print(
            f"Expanded the V4 action head into {target_config.task_count} "
            "language-routed task experts."
        )
    else:
        raise ValueError("Initial checkpoint action head cannot initialize the new task experts.")

    model.load_state_dict(target_state)
    print(f"Warm-started {loaded_tensors}/{len(target_state)} model tensors from {checkpoint_path}.")


def _save_small_vla_checkpoint(
    path: Path,
    model: SmallVLAModel,
    model_config: SmallVLAModelConfig,
    training_config: SmallVLATrainingConfig,
    vocabulary: Vocabulary,
    statistics: Mapping[str, np.ndarray],
    history: Sequence[Mapping[str, float]],
    best_epoch: int,
) -> None:
    torch.save(
        {
            "checkpoint_version": SMALL_VLA_CHECKPOINT_VERSION,
            "action_mode": ACTION_MODE,
            "scene_id": VLA_SCENE_ID,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "vocabulary": list(vocabulary.tokens),
            "state_mean": statistics["state_mean"],
            "state_std": statistics["state_std"],
            "action_mean": statistics["action_mean"],
            "action_std": statistics["action_std"],
            "history": list(history),
            "best_epoch": best_epoch,
        },
        path,
    )
