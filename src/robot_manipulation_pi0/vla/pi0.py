from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from robot_manipulation_pi0.sim import VLA_SCENE_ID

from .lerobot_dataset import (
    PI0_CAMERA_RENAME_MAP,
    SUPPORTED_LEROBOT_MINOR,
    read_conversion_metadata,
    split_indices_from_metadata,
)
from .observation import VLAObservation


DEFAULT_PI0_BASE_MODEL = "lerobot/pi0_base"
MIN_PI0_INFERENCE_VRAM_GIB = 14.0


@dataclass(frozen=True)
class Pi0TrainingRequest:
    dataset_directory: Path
    output_directory: Path
    split: str = "train"
    base_model: str = DEFAULT_PI0_BASE_MODEL
    steps: int = 30_000
    batch_size: int = 1
    num_workers: int = 2
    save_frequency: int = 5_000
    log_frequency: int = 50
    device: str = "cuda"
    job_name: str = "pi0_franka_multi_object"
    job_target: str | None = None
    policy_repo_id: str | None = None
    allow_legacy_scene: bool = False


def build_pi0_training_command(request: Pi0TrainingRequest) -> list[str]:
    _validate_training_request(request)
    info_path = request.dataset_directory / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Not a finalized LeRobot dataset (missing {info_path}). "
            "Run scripts/convert_vla_to_lerobot.py first."
        )
    metadata = read_conversion_metadata(request.dataset_directory)
    if metadata.get("scene_id") != VLA_SCENE_ID and not request.allow_legacy_scene:
        raise ValueError(
            f"Converted dataset scene {metadata.get('scene_id')!r} does not match {VLA_SCENE_ID!r}."
        )
    episodes = split_indices_from_metadata(metadata, request.split)
    repo_id = str(metadata["repo_id"])
    rename_map = metadata.get("camera_rename_map", PI0_CAMERA_RENAME_MAP)
    if set(rename_map) != set(PI0_CAMERA_RENAME_MAP):
        raise ValueError(f"Converted dataset has unexpected camera mappings: {rename_map}")

    command = [
        "lerobot-train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={request.dataset_directory.resolve()}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        "--dataset.eval_split=0.0",
        "--dataset.return_uint8=true",
        f"--policy.path={request.base_model}",
        f"--policy.device={request.device}",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        "--policy.train_expert_only=true",
        f"--rename_map={json.dumps(rename_map, separators=(',', ':'))}",
        f"--output_dir={request.output_directory}",
        f"--job_name={request.job_name}",
        f"--steps={request.steps}",
        f"--batch_size={request.batch_size}",
        f"--num_workers={request.num_workers}",
        f"--save_freq={request.save_frequency}",
        f"--log_freq={request.log_frequency}",
        "--env_eval_freq=0",
        "--wandb.enable=false",
    ]
    if request.job_target:
        command.append(f"--job.target={request.job_target}")
    if request.policy_repo_id:
        command.extend(
            [
                f"--policy.repo_id={request.policy_repo_id}",
                "--policy.push_to_hub=true",
                "--policy.private=true",
            ]
        )
    else:
        command.append("--policy.push_to_hub=false")
    return command


def run_pi0_training(request: Pi0TrainingRequest) -> None:
    command = build_pi0_training_command(request)
    _require_lerobot_runtime(require_cuda=request.job_target is None and request.device.startswith("cuda"))
    if request.output_directory.exists():
        raise FileExistsError(
            f"Training output already exists: {request.output_directory}. Choose another --output-dir."
        )
    subprocess.run(command, check=True)


class LeRobotPi0Policy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        camera_rename_map: Mapping[str, str] = PI0_CAMERA_RENAME_MAP,
    ) -> None:
        _require_lerobot_runtime(require_cuda=False)
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference

        self.checkpoint = str(checkpoint)
        self.device = torch.device(device)
        policy_class = get_policy_class("pi0")
        try:
            policy_config = PreTrainedConfig.from_pretrained(self.checkpoint)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"pi0 checkpoint {self.checkpoint!r} is missing or incomplete. A trained policy "
                "repository must contain config.json and model weights. Converting or uploading "
                "the LeRobot dataset does not create a policy checkpoint; training must finish "
                "successfully before running preview_pi0_policy.py or interactive_pi0_tasks.py."
            ) from None
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. pi0 inference is not practical on this CPU-only host."
            )
        if self.device.type == "cuda":
            _validate_cuda_device_for_pi0(self.device)
        if getattr(policy_config, "type", None) != "pi0":
            raise ValueError(f"Checkpoint is not a pi0 policy: {checkpoint}")
        policy_config.device = str(self.device)
        policy_config.gradient_checkpointing = False
        self.policy = policy_class.from_pretrained(self.checkpoint, config=policy_config)
        if getattr(self.policy, "name", None) != "pi0":
            raise ValueError(f"Checkpoint is not a pi0 policy: {checkpoint}")
        self.policy.to(self.device)
        self.policy.eval()
        device_override = {"device": str(self.device)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": dict(camera_rename_map)},
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._prepare_observation = prepare_observation_for_inference

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict_action_chunk(self, observation: VLAObservation) -> torch.Tensor:
        raw_observation: dict[str, Any] = {"observation.state": observation.state.copy()}
        for camera_key, image in observation.images.items():
            raw_observation[f"observation.images.{camera_key}"] = image.copy()
        prepared = self._prepare_observation(
            raw_observation,
            self.device,
            task=observation.instruction,
            robot_type="franka_panda",
        )
        prepared = self.preprocessor(prepared)
        with torch.inference_mode():
            action_chunk = self.policy.predict_action_chunk(prepared)
        if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
            raise RuntimeError(f"pi0 returned an invalid action chunk shape: {tuple(action_chunk.shape)}")

        processed_actions = []
        for action_index in range(action_chunk.shape[1]):
            processed = self.postprocessor(action_chunk[:, action_index, :])
            processed_actions.append(processed.squeeze(0).detach().cpu())
        actions = torch.stack(processed_actions)
        if not torch.isfinite(actions).all():
            raise RuntimeError("pi0 returned NaN or infinite actions.")
        return actions.clamp(-1.0, 1.0)


def _validate_training_request(request: Pi0TrainingRequest) -> None:
    if request.steps <= 0 or request.batch_size <= 0:
        raise ValueError("Training steps and batch size must be positive.")
    if request.num_workers < 0:
        raise ValueError("Number of data loader workers cannot be negative.")
    if request.save_frequency <= 0 or request.log_frequency <= 0:
        raise ValueError("Save and log frequencies must be positive.")
    if request.job_target and not request.policy_repo_id:
        raise ValueError("Remote Hugging Face Jobs training requires --policy-repo-id.")
    if request.job_target and str(request.device) != "cuda":
        raise ValueError("Remote GPU training must use --device cuda.")


def _require_lerobot_runtime(*, require_cuda: bool) -> None:
    if shutil.which("lerobot-train") is None:
        raise ModuleNotFoundError(
            "LeRobot training tools are not installed. Install this project with: pip install -e '.[pi0]'"
        )
    try:
        version = importlib_metadata.version("lerobot")
    except importlib_metadata.PackageNotFoundError as error:
        raise ModuleNotFoundError("The lerobot package is not installed.") from error
    major_minor = _major_minor(version)
    if major_minor != SUPPORTED_LEROBOT_MINOR:
        expected = ".".join(str(value) for value in SUPPORTED_LEROBOT_MINOR)
        raise RuntimeError(f"Expected LeRobot {expected}.x, found {version}.")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. pi0 training/inference is not practical on this CPU-only host; "
            "use a CUDA machine or pass --job-target for Hugging Face Jobs."
        )
    if require_cuda:
        _validate_cuda_device_for_pi0(torch.device("cuda"))


def _validate_cuda_device_for_pi0(device: torch.device) -> None:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    required_architecture = f"sm_{capability[0]}{capability[1]}"
    supported_architectures = set(torch.cuda.get_arch_list())
    problems: list[str] = []
    if supported_architectures and not _cuda_architecture_supported(
        capability,
        supported_architectures,
    ):
        supported = ", ".join(sorted(supported_architectures))
        problems.append(
            f"PyTorch does not include {required_architecture} kernels (installed: {supported})"
        )

    total_vram_gib = properties.total_memory / 1024**3
    if total_vram_gib < MIN_PI0_INFERENCE_VRAM_GIB:
        problems.append(
            f"only {total_vram_gib:.1f} GiB VRAM is available; the pi0 checkpoint alone is "
            f"approximately {MIN_PI0_INFERENCE_VRAM_GIB:.0f} GB before runtime memory"
        )
    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            f"CUDA device {properties.name!r} cannot run this LeRobot pi0 pipeline: {details}. "
            "Use a machine with a supported CUDA GPU and substantially more VRAM."
        )


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


def _major_minor(version: str) -> tuple[int, int]:
    pieces = version.split("+", maxsplit=1)[0].split("-", maxsplit=1)[0].split(".")
    if len(pieces) < 2 or not pieces[0].isdigit() or not pieces[1].isdigit():
        raise RuntimeError(f"Cannot parse LeRobot version {version!r}.")
    return int(pieces[0]), int(pieces[1])
