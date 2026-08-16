from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from robot_manipulation_pi0.sim import ACTION_MODE
from robot_manipulation_pi0.sim.oracle import STAGE_NAMES

from .dataset import DemonstrationDataset, split_dataset


@dataclass(frozen=True)
class TrainingConfig:
    hidden_size: int = 128
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    seed: int = 0
    device: str = "cpu"
    stage_balanced_sampling: bool = True
    log_every: int = 10


class MLPPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


@dataclass
class BehaviorCloningPolicy:
    model: MLPPolicy
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    device: str = "cpu"

    def predict(self, observations: np.ndarray) -> np.ndarray:
        self.model.eval()
        normalized = (observations.astype(np.float32) - self.observation_mean) / self.observation_std
        with torch.no_grad():
            tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            normalized_action = self.model(tensor).cpu().numpy()
        actions = normalized_action * self.action_std + self.action_mean
        return np.clip(actions, -1.0, 1.0)

    def save(self, path: Path, config: TrainingConfig, metrics: dict[str, float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_mode": ACTION_MODE,
                "model_state_dict": self.model.cpu().state_dict(),
                "observation_mean": self.observation_mean,
                "observation_std": self.observation_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
                "config": asdict(config),
                "metrics": metrics,
            },
            path,
        )
        self.model.to(self.device)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "BehaviorCloningPolicy":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        action_mode = checkpoint.get("action_mode")
        if action_mode != ACTION_MODE:
            raise ValueError(
                f"Incompatible checkpoint {path}: expected action_mode={ACTION_MODE!r}, got "
                f"{action_mode!r}. Recollect demonstrations and retrain the policy."
            )
        config = checkpoint["config"]
        observation_mean = np.asarray(checkpoint["observation_mean"], dtype=np.float32)
        action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
        model = MLPPolicy(
            observation_dim=observation_mean.shape[0],
            action_dim=action_mean.shape[0],
            hidden_size=int(config["hidden_size"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return cls(
            model=model,
            observation_mean=observation_mean,
            observation_std=np.asarray(checkpoint["observation_std"], dtype=np.float32),
            action_mean=action_mean,
            action_std=np.asarray(checkpoint["action_std"], dtype=np.float32),
            device=device,
        )


def train_behavior_cloning(
    dataset: DemonstrationDataset,
    config: TrainingConfig,
) -> tuple[BehaviorCloningPolicy, dict[str, float]]:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    split = split_dataset(dataset, config.validation_fraction, config.seed)
    obs_mean, obs_std = _normalization_stats(split.train.observations)
    action_mean, action_std = _normalization_stats(split.train.actions)

    train_x = (split.train.observations - obs_mean) / obs_std
    train_y = (split.train.actions - action_mean) / action_std
    val_x = (split.validation.observations - obs_mean) / obs_std
    val_y = (split.validation.actions - action_mean) / action_std

    train_dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    sampler = None
    if config.stage_balanced_sampling:
        sample_weights = _stage_sample_weights(split.train.observations)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
    )

    model = MLPPolicy(train_x.shape[1], train_y.shape[1], config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    val_x_tensor = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    val_y_tensor = torch.as_tensor(val_y, dtype=torch.float32, device=device)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        val_loss = _torch_mse(model, val_x_tensor, val_y_tensor)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if config.log_every > 0 and (epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs):
            print(f"Epoch {epoch:4d}/{config.epochs}: validation loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    train_loss = _torch_mse(
        model,
        torch.as_tensor(train_x, dtype=torch.float32, device=device),
        torch.as_tensor(train_y, dtype=torch.float32, device=device),
    )
    metrics = {
        "train_loss": float(train_loss),
        "validation_loss": float(best_val_loss),
        "train_samples": float(train_x.shape[0]),
        "validation_samples": float(val_x.shape[0]),
        "best_epoch": float(best_epoch),
    }
    policy = BehaviorCloningPolicy(
        model=model,
        observation_mean=obs_mean,
        observation_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        device=str(device),
    )
    return policy, metrics


def _normalization_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _stage_sample_weights(observations: np.ndarray) -> np.ndarray:
    stage_features = observations[:, -len(STAGE_NAMES) :]
    stage_ids = np.argmax(stage_features, axis=1)
    counts = np.bincount(stage_ids, minlength=len(STAGE_NAMES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return 1.0 / counts[stage_ids]


def _torch_mse(model: nn.Module, observations: torch.Tensor, actions: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(observations), actions).item())
