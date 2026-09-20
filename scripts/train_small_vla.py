from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_manipulation_pi0.vla.small_vla import (
    JOINT_DELTA_ACTION_REPRESENTATION,
    SmallVLAModelConfig,
    SmallVLATrainingConfig,
    resolve_torch_device,
    train_small_vla,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact camera-and-language-conditioned action policy."
    )
    parser.add_argument("--demo-dir", type=Path, default=Path("data/demos/vla_centered_v4"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/small_vla_strict_v7.pt"),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--stage-loss-weight", type=float, default=0.1)
    parser.add_argument("--task-loss-weight", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=("top", "side", "wrist"),
        default=("top", "side", "wrist"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start compatible features, including a checkpoint from the previous camera calibration.",
    )
    parser.add_argument(
        "--control-repeat",
        type=int,
        default=1,
        help="Simulator steps per policy prediction. Direct oracle commands should use one.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=0,
        help="Limit each object-target pair for a quick experiment; zero uses every episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.control_repeat <= 0:
        raise ValueError("--control-repeat must be positive.")
    dataset_record_every = _dataset_record_every(args.demo_dir)
    model_config = SmallVLAModelConfig(
        camera_keys=tuple(args.cameras),
        image_size=args.image_size,
        hidden_size=args.hidden_size,
        action_representation=JOINT_DELTA_ACTION_REPRESENTATION,
        joint_delta_scale=0.02,
        control_repeat=args.control_repeat,
    )
    training_config = SmallVLATrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        stage_loss_weight=args.stage_loss_weight,
        task_loss_weight=args.task_loss_weight,
        seed=args.seed,
        device=args.device,
        episodes_per_task=args.episodes_per_task or None,
    )
    device = resolve_torch_device(args.device)
    print(f"Training device: {device}")
    print(f"Cameras: {', '.join(model_config.camera_keys)}")
    print(f"Image size: {model_config.image_size}x{model_config.image_size}")
    print(
        f"Visual feature grid: {model_config.visual_grid_size}x"
        f"{model_config.visual_grid_size}"
    )
    print(f"Action representation: {model_config.action_representation}")
    print(f"Learned language task routing: {model_config.learned_task_routing}")
    print(f"Strict RGB + language + proprioception inference: {model_config.strict_vla_inference}")
    print(f"Task-conditioned action experts: {model_config.task_count}")
    print(f"Dataset sampling interval: {dataset_record_every} simulator steps")
    print(f"Control horizon: {model_config.control_repeat} simulator steps")
    result = train_small_vla(
        args.demo_dir,
        args.output,
        model_config,
        training_config,
        initial_checkpoint=args.init_checkpoint,
    )

    metrics = {
        "train_episodes": result.train_episodes,
        "validation_episodes": result.validation_episodes,
        "train_transitions": result.train_transitions,
        "validation_transitions": result.validation_transitions,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "best_validation_action_mae": result.best_validation_action_mae,
        "action_representation": model_config.action_representation,
        "control_repeat": model_config.control_repeat,
        "visual_grid_size": model_config.visual_grid_size,
        "task_count": model_config.task_count,
        "learned_task_routing": model_config.learned_task_routing,
        "strict_vla_inference": model_config.strict_vla_inference,
        "task_loss_weight": training_config.task_loss_weight,
        "initial_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        "dataset_record_every": dataset_record_every,
        "history": list(result.history),
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best validation loss: {result.best_validation_loss:.6f}")
    print(f"Best validation action MAE: {result.best_validation_action_mae:.6f}")
    print(
        "Best validation task accuracy: "
        f"{100.0 * result.best_validation_task_accuracy:.1f}%"
    )
    print(f"Saved checkpoint: {result.checkpoint}")
    print(f"Saved metrics: {metrics_path}")


def _dataset_record_every(demo_directory: Path) -> int:
    manifest_path = demo_directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing VLA manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_every = int(manifest.get("record_every", 0))
    if record_every <= 0:
        raise ValueError(f"Invalid record_every value in {manifest_path}: {record_every}")
    return record_every


if __name__ == "__main__":
    main()
