from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_manipulation_pi0.learning import TrainingConfig, load_demonstration_dataset, train_behavior_cloning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a state-based behavior cloning baseline.")
    parser.add_argument("--demo-dir", type=Path, default=Path("data/demos/pick_place_oracle"))
    parser.add_argument("--output", type=Path, default=Path("outputs/bc_policy.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--no-stage-balancing",
        action="store_true",
        help="Disable inverse-frequency sampling of the eight oracle stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_demonstration_dataset(args.demo_dir)
    config = TrainingConfig(
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
        stage_balanced_sampling=not args.no_stage_balancing,
        log_every=args.log_every,
    )
    policy, metrics = train_behavior_cloning(dataset, config)
    policy.save(args.output, config, metrics)

    print(f"Loaded transitions: {len(dataset.actions)}")
    print(f"Observation dimension: {dataset.observations.shape[1]}")
    print(f"Action dimension: {dataset.actions.shape[1]}")
    stage_features = [name for name in dataset.feature_names if name.startswith("stage_")]
    if stage_features:
        print(f"Stage features: {len(stage_features)}")
    print(f"Train loss: {metrics['train_loss']:.6f}")
    print(f"Validation loss: {metrics['validation_loss']:.6f}")
    print(f"Saved checkpoint: {args.output}")
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
