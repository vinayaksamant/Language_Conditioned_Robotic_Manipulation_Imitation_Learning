from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_manipulation_pi0.vla import resolve_pi0_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a pi0 training checkpoint is complete.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_pi0_checkpoint(args.output_dir)
    train_config = checkpoint / "train_config.json"
    training_state = checkpoint.parent / "training_state"
    if not train_config.is_file():
        raise FileNotFoundError(f"Missing training configuration: {train_config}")
    if not training_state.is_dir():
        raise FileNotFoundError(f"Missing optimizer/training state: {training_state}")

    config = json.loads(train_config.read_text(encoding="utf-8"))
    if config.get("policy", {}).get("type") not in (None, "pi0"):
        raise ValueError(f"Unexpected policy in {train_config}")
    step = _checkpoint_step(checkpoint, training_state)
    if args.expected_steps is not None and step < args.expected_steps:
        raise RuntimeError(
            f"Latest checkpoint reached step {step}, expected at least {args.expected_steps}. "
            "Submit the training job again with the same output directory to resume."
        )

    weight_bytes = sum(path.stat().st_size for path in checkpoint.glob("*.safetensors"))
    print("pi0 checkpoint verification passed")
    print(f"Checkpoint: {checkpoint}")
    print(f"Training step: {step}")
    print(f"Weight files: {weight_bytes / 1024**3:.2f} GiB")
    print("The checkpoint is ready for validation, test rollouts, or interactive inference.")


def _checkpoint_step(checkpoint: Path, training_state: Path) -> int:
    state_file = training_state / "training_step.json"
    if state_file.is_file():
        value = _find_step(json.loads(state_file.read_text(encoding="utf-8")))
        if value is not None:
            return value
    checkpoint_name = checkpoint.parent.name
    if checkpoint_name.isdigit():
        return int(checkpoint_name)
    raise RuntimeError(f"Could not determine the training step from {checkpoint.parent}.")


def _find_step(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("step", "training_step", "global_step"):
            if key in value and isinstance(value[key], int):
                return value[key]
        for child in value.values():
            found = _find_step(child)
            if found is not None:
                return found
    return None


if __name__ == "__main__":
    main()
