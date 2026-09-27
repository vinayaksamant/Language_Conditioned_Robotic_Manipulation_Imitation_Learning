from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from robot_manipulation_pi0.vla import (
    DEFAULT_PI0_BASE_MODEL,
    Pi0TrainingRequest,
    build_pi0_training_command,
    find_pi0_resume_config,
    run_pi0_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or execute the guarded LeRobot pi0 fine-tuning command."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/lerobot/vla_centered_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pi0_vla_v2"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--base-model", default=DEFAULT_PI0_BASE_MODEL)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-frequency", type=int, default=5_000)
    parser.add_argument("--log-frequency", type=int, default=50)
    parser.add_argument(
        "--validation-frequency",
        type=int,
        default=0,
        help="Evaluate loss on a 10% held-out portion of the train+val pool every N steps; "
        "0 disables it.",
    )
    parser.add_argument("--max-validation-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--job-name", default="pi0_franka_multi_object")
    parser.add_argument(
        "--job-target",
        default=None,
        help="Hugging Face Jobs GPU flavor, for example a10g-small. Omit for local training.",
    )
    parser.add_argument(
        "--policy-repo-id",
        default=None,
        help="Hugging Face model repository. Required when --job-target is used.",
    )
    parser.add_argument("--allow-legacy-scene", action="store_true")
    parser.add_argument(
        "--resume",
        choices=("never", "auto", "require"),
        default="never",
        help="Resume the latest complete checkpoint, or start fresh when set to auto and none exists.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute training. Without this flag the exact command is printed as a dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume = _resolve_resume_mode(args.resume, args.output_dir)
    request = Pi0TrainingRequest(
        dataset_directory=args.dataset_dir,
        output_directory=args.output_dir,
        split=args.split,
        base_model=args.base_model,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_frequency=args.save_frequency,
        log_frequency=args.log_frequency,
        validation_frequency=args.validation_frequency,
        max_validation_samples=args.max_validation_samples,
        device=args.device,
        job_name=args.job_name,
        job_target=args.job_target,
        policy_repo_id=args.policy_repo_id,
        allow_legacy_scene=args.allow_legacy_scene,
        resume=resume,
    )
    command = build_pi0_training_command(request)
    print(f"Dataset split: {request.split}")
    print(f"Training steps: {request.steps}")
    print(f"Mode: {'resume' if request.resume else 'fresh training'}")
    print(f"Execution: {'Hugging Face Jobs ' + request.job_target if request.job_target else 'local CUDA'}")
    print("Command:")
    print(shlex.join(command))
    if not args.run:
        print("Dry run only. Add --run after checking the command.")
        return
    checkpoint = run_pi0_training(request)
    if checkpoint is None:
        print(f"Remote training submitted. Policy output: {request.policy_repo_id}")
    else:
        print("Training complete.")
        print(f"Saved loadable checkpoint: {checkpoint}")


def _resolve_resume_mode(mode: str, output_directory: Path) -> bool:
    if mode == "never":
        return False
    try:
        find_pi0_resume_config(output_directory)
    except FileNotFoundError:
        if mode == "require":
            raise
        return False
    return True


if __name__ == "__main__":
    main()
