from __future__ import annotations

import argparse
import shutil
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch

from robot_manipulation_pi0.vla import existing_lerobot_conversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the pi0 installation, dataset, disk, and GPU.")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pi0_hpc"))
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lerobot_version = importlib_metadata.version("lerobot")
    if not lerobot_version.startswith("0.6."):
        raise RuntimeError(f"Expected LeRobot 0.6.x, found {lerobot_version}.")
    if shutil.which("lerobot-train") is None:
        raise RuntimeError("lerobot-train is not on PATH. Activate the pi0 virtual environment.")

    print(f"Python/PyTorch: {torch.__version__}")
    print(f"LeRobot: {lerobot_version}")
    print(f"lerobot-train: {shutil.which('lerobot-train')}")

    if args.dataset_dir is not None:
        result = existing_lerobot_conversion(args.dataset_dir)
        if result is None:
            raise FileNotFoundError(
                f"Prepared LeRobot dataset not found at {args.dataset_dir}. "
                "Run scripts/prepare_pi0_dataset.py first."
            )
        print(
            f"Dataset: {result.episode_count} episodes, {result.frame_count} frames, "
            f"train/val/test={len(result.splits.train)}/{len(result.splits.val)}/{len(result.splits.test)}"
        )

    output_parent = args.output_dir.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_parent).free / 1024**3
    print(f"Free output disk: {free_gib:.1f} GiB at {output_parent}")
    if free_gib < 60:
        print("WARNING: less than 60 GiB is free; pi0 model caches and checkpoints can fill this disk.")

    if not torch.cuda.is_available():
        if args.require_cuda:
            raise RuntimeError("CUDA is required but PyTorch cannot see a GPU.")
        print("CUDA: unavailable (allowed for installation/data preparation)")
        return

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    vram_gib = properties.total_memory / 1024**3
    capability = torch.cuda.get_device_capability(device)
    print(f"CUDA device: {properties.name}")
    print(f"CUDA capability: {capability[0]}.{capability[1]}")
    print(f"GPU memory: {vram_gib:.1f} GiB")
    print(f"bfloat16 supported: {torch.cuda.is_bf16_supported()}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training configuration requires a GPU with bfloat16 support.")
    if vram_gib < 24:
        print("WARNING: pi0 fine-tuning may run out of memory below 24 GiB even with batch size 1.")


if __name__ == "__main__":
    main()
