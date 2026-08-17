from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robot_manipulation_pi0.vla import validate_vla_episode_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a contact sheet from a saved VLA demonstration episode.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8, help="Number of evenly spaced timesteps to display.")
    parser.add_argument("--output", type=Path, default=Path("outputs/vla_episode_preview.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive.")
    metadata = validate_vla_episode_file(args.episode)
    camera_keys = tuple(camera["key"] for camera in metadata["cameras"])
    with np.load(args.episode, allow_pickle=False) as data:
        steps = int(metadata["steps"])
        indices = np.unique(np.linspace(0, steps - 1, min(args.frames, steps), dtype=int))
        frames = {
            camera_key: data[f"observation.images.{camera_key}"][indices]
            for camera_key in camera_keys
        }
        stages = data["oracle_stage_index"][indices]
    _save_contact_sheet(frames, indices, stages, metadata, args.output)

    print(f"Instruction: {metadata['task']['instruction']}")
    print(f"Selected object: {metadata['task']['object_key']}")
    print(f"Episode steps: {metadata['steps']}")
    print(f"Cameras: {', '.join(camera_keys)}")
    print(f"Saved preview: {args.output}")


def _save_contact_sheet(
    frames: dict[str, np.ndarray],
    indices: np.ndarray,
    stages: np.ndarray,
    metadata: dict[str, object],
    output: Path,
) -> None:
    camera_keys = tuple(frames)
    first = frames[camera_keys[0]][0]
    height, width = first.shape[:2]
    header_height = 22
    row_label_height = 22
    canvas = Image.new(
        "RGB",
        (width * len(camera_keys), header_height + (height + row_label_height) * len(indices)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    stage_names = metadata["oracle_stage_names"]
    for column, camera_key in enumerate(camera_keys):
        draw.text((column * width + 4, 4), camera_key, fill="black")
    for row, (frame_index, stage_index) in enumerate(zip(indices, stages, strict=True)):
        top = header_height + row * (height + row_label_height)
        stage_name = stage_names[int(stage_index)]
        draw.text((4, top + 4), f"t={int(frame_index)} stage={stage_name}", fill="black")
        for column, camera_key in enumerate(camera_keys):
            left = column * width
            canvas.paste(Image.fromarray(frames[camera_key][row]), (left, top + row_label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == "__main__":
    main()
