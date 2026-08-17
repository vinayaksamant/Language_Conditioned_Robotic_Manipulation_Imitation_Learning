from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from robot_manipulation_pi0.sim import MultiObjectPickPlaceEnvironment, PickPlaceConfig
from robot_manipulation_pi0.vla import VLA_CAMERA_SPECS, CameraSpec, MujocoCameraRig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check fixed-camera visibility using only rendered RGB pixels across randomized scenes."
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("--seeds, --width, and --height must be positive.")

    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("red_cube", "blue_cylinder"),
        workspace_size=0.5,
        max_steps=500,
    )
    environment = MultiObjectPickPlaceEnvironment(config)
    fixed_specs = tuple(
        CameraSpec(spec.key, spec.camera_name, args.width, args.height)
        for spec in VLA_CAMERA_SPECS
        if spec.key in {"top", "side"}
    )
    pixel_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    box_heights: dict[tuple[str, str], list[int]] = defaultdict(list)

    with MujocoCameraRig(environment.model, fixed_specs) as cameras:
        for offset in range(args.seeds):
            seed = args.seed_start + offset
            object_key = "red_cube" if offset % 2 == 0 else "blue_cylinder"
            environment.reset(seed=seed, object_key=object_key)
            images = cameras.capture(environment.data)
            for camera_key, image in images.items():
                for color_key in ("red", "blue", "green", "yellow"):
                    mask = _color_mask(image, color_key)
                    count = int(mask.sum())
                    pixel_counts[(camera_key, color_key)].append(count)
                    box_heights[(camera_key, color_key)].append(_bounding_box_height(mask))

    minimum_visible_pixels = max(3, round(20 * args.width * args.height / (256 * 256)))
    failed: list[str] = []
    for camera_key in ("top", "side"):
        for color_key in ("red", "blue", "green", "yellow"):
            counts = pixel_counts[(camera_key, color_key)]
            heights = box_heights[(camera_key, color_key)]
            minimum = min(counts)
            visible_scenes = sum(count >= minimum_visible_pixels for count in counts)
            print(
                f"{camera_key:4s} {color_key:5s}: min_pixels={minimum:4d} "
                f"mean_pixels={np.mean(counts):7.1f} mean_box_height={np.mean(heights):5.1f}px "
                f"visible={visible_scenes}/{args.seeds}"
            )
            required_camera = {
                "red": "side",
                "blue": "side",
                "green": "top",
                "yellow": "top",
            }[color_key]
            if camera_key == required_camera and minimum < minimum_visible_pixels:
                failed.append(f"{camera_key}/{color_key} (minimum {minimum} pixels)")
            if camera_key != required_camera and visible_scenes < max(1, args.seeds // 2):
                failed.append(f"{camera_key}/{color_key} (visible in only {visible_scenes} scenes)")

    red_side_height = float(np.mean(box_heights[("side", "red")]))
    blue_side_height = float(np.mean(box_heights[("side", "blue")]))
    print(f"Side-view mean object heights: red={red_side_height:.1f}px blue={blue_side_height:.1f}px")
    if blue_side_height <= red_side_height:
        failed.append("side camera does not show the taller blue object as taller on average")
    if failed:
        raise SystemExit("Camera visibility check failed: " + ", ".join(failed))
    print(f"Camera visibility check passed across {args.seeds} randomized scenes.")


def _color_mask(image: np.ndarray, color_key: str) -> np.ndarray:
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    if color_key == "red":
        return (red > 110) & (red > green * 2) & (red > blue * 9 // 5)
    if color_key == "blue":
        return (blue > 110) & (blue > red * 2) & (blue > green * 17 // 10)
    if color_key == "green":
        return (green > 90) & (green > red * 9 // 5) & (green > blue * 7 // 5)
    if color_key == "yellow":
        return (red > 140) & (green > 110) & (blue < 70) & (red < green * 3 // 2)
    raise ValueError(f"Unsupported color key: {color_key}")


def _bounding_box_height(mask: np.ndarray) -> int:
    rows = np.flatnonzero(mask.any(axis=1))
    return 0 if rows.size == 0 else int(rows[-1] - rows[0] + 1)


if __name__ == "__main__":
    main()
