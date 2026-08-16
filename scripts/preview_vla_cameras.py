from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from robot_manipulation_pi0.sim import PickPlaceConfig, PickPlaceEnvironment, ScriptedOraclePolicy
from robot_manipulation_pi0.task_spec import DEFAULT_TASK_INSTRUCTION
from robot_manipulation_pi0.vla import (
    DEFAULT_CAMERA_SPECS,
    CameraSpec,
    MujocoCameraRig,
    capture_vla_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render front and wrist RGB observations from one oracle rollout.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("outputs/vla_camera_preview.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    config = PickPlaceConfig(
        robot_name="franka_panda",
        object_names=("cube",),
        workspace_size=0.5,
        max_steps=args.max_steps,
    )
    environment = PickPlaceEnvironment(config)
    policy = ScriptedOraclePolicy(environment)
    observation = environment.reset(seed=args.seed)
    camera_specs = tuple(
        CameraSpec(
            key=spec.key,
            camera_name=spec.camera_name,
            width=args.width,
            height=args.height,
        )
        for spec in DEFAULT_CAMERA_SPECS
    )
    captures: list[tuple[str, dict[str, object]]] = []

    with MujocoCameraRig(environment.model, camera_specs) as cameras:
        last_stage: str | None = None
        result = None
        for _ in range(config.max_steps):
            action = policy.act(observation)
            if policy.stage != last_stage:
                vla_observation = capture_vla_observation(
                    environment.model,
                    environment.data,
                    cameras,
                    DEFAULT_TASK_INSTRUCTION.instruction,
                )
                captures.append((policy.stage, vla_observation.as_dict()))
                last_stage = policy.stage

            result = environment.step(action)
            observation = result.observation
            if result.terminated or result.truncated:
                break

        if result is None:
            raise RuntimeError("Camera preview rollout did not execute any steps.")
        final_label = "final_success" if result.info["success"] else "final_timeout"
        final_observation = capture_vla_observation(
            environment.model,
            environment.data,
            cameras,
            DEFAULT_TASK_INSTRUCTION.instruction,
        )
        captures.append((final_label, final_observation.as_dict()))

    _save_contact_sheet(captures, camera_specs, args.output)
    print(f"Rollout success: {bool(result.info['success'])}")
    print(f"Steps: {observation['step_count']}")
    print(f"Robot state dimension: {final_observation.state.shape[0]}")
    for key, image in final_observation.images.items():
        print(f"Camera {key}: shape={image.shape} dtype={image.dtype}")
    print(f"Saved camera preview: {args.output}")


def _save_contact_sheet(
    captures: list[tuple[str, dict[str, object]]],
    camera_specs: tuple[CameraSpec, ...],
    output: Path,
) -> None:
    label_height = 24
    cell_width = camera_specs[0].width
    cell_height = camera_specs[0].height
    canvas = Image.new(
        "RGB",
        (cell_width * len(camera_specs), (cell_height + label_height) * len(captures)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for row, (stage, observation) in enumerate(captures):
        top = row * (cell_height + label_height)
        for column, spec in enumerate(camera_specs):
            left = column * cell_width
            image = observation[f"observation.images.{spec.key}"]
            if not hasattr(image, "shape"):
                raise TypeError(f"Camera observation {spec.key!r} is not an image array.")
            canvas.paste(Image.fromarray(image), (left, top + label_height))
            draw.text((left + 4, top + 4), f"{stage} / {spec.key}", fill="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == "__main__":
    main()
