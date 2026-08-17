from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    key: str
    camera_name: str
    width: int = 256
    height: int = 256

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Camera key cannot be empty.")
        if not self.camera_name:
            raise ValueError("MuJoCo camera name cannot be empty.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive.")


DEFAULT_CAMERA_SPECS = (
    CameraSpec(key="front", camera_name="front_camera"),
    CameraSpec(key="wrist", camera_name="wrist_camera"),
)

VLA_CAMERA_SPECS = (
    CameraSpec(key="top", camera_name="top_camera"),
    CameraSpec(key="side", camera_name="side_camera"),
    CameraSpec(key="wrist", camera_name="wrist_camera"),
)


class MujocoCameraRig:
    def __init__(
        self,
        model: mujoco.MjModel,
        specs: Sequence[CameraSpec] = DEFAULT_CAMERA_SPECS,
    ) -> None:
        if not specs:
            raise ValueError("At least one camera specification is required.")
        keys = [spec.key for spec in specs]
        if len(keys) != len(set(keys)):
            raise ValueError("Camera keys must be unique.")

        for spec in specs:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, spec.camera_name)
            if camera_id < 0:
                raise ValueError(f"MuJoCo camera {spec.camera_name!r} does not exist in the model.")

        self.model = model
        self.specs = tuple(specs)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def capture(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for spec in self.specs:
            renderer = self._renderer(spec)
            renderer.update_scene(data, camera=spec.camera_name)
            image = np.asarray(renderer.render(), dtype=np.uint8).copy()
            expected_shape = (spec.height, spec.width, 3)
            if image.shape != expected_shape:
                raise RuntimeError(
                    f"Camera {spec.camera_name!r} returned shape {image.shape}, expected {expected_shape}."
                )
            images[spec.key] = image
        return images

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self) -> "MujocoCameraRig":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _renderer(self, spec: CameraSpec) -> mujoco.Renderer:
        resolution = (spec.height, spec.width)
        if resolution not in self._renderers:
            self._renderers[resolution] = mujoco.Renderer(
                self.model,
                height=spec.height,
                width=spec.width,
            )
        return self._renderers[resolution]
