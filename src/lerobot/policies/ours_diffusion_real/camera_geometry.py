"""Real-world pinhole camera calibration and world-to-image projection.

Preferred JSON convention:

    p_camera_h = world_to_camera_matrix @ p_world_h
    p_pixel_h = camera_intrinsics @ p_camera

Both equations use column vectors. camera_image_size is (height, width), and K is
expressed in pixels at exactly that resolution. Legacy SCDP pose fields are also
accepted: p_camera_row = (p_world_row - camera_world_position) @
camera_world_rotation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _vector(values: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{name} must contain {size} values.")
    result = tuple(float(value) for value in values)
    if not torch.isfinite(torch.tensor(result, dtype=torch.float64)).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _matrix(values: Any, rows: int, columns: int, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(values, (list, tuple)) or len(values) != rows:
        raise ValueError(f"{name} must be a {rows}x{columns} matrix.")
    result = tuple(_vector(row, columns, name) for row in values)
    return result


@dataclass(frozen=True)
class PinholeCameraCalibration:
    """A fixed pinhole camera pose and pixel-space intrinsic matrix.

    camera_axes_world is the legacy row-vector matrix. Its columns are the
    camera x/y/z axes expressed in world coordinates. The standard column-vector
    world-to-camera rotation is camera_axes_world.T.
    """

    image_size: tuple[int, int]
    intrinsics: tuple[tuple[float, ...], ...]
    camera_position_world: tuple[float, ...]
    camera_axes_world: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(int(value) <= 0 for value in self.image_size):
            raise ValueError("camera_image_size must be positive (height, width).")

        intrinsic = torch.tensor(self.intrinsics, dtype=torch.float64)
        if intrinsic.shape != (3, 3) or not torch.isfinite(intrinsic).all():
            raise ValueError("camera_intrinsics must be a finite 3x3 matrix.")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("camera_intrinsics focal lengths fx and fy must be positive.")
        expected_bottom = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        if not torch.allclose(intrinsic[2], expected_bottom, atol=1e-8, rtol=0):
            raise ValueError("camera_intrinsics bottom row must be [0, 0, 1].")

        position = torch.tensor(self.camera_position_world, dtype=torch.float64)
        rotation = torch.tensor(self.camera_axes_world, dtype=torch.float64)
        if position.shape != (3,) or not torch.isfinite(position).all():
            raise ValueError("Camera position must be a finite 3-vector.")
        if rotation.shape != (3, 3) or not torch.isfinite(rotation).all():
            raise ValueError("Camera rotation must be a finite 3x3 matrix.")
        identity = torch.eye(3, dtype=torch.float64)
        if not torch.allclose(rotation.T @ rotation, identity, atol=1e-4, rtol=1e-4):
            raise ValueError("Camera rotation must be orthonormal.")
        if not torch.isclose(
            torch.linalg.det(rotation), torch.tensor(1.0, dtype=torch.float64), atol=1e-4, rtol=1e-4
        ):
            raise ValueError("Camera rotation determinant must be +1.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> PinholeCameraCalibration:
        raw_image_size = _vector(values.get("camera_image_size"), 2, "camera_image_size")
        if any(not value.is_integer() for value in raw_image_size):
            raise ValueError("camera_image_size must contain integer pixel dimensions.")
        image_size = tuple(int(value) for value in raw_image_size)
        intrinsics = _matrix(values.get("camera_intrinsics"), 3, 3, "camera_intrinsics")

        has_matrix = "world_to_camera_matrix" in values
        legacy_names = ("camera_world_position", "camera_world_rotation")
        has_legacy = any(name in values for name in legacy_names)
        if has_matrix and has_legacy:
            raise ValueError(
                "Specify world_to_camera_matrix or the legacy camera pose fields, not both."
            )

        if has_matrix:
            transform = torch.tensor(
                _matrix(values["world_to_camera_matrix"], 4, 4, "world_to_camera_matrix"),
                dtype=torch.float64,
            )
            expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
            if not torch.allclose(transform[3], expected_bottom, atol=1e-8, rtol=0):
                raise ValueError("world_to_camera_matrix bottom row must be [0, 0, 0, 1].")
            rotation_world_to_camera = transform[:3, :3]
            translation_world_to_camera = transform[:3, 3]
            camera_axes_world = rotation_world_to_camera.T
            camera_position_world = -(camera_axes_world @ translation_world_to_camera)
            position = tuple(float(value) for value in camera_position_world.tolist())
            axes = tuple(
                tuple(float(value) for value in row)
                for row in camera_axes_world.tolist()
            )
        else:
            missing = [name for name in legacy_names if name not in values]
            if missing:
                raise ValueError(
                    "Calibration requires world_to_camera_matrix; "
                    f"missing legacy fallback fields: {missing}."
                )
            position = _vector(values["camera_world_position"], 3, "camera_world_position")
            axes = _matrix(values["camera_world_rotation"], 3, 3, "camera_world_rotation")

        return cls(
            image_size=image_size,
            intrinsics=intrinsics,
            camera_position_world=position,
            camera_axes_world=axes,
        )

    @classmethod
    def from_policy_config(
        cls,
        *,
        camera_image_size: Any,
        camera_intrinsics: Any,
        camera_world_position: Any,
        camera_world_rotation: Any,
    ) -> PinholeCameraCalibration:
        return cls.from_mapping(
            {
                "camera_image_size": camera_image_size,
                "camera_intrinsics": camera_intrinsics,
                "camera_world_position": camera_world_position,
                "camera_world_rotation": camera_world_rotation,
            }
        )

    @property
    def world_to_camera_matrix(self) -> tuple[tuple[float, ...], ...]:
        axes = torch.tensor(self.camera_axes_world, dtype=torch.float64)
        position = torch.tensor(self.camera_position_world, dtype=torch.float64)
        rotation = axes.T
        translation = -(rotation @ position)
        transform = torch.eye(4, dtype=torch.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return tuple(tuple(float(value) for value in row) for row in transform.tolist())

    def as_policy_config(self) -> dict[str, Any]:
        """Return legacy config names to keep existing checkpoints compatible."""
        return {
            "camera_image_size": self.image_size,
            "camera_intrinsics": self.intrinsics,
            "camera_world_position": self.camera_position_world,
            "camera_world_rotation": self.camera_axes_world,
        }


class PinholeCameraProjector(nn.Module):
    """Project arbitrary leading-shape world XYZ tensors to grid_sample coordinates."""

    def __init__(
        self,
        calibration: PinholeCameraCalibration,
        crop_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        image_height, image_width = calibration.image_size
        intrinsic = torch.tensor(calibration.intrinsics, dtype=torch.float32)

        if crop_shape is not None:
            if len(crop_shape) != 2 or any(int(value) <= 0 for value in crop_shape):
                raise ValueError("crop_shape must be positive (height, width).")
            crop_height, crop_width = crop_shape
            if crop_height > image_height or crop_width > image_width:
                raise ValueError(
                    f"crop_shape {crop_shape} exceeds camera image size {calibration.image_size}."
                )
            top = round((image_height - crop_height) / 2)
            left = round((image_width - crop_width) / 2)
            intrinsic[0, 2] -= left
            intrinsic[1, 2] -= top
            image_height, image_width = crop_height, crop_width

        self.register_buffer(
            "camera_position_world",
            torch.tensor(calibration.camera_position_world, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "camera_axes_world",
            torch.tensor(calibration.camera_axes_world, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("intrinsics", intrinsic, persistent=False)
        self.image_size = (image_height, image_width)

    def forward(self, points_world: Tensor) -> Tensor:
        points_camera = (points_world - self.camera_position_world) @ self.camera_axes_world
        inverse_z = points_camera[..., 2].reciprocal()
        normalized_x = points_camera[..., 0] * inverse_z
        normalized_y = points_camera[..., 1] * inverse_z

        u = (
            self.intrinsics[0, 0] * normalized_x
            + self.intrinsics[0, 1] * normalized_y
            + self.intrinsics[0, 2]
        )
        v = (
            self.intrinsics[1, 0] * normalized_x
            + self.intrinsics[1, 1] * normalized_y
            + self.intrinsics[1, 2]
        )
        image_height, image_width = self.image_size
        return torch.stack(
            (2 * (u / image_width) - 1, 2 * (v / image_height) - 1),
            dim=-1,
        )


def load_camera_calibration(path: str | Path) -> PinholeCameraCalibration:
    calibration_path = Path(path).expanduser()
    with calibration_path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, Mapping):
        raise ValueError("Camera calibration JSON must contain an object.")
    return PinholeCameraCalibration.from_mapping(values)


__all__ = [
    "PinholeCameraCalibration",
    "PinholeCameraProjector",
    "load_camera_calibration",
]
