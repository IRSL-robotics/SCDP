#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.ours_diffusion.configuration_diffusion import OursDiffusionConfig
from lerobot.policies.ours_diffusion_real.camera_geometry import (
    PinholeCameraCalibration,
)


@PreTrainedConfig.register_subclass("ours_diffusion_real")
@dataclass
class OursDiffusionRealConfig(OursDiffusionConfig):
    """SCDP configuration for a calibrated fixed real-world RGB camera.

    Intrinsics are expressed in pixels of camera_image_size. These fields retain
    the legacy row-vector checkpoint representation. New calibration JSON files
    should use the standard world_to_camera_matrix format documented in
    camera_geometry.py and configs/real_camera.example.json.
    """

    camera_image_size: tuple[int, int] = (240, 320)
    camera_intrinsics: tuple[tuple[float, float, float], ...] = (
        (304.325225, 0.0, 163.35962),
        (0.0, 303.736235, 124.355475),
        (0.0, 0.0, 1.0),
    )
    camera_world_position: tuple[float, float, float] = (1.322, -0.013, 0.609)
    camera_world_rotation: tuple[tuple[float, float, float], ...] = (
        (-0.00662099, 0.47824121, -0.87820357),
        (0.99996514, 0.00763364, -0.00338195),
        (0.00508650, -0.87819535, -0.47827509),
    )
    state_embedding_dim: int = 64

    def __post_init__(self):
        super().__post_init__()
        if len(self.camera_image_size) != 2 or any(value <= 0 for value in self.camera_image_size):
            raise ValueError(
                f"`camera_image_size` must be positive (height, width), got {self.camera_image_size}."
            )
        PinholeCameraCalibration.from_policy_config(
            camera_image_size=self.camera_image_size,
            camera_intrinsics=self.camera_intrinsics,
            camera_world_position=self.camera_world_position,
            camera_world_rotation=self.camera_world_rotation,
        )
        if self.state_embedding_dim <= 0:
            raise ValueError("`state_embedding_dim` must be positive.")
        if self.crop_shape is not None and self.crop_is_random:
            raise ValueError(
                "Random cropping is incompatible with calibrated SCDP projection. "
                "Use a center crop (`crop_is_random=False`) or disable cropping."
            )

    def validate_features(self) -> None:
        super().validate_features()
        if len(self.image_features) != 1:
            raise ValueError("Real-world SCDP currently requires exactly one calibrated RGB camera.")
        if self.env_state_feature is not None:
            raise ValueError("Real-world SCDP does not support an environment-state feature.")

        image_feature = next(iter(self.image_features.values()))
        image_size = tuple(image_feature.shape[-2:])
        if image_size != tuple(self.camera_image_size):
            raise ValueError(
                "Camera calibration and dataset image size differ: "
                f"calibration={self.camera_image_size}, dataset={image_size}."
            )
