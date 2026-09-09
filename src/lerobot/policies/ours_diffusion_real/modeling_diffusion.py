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

import json

import einops
import torch
from torch import Tensor, nn

from lerobot.policies.ours_diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    OursDiffusionPolicy,
    _make_noise_scheduler,
)
from lerobot.policies.ours_diffusion_real.configuration_diffusion import OursDiffusionRealConfig
from lerobot.policies.ours_diffusion_real.ours_encoder import OursDiffusionRealRgbEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE


class OursDiffusionRealPolicy(OursDiffusionPolicy):
    """SCDP policy using a calibrated fixed camera for real-world observations."""

    config_class = OursDiffusionRealConfig
    name = "ours_diffusion_real"

    def __init__(self, config: OursDiffusionRealConfig):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = OursDiffusionRealModel(config)
        self.reset()


class OursDiffusionRealModel(DiffusionModel):
    """Real-world geometry on top of the optimized SCDP denoising loop."""

    def __init__(self, config: OursDiffusionRealConfig):
        nn.Module.__init__(self)
        self.config = config

        state_dim = config.robot_state_feature.shape[0]
        global_cond_dim = config.state_embedding_dim
        self.rgb_encoder = OursDiffusionRealRgbEncoder(config)
        global_cond_dim += self.rgb_encoder.feature_dim

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, config.state_embedding_dim),
            nn.LayerNorm(config.state_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.state_embedding_dim, config.state_embedding_dim),
            nn.LayerNorm(config.state_embedding_dim),
            nn.SiLU(),
        )
        self.unet = DiffusionConditionalUnet1d(
            config,
            global_cond_dim=global_cond_dim * config.n_obs_steps,
        )
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )
        self._inference_schedule_key = None
        self._model_timesteps = None

        action_min = config.action_min
        action_max = config.action_max
        if action_min is None or action_max is None:
            if config.dataset_stats_path is None:
                raise ValueError(
                    "Real-world SCDP needs action bounds. Set `action_min`/`action_max`, "
                    "or provide `dataset_stats_path` for a legacy checkpoint."
                )
            with open(config.dataset_stats_path) as stream:
                stats = json.load(stream)
            action_min = tuple(stats["action"]["min"][:3])
            action_max = tuple(stats["action"]["max"][:3])

        self.register_buffer("action_min", torch.tensor(action_min, dtype=torch.float32), persistent=False)
        self.register_buffer("action_max", torch.tensor(action_max, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "camera_world_position",
            torch.tensor(config.camera_world_position, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "camera_world_rotation",
            torch.tensor(config.camera_world_rotation, dtype=torch.float32),
            persistent=False,
        )

        image_height, image_width = config.camera_image_size
        intrinsic = torch.tensor(config.camera_intrinsics, dtype=torch.float32)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        if config.crop_shape is not None:
            crop_height, crop_width = config.crop_shape
            top = round((image_height - crop_height) / 2)
            left = round((image_width - crop_width) / 2)
            cx = cx - left
            cy = cy - top
            image_height, image_width = crop_height, crop_width
        self.register_buffer(
            "projection_intrinsics",
            torch.stack([fx, fy, cx, cy]),
            persistent=False,
        )
        self.projection_image_size = (image_height, image_width)

    def _prepare_state_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        state_features = self.state_mlp(batch[OBS_STATE])
        return einops.rearrange(state_features, "b s ... -> (b s) ...")

    def _real_projection(self, raw_states: Tensor) -> Tensor:
        camera_states = (raw_states - self.camera_world_position) @ self.camera_world_rotation
        fx, fy, cx, cy = self.projection_intrinsics
        u = (fx * (camera_states[:, :, 0] / camera_states[:, :, 2]) + cx).unsqueeze(-1)
        v = (fy * (camera_states[:, :, 1] / camera_states[:, :, 2]) + cy).unsqueeze(-1)
        projected = torch.cat([u, v], dim=-1)

        image_height, image_width = self.projection_image_size
        projected[:, :, 0] = 2 * (projected[:, :, 0] / image_width) - 1
        projected[:, :, 1] = 2 * (projected[:, :, 1] / image_height) - 1
        return projected

    def _real_movement(self, current_states: Tensor, delta_states: Tensor) -> Tensor:
        return current_states + torch.clamp(delta_states, min=-1, max=1)

    # The shared optimized loop calls these hooks. Aliases keep the Meta-World
    # implementation untouched while replacing only the real-world geometry.
    def _metaworld_projection(self, raw_states: Tensor) -> Tensor:
        return self._real_projection(raw_states)

    def _metaworld_movement(
        self,
        current_states: Tensor,
        delta_states: Tensor,
        action_scale: float = 1.0,
    ) -> Tensor:
        del action_scale
        return self._real_movement(current_states, delta_states)
