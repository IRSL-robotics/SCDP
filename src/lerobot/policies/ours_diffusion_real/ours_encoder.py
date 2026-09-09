#!/usr/bin/env python

import torch
import torchvision.transforms.functional as TVF

from lerobot.policies.ours_diffusion.ours_encoder import OursDiffusionRgbEncoder


class OursDiffusionRealRgbEncoder(OursDiffusionRgbEncoder):
    """SCDP RGB encoder with an optional calibration-preserving center crop."""

    def __init__(self, config):
        super().__init__(config)
        self.crop_shape = config.crop_shape

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        if self.crop_shape is not None:
            images = TVF.center_crop(images, list(self.crop_shape))
        return super().forward(images)
