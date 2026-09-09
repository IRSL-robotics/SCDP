import json

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.ours_diffusion_real.configuration_diffusion import OursDiffusionRealConfig
from lerobot.policies.ours_diffusion_real.modeling_diffusion import OursDiffusionRealPolicy

INPUT_FEATURES = {
    "observation.images.cam_3": PolicyFeature(FeatureType.VISUAL, (3, 240, 320)),
    "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
}
OUTPUT_FEATURES = {"action": PolicyFeature(FeatureType.ACTION, (7,))}


def make_config(**overrides):
    values = {
        "input_features": INPUT_FEATURES,
        "output_features": OUTPUT_FEATURES,
        "device": "cpu",
        "crop_shape": None,
        "down_dims": (32, 64, 128),
        "action_min": (-0.05, -0.04, -0.03),
        "action_max": (0.05, 0.04, 0.03),
        "camera_image_size": (240, 320),
        "camera_intrinsics": ((320.0, 0.0, 160.0), (0.0, 240.0, 120.0), (0.0, 0.0, 1.0)),
        "camera_world_position": (0.0, 0.0, 0.0),
        "camera_world_rotation": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    }
    values.update(overrides)
    return OursDiffusionRealConfig(**values)


def test_real_scdp_constructs_and_projects_on_cpu():
    policy = OursDiffusionRealPolicy(make_config())
    raw_states = torch.tensor([[[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]]])
    projected = policy.diffusion._real_projection(raw_states)

    torch.testing.assert_close(projected, torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]))
    assert projected.device.type == "cpu"


def test_real_config_keeps_calibration_and_action_bounds(tmp_path):
    config = make_config()
    config._save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())

    assert saved["camera_image_size"] == [240, 320]
    assert saved["camera_world_position"] == [0.0, 0.0, 0.0]
    assert saved["action_min"] == [-0.05, -0.04, -0.03]

    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, OursDiffusionRealConfig)


def test_real_scdp_rejects_random_crop():
    with pytest.raises(ValueError, match="Random cropping"):
        make_config(crop_shape=(224, 224), crop_is_random=True)
