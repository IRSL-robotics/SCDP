import json

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.ours_diffusion.configuration_diffusion import OursDiffusionConfig
from lerobot.policies.ours_diffusion.modeling_diffusion import OursDiffusionPolicy

INPUT_FEATURES = {
    "observation.images.world": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
    "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
}
OUTPUT_FEATURES = {
    "action": PolicyFeature(FeatureType.ACTION, (4,)),
}


def make_scdp_config():
    return OursDiffusionConfig(
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        device="cpu",
        crop_shape=None,
        down_dims=(32, 64, 128),
        action_min=(-1.0, -0.5, -0.25),
        action_max=(1.0, 0.5, 0.25),
        sample_step=8,
    )


def test_baseline_uses_identity_image_normalization():
    config = DiffusionConfig(
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        device="cpu",
    )
    assert config.normalization_mapping["VISUAL"] is NormalizationMode.IDENTITY


def test_scdp_constructs_and_projects_on_cpu():
    policy = OursDiffusionPolicy(make_scdp_config())
    raw_states = torch.tensor([[[0.0, 0.6, 0.2], [0.01, 0.61, 0.21]]])
    projected = policy.diffusion._metaworld_projection(raw_states)

    assert projected.shape == (1, 2, 2)
    assert projected.device.type == "cpu"
    assert torch.isfinite(projected).all()


def test_sampled_trajectory_projects_all_points_in_one_batch():
    policy = OursDiffusionPolicy(make_scdp_config())
    model = policy.diffusion
    raw_states = torch.tensor([[[0.0, 0.6, 0.2], [0.01, 0.61, 0.21]]])
    sample = torch.linspace(-0.5, 0.5, 64).reshape(1, 16, 4)

    actual = model._project_sampled_trajectory(raw_states, sample)

    dynamic_states = raw_states
    expected = [model._metaworld_projection(raw_states)]
    for index in range(1, 1 + model.config.sample_step):
        delta = model._action_unnormalizer(sample[:, index, :3]).unsqueeze(1)
        dynamic_states = model._metaworld_movement(dynamic_states, delta)
        expected.append(model._metaworld_projection(dynamic_states))
    expected = torch.stack(expected, dim=2)

    assert actual.shape == (1, 2, model.config.sample_step + 1, 2)
    torch.testing.assert_close(actual, expected)


def test_scdp_can_construct_legacy_shared_group_norm_encoder():
    config = make_scdp_config()
    config.use_shared_group_norm_in_residual_blocks = True
    policy = OursDiffusionPolicy(config)
    keys = set(policy.state_dict())

    assert "diffusion.rgb_encoder.blockB.res1.normalize.weight" in keys
    assert "diffusion.rgb_encoder.blockB.res1.normalize1.weight" not in keys


def test_scdp_config_keeps_action_bounds(tmp_path):
    config = make_scdp_config()
    config._save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["action_min"] == [-1.0, -0.5, -0.25]
    assert saved["action_max"] == [1.0, 0.5, 0.25]
