#!/usr/bin/env python
"""Policy helpers retained by the focused SCDP LeRobot fork."""

from __future__ import annotations

from typing import Any, TypedDict

import torch
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.ours_diffusion.configuration_diffusion import OursDiffusionConfig
from lerobot.policies.ours_diffusion_real.configuration_diffusion import OursDiffusionRealConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    if name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    if name == "ours_diffusion":
        from lerobot.policies.ours_diffusion.modeling_diffusion import OursDiffusionPolicy

        return OursDiffusionPolicy
    if name == "ours_diffusion_real":
        from lerobot.policies.ours_diffusion_real.modeling_diffusion import OursDiffusionRealPolicy

        return OursDiffusionRealPolicy
    raise NotImplementedError(f"Policy with name {name!r} is not included in this focused fork.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    if policy_type == "ours_diffusion":
        return OursDiffusionConfig(**kwargs)
    if policy_type == "ours_diffusion_real":
        return OursDiffusionRealConfig(**kwargs)
    raise ValueError(f"Policy type {policy_type!r} is not included in this focused fork.")


class ProcessorConfigKwargs(TypedDict, total=False):
    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if pretrained_path:
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    if isinstance(policy_cfg, DiffusionConfig):
        from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
    elif isinstance(policy_cfg, (OursDiffusionConfig, OursDiffusionRealConfig)):
        from lerobot.policies.ours_diffusion.processor_diffusion import make_diffusion_pre_post_processors
    else:
        raise NotImplementedError(f"Processor for policy type {policy_cfg.type!r} is not included.")

    return make_diffusion_pre_post_processors(
        config=policy_cfg,
        dataset_stats=kwargs.get("dataset_stats"),
    )
