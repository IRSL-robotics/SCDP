#!/usr/bin/env python
"""Processor API required by Diffusion Policy and SCDP."""

from .batch_processor import AddBatchDimensionProcessorStep
from .converters import batch_to_transition, create_transition, transition_to_batch
from .core import PolicyAction, RobotAction, RobotObservation
from .device_processor import DeviceProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from .pipeline import PolicyProcessorPipeline
from .rename_processor import RenameObservationsProcessorStep

__all__ = [
    "AddBatchDimensionProcessorStep",
    "DeviceProcessorStep",
    "NormalizerProcessorStep",
    "PolicyAction",
    "PolicyProcessorPipeline",
    "RenameObservationsProcessorStep",
    "RobotAction",
    "RobotObservation",
    "UnnormalizerProcessorStep",
    "batch_to_transition",
    "create_transition",
    "transition_to_batch",
]
