"""Focused LeRobot 0.4.1 fork for the SCDP Meta-World experiments."""

from lerobot.__version__ import __version__

available_envs = ["metaworld"]
available_policies = ["diffusion", "ours_diffusion"]

__all__ = ["__version__", "available_envs", "available_policies"]
