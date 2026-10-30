"""Continual embodied-transfer experiments on Gymnasium Ant."""

from .env import ContinualAntEnv, load_friction_schedule

__all__ = ["ContinualAntEnv", "load_friction_schedule"]
