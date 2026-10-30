import os

# Headless (no DISPLAY): use EGL for offscreen OpenGL so vision/rendering works
# e.g. SSH, CI, Docker. Must be set before dm_control/mujoco are imported.
if "DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from .simulation import Simulation, SingleFlySimulation
from .fly import Fly
from .camera import Camera, YawOnlyCamera, ZStabilizedCamera, GravityAlignedCamera
from .util import get_data_path, load_config
from dm_control.rl.control import PhysicsError

from os import environ

is_rendering_skipped = (
    "SKIP_RENDERING" in environ and environ["SKIP_RENDERING"] == "true"
)

__all__ = [
    "Simulation",
    "SingleFlySimulation",
    "Fly",
    "Camera",
    "YawOnlyCamera",
    "ZStabilizedCamera",
    "GravityAlignedCamera",
    "PhysicsError",
    "get_data_path",
    "load_config",
    "is_rendering_skipped",
]
