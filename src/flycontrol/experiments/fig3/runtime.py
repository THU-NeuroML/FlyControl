"""Activate the turning namespace without replacing the speed adapter."""

import os
import sys
from flycontrol.experiments.shared.paths import ROOT


def activate():
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for part in reversed(("turning", "nmf", "common")):
        path = str(ROOT / "engines" / part)
        if path not in sys.path:
            sys.path.insert(0, path)
