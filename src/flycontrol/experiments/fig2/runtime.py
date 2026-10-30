"""Activate only the terrain adapter and the shared numerical engines."""

import os
import sys
from fig6.paths import ROOT


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
    sys.path[:0] = [str(ROOT / "engines" / p) for p in ("terrain", "nmf", "common")]
