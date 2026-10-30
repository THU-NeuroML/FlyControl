"""Each numerical engine runs in a separate interpreter with an explicit root."""

from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import subprocess
import sys

from flycontrol.paths import PACKAGE, ENGINES, read_json


def environment(root, *engines):
    root = Path(root).resolve()
    env = os.environ.copy()
    paths = [PACKAGE.parent] + [ENGINES / name for name in engines]
    env.update(
        FLYCONTROL_RUN=str(root),
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        FIG6_ASSET_ROOT=str(root / "assets/external"),
        NMF_DATA_ROOT=str(root / "assets/nmf/data"),
        TERRAIN_STUDY_ROOT=str(root / "training"),
        SPEED_STUDY_ROOT=str(root / "training"),
        FIGURE2_ANALYSIS_OUTPUT=str(root / "data/figure2"),
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(root / ".cache/matplotlib"),
        MUJOCO_GL=env.get("MUJOCO_GL", "egl"),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    return env


def worker(root, family, action, *args, device="cpu"):
    parts = [family, "common"]
    if family in {"turning", "speed", "terrain"}:
        parts.insert(1, "nmf")
    env = environment(root, *parts)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-m", "flycontrol.worker", family, action, *map(str, args)]
    print(" ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=root, env=env, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{family} {action} failed (exit {error.returncode}); inspect the worker output and {root / 'training'}") from error


@contextmanager
def run_lock(root):
    """Reject concurrent mutations of a run instead of mixing outputs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is modifying {root}") from error
        yield
