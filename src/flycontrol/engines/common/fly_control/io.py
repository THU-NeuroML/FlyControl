"""Checkpoint I/O, JSON encoding and reward-only execution guards."""

import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer):
            return int(v)
        return v

    _write_json(path, clean(value))


class NoTeacherImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("fly_control.", "body_control.")) and (
            "teacher" in fullname or "distill" in fullname
        ):
            raise ImportError(
                "Teachers and distillation are disabled in reward-only PPO: " + fullname
            )
        return None


class CheckpointYield(Exception):
    pass
