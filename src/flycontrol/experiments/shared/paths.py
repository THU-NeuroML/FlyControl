"""Explicit run-local paths for internal analysis modules."""
from pathlib import Path
from flycontrol.paths import run_root, read_json, write_json, sha256
ROOT = run_root()
def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
