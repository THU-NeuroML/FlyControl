"""Package inputs and explicit run locations; no workstation paths."""

from pathlib import Path
import hashlib
import json
import os

PACKAGE = Path(__file__).resolve().parent
ENGINES = PACKAGE / "engines"
EXPERIMENTS = PACKAGE / "experiments"
CONFIGS = PACKAGE / "configs"
if not CONFIGS.exists():  # Editable source checkout.
    CONFIGS = PACKAGE.parents[1] / "configs"


def run_root():
    value = os.environ.get("FLYCONTROL_RUN")
    if not value:
        raise RuntimeError("Use flycontrol with --run-dir; a run directory is required")
    return Path(value).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*.py")):
        digest.update(str(path.relative_to(PACKAGE)).encode())
        digest.update(path.read_bytes())
    for path in sorted(CONFIGS.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}:
            digest.update(("configs/" + str(path.relative_to(CONFIGS))).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def configuration_manifest(root):
    root = Path(root)
    paths = [root / "protocol.json"]
    paths.extend(p for p in (root / "configs").rglob("*")
                 if p.is_file() and "resolved_maze" not in p.relative_to(root / "configs").parts)
    if (root / "training/protocol.json").exists():
        paths.append(root / "training/protocol.json")
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}
