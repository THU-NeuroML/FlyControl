"""Generated turning artifacts and checkpoints; no archive fallback."""
from flycontrol.paths import run_root, read_json
from flycontrol.pipeline import checkpoint as resolve_checkpoint
ROOT = run_root()
DATA = ROOT / 'data/figure3'
MECHANISMS = DATA / 'mechanisms'
BANK = MECHANISMS / 'landscape/flatness'

def checkpoint(method, seed, initial=False):
    path = resolve_checkpoint(ROOT, read_json(ROOT/'run.json'), method, seed)
    return path.parent/'initial_checkpoint.pt' if initial else path

def generated_path(path):
    from pathlib import Path
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Neural history is outside this run')
    return path
