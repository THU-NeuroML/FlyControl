"""Read-only evidence inputs and separate destinations for fresh analyses."""

from pathlib import Path
import os
import torch
from flycontrol.experiments.shared.paths import ROOT, read_json, sha256 as sha
from body_control.policy import build_policy
from body_control.context import write_json

DATA = ROOT / "data/figure4"
O = Path(os.environ.get("SPEED_ANALYSIS_ROOT", ROOT / "build/figure4_analysis"))
ANATOMY = DATA / "anatomy"
NODE_SOURCE = ANATOMY / "nodes.csv.gz"
CONFIG = read_json(ROOT / "configs/figure4/evaluation.json")
SEEDS = read_json(ROOT / "run.json")["seeds"]
METHODS = ["connectome", "mlp"]
CASES = read_json(ROOT / "configs/figure4/analysis.json")["mechanism_cases"]


def checkpoint(method, seed):
    from flycontrol.pipeline import checkpoint as resolve_checkpoint
    return resolve_checkpoint(ROOT, read_json(ROOT / 'run.json'), method, seed)


def load_policy(method, seed, device="cpu"):
    policy = build_policy(method, seed, device, 40)
    payload = torch.load(
        checkpoint(method, seed), map_location="cpu", weights_only=False
    )
    assert payload["obs_rms_state"] is None
    policy.load_state_dict(payload["policy_state"])
    policy.eval()
    return policy


def name_case(case):
    return f"{case['kind']}_c{case['command_seed']}_p{case['physics_seed']}"


def baseline_path(method, seed, case):
    relative = Path("rollouts") / f"{method}_{seed}" / (name_case(case) + ".npz")
    fresh = O / relative
    return fresh


def evaluation_path(method, seed, case):
    return ROOT / "evaluation" / f"{method}_seed{seed}" / name_case(case)


def event_path(seed):
    relative = Path("events") / str(seed) / "neural_responses.npz"
    return O / relative


def setup():
    O.mkdir(parents=True, exist_ok=True)
    write_json(
        O / "INPUTS.json",
        dict(
            seeds=SEEDS,
            steps=15872,
            models={f"{m}_{s}": sha(checkpoint(m, s)) for s in SEEDS for m in METHODS},
        ),
    )
