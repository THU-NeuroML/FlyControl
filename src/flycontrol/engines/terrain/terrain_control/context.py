"""Run-local inputs and deterministic process setup."""
from flycontrol.paths import run_root, read_json, write_json, sha256 as sha
REPO = run_root()
CONFIG = REPO / "configs/figure2"
DATA = REPO / "data/figure2"
ROOT = REPO / "training"
GRAPH = DATA / "graph.npz"
PROTOCOL = CONFIG / "protocol.json"
_CONFIGURED = False

def configure_process():
    global _CONFIGURED
    if _CONFIGURED:
        return
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    _CONFIGURED = True
