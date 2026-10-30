"""Portable paths for all original registered statistical controls."""

import os
from pathlib import Path
from flycontrol.experiments.shared.paths import ROOT as REPO, sha256 as sha, write_json
from flycontrol.experiments.fig3.runtime import activate

activate()
from turning_control.context import configure_process

ROOT = Path(
    os.environ.get("FIGURE3_REGISTERED_OUTPUT", REPO / "build/figure3_registered")
)
DATA = REPO / "data/figure3"
PROTOCOL = REPO / "configs/figure3/registered_protocol.json"
