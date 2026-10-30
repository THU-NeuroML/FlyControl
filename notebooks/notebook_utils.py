"""Notebook presentation helpers; numerical work stays in the public CLI."""

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from IPython.display import Image, Markdown, display
import pandas as pd


STAGES = ("prepare", "train", "evaluate", "probe", "analyze", "plot")


def run_stage(stage, config, run_dir, device="cpu", workers=1):
    """Stream a smoke stage using the notebook kernel's Python interpreter."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    run_dir = Path(run_dir).resolve()
    command = [sys.executable, "-u", "-m", "flycontrol", stage,
               "--run-dir", str(run_dir), "--device", device,
               "--workers", str(workers)]
    if stage == "prepare":
        command += ["--config", config, "--smoke"]
    else:
        spec = json.loads((run_dir / "run.json").read_text())
        if spec.get("smoke") is not True or spec["experiment"] != config:
            raise ValueError("Choose the matching smoke run; use the terminal for full runs.")
    print(shlex.join(command), flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, start_new_session=True) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            returncode = process.wait()
        except BaseException:
            # Stop simulation workers too when the notebook kernel is interrupted.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            except ProcessLookupError:
                pass
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    print(f"Completed: {stage}", flush=True)


def show_outputs(run_dir):
    """Show recorded settings, completion times and a few diagnostic outputs."""
    run_dir = Path(run_dir).resolve()
    spec = json.loads((run_dir / "run.json").read_text())
    if spec.get("smoke") is not True:
        raise ValueError("This viewer expects a smoke run.")
    display(Markdown("**SMOKE TEST — not paper results.**"))
    display(pd.DataFrame([{
        "experiment": spec["experiment"], "methods": ", ".join(spec["methods"]),
        "training seeds": str(spec["seeds"]), "source SHA256": spec["source_sha256"],
    }]))
    stages = []
    for stage in STAGES:
        record = json.loads((run_dir / "status" / f"{stage}.json").read_text())
        stages.append({"stage": stage, "completed": True,
                       "elapsed seconds": round(record["elapsed_s"], 2)})
    display(pd.DataFrame(stages))
    for path in sorted((run_dir / "analysis").glob("*.csv"))[:3]:
        display(Markdown(f"`{path.relative_to(run_dir)}` (first eight rows)"))
        try:
            display(pd.read_csv(path).head(8))
        except pd.errors.EmptyDataError:
            print("No tabular records; check the analysis manifest for exclusions.")
    figures = sorted((run_dir / "figures/paper").glob("*.png"))
    if not figures:
        figures = sorted((run_dir / "figures").rglob("*.png"))
    for path in figures[:3]:
        display(Markdown(f"`{path.relative_to(run_dir)}`"))
        display(Image(filename=str(path)))
    print(f"All generated outputs: {run_dir}")
