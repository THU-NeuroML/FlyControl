"""Notebook presentation helpers; numerical work stays in the public CLI."""

import json
from collections import deque
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from IPython.display import Image, Markdown, display
import pandas as pd


STAGES = ("prepare", "train", "evaluate", "probe", "analyze", "plot")


def run_stage(stage, config, run_dir, device="cpu", workers=1):
    """Execute a smoke stage, retaining a compact display and a complete log."""
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
    log_path = run_dir / "logs" / f"notebook-{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=8)
    progress = display(Markdown(f"**Running {stage}…**"), display_id=True)
    refreshed = time.monotonic()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, start_new_session=True) as process, log_path.open("w") as log:
        log.write(shlex.join(command) + "\n")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                if time.monotonic() - refreshed >= 2:
                    progress.update(Markdown(f"**Running {stage}…**\n\n```text\n" + "\n".join(tail) + "\n```"))
                    refreshed = time.monotonic()
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
            progress.update(Markdown(f"**Interrupted: {stage}.** Full log: `logs/{log_path.name}` within the run directory."))
            raise
    if returncode:
        progress.update(Markdown(f"**Failed: {stage}.**\n\n```text\n" + "\n".join(tail) + "\n```"))
        raise subprocess.CalledProcessError(returncode, command)
    record = json.loads((run_dir / "status" / f"{stage}.json").read_text())
    progress.update(Markdown(
        f"**Completed: {stage}** · {record['elapsed_s']:.1f} seconds · smoke test\n\n"
        f"Full execution log: `logs/{log_path.name}` within the run directory."
    ))


def show_outputs(run_dir):
    """Show recorded settings, completion times and a few diagnostic outputs."""
    run_dir = Path(run_dir).resolve()
    from flycontrol.pipeline import load_run
    spec = load_run(run_dir)
    if spec.get("smoke") is not True:
        raise ValueError("This viewer expects a smoke run.")
    display(Markdown("**SMOKE TEST — not paper results.**"))
    display(pd.DataFrame([{
        "experiment": spec["experiment"], "methods": ", ".join(spec["methods"]),
        "training seeds": str(spec["seeds"]),
    }]))
    display(Markdown(f"Source/configuration SHA-256: `{spec['source_sha256']}`"))
    stages = []
    for stage in STAGES:
        record = json.loads((run_dir / "status" / f"{stage}.json").read_text())
        if record.get("source_sha256") != spec["source_sha256"] or record.get("smoke") is not True:
            raise ValueError(f"Stage record does not match the smoke run: {stage}")
        stages.append({"stage": stage, "completed": True,
                       "elapsed seconds": round(record["elapsed_s"], 2)})
    display(pd.DataFrame(stages))
    analysis = run_dir / "analysis"
    preferred = ["seed_means.csv", "seed_sensitivity.csv", "regional_seed_effects.csv",
                 "summary.csv", "episodes.csv"]
    tables = [analysis / name for name in preferred if (analysis / name).exists()]
    tables += [p for p in sorted(analysis.glob("*.csv")) if p not in tables]
    for path in tables[:3]:
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
    display(Markdown("The complete numerical files and figures are saved in `RUN_DIR`. "
                     "The tables and images displayed here are embedded in this notebook."))
