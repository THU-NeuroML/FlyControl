"""Isolated numerical workers. Invoke through the public flycontrol CLI."""

from pathlib import Path
import sys

from flycontrol.paths import run_root, read_json, write_json
from flycontrol.pipeline import BODIES, checkpoint


def main():
    family, action, method, seed, option = sys.argv[1:]
    seed = int(seed)
    root = run_root()
    spec = read_json(root / "run.json")
    folder = root / f"training/{method}_seed{seed}"
    out = root / f"evaluation/{method}_seed{seed}"
    cp = checkpoint(root, spec, method, seed)
    if family == "ant":
        if spec["experiment"] == "ant-flat":
            from flycontrol.experiments.fig5.flat_workflow import train, evaluate
        else:
            from flycontrol.experiments.fig5.workflow import train, evaluate
        if action == "train":
            train(root, spec, method, seed, option)
        else:
            evaluate(root, spec, method, seed)
    elif family in BODIES:
        from flycontrol.experiments.fig6.workflow import body_train, body_evaluate
        if action == "train":
            body_train(root, spec, method, seed, option)
        else:
            body_evaluate(root, spec, method, seed)
    elif family == "turning":
        if action == "train":
            from turning_control.train import train
            train(method, seed, folder, option)
        else:
            from turning_control.evaluate import evaluate
            if not (out / "COMPLETE.json").exists():
                evaluate(method, seed, out, cp, workers=int(option))
    elif family == "speed":
        if action == "train":
            from body_control.train import train
            base = folder / "base"
            if not (base / "COMPLETE.json").exists():
                train(method, seed, base, option, stage="base")
            train(method, seed, folder / "continuation", option, stage="continue", source=base / "checkpoint.pt")
        else:
            from body_control.evaluate import evaluate
            evaluate(method, seed, out, cp, workers=int(option))
    elif family == "terrain":
        if action == "train":
            from flycontrol.experiments.fig2.current.train import main as train
            sys.argv = ["train", "--method", method, "--seed", str(seed), "--device", option]
            train()
        else:
            from flycontrol.experiments.fig2.workflow import evaluate
            evaluate(root, spec, method, seed, cp, workers=int(option))
    else:
        raise ValueError(family)


if __name__ == "__main__":
    main()
