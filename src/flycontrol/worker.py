"""Isolated numerical workers. Invoke through the public flycontrol CLI."""

from pathlib import Path
import sys

from .paths import run_root, read_json, write_json
from .pipeline import BODIES, checkpoint


def body_train(root, spec, method, seed, device):
    conf = read_json(root / "protocol.json")
    sys.argv = ["train", "--root", str(root / "training"), "--body", spec["experiment"],
                "--method", method, "--seed", str(seed), "--readout", conf["readout"], "--device", device]
    from fly_control.training import main
    main()


def body_evaluate(root, spec, method, seed):
    import subprocess
    from .runtime import environment
    conf = read_json(root / "protocol.json")
    name = f"{spec['experiment']}_{method}_{conf['readout']}_seed{seed}"
    for evaluation_seed in conf["evaluation_seeds"]:
        out = root / f"evaluation/{method}_seed{seed}/eval{evaluation_seed}"
        job = dict(id=f"{name}_eval{evaluation_seed}", body=spec["experiment"], method=method,
                   run=name, training_root=str(root / "training"), training_steps=conf["ppo"]["total_timesteps"],
                   evaluation_seed=evaluation_seed, command=conf["command"], perturb_onset_s=conf["perturb_onset_s"])
        if spec["smoke"]:
            job["smoke_horizon_s"] = 0.2
        write_json(out / "protocol.json", dict(jobs=[job], conditions=conf["conditions"]))
        write_json(out / "control.json", dict(stop_all=False, yield_all=False))
        subprocess.run([sys.executable, "-m", "fly_control.evaluate", "--root", str(out), "--job", "0"],
                       cwd=root, env=environment(root, spec["experiment"], "common"), check=True)


def main():
    family, action, method, seed, option = sys.argv[1:]
    seed = int(seed)
    root = run_root()
    spec = read_json(root / "run.json")
    folder = root / f"training/{method}_seed{seed}"
    out = root / f"evaluation/{method}_seed{seed}"
    cp = checkpoint(root, spec, method, seed)
    if family == "ant":
        from .ant import train, evaluate
        if action == "train":
            train(root, spec, method, seed, option)
        else:
            evaluate(root, spec, method, seed)
    elif family in BODIES:
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
            from fig2.current.train import main as train
            sys.argv = ["train", "--method", method, "--seed", str(seed), "--device", option]
            train()
        else:
            from .terrain import evaluate
            evaluate(root, spec, method, seed, cp, workers=int(option))
    else:
        raise ValueError(family)


if __name__ == "__main__":
    main()
