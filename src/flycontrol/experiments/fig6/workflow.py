"""Figure 6 body training and evaluation through isolated adapters."""
import sys
from flycontrol.paths import read_json, write_json


def body_train(root, spec, method, seed, device):
    conf = read_json(root / "protocol.json")
    sys.argv = ["train", "--root", str(root / "training"), "--body", spec["experiment"],
                "--method", method, "--seed", str(seed), "--readout", conf["readout"], "--device", device]
    from fly_control.training import main
    main()


def body_evaluate(root, spec, method, seed):
    import subprocess
    from flycontrol.runtime import environment
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
