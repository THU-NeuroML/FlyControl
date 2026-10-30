"""Fixed-budget CPU evaluation on paired, explicitly recorded initial states."""

from flycontrol.experiments.fig3.registered.common import DATA, PROTOCOL, REPO, ROOT, write_json, configure_process, sha
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, json, multiprocessing as mp, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from flycontrol.experiments.fig3.registered.task import PushTurnTask, ANGLES, OBS_DIM, ACTION_DIM
from flycontrol.experiments.fig3.registered.policy_adapter import build_policy, graph_path

STATE = {}


def initialize(args):
    configure_process()
    STATE.update(args)
    policy = None
    if args["method"] != "cpg":
        cp = torch.load(args["checkpoint"], map_location="cpu", weights_only=False)
        policy = build_policy(
            DATA / "graph.npz",
            OBS_DIM,
            ACTION_DIM,
            args["method"],
            args["seed"],
            "cpu",
            "normalized",
            -1.0,
        )
        policy.load_state_dict(cp["policy_state"])
        policy.eval()
        assert cp["obs_rms_state"] is None, (
            "This protocol uses fixed physical observation scales"
        )
        assert int(cp["timesteps"]) == args["steps"]
        STATE["checkpoint_sha256"] = sha(args["checkpoint"])
    STATE["policy"] = policy
    STATE["env"] = PushTurnTask(
        family=args["family"], objective=args["objective"], fixed_case={}
    )


def episode(spec):
    phase, case = spec
    key = f"phase{phase}_" + ("_".join(f"{k}{v}" for k, v in case.items()))
    path = ROOT / STATE["folder"] / f"{key}.json"
    if path.exists():
        result = json.loads(path.read_text())
        assert result["protocol_sha256"] == STATE["protocol_sha256"]
        assert result.get("checkpoint_sha256") == STATE.get("checkpoint_sha256")
        return result
    env = STATE["env"]
    env.fixed_case = case
    obs, initial = env.reset(seed=phase)
    policy = STATE["policy"]
    hidden = policy.init_state(1, torch.device("cpu")) if policy is not None else None
    trajectory = []
    micro_trajectory = []
    last_action = np.zeros(2)
    variation = 0.0
    for t in range(200):
        if policy is None:
            action = np.zeros(2, dtype=np.float32)
        else:
            with torch.no_grad():
                action, _, _, hidden = policy.act(
                    torch.as_tensor(obs).unsqueeze(0), hidden, deterministic=True
                )
            action = action[0].numpy()
        physical = np.tanh(action)
        variation += float(np.linalg.norm(physical - last_action))
        last_action = physical
        obs, reward, term, trunc, info = env.step(action)
        trajectory.append(
            {
                **{
                    k: info[k]
                    for k in [
                        "elapsed_s",
                        "heading_error_rad",
                        "yaw_rate_rad_s",
                        "body_forward_speed_mm_s",
                        "up_z",
                        "total_rotation_deg",
                    ]
                },
                "action": physical.tolist(),
                "reward": reward,
            }
        )
        for micro in env.last_micro_infos:
            micro_trajectory.append(
                {
                    **{
                        k: micro[k]
                        for k in [
                            "elapsed_s",
                            "heading_error_rad",
                            "yaw_rate_rad_s",
                            "body_forward_speed_mm_s",
                            "up_z",
                            "total_rotation_deg",
                        ]
                    },
                    "action": physical.tolist(),
                    "reward": micro["micro_reward"],
                    "unshaped_reward": micro["unshaped_reward"],
                }
            )
        if term:
            break
    result = dict(
        family=STATE["family"],
        objective=STATE["objective"],
        method=STATE["method"],
        training_seed=STATE["seed"] if policy else None,
        steps=STATE["steps"] if policy else 0,
        phase_seed=phase,
        case=case,
        initial=initial,
        final=info,
        trajectory=trajectory,
        micro_trajectory=micro_trajectory,
        primary_angle_error_deg=180.0
        if info["physical_failure"]
        else info["final_heading_error_deg"],
        primary_push_auc_rad_s=info["postpush_auc_rad_s"],
        action_total_variation=variation,
        checkpoint_sha256=STATE.get("checkpoint_sha256"),
        protocol_sha256=STATE["protocol_sha256"],
        graph_sha256=sha(graph_path(STATE["method"], STATE["seed"]))
        if policy
        else None,
    )
    write_json(path, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["push", "angle"], required=True)
    p.add_argument("--objective", choices=["legacy", "aligned"], default="aligned")
    p.add_argument(
        "--method", choices=["connectome", "mlp", "er", "cpg"], required=True
    )
    p.add_argument("--seed", type=int, choices=range(4301, 4325), default=4301)
    p.add_argument("--steps", type=int, default=8192)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--phase-set", choices=["development", "confirmation", "final"], default="final"
    )
    a = p.parse_args()
    conf = json.loads((PROTOCOL).read_text())
    run = f"{a.family}_{a.objective}_{a.method}_seed{a.seed}"
    index = json.loads((DATA / "registered_models.json").read_text())
    checkpoint = None
    if a.method != "cpg":
        entry = next(
            x
            for x in index
            if (x["family"], x["method"], x["seed"]) == (a.family, a.method, a.seed)
        )
        checkpoint = REPO / entry["path"]
    name = f"{a.family}_cpg" if a.method == "cpg" else f"{run}_step{a.steps}"
    folder = f"evaluation/{a.phase_set}/{name}"
    phases = conf[f"{a.phase_set}_phase_seeds"]
    cases = (
        [
            dict(force=f, side=s, onset=0.75)
            for f, s in [(0.0, 1.0), (5.0, -1.0), (5.0, 1.0)]
        ]
        if a.family == "push"
        else [dict(angle_deg=d, onset=0.3) for d in ANGLES]
    )
    args = {
        **vars(a),
        "folder": folder,
        "checkpoint": str(checkpoint),
        "protocol_sha256": sha(PROTOCOL),
    }
    rows = []
    with ProcessPoolExecutor(
        max_workers=a.workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(args,),
    ) as pool:
        futures = [
            pool.submit(episode, (phase, case)) for phase in phases for case in cases
        ]
        for f in as_completed(futures):
            rows.append(f.result())
            print(f"{len(rows)}/{len(futures)} complete", flush=True)

    def summarize(group):
        return dict(
            episodes=len(group),
            success_rate=float(np.mean([r["final"]["success"] for r in group])),
            physical_failure_rate=float(
                np.mean([r["final"]["physical_failure"] for r in group])
            ),
            mean_angle_error_deg=float(
                np.mean([r["primary_angle_error_deg"] for r in group])
            ),
            mean_push_auc_rad_s=float(
                np.mean([r["primary_push_auc_rad_s"] for r in group])
            ),
            mean_action_total_variation=float(
                np.mean([r["action_total_variation"] for r in group])
            ),
        )

    by_case = []
    for case in cases:
        by_case.append(
            dict(case=case, **summarize([r for r in rows if r["case"] == case]))
        )
    summary = dict(
        **{
            k: args[k]
            for k in [
                "family",
                "objective",
                "method",
                "seed",
                "steps",
                "phase_set",
                "protocol_sha256",
            ]
        },
        **summarize(rows),
        by_case=by_case,
        initial_states={
            str((r["phase_seed"], tuple(r["case"].items()))): r["initial"][
                "initial_state_sha256"
            ]
            for r in rows
        },
        initial_observations={
            str((r["phase_seed"], tuple(r["case"].items()))): r["initial"][
                "initial_observation_sha256"
            ]
            for r in rows
        },
        completed=time.time(),
    )
    write_json(ROOT / folder / "SUMMARY.json", summary)
    write_json(
        ROOT / folder / "COMPLETE.json", dict(episodes=len(rows), completed=time.time())
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["initial_states", "initial_observations"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
