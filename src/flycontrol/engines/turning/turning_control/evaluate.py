"""Original fixed-budget CPU evaluation; all six angles and optional zero extension."""

import json, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from .context import DATA, PROTOCOL, write_json, configure_process, sha
from .task import PushTurnTask, ANGLES, OBS_DIM, ACTION_DIM
from .policy import build_policy, graph_path

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
        assert (
            cp["obs_rms_state"] is None
        ), "This protocol uses fixed physical observation scales"
        assert int(cp["timesteps"]) == args["steps"]
        STATE["checkpoint_sha256"] = sha(args["checkpoint"])
    STATE["policy"] = policy
    STATE["env"] = PushTurnTask(
        family=args["family"], objective=args["objective"], fixed_case={}
    )


def episode(spec):
    phase, case = spec
    key = f"phase{phase}_" + ("_".join(f"{k}{v}" for k, v in case.items()))
    path = Path(STATE["folder"]) / f"{key}.json"
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
        primary_angle_error_deg=(
            180.0 if info["physical_failure"] else info["final_heading_error_deg"]
        ),
        primary_push_auc_rad_s=info["postpush_auc_rad_s"],
        action_total_variation=variation,
        checkpoint_sha256=STATE.get("checkpoint_sha256"),
        protocol_sha256=STATE["protocol_sha256"],
        graph_sha256=(
            sha(graph_path(STATE["method"], STATE["seed"])) if policy else None
        ),
    )
    write_json(path, result)
    return result


def evaluate(method, seed, output, checkpoint, workers=4, zero=False):
    output.mkdir(parents=True, exist_ok=False)
    conf = json.loads(PROTOCOL.read_text())
    cases = [dict(angle_deg=d, onset=0.3) for d in ([0.0] if zero else ANGLES)]
    args = dict(
        method=method,
        seed=seed,
        steps=conf["ppo"]["total_timesteps"],
        family="angle",
        objective="aligned",
        folder=str(output),
        checkpoint=str(checkpoint),
        protocol_sha256=sha(PROTOCOL),
    )
    jobs = [(phase, case) for phase in conf["final_phase_seeds"] for case in cases]
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(args,),
    ) as pool:
        for future in as_completed([pool.submit(episode, job) for job in jobs]):
            rows.append(future.result())
    write_json(
        output / "COMPLETE.json",
        dict(
            episodes=len(rows), checkpoint_sha256=sha(checkpoint), completed=time.time()
        ),
    )
