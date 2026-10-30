"""Original reversible edge cuts, matched initial states and intact replay checks."""

from pathlib import Path
import json, time, hashlib
import numpy as np
import pandas as pd
import torch
from fig6.paths import sha256 as sha
from fig3.paths import DATA, MECHANISMS, checkpoint as cp_path
from turning_control.policy import build_policy

S, I, D = 18756, 118538, 1290
N = S + I + D
STARTS = [8, 20, 32]
NAMES = [
    f"input_{m}_{a:+d}"
    for m in ["connectome", "mlp"]
    for a in [-90, -60, -30, 30, 60, 90]
]
from turning_control.context import configure_process
from turning_control.task import PushTurnTask

MODES = [
    "intact",
    "AN_off",
    "AN_half",
    "AN_restore",
    "matched_off",
    "mech_off",
    "ocellar_off",
    "AN_DN_cut",
    "AN_I_cut",
    "AN_DN_matchcut",
    "steering_off",
]
CONFIGURED = False


class Env(PushTurnTask):
    def _apply_external_perturbation(self, t):
        super()._apply_external_perturbation(t)
        if self._sim is not None:
            self.impulse += (
                float(self._sim.physics.data.xfrc_applied[:, 1].sum()) * self.timestep
            )


def digest(env, h, obs):
    return hashlib.sha256(
        b"".join(
            np.ascontiguousarray(a).tobytes()
            for a in [
                env._sim.physics.data.qpos,
                env._sim.physics.data.qvel,
                h.cpu().numpy(),
                obs,
            ]
        )
    ).hexdigest()


def auc(micro, start, stop, end):
    out = 0.0
    last = 0.0
    for r in micro:
        out += max(0.0, min(stop, r["elapsed_s"]) - max(start, last)) * abs(
            r["heading_error_rad"]
        )
        last = r["elapsed_s"]
    return out + np.pi * max(0.0, stop - max(start, end))


def worker(spec, output):
    global CONFIGURED
    if not CONFIGURED:
        configure_process()
        CONFIGURED = True
    seed, phase, angle = spec
    begin = time.time()
    dest = Path(output) / f"seed{seed}_phase{phase}_angle{angle:+d}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETE.json").exists():
        return str(dest)
    cp = cp_path("connectome", seed)
    before = sha(cp)
    p = build_policy(
        DATA / "graph.npz", 64, 2, "connectome", seed, "cpu", "normalized", -1.0
    )
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    p.load_state_dict(payload["policy_state"])
    p.eval()
    for par in p.parameters():
        par.requires_grad_(False)
    core = p.actor.core
    fg = core.fixed_graph
    original_f = fg.f_weight.clone()
    original_b = fg.b_weight.clone()
    pre = core.edge_pre.numpy()
    post = core.edge_post.numpy()
    ff = np.argsort(post, kind="stable")
    bb = np.argsort(pre, kind="stable")
    invf = np.argsort(ff)
    invb = np.argsort(bb)
    assert torch.equal(original_f, core.edge_weight[ff])
    assert torch.equal(original_b, core.edge_weight[bb])
    edges = dict(np.load(MECHANISMS / "interventions/source_data/edges.npz"))
    cells = dict(np.load(MECHANISMS / "interventions/source_data/cells.npz"))
    selected = np.unique(
        np.concatenate(
            [
                cells[k]
                for k in [
                    "steering",
                    "AN_GNG_172",
                    "AN_GNG_175",
                    "AN_GNG_196",
                    "AN_multi_1",
                    "AN_top50_relays",
                ]
            ]
        )
    )
    dnidx = torch.as_tensor(cells["steering"][cells["steering"] >= N - D] - (N - D))
    active = [False]
    mode_state = ["intact"]

    def readout_hook(module, args):
        if active[0] and mode_state[0] == "steering_off":
            r = args[0].clone()
            r[:, dnidx] = 0.0
            return (r,)
        return args

    handle = core.output_layer[0].register_forward_pre_hook(readout_hook)
    env = Env(
        family="angle",
        objective="aligned",
        fixed_case=dict(angle_deg=float(angle), onset=0.3),
    )
    env.impulse = 0.0
    rows = []
    traces = []
    initial_hash = None
    prefix_hash = None
    baseline_inputs = {}
    max_sham_error = 0.0
    edgeqa = []
    signed_force = 5.0 * np.sign(angle) * (1 if phase == 790101 else -1)
    for mode in MODES:
        for force in [0.0, signed_force]:
            fg.f_weight.copy_(original_f)
            fg.b_weight.copy_(original_b)
            active[0] = False
            mode_state[0] = mode
            obs, initial = env.reset(seed=phase)
            env.impulse = 0.0
            env.perturbation = dict(
                type="side_push",
                force=abs(force),
                start_time_s=0.75,
                duration_s=0.04,
                direction="lateral",
            )
            env._perturbation_side = float(np.sign(force)) if force else 1.0
            if initial_hash is None:
                initial_hash = initial["initial_state_sha256"]
            assert initial_hash == initial["initial_state_sha256"]
            h = p.init_state(1, "cpu")
            micro = []
            observations = []
            actions = []
            rates = []
            phases = []
            prefix = None
            applied = False
            restored = False
            cut = edges[mode]
            fidx = torch.as_tensor(invf[cut])
            bidx = torch.as_tensor(invb[cut])
            scale = 0.5 if mode == "AN_half" else 0.0
            for t in range(50):
                if t == 8:
                    prefix = digest(env, h, obs)
                    if prefix_hash is None:
                        prefix_hash = prefix
                    assert prefix == prefix_hash, (
                        seed,
                        phase,
                        angle,
                        mode,
                        force,
                        "prefix",
                    )
                    fg.f_weight[fidx] = original_f[fidx] * scale
                    fg.b_weight[bidx] = original_b[bidx] * scale
                    changed = torch.nonzero(fg.f_weight != original_f, as_tuple=True)[
                        0
                    ].numpy()
                    assert np.array_equal(np.sort(changed), np.sort(invf[cut]))
                    active[0] = True
                    applied = True
                    edgeqa.append(
                        dict(
                            mode=mode,
                            force=force,
                            changed_edges=len(changed),
                            selected_edges=len(cut),
                        )
                    )
                if t == 25 and mode == "AN_restore":
                    fg.f_weight.copy_(original_f)
                    fg.b_weight.copy_(original_b)
                    active[0] = False
                    restored = True
                observations.append(obs.copy())
                phases.append(env._controller.cpg.curr_phases.copy())
                with torch.no_grad():
                    a, _, _, h = p.act(
                        torch.as_tensor(obs).unsqueeze(0), h, deterministic=True
                    )
                actions.append(a[0].numpy().copy())
                rates.append(h[0, selected].tanh().numpy().copy())
                obs, _, term, trunc, info = env.step(a[0].numpy())
                for v in env.last_micro_infos:
                    r = {
                        k: v[k]
                        for k in [
                            "elapsed_s",
                            "heading_error_rad",
                            "up_z",
                            "yaw_rate_rad_s",
                            "body_forward_speed_mm_s",
                            "physical_failure",
                        ]
                    }
                    r.update(
                        seed=seed,
                        phase=phase,
                        angle=angle,
                        mode=mode,
                        force=force,
                        drive=float(np.tanh(a[0, 0].item())),
                        turn=float(np.tanh(a[0, 1].item())),
                    )
                    micro.append(r)
                if term or trunc:
                    break
            end = info["elapsed_s"]
            expected = force * max(0.0, min(end, 0.79) - 0.75)
            assert abs(env.impulse - expected) < 1e-6, (env.impulse, expected)
            assert prefix is not None and applied
            if mode == "AN_restore" and end > 1.0:
                assert restored
            if mode == "intact":
                baseline_inputs[force] = (np.asarray(observations), np.asarray(actions))
            # Prefix state matching is stronger than matching visible trajectories.
            z = dict(
                seed=seed,
                phase=phase,
                angle=angle,
                mode=mode,
                force=force,
                auc_rad_s=auc(micro, 0.75, 1.5, end),
                command_auc_rad_s=auc(micro, 0.32, 2.0, end),
                late_auc_rad_s=auc(micro, 1.0, 2.0, end),
                terminal_error_deg=(
                    180.0
                    if info["physical_failure"]
                    else info["final_heading_error_deg"]
                ),
                physical_failure=info["physical_failure"],
                success=info["success"],
                duration_s=end,
                impulse=env.impulse,
                expected_impulse=expected,
                initial_hash=initial_hash,
                preintervention_hash=prefix,
                restored=restored,
                min_up_z=min(v["up_z"] for v in micro),
                mean_forward_speed=float(
                    np.mean(
                        [
                            v["body_forward_speed_mm_s"]
                            for v in micro
                            if v["elapsed_s"] >= 0.32
                        ]
                    )
                ),
            )
            rows.append(z)
            traces.extend(micro)
            np.savez_compressed(
                dest / f"{mode}_{force:+.0f}.npz",
                observations=np.asarray(observations),
                actions=np.asarray(actions),
                selected_state_indices=selected,
                rates=np.asarray(rates),
                cpg_phases=np.asarray(phases),
            )
            pd.DataFrame(rows).to_csv(dest / "episodes.csv", index=False)
            print(
                "EPISODE",
                seed,
                phase,
                angle,
                mode,
                force,
                round(time.time() - begin, 1),
                flush=True,
            )
    fg.f_weight.copy_(original_f)
    fg.b_weight.copy_(original_b)
    active[0] = False
    # Native intact replay after all mutations verifies no residual graph/readout change.
    h = p.init_state(1, "cpu")
    xx, aa = baseline_inputs[0.0]
    for obs, a in zip(xx, aa):
        with torch.no_grad():
            b, _, _, h = p.act(torch.as_tensor(obs).unsqueeze(0), h, deterministic=True)
        max_sham_error = max(max_sham_error, float(np.max(abs(b[0].numpy() - a))))
    assert max_sham_error == 0.0, max_sham_error
    handle.remove()
    env.close()
    pd.DataFrame(traces).to_csv(dest / "micro_trajectories.csv.gz", index=False)
    pd.DataFrame(edgeqa).to_csv(dest / "intervention_qa.csv", index=False)
    assert sha(cp) == before
    (dest / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                phase=phase,
                angle=angle,
                episodes=len(rows),
                checkpoint_sha256=before,
                script_sha256=sha(__file__),
                seconds=time.time() - begin,
                all_initial_and_preintervention_hashes_equal=True,
                max_intact_replay_after_restoration_error=max_sham_error,
                max_impulse_error=max(
                    abs(v["impulse"] - v["expected_impulse"]) for v in rows
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    return str(dest)
