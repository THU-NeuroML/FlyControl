"""Fixed-observation causal probe: locate VNC influence on the chosen readout."""
from pathlib import Path
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from .regional_evaluate import RegionalReplay, regional_weights
from flycontrol.paths import run_root, sha256 as sha, write_json as write
REPO = run_root()
HERE = REPO / 'mechanisms/regional'
OUT = REPO / 'evaluation/regional'
AUDIT = REPO / 'inputs/anatomy'
PYTHON = sys.executable
from larva.locomotion.scripts.eval_same_state_paired import load_saved_policy

MODES = ['brain_reference', 'vnc_reference', 'an_brain_reference',
         'vnc_intrinsic_to_dn_reference', 'vnc_intrinsic_outputs_reference']


@torch.no_grad()
def one(job_file):
    job = json.loads(Path(job_file).read_text())
    torch.set_num_threads(1)
    assert sha(job['checkpoint']) == job['checkpoint_sha256']
    policy, _ = load_saved_policy(Path(job['checkpoint']), job['core_n'])
    core = policy.actor.core
    def read(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    fractions = read(AUDIT / (job['condition'] + '_regional_edge_fractions.npz'))
    masks = read(AUDIT / (job['condition'] + '_annotated_neuron_masks.npz'))
    weights = regional_weights(core, fractions, masks)
    pre, post = fractions['edge_pre'], fractions['edge_post']
    for mode, keep in [
        ('vnc_intrinsic_to_dn_reference', masks['vnc_intrinsic'][pre] & masks['descending'][post]),
        ('vnc_intrinsic_outputs_reference', masks['vnc_intrinsic'][pre])]:
        ids = torch.from_numpy(np.flatnonzero(keep))
        weights[mode] = (ids, torch.ones(len(ids)))
    baseline = read(OUT / 'results' / job['job_id'] / 'baseline_neural.npz')
    phase = Path(job['source_phase_dir'])
    rows, stored = [], {}
    native_zero = read(phase / 'force_+0_intact.npz')
    for force in json.loads((OUT / 'protocol.json').read_text())['signed_forces']:
        native = read(phase / f'force_{force:+g}_intact.npz')
        observations = np.concatenate([baseline['inputs'][:1], native['observation'][:-1]])
        n = len(observations)
        for mode in ['intact'] + MODES:
            hidden = torch.from_numpy(baseline['hidden_before'][:1].copy())
            proxy = RegionalReplay(policy, mode, baseline, weights)
            actions, outputs = [], []
            with proxy.messages():
                for obs in observations:
                    action, _, _, hidden = proxy.act(torch.from_numpy(obs[None]), hidden, deterministic=True)
                    actions.append(action[0].numpy().copy())
                    outputs.append(hidden[0, -core.O:].numpy().copy())
            actions, outputs = np.asarray(actions), np.asarray(outputs)
            action_diff = actions - native['action']
            output_diff = outputs - native['hidden_dn']
            if mode == 'intact' or force == 0:
                assert np.array_equal(actions, native['action']), (mode, force, 'native action replay')
                assert np.array_equal(outputs, native['hidden_dn']), (mode, force, 'native readout replay')
            stored[f'{force:+g}_{mode}_action'] = actions
            stored[f'{force:+g}_{mode}_readout_delta_rms'] = np.sqrt(np.mean(output_diff.astype(float)**2, axis=1))
            times = np.arange(n) * .01
            changed = np.flatnonzero(np.max(np.abs(action_diff), axis=1) > 1e-7)
            changed_output = np.flatnonzero(np.max(np.abs(output_diff), axis=1) > 1e-7)
            for start, stop in [(0., .05), (.05, .15), (.15, .5), (0., .5)]:
                selected = (times >= start - 1e-9) & (times < stop - 1e-9)
                if not selected.any():
                    continue
                response = native['action'] - native_zero['action'][:n]
                denominator = float(np.linalg.norm(response[selected]))
                rows.append(dict(model_id=job['model_id'], condition=job['condition'], force=force,
                    mode=mode, start_s=start, stop_s=stop, actual_samples=int(selected.sum()),
                    full_window_observed=bool(n * .01 >= stop - 1e-8),
                    action_change_rms=float(np.sqrt(np.mean(action_diff[selected].astype(float)**2))),
                    output_change_rms=float(np.sqrt(np.mean(output_diff[selected].astype(float)**2))),
                    action_response_change_norm_ratio=(float(np.linalg.norm(action_diff[selected])) / denominator if denominator > 1e-12 else None),
                    first_action_change_s=float(changed[0] * .01) if len(changed) else None,
                    first_readout_change_s=float(changed_output[0] * .01) if len(changed_output) else None,
                    targeted_edges=len(weights[mode][0]) if mode in weights else 0))
    destination = HERE / 'neural_probe'
    destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / (job['model_id'] + '.csv'), index=False)
    np.savez_compressed(destination / (job['model_id'] + '.npz'), **stored)
    print(job['model_id'], 'fixed-observation native and sham replay bit-exact', flush=True)
