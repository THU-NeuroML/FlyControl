"""Same-state force and acute-recurrence counterfactuals for saved EXP03 models."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from larva.locomotion.connectome import ConnectomeData
from larva.locomotion.env import TerrainLocomotionEnv
from larva.locomotion.models import build_policy
from larva.locomotion.paired_state import PairedSnapshot
from larva.locomotion.scripts.eval_command2_policy import make_eval_env_config


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    os.replace(temporary, path)


def load_saved_policy(path, n_total):
    """Use checkpoint edge buffers verbatim; never regenerate a random graph.

    n_total is read from the archived evaluator's core_n. S and O are checked
    against saved projections. Root IDs are deliberately absent: this loader
    supports numerical evaluation, not anatomical labeling.
    """
    c = torch.load(path, map_location='cpu', weights_only=False)
    cfg = c['config']
    assert cfg['policy']['name'] == 'connectome_cfc'
    assert cfg['policy']['actor_kwargs']['mode'] == 'rate'
    assert cfg['policy']['actor_kwargs']['input_target'] == 'sensory'
    state = c['policy_state']
    s = state['actor.core.input_proj.weight'].shape[0]
    o = state['actor.core.output_layer.weight'].shape[1]
    arrays = [state['actor.core.' + name].numpy() for name in ['edge_pre', 'edge_post', 'edge_weight']]
    assert max(arrays[0].max(), arrays[1].max()) < n_total
    graph = ConnectomeData(None, s, n_total-s-o, o, (), (), (), *arrays,
                           metadata={'source': 'saved_checkpoint_edges', 'root_ids_available': False})
    policy = build_policy(cfg['policy']['name'], c['obs_dim'], c['action_dim'],
                          actor_kwargs=cfg['policy'].get('actor_kwargs'),
                          value_hidden=cfg['policy'].get('value_hidden'),
                          log_std_init=cfg['policy'].get('log_std_init', -.5), connectome=graph)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    for k, v in policy.state_dict().items():
        assert torch.equal(v, state[k]), k
    return policy, cfg


class AuditedHeadingEnv(TerrainLocomotionEnv):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._audit_body_id = int(self._sim.physics.bind(self._fly.thorax).element_id)
        self._audit_impulse_y = 0.
        self._audit_nonzero_steps = 0
        self._audit_force_max = 0.
        self._audit_wrong_body_or_torque_max = 0.
        return obs, info

    def _apply_external_perturbation(self, elapsed_s):
        super()._apply_external_perturbation(elapsed_s)
        forces = self._sim.physics.data.xfrc_applied
        applied = forces[self._audit_body_id]
        self._audit_impulse_y += float(applied[1]) * self.timestep
        if applied[1] != 0:
            self._audit_nonzero_steps += 1
            self._audit_force_max = max(self._audit_force_max, abs(float(applied[1])))
            other = forces.copy()
            other[self._audit_body_id, 1] = 0
            self._audit_wrong_body_or_torque_max = max(self._audit_wrong_body_or_torque_max, float(np.abs(other).max()))


@torch.no_grad()
def decision(policy, obs, hidden):
    action, _, _, hidden = policy.act(torch.as_tensor(obs[None], dtype=torch.float32), hidden, deterministic=True)
    return action[0].numpy().astype(np.float32), hidden


def branch(env, policy, snapshot, signed_force, mode, window_s, duration_s):
    obs, hidden = snapshot.restore(env)
    start = env._physics_steps * env.timestep
    env.perturbation = {'type': 'side_push', 'force': abs(float(signed_force)), 'direction': 'lateral',
                        'start_time_s': start, 'duration_s': duration_s}
    env._perturbation_side = float(np.sign(signed_force)) if signed_force else 1.
    initial_y = float(env._last_raw_obs['fly'][0, 1])
    records = {k: [] for k in ['relative_time_s', 'observation', 'action', 'heading_error_rad', 'body_euler_rad',
                              'body_position_mm', 'end_effectors_mm', 'contact_forces', 'cpg_phase_rad',
                              'hidden_dn', 'instantaneous_inversion', 'actual_step_duration_s']}
    auc = valid = 0.
    flipped = numeric = terminated = truncated = False
    t0 = time.monotonic()
    with policy.actor.core.intervention(recurrent_gain=0. if mode == 'acute_recurrent_off' else 1.):
        while env._physics_steps * env.timestep - start < window_s - 1e-10:
            before = env._physics_steps * env.timestep
            action, hidden = decision(policy, obs, hidden)
            obs, reward, terminated, truncated, info = env.step(action)
            after = env._physics_steps * env.timestep
            dt = min(after-start, window_s)-min(before-start, window_s)
            raw = env._last_raw_obs
            heading = env._raw_heading_angle(raw)
            err = float(np.arctan2(np.sin(heading-env._target_heading_rad), np.cos(heading-env._target_heading_rad)))
            numeric = bool(info.get('physics_error', False) or info.get('nan_detected', False))
            finite = np.isfinite(raw['fly']).all() and np.isfinite(err)
            if finite and dt > 0:
                auc += abs(err) * dt
                valid += dt
            flipped |= bool(info.get('flip', False))
            sample = {'relative_time_s': after-start, 'observation': obs.copy(), 'action': action,
                      'heading_error_rad': err, 'body_euler_rad': raw['fly'][2].copy(),
                      'body_position_mm': raw['fly'][0].copy(), 'end_effectors_mm': raw['end_effectors'].copy(),
                      'contact_forces': raw['contact_forces'].copy(), 'cpg_phase_rad': env._controller.cpg.curr_phases.copy(),
                      'hidden_dn': hidden[0, -policy.actor.core.O:].numpy().copy(),
                      'instantaneous_inversion': bool(info.get('instantaneous_inversion', False)),
                      'actual_step_duration_s': dt}
            for k, v in sample.items():
                records[k].append(v)
            if terminated or truncated or numeric or not finite:
                break
    arrays = {k: np.asarray(v) for k, v in records.items()}
    missing = max(0., window_s-valid)
    complete = missing <= 1e-8 and not numeric
    final_error = float(abs(arrays['heading_error_rad'][-1]))
    summary = dict(signed_force=float(signed_force), intervention=mode,
                   onset_s=start, window_s=window_s, duration_s=valid,
                   auc_observed=auc, auc_missing_penalty=np.pi*missing, auc=auc+np.pi*missing,
                   missing_duration_s=missing, flipped=flipped, numerical_failure=numeric,
                   terminated=bool(terminated), truncated=bool(truncated), survived_window=bool(complete and not flipped),
                   success=bool(complete and not flipped and final_error <= .25), final_heading_error_abs_rad=final_error,
                   lateral_displacement_mm=float(arrays['body_position_mm'][-1, 1]-initial_y),
                   # FlyGym stores the ZYX Euler array in yaw, pitch, roll order.
                   max_abs_roll_rad=float(np.max(np.abs(arrays['body_euler_rad'][:, 2]))),
                   max_abs_pitch_rad=float(np.max(np.abs(arrays['body_euler_rad'][:, 1]))),
                   action_saturation_fraction=float(np.mean(np.abs(arrays['action']) >= .95)),
                   action_temporal_std=float(np.std(arrays['action'], axis=0).mean()),
                   applied_impulse_y=float(env._audit_impulse_y), force_active_physics_steps=env._audit_nonzero_steps,
                   actual_force_max=env._audit_force_max, wrong_body_or_torque_max=env._audit_wrong_body_or_torque_max,
                   wall_seconds=time.monotonic()-t0, prebranch_digest=snapshot.digest)
    assert summary['wrong_body_or_torque_max'] == 0
    assert np.isclose(summary['actual_force_max'], abs(signed_force))
    if signed_force == 0:
        assert summary['applied_impulse_y'] == 0
    elif valid >= duration_s:
        assert abs(summary['applied_impulse_y']-signed_force*duration_s) <= abs(signed_force)*env.timestep*1.1
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode()); h.update(np.ascontiguousarray(arrays[k]).tobytes())
    summary['trajectory_digest'] = h.hexdigest()
    return summary, arrays


def evaluate_job(job, protocol, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for source, expected in protocol.get('source_sha256', {}).items():
        if sha256(source) != expected:
            raise ValueError(f'frozen evaluator source changed: {source}')
    complete_path = output / 'complete.json'
    if complete_path.exists():
        old = json.loads(complete_path.read_text())
        if old['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('refusing to mix protocols')
        return old
    assert sha256(job['checkpoint']) == job['checkpoint_sha256']
    torch.set_num_threads(1)
    torch.manual_seed(20260917)
    policy, cfg = load_saved_policy(Path(job['checkpoint']), int(job['core_n']))
    env_cfg = make_eval_env_config(cfg, 'heading_hold', 'flat', job['eval_seed'], protocol['max_time_s'])
    env_cfg['perturbation'] = {'type': 'side_push', 'force': 0., 'direction': 'lateral',
                               'start_time_s': 10., 'duration_s': protocol['push_duration_s']}
    env = AuditedHeadingEnv(**env_cfg)
    summaries, qc, unavailable = [], [], []
    try:
        obs, _ = env.reset(seed=job['eval_seed'])
        hidden = torch.zeros(1, policy.state_dim)
        snapshots = {}
        target_bins = set(protocol['phase_bins'])
        done = False
        while env._physics_steps * env.timestep <= protocol['last_snapshot_s'] + 1e-10:
            elapsed = env._physics_steps * env.timestep
            phase = float(env._controller.cpg.curr_phases[0] % (2*np.pi))
            phase_bin = int(phase / (2*np.pi/3))
            if elapsed >= protocol['first_snapshot_s'] - 1e-10 and phase_bin in target_bins and phase_bin not in snapshots:
                snapshots[phase_bin] = (phase, PairedSnapshot.capture(env, obs, hidden))
            if len(snapshots) == len(target_bins):
                break
            action, hidden = decision(policy, obs, hidden)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                done = True
                break
        for phase_bin in sorted(target_bins - set(snapshots)):
            unavailable.append(dict(phase_bin=phase_bin, reason='prebranch_termination' if done else 'phase_not_reached',
                                    elapsed_s=env._physics_steps*env.timestep))
        for phase_bin, (phase, snap) in sorted(snapshots.items()):
            phase_dir = output / f'phase_{phase_bin}'
            phase_dir.mkdir(exist_ok=True)
            with gzip.open(phase_dir / 'snapshot.pkl.gz', 'wb', compresslevel=1) as handle:
                pickle.dump(snap.archive(), handle, protocol=5)
            local = []
            combinations = [(f, m) for f in protocol['signed_forces'] for m in protocol['interventions']]
            # Order is outcome independent and unique per job/phase.
            order = np.random.default_rng(np.random.SeedSequence([20260917, job['eval_seed'], job['job_index'], phase_bin])).permutation(len(combinations))
            for i in order:
                force, mode = combinations[int(i)]
                summary, arrays = branch(env, policy, snap, force, mode, protocol['window_s'], protocol['push_duration_s'])
                name = f'force_{force:+g}_{mode}'
                np.savez_compressed(phase_dir / (name+'.npz'), **arrays)
                summary.update({**job, 'phase_bin': phase_bin, 'prepush_lf_phase_rad': phase,
                                'trace': str((phase_dir/(name+'.npz')).resolve()), 'protocol_sha256': protocol['protocol_sha256']})
                local.append(summary)
                print(json.dumps({'job': job['job_id'], 'phase': phase_bin, 'force': force,
                                  'mode': mode, 'wall_seconds': summary['wall_seconds']}), flush=True)
            # Same branch after all other interventions must match bit for bit.
            reference = next(a for a in local if a['signed_force'] == 0 and a['intervention'] == 'intact')
            repeated, _ = branch(env, policy, snap, 0, 'intact', protocol['window_s'], protocol['push_duration_s'])
            identical = repeated['trajectory_digest'] == reference['trajectory_digest']
            qc.append(dict(phase_bin=phase_bin, repeated_zero_branch_exact=identical,
                           prebranch_digest=snap.digest, reference_digest=reference['trajectory_digest'],
                           repeat_digest=repeated['trajectory_digest']))
            atomic_json(phase_dir/'replay_qc.json', qc[-1])
            if not identical:
                raise RuntimeError('repeated branch differs after snapshot restoration')
            summaries.extend(local)
            atomic_json(output/'episodes.json', summaries)
            print(json.dumps({'job': job['job_id'], 'phase': phase_bin, 'branches': len(local), 'replay_exact': identical}), flush=True)
        result = dict(job=job, protocol_sha256=protocol['protocol_sha256'], episodes=len(summaries),
                      eligible_snapshots=len(snapshots), unavailable=unavailable, replay_qc=qc)
        atomic_json(complete_path, result)
        return result
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    job = json.loads(args.job.read_text())
    protocol = json.loads(args.protocol.read_text())
    evaluate_job(job, protocol, args.output)


if __name__ == '__main__':
    main()
