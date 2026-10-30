"""Anatomically weighted message-reference interventions from freshly simulated states."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import json
import numpy as np
import torch

from larva.locomotion.scripts.eval_same_state_paired import (
    load_saved_policy, AuditedHeadingEnv, branch, atomic_json, sha256)
from larva.locomotion.scripts.eval_feedback_pathways import (
    import_archive, ReplayPolicy, extended_metrics)
from larva.locomotion.scripts.eval_command2_policy import make_eval_env_config


def regional_weights(core, fractions, masks):
    for k in ['edge_pre', 'edge_post', 'edge_weight']:
        assert np.array_equal(getattr(core, k).numpy(), fractions[k]), k
    b, v = fractions['brain_fraction'], fractions['vnc_fraction']
    assert np.all(b >= 0) and np.all(v >= 0)
    assert np.max(np.abs(b + v + fractions['neck_fraction'] - 1)) < 2e-7
    pre = fractions['edge_pre']
    f = {'brain_reference': b, 'vnc_reference': v,
         'brain_vnc_reference': b + v,
         'dn_vnc_reference': v * masks['descending'][pre],
         'an_brain_reference': b * masks['ascending'][pre]}
    return {name: (torch.from_numpy(np.flatnonzero(a).copy()),
                   torch.from_numpy(a[a != 0].copy())) for name, a in f.items()}


class RegionalReplay(ReplayPolicy):
    def __init__(self, policy, mode, baseline, weights):
        super().__init__(policy, mode, baseline)
        self.weights = weights
        self.correction_l1 = []

    @contextmanager
    def messages(self):
        if self.mode not in self.weights:
            yield
            return
        core = self.actor.core
        ids, fraction = self.weights[self.mode]
        ep, eo = core.edge_pre[ids], core.edge_post[ids]
        ew = core.edge_weight[ids] * fraction
        self.message_count = len(ids)
        self.fractional_abs_weight_sum = float(ew.abs().sum())
        original = core._recurrent_drive
        assert '_recurrent_drive' not in core.__dict__

        def replay_drive(r):
            drive = original(r)
            ref = torch.tanh(torch.from_numpy(self.baseline['hidden_before'][self.index]))
            difference = (ref[ep].unsqueeze(0) - r[:, ep]) * ew.unsqueeze(0)
            correction = torch.zeros_like(drive).index_add_(1, eo, difference)
            self.correction_l1.append(float(correction.abs().sum()))
            return drive + correction

        core._recurrent_drive = replay_drive
        try:
            yield
        finally:
            del core._recurrent_drive


def evaluate(job, protocol, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for path, h in protocol['source_sha256'].items():
        assert sha256(path) == h, ('source changed', path)
    if (output / 'complete.json').exists():
        assert json.loads((output / 'complete.json').read_text())['protocol_sha256'] == protocol['protocol_sha256']
        return
    torch.set_num_threads(1)
    torch.manual_seed(20260917)
    assert sha256(job['checkpoint']) == job['checkpoint_sha256']
    phase_dir = Path(job['source_phase_dir'])
    assert sha256(phase_dir / 'snapshot.pkl.gz') == job['snapshot_sha256']
    policy, cfg = load_saved_policy(Path(job['checkpoint']), int(job['core_n']))
    audit = Path(protocol['anatomy_directory'])
    with np.load(audit / (job['condition'] + '_regional_edge_fractions.npz')) as z:
        fractions = {k: z[k] for k in z.files}
    with np.load(audit / (job['condition'] + '_annotated_neuron_masks.npz')) as z:
        masks = {k: z[k] for k in z.files}
    weights = regional_weights(policy.actor.core, fractions, masks)
    ec = make_eval_env_config(cfg, 'heading_hold', 'flat', job['eval_seed'], 1.7)
    ec['perturbation'] = {'type': 'side_push', 'force': 0., 'direction': 'lateral',
                          'start_time_s': 10., 'duration_s': .04}
    env = AuditedHeadingEnv(**ec)
    try:
        env.reset(seed=job['eval_seed'])
        snap, old_digest = import_archive(env, phase_dir / 'snapshot.pkl.gz')
        reference_qc = json.loads((phase_dir / 'replay_qc.json').read_text())
        collector = ReplayPolicy(policy)
        base, base_arrays = branch(env, collector, snap, 0, 'intact', .5, .04)
        assert base['trajectory_digest'] == reference_qc['reference_digest'], 'archive replay differs'
        assert base['survived_window'], 'full no-force reference unavailable'
        baseline = {'inputs': np.asarray(collector.inputs), 'hidden_before': np.asarray(collector.hidden_before)}
        assert len(baseline['inputs']) == 50
        np.savez_compressed(output / 'baseline_neural.npz', **baseline)
        old_rows = json.loads((phase_dir.parent / 'episodes.json').read_text())
        rows = []
        for f in protocol['signed_forces']:
            old = next(r for r in old_rows if r['phase_bin'] == job['phase_bin']
                       and r['signed_force'] == f and r['intervention'] == 'intact')
            with np.load(old['trace']) as z:
                arr = {k: z[k] for k in z.files}
            row = extended_metrics(dict(old), arr)
            row.update(**job, protocol_sha256=protocol['protocol_sha256'],
                       source_protocol_sha256=old['protocol_sha256'], archive_digest=old_digest,
                       prebranch_digest=snap.digest, reused_intact_reference=True)
            rows.append(row)
        combinations = [(f, m) for f in protocol['signed_forces'] for m in protocol['modes']]
        order = np.random.default_rng(np.random.SeedSequence([2026091711, job['job_index']])).permutation(len(combinations))
        zero_exact = first_exact = 0
        for i in order:
            f, mode = combinations[int(i)]
            proxy = RegionalReplay(policy, mode, baseline, weights)
            with proxy.messages():
                summary, arrays = branch(env, proxy, snap, f, 'intact', .5, .04)
            assert np.array_equal(arrays['action'][0], base_arrays['action'][0]), (mode, 'onset action changed')
            first_exact += 1
            if f == 0:
                assert summary['trajectory_digest'] == base['trajectory_digest'], (mode, 'zero-force sham differs')
                assert not any(proxy.correction_l1)
                zero_exact += 1
            arrays['policy_input'] = np.asarray(proxy.inputs)
            arrays['recurrent_correction_l1'] = np.asarray(proxy.correction_l1)
            path = output / f'force_{f:+g}_{mode}.npz'
            np.savez_compressed(path, **arrays)
            summary = extended_metrics(summary, arrays)
            summary.update(**job, intervention=mode, trace=str(path.resolve()),
                protocol_sha256=protocol['protocol_sha256'], archive_digest=old_digest,
                reused_intact_reference=False, replayed_edge_count=getattr(proxy, 'message_count', 0),
                fractional_abs_weight_sum=getattr(proxy, 'fractional_abs_weight_sum', 0.))
            rows.append(summary)
            atomic_json(output / 'episodes.json', rows)
            print(json.dumps({'job': job['job_id'], 'force': f, 'mode': mode, 'wall_s': summary['wall_seconds']}), flush=True)
        repeated, _ = branch(env, policy, snap, 0, 'intact', .5, .04)
        assert repeated['trajectory_digest'] == base['trajectory_digest'], 'intervention leaked'
        atomic_json(output / 'complete.json', dict(protocol_sha256=protocol['protocol_sha256'],
            episodes=len(rows), archived_reference_exact=True, zero_replay_exact=zero_exact,
            first_action_exact=first_exact, final_reference_exact=True,
            source_archive_digest=old_digest, local_digest=snap.digest))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['job', 'protocol', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    evaluate(json.loads(a.job.read_text()), json.loads(a.protocol.read_text()), a.output)
