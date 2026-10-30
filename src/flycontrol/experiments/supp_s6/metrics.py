import hashlib
import numpy as np,pandas as pd
TIME=np.arange(1,51)/100
SEED=2026091712
def intervals(model, keys, metrics):
    rows = []
    rng = np.random.default_rng(SEED)
    for name, g in model.groupby(keys, dropna=False, sort=True):
        if not isinstance(name, tuple):
            name = (name,)
        row = dict(zip(keys, name))
        row['n_models'] = len(g)
        ids = rng.integers(0, len(g), (20000, len(g)))
        for metric in metrics:
            a = g[metric].to_numpy(float)
            samples = a[ids].mean(axis=1)
            row[metric + '_mean'] = a.mean()
            row[metric + '_lo'], row[metric + '_hi'] = np.quantile(samples, [.025, .975])
        rows.append(row)
    return pd.DataFrame(rows)
def outcome(r):
    if r['numerical_failure']:
        return 'numerical_failure'
    if r['flipped']:
        return 'flip'
    if not r['survived_window']:
        return 'other_incomplete'
    return 'recovered' if r['success'] else 'heading_only_failure'
def load_raw(r):
    with np.load(r['trace'], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
def metrics(r, a):
    t = a['relative_time_s']
    dt = a['actual_step_duration_s']
    err = np.abs(a['heading_error_rad'])
    assert np.all(np.diff(t) > 0) and np.all(dt > 0)
    assert abs(dt.sum() - r['duration_s']) < 1e-8
    missing = max(0., .5 - dt.sum())
    reconstructed = float(np.dot(err, dt) + np.pi * missing)
    assert abs(reconstructed - r['auc']) < 1e-10
    h = hashlib.sha256()
    for k in sorted(set(a) - {'policy_input', 'recurrent_correction_l1'}):
        h.update(k.encode())
        h.update(np.ascontiguousarray(a[k]).tobytes())
    assert h.hexdigest() == r['trajectory_digest'], r['trace']
    assert bool(r['success']) == bool(r['survived_window'] and err[-1] <= .25)
    assert abs(r['actual_force_max'] - abs(r['signed_force'])) < 1e-8
    assert r['wrong_body_or_torque_max'] == 0
    euler = a['body_euler_rad'].astype(float)
    tilt = np.arccos(np.clip(np.cos(euler[:, 1]) * np.cos(euler[:, 2]), -1, 1))
    arr_action = a['action']
    r.update(amplitude=abs(r['signed_force']), direction=int(np.sign(r['signed_force'])),
             heading_only_failure=bool(r['survived_window'] and not r['success']),
             outcome=outcome(r), tilt_auc=float(np.dot(tilt, dt) + np.pi * missing),
             max_tilt_rad=float(tilt.max()), sustained_success_last100ms=bool(
                 r['survived_window'] and np.all(err[t > .4 + 1e-9] <= .25)),
             raw_auc_error=abs(reconstructed - r['auc']),
             forward_command_mean=float(arr_action[:, 0].mean()),
             turn_command_mean=float(arr_action[:, 1].mean()),
             forward_command_saturation=float(np.mean(np.abs(arr_action[:, 0]) >= .95)),
             turn_command_saturation=float(np.mean(np.abs(arr_action[:, 1]) >= .95)),
             complete_pulse=bool(abs(r['applied_impulse_y'] - r['signed_force'] * .04)
                                 <= abs(r['signed_force']) * .0001 * 1.1 + 1e-12))
    # Endpoint sampling, with explicit absorbing failures; no survivor-only averaging.
    ix = np.clip(np.searchsorted(t, TIME + 1e-8, side='right') - 1, 0, len(t) - 1)
    observed = TIME <= t[-1] + 1e-8
    error_curve = np.where(observed, err[ix], np.pi)
    flip_curve = bool(r['flipped']) & (TIME >= t[-1] - 1e-8)
    alive = ~(flip_curve | ~observed)
    temporal = dict(error_with_missing_penalty=error_curve, flip=flip_curve.astype(float),
                    success_at_time=(alive & (error_curve <= .25)).astype(float),
                    tilt_with_missing_penalty=np.where(observed, tilt[ix], np.pi))
    return r, temporal
