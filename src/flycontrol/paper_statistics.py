"""Prespecified behavioral comparison families, separate from diagnostics."""

import numpy as np
import pandas as pd

from .paths import CONFIGS, read_json, write_json
from .analysis import signflip, holm

TERRAIN_METRICS = [
    'success', 'crossed_within_corridor', 'completion_time_with_failure_horizon_s',
    'max_lateral_mm', 'physical_failure', 'numerical_failure', 'final_last_foot_x_mm',
    'final_body_x_mm', 'final_up_z', 'minimum_up_z', 'final_stable_dwell_s',
    'residual_variation_per_observed_second', 'residual_max_step_change_mean6_l1', 'action_rms',
]


def behavior(root, spec, episodes):
    family = spec['family']
    if family not in {'terrain', 'speed'}:
        return
    planned = read_json(CONFIGS / (spec['experiment'] + '.json'))
    if spec['smoke'] or set(spec['seeds']) != set(planned['seeds']) or set(spec['methods']) != {'connectome','mlp'}:
        write_json(root / 'analysis/paper_statistics.json', dict(computed=False,
            reason='The complete fixed training-seed cohort and both methods are required.'))
        return
    metrics = TERRAIN_METRICS if family == 'terrain' else read_json(root / 'configs/figure4/analysis.json')['statistical_family']
    episodes = episodes.copy()
    for metric in metrics:
        if metric not in episodes:
            episodes[metric] = np.nan
    seeds = sorted(spec['seeds'])
    means = episodes.groupby(['method','seed','condition'])[metrics].mean()
    rows = []
    draw = np.random.default_rng(20261021 if family == 'terrain' else 20261020).integers(
        0, len(seeds), (50000 if family == 'terrain' else 20000, len(seeds)))
    for metric in metrics:
        for condition in sorted(episodes.condition.unique()):
            pair = means.xs(condition, level='condition')[metric].unstack('method').reindex(seeds)
            a, b = pair.connectome.to_numpy(), pair.mlp.to_numpy()
            delta = a-b if family=='terrain' else b-a
            valid = bool(np.isfinite(delta).all())
            lo,hi = np.quantile(delta[draw].mean(1), [.025,.975]) if valid else (np.nan,np.nan)
            rows.append(dict(metric=metric, condition=condition, n_planned_pairs=len(seeds),
                n_observed_pairs=int(np.isfinite(delta).sum()), adult_mean=float(np.mean(a)), mlp_mean=float(np.mean(b)),
                difference=float(np.mean(delta)), ci95_low=lo, ci95_high=hi,
                direction='Adult minus MLP' if family=='terrain' else 'MLP minus Adult',
                p_raw=signflip(delta) if valid else np.nan))
    table = pd.DataFrame(rows)
    if family == 'terrain':
        table['p_holm_within_metric'] = np.nan
        for _, group in table.groupby('metric'):
            table.loc[group.index, 'p_holm_within_metric'] = holm(group.p_raw.fillna(1).to_numpy())
        secondary = table.metric != 'success'
        table.loc[secondary, 'p_holm_65_secondary'] = holm(table.loc[secondary,'p_raw'].fillna(1).to_numpy())
        table.loc[table.p_raw.isna(), ['p_holm_within_metric','p_holm_65_secondary']] = np.nan
        correction = 'Success: 5 heights; secondary: all 13 metrics × 5 heights. Non-estimable slots retained.'
    else:
        table['p_holm_three_endpoints'] = holm(table.p_raw.fillna(1).to_numpy())
        table.loc[table.p_raw.isna(), 'p_holm_three_endpoints'] = np.nan
        correction = 'All three fixed-budget endpoints. Non-estimable slots retained.'
    table.to_csv(root / 'analysis/paper_behavior.csv', index=False)
    write_json(root / 'analysis/paper_statistics.json', dict(computed=True, correction=correction,
        unit='paired training seed', bootstrap_draws=len(draw),
        missing_values='No imputation; a missing planned pair makes that endpoint non-estimable.'))
