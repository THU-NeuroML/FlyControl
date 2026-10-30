"""Combine the four complete body experiments using the paper's fixed families."""

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

from .paths import CONFIGS, EXPERIMENTS, read_json, write_json, sha256
from .pipeline import load_run, BODIES


def combine(run_dirs, output):
    runs = {}
    sources, episodes, windows = {}, [], []
    output = Path(output).resolve()
    for directory in run_dirs:
        root = Path(directory).resolve()
        if output == root or output.is_relative_to(root):
            raise ValueError('Combined output must be separate from the individual runs')
        spec = load_run(root)
        name = spec['experiment']
        planned = read_json(CONFIGS / (name + '.json'))
        if name not in BODIES or name in runs or spec['smoke'] or set(spec['seeds']) != set(planned['seeds']) or set(spec['methods']) != {'connectome','mlp'}:
            raise ValueError('Exactly four distinct, complete body cohorts are required')
        manifest = read_json(root / 'analysis/manifest.json')
        for relative,digest in manifest['sources'].items():
            if sha256(root / relative) != digest:
                raise ValueError('Evaluation changed after analysis')
        for relative,digest in manifest['tables'].items():
            if sha256(root / 'analysis' / relative) != digest:
                raise ValueError('Body analysis table changed')
        runs[name] = root
        episodes.extend(pd.read_csv(root / 'analysis/episodes.csv').to_dict('records'))
        windows.append(pd.read_csv(root / 'analysis/body_windows.csv'))
        sources[name] = dict(run=str(root), analysis_sha256=sha256(root / 'analysis/manifest.json'))
    if set(runs) != set(BODIES):
        raise ValueError('Expected cassie, tripod, go1 and hexapod')
    os.environ['FLYCONTROL_RUN'] = str(runs['cassie'])
    sys.path.insert(0, str(EXPERIMENTS))
    from fig6.analysis import summarize
    from fig6.comprehensive import summarize as summarize_details
    config = read_json(CONFIGS / 'internal/figure6/analysis.json')
    cells,effects,_ = summarize(episodes, config)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(cells).to_csv(output / 'condition_seed_means.csv',index=False)
    pd.DataFrame(effects).to_csv(output / 'four_body_effects.csv',index=False)
    details = output / 'details';details.mkdir(exist_ok=True)
    summarize_details(pd.DataFrame(episodes), pd.concat(windows,ignore_index=True), details, expected_seeds=20)
    write_json(output / 'manifest.json', dict(sources=sources, statistical_unit='paired training seed',
        primary='Equal-weight mean across all seven conditions; Holm across four bodies',
        secondary='Condition/window comparisons; global and within-body exploratory Holm families',
        tables={str(p.relative_to(output)): sha256(p) for p in output.rglob('*.csv')}))
