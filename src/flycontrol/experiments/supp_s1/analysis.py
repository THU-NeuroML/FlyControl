"""S1 state-sensitivity and activity-scaled sensitivity summaries."""
import pandas as pd
from flycontrol.paths import write_json, sha256


def analyze(root, spec):
    out=root/'analysis';out.mkdir(exist_ok=True)
    paths=sorted((root/'mechanisms/historical/raw').glob('sensitivity_*/SENSITIVITY.csv.gz'))
    if not paths:raise FileNotFoundError('Historical state sensitivities have not been generated')
    d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    d.to_csv(out/'state_sensitivity.csv.gz',index=False)
    keys=['training_seed','pulse_time_s','horizon_ms','scale','category']
    means=d.groupby(keys)[['energy','rms','full_share','afferent_enrichment']].mean().reset_index();means.to_csv(out/'seed_sensitivity.csv',index=False)
    from flycontrol.experiments.supp_s1.workflow import worker
    worker(root,'natural_analysis')
    write_json(out/'manifest.json',dict(smoke=spec['smoke'],statistical_unit='training seed',sources={str(p.relative_to(root)):sha256(p) for p in paths}))
