import importlib.util
from itertools import product

import numpy as np
import pytest

from flycontrol.paths import CONFIGS, EXPERIMENTS, read_json


def load_body_analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location('body_primary_statistics',EXPERIMENTS/'fig6/analysis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def body_fixture():
    config = read_json(CONFIGS/'internal/figure6/analysis.json')
    config['bodies'] = ['tripod']
    config['training_seeds'] = {'tripod':[861,862]}
    config['evaluation_seeds'] = [899200,899201]
    config['bootstrap_resamples'] = 100
    rows = []
    for seed,method,evaluation,condition in product([861,862],config['methods'],config['evaluation_seeds'],config['conditions']):
        rows.append(dict(body='tripod',training_seed=seed,method=method,evaluation_seed=evaluation,
            condition=condition,training_steps=131072,teacher_used=False,external_support_used=False,
            evaluation_device='cpu',duration_s=20.,horizon_s=20.,velocity_mps=.4,lateral_max_m=.1,
            success=method=='connectome' and condition=='flat'))
    return config,rows


def test_body_primary_weights_all_seven_conditions(monkeypatch):
    module = load_body_analysis(monkeypatch)
    config,rows = body_fixture()
    cells,effects,_ = module.summarize(rows,config)
    assert len(cells) == 2*2*7
    assert effects[0]['mean_delta'] == pytest.approx(1/7)
    assert effects[0]['paired_seed_differences'] == pytest.approx([1/7,1/7])


@pytest.mark.parametrize('mutation',['missing','duplicate','wrong_budget'])
def test_body_primary_rejects_invalid_evidence(monkeypatch,mutation):
    module = load_body_analysis(monkeypatch)
    config,rows = body_fixture()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(rows[0].copy())
    else:rows[0]['training_steps'] = 100
    with pytest.raises(ValueError):module.summarize(rows,config)
