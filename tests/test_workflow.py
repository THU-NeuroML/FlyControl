from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from flycontrol import pipeline
from flycontrol.paths import CONFIGS, read_json, write_json, sha256
from flycontrol.analysis import signflip, holm, summarize
from flycontrol.inputs import fetch
from flycontrol.runtime import environment, run_lock


def test_stage_order_and_configuration_integrity(tmp_path):
    root = tmp_path / 'run'
    spec = pipeline.initialize(root, 'tripod', smoke=True, methods=['mlp'])
    assert spec['smoke'] and spec['seeds'] == [861]
    with pytest.raises(FileNotFoundError, match='First run'):
        pipeline.execute(root, 'evaluate')
    protocol = read_json(root / 'protocol.json')
    protocol['ppo']['total_timesteps'] += 1
    write_json(root / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='configurations changed'):
        pipeline.load_run(root)


def test_changed_code_rejects_existing_run(tmp_path, monkeypatch):
    pipeline.initialize(tmp_path, 'tripod', True)
    monkeypatch.setattr(pipeline, 'source_digest', lambda: 'different')
    with pytest.raises(ValueError, match='Code has changed'):
        pipeline.load_run(tmp_path)


@pytest.mark.parametrize('kwargs', [dict(methods=['unknown']), dict(seeds=[0]), dict(seeds=[861,861]), dict(methods=['mlp','mlp'])])
def test_invalid_cohorts(tmp_path, kwargs):
    with pytest.raises(ValueError):
        pipeline.initialize(tmp_path, 'tripod', **kwargs)


def test_run_lock_excludes_other_writer(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another process'):
            with run_lock(tmp_path):
                pass


def test_input_checksums_and_traversal(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('public input')
    record = dict(path='safe/file.txt', url=source.as_uri(), sha256=sha256(source), bytes=source.stat().st_size)
    dest = tmp_path / 'inputs'
    got = fetch(record, dest)
    assert got.read_text() == 'public input'
    got.write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        fetch(record, dest)
    with pytest.raises(ValueError, match='escapes'):
        fetch(dict(record, path='../escape'), dest)


def test_no_source_repository_in_child_imports(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/imaginary/old/repository')
    env = environment(tmp_path, 'speed', 'common')
    assert '/imaginary/old/repository' not in env['PYTHONPATH']
    command = [sys.executable, '-c', 'import body_control; import fly_control; print(body_control.__file__); print(fly_control.__file__)']
    result = subprocess.run(command, env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert '/flycontrol/engines/speed/' in result.stdout
    assert '/flycontrol/engines/common/' in result.stdout
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'


def test_inference_uses_training_seed_not_episode_count():
    # Different numbers of evaluations must not weight the trained models.
    rows = []
    for seed, n, adult in [(1, 1, 4.), (2, 9, 2.)]:
        for method, value in [('connectome', adult), ('mlp', 1.)]:
            rows.extend(dict(method=method, seed=seed, condition='flat', score=value) for _ in range(n))
    sm, counts, tests = summarize(pd.DataFrame(rows), dict(seeds=[1,2], smoke=False), ['score'])
    assert len(sm) == 4
    assert tests.iloc[0].n_pairs == 2
    assert tests.iloc[0].adult_minus_mlp == 2.
    assert tests.iloc[0].p_raw == .5


def test_smoke_has_no_inferential_intervals():
    df = pd.DataFrame([dict(method=m, seed=1, condition='flat', score=v) for m,v in [('connectome',2.),('mlp',1.)]])
    _,_,tests = summarize(df, dict(seeds=[1], smoke=True), ['score'])
    assert tests.p_raw.isna().all() and tests.ci95_low.isna().all()


def test_missing_metric_does_not_shrink_holm_family():
    rows = [dict(method=m, seed=s, condition='flat', score=2. if m=='connectome' else 1., undefined=np.nan)
            for s in range(3) for m in ['connectome','mlp']]
    _,_,tests = summarize(pd.DataFrame(rows), dict(seeds=list(range(3)), smoke=False), ['score','undefined'])
    assert tests.loc[0,'p_raw'] == .25
    assert tests.loc[0,'p_holm_behavior_family'] == .5
    assert tests.loc[1,'n_pairs'] == 0


def test_exact_signflip_matches_enumeration():
    assert signflip([1., 2., 3.]) == .25
    assert signflip([0., 0., 0.]) == 1.
    np.testing.assert_allclose(holm([.01,.04,.03]), [.03,.06,.06])


def test_maze_prerequisites_are_a_topologically_ordered_graph():
    nodes = read_json(CONFIGS / 'internal/maze/prerequisites.json')
    seen = set()
    for node in nodes:
        assert node['id'] not in seen
        assert node['source'] is None or node['source'] in seen
        assert (CONFIGS / 'internal/maze' / (node['id']+'.yaml')).is_file()
        seen.add(node['id'])
    assert len(nodes) == 10


def test_model_cohort_is_complete_before_evaluation(tmp_path):
    spec = pipeline.initialize(tmp_path, 'tripod', True)
    write_json(tmp_path / 'training/models.json', [])
    with pytest.raises(ValueError, match='cohort is incomplete'):
        pipeline.evaluate(tmp_path, spec, 1)
