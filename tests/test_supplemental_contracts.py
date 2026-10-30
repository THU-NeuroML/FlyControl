"""Regression checks for registered history selection and independent stage dispatch."""
from pathlib import Path
import json
import numpy as np
import pytest
from flycontrol import pipeline
from flycontrol.paths import CONFIGS, read_json, write_json, sha256
from flycontrol.mechanisms import terrain_history_phases


def test_terrain_uses_eighty_histories_not_the_full_physical_grid():
    study=read_json(CONFIGS/'internal/figure2/study.json')
    grid={'phases':[s for b in study['phase_bins'] for s in b]}
    assert len(grid['phases'])==64
    phases=terrain_history_phases(grid,study)
    assert phases==[b[0] for b in study['phase_bins']]
    assert 5*2*len(phases)==80
    assert terrain_history_phases({'phases':phases[:1]},study)==phases[:1]
    with pytest.raises(ValueError):terrain_history_phases({'phases':[-1]},study)


def test_speed_smoke_retains_both_neural_histories(tmp_path):
    pipeline.initialize(tmp_path,'speed',smoke=True)
    cases=read_json(tmp_path/'configs/figure4/evaluation.json')['cases']
    assert cases==read_json(tmp_path/'configs/figure4/analysis.json')['mechanism_cases']
    assert len(cases)==2


def test_main_engine_evaluation_dispatch_is_not_shadowed(tmp_path,monkeypatch):
    p=tmp_path/'policy.pt';p.write_bytes(b'checkpoint fixture')
    write_json(tmp_path/'training/models.json',[dict(method='mlp',seed=4301,checkpoint='policy.pt',sha256=sha256(p))])
    calls=[];monkeypatch.setattr(pipeline,'worker',lambda *a,**kw:calls.append(a))
    pipeline.evaluate(tmp_path,dict(family='turning',methods=['mlp'],seeds=[4301]),2)
    assert calls==[(tmp_path,'turning','evaluate','mlp',4301,2)]
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Checkpoint changed'):pipeline.evaluate(tmp_path,dict(family='turning',methods=['mlp'],seeds=[4301]),2)


def test_video_recipes_cover_caption_numbering():
    rows=read_json(CONFIGS/'videos.json')
    assert [r['number'] for r in rows]==list(range(1,21))
    for r in rows:assert (CONFIGS/(r['config']+'.json')).exists()
    assert [r['config'] for r in rows[:5]]==['turning','turning','terrain','speed','speed']


def test_turning_family_keeps_undisplayed_modes_and_rejects_missing_states(tmp_path):
    import pandas as pd
    from flycontrol.analysis.mechanism_statistics import turning_intervention_comparisons
    modes=['intact','AN_off','AN_half','AN_restore','matched_off','mech_off',
           'ocellar_off','AN_DN_cut','AN_I_cut','AN_DN_matchcut','steering_off']
    rows=[]
    for seed in range(4301,4325):
        for phase in (790101,790102):
            for angle in (-60,60):
                for i,mode in enumerate(modes):
                    rows.append(dict(seed=seed,phase=phase,angle=angle,mode=mode,force=0,
                                     command_auc_rad_s=1+i*(seed-4310)/100))
    raw=pd.DataFrame(rows)
    turning_intervention_comparisons(raw,tmp_path)
    q=pd.read_csv(tmp_path/'intervention_vs_intact_statistics.csv')
    assert set(q['mode'])==set(modes[1:]) and len(q)==10
    assert (q.n==24).all() and (q.p_holm10>=q.p_raw).all()
    assert len(pd.read_csv(tmp_path/'specificity_statistics.csv'))==4
    # Dropping an undisplayed mode or one physical state must not shrink the family.
    for incomplete in (raw[raw['mode'].ne('AN_half')],raw.iloc[11:]):
        with pytest.raises(ValueError):turning_intervention_comparisons(incomplete,tmp_path)
