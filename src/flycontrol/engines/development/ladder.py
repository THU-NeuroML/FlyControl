"""Evaluation-only terrain adapter; all controllers call the frozen evaluator."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = __import__("flycontrol.paths",fromlist=["run_root"]).run_root()
MODEL_ROOT = ROOT.parent
from experiment import ManagedLegacyEnv, write_json, sha256
import cpg_rl_env
from flygym.arena import GappedTerrain, BlocksTerrain

CONFIG = json.loads((ROOT/'configs/development_terrain/ladder.json').read_text())


def spec_for(terrain, level):
    settings = CONFIG['terrains'][terrain]
    return {'terrain': terrain, 'level': level, 'parameter': settings['parameter'],
            'value_mm': settings['levels'][level], 'obstacle_start_x_mm': CONFIG['obstacle_start_x_mm']}


def make_arena(spec, seed):
    start = spec['obstacle_start_x_mm']
    if spec['terrain'] == 'gapped':
        arena = GappedTerrain(x_range=(start+spec['value_mm'], 25.),
                              gap_width=spec['value_mm'], block_width=1., gap_depth=2.)
        # Original platform elevation and bottom plane; first gap starts at x=2.
        arena.root_element.worldbody.add('geom', type='box', name='safe_spawn_apron',
            size=((start+10.)/2, 20., 1.), pos=((start-10.)/2, 0, 0),
            friction=arena.friction, rgba=(.3,.3,.3,1.))
    else:
        # Original blocks overlap by 5% per edge: offset keeps the leading face at x=2.
        arena = BlocksTerrain(x_range=(start+.065, 25.), block_size=1.3,
            height_range=(spec['value_mm'], spec['value_mm']), rand_seed=seed)
        arena.root_element.worldbody.add('geom', type='box', name='safe_spawn_apron',
            size=((start+10.)/2, 20., .7), pos=((start-10.)/2, 0, -.6),
            friction=arena.friction, rgba=(.3,.3,.3,1.))
    return arena


class DifficultyEnv(ManagedLegacyEnv):
    def __init__(self, spec, **kwargs):
        self.difficulty_spec = spec
        kwargs.setdefault("timestep", CONFIG["physics_dt"])
        kwargs.setdefault("decision_interval", CONFIG["decision_interval"])
        super().__init__(**kwargs)

    def _build_sim(self, terrain_name, seed):
        # Process-local, single-threaded adapter, restored even on constructor failure.
        original = cpg_rl_env._make_arena
        cpg_rl_env._make_arena = lambda name, sd: make_arena(self.difficulty_spec, sd)
        try:
            super()._build_sim(terrain_name, seed)
        finally:
            cpg_rl_env._make_arena = original


def run_path(phase, terrain, level, kind, train_seed):
    label = f'{kind}_seed{train_seed:02d}' if kind in ['connectome','er'] else kind
    return ROOT/phase/f'{terrain}_L{level}'/label


def extend_result(path, spec, phase, kind, train_seed):
    row = json.loads(path.read_text())
    with np.load(path.parent/'trace.npz') as trace:
        x = trace['position'][:, 0]
        feet = trace['feet'][:, :, 0]
        near = spec['obstacle_start_x_mm']
        far = near + (spec['value_mm'] if spec['terrain']=='gapped' else 1.43)
        finite = np.isfinite(x).all() and np.isfinite(feet).all()
        row.update({'difficulty': spec, 'phase': phase,
            'body_reaches_first_near_edge': bool(finite and np.any(x>=near)),
            'body_passes_first_far_edge': bool(finite and np.any(x>=far)),
            'body_beyond_first_far_edge_at_end_and_alive': bool(finite and row['survived_0_5_s'] and x[-1]>=far),
            'any_foot_enters_obstacle_zone': bool(finite and np.any(feet>=near)),
            'first_near_edge_x_mm': near, 'first_far_edge_x_mm': far,
            'progress_definition': 'Thorax x coordinate reaches/passes fixed terrain edges; not a whole-body clearance or new termination criterion.'})
    if kind in ['connectome','er']:
        row['original_checkpoint'] = str(MODEL_ROOT/'runs'/f'{kind}_seed{train_seed:02d}'/'final_model.zip')
    write_json(path,row)
    return row
