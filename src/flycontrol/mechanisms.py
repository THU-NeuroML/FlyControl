"""Generate neural histories and assays from this run's freshly trained policies."""
from pathlib import Path
import os
import subprocess
import sys
from flycontrol.paths import read_json, write_json, sha256
from flycontrol.runtime import environment


def terrain_history_phases(grid,study):
    """One prespecified phase per bin, intersected with a smoke evaluation grid."""
    phases=[int(b[0]) for b in study['phase_bins'] if int(b[0]) in grid['phases']]
    if not phases:raise ValueError('Evaluation grid lacks all prespecified neural-history phases')
    return phases


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve(): raise ValueError(f'Conflicting generated link: {target}')
    elif target.exists():
        raise FileExistsError(target)
    else:
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def run_job(root, family, assay, seed, method='connectome', device='cpu'):
    env = environment(root, family, 'nmf', 'common')
    env.update(FLYCONTROL_DEVICE=device, SPEED_ANALYSIS_ROOT=str(root/'data/figure4'),
               FIGURE3_REGISTERED_OUTPUT=str(root/'registered'))
    subprocess.run([sys.executable, '-m', 'flycontrol.mechanisms', assay, method, str(seed), device],
                   cwd=root, env=env, check=True)


def prepare(root, spec):
    from flycontrol.connectomes.anatomy import prepare as prepare_anatomy
    prepare_anatomy(root, spec)
    if spec['family'] == 'terrain' and 'er' in spec['methods']:
        for seed in spec['seeds']: run_job(root, 'terrain', 'terrain_graph', seed)
    if spec['family'] == 'turning':
        for seed in spec['seeds']: run_job(root, 'turning', 'turning_graph', seed)


def probe(root, spec, device='cpu'):
    family = spec['family']
    if family=='historical':
        from flycontrol.experiments.supp_s1.workflow import worker
        worker(root,'probe',device=device)
        write_json(root/'mechanisms/complete.json',dict(assays=spec['mechanisms'],smoke=spec['smoke']))
        return
    if family=='supplemental':
        from flycontrol.experiments.shared.workflow import worker
        worker(root,'probe',device=device)
        write_json(root/'mechanisms/complete.json',dict(assays=spec.get('mechanisms',[]),smoke=spec['smoke']))
        return
    if not spec.get('mechanisms'):
        write_json(root/'mechanisms/complete.json', {'assays': [], 'reason': 'No neural assay in this preset'})
        return
    if not {'connectome','mlp'} <= set(spec['methods']):
        if not spec['smoke']:
            raise ValueError('Neural assays require the paired Adult and MLP histories; select both methods')
        write_json(root/'mechanisms/complete.json', {'assays': [], 'smoke': True, 'reason': 'Single-method smoke; paired neural assays not run'})
        return
    if family == 'terrain':
        data = root/'data/figure2'
        link(root/'evaluation', data/'evaluation')
        rows = read_json(root/'training/models.json')
        write_json(data/'evaluation_protocol.json', {'checkpoints': [dict(r,path=r['checkpoint']) for r in rows]})
        for seed in spec['seeds']:
            for assay in ['coarse','dense','circuit']:
                run_job(root,family,'terrain_'+assay,seed,device=device)
            if 'er' in spec['methods']: run_job(root,family,'terrain_er',seed,device=device)
            for method in ('connectome','mlp'): run_job(root,family,'terrain_gain',seed,method,device)
    elif family == 'speed':
        data = root/'data/figure4'
        link(root/'evaluation',data/'evaluation')
        for seed in spec['seeds']:
            for method in ('connectome','mlp'): run_job(root,family,'speed_collect',seed,method,device)
            for method in ('connectome','mlp'): run_job(root,family,'speed_local_response',seed,method,device)
            for assay in ('neurons','events','signed'): run_job(root,family,'speed_'+assay,seed,device=device)
    elif family == 'turning':
        from flycontrol.experiments.fig3.workflow import probe as run_turning
        run_turning(root,spec,device)
    write_json(root/'mechanisms/complete.json', dict(assays=spec['mechanisms'], smoke=spec['smoke'],
        checkpoints={r['checkpoint']:sha256(root/r['checkpoint']) for r in read_json(root/'training/models.json')}))


def main():
    from flycontrol.paths import run_root
    assay, method, seed, device = sys.argv[1:]
    seed = int(seed)
    root = run_root()
    if assay == 'terrain_graph':
        from flycontrol.experiments.fig2.current.er_graph import generate
        generate(seed,root/'data/figure2/er_inputs')
    elif assay.startswith('terrain_'):
        from importlib import import_module
        names={'coarse':'future_sensitivity','dense':'dense_future','circuit':'circuit_probe','er':'er_probe','gain':'input_gain'}
        kind=assay.removeprefix('terrain_')
        module=import_module('flycontrol.experiments.fig2.current.'+names[kind])
        if kind=='gain': module.run(method,seed)
        else: module.run(seed)
    elif assay.startswith('speed_'):
        kind=assay.removeprefix('speed_')
        from importlib import import_module
        module=import_module('flycontrol.experiments.fig4.mechanisms.'+kind)
        if kind=='collect': module.run(method,seed)
        elif kind=='local_response': module.run(method,seed,device)
        elif kind=='events': module.event_run(seed,device)
        elif kind=='signed': module.run_signed(seed,device)
        else: module.run(seed,device)
    elif assay.startswith('turning_'):
        from flycontrol.experiments.fig3.workflow import job
        job(root,assay.removeprefix('turning_'),method,seed,device)
    else: raise ValueError(assay)


if __name__=='__main__': main()
