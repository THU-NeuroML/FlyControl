"""The installed package must not need short-name experiment import roots."""
import subprocess
import sys

from flycontrol.runtime import environment


def test_figure_entry_points_import_in_isolated_worker(tmp_path):
    env = environment(tmp_path, 'common')
    assert '/experiments' not in env['PYTHONPATH']
    # These are actual implementations, not forwarding aliases. Importing them
    # must not require an old checkout, prepared dataset or experiments/ on sys.path.
    program = '''
from importlib import import_module
from importlib.abc import MetaPathFinder
import sys

class RejectLegacyExperiments(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"fig2", "fig3", "fig4", "fig5", "fig6", "historical", "banc", "supplemental"}:
            raise AssertionError("Unqualified experiment import: " + fullname)

sys.meta_path.insert(0, RejectLegacyExperiments())
entries = {
    'fig1.materials': 'export',
    'fig2.workflow': 'evaluate',
    'fig3.workflow': 'probe',
    'fig4.statistics': 'contrast',
    'fig5.workflow': 'train',
    'fig5.analysis': 'analyze',
    'fig6.workflow': 'body_evaluate',
    'fig6.comparison': 'combine',
    'supp_s1.workflow': 'train',
    'supp_s1.analysis': 'analyze',
    'supp_s3.terrain': 'train',
    'supp_s3.turning': 'evaluate',
    'supp_s3.analysis': 'combine_development',
    'supp_s4.analysis': 'summarize',
    'supp_s6.workflow': 'evaluate',
    'supp_s6.analysis': 'regional',
    'shared.workflow': 'worker',
    'shared.speed.workflow': 'evaluate',
}
for name, function in entries.items():
    module = import_module('flycontrol.experiments.' + name)
    assert callable(getattr(module, function)), name
'''
    subprocess.run([sys.executable, '-c', program], env=env, cwd=tmp_path,
                   check=True, capture_output=True, text=True)
