"""Check fused sequence replay against actual recurrent execution at resets."""
import importlib.util
from pathlib import Path

import pytest
import torch
from flycontrol.paths import ENGINES

spec = importlib.util.spec_from_file_location('test_actor_module', ENGINES / 'ant/body_control/extension_actors.py')
actors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actors)
torch.set_num_threads(1)


@pytest.mark.parametrize('kind', ['GruActor','LstmActor','TransformerActor','ReservoirActor'])
def test_sequence_matches_steps_and_preserves_gradients(kind):
    torch.manual_seed(77)
    model = getattr(actors,kind)(11,3)
    observations = torch.randn(7,3,11)
    dones = torch.zeros(7,3)
    dones[2,0] = 1
    dones[4,2] = 1
    dones[-1,1] = 1
    state = model.init_state(3,'cpu')
    step_outputs = []
    for t in range(len(observations)):
        value,state = model(observations[t],state)
        step_outputs.append(value)
        state = state * (1-dones[t,:,None])
    step = torch.stack(step_outputs)
    sequence,final = model.forward_sequence(observations, model.init_state(3,'cpu'), dones)
    torch.testing.assert_close(sequence,step,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(final,state,atol=2e-6,rtol=2e-5)
    parameters = list(model.parameters())
    expected = torch.autograd.grad(step.square().sum(),parameters)
    actual = torch.autograd.grad(sequence.square().sum(),parameters)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-4)


@pytest.mark.parametrize('readout_only',[False,True])
def test_reservoir_fixed_tensors_survive_training_mode(readout_only):
    model = actors.ReservoirActor(11,3,readout_only=readout_only)
    buffers = dict(model.named_buffers())
    assert 'recurrent_weight' in buffers
    assert ('input_weight' in buffers) == readout_only
    original = {k:v.clone() for k,v in buffers.items()}
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    value,_ = model(torch.randn(2,11),model.init_state(2,'cpu'))
    value.square().mean().backward()
    optimizer.step()
    for key,value in model.named_buffers():
        torch.testing.assert_close(value,original[key],atol=0,rtol=0)


def test_every_added_architecture_builds_from_each_stage_config(tmp_path):
    import subprocess,sys
    from flycontrol.runtime import environment
    script = '''
import torch,yaml,numpy as np
from flycontrol.paths import CONFIGS
from body_control.train import build_policy_from_config
torch.set_num_threads(1)
count=0
for method in ['gru','lstm','transformer','reservoir','reservoir_ro']:
 for stage in ['base','flat','umaze','medium','large']:
  config=yaml.safe_load((CONFIGS/'internal/maze'/f'{method}_{stage}.yaml').read_text())
  policy=build_policy_from_config(config,109,8)
  configure=getattr(policy.actor,'configure_goal_router_normalizer',None)
  if configure is not None:
   configure(np.zeros(109),np.ones(109))
  mean,state,value=policy(torch.zeros(2,109),policy.init_state(2,'cpu'))
  assert mean.shape==(2,8) and torch.isfinite(mean).all()
  assert torch.isfinite(state).all() and torch.isfinite(value).all()
  count+=1
print('CONFIGURED_POLICIES',count)
'''
    result=subprocess.run([sys.executable,'-c',script],env=environment(tmp_path,'ant','common'),cwd=tmp_path,check=True,text=True,capture_output=True)
    assert 'CONFIGURED_POLICIES 25' in result.stdout
