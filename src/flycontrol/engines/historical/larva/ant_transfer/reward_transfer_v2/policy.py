"""Fixed-order graph propagation and PPO in the stored Gaussian coordinates."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

from larva.locomotion.connectome import ConnectomeData
from larva.locomotion.models import ConnectomeRateCore, Policy, ValueHead


def segment_product(x, source, weight, lengths):
    # One output row has one fixed summation order, for every batch size.
    messages = x.index_select(1, source) * weight
    # Always reduce along axis 1, including batch size 1. This fixed reduction
    # is independent of batch size and avoids launching one kernel per row.
    batch_lengths = lengths.unsqueeze(0).expand(x.shape[0], -1).contiguous()
    return torch.segment_reduce(messages, 'sum', lengths=batch_lengths, axis=1)


class FixedGraphProduct(torch.autograd.Function):
    """Exact linear adjoint, without atomic accumulation in gather's backward."""
    @staticmethod
    def forward(ctx, x, f_source, f_weight, f_lengths, b_source, b_weight, b_lengths):
        ctx.save_for_backward(b_source, b_weight, b_lengths)
        return segment_product(x, f_source, f_weight, f_lengths)

    @staticmethod
    def backward(ctx, gradient):
        source, weight, lengths = ctx.saved_tensors
        return segment_product(gradient, source, weight, lengths), None, None, None, None, None, None


class FixedGraph(nn.Module):
    def __init__(self, pre, post, weights, n):
        super().__init__()
        # Stable sorting preserves the archive's relative edge order per row.
        forward = np.argsort(post, kind='stable')
        backward = np.argsort(pre, kind='stable')
        arrays = dict(f_source=pre[forward], f_weight=weights[forward],
                      f_lengths=np.bincount(post, minlength=n),
                      b_source=post[backward], b_weight=weights[backward],
                      b_lengths=np.bincount(pre, minlength=n))
        for name, value in arrays.items():
            dtype = torch.float32 if 'weight' in name else torch.long
            self.register_buffer(name, torch.as_tensor(value.copy(), dtype=dtype), persistent=False)

    def forward(self, x):
        return FixedGraphProduct.apply(x, self.f_source, self.f_weight, self.f_lengths,
                                       self.b_source, self.b_weight, self.b_lengths)


class FixedRateCore(ConnectomeRateCore):
    def __init__(self, graph, obs_dim, action_dim, readout='ant'):
        super().__init__(graph, obs_dim, action_dim, dt=0.01, tau=0.02,
                         input_target='sensory', internal_substeps=1, initial_state_std=0.)
        self.fixed_graph = FixedGraph(graph.edge_pre, graph.edge_post, graph.edge_weight, graph.N)
        if readout in {'normalized','reflex'}:
            self.output_layer = nn.Sequential(nn.LayerNorm(graph.output_dim, eps=1e-5),
                                              nn.Linear(graph.output_dim, action_dim))
        elif readout != 'ant':
            raise ValueError(readout)

    def recurrent_drive(self, rates):
        return self.fixed_graph(rates)


class GraphActor(nn.Module):
    has_state = True

    def __init__(self, path, obs_dim, action_dim, readout):
        super().__init__()
        with np.load(path) as archive:
            graph = ConnectomeData(W=None, sensory_dim=2279, internal_dim=135015, output_dim=1290,
                sensory_ids=(), internal_ids=(), output_ids=(), edge_pre=archive['edge_pre'],
                edge_post=archive['edge_post'], edge_weight=archive['edge_weight'])
        self.frame_obs_dim = obs_dim
        self.core = FixedRateCore(graph, obs_dim, action_dim, readout)
        self.state_dim = graph.N
        self.reflex = None
        if readout == 'reflex':
            self.reflex = nn.Sequential(nn.Linear(obs_dim,64),nn.ELU(),nn.Linear(64,action_dim))

    def init_state(self, batch, device):
        return self.core.init_state(batch, device)

    def forward(self, obs, state):
        assert obs.shape[-1] == 4*self.frame_obs_dim
        for frame in obs.reshape(obs.shape[0],4,self.frame_obs_dim).unbind(1):
            state, mean = self.core.step(state, frame)
        if self.reflex is not None:
            mean = mean + self.reflex(obs)
        return mean, state


class MLPActor(nn.Module):
    has_state = False
    state_dim = 0

    def __init__(self, obs_dim, action_dim, target_parameters):
        super().__init__()
        width = min(range(64,2049), key=lambda h:abs((obs_dim+1)*h+(h+1)*h+(h+1)*action_dim-target_parameters))
        self.net = nn.Sequential(nn.Linear(obs_dim,width),nn.ELU(),nn.Linear(width,width),nn.ELU(),nn.Linear(width,action_dim))

    def init_state(self, batch, device):
        return torch.empty(batch,0,device=device)

    def forward(self, obs, state):
        return self.net(obs), state


class ReflexOnlyActor(nn.Module):
    """Exact local-control branch of the composite actor, without the graph."""
    has_state = False
    state_dim = 0

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim,64),nn.ELU(),nn.Linear(64,action_dim))

    def init_state(self,batch,device):
        return torch.empty(batch,0,device=device)

    def forward(self,obs,state):
        return self.net(obs),state


class GaussianCoordinatePolicy(Policy):
    """Buffer actions are Gaussian samples; the environment alone applies tanh.

    PPO ratios may be computed in these coordinates because the fixed bijection
    has the same Jacobian for old/new policies and it cancels. No atanh, no
    action clipping or density inversion appears in the update.
    """
    def forward(self, obs, hidden):
        mean, hidden, value = super().forward(obs, hidden)
        return 3.*torch.tanh(mean/3.), hidden, value

    @torch.no_grad()
    def act(self, obs, hidden, deterministic=False, generator=None):
        mean, state, value = self.forward(obs,hidden)
        if deterministic:
            return mean, torch.zeros(mean.shape[0],device=mean.device), value, state
        noise = torch.randn(mean.shape,dtype=mean.dtype,device='cpu',generator=generator)
        distribution = self.distribution(mean)
        raw = (mean.cpu() + self.log_std.cpu().exp()*noise).to(mean.device)
        return raw, distribution.log_prob(raw).sum(-1), value, state

    def evaluate(self, obs, hidden, actions):
        mean, state, value = self.forward(obs,hidden)
        distribution = self.distribution(mean)
        return distribution.log_prob(actions).sum(-1), distribution.entropy().sum(-1), value, state


def build_policy(graph_path: Path, obs_dim: int, action_dim: int, method: str,
                 seed: int, device: str, readout='ant', log_std=-1.):
    assert obs_dim==688 and readout=='normalized' and method in {'connectome','mlp'}
    torch.manual_seed(seed)
    reflex_parameters=(obs_dim+1)*64+65*action_dim
    expected = 2279*(172+1) + (1290+1)*action_dim + (2580 if readout in {'normalized','reflex'} else 0)
    if readout=='reflex':expected+=reflex_parameters
    if method=='connectome':actor=GraphActor(graph_path,172,action_dim,readout)
    elif method=='mlp':actor=MLPActor(obs_dim,action_dim,expected)
    elif method=='reflex_only' and readout=='reflex':actor=ReflexOnlyActor(obs_dim,action_dim)
    else:raise ValueError((method,readout))
    final = actor.core.output_layer if method=='connectome' and readout=='ant' else actor.core.output_layer[-1] if method=='connectome' else actor.net[-1]
    nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)
    if readout=='reflex' and method in {'connectome','reflex_only'}:
        # Identical local branch initialization for its explicit ablation.
        branch=actor.reflex if method=='connectome' else actor.net
        with torch.random.fork_rng():
            torch.manual_seed(seed+400000)
            branch[0].reset_parameters()
        nn.init.zeros_(branch[-1].weight);nn.init.zeros_(branch[-1].bias)
    actual = sum(v.numel() for v in actor.parameters() if v.requires_grad)
    assert actual==reflex_parameters if method=='reflex_only' else abs(actual/expected-1)<.005
    with torch.random.fork_rng():
        torch.manual_seed(seed+200000)
        critic = ValueHead(obs_dim,[256,256],activation='relu')
    return GaussianCoordinatePolicy(actor,critic,action_dim,log_std_init=log_std).to(device)
