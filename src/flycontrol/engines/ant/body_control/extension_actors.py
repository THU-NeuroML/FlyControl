"""Conventional actors with the archived PPO recurrent-state interface.

Reservoir+RO means readout-only, as explicitly selected by the user.
Frozen tensors are buffers, so stage-specific optimizer setup cannot unfreeze them.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class GruActor(nn.Module):
    has_state = True

    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super().__init__()
        self.state_dim = hidden_size
        self.cell = nn.GRU(obs_dim, hidden_size)
        self.head = nn.Linear(hidden_size, action_dim)

    def init_state(self, batch, device):
        return torch.zeros(batch, self.state_dim, device=device)

    def forward(self, obs, hidden):
        _, state = self.cell(obs[None], hidden[None].contiguous())
        next_hidden = state[0]
        return self.head(next_hidden), next_hidden

    def forward_sequence(self, obs, hidden, dones):
        return recurrent_sequence(self, obs, hidden, dones, lstm=False)


class LstmActor(nn.Module):
    has_state = True

    def __init__(self, obs_dim, action_dim, hidden_size=108):
        super().__init__()
        self.state_dim = 2 * hidden_size
        self.cell = nn.LSTM(obs_dim, hidden_size)
        self.head = nn.Linear(hidden_size, action_dim)

    def init_state(self, batch, device):
        return torch.zeros(batch, self.state_dim, device=device)

    def forward(self, obs, hidden):
        h, c = hidden.chunk(2, dim=-1)
        value, (h, c) = self.cell(obs[None], (h[None].contiguous(), c[None].contiguous()))
        return self.head(value[0]), torch.cat((h[0], c[0]), -1)

    def forward_sequence(self, obs, hidden, dones):
        return recurrent_sequence(self, obs, hidden, dones, lstm=True)


def recurrent_sequence(actor, obs, hidden, dones, lstm):
    """Fused recurrent replay; split only at episode boundaries, retain BPTT."""
    ends = (dones.any(dim=1).nonzero().flatten() + 1).tolist()
    if not ends or ends[-1] != len(obs):
        ends.append(len(obs))
    outputs, begin = [], 0
    for end in ends:
        if lstm:
            h, c = hidden.chunk(2, -1)
            value, (h, c) = actor.cell(obs[begin:end], (h[None].contiguous(), c[None].contiguous()))
            hidden = torch.cat((h[0], c[0]), -1)
        else:
            value, h = actor.cell(obs[begin:end], hidden[None].contiguous())
            hidden = h[0]
        outputs.append(actor.head(value))
        hidden = hidden * (1 - dones[end - 1, :, None])
        begin = end
    return torch.cat(outputs), hidden


class TransformerActor(nn.Module):
    """One causal attention block over the last 16 observed control samples.

Cache raw observations, not stale parameter-dependent embeddings. Padding is
masked; dropout is disabled so PPO collection and replay use the same policy.
The current query only attends to past/current tokens, never future samples.
"""
    has_state = True

    def __init__(self, obs_dim, action_dim, width=64, heads=4,
                 feedforward=512, context=16):
        super().__init__()
        self.obs_dim, self.context = obs_dim, context
        self.state_dim = (context - 1) * obs_dim + 1
        self.input = nn.Linear(obs_dim, width)
        self.position = nn.Parameter(torch.zeros(context, width))
        nn.init.normal_(self.position, std=0.02)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(),
                                nn.Linear(feedforward, width))
        self.head = nn.Linear(width, action_dim)

    def init_state(self, batch, device):
        return torch.zeros(batch, self.state_dim, device=device)

    def forward(self, obs, hidden):
        history = hidden[:, :-1].reshape(-1, self.context - 1, self.obs_dim)
        tokens = torch.cat((history, obs[:, None]), dim=1)
        count = hidden[:, -1:].clamp(0, self.context - 1)
        mean = self._read(tokens, count)
        state = torch.cat((tokens[:, 1:].reshape(len(obs), -1),
                           (count + 1).clamp(max=self.context - 1)), dim=-1)
        return mean, state

    def _read(self, tokens, count):
        padding = torch.arange(self.context, device=tokens.device)[None] < (self.context - 1 - count)
        encoded = self.input(tokens) + self.position
        query = encoded[:, -1:]
        attended, _ = self.attention(query, encoded, encoded,
                                     key_padding_mask=padding, need_weights=False)
        value = self.norm1(query + attended)
        value = self.norm2(value + self.ff(value))
        return self.head(value[:, 0])

    def forward_sequence(self, obs, hidden, dones):
        windows, counts = [], []
        for t in range(len(obs)):
            history = hidden[:, :-1].reshape(-1, self.context - 1, self.obs_dim)
            tokens = torch.cat((history, obs[t, :, None]), 1)
            count = hidden[:, -1:].clamp(0, self.context - 1)
            windows.append(tokens)
            counts.append(count)
            hidden = torch.cat((tokens[:, 1:].reshape(len(obs[t]), -1),
                                (count + 1).clamp(max=self.context - 1)), -1)
            hidden = hidden * (1 - dones[t, :, None])
        means = self._read(torch.cat(windows), torch.cat(counts))
        return means.reshape(len(obs), obs.shape[1], -1), hidden


class ReservoirActor(nn.Module):
    """Leaky tanh reservoir; frozen orthogonal recurrent matrix, radius .9.

Both variants use 512 units and leak .5. Reservoir trains input plus readout;
Reservoir+RO freezes input too. Readout widths roughly match trainable actor
capacity to the historical 256x256 MLP (counts are recorded, not hidden).
"""
    has_state = True

    def __init__(self, obs_dim, action_dim, readout_only=False, hidden_size=512,
                 readout_hidden=None, leak=.5, radius=.9):
        super().__init__()
        self.state_dim = hidden_size
        self.leak = float(leak)
        weight = torch.empty(hidden_size, obs_dim).uniform_(-1 / math.sqrt(obs_dim),
                                                          1 / math.sqrt(obs_dim))
        bias = torch.zeros(hidden_size)
        if readout_only:
            self.register_buffer('input_weight', weight)
            self.register_buffer('input_bias', bias)
        else:
            self.input_weight = nn.Parameter(weight)
            self.input_bias = nn.Parameter(bias)
        recurrent = torch.empty(hidden_size, hidden_size)
        nn.init.orthogonal_(recurrent, gain=radius)
        self.register_buffer('recurrent_weight', recurrent)
        width = readout_hidden or (176 if readout_only else 72)
        self.head = nn.Sequential(nn.Linear(hidden_size, width), nn.Tanh(),
                                  nn.Linear(width, action_dim))

    def init_state(self, batch, device):
        return torch.zeros(batch, self.state_dim, device=device)

    def forward(self, obs, hidden):
        proposal = torch.tanh(F.linear(obs, self.input_weight, self.input_bias)
                              + F.linear(hidden, self.recurrent_weight))
        state = (1 - self.leak) * hidden + self.leak * proposal
        return self.head(state), state

    def forward_sequence(self, obs, hidden, dones):
        drive = F.linear(obs, self.input_weight, self.input_bias)
        states = []
        for t in range(len(obs)):
            proposal = torch.tanh(drive[t] + F.linear(hidden, self.recurrent_weight))
            hidden = (1 - self.leak) * hidden + self.leak * proposal
            states.append(hidden)
            hidden = hidden * (1 - dones[t, :, None])
        return self.head(torch.stack(states)), hidden
