"""Recurrent PPO interface for the original shared CfC; no LSTM is constructed.

sb3-contrib calls its storage slots ``lstm_*``. Only the hidden slot contains
CfC membrane state. Cell slots are zeros required by that library's ABI.
Actor and critic share the CfC, with both losses differentiating through it.
"""
from types import SimpleNamespace
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class RecurrentCfCPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Intentionally bypass the constructor that would allocate LSTM weights.
        ActorCriticPolicy.__init__(self, observation_space, action_space,
                                   lr_schedule, **kwargs)
        assert self.share_features_extractor
        n = self.features_extractor.rnn.total_dim
        self.lstm_actor = SimpleNamespace(num_layers=1, hidden_size=n)
        self.lstm_critic = None
        self.lstm_hidden_state_shape = (1, 1, n)
        assert not any(isinstance(m, torch.nn.LSTM) for m in self.modules())

    def _build_mlp_extractor(self):
        ActorCriticPolicy._build_mlp_extractor(self)

    def sequence_features(self, obs, states, episode_starts):
        """Unroll environment-major padded sequences without detaching time steps."""
        obs = preprocess_obs(obs, self.observation_space,
                             normalize_images=self.normalize_images)
        n_seq = states[0].shape[1]
        observations = obs.reshape(n_seq, -1, obs.shape[-1]).transpose(0, 1)
        starts = episode_starts.reshape(n_seq, -1).transpose(0, 1)
        h = states[0][0]
        outputs = []
        for observation, start in zip(observations, starts):
            h = h * (1.0 - start[:, None])
            h, feature = self.features_extractor.advance(observation, h)
            outputs.append(feature)
        features = torch.stack(outputs).transpose(0, 1).flatten(0, 1)
        hidden = h.unsqueeze(0)
        return features, (hidden, torch.zeros_like(hidden))

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        features, state = self.sequence_features(obs, lstm_states.pi, episode_starts)
        pi, vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return (actions.reshape((-1, *self.action_space.shape)),
                self.value_net(vf), log_prob, RNNStates(state, state))

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        features, _ = self.sequence_features(obs, lstm_states.pi, episode_starts)
        pi, vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(pi)
        return self.value_net(vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs, lstm_states, episode_starts):
        features, state = self.sequence_features(obs, lstm_states, episode_starts)
        pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(pi), state

    def predict_values(self, obs, lstm_states, episode_starts):
        features, _ = self.sequence_features(obs, lstm_states, episode_starts)
        return self.value_net(self.mlp_extractor.forward_critic(features))
