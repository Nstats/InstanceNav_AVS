"""VLFM's discrete PointNav policy without habitat_baselines.

The network is VLFM's own non-habitat re-implementation
(vlfm/policy/utils/non_habitat_policy) with discrete actions; the head is
habitat_baselines' CategoricalNet (one linear layer, 512 -> 4). Deterministic actions
are the argmax, as CategoricalNet.mode(). tools/export_vlfm_pointnav.py --check compares
it with habitat_baselines' PointNavResNetPolicy on the same weights.
"""
import torch
from torch import nn

# Checkpoint key -> key of VLFM's non-habitat network (the only renamed parameter).
_RENAMED = {'net.prev_action_embedding.weight': 'net.prev_action_embedding_discrete.weight'}
# Parameters the policy does not use at inference.
_UNUSED = ('critic.',)


class _CategoricalHead(nn.Module):
    def __init__(self, num_inputs=512, num_outputs=4):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return self.linear(x)


class DiscretePointNavPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import PointNavResNetNet
        self.net = PointNavResNetNet(discrete_actions=True)
        self.action_distribution = _CategoricalHead()

    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False):
        features, rnn_hidden_states, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        logits = self.action_distribution(features)
        if deterministic:
            action = logits.argmax(dim=-1, keepdim=True)
        else:
            action = torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
        return action, rnn_hidden_states


def load_discrete_pointnav_policy(file_path):
    """Load the state_dict written by tools/export_vlfm_pointnav.py; every key must match."""
    state = torch.load(file_path, map_location='cpu', weights_only=True)
    state = {_RENAMED.get(k, k): v for k, v in state.items() if not k.startswith(_UNUSED)}
    policy = DiscretePointNavPolicy()
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy
