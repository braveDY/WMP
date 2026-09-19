import torch
import torch.nn as nn
from torch import autograd

class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        amp_reward_coef,
        hidden_layer_sizes,
        device,
        task_reward_lerp=0.0,
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)
        self.trunk.train()
        self.amp_linear.train()
        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10.0):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad_(True)
        disc = self.forward(expert_data)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=torch.ones_like(disc),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_amp_reward(self, state, next_state, task_reward=None, normalizer=None):
        was_training = self.training
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)
            d = self.forward(torch.cat([state, next_state], dim=-1))
            amp_reward = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * torch.square(d - 1.0), min=0.0)
            if task_reward is not None and self.task_reward_lerp > 0.0:
                reward = task_reward.view_as(amp_reward) + amp_reward
            else:
                reward = amp_reward
            self.train(was_training)
        return reward.squeeze(-1), amp_reward.squeeze(-1), d
