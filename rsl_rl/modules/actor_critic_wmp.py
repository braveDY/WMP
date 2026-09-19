import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticWMP(nn.Module):
    """Pure World Model-based Perception (WMP) Actor-Critic Architecture.
    
    In the native WMP design:
    - Actor policy inputs: [latent_vector (proprio history), command, wm_latent_vector (world model feature)]
    - Critic value inputs: [critic_observations, wm_latent_vector]
    """
    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 encoder_hidden_dims=[256, 128],
                 wm_encoder_hidden_dims=[64, 32],
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 latent_dim=35,
                 height_dim=187,
                 privileged_dim=3 + 24,
                 history_dim=42 * 5,
                 wm_feature_dim=512,
                 wm_latent_dim=32,
                 commands_begin_dim=6,
                 wm_prop_dim=33,
                 **kwargs):
        if kwargs:
            print("ActorCriticWMP.__init__ ignored unexpected arguments: " + str(list(kwargs.keys())))
        super(ActorCriticWMP, self).__init__()

        activation_fn = get_activation(activation)

        self.latent_dim = latent_dim
        self.height_dim = int(height_dim)
        self.privileged_dim = privileged_dim
        self.commands_begin_dim = int(commands_begin_dim)
        self.wm_prop_dim = int(wm_prop_dim)
        self.wm_feature_dim = int(wm_feature_dim)
        self.wm_latent_dim = int(wm_latent_dim)

        mlp_input_dim_a = latent_dim + 3 + wm_latent_dim  # latent_vector + command + wm_latent
        mlp_input_dim_c = num_critic_obs + wm_latent_dim   # privileged_obs + wm_latent

        # 1. History Encoder (Extracts latent dynamics vector from proprioceptive history)
        encoder_layers = [nn.Linear(history_dim, encoder_hidden_dims[0]), activation_fn]
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation_fn)
        self.history_encoder = nn.Sequential(*encoder_layers)

        # 2. World Model Feature Encoder for Actor
        wm_encoder_layers = [nn.Linear(self.wm_feature_dim, wm_encoder_hidden_dims[0]), activation_fn]
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], self.wm_latent_dim))
            else:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                wm_encoder_layers.append(activation_fn)
        self.wm_feature_encoder = nn.Sequential(*wm_encoder_layers)

        # 3. World Model Feature Encoder for Critic
        critic_wm_encoder_layers = [nn.Linear(self.wm_feature_dim, wm_encoder_hidden_dims[0]), activation_fn]
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], self.wm_latent_dim))
            else:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                critic_wm_encoder_layers.append(activation_fn)
        self.critic_wm_feature_encoder = nn.Sequential(*critic_wm_encoder_layers)

        # 4. Actor MLP (Policy)
        actor_layers = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), activation_fn]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # 5. Critic MLP (Value Function)
        critic_layers = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), activation_fn]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        print(f"[ActorCriticWMP] Actor MLP: {self.actor}")
        print(f"[ActorCriticWMP] Critic MLP: {self.critic}")

        # Action noise distribution
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, history, wm_feature, **kwargs):
        latent_vector = self.history_encoder(history)
        command = observations[:, self.commands_begin_dim:self.commands_begin_dim + 3]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.cat((latent_vector, command, wm_latent_vector), dim=-1)
        self.update_distribution(concat_observations)
        return self.distribution.sample()

    def get_latent_vector(self, observations, history, **kwargs):
        return self.history_encoder(history)

    def get_linear_vel(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        linear_vel = latent_vector[:, -3:]
        return linear_vel

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, wm_feature, **kwargs):
        latent_vector = self.history_encoder(history)
        command = observations[:, self.commands_begin_dim:self.commands_begin_dim + 3]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.cat((latent_vector, command, wm_latent_vector), dim=-1)
        return self.actor(concat_observations)

    def evaluate(self, critic_observations, wm_feature, **kwargs):
        wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)
        concat_observations = torch.cat((critic_observations, wm_latent_vector), dim=-1)
        return self.critic(concat_observations)


class ActorCriticWMPDeployment(nn.Module):
    """Deployment-only actor graph for TorchScript JIT export."""

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = actor_critic.actor
        self.history_encoder = actor_critic.history_encoder
        self.wm_feature_encoder = actor_critic.wm_feature_encoder
        self.commands_begin_dim = actor_critic.commands_begin_dim

    def forward(self, observations, history, wm_feature):
        latent_vector = self.history_encoder(history)
        command = observations[:, self.commands_begin_dim:self.commands_begin_dim + 3]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        return self.actor(torch.cat((latent_vector, command, wm_latent_vector), dim=-1))


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print(f"Invalid activation function: {act_name}!")
        return nn.ELU()
