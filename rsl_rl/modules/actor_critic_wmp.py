# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import torch
import torch.nn as nn
from torch.distributions import Normal

from dreamer.networks import CrossAttentionTerrainEncoder


class ActorCriticWMP(nn.Module):
    is_recurrent = False

    @property
    def is_ame(self):
        """Compatibility flag for callers that require AME raw observation handling."""
        return self.uses_ame_raw_inputs

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 encoder_hidden_dims=[256, 128],
                 wm_encoder_hidden_dims = [64, 32],
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 latent_dim = 35,  
                 height_dim=187,
                 privileged_dim=3 + 24,
                 history_dim = 42*5,
                 wm_feature_dim = 512,  
                 wm_latent_dim=32,  
                 commands_begin_dim=6,
                 wm_prop_dim=33,
                 terrain_grid_shape=None,
                 terrain_embedding_dim=64,
                 terrain_attention_heads=16,
                 terrain_cnn_downsample=True,
                 terrain_attach_global=False,
                 critic_terrain_query_skip_dim=0,
                 use_estimation=False,
                 est_encoder_hidden_dims=[256, 128],
                 est_history_length=4,
                 architecture="wmp",
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCriticWMP, self).__init__()

        activation_fn = get_activation(activation)

        self.commands_begin_dim = commands_begin_dim
        self.wm_prop_dim = int(wm_prop_dim)
        self.architecture = str(architecture).lower()
        if self.architecture not in ("wmp", "ame", "ame_wm"):
            raise ValueError(f"Unsupported ActorCriticWMP architecture: {architecture!r}")
        self.uses_ame_raw_inputs = self.architecture in ("ame", "ame_wm")
        self.uses_world_model_feature = self.architecture in ("wmp", "ame_wm")
        self.uses_history_latent = self.architecture == "wmp"
        # MGDP-style proprioception estimation head (3-D lin vel + 4-D foot
        # height) is only wired into the AME raw-input branch.
        self.use_estimation = bool(use_estimation) and self.uses_ame_raw_inputs
        self.est_encoder_hidden_dims = list(est_encoder_hidden_dims)
        self.est_history_length = int(est_history_length)
        self.estimation_token = None

        self.latent_dim = latent_dim
        self.height_dim = int(height_dim)
        self.privileged_dim = privileged_dim
        self.terrain_query_prop_dim = int(num_actor_obs - self.height_dim)
        self.critic_prop_dim = int(num_critic_obs - self.height_dim)
        self.critic_terrain_query_skip_dim = int(critic_terrain_query_skip_dim)
        self.critic_terrain_query_prop_dim = (
            self.terrain_query_prop_dim
            if self.uses_ame_raw_inputs
            else self.critic_prop_dim - self.critic_terrain_query_skip_dim
        )

        if terrain_grid_shape is None:
            raise ValueError("terrain_grid_shape is required for ActorCriticWMP.")
        if self.wm_prop_dim <= 0 or self.wm_prop_dim > num_actor_obs:
            raise ValueError(
                f"wm_prop_dim must be in [1, num_actor_obs], got {self.wm_prop_dim} "
                f"for num_actor_obs={num_actor_obs}."
            )
        if num_actor_obs < self.wm_prop_dim + self.height_dim:
            raise ValueError(
                f"Actor observations must contain wm_prop and terrain map, got "
                f"num_actor_obs={num_actor_obs}, wm_prop_dim={self.wm_prop_dim}, "
                f"height_dim={self.height_dim}."
            )
        if self.terrain_query_prop_dim <= 0:
            raise ValueError(
                f"Actor observations must contain policy proprioception before the terrain map, "
                f"got num_actor_obs={num_actor_obs}, height_dim={self.height_dim}."
            )
        if self.uses_ame_raw_inputs and not (
            0 <= self.critic_terrain_query_skip_dim
            and self.critic_terrain_query_skip_dim + self.critic_terrain_query_prop_dim
            <= self.critic_prop_dim
        ):
            raise ValueError(
                "Critic terrain query does not fit in critic proprioception: "
                f"skip={self.critic_terrain_query_skip_dim}, query="
                f"{self.critic_terrain_query_prop_dim}, critic_prop_dim="
                f"{self.critic_prop_dim}."
            )

        self.terrain_encoder = CrossAttentionTerrainEncoder(
            prop_shape=(self.terrain_query_prop_dim,),
            terrain_shape=(self.height_dim,),
            mha_dim=int(terrain_embedding_dim),
            num_heads=int(terrain_attention_heads),
            act="SiLU",
            norm=True,
            cnn_downsample=bool(terrain_cnn_downsample),
            attach_global=bool(terrain_attach_global),
            terrain_grid_shape=tuple(terrain_grid_shape),
        )
        self.terrain_embedding_dim = self.terrain_encoder.outdim

        if self.uses_ame_raw_inputs:
            mlp_input_dim_a = self.terrain_query_prop_dim + self.terrain_embedding_dim
            mlp_input_dim_c = self.critic_prop_dim + self.terrain_embedding_dim
            if self.use_estimation:
                # Actor also consumes the 7-D estimation token (3-D lin vel + 4-D foot height).
                mlp_input_dim_a += 7
            if self.uses_world_model_feature:
                mlp_input_dim_a += wm_latent_dim
                mlp_input_dim_c += wm_latent_dim
        else:
            mlp_input_dim_a = latent_dim + 3 + self.terrain_embedding_dim + wm_latent_dim
            mlp_input_dim_c = num_critic_obs + self.terrain_embedding_dim + wm_latent_dim

        if self.use_estimation:
            # MGDP-style MLPModule: Linear(history_dim, 256) -> ReLU -> Linear(256, 128)
            # -> ReLU -> Linear(128, 7). history_dim = est_history_length * prop_dim.
            if history_dim != self.est_history_length * self.terrain_query_prop_dim:
                raise ValueError(
                    f"Estimation head expects history_dim={self.est_history_length * self.terrain_query_prop_dim} "
                    f"(est_history_length={self.est_history_length} x prop_dim="
                    f"{self.terrain_query_prop_dim}), got history_dim={history_dim}."
                )
            est_layers = [nn.Linear(history_dim, self.est_encoder_hidden_dims[0]), nn.ReLU()]
            for l in range(len(self.est_encoder_hidden_dims)):
                if l == len(self.est_encoder_hidden_dims) - 1:
                    est_layers.append(nn.Linear(self.est_encoder_hidden_dims[l], 7))
                else:
                    est_layers.append(
                        nn.Linear(self.est_encoder_hidden_dims[l], self.est_encoder_hidden_dims[l + 1])
                    )
                    est_layers.append(nn.ReLU())
            self.est_encoder = nn.Sequential(*est_layers)

        if self.uses_history_latent:
            # Only WMP consumes a history-derived policy latent.
            encoder_layers = [nn.Linear(history_dim, encoder_hidden_dims[0]), activation_fn]
            for l in range(len(encoder_hidden_dims)):
                if l == len(encoder_hidden_dims) - 1:
                    encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
                else:
                    encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                    encoder_layers.append(activation_fn)
            self.history_encoder = nn.Sequential(*encoder_layers)

        if self.uses_world_model_feature:
            wm_encoder_layers = [nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]), activation_fn]
            for l in range(len(wm_encoder_hidden_dims)):
                if l == len(wm_encoder_hidden_dims) - 1:
                    wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
                else:
                    wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                    wm_encoder_layers.append(activation_fn)
            self.wm_feature_encoder = nn.Sequential(*wm_encoder_layers)

            critic_wm_encoder_layers = [nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]), activation_fn]
            for l in range(len(wm_encoder_hidden_dims)):
                if l == len(wm_encoder_hidden_dims) - 1:
                    critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
                else:
                    critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                    critic_wm_encoder_layers.append(activation_fn)
            self.critic_wm_feature_encoder = nn.Sequential(*critic_wm_encoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)

        self.critic = nn.Sequential(*critic_layers)



        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @property
    def uses_estimation(self):
        return self.use_estimation

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

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

    def encode_terrain(self, query_prop, height_map):
        if query_prop.shape[-1] != self.terrain_query_prop_dim:
            raise ValueError(
                f"Terrain encoder expected query prop dim {self.terrain_query_prop_dim}, "
                f"got {query_prop.shape}."
            )
        if height_map.shape[-1] != self.height_dim:
            raise ValueError(
                f"Terrain encoder expected height dim {self.height_dim}, got {height_map.shape}."
            )
        return self.terrain_encoder(query_prop, height_map)

    def encode_policy_observation(self, observations):
        if observations.shape[-1] < self.terrain_query_prop_dim + self.height_dim:
            raise ValueError(
                f"Policy observation dim {observations.shape[-1]} is too small for "
                f"query_prop_dim={self.terrain_query_prop_dim} and height_dim={self.height_dim}."
            )
        query_prop = observations[..., : self.terrain_query_prop_dim]
        height_map = observations[..., -self.height_dim :]
        return self.encode_terrain(query_prop, height_map)

    def encode_critic_observation(self, observations):
        if observations.shape[-1] < self.critic_prop_dim + self.height_dim:
            raise ValueError(
                f"Critic observation dim {observations.shape[-1]} is too small for "
                f"critic_prop_dim={self.critic_prop_dim} and height_dim={self.height_dim}."
            )
        query_prop = observations[
            ...,
            self.critic_terrain_query_skip_dim : self.critic_terrain_query_skip_dim
            + self.critic_terrain_query_prop_dim,
        ]
        height_map = observations[..., -self.height_dim :]
        return self.encode_terrain(query_prop, height_map)

    def get_last_attention_map(self, size=None):
        return self.terrain_encoder.get_last_attention_map(size=size)

    def act(self, observations, history, wm_feature, terrain_embedding=None, **kwargs):
        if terrain_embedding is None:
            terrain_embedding = self.encode_policy_observation(observations)
        if self.uses_ame_raw_inputs:
            prop = observations[..., : self.terrain_query_prop_dim]
            inputs = (terrain_embedding, prop)
            if self.use_estimation:
                est_token = self.est_encoder(history)
                self.estimation_token = est_token
                inputs = (terrain_embedding, est_token, prop)
            if self.uses_world_model_feature:
                inputs += (self.wm_feature_encoder(wm_feature),)
            self.update_distribution(torch.cat(inputs, dim=-1))
            return self.distribution.sample()
        latent_vector = self.history_encoder(history)
        command = observations[:, self.commands_begin_dim:self.commands_begin_dim + 3]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.cat(
            (latent_vector, command, terrain_embedding, wm_latent_vector), dim=-1
        )
        self.update_distribution(concat_observations)
        return self.distribution.sample()

    def get_latent_vector(self, observations, history, **kwargs):
        if not self.uses_history_latent:
            raise RuntimeError("This architecture does not expose a history latent vector.")
        latent_vector = self.history_encoder(history)
        return latent_vector

    def get_linear_vel(self, observations, history, **kwargs):
        if not self.uses_history_latent:
            raise RuntimeError("This architecture does not use history-based velocity prediction.")
        latent_vector = self.history_encoder(history)
        linear_vel = latent_vector[:,-3:]
        return linear_vel

    def get_estimation_token(self, history, **kwargs):
        """Return the MGDP-style 7-D estimation token (3-D lin vel + 4-D foot height)."""
        if not self.use_estimation:
            raise RuntimeError("The estimation head is disabled for this architecture.")
        return self.est_encoder(history)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, wm_feature, terrain_embedding=None):
        if terrain_embedding is None:
            terrain_embedding = self.encode_policy_observation(observations)
        if self.uses_ame_raw_inputs:
            prop = observations[..., : self.terrain_query_prop_dim]
            inputs = (terrain_embedding, prop)
            if self.use_estimation:
                est_token = self.est_encoder(history)
                self.estimation_token = est_token
                inputs = (terrain_embedding, est_token, prop)
            if self.uses_world_model_feature:
                inputs += (self.wm_feature_encoder(wm_feature),)
            return self.actor(torch.cat(inputs, dim=-1))
        latent_vector = self.history_encoder(history)
        command = observations[:, self.commands_begin_dim:self.commands_begin_dim + 3]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.cat(
            (latent_vector, command, terrain_embedding, wm_latent_vector), dim=-1
        )
        actions_mean = self.actor(concat_observations)
        return actions_mean

    def evaluate(
        self,
        critic_observations,
        wm_feature,
        terrain_embedding=None,
        policy_observations=None,
        **kwargs,
    ):
        if self.uses_ame_raw_inputs:
            critic_terrain_embedding = kwargs.get("critic_terrain_embedding")
            if critic_terrain_embedding is None:
                critic_terrain_embedding = self.encode_critic_observation(critic_observations)
            prop = critic_observations[..., : self.critic_prop_dim]
            inputs = (critic_terrain_embedding, prop)
            if self.uses_world_model_feature:
                inputs += (self.critic_wm_feature_encoder(wm_feature),)
            return self.critic(torch.cat(inputs, dim=-1))
        if terrain_embedding is None:
            if policy_observations is None:
                raise ValueError(
                    "evaluate requires terrain_embedding or policy_observations."
                )
            terrain_embedding = self.encode_policy_observation(policy_observations)
        wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)
        concat_observations = torch.cat(
            (critic_observations, terrain_embedding, wm_latent_vector), dim=-1
        )
        value = self.critic(concat_observations)
        return value


class ActorCriticWMPDeployment(nn.Module):
    """Deployment-only actor graph for TorchScript export."""

    def __init__(self, actor_critic):
        super().__init__()
        self.terrain_encoder = actor_critic.terrain_encoder
        self.actor = actor_critic.actor
        self.uses_ame_raw_inputs = actor_critic.uses_ame_raw_inputs
        self.uses_world_model_feature = actor_critic.uses_world_model_feature
        self.uses_history_latent = actor_critic.uses_history_latent
        self.use_estimation = actor_critic.use_estimation
        self.terrain_query_prop_dim = actor_critic.terrain_query_prop_dim
        self.height_dim = actor_critic.height_dim
        self.commands_begin_dim = actor_critic.commands_begin_dim
        if self.uses_history_latent:
            self.history_encoder = actor_critic.history_encoder
        if self.use_estimation:
            self.est_encoder = actor_critic.est_encoder
        if self.uses_world_model_feature:
            self.wm_feature_encoder = actor_critic.wm_feature_encoder

    def forward(self, observations, history, wm_feature):
        query_prop = observations[..., : self.terrain_query_prop_dim]
        height_map = observations[..., -self.height_dim :]
        terrain_embedding = self.terrain_encoder(query_prop, height_map)
        if self.uses_ame_raw_inputs:
            inputs = (terrain_embedding, query_prop)
            if self.use_estimation:
                est_token = self.est_encoder(history)
                inputs = (terrain_embedding, est_token, query_prop)
            if self.uses_world_model_feature:
                inputs += (self.wm_feature_encoder(wm_feature),)
            return self.actor(torch.cat(inputs, dim=-1))
        latent_vector = self.history_encoder(history)
        command = observations[:, self.commands_begin_dim : self.commands_begin_dim + 3]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        return self.actor(torch.cat((latent_vector, command, terrain_embedding, wm_latent_vector), dim=-1))


class ActorCriticAMEDeployment(nn.Module):
    """Deployment graph for the AME policy (optionally with the estimation head)."""

    def __init__(self, actor_critic):
        super().__init__()
        if not actor_critic.uses_ame_raw_inputs or actor_critic.uses_world_model_feature:
            raise ValueError("ActorCriticAMEDeployment requires architecture='ame'.")
        self.terrain_encoder = actor_critic.terrain_encoder
        self.actor = actor_critic.actor
        self.terrain_query_prop_dim = actor_critic.terrain_query_prop_dim
        self.height_dim = actor_critic.height_dim
        self.use_estimation = actor_critic.use_estimation
        if self.use_estimation:
            self.est_encoder = actor_critic.est_encoder

    def forward(self, observations, history=None):
        query_prop = observations[..., : self.terrain_query_prop_dim]
        height_map = observations[..., -self.height_dim :]
        terrain_embedding = self.terrain_encoder(query_prop, height_map)
        if self.use_estimation:
            if history is None:
                raise ValueError(
                    "ActorCriticAMEDeployment with estimation requires the proprio history input."
                )
            est_token = self.est_encoder(history)
            return self.actor(torch.cat((terrain_embedding, est_token, query_prop), dim=-1))
        return self.actor(torch.cat((terrain_embedding, query_prop), dim=-1))


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
        print("invalid activation function!")
        return None
