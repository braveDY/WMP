# MIT License

# Copyright (c) 2023 NM512

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from . import tools

class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
        multi_step_length=0,
        multi_step_shift=False,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        self._multi_step_length = multi_step_length
        self._multi_step_shift = multi_step_shift

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size, device=None):
        # Use provided device or fallback to self._device
        target_device = device if device is not None else self._device
        deter = torch.zeros(batch_size, self._deter, device=target_device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete], device=target_device),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete], device=target_device),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=target_device),
                std=torch.zeros([batch_size, self._stoch], device=target_device),
                stoch=torch.zeros([batch_size, self._stoch], device=target_device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None, return_rollouts=False):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        if not return_rollouts or self._multi_step_length <= 0:
            return post, prior
        return post, prior, self.multi_step_rollout(prior, swap(action))

    def multi_step_rollout(self, prior, action):
        horizon = self._multi_step_length
        shift = int(bool(self._multi_step_shift))
        valid_starts = action.shape[1] - horizon - shift + 1
        if valid_starts <= 0:
            return {
                key: value[:, :0].unsqueeze(2).expand(
                    value.shape[0], 0, horizon, *value.shape[2:]
                )
                for key, value in prior.items()
            }

        rollouts = []
        for t in range(valid_starts):
            start = {key: value[:, t] for key, value in prior.items()}
            act = action[:, t + shift : t + shift + horizon]
            rollouts.append(self.imagine_with_action(act, start))
        return {
            key: torch.stack([rollout[key] for rollout in rollouts], dim=1)
            for key in prior.keys()
        }

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_deter_feat(self, state):
        return state["deter"]

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior
    
    def obs_step_deter(self, prev_deter, prev_stoch, prev_action, embed, is_first): #推理使用的obs_step
        """
        Deterministic obs_step for TorchScript inference.
        
        Args:
            prev_deter: Tensor [batch, deter_dim] - previous deterministic state
            prev_stoch: Tensor [batch, stoch, discrete] or [batch, stoch] - previous stochastic state
            prev_action: Tensor [batch, action_dim] - previous action  
            embed: Tensor [batch, embed_dim] - encoded observation
            is_first: Tensor [batch] - reset flag (1.0 = reset, 0.0 = continue)
        
        Returns:
            wm_latent_dict: dict with "deter", "stoch", etc.
            prior: dict (unused, return for compatibility)
        """
        batch_size = is_first.shape[0]

        # Use embed.device for all tensors to ensure device consistency
        target_device = embed.device
        initial_state = self.initial(batch_size, device=target_device)
        initial_deter = initial_state["deter"]  # [batch, deter]
        initial_stoch = initial_state["stoch"]  # [batch, stoch, discrete] or [batch, stoch]
        initial_action = torch.zeros(
            (batch_size, self._num_actions),
            device=target_device,
            dtype=embed.dtype
        )
        
        # === 用 torch.where 处理状态重置 (兼容 trace，无 Python if) ===
        # 对于 deter: [batch] -> [batch, 1] -> expand to [batch, deter]
        reset_mask_deter = is_first.unsqueeze(-1).expand_as(prev_deter)
        new_prev_deter = torch.where(
            reset_mask_deter.bool(),
            initial_deter,
            prev_deter
        )
        
        # 对于 stoch: 需要根据维度处理
        # prev_stoch 形状: discrete模式 [batch, stoch, discrete], 连续模式 [batch, stoch]
        # 使用 reshape 来统一处理，避免 if 语句
        # 将 is_first 扩展到与 prev_stoch 相同的形状
        stoch_shape = prev_stoch.shape
        # 创建与 stoch_shape 相同形状的 mask
        # is_first: [batch] -> 扩展到 stoch_shape
        reset_mask_stoch = is_first.view(-1, *([1] * (len(stoch_shape) - 1))).expand(stoch_shape)
        new_prev_stoch = torch.where(
            reset_mask_stoch.bool(),
            initial_stoch,
            prev_stoch
        )
        
        # 对于 action: [batch] -> [batch, 1] -> expand to [batch, action]
        action_mask = is_first.unsqueeze(-1).expand_as(prev_action)
        new_prev_action = torch.where(
            action_mask.bool(),
            initial_action,
            prev_action
        )
        
        # === 构建临时 prev_state dict 供 img_step 使用 ===
        # 注意: 这里的 if 语句在 trace 时会被固化，但由于 discrete 配置是固定的，所以行为正确
        if self._discrete:
            prev_state = {"stoch": new_prev_stoch, "deter": new_prev_deter}
        else:
            prev_state = {
                "mean": torch.zeros(batch_size, self._stoch, device=target_device),
                "std": torch.ones(batch_size, self._stoch, device=target_device),
                "stoch": new_prev_stoch,
                "deter": new_prev_deter
            }
        
        # === 4. 计算 prior (img_step, sample=False 确定性) ===
        prior = self.img_step(prev_state, new_prev_action, sample=False)
        
        # === 5. 计算 posterior (obs 路径) ===
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)
        stats = self._suff_stats_layer("obs", x)
        
        # 使用 mode() 而非 sample() (确定性推理)
        # 注意: 这里的 if 语句在 trace 时会被固化
        if self._discrete:
            stoch = self.get_dist(stats).mode()
        else:
            stoch = stats["mean"]  # mode = mean for normal dist
        
        # === 6. 构建输出字典 ===
        wm_latent_dict = {"stoch": stoch, "deter": prior["deter"], **stats}
        
        return wm_latent_dict, prior


    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
        cross_attention=False,
        cross_attention_prop_key="prop",
        cross_attention_terrain_key="height_map",
        cross_attention_dim=64,
        cross_attention_heads=8,
        cross_attention_cnn_downsample=True,
        cross_attention_attach_global=False,
        cross_attention_terrain_grid=None,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self._cross_attention = None
        self._cross_attention_prop_key = cross_attention_prop_key
        self._cross_attention_terrain_key = cross_attention_terrain_key
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        if cross_attention:
            self.cnn_shapes = {
                k: v
                for k, v in self.cnn_shapes.items()
                if k not in (cross_attention_prop_key, cross_attention_terrain_key)
            }
            self.mlp_shapes = {
                k: v
                for k, v in self.mlp_shapes.items()
                if k not in (cross_attention_prop_key, cross_attention_terrain_key)
            }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
            print('cnn outdim', self._cnn.outdim)
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units
            print('mlp outdim', mlp_units)
        if cross_attention:
            if cross_attention_prop_key not in shapes:
                raise KeyError(
                    f"Cross-attention proprio key '{cross_attention_prop_key}' not found in observation shapes."
                )
            if cross_attention_terrain_key not in shapes:
                raise KeyError(
                    f"Cross-attention terrain key '{cross_attention_terrain_key}' not found in observation shapes."
                )
            prop_shape = shapes[cross_attention_prop_key]
            terrain_shape = shapes[cross_attention_terrain_key]
            if len(prop_shape) not in (1, 2):
                raise ValueError(
                    f"Cross-attention proprio input must be vector-like, got {prop_shape}."
                )
            if len(terrain_shape) not in (1, 3):
                raise ValueError(
                    f"Cross-attention terrain input must be a flattened elevation scan or image-like, got {terrain_shape}."
                )
            self._cross_attention = CrossAttentionTerrainEncoder(
                prop_shape,
                terrain_shape,
                cross_attention_dim,
                cross_attention_heads,
                act,
                norm,
                cross_attention_cnn_downsample,
                cross_attention_attach_global,
                terrain_grid_shape=cross_attention_terrain_grid,
            )
            self.outdim += self._cross_attention.outdim
            self.outdim += int(np.prod(prop_shape))
            print('cross attention terrain outdim', self._cross_attention.outdim)
            print('cross attention raw prop outdim', int(np.prod(prop_shape)))

        print('total outdim:', self.outdim)

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        if self._cross_attention is not None:
            prop = obs[self._cross_attention_prop_key]
            outputs.append(
                self._cross_attention(
                    prop,
                    obs[self._cross_attention_terrain_key],
                )
            )
            outputs.append(prop.reshape(list(prop.shape[:-len(self._cross_attention._prop_shape)]) + [-1]))
        outputs = torch.cat(outputs, -1)
        return outputs

    def get_last_cross_attention_map(self, size=None):
        if self._cross_attention is None:
            return None
        return self._cross_attention.get_last_attention_map(size=size)


class CrossAttentionTerrainEncoder(nn.Module):
    def __init__(
        self,
        prop_shape,
        terrain_shape,
        mha_dim=64,
        num_heads=8,
        act="SiLU",
        norm=True,
        cnn_downsample=True,
        attach_global=False,
        terrain_grid_shape=None,
    ):
        super(CrossAttentionTerrainEncoder, self).__init__()
        if mha_dim % num_heads != 0:
            raise ValueError(
                f"cross_attention_dim ({mha_dim}) must be divisible by cross_attention_heads ({num_heads})."
            )
        act_name = act
        self._prop_dim = int(np.prod(prop_shape))
        self._prop_shape = tuple(prop_shape)
        self._terrain_input_shape = tuple(terrain_shape)
        self._mha_dim = mha_dim
        self._cnn_downsample = cnn_downsample
        self._attach_global = attach_global

        if len(self._terrain_input_shape) == 1:
            if terrain_grid_shape is None:
                raise ValueError(
                    "cross_attention_terrain_grid is required for a flattened elevation map."
                )
            self._map_scan_dim = tuple(terrain_grid_shape)
        else:
            self._map_scan_dim = self._terrain_input_shape
        if len(self._map_scan_dim) != 3:
            raise ValueError(f"Terrain map_scan_dim must be (L, W, coord_dim), got {self._map_scan_dim}.")
        self._map_length, self._map_width, input_ch = self._map_scan_dim
        if input_ch not in (1, 3):
            raise ValueError(
                f"Cross-attention terrain input channel must be 1 or 3, got {self._map_scan_dim}."
            )
        self._coord_dim = input_ch
        if mha_dim <= self._coord_dim:
            raise ValueError(
                f"cross_attention_dim ({mha_dim}) must be greater than coordinate dim ({self._coord_dim})."
            )
        self._cnn_output_dim = mha_dim - self._coord_dim
        self.outdim = mha_dim + (mha_dim if attach_global else 0)

        expected_flat_dim = int(np.prod(self._map_scan_dim))
        if len(self._terrain_input_shape) == 1 and self._terrain_input_shape[0] != expected_flat_dim:
            raise ValueError(
                f"Flattened elevation-map dim {self._terrain_input_shape[0]} does not match "
                f"map_scan_dim={self._map_scan_dim} ({expected_flat_dim})."
            )

        # Flattened (L, W, 3) scan is restored as (W, L, 3)
        # to preserve the GridPattern spatial ordering.
        h, w = self._map_width, self._map_length
        stride = 2 if cnn_downsample else 1
        self._token_hw = (math.ceil(h / stride), math.ceil(w / stride))
        self._token_count = self._token_hw[0] * self._token_hw[1]
        self.last_attention_weights = None
        self.map_cnn = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=5,
                padding=2,
                stride=stride,
                bias=False,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.GroupNorm(1, 16),
            nn.Conv2d(
                16,
                self._cnn_output_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.GroupNorm(1, self._cnn_output_dim),
        )

        self.proprio_embedding = nn.Linear(self._prop_dim, mha_dim)
        if attach_global:
            self.global_encoder = MLP(
                mha_dim,
                None,
                2,
                mha_dim,
                act_name,
                norm,
                symlog_inputs=False,
                name="CrossAttentionGlobal",
            )
            self.query_projector = nn.Linear(mha_dim * 2, mha_dim)
        else:
            self.global_encoder = None
            self.query_projector = None

        self.mha = nn.MultiheadAttention(
            embed_dim=mha_dim, num_heads=num_heads, batch_first=True
        )
        print(
            f"Cross Attention Terrain Encoder: flat_terrain={self._terrain_input_shape}, "
            f"map_scan_dim={self._map_scan_dim}, prop_embedding_dim={self._prop_dim}, "
            f"MHA dim={mha_dim}, heads={num_heads}, tokens={self._token_count}, "
            f"terrain_cnn_channels=1, coord_concat=True"
        )

    def forward(self, prop, terrain):
        leading_shape = prop.shape[:-len(self._prop_shape)]
        terrain_leading_shape = terrain.shape[:-len(self._terrain_input_shape)]
        if leading_shape != terrain_leading_shape:
            raise ValueError(
                f"Cross-attention inputs must share leading dimensions, got {leading_shape} and {terrain_leading_shape}."
            )
        prop = prop.reshape(-1, self._prop_dim)
        terrain = terrain.reshape(
            -1,
            self._map_width,
            self._map_length,
            self._coord_dim,
        )
        if self._coord_dim == 1:
            height = terrain.permute(0, 3, 1, 2)
        else:
            height = terrain[..., 2:3].permute(0, 3, 1, 2)

        cnn_features = self.map_cnn(height)
        cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(
            terrain.shape[0], -1, self._cnn_output_dim
        )
        if self._cnn_downsample:
            coords = terrain[:, ::2, ::2, : self._coord_dim]
        else:
            coords = terrain[..., : self._coord_dim]
        coords = coords.reshape(terrain.shape[0], -1, self._coord_dim)
        local_features = torch.cat([cnn_features, coords], dim=-1)
        if local_features.shape[-1] != self._mha_dim:
            raise RuntimeError(
                f"Cross-attention local feature dim must equal {self._mha_dim}, got {local_features.shape[-1]}."
            )
        proprio_embedding = self.proprio_embedding(prop)
        query_embedding = proprio_embedding

        if self._attach_global:
            global_features = self.global_encoder(local_features)
            global_features_max, _ = torch.max(global_features, dim=1)
            query_embedding = self.query_projector(
                torch.cat([global_features_max, proprio_embedding], dim=-1)
            )

        mha_output, attention_weights = self.mha(
            query=query_embedding.unsqueeze(1),
            key=local_features,
            value=local_features,
        )
        self.last_attention_weights = attention_weights.squeeze(1).reshape(
            list(leading_shape) + list(self._token_hw)
        ).detach()
        output = mha_output.squeeze(1)
        if self._attach_global:
            output = torch.cat([global_features_max, output], dim=-1)
        return output.reshape(list(leading_shape) + [self.outdim])

    def get_last_attention_map(self, size=None):
        if self.last_attention_weights is None:
            return None
        attention = self.last_attention_weights
        if size is None or tuple(attention.shape[-2:]) == tuple(size):
            return attention
        flat = attention.reshape((-1, 1) + tuple(attention.shape[-2:]))
        flat = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
        return flat.reshape(list(attention.shape[:-2]) + list(size))


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(w) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = (h+1) // 2, (w+1) // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        # print('init encoder shape:', x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        # add this to fully recover the process of conv encoder
        input_ch, h, w = shape
        stages = int(np.log2(w) - np.log2(minres))
        self.h_list = []
        self.w_list = []
        for i in range(stages):
            h, w = (h+1) // 2, (w+1) // 2
            self.h_list.append(h)
            self.w_list.append(w)
        self.h_list = self.h_list[::-1]
        self.w_list = self.w_list[::-1]
        self.h_list.append(shape[1])
        self.w_list.append(shape[2])

        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = len(self.h_list) - 1
        # layer_num = int(np.log2(shape[2]) - np.log2(minres))
        # self._minres = minres
        # out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2

        layers = []
        # h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            # pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            # pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)

            if(self.h_list[i] * 2 == self.h_list[i+1]):
                pad_h, outpad_h = 1, 0
            else:
                pad_h, outpad_h = 2, 1

            if(self.w_list[i] * 2 == self.w_list[i+1]):
                pad_w, outpad_w = 1, 0
            else:
                pad_w, outpad_w = 2, 1

            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            # h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self.h_list[0], self.w_list[0], self._embed_size // (self.h_list[0] * self.w_list[0])]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        # print('init decoder shape:', x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        # else:
        #     mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class Projector(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Projector, self).__init__()
        self.linear = nn.Linear(inp_dim, out_dim, bias=False)
        self.apply(tools.weight_init)

    def forward(self, x):
        return self.linear(x)


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
