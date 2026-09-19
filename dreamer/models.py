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

import copy
import torch
from torch import nn

from . import tools
from . import networks

to_np = lambda x: x.detach().cpu().numpy()

class WorldModel(nn.Module):
    def __init__(self, config, obs_shape, embed_size):
        super(WorldModel, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.device = self._config.device

        self.embed_size = int(embed_size)
        if self.embed_size <= 0:
            raise ValueError(f"embed_size must be positive, got {self.embed_size}.")
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            getattr(config, "multi_step_length", 0),
            getattr(config, "multi_step_shift", False),
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self._barlow_scale = float(getattr(config, "barlow_loss_scale", 0.0))
        self._barlow_lambd = float(getattr(config, "barlow_lambd", 5e-4))
        self.barlow_projector = None
        if self._barlow_scale > 0.0:
            self.barlow_projector = networks.Projector(config.dyn_deter, self.embed_size)
        if "decoder" in config.grad_heads:
            self.heads["decoder"] = networks.MultiDecoder(
                feat_size, obs_shape, **config.decoder
            )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        # can set different scale for terms in decoder here
        self._scales = dict(
            prop=getattr(config, "prop_loss_scale", 1.0),
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
            height_map=getattr(config, "height_map_loss_scale", 0.0),
            rollout_reward=getattr(config, "rollout_reward_loss_scale", 1.0),
            rollout_cont=getattr(config, "rollout_cont_loss_scale", 1.0),
        )

    def _train(self, data, embed):
        # action (batch_size, batch_length, act_dim)
        # height_map (batch_size, batch_length, L * W * 3), matching AME elevation_map
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)
        embed = torch.as_tensor(embed, device=self.device).detach()
        if embed.shape[:2] != data["action"].shape[:2]:
            raise ValueError(
                f"External embedding leading shape {embed.shape[:2]} does not match "
                f"action shape {data['action'].shape[:2]}."
            )
        if embed.shape[-1] != self.embed_size:
            raise ValueError(
                f"External embedding dim {embed.shape[-1]} does not match "
                f"WorldModel embed_size={self.embed_size}."
            )
        barlow_loss = None

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                multi_step_length = getattr(self._config, "multi_step_length", 0)
                if multi_step_length > 0:
                    post, prior, rollouts = self.dynamics.observe(
                        embed, data["action"], data["is_first"], return_rollouts=True
                    )
                else:
                    post, prior = self.dynamics.observe(
                        embed, data["action"], data["is_first"]
                    )
                    rollouts = {}
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    if name not in data:
                        continue
                    if name == "height_map":
                        loss = self._height_map_loss(pred, data[name])
                    else:
                        loss = -pred.log_prob(self._head_target(name, data[name]))
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                if rollouts:
                    rollout_losses = self._rollout_losses(
                        rollouts, data, multi_step_length
                    )
                    losses.update(rollout_losses)
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                    if not key.startswith("rollout_")
                }
                rollout_scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                    if key.startswith("rollout_")
                }
                model_loss = sum(scaled.values()) + kl_loss
                total_model_loss = torch.mean(model_loss)
                if rollout_scaled:
                    total_model_loss = total_model_loss + sum(
                        torch.mean(value) for value in rollout_scaled.values()
                    )
                if self.barlow_projector is not None:
                    deter_feat = self.dynamics.get_deter_feat(post)
                    barlow_loss, barlow_inv, barlow_red, barlow_metrics = self._barlow_loss(
                        deter_feat, embed.detach()
                    )
                    total_model_loss = total_model_loss + self._barlow_scale * barlow_loss
            metrics = self._model_opt(total_model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        if barlow_loss is not None:
            metrics["barlow_loss"] = to_np(barlow_loss)
            metrics["barlow_invariance_loss"] = to_np(barlow_inv)
            metrics["barlow_redundancy_loss"] = to_np(barlow_red)
            metrics["barlow_loss_scale"] = self._barlow_scale
            metrics.update({name: to_np(value) for name, value in barlow_metrics.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        if "height_map" in data:
            height_map = self._reshape_height_map(data["height_map"])
            height_channel = 2
            height_values = height_map[..., height_channel]
            metrics["height_map_valid_ratio"] = 1.0
            metrics["height_map_valid_std"] = to_np(torch.std(height_values))
            metrics["height_map_valid_abs_mean"] = to_np(torch.mean(torch.abs(height_values)))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def _head_target(self, name, target):
        if name == "cont" and target.ndim >= 2:
            return target.unsqueeze(-1)
        return target

    def _barlow_loss(self, feat, target, eps=1e-8):
        assert feat.shape[:-1] == target.shape[:-1], (feat.shape, target.shape)
        feat = feat.reshape(-1, feat.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        proj = self.barlow_projector(feat)
        invariance, redundancy, metrics = self.barlow_statistics(proj, target, eps)
        return (
            invariance + self._barlow_lambd * redundancy,
            invariance,
            redundancy,
            metrics,
        )

    @staticmethod
    def barlow_statistics(proj, target, eps=1e-8):
        """Return Barlow terms and diagnostics without changing the objective."""
        assert proj.shape == target.shape, (proj.shape, target.shape)
        # Keep the original Barlow normalization exactly unchanged.
        proj_std = proj.std(0)
        target_std = target.std(0)
        normalized_proj = (proj - proj.mean(0)) / (proj_std + eps)
        normalized_target = (target - target.mean(0)) / (target_std + eps)
        batch = max(normalized_proj.shape[0], 1)
        corr = torch.mm(normalized_proj.T, normalized_target) / batch
        diagonal = torch.diagonal(corr)
        invariance = (diagonal - 1.0).pow(2).sum()
        off_diag = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
        redundancy = corr[off_diag].pow(2).sum()
        off_diag_values = corr[off_diag]
        metrics = {
            "barlow_diag_corr_mean": diagonal.mean(),
            "barlow_diag_corr_min": diagonal.min(),
            "barlow_diag_corr_std": diagonal.std(unbiased=False),
            "barlow_offdiag_abs_mean": (
                off_diag_values.abs().mean()
                if off_diag_values.numel()
                else torch.zeros((), device=corr.device)
            ),
            "barlow_projected_std": proj_std.mean(),
            "barlow_target_std": target_std.mean(),
            "barlow_normalized_mse": (
                normalized_proj - normalized_target
            ).pow(2).mean(),
        }
        metrics = {name: value.detach() for name, value in metrics.items()}
        return invariance, redundancy, metrics

    def _rollout_losses(self, rollouts, data, horizon):
        valid_starts = rollouts["deter"].shape[1]
        if valid_starts <= 0:
            return {}

        losses = {}
        shift = int(bool(getattr(self._config, "multi_step_shift", False)))
        feat = self.dynamics.get_feat(rollouts)
        for name in ("reward", "cont"):
            if name not in self.heads or name not in data:
                continue
            grad_head = name in self._config.grad_heads
            pred = self.heads[name](feat if grad_head else feat.detach())
            target = torch.stack(
                [
                    data[name][:, t + shift : t + shift + horizon]
                    for t in range(valid_starts)
                ],
                dim=1,
            )
            target = self._head_target(name, target)
            loss = -pred.log_prob(target)
            assert loss.shape == rollouts["deter"].shape[:3], (
                f"rollout_{name}",
                loss.shape,
                rollouts["deter"].shape[:3],
            )
            losses[f"rollout_{name}"] = loss
        return losses

    def _height_map_loss(self, pred, target, eps=1e-6):
        pred_mean = self._reshape_height_map(pred.mode())
        target = self._reshape_height_map(target)
        assert pred_mean.shape == target.shape, (pred_mean.shape, target.shape)

        height_channel = 2
        pred_height = pred_mean[..., height_channel]
        target_height = target[..., height_channel]

        height_error = (pred_height - target_height) ** 2
        height_loss = height_error.mean(dim=(-2, -1))

        extra_loss = 0.1 * ((pred_mean[..., :2] - target[..., :2]) ** 2).mean(
            dim=tuple(range(2, target.ndim))
        )
        return height_loss + extra_loss

    def _reshape_height_map(self, height_map):
        """Restore AME's flattened elevation map to its (W, L, 3) grid."""
        channels = int(getattr(self._config, "height_map_channels", 3))
        length = int(getattr(self._config, "height_map_grid_rows", 0))
        width = int(getattr(self._config, "height_map_grid_cols", 0))
        if height_map.shape[-1] == channels and height_map.ndim >= 3:
            return height_map
        expected = length * width * channels
        if expected <= 0 or height_map.shape[-1] != expected:
            raise ValueError(
                f"Flattened height_map must have {expected} values for AME map_scan_dim "
                f"({length}, {width}, {channels}), got {height_map.shape}."
            )
        return height_map.reshape(list(height_map.shape[:-1]) + [width, length, channels])

    # this function is called during both rollout and training
    def preprocess(self, obs):
        assert "is_first" in obs
        obs = {
            k: torch.as_tensor(v, device=self._config.device)
            for k, v in obs.items()
        }
        if "is_terminal" in obs:
            obs["cont"] = 1.0 - obs["is_terminal"].float()
        return obs

    def video_pred(self, data, embed):
        if "decoder" not in self.heads:
            return None
        if "height_map" not in getattr(self.heads["decoder"], "cnn_shapes", {}):
            return None
        data = self.preprocess(data)
        embed = torch.as_tensor(embed, device=self.device).detach()

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["height_map"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["height_map"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["height_map"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model], 2)
