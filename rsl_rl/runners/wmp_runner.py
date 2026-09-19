import time
import os
from collections import deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pathlib
import yaml

from ..algorithms import AMPPPO
from ..modules import ActorCriticWMP, ActorCriticWMPDeployment
from ..algorithms.amp_discriminator import AMPDiscriminator
from ..datasets import IsaacLabAMPLoader
from ..utils.utils import Normalizer

# Import Dreamer components
from dreamer.models import WorldModel
from dreamer.networks import CrossAttentionTerrainEncoder


class WMPRunner:
    ARCHITECTURE_VERSION = "wmp_world_model_pure_v1"

    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 history_length=5,
                 ):

        cfg_obj = train_cfg
        if not isinstance(train_cfg, dict):
            converted_cfg = None
            if hasattr(train_cfg, "to_dict") and callable(train_cfg.to_dict):
                converted_cfg = train_cfg.to_dict()
            if (
                isinstance(converted_cfg, dict)
                and "runner" in converted_cfg
                and "algorithm" in converted_cfg
                and "policy" in converted_cfg
            ):
                train_cfg = converted_cfg
            else:
                train_cfg = self._class_to_dict(cfg_obj)

        # Support both nested cfg format and flattened RslRlBaseRunnerCfg format.
        if "runner" in train_cfg and "algorithm" in train_cfg and "policy" in train_cfg:
            self.cfg = train_cfg["runner"]
            self.alg_cfg = train_cfg["algorithm"]
            self.policy_cfg = train_cfg["policy"]
        else:
            self.cfg = train_cfg
            self.alg_cfg = train_cfg.get("algorithm", {})
            self.policy_cfg = train_cfg.get("policy", {})

        if not self.policy_cfg:
            raise ValueError(
                "WMPRunner received empty policy config. "
                "Please check Hydra/CLI config conversion before initializing ActorCriticWMP."
            )
        self.device = device
        self.env = env
        self.history_length = history_length
        self.enable_world_model = bool(self.cfg.get("enable_world_model", True))
        self.policy_architecture = "wmp"
        self.uses_world_model_feature = self.enable_world_model
        self.architecture_version = self.ARCHITECTURE_VERSION
        
        # Isaac Lab environment properties
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.num_actions

        obs_dict = self.env.get_observations()
        policy_obs = obs_dict["policy"]
        if "critic" in obs_dict.keys():
            privileged_obs = obs_dict["critic"]
        else:
            privileged_obs = policy_obs

        
        self.num_privileged_obs = privileged_obs.shape[-1]
        self.num_obs = policy_obs.shape[-1]
        
        self.dt = float(self.env.unwrapped.cfg.sim.dt * self.env.unwrapped.cfg.decimation)
        
        # In Isaac Lab, we might need to define these manually or extract from config
        self.privileged_dim = getattr(self.env.unwrapped.cfg, "privileged_dim", 53) 
        self.height_dim = getattr(self.env.unwrapped.cfg, "height_dim", 187) 
        self.prop_dim = getattr(self.env.unwrapped.cfg, "prop_dim", 33) 
        self.wm_prop_dim = int(getattr(self.env.unwrapped.cfg, "wm_prop_dim", self.prop_dim))

        self.commands_begin_dim = int(getattr(self.env.unwrapped.cfg, "commands_begin_dim", self.policy_cfg.get("commands_begin_dim", 6)))
        self.policy_cfg.setdefault("commands_begin_dim", self.commands_begin_dim)

        self.history_obs_dim = self.prop_dim - 3
        if self.commands_begin_dim < 0 or self.commands_begin_dim + 3 > self.prop_dim:
            raise ValueError(
                f"Invalid command slice [{self.commands_begin_dim}:{self.commands_begin_dim + 3}] "
                f"for prop_dim={self.prop_dim}."
            )
        if self.enable_world_model and (self.wm_prop_dim <= 0 or self.wm_prop_dim > self.prop_dim):
            raise ValueError(
                f"wm_prop_dim must be in [1, prop_dim], got wm_prop_dim={self.wm_prop_dim}, "
                f"prop_dim={self.prop_dim}."
            )
        if self.enable_world_model and self.commands_begin_dim + 3 > self.wm_prop_dim:
            raise ValueError(
                f"World-model prop must include the command slice "
                f"[{self.commands_begin_dim}:{self.commands_begin_dim + 3}], "
                f"got wm_prop_dim={self.wm_prop_dim}."
            )
        self.terrain_query_extra_dim = self.prop_dim - self.wm_prop_dim if self.enable_world_model else 0

        self.wm_update_interval = self.cfg.get("wm_update_interval", 5) if self.enable_world_model else 1

        self.height_scanner = self.env.unwrapped.scene.sensors.get("height_scanner", None)
        env_cfg = self.env.unwrapped.cfg
        self.height_map_grid_rows = int(getattr(env_cfg, "height_map_grid_rows", 17))
        self.height_map_grid_cols = int(getattr(env_cfg, "height_map_grid_cols", 11))
        self.height_map_channels = int(getattr(env_cfg, "height_map_channels", 1))
        self.height_map_height_range = float(getattr(env_cfg, "height_map_height_range", 1.0))
        self.height_map_flat_dim = (
            self.height_map_grid_rows * self.height_map_grid_cols * self.height_map_channels
        )
        if self.height_dim != self.height_map_flat_dim:
            raise ValueError(
                f"height_dim={self.height_dim} does not match elevation-map shape "
                f"{self.height_map_grid_rows}x{self.height_map_grid_cols}x{self.height_map_channels} "
                f"({self.height_map_flat_dim})."
            )
        if self.num_obs < self.prop_dim + self.height_map_flat_dim:
            raise ValueError(
                f"Policy observation is too small for prop + elevation_map: "
                f"num_obs={self.num_obs}, prop_dim={self.prop_dim}, "
                f"height_dim={self.height_map_flat_dim}."
            )

        if self.enable_world_model:
            self._build_world_model()
        else:
            self._world_model = None
            self.wm_feature_dim = 0
            self.wm_embed_size = 0

        self.history_dim = self.history_length * self.history_obs_dim
        
        actor_critic = ActorCriticWMP(
            num_actor_obs=self.num_obs,
            num_critic_obs=self.num_privileged_obs,
            num_actions=self.num_actions,
            height_dim=self.height_dim,
            privileged_dim=self.privileged_dim,
            history_dim=self.history_dim,
            wm_feature_dim=self.wm_feature_dim,
            **self.policy_cfg
        ).to(self.device)

        # AMP integration
        motion_files = self.cfg.get("amp_motion_files", None)
        if not motion_files:
            raise ValueError(
                "AMPPPO requires at least one AMP motion file. "
                "Check runner.amp_motion_files, for example datasets/mocap_motions/*."
            )
        amp_data = IsaacLabAMPLoader(
            motion_files=motion_files,
            device=self.device,
            time_between_frames=self.dt,
            preload_transitions=True,
            num_preload_transitions=self.cfg.get("amp_num_preload_transitions", 10000),
            motion_fps=self.cfg.get("amp_motion_fps", None),
        )
        self.amp_obs_dim = amp_data.observation_dim
        env_amp_dim = self.env.get_observations()["amp"].shape[-1] if "amp" in self.env.get_observations() else 30
        if self.amp_obs_dim != env_amp_dim:
            raise RuntimeError(
                f"AMP expert obs dim ({self.amp_obs_dim}) does not match "
                f"environment AMP obs dim ({env_amp_dim})."
            )

        amp_normalizer = Normalizer(amp_data.observation_dim)
        amp_reward_coef = self.cfg.get("amp_reward_coef", 1.0)
        amp_task_reward_lerp = self.cfg.get("amp_task_reward_lerp", 1.0)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            amp_reward_coef=amp_reward_coef,
            hidden_layer_sizes=self.cfg.get("amp_discr_hidden_dims", [256, 128]),
            device=self.device,
            task_reward_lerp=amp_task_reward_lerp,
        ).to(self.device)
        print(
            "[INFO]: AMP enabled: "
            f"obs_dim={self.amp_obs_dim}, transition_dim={amp_data.observation_dim * 2}, "
            f"dt={self.dt:.6f}, reward_coef={amp_reward_coef}, "
            f"task_reward_lerp={amp_task_reward_lerp}"
        )

        min_std = self.cfg.get("min_normalized_std", None)
        if min_std is not None:
            min_std = torch.tensor(min_std, device=self.device, dtype=torch.float32)
        self.alg = AMPPPO(actor_critic, discriminator, 
                          amp_data, amp_normalizer,
                          device=self.device,
                          amp_replay_buffer_size=self.cfg.get("amp_replay_buffer_size", 100000),
                          amp_grad_penalty_coef=self.cfg.get("amp_grad_penalty_coef", 10.0),
                          min_std=min_std,
                          **self.alg_cfg)

        ppo_param_ids = {id(param) for param in self.alg.actor_critic.parameters()}
        wm_param_ids = {id(param) for param in self._world_model.parameters()} if self.enable_world_model else set()
        if ppo_param_ids & wm_param_ids:
            raise RuntimeError(
                "PPO and WorldModel optimizers must not share parameter objects."
            )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.gamma = float(self.alg_cfg.get("gamma", 0.99))
        self.lam = float(self.alg_cfg.get("lam", 0.95))
        # init storage
        self.alg.init_storage(
            self.num_envs,
            self.num_steps_per_env,
            [self.num_obs],
            [self.num_privileged_obs],
            [self.num_actions],
            history_dim=self.history_dim,
            wm_feature_dim=self.wm_feature_dim,
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self._set_env_learning_iteration(self.current_learning_iteration)
        self.env.reset()

    def _set_env_learning_iteration(self, iteration):
        """Expose the PPO iteration to environment-side curriculum terms."""
        if not hasattr(self, "env"):
            return
        env = getattr(self.env, "unwrapped", self.env)
        env.ppo_learning_iteration = int(iteration)

    def _get_height_map_observation(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """Return the flattened elevation_map term used by the world model."""
        if policy_obs.shape[-1] < self.height_map_flat_dim:
            raise ValueError(
                f"Policy observation last dim {policy_obs.shape[-1]} is smaller than "
                f"height_map_flat_dim={self.height_map_flat_dim}."
            )
        height_map = policy_obs[..., -self.height_map_flat_dim :]
        if height_map.shape[-1] != self.height_map_flat_dim:
            raise RuntimeError(
                f"Elevation-map slice has shape {height_map.shape}; expected last dim "
                f"{self.height_map_flat_dim}."
            )
        return height_map

    def _get_terminal_policy_observation(self, infos, policy_obs: torch.Tensor):
        """Return reset-free policy observations supplied by the environment, if any."""
        if not isinstance(infos, dict):
            return None
        terminal_obs = None
        for key in ("terminal_observation", "final_observation"):
            if key in infos:
                terminal_obs = infos[key]
                break
        if terminal_obs is None:
            return None
        if isinstance(terminal_obs, dict):
            terminal_obs = terminal_obs.get("policy")
        elif hasattr(terminal_obs, "get") and not isinstance(terminal_obs, torch.Tensor):
            terminal_obs = terminal_obs.get("policy", terminal_obs)
        if terminal_obs is None:
            return None
        terminal_obs = torch.as_tensor(terminal_obs, device=self.device)
        if terminal_obs.shape != policy_obs.shape:
            return None
        return terminal_obs

    @staticmethod
    def _wm_termination_masks(dones, infos):
        """Return true terminations separately from episode truncations."""
        done_mask = dones.bool()
        infos = infos if isinstance(infos, dict) else {}
        truncated_mask = torch.as_tensor(
            infos.get("truncated", infos.get("time_outs", torch.zeros_like(done_mask))),
            device=done_mask.device,
        ).bool()
        terminated_mask = torch.as_tensor(
            infos.get("terminated", done_mask & ~truncated_mask),
            device=done_mask.device,
        ).bool()
        return terminated_mask, truncated_mask

    def _switch_wm_slots(self, reset_env_ids):
        """Start new episode buffers while preserving the just-completed slots."""
        if len(reset_env_ids) == 0:
            return
        self.wm_active_slot[reset_env_ids] ^= 1
        self.wm_buffer_index[reset_env_ids] = 0
        self.wm_dataset_size[
            reset_env_ids, self.wm_active_slot[reset_env_ids]
        ] = 0

    def _get_reward_term_cfg(self, name):
        reward_manager = getattr(self.env.unwrapped, "reward_manager", None)
        if reward_manager is None:
            return None
        try:
            return reward_manager.get_term_cfg(name)
        except Exception:
            return None

    def _get_amp_observations(self, obs_dict=None):
        if obs_dict is not None and "amp" in obs_dict:
            amp_obs = obs_dict["amp"]
            if isinstance(amp_obs, dict):
                amp_obs = next(iter(amp_obs.values()))
            return amp_obs

        robot_data = self.env.unwrapped.scene["robot"].data
        joint_pos = robot_data.joint_pos
        joint_vel = robot_data.joint_vel
        base_lin_vel = robot_data.root_lin_vel_b
        base_ang_vel = robot_data.root_ang_vel_b
        return torch.cat([joint_pos, base_lin_vel, base_ang_vel, joint_vel], dim=-1)

    def _class_to_dict(self, obj):
        if not hasattr(obj, "__dict__") and not isinstance(obj, type):
            return obj
        
        result = {}
        # Get all attributes that don't start with __
        for key in dir(obj):
            if key.startswith("__"):
                continue

            value = getattr(obj, key)
            if isinstance(value, type) or hasattr(value, "__dict__"):
                result[key] = self._class_to_dict(value)
            else:
                result[key] = value
        return result

    def _build_world_model(self):
        # Load dreamer config
        config_path = pathlib.Path(__file__).parent.parent.parent / "dreamer" / "configs.yaml"
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)

        defaults = configs["defaults"]
        
        # We'll use a simple Namespace-like object for config
        class Config:
            pass
        self.wm_config = Config()
        for k, v in defaults.items():
            setattr(self.wm_config, k, v)
        for key in ("wm_barlow_loss_scale", "wm_barlow_lambd"):
            value = self.cfg.get(key)
            if value is not None:
                setattr(self.wm_config, key.removeprefix("wm_"), value)
        self.wm_config.barlow_loss_scale = float(
            getattr(self.wm_config, "barlow_loss_scale", 0.0)
        )
        self.wm_config.barlow_lambd = float(
            getattr(self.wm_config, "barlow_lambd", 5e-4)
        )
        
        self.wm_config.num_actions = self.num_actions * self.wm_update_interval
        self.wm_config.device = self.device
        
        self.wm_config.height_map_grid_rows = self.height_map_grid_rows
        self.wm_config.height_map_grid_cols = self.height_map_grid_cols
        self.wm_config.height_map_channels = self.height_map_channels
        height_map_shape = (self.height_map_flat_dim,)
        obs_shape = {'prop': (self.wm_prop_dim,), 'height_map': height_map_shape}
        terrain_embed_dim = int(self.policy_cfg.get("terrain_embedding_dim", 64))
        terrain_grid_shape = (
            self.height_map_grid_rows,
            self.height_map_grid_cols,
            self.height_map_channels,
        )
        self.terrain_encoder = CrossAttentionTerrainEncoder(
            prop_shape=(self.prop_dim,),
            terrain_shape=(self.height_dim,),
            mha_dim=terrain_embed_dim,
            num_heads=int(self.policy_cfg.get("terrain_attention_heads", 16)),
            act="SiLU",
            norm=True,
            cnn_downsample=True,
            attach_global=False,
            terrain_grid_shape=terrain_grid_shape,
        ).to(self.device)
        self.wm_embed_size = self.wm_prop_dim + self.terrain_encoder.outdim

        self._world_model = WorldModel(
            self.wm_config, obs_shape, embed_size=self.wm_embed_size
        )
        self._world_model = self._world_model.to(self.device)
        self.wm_feature_dim = self.wm_config.dyn_deter

    def encode_terrain(self, observations):
        query_prop = observations[..., : self.prop_dim]
        height_map = observations[..., -self.height_dim :]
        return self.terrain_encoder(query_prop, height_map)

    @staticmethod
    def summarize_macro_dones(dones_by_step, update_interval):
        """Summarize physical-step dones at completed macro boundaries."""
        dones_by_step = torch.as_tensor(dones_by_step, dtype=torch.bool)
        if dones_by_step.ndim != 2:
            raise ValueError("dones_by_step must have shape [steps, envs].")
        pending = torch.zeros(dones_by_step.shape[1], dtype=torch.bool)
        macro_done_count = 0
        first_step_count = 0
        last_step_count = 0
        terminal_written_count = 0
        terminal_dropped_count = 0
        for step, dones in enumerate(dones_by_step):
            pending |= dones
            phase = step % update_interval
            if phase == 0:
                first_step_count += int(dones.sum())
            if phase == update_interval - 1:
                last_step_count += int(dones.sum())
                terminal_written_count += int(dones.sum())
                macro_done_count += int(pending.sum())
                pending.zero_()
            else:
                terminal_dropped_count += int(dones.sum())
        return {
            "env_done_count": int(dones_by_step.sum()),
            "macro_done_count": macro_done_count,
            "done_on_macro_first_step": first_step_count,
            "done_on_macro_last_step": last_step_count,
            "terminal_written_count": terminal_written_count,
            "terminal_dropped_count": terminal_dropped_count,
        }

    def _make_rollout_probe(self, max_samples=1024):
        storage = self.alg.storage
        observations = storage.observations.flatten(0, 1)
        history = storage.history.flatten(0, 1)
        wm_feature = storage.wm_feature.flatten(0, 1)
        count = min(max_samples, observations.shape[0])
        if count == 0:
            return None
        indices = torch.linspace(
            0, observations.shape[0] - 1, count, device=observations.device
        ).long()
        return {
            "observations": observations[indices].detach(),
            "history": history[indices].detach(),
            "wm_feature": wm_feature[indices].detach(),
        }

    def _probe_policy_before_update(self, probe):
        if probe is None:
            return None, {}
        actor_critic = self.alg.actor_critic
        was_training = actor_critic.training
        actor_critic.eval()
        try:
            with torch.inference_mode():
                metrics = {}
                if self.uses_world_model_feature and probe["wm_feature"].shape[-1] > 0:
                    action_wm = actor_critic.act_inference(
                        probe["observations"],
                        probe["history"],
                        probe["wm_feature"],
                    )
                    action_zero = actor_critic.act_inference(
                        probe["observations"],
                        probe["history"],
                        torch.zeros_like(probe["wm_feature"]),
                    )
                    action_delta = action_wm - action_zero
                    wm_encoder_output = actor_critic.wm_feature_encoder(
                        probe["wm_feature"]
                    )
                    metrics = {
                        "Policy/wm_action_delta_l2": action_delta.norm(dim=-1).mean(),
                        "Policy/wm_action_delta_abs_mean": action_delta.abs().mean(),
                        "Policy/wm_action_delta_relative": (
                            action_delta.norm(dim=-1)
                            / (action_wm.norm(dim=-1) + 1e-8)
                        ).mean(),
                        "Policy/wm_encoder_output_norm": wm_encoder_output.norm(dim=-1).mean(),
                        "Policy/wm_encoder_output_std": wm_encoder_output.std(unbiased=False),
                    }
        finally:
            actor_critic.train(was_training)
        return None, metrics

    def _probe_representation_after_update(self, probe, embedding_before):
        return {}

    @staticmethod
    def _mean_metric_values(metric_values):
        means = {}
        for name, values in metric_values.items():
            if not values:
                means[name] = 0.0
            elif isinstance(values[0], torch.Tensor):
                means[name] = torch.stack(values).float().mean().item()
            else:
                means[name] = float(np.mean(values))
        return means

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        obs_dict = self.env.get_observations()
        obs = obs_dict["policy"]
        critic_obs = obs_dict.get("critic", obs)
        
        if hasattr(self.alg, "train_mode"):
            self.alg.train_mode()
        else:
            self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        reward_buffers = {}
        cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_task_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_amp_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_ppo_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        if self.enable_world_model:
            self.init_wm_dataset()
        wm_reward = torch.zeros(self.num_envs, device=self.device) if self.enable_world_model else None
        wm_metrics = None
        
        self.trajectory_history = torch.zeros(
            size=(self.num_envs, self.history_length, self.history_obs_dim), device=self.device
        )
        obs_without_command = torch.zeros((self.num_envs, self.history_obs_dim), device=self.device)
        # A reset has no temporal context.  Repeat the first valid proprioceptive
        # frame instead of exposing zero-padded history as an episode-start token.
        if self.history_length > 0:
            prop_obs = obs[:, : self.prop_dim]
            obs_without_command[:, : self.commands_begin_dim] = prop_obs[:, : self.commands_begin_dim]
            obs_without_command[:, self.commands_begin_dim :] = prop_obs[:, self.commands_begin_dim + 3 :]
            self.trajectory_history[:] = obs_without_command.unsqueeze(1)
        
        # World model state
        wm_latent = None
        wm_is_first = torch.ones(self.num_envs, device=self.device) if self.enable_world_model else None
        wm_feature = torch.zeros((self.num_envs, self.wm_feature_dim), device=self.device)
        wm_action_history = (
            torch.zeros((self.num_envs, self.wm_update_interval, self.num_actions), device=self.device)
            if self.enable_world_model
            else None
        )
        wm_action_order = torch.arange(self.wm_update_interval, device=self.device) if self.enable_world_model else None
        wm_action_cursor = 0
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        all_env_ids_cpu = np.arange(self.num_envs)
        
        use_amp = isinstance(self.alg, AMPPPO)
        if use_amp:
            amp_obs = self._get_amp_observations(obs_dict)
            if hasattr(self, "amp_obs_dim") and amp_obs.shape[-1] != self.amp_obs_dim:
                raise RuntimeError(
                    f"AMP obs dim mismatch: env={amp_obs.shape[-1]} vs dataset={self.amp_obs_dim}"
                )
            print(
                "[INFO]: AMP observation check passed: "
                f"env_obs_dim={amp_obs.shape[-1]}, expert_obs_dim={self.amp_obs_dim}, "
                "order=joint_pos+root_lin_vel_b+root_ang_vel_b+joint_vel+foot_contact"
            )
        
        wm_update_counter = 0
        macro_done_mask = (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if self.enable_world_model
            else None
        )
        macro_corrupt_mask = (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if self.enable_world_model
            else None
        )
        for it in range(self.current_learning_iteration, tot_iter):
            self._set_env_learning_iteration(it)
            start = time.time()
            rssm_metric_values = {
                "RSSM/deter_norm": [],
                "RSSM/deter_mean": [],
                "RSSM/deter_std": [],
                "RSSM/deter_dim_std_mean": [],
                "RSSM/deter_active_dim_ratio": [],
                "RSSM/deter_temporal_delta": [],
                "RSSM/deter_reset_norm": [],
            }
            wm_data_counts = {
                name: torch.zeros((), dtype=torch.long, device=self.device)
                for name in (
                    "env_done_count",
                    "terminated_count",
                    "truncated_count",
                    "macro_done_count",
                    "terminal_written_count",
                    "terminal_proxy_count",
                    "terminal_dropped_off_boundary",
                    "terminal_dropped_corrupt_macro",
                    "done_on_macro_first_step",
                    "done_on_macro_last_step",
                    "record_written_count",
                )
            }
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    wm_update_counter += 1
                    macro_phase = (wm_update_counter - 1) % self.wm_update_interval
                    is_macro_boundary = (
                        wm_update_counter % self.wm_update_interval == 0
                    )
                    history = self.trajectory_history.flatten(1)
                    wmp_obs = obs.clone() 
                    current_critic_obs = critic_obs
                    if use_amp:
                        actions = self.alg.act(
                            wmp_obs,
                            current_critic_obs,
                            amp_obs,
                            history,
                            wm_feature,
                        )
                    else:
                        actions = self.alg.actor_critic.act(
                            wmp_obs,
                            history,
                            wm_feature,
                        )
                        values = self.alg.actor_critic.evaluate(
                            current_critic_obs,
                            wm_feature,
                        ).detach()
                        actions_log_prob = self.alg.actor_critic.get_actions_log_prob(actions).detach()
                        action_mean = self.alg.actor_critic.action_mean.detach()
                        action_sigma = self.alg.actor_critic.action_std.detach()
                    
                    # Env step
                    prev_obs = obs
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    obs = obs_dict["policy"]
                    critic_obs = obs_dict.get("critic", obs)
                    resets = dones.nonzero(as_tuple=False).flatten()
                    has_resets = resets.numel() > 0

                    if self.enable_world_model:
                        done_mask = dones.bool()
                        terminated_mask, truncated_mask = self._wm_termination_masks(
                            dones, infos
                        )
                        done_count = done_mask.sum()
                        wm_data_counts["env_done_count"] += done_count
                        wm_data_counts["terminated_count"] += terminated_mask.sum()
                        wm_data_counts["truncated_count"] += truncated_mask.sum()
                        macro_done_mask |= done_mask
                        if macro_phase == 0:
                            wm_data_counts["done_on_macro_first_step"] += done_count
                        if is_macro_boundary:
                            wm_data_counts["done_on_macro_last_step"] += done_count
                            wm_data_counts["macro_done_count"] += macro_done_mask.sum()
                            macro_done_mask.zero_()
                        else:
                            macro_corrupt_mask |= done_mask
                            wm_data_counts["terminal_dropped_off_boundary"] += (
                                terminated_mask.sum()
                            )
                    
                    if self.enable_world_model:
                        wm_action_history[:, wm_action_cursor] = actions
                        wm_action_cursor = (wm_action_cursor + 1) % self.wm_update_interval
                        wm_reward += rewards

                    # A World Model record describes one completed macro transition:
                    # previous boundary -- action chunk --> current observation.
                    if self.enable_world_model and is_macro_boundary:
                        wm_action_for_obs = wm_action_history[
                            :, (wm_action_order + wm_action_cursor) % self.wm_update_interval
                        ].flatten(1)
                        wm_arrival_obs = obs
                        wm_write_mask = torch.ones(
                            self.num_envs, dtype=torch.bool, device=self.device
                        )
                        wm_write_mask &= ~macro_corrupt_mask
                        terminal_proxy_mask = torch.zeros_like(wm_write_mask)
                        if has_resets:
                            terminal_obs = self._get_terminal_policy_observation(infos, obs)
                            wm_arrival_obs = obs.clone()
                            if terminal_obs is None:
                                wm_arrival_obs[resets] = prev_obs[resets]
                                terminal_proxy_mask[resets] = True
                                if not getattr(self, "_warned_missing_terminal_observation", False):
                                    print(
                                        "[WARN]: World Model is using the pre-step observation as a "
                                        "terminal-state proxy because the environment did not provide "
                                        "terminal_observation or final_observation."
                                    )
                                    self._warned_missing_terminal_observation = True
                            else:
                                wm_arrival_obs[resets] = terminal_obs[resets]
                        height_map_obs = self._get_height_map_observation(wm_arrival_obs)
                        wm_obs = {
                            "prop": wm_arrival_obs[:, :self.wm_prop_dim],
                            "height_map": height_map_obs,
                            "is_first": wm_is_first,
                        }
                        if self.terrain_query_extra_dim:
                            wm_obs["query_extra"] = wm_arrival_obs[
                                :, self.wm_prop_dim : self.prop_dim
                            ]
                        arrival_terrain_embedding = self.encode_terrain(wm_arrival_obs)
                        wm_embed = torch.cat(
                            (arrival_terrain_embedding.detach(), wm_obs["prop"]), dim=-1
                        )
                        previous_wm_feature = wm_feature
                        wm_latent, _ = self._world_model.dynamics.obs_step(
                            wm_latent, wm_action_for_obs, wm_embed, wm_obs["is_first"]
                        )
                        wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                        deter = wm_feature.detach()
                        deter_dim_std = deter.std(dim=0, unbiased=False)
                        rssm_metric_values["RSSM/deter_norm"].append(
                            deter.norm(dim=-1).mean()
                        )
                        rssm_metric_values["RSSM/deter_mean"].append(deter.mean())
                        rssm_metric_values["RSSM/deter_std"].append(
                            deter.std(unbiased=False)
                        )
                        rssm_metric_values["RSSM/deter_dim_std_mean"].append(
                            deter_dim_std.mean()
                        )
                        rssm_metric_values["RSSM/deter_active_dim_ratio"].append(
                            (deter_dim_std > 1e-3).float().mean()
                        )
                        temporal_mask = ~wm_obs["is_first"].bool()
                        temporal_delta = (deter - previous_wm_feature).norm(dim=-1)
                        rssm_metric_values["RSSM/deter_temporal_delta"].append(
                            (temporal_delta * temporal_mask.float()).sum()
                            / temporal_mask.sum().clamp_min(1)
                        )
                        wm_is_first[:] = 0
                        write_env_ids = all_env_ids[wm_write_mask]
                        write_env_ids_cpu = write_env_ids.detach().cpu().numpy()
                        if write_env_ids.numel() > 0:
                            active_slots_cpu = self.wm_active_slot[write_env_ids_cpu]
                            active_slots = torch.as_tensor(
                                active_slots_cpu, device=self.device, dtype=torch.long
                            )
                            buffer_indices_cpu = self.wm_buffer_index[write_env_ids_cpu]
                            buffer_indices = torch.as_tensor(buffer_indices_cpu, device=self.device, dtype=torch.long)
                            if np.any(buffer_indices_cpu >= self.wm_sequence_length):
                                raise RuntimeError("World-model episode buffer overflow.")
                            dataset_index = (write_env_ids, active_slots, buffer_indices)
                            self.wm_dataset["terrain_embed"][dataset_index] = (
                                arrival_terrain_embedding[wm_write_mask].detach()
                            )
                            self.wm_dataset["prop"][dataset_index] = wm_obs["prop"][wm_write_mask]
                            cpu_dataset_index = tuple(
                                torch.as_tensor(values, dtype=torch.long)
                                for values in (
                                    write_env_ids_cpu,
                                    active_slots_cpu,
                                    buffer_indices_cpu,
                                )
                            )
                            self.wm_dataset["height_map"][cpu_dataset_index] = (
                                wm_obs["height_map"][wm_write_mask]
                                .detach()
                                .to(device="cpu", dtype=torch.float16)
                            )
                            if "query_extra" in self.wm_dataset:
                                self.wm_dataset["query_extra"][cpu_dataset_index] = (
                                    wm_obs["query_extra"][wm_write_mask]
                                    .detach()
                                    .to(device="cpu", dtype=torch.float16)
                                )
                            self.wm_dataset["action"][dataset_index] = wm_action_for_obs[wm_write_mask]
                            self.wm_dataset["reward"][dataset_index] = wm_reward[wm_write_mask]
                            self.wm_dataset["is_terminal"][dataset_index] = terminated_mask[wm_write_mask].float()
                            self.wm_dataset["is_first"][dataset_index] = torch.as_tensor(
                                buffer_indices_cpu == 0, device=self.device
                            )

                            self.wm_buffer_index[write_env_ids_cpu] = buffer_indices_cpu + 1
                            self.wm_dataset_size[write_env_ids_cpu, active_slots_cpu] = (
                                self.wm_buffer_index[write_env_ids_cpu]
                            )
                        wm_data_counts["record_written_count"] += wm_write_mask.sum()
                        wm_data_counts["terminal_written_count"] += terminated_mask[
                            wm_write_mask
                        ].sum()
                        wm_data_counts["terminal_proxy_count"] += (
                            terminated_mask & wm_write_mask & terminal_proxy_mask
                        ).sum()
                        wm_data_counts["terminal_dropped_corrupt_macro"] += (
                            terminated_mask & ~wm_write_mask
                        ).sum()
                        wm_reward[:] = 0
                        macro_corrupt_mask.zero_()

                    if use_amp:
                        next_amp_obs = self._get_amp_observations(obs_dict)
                        next_amp_obs_with_term = next_amp_obs.clone()
                        if has_resets:
                            next_amp_obs_with_term[resets] = amp_obs[resets]
                        
                        raw_rewards_for_ppo, raw_amp_rewards, _ = self.alg.discriminator.predict_amp_reward(
                            amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                        )
                        amp_rewards = raw_amp_rewards * self.dt
                        if self.alg.discriminator.task_reward_lerp > 0.0:
                            rewards_for_ppo = rewards + amp_rewards
                        else:
                            rewards_for_ppo = raw_rewards_for_ppo * self.dt
                        
                        self.alg.process_env_step(rewards_for_ppo, dones, infos, next_amp_obs_with_term)
                        amp_obs = next_amp_obs
                    else:
                        self.alg.transition.observations = wmp_obs
                        self.alg.transition.critic_observations = current_critic_obs
                        self.alg.transition.actions = actions
                        self.alg.transition.rewards = rewards
                        self.alg.transition.dones = dones
                        self.alg.transition.values = values
                        self.alg.transition.actions_log_prob = actions_log_prob
                        self.alg.transition.action_mean = action_mean
                        self.alg.transition.action_sigma = action_sigma
                        self.alg.transition.history = history
                        self.alg.transition.wm_feature = wm_feature
                        self.alg.storage.add_transitions(self.alg.transition)
                    
                    # Update history
                    prop_obs = obs[:, : self.prop_dim]
                    obs_without_command[:, : self.commands_begin_dim] = prop_obs[:, : self.commands_begin_dim]
                    obs_without_command[:, self.commands_begin_dim :] = prop_obs[:, self.commands_begin_dim + 3 :]
                    self.trajectory_history[:, :-1] = self.trajectory_history[:, 1:].clone()
                    self.trajectory_history[:, -1] = obs_without_command
                    reset_history_frame = obs_without_command
                    
                    # Handle resets
                    if has_resets:
                        resets_cpu = resets.cpu().numpy()
                        if self.enable_world_model:
                            self._switch_wm_slots(resets_cpu)
                        # ``obs`` already contains each reset environment's new
                        # initial observation.  Fill its whole history window with
                        # that valid frame, matching MGDP's reset convention.
                        if self.history_length > 0:
                            self.trajectory_history[resets] = reset_history_frame[resets].unsqueeze(1)
                        wm_feature[resets] = 0
                        if self.enable_world_model:
                            rssm_metric_values["RSSM/deter_reset_norm"].append(
                                wm_feature[resets].norm(dim=-1).mean()
                            )
                        if self.enable_world_model:
                            wm_action_history[resets] = 0
                            wm_reward[resets] = 0
                            wm_is_first[resets] = 1
                            if wm_latent is not None:
                                wm_latent["deter"][resets] = 0
                                wm_latent["stoch"][resets] = 0

                    if self.log_dir is not None:
                        if use_amp:
                            cur_task_reward_sum += rewards
                            cur_amp_reward_sum += amp_rewards
                            cur_ppo_reward_sum += rewards_for_ppo
                            cur_reward_sum += rewards_for_ppo
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = resets
                        if len(new_ids) > 0:
                            # 使用 detach().cpu().flatten().tolist() 确保得到的是一维纯数字列表
                            rewbuffer.extend(cur_reward_sum[new_ids].detach().cpu().flatten().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids].detach().cpu().flatten().tolist())
                            if use_amp:
                                reward_buffers.setdefault("amp_task_reward", deque(maxlen=100)).extend(
                                    cur_task_reward_sum[new_ids].detach().cpu().flatten().tolist()
                                )
                                reward_buffers.setdefault("amp_reward", deque(maxlen=100)).extend(
                                    cur_amp_reward_sum[new_ids].detach().cpu().flatten().tolist()
                                )
                                reward_buffers.setdefault("amp_ppo_reward", deque(maxlen=100)).extend(
                                    cur_ppo_reward_sum[new_ids].detach().cpu().flatten().tolist()
                                )

                            if 'episode' in infos:
                                ep_infos.append(infos['episode'])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                            
                            # Log per-reward items
                            if "episode" in infos:
                                for key, value in infos["episode"].items():
                                    if key not in reward_buffers:
                                        reward_buffers[key] = deque(maxlen=100)
                                    if isinstance(value, torch.Tensor):
                                        if value.ndim > 0:
                                            reward_buffers[key].extend(value[new_ids].detach().cpu().tolist())
                                        else:
                                            reward_buffers[key].append(value.item())
                                    else:
                                        reward_buffers[key].append(value)
                            elif "log" in infos:
                                for key, value in infos["log"].items():
                                    if key not in reward_buffers:
                                        reward_buffers[key] = deque(maxlen=100)
                                    if isinstance(value, torch.Tensor):
                                        reward_buffers[key].append(value.mean().item())
                                    else:
                                        reward_buffers[key].append(value)

                        cur_reward_sum[new_ids] = 0
                        if use_amp:
                            cur_task_reward_sum[new_ids] = 0
                            cur_amp_reward_sum[new_ids] = 0
                            cur_ppo_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if use_amp:
                    self.alg.compute_returns(obs, critic_obs, wm_feature)
                else:
                    last_values = self.alg.actor_critic.evaluate(
                        critic_obs,
                        wm_feature,
                    ).detach()
                    self.alg.storage.compute_returns(last_values, self.gamma, self.lam)

            rollout_probe = self._make_rollout_probe() if self.log_dir is not None else None
            embedding_before, policy_probe_metrics = self._probe_policy_before_update(
                rollout_probe
            )
            update_out = self.alg.update(clear_storage=False)
            mean_value_loss = update_out[0]
            mean_surrogate_loss = update_out[1]
            if len(update_out) > 2:
                mean_vel_predict_loss = update_out[2]
                mean_amp_loss = update_out[3]
                mean_grad_pen_loss = update_out[4]
                mean_policy_pred = update_out[5]
                mean_expert_pred = update_out[6]
            representation_metrics = self._probe_representation_after_update(
                rollout_probe, embedding_before
            )
            self.alg.storage.clear()
            if self.log_dir is not None:
                for name, value in getattr(self.alg, "last_diagnostics", {}).items():
                    self.writer.add_scalar("PPO/" + name, float(value), it)
                for name, value in policy_probe_metrics.items():
                    self.writer.add_scalar(name, float(value), it)
                for name, value in representation_metrics.items():
                    self.writer.add_scalar(name, float(value), it)
                if self.enable_world_model:
                    for name, value in self._mean_metric_values(
                        rssm_metric_values
                    ).items():
                        self.writer.add_scalar(name, value, it)
                    sequence_lengths = self.wm_dataset_size
                    populated_sequence_lengths = sequence_lengths[sequence_lengths > 0]
                    if populated_sequence_lengths.size == 0:
                        populated_sequence_lengths = np.zeros(1)
                    wm_data_metrics = {
                        "WM_data/env_done_count": wm_data_counts["env_done_count"],
                        "WM_data/terminated_count": wm_data_counts["terminated_count"],
                        "WM_data/truncated_count": wm_data_counts["truncated_count"],
                        "WM_data/macro_done_count": wm_data_counts["macro_done_count"],
                        "WM_data/terminal_written_count": wm_data_counts["terminal_written_count"],
                        "WM_data/terminal_proxy_count": wm_data_counts["terminal_proxy_count"],
                        "WM_data/terminal_dropped_off_boundary": wm_data_counts[
                            "terminal_dropped_off_boundary"
                        ],
                        "WM_data/terminal_dropped_corrupt_macro": wm_data_counts[
                            "terminal_dropped_corrupt_macro"
                        ],
                        "WM_data/terminal_dropped_count": (
                            wm_data_counts["terminal_dropped_off_boundary"]
                            + wm_data_counts["terminal_dropped_corrupt_macro"]
                        ),
                        "WM_data/done_on_macro_first_step": wm_data_counts["done_on_macro_first_step"],
                        "WM_data/done_on_macro_last_step": wm_data_counts["done_on_macro_last_step"],
                        "WM_data/terminal_target_ratio": (
                            wm_data_counts["terminal_written_count"]
                            / wm_data_counts["record_written_count"].clamp_min(1)
                        ),
                        "WM_data/sequence_length_mean": float(np.mean(populated_sequence_lengths)),
                        "WM_data/sequence_length_min": float(np.min(populated_sequence_lengths)),
                        "WM_data/sequence_length_max": float(np.max(populated_sequence_lengths)),
                    }
                    for name, value in wm_data_metrics.items():
                        self.writer.add_scalar(name, float(value), it)
            stop = time.time()
            ppo_learn_time = stop - start
            
            # Train world model
            wm_learn_time = 0.0
            train_start_steps = (
                getattr(self.wm_config, "train_start_steps", 1000)
                if self.enable_world_model
                else 0
            )
            wm_dataset_total = float(np.sum(self.wm_dataset_size)) if self.enable_world_model else 0.0
            wm_train_ready = self.enable_world_model and wm_dataset_total > train_start_steps
            if self.log_dir is not None and self.enable_world_model:
                self.writer.add_scalar("World_model/dataset_size", wm_dataset_total, it)
                self.writer.add_scalar("World_model/train_start_steps", train_start_steps, it)
                self.writer.add_scalar("World_model/train_ready", float(wm_train_ready), it)
            if wm_train_ready:
                wm_start = time.time()
                wm_metrics = self.train_world_model()
                wm_learn_time = time.time() - wm_start
                if self.log_dir is not None and wm_metrics:
                    for name, values in wm_metrics.items():
                        writer_name = name if name.startswith("WM_batch/") else "World_model/" + name
                        self.writer.add_scalar(writer_name, float(np.mean(values)), it)
            learn_time = ppo_learn_time + wm_learn_time

            if self.log_dir is not None:
                self.log(locals())
                if it in {0, 200, 500, 800, 1000, 1200, 1500, 2000, 2500}:
                    self.writer.add_scalar("Diagnostics/milestone", 1.0, it)
                    self.writer.flush()
            ep_infos.clear()
            if it % self.save_interval == 0:
                # The checkpoint contains the next iteration to execute. The
                # filename keeps the conventional completed-iteration index.
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), iteration=it + 1)

        self.current_learning_iteration = tot_iter
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def init_wm_dataset(self):
        max_episode_length = getattr(self.env.unwrapped.cfg, "episode_length_steps", 1000)
        wm_sequence_length = int(max_episode_length / self.wm_update_interval) + 3
        self.wm_sequence_length = wm_sequence_length
        terrain_embedding_dim = self.terrain_encoder.outdim
        self.wm_dataset = {
            "terrain_embed": torch.zeros(
                (self.num_envs, 2, wm_sequence_length, terrain_embedding_dim),
                device=self.device,
            ),
            "prop": torch.zeros(
                (self.num_envs, 2, wm_sequence_length, self.wm_prop_dim),
                device=self.device,
            ),
            "action": torch.zeros(
                (self.num_envs, 2, wm_sequence_length, self.num_actions * self.wm_update_interval),
                device=self.device,
            ),
            "reward": torch.zeros((self.num_envs, 2, wm_sequence_length), device=self.device),
            "is_terminal": torch.zeros((self.num_envs, 2, wm_sequence_length), device=self.device),
            "is_first": torch.zeros(
                (self.num_envs, 2, wm_sequence_length), dtype=torch.bool, device=self.device
            ),
        }
        self.wm_cpu_dataset_keys = {"height_map"}
        self.wm_dataset["height_map"] = torch.zeros(
            (self.num_envs, 2, wm_sequence_length, self.height_map_flat_dim),
            dtype=torch.float16,
            device="cpu",
        )
        if self.terrain_query_extra_dim:
            self.wm_cpu_dataset_keys.add("query_extra")
            self.wm_dataset["query_extra"] = torch.zeros(
                (
                    self.num_envs,
                    2,
                    wm_sequence_length,
                    self.terrain_query_extra_dim,
                ),
                dtype=torch.float16,
                device="cpu",
            )
        self.wm_dataset_size = np.zeros((self.num_envs, 2), dtype=np.int32)
        self.wm_active_slot = np.zeros(self.num_envs, dtype=np.int8)
        self.wm_buffer_index = np.zeros(self.num_envs, dtype=np.int32)

    def train_world_model(self):
        wm_metrics = {}
        
        # Determine how many steps to train
        train_steps = getattr(self.wm_config, "train_steps_per_iter", 100)
        batch_size = getattr(self.wm_config, "batch_size", 16)
        batch_length = getattr(self.wm_config, "batch_length", 50)
        
        for i in range(train_steps):
            if np.sum(self.wm_dataset_size) == 0:
                continue
                
            flat_sizes = self.wm_dataset_size.reshape(-1)
            p = flat_sizes / np.sum(flat_sizes)
            batch_flat_idx = np.random.choice(
                len(flat_sizes), batch_size, replace=True, p=p
            )
            batch_env_idx, batch_slot_idx = np.unravel_index(
                batch_flat_idx, self.wm_dataset_size.shape
            )
            
            # Check min dataset size for chosen indices
            selected_sizes = self.wm_dataset_size[batch_env_idx, batch_slot_idx]
            min_size = int(selected_sizes.min())
            current_batch_length = min(min_size, batch_length)
            
            if current_batch_length <= 1:
                continue
                
            batch_end_idx = [
                np.random.randint(current_batch_length, int(size) + 1)
                for size in selected_sizes
            ]

            batch_env_cpu = torch.as_tensor(
                batch_env_idx.copy(), dtype=torch.long
            )
            batch_slot_cpu = torch.as_tensor(
                batch_slot_idx.copy(), dtype=torch.long
            )
            batch_end_cpu = torch.as_tensor(batch_end_idx, dtype=torch.long)
            batch_time_cpu = (
                batch_end_cpu[:, None]
                - current_batch_length
                + torch.arange(current_batch_length, dtype=torch.long)[None, :]
            )
            target_device = torch.device(self.device)
            if target_device.type == "cuda":
                batch_env_device = batch_env_cpu.to(target_device)
                batch_slot_device = batch_slot_cpu.to(target_device)
                batch_time_device = batch_time_cpu.to(target_device)
            else:
                batch_env_device = batch_env_cpu
                batch_slot_device = batch_slot_cpu
                batch_time_device = batch_time_cpu

            batch_data = {}
            for k, v in self.wm_dataset.items():
                if v.device.type == "cpu":
                    value = v[
                        batch_env_cpu[:, None],
                        batch_slot_cpu[:, None],
                        batch_time_cpu,
                    ]
                    if target_device.type == "cuda":
                        value = value.pin_memory().to(
                            target_device, non_blocking=True
                        )
                else:
                    value = v[
                        batch_env_device[:, None],
                        batch_slot_device[:, None],
                        batch_time_device,
                    ]
                    if value.device != target_device:
                        value = value.to(target_device)
                batch_data[k] = value

            batch_metrics = {
                "WM_batch/terminal_ratio": batch_data["is_terminal"].float().mean(),
                "WM_batch/is_first_ratio": batch_data["is_first"].float().mean(),
                "WM_batch/reward_mean": batch_data["reward"].float().mean(),
                "WM_batch/reward_std": batch_data["reward"].float().std(unbiased=False),
                "WM_batch/batch_length": float(current_batch_length),
            }

            wm_embed = torch.cat(
                (batch_data["terrain_embed"], batch_data["prop"]), dim=-1
            )

            post, context, mets = self._world_model._train(batch_data, wm_embed)
            for name, value in {**mets, **batch_metrics}.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().mean().item()
                elif isinstance(value, np.ndarray):
                    value = float(np.mean(value))
                else:
                    value = float(value)
                wm_metrics.setdefault(name, []).append(value)

        return wm_metrics

    def _mean_log_value(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(self.device)
            if tensor.numel() == 0:
                return None
            return torch.mean(tensor.float())
        try:
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float)
        except (TypeError, ValueError):
            return None
        if tensor.numel() == 0:
            return None
        return torch.mean(tensor.float())

    def _curriculum_log_values(self):
        values = {}
        terrain = getattr(self.env.unwrapped.scene, "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is not None:
            terrain_levels = terrain_levels.detach().float()
            if terrain_levels.numel() > 0:
                values["Curriculum/terrain_levels"] = terrain_levels.mean()
        return values

    @staticmethod
    def _display_log_key(key):
        if key.startswith("Episode_Reward/"):
            return "Episode_Reward/" + key.removeprefix("Episode_Reward/")
        if key.startswith("Episode_Termination/"):
            return "Episode_Termination/" + key.removeprefix("Episode_Termination/")
        if key.startswith("Metrics/"):
            return "Metrics/" + key.removeprefix("Metrics/")
        if key.startswith("Curriculum/"):
            return "Curriculum/" + key.removeprefix("Curriculum/")
        if key.startswith("Episode/"):
            return "Episode/" + key.removeprefix("Episode/")
        return key

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        
        ep_values = {}
        if locs["ep_infos"]:
            all_keys = sorted({key for ep_info in locs["ep_infos"] for key in ep_info.keys()})
            for key in all_keys:
                values = []
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    value = self._mean_log_value(ep_info[key])
                    if value is not None:
                        values.append(value)
                if values:
                    ep_values[key] = torch.mean(torch.stack(values))

        ep_values.update(self._curriculum_log_values())
        display_ep_values = []
        for key, value in ep_values.items():
            writer_key = key if "/" in key else "Episode/" + key
            self.writer.add_scalar(writer_key, value, locs["it"])
            display_ep_values.append((self._display_log_key(writer_key), value))

        ep_string = ""
        for key, value in sorted(display_ep_values, key=lambda item: item[0]):
            ep_string += f"{(key + ':'):>{pad}} {value.item():.4f}\n"
        
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.num_envs / iteration_time)

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        self.writer.add_scalar('Perf/ppo_learning_time', locs.get('ppo_learn_time', locs['learn_time']), locs['it'])
        self.writer.add_scalar('Perf/wm_learning_time', locs.get('wm_learn_time', 0.0), locs['it'])

        if 'mean_amp_loss' in locs:
            self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
            self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
            self.writer.add_scalar('Loss/AMP_mean_policy_pred', locs['mean_policy_pred'], locs['it'])
            self.writer.add_scalar('Loss/AMP_mean_expert_pred', locs['mean_expert_pred'], locs['it'])
            self.writer.add_scalar('Loss/vel_predict', locs['mean_vel_predict_loss'], locs['it'])

        if 'mean_wm_reward' in locs:
            self.writer.add_scalar('World_model/mean_reward', locs['mean_wm_reward'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', np.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', np.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', np.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', np.mean(locs['lenbuffer']), self.tot_time)
            
            # Log per-reward items
            for key, value in locs['reward_buffers'].items():
                if "/" in key:
                    continue
                if len(value) > 0:
                    self.writer.add_scalar(f'Reward/{key}', np.mean(value), locs['it'])
            if "amp_reward" in locs["reward_buffers"] and len(locs["reward_buffers"]["amp_reward"]) > 0:
                mean_task_reward = np.mean(locs["reward_buffers"].get("amp_task_reward", [0.0]))
                mean_amp_reward = np.mean(locs["reward_buffers"]["amp_reward"])
                mean_ppo_reward = np.mean(locs["reward_buffers"].get("amp_ppo_reward", locs["rewbuffer"]))
                self.writer.add_scalar("Amp/mean_task_reward", mean_task_reward, locs["it"])
                self.writer.add_scalar("Amp/mean_reward", mean_amp_reward, locs["it"])
                self.writer.add_scalar("Amp/mean_ppo_reward", mean_ppo_reward, locs["it"])
                self.writer.add_scalar("Amp/amp_fraction", mean_amp_reward / (mean_ppo_reward + 1.0e-8), locs["it"])

        str_title = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_title.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s, ppo {locs.get('ppo_learn_time', locs['learn_time']):.3f}s, wm {locs.get('wm_learn_time', 0.0):.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {np.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {np.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_title.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s, ppo {locs.get('ppo_learn_time', locs['learn_time']):.3f}s, wm {locs.get('wm_learn_time', 0.0):.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
        
        if 'mean_amp_loss' in locs:
            log_string += (f"""{'Vel predict loss:':>{pad}} {locs['mean_vel_predict_loss']:.4f}\n"""
                           f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                           f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                           f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                           f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n""")
            if "amp_reward" in locs["reward_buffers"] and len(locs["reward_buffers"]["amp_reward"]) > 0:
                mean_task_reward = np.mean(locs["reward_buffers"].get("amp_task_reward", [0.0]))
                mean_amp_reward = np.mean(locs["reward_buffers"]["amp_reward"])
                mean_ppo_reward = np.mean(locs["reward_buffers"].get("amp_ppo_reward", locs["rewbuffer"]))
                amp_fraction = mean_amp_reward / (mean_ppo_reward + 1.0e-8)
                log_string += (f"""{'Mean task reward:':>{pad}} {mean_task_reward:.4f}\n"""
                               f"""{'Mean AMP reward:':>{pad}} {mean_amp_reward:.4f}\n"""
                               f"""{'AMP reward fraction:':>{pad}} {amp_fraction:.3f}\n""")

        eta_seconds = self.tot_time / (locs['it'] + 1) * (
            locs['num_learning_iterations'] - locs['it']
        )

        def format_time_hours(seconds):
            hours = seconds / 3600.0
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            if h >= 24:
                d = h // 24
                rem_h = h % 24
                return f"{hours:.2f}h ({d}d {rem_h}h {m}m)"
            elif h > 0:
                return f"{hours:.2f}h ({h}h {m}m)"
            else:
                s = int(seconds % 60)
                return f"{hours:.2f}h ({m}m {s}s)"

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {format_time_hours(self.tot_time)}\n"""
                       f"""{'ETA:':>{pad}} {format_time_hours(eta_seconds)}\n""")
        print(log_string)

    def save(self, path, iteration=None):
        next_iteration = self.current_learning_iteration if iteration is None else int(iteration)
        enable_world_model = getattr(self, "enable_world_model", True)
        architecture_version = getattr(self, "architecture_version", self.ARCHITECTURE_VERSION)
        save_dict = {
            'architecture_version': architecture_version,
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            # ``iter`` is the next loop index to execute after loading.
            'iter': next_iteration,
            'completed_iteration': next_iteration - 1,
            'tot_timesteps': int(self.tot_timesteps),
            'tot_time': float(self.tot_time),
            'num_envs': int(self.num_envs),
            'num_steps_per_env': int(self.num_steps_per_env),
        }
        if enable_world_model:
            save_dict['world_model_dict'] = self._world_model.state_dict()
            save_dict['terrain_encoder_dict'] = self.terrain_encoder.state_dict()
        if hasattr(self.alg, "discriminator"):
            save_dict["discriminator_state_dict"] = self.alg.discriminator.state_dict()
        if hasattr(self.alg, "amp_normalizer") and self.alg.amp_normalizer is not None:
            save_dict["amp_normalizer"] = {
                "mean": self.alg.amp_normalizer.mean,
                "var": self.alg.amp_normalizer.var,
                "count": self.alg.amp_normalizer.count,
            }
        torch.save(save_dict, path)
        print(
            f"[INFO]: Saved checkpoint: {path} "
            f"(completed_iteration={next_iteration - 1}, "
            f"next_iteration={next_iteration}, total_timesteps={self.tot_timesteps})"
        )

    def load(self, path):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        architecture_version = loaded_dict.get('architecture_version')
        expected_architecture = getattr(self, "architecture_version", self.ARCHITECTURE_VERSION)
        enable_world_model = getattr(self, "enable_world_model", True)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if enable_world_model:
            if 'world_model_dict' in loaded_dict:
                self._world_model.load_state_dict(loaded_dict['world_model_dict'])
            if 'terrain_encoder_dict' in loaded_dict:
                self.terrain_encoder.load_state_dict(loaded_dict['terrain_encoder_dict'])
        self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        if hasattr(self.alg, "discriminator") and "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if hasattr(self.alg, "amp_normalizer") and "amp_normalizer" in loaded_dict and self.alg.amp_normalizer is not None:
            norm = loaded_dict["amp_normalizer"]
            self.alg.amp_normalizer.mean = norm["mean"]
            self.alg.amp_normalizer.var = norm["var"]
            self.alg.amp_normalizer.count = norm["count"]
        stored_iteration = int(loaded_dict.get('iter', 0))
        checkpoint_index = None
        checkpoint_stem = pathlib.Path(path).stem
        if checkpoint_stem.startswith("model_"):
            try:
                checkpoint_index = int(checkpoint_stem.removeprefix("model_"))
            except ValueError:
                pass

        # Older periodic checkpoints wrote the training-start iteration into
        # every file. Recover their next iteration from model_<index>.pt.
        if (
            "completed_iteration" not in loaded_dict
            and checkpoint_index is not None
            and stored_iteration != checkpoint_index
        ):
            self.current_learning_iteration = checkpoint_index + 1
            print(
                "[WARNING]: Legacy checkpoint iteration metadata does not match its filename: "
                f"stored_iter={stored_iteration}, filename_index={checkpoint_index}. "
                f"Resuming from iteration {self.current_learning_iteration}."
            )
        else:
            self.current_learning_iteration = stored_iteration

        self._set_env_learning_iteration(self.current_learning_iteration)
        self.tot_timesteps = int(loaded_dict.get('tot_timesteps', 0))
        self.tot_time = float(loaded_dict.get('tot_time', 0.0))
        print(
            f"[INFO]: Loaded checkpoint: {path} "
            f"(next_iteration={self.current_learning_iteration}, "
            f"total_timesteps={self.tot_timesteps}, total_time={self.tot_time:.2f}s)"
        )

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def export_policy_to_jit(self, path, filename="policy.pt"):
        """Export only the action-producing graph as a standalone TorchScript module."""
        os.makedirs(path, exist_ok=True)
        actor_critic = self.alg.actor_critic.to(self.device).eval()
        policy_obs = self.env.get_observations()["policy"]
        example_obs = policy_obs.to(self.device)
        with torch.inference_mode():
            deployment_policy = ActorCriticWMPDeployment(actor_critic).to(self.device).eval()
            batch_size = example_obs.shape[0]
            example_history = torch.zeros(
                (batch_size, self.history_dim), device=self.device, dtype=example_obs.dtype
            )
            example_wm_feature = torch.zeros(
                (batch_size, self.wm_feature_dim), device=self.device, dtype=example_obs.dtype
            )
            scripted_policy = torch.jit.trace(
                deployment_policy,
                (example_obs, example_history, example_wm_feature),
                strict=False,
            )
            input_description = (
                f"observations[{self.num_obs}], history[{self.history_dim}], "
                f"wm_feature[{self.wm_feature_dim}]"
            )
            scripted_policy.save(os.path.join(path, filename))
        print(
            f"[INFO] Exported deployment TorchScript policy to "
            f"{os.path.join(path, filename)} "
            f"(inputs: {input_description}; outputs: actions[{self.num_actions}])"
        )
        return scripted_policy
