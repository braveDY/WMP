# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent trained with WMP on Isaac Lab."""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and WMP.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps before exiting.")
parser.add_argument("--task", type=str, default="Velocity-Rough-WMP-Play", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--terrain",
    type=str,
    default="all",
    choices=["all", "flat", "stair", "slope", "gap", "pit", "rough"],
    help="Select terrain type to evaluate on: all, flat, stair, slope, gap, pit, rough (matching master play.py).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

project_root = Path(__file__).resolve().parents[2]
local_wmp_path = project_root / "source" / "wmp"
if local_wmp_path.is_dir() and str(local_wmp_path) not in sys.path:
    sys.path.insert(0, str(local_wmp_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rsl_rl_compat import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import wmp  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.runners.wmp_runner import WMPRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        load_run = agent_cfg.load_run if agent_cfg.load_run else ".*"
        load_checkpoint = agent_cfg.load_checkpoint if agent_cfg.load_checkpoint else r"model_.*\.pt"
        resume_path = get_checkpoint_path(log_root_path, load_run, load_checkpoint)
    if hasattr(env_cfg.scene, "terrain"):
        if args_cli.terrain == "flat":
            env_cfg.scene.terrain.terrain_type = "plane"
            env_cfg.scene.terrain.terrain_generator = None
        elif args_cli.terrain != "all" and getattr(env_cfg.scene.terrain, "terrain_generator", None) is not None:
            tg = env_cfg.scene.terrain.terrain_generator
            keyword_map = {
                "stair": "stair",
                "slope": "slope",
                "gap": "gap",
                "pit": "pit",
                "rough": "rough",
            }
            target_kw = keyword_map[args_cli.terrain]
            matched = {k: v for k, v in tg.sub_terrains.items() if target_kw in k}
            if matched:
                tg.sub_terrains = matched
                for v in tg.sub_terrains.values():
                    v.proportion = 1.0 / len(tg.sub_terrains)
            print(f"[INFO]: Selected terrain types: {list(tg.sub_terrains.keys())}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(resume_path), "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    if agent_cfg.class_name == "WMPRunner":
        runner = WMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        from rsl_rl.utils import resolve_callable
        runner_class = resolve_callable(agent_cfg.class_name)
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)
    class WMPInferencePolicyWrapper:
        def __init__(self, r):
            self.r = r
            self.actor_critic = r.alg.actor_critic.eval()
            self.world_model = r._world_model.eval() if r.enable_world_model else None
            self.terrain_encoder = r.terrain_encoder.eval() if r.enable_world_model else None
            self.device = r.device
            self.num_envs = env.num_envs
            self.num_actions = r.num_actions
            self.wm_update_interval = r.wm_update_interval
            self.prop_dim = r.prop_dim
            self.wm_prop_dim = r.wm_prop_dim
            self.commands_begin_dim = r.commands_begin_dim
            self.history_length = r.history_length
            self.history_obs_dim = r.history_obs_dim
            self.height_map_flat_dim = r.height_map_flat_dim

            self.trajectory_history = torch.zeros(
                (self.num_envs, self.history_length, self.history_obs_dim), device=self.device
            )
            self.wm_latent = None
            self.wm_is_first = torch.ones(self.num_envs, device=self.device) if r.enable_world_model else None
            self.wm_feature = torch.zeros((self.num_envs, r.wm_feature_dim), device=self.device)
            self.wm_action_history = (
                torch.zeros((self.num_envs, self.wm_update_interval, self.num_actions), device=self.device)
                if r.enable_world_model
                else None
            )
            self.wm_action_order = torch.arange(self.wm_update_interval, device=self.device) if r.enable_world_model else None
            self.wm_action_cursor = 0
            self.wm_update_counter = 0

        def __call__(self, obs):
            prop_obs = obs[:, : self.prop_dim]
            obs_without_cmd = torch.cat(
                [prop_obs[:, : self.commands_begin_dim], prop_obs[:, self.commands_begin_dim + 3 :]], dim=-1
            )
            self.trajectory_history = torch.cat(
                [self.trajectory_history[:, 1:], obs_without_cmd.unsqueeze(1)], dim=1
            )
            history = self.trajectory_history.flatten(1)

            self.wm_update_counter += 1
            if self.r.enable_world_model and (self.wm_update_counter % self.wm_update_interval == 0):
                arrival_terrain_emb = self.r.encode_terrain(obs)
                wm_obs = {"prop": prop_obs[:, : self.wm_prop_dim], "is_first": self.wm_is_first}
                wm_embed = torch.cat((arrival_terrain_emb.detach(), wm_obs["prop"]), dim=-1)
                wm_action_for_obs = self.wm_action_history[
                    :, (self.wm_action_order + self.wm_action_cursor) % self.wm_update_interval
                ].flatten(1)
                self.wm_latent, _ = self.world_model.dynamics.obs_step(
                    self.wm_latent, wm_action_for_obs, wm_embed, wm_obs["is_first"]
                )
                self.wm_feature = self.world_model.dynamics.get_deter_feat(self.wm_latent)
                self.wm_is_first[:] = 0

            actions = self.actor_critic.act_inference(obs, history, self.wm_feature)
            if self.r.enable_world_model:
                self.wm_action_history[:, self.wm_action_cursor] = actions
                self.wm_action_cursor = (self.wm_action_cursor + 1) % self.wm_update_interval
            return actions

    if agent_cfg.class_name == "WMPRunner":
        policy = WMPInferencePolicyWrapper(runner)
    else:
        policy = runner.get_inference_policy(device=env.device)

    obs_dict = env.get_observations()
    obs = obs_dict["policy"]

    total_steps = args_cli.num_steps or 1000
    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs_dict, _, _, _ = env.step(actions)
            obs = obs_dict["policy"]


    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
