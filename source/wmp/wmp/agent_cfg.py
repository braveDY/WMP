from __future__ import annotations

import glob
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


MOTION_FILES = sorted(glob.glob("datasets/mocap_motions/*.txt"))

EXPERIMENT_NAME = "unitree_a1_wmp"
RUN_NAME = "WMP_A1"
MAX_ITERATIONS = 20000
SAVE_ITERATIONS = 1000


@configclass
class PPOAlgorithmCfg:
    """Shared plain-PPO hyperparameters."""

    entropy_coef: float = 0.008
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class AMPPPOAlgorithmCfg(PPOAlgorithmCfg):
    class_name: str = "AMPPPO"


@configclass
class WMPPolicyCfg:
    architecture: str = "wmp"
    init_noise_std: float = 1.0
    encoder_hidden_dims: list = [256, 128]
    wm_encoder_hidden_dims: list = [64, 64]
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    latent_dim: int = 35
    wm_latent_dim: int = 32
    activation: str = "elu"
    commands_begin_dim: int = 6
    wm_prop_dim: int = 33
    terrain_grid_shape: list = [25, 17, 3]
    terrain_embedding_dim: int = 64
    terrain_attention_heads: int = 16
    terrain_cnn_downsample: bool = True
    terrain_attach_global: bool = False


@configclass
class WMPRunnerSubCfg:
    enable_world_model: bool = True
    algorithm_class_name: str = "AMPPPO"
    experiment_name: str = EXPERIMENT_NAME
    run_name: str = RUN_NAME
    max_iterations: int = MAX_ITERATIONS
    save_interval: int = SAVE_ITERATIONS
    num_steps_per_env: int = 24
    wm_update_interval: int = 5
    amp_motion_files: list = MOTION_FILES
    amp_reward_coef: float = 2.0
    amp_num_preload_transitions: int = 100000
    amp_task_reward_lerp: float = 0.3
    amp_discr_hidden_dims: list = [1024, 512]
    amp_replay_buffer_size: int = 100000
    amp_grad_penalty_coef: float = 10.0
    min_normalized_std: list = [0.05, 0.02, 0.05] * 4


@configclass
class UnitreeA1WMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "WMPRunner"
    experiment_name: str = EXPERIMENT_NAME
    run_name: str = RUN_NAME
    seed: int = 42
    max_iterations: int = MAX_ITERATIONS
    save_interval: int = SAVE_ITERATIONS
    resume: bool = False
    load_run = None
    load_checkpoint = None

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "amp": ["amp"],
    }

    policy = WMPPolicyCfg()
    algorithm = AMPPPOAlgorithmCfg()
    runner = WMPRunnerSubCfg()
