# World Model-based Perception (WMP) for Visual Legged Locomotion on NVIDIA IsaacLab

[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.x-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1-silver.svg)](https://developer.nvidia.com/isaac-sim)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)

This repository provides the official modern **NVIDIA IsaacLab (2.x)** refactored implementation of:
> **World Model-based Perception for Visual Legged Locomotion (WMP)**  
> [Hang Lai](https://apex.sjtu.edu.cn/members/laihang@apexlab.org), [Jiahang Cao](https://apex.sjtu.edu.cn/members/jhcao@apexlab.org), [JiaFeng Xu](https://scholar.google.com/citations?user=GPmUxtIAAAAJ), [Hongtao Wu](https://scholar.google.com/citations?user=7u0TYgIAAAAJ), [Yunfeng Lin](https://apex.sjtu.edu.cn/members/yflin@apexlab.org), [Tao Kong](https://www.taokong.org/), [Yong Yu](https://scholar.google.com.hk/citations?user=-84M1m0AAAAJ), [Weinan Zhang](https://wnzhang.net/)  
> *arXiv preprint arXiv:2409.16784*  
> [🌐 Project Website](https://wmp-loco.github.io/) | [📄 Paper](https://arxiv.org/abs/2409.16784)

Legacy IsaacGym Preview 4 and `legged_gym` dependencies have been completely replaced with IsaacLab's modular `ManagerBasedRLEnv` architecture. The framework combines **DreamerV3 Recurrent State-Space Models (RSSM)**, **Adversarial Motion Priors (AMP)**, and **3D Cross-Attention Elevation Encoding** for high-performance legged obstacle traversal.

---

## Key Highlights

- **DreamerV3 RSSM World Model**: Captures environment dynamics in latent space (512-dim deterministic and categorical stochastic representations) with macro-action sequence predictions.
- **Adversarial Motion Priors (AMP)**: Enforces natural and agile gaits learned from real quadruped motion capture demonstrations (Unitree A1, 30-dim kinematics: joint positions, velocities, and base twist), eliminating unphysical locomotion behaviors.
- **Cross-Attention 3D Elevation Perception**: A yaw-aligned 25×17 RayCaster scanner extracts 1275-dim `[x, y, z]` spatial point coordinate grids, processed by a multi-head cross-attention encoder.
- **Standalone `rsl_rl` RL Library**: Maintained locally with end-to-end support for PPO actor-critic optimization, AMP discriminator updates with WGAN gradient penalty, and RSSM world-model representation training.

---

## Project Structure

```text
WMP/
├── source/wmp/                 # IsaacLab extension package
│   ├── config/extension.toml   # Omniverse Kit extension metadata
│   ├── setup.py / pyproject.toml
│   └── wmp/
│       ├── env_cfg.py          # Scene, A1 robot articulation, sensors & environment
│       ├── agent_cfg.py        # WMPRunner configuration (PPO, AMP, Dreamer hyperparameters)
│       ├── mdp/                # Modular MDP terms (observations, rewards, terminations, curriculum)
│       └── terrains/           # Diverse rough and obstacle terrain generators
├── rsl_rl/                     # Local reinforcement learning algorithms (WMPRunner, AMPPPO, ActorCriticWMP)
├── dreamer/                    # DreamerV3 RSSM architecture and configuration
├── datasets/mocap_motions/     # Unitree A1 motion capture clips for AMP
└── scripts/                    # Command-line entry points
    ├── list_envs.py            # Task listing utility
    ├── export_terrain.py       # 3D terrain export utility
    └── rsl_rl/
        ├── train.py            # Training pipeline for WMP + AMP
        └── play.py             # Inference and checkpoint evaluation
```

---

## Getting Started

### 1. Installation

Activate your Conda environment with IsaacLab / Isaac Sim 5.1 installed, then install the `wmp` extension in editable mode:

```bash
conda activate isaaclab
pip install -e source/wmp --no-deps
```

### 2. Verify Available Tasks

List registered environments to verify installation:

```bash
python scripts/list_envs.py
```

Registered tasks include:
- `Velocity-Rough-WMP-Train`: Multi-environment parallel parkour and rough terrain training.
- `Velocity-Rough-WMP-Play`: Lightweight evaluation and visualization environment.

### 3. Training

Run headless training with parallel environments (e.g., 16 environments for smoke testing or 4096 for full training):

```bash
# Lightweight smoke test / debugging
python scripts/rsl_rl/train.py --task Velocity-Rough-WMP-Train --num_envs 16 --headless

# Full-scale training
python scripts/rsl_rl/train.py --task Velocity-Rough-WMP-Train --num_envs 4096 --headless
```

Experiment logs, configuration snapshots, and policy checkpoints are saved to:
`logs/rsl_rl/unitree_a1_wmp/<timestamp>_WMP_A1/`

### 4. Evaluation & Play

Evaluate a trained model checkpoint in the evaluation environment:

```bash
python scripts/rsl_rl/play.py \
  --task Velocity-Rough-WMP-Play \
  --num_envs 2 \
  --checkpoint logs/rsl_rl/unitree_a1_wmp/<RUN_DIR>/model_<ITER>.pt
```

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{lai2024world,
  title={World Model-based Perception for Visual Legged Locomotion},
  author={Lai, Hang and Cao, Jiahang and Xu, Jiafeng and Wu, Hongtao and Lin, Yunfeng and Kong, Tao and Yu, Yong and Zhang, Weinan},
  journal={arXiv preprint arXiv:2409.16784},
  year={2024}
}
```

---

## License

This project is licensed under the BSD-3-Clause License. See [LICENSE.txt](LICENSE.txt) for details.