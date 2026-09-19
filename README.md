# WMP on NVIDIA IsaacLab

[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.x-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)

本工程是论文 **World Model-based Perception for Visual Legged Locomotion (WMP)** 在现代 **NVIDIA IsaacLab (2.x)** 仿真平台上的重构与升级实现。

项目完整移除了对已淘汰的 IsaacGym Preview 4 及 `legged_gym` 的依赖，基于 IsaacLab 规范的 ManagerBasedRLEnv 架构，直接实现了 **DreamerV3 RSSM 表征世界模型 + AMP (Adversarial Motion Priors) 对抗先验步态约束 + Cross-Attention 3D 高程空间感知** 的全套核心算法体系。

---

## 核心架构特性

1. **DreamerV3 RSSM 时序世界模型**：
   - 采用确定性（`dyn_deter` 512维）与随机性离散隐状态，时序建模本体感受与高程环境演化；
   - 支持宏观时间步长（Macro Phase）更新与隐状态自回归表征。
2. **AMP 真实步态先验约束**：
   - 对齐真机动捕数据（Unitree A1 动捕轨迹，30 维观测）；
   - 使用对抗判别器（Discriminator）提供步态风格奖励与 WGAN-GP 梯度惩罚约束，消除非自然步态。
3. **Cross-Attention 3D 空间高程感知**：
   - 搭载基座对齐的 RayCaster 网格雷达（25×17 网格，425 射线点）；
   - 提取包含 `[x, y, z]` 三维坐标的局部高程图（1275 维），通过多头交叉注意力网络实现空间几何对齐。
4. **自主维护本地算法库**：
   - 本地源码级集成 [`rsl_rl/`](rsl_rl/)，支持与 IsaacLab 观测体系及世界模型的联合反向传播。

---

## 工程目录结构

```text
WMP/
├── source/wmp/                 # IsaacLab 核心扩展包
│   ├── config/extension.toml   # Omniverse Kit 扩展元数据
│   ├── setup.py / pyproject.toml
│   └── wmp/
│       ├── env_cfg.py          # A1 机器人场景、传感器与越障环境配置
│       ├── agent_cfg.py        # WMPRunner 算法超参数与 PPO/AMP 配置
│       ├── mdp/                # 解耦 MDP 体系 (observations, reward, terminations, curriculum)
│       └── terrains/           # 复杂越障地形库 (rough, discrete, stairs, obstacles)
├── rsl_rl/                     # 本地强化学习算法库 (WMPRunner, AMPPPO, ActorCriticWMP)
├── dreamer/                    # DreamerV3 RSSM 时序网络与模型配置
├── datasets/mocap_motions/     # Unitree A1 真实动捕数据
├── docs/                       # 重构方案与开发文档
└── scripts/                    # 用户执行入口脚本
    ├── list_envs.py            # 环境与任务查询工具
    ├── export_terrain.py       # 3D 地形导出工具
    └── rsl_rl/
        ├── train.py            # WMP + AMP 强化学习训练入口
        └── play.py             # 训练模型检查点评估与推理回放
```

---

## 快速上手

### 1. 环境准备与扩展包安装
激活已配置好 Isaac Sim 5.1 / IsaacLab 2.x 的 Conda 环境，并在本项目根目录下以可编辑模式安装扩展包：

```bash
conda activate isaaclab
pip install -e source/wmp --no-deps
```

### 2. 查看已注册的任务环境
```bash
python scripts/list_envs.py
```
终端将输出注册的两个核心任务：
- `Velocity-Rough-WMP-Train`：多并行环境越障地形训练；
- `Velocity-Rough-WMP-Play`：单步评估回放与轻量可视化测试。

### 3. 启动模型训练 (Train)
```bash
# 启动 16 并行环境轻量测试 / 无头模式训练
python scripts/rsl_rl/train.py --task Velocity-Rough-WMP-Train --num_envs 16 --headless

# 启动全规模训练 (默认 4096 环境，可根据 GPU 显存调整)
python scripts/rsl_rl/train.py --task Velocity-Rough-WMP-Train --num_envs 4096 --headless
```
训练日志、模型检查点和参数快照将自动保存至 `logs/rsl_rl/unitree_a1_wmp/` 目录。

### 4. 评估与推理推断 (Play)
```bash
# 加载指定检查点进行可视化回放评估
python scripts/rsl_rl/play.py \
  --task Velocity-Rough-WMP-Play \
  --num_envs 2 \
  --checkpoint logs/rsl_rl/unitree_a1_wmp/<RUN_DIR>/model_<ITER>.pt
```

---

## 引用

如果您在研究或工作中参考了本项目，请引用原论文：
```bibtex
@article{lai2024world,
  title={World Model-based Perception for Visual Legged Locomotion},
  author={Lai, Hang and Cao, Jiahang and Xu, Jiafeng and Wu, Hongtao and Lin, Yunfeng and Kong, Tao and Yu, Yong and Zhang, Weinan},
  journal={arXiv preprint arXiv:2409.16784},
  year={2024}
}
```