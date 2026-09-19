# WMP (World Model + AMP) 迁移重构方案：基于 IsaacLab 现代架构

## 1. 重构目标与核心定位

### 1.1 核心目标
本项目旨在将 **原版 WMP (World Model-based Perception for Visual Legged Locomotion)** 从旧版 NVIDIA IsaacGym 完整重构为现代 **NVIDIA IsaacLab (2.x)** 工程。

**本项目直接落地原版 WMP 核心算法与步态约束，绝不做任何纯 CNN 的降级替代**：
1. **WMP 世界模型 (World Model - DreamerV3)**：
   - 基于循环状态空间模型 (RSSM) 进行自监督环境表征学习；
   - 学习确定性隐状态 $h_t$ 与随机隐状态 $z_t$，生成环境隐空间特征向量 $s_t$；
   - 支持前向高程/深度重构预测 (`DepthPredictor`)。
2. **AMP 步态约束 (Adversarial Motion Priors)**：
   - 基于动捕数据集 (`datasets/mocap_motions/*.txt`) 训练对抗判别器 (`AMPDiscriminator`)；
   - 提取机器人当前状态与动捕参考状态进行对抗对比，输出平滑、自然的动物步态奖励 $r^{\text{AMP}}$。
3. **ActorCriticWMP 策略网络**：
   - Policy 网络融合本体感觉 (Proprioception)、历史时序观测 (History Obs) 以及世界模型特征隐向量 $s_t$；
   - Critic 网络接收特权线速度、质量、接触状态与全局高度图。
4. **WMPRunner 协同训练调度**：
   - 深度集成 RolloutStorage（供 PPO 策略更新）与 ReplayBuffer（供世界模型自监督训练）；
   - 单步交互中同步驱动环境步进、经验存入、世界模型前向推断、判别器对抗更新与策略网络梯度优化。

### 1.2 工程架构范式 (完全对齐 `wmp_cnn` 规范)
项目代码组织完全拥抱 IsaacLab 现代化扩展包架构，参考 `/home/brave/isaaclab_pj/wmp_cnn` 的工程组织标准：
- 采用 **ManagerBasedRLEnv**，将旧版单体 `LeggedRobot` 彻底解耦为 Scene, Observations, Actions, Commands, Rewards, Terminations, Events, Curriculum 等管理器；
- 采用标准的 `source/wmp` 模块打包与可编辑安装；
- 采用清晰独立的 `scripts/rsl_rl/` 作为训练与推理的统一入口；
- 算法库 `rsl_rl/` 本地化维护，以便随时针对网络架构、损失函数与训练流程进行深度定制与断点调试。

---

## 2. 系统数据流与架构设计

```text
               +-------------------------------------------------------------+
               |                  IsaacLab ManagerBasedRLEnv                 |
               |       (Unitree A1 + RayCaster 网格扫描 + 复杂越障地形)        |
               +-------------------------------------------------------------+
                     │                    │                       │
                     │ amp_obs            │ obs (proprio + hist)  │ height grid
                     ▼                    ▼                       ▼
          +--------------------+  +---------------+     +--------------------+
          |  AMP 判别器        |  | ActorCriticWMP| <───┤   DreamerV3        |
          | (对比 Mocap 动捕)  |  | (Policy/Critic|     |   世界模型 (RSSM)  |
          +--------------------+  +---------------+     +--------------------+
                     │                    ▲                       │
          amp_reward ▼                    │ action                │ 隐向量 s_t
               +─────────────────────────────────────────────────────────────+
               |                          WMPRunner                          |
               |    (RolloutStorage + ReplayBuffer + 多目标协同更新调度)      |
               +─────────────────────────────────────────────────────────────+
```

### 2.1 观测系统规划 (`ObservationsCfg`)
在 `source/wmp/wmp/env_cfg.py` 中规划解耦的观测组：
1. **`policy`**：Actor 输入的基础本体感受（基座角速度、重力投影、速度指令、关节角度/角速度、历史动作）；
2. **`history`**：多步本体感受历史窗口（时序卷积或平铺拼接）；
3. **`critic`**：Critic 专用特权观测（真实全局线速度、质心偏移量、刚体质量、地面摩擦系数、足端接触力等）；
4. **`amp`**：专用动捕特征匹配观测（基座离地高度、基座朝向姿态、线速度、角速度、关节角度、关节角速度、四足相对基座坐标）；
5. **`wm_perception`**：供给世界模型 encoder 的环境扫描网格（通过 `RayCaster` 高精度扫描生成）。

---

## 3. 目标工程完整目录树

```text
WMP/
├── datasets/                       # AMP 参考动捕数据集
│   └── mocap_motions/              # trot1.txt, hop1.txt, etc.
├── docs/                           # 项目技术文档
│   ├── isaaclab_refactor_plan.md   # 本重构计划文档
│   ├── amp_integration.md          # AMP 步态先验与判别器技术文档
│   └── world_model.md              # DreamerV3 世界模型设计文档
├── dreamer/                        # DreamerV3 核心模型
│   ├── configs.yaml                # 世界模型超参数 (RSSM 维度、网络层级等)
│   ├── models.py                   # RSSM, Encoder, Decoder
│   ├── networks.py                 # MLP, Conv, GRU 基础网络单元
│   └── tools.py                    # 辅助工具函数
├── rsl_rl/                         # ★ 本地 WMP 强化学习算法库 (源码维护与调试)
│   ├── algorithms/
│   │   ├── amp_discriminator.py    # AMP 对抗判别器 (LSGAN + 梯度惩罚)
│   │   ├── amp_ppo.py              # 融合 AMP 奖励的 PPO 算法
│   │   └── ppo.py                  # 基础 PPO 算法
│   ├── datasets/
│   │   ├── isaaclab_amp_loader.py  # IsaacLab 动捕数据加载器 (对齐 A1 关节)
│   │   ├── motion_loader.py        # 动捕解析工具
│   │   └── pose3d.py
│   ├── modules/
│   │   ├── actor_critic_wmp.py     # 融合 World Model 隐向量的策略网络
│   │   └── depth_predictor.py      # 前向高程/深度重构预测器
│   ├── runners/
│   │   └── wmp_runner.py           # IsaacLab 适配版 WMPRunner 训练总调度
│   ├── storage/
│   │   ├── rollout_storage.py      # PPO 轨迹存储
│   │   └── replay_buffer.py        # 世界模型离线/在线经验回放池
│   └── utils/
│       └── utils.py
├── source/
│   └── wmp/                        # 标准 IsaacLab 扩展包 (架构完全对齐 wmp_cnn)
│       ├── setup.py                # pip install -e 安装入口
│       ├── pyproject.toml          # 打包规范
│       ├── config/
│       │   └── extension.toml      # Omniverse 扩展元数据
│       └── wmp/
│           ├── __init__.py         # 注册 Gymnasium 任务 (Velocity-Rough-WMP-Train 等)
│           ├── agent_cfg.py        # WMPRunner, PPO, Dreamer, AMP 超参数配置
│           ├── env_cfg.py          # 基于 ManagerBasedRLEnvCfg 的完整场景定义
│           ├── mdp/                # 解耦的 MDP 函数库
│           │   ├── __init__.py
│           │   ├── observations.py # policy, critic, amp, heightmap 观测生成
│           │   ├── reward.py       # 速度跟踪、平滑惩罚、AMP 对抗奖励集成
│           │   ├── terminations.py # 摔倒、碰撞、边界终止
│           │   └── curriculum.py   # 地形课程自适应
│           └── terrains/           # 复杂越障与跑酷地形系统
│               ├── __init__.py
│               ├── terrain_cfg.py  # 障碍/台阶/斜坡/跨缝等训练地形
│               ├── finetune_terrain_cfg.py
│               └── loco_hf_terrains.py
├── scripts/
│   ├── list_envs.py                # 环境检查工具
│   ├── export_terrain.py           # 地形导出工具
│   └── rsl_rl/
│       ├── cli_args.py             # 命令行参数配置
│       ├── train.py                # 训练主入口 (加载本地 WMPRunner)
│       └── play.py                 # 评估与推理可视化
├── logs/                           # 模型权重与 Tensorboard
└── README.md                       # 项目运行指南
```

---

## 4. 关键技术模块与迁移方案

### 4.1 AMP 步态约束模块 (`rsl_rl/algorithms/amp_discriminator.py` & `mdp/observations.py`)
1. **动捕数据加载器 (`IsaacLabAMPLoader`)**：
   - 适配 Unitree A1 在 IsaacLab 中的关节顺序 (`[FL, FR, RL, RR]`)；
   - 载入 `datasets/mocap_motions/` 下的各类动物步态数据，支持参考运动随机采样与时间步对齐。
2. **`amp_obs` 状态提取**：
   - 在 `mdp/observations.py` 中编写 `amp_observations` 函数，严格提取包含：
     - 基座高度 $z$
     - 基座旋转四元数/欧拉角
     - 基座线速度与角速度
     - 12个关节位置与角速度
     - 四足相对基座坐标
   - 保证维度与动捕特征完全一致。
3. **对抗判别器 (`AMPDiscriminator`)**：
   - 训练判别器以区分专家动捕状态 $s^E$ 与策略生成状态 $s^\pi$；
   - 计算最小二乘 GAN 损失及 $R_1$ 梯度惩罚；
   - 实时输出平滑奖励 $r^{\text{AMP}} = \max(0, 1 - 0.25(D(s) - 1)^2)$。

### 4.2 WMP 世界模型与前向高程预测 (`dreamer/` & `rsl_rl/modules/`)
1. **循环状态空间模型 (RSSM)**：
   - 输入本体感觉时序 + `RayCaster` 高程网格；
   - 通过确定性状态 $h_t$ 与随机状态 $z_t$ 自监督学习环境转移概率；
   - 输出维度如 1024 维的环境表征特征 $s_t = [h_t, z_t]$。
2. **前向高程预测器 (`DepthPredictor`)**：
   - 从隐向量 $s_t$ 解码预测前向网格点高度，辅助表征学习并强化对复杂地形特征的感知能力。
3. **策略融合 (`ActorCriticWMP`)**：
   - Actor 策略接收本体感觉 + 历史序列 + 隐特征 $s_t$，输出关节控制目标。

### 4.3 训练总调度器 (`rsl_rl/runners/wmp_runner.py`)
- 全面适配 IsaacLab 的 `RslRlVecEnvWrapper`；
- 单步 Rollout 过程中：
  1. 向 `ReplayBuffer` 写入世界模型转移数据，定时更新 DreamerV3 损失；
  2. 向 `AMPDiscriminator` 写入策略轨迹，对抗更新判别器；
  3. 执行 `DepthPredictor` 前向预测损失反向传播；
  4. 综合任务速度跟踪奖励与 AMP 奖励，更新 PPO 策略与价值网络。

---

## 5. 分阶段落地实施路线图 (Milestones)

### 阶段 1：项目工程骨架与本地算法库构建
- [x] 创建 `source/wmp/` 扩展目录结构，编写 `setup.py`、`pyproject.toml`、`config/extension.toml`。
- [x] 整合并适配本地 `rsl_rl/`：
  - 接入 `WMPRunner`（统一支持 RSSM 世界模型与 AMP 对抗训练调度）；
  - 接入 `AMPDiscriminator` 与 `IsaacLabAMPLoader`；
  - 接入 `ActorCriticWMP`、`CrossAttentionTerrainEncoder` 与 `AMPPPO`。
- [x] 适配 `dreamer/` 模块与 `dreamer/configs.yaml`。
- [x] 在 conda 的 `isaaclab` 环境下执行 `pip install -e source/wmp --no-deps` 验证通过。

### 阶段 2：场景、机器人与传感器构建 (`env_cfg.py`)
- [x] 编写 `VelocitySceneCfg`：引入 Unitree A1 资产，精确配置关节限制、驱动器刚度与阻尼。
- [x] 配置足端接触传感器 `contact_forces`。
- [x] 配置高精度网格扫描器 `RayCasterCfg`（25x17 网格，分辨率 0.05m，范围 1.2mx0.8m）。
- [x] 迁移 `terrains/` 复杂越障地形库（台阶、斜坡、离散金字塔、越障地形）。

### 阶段 3：MDP 逻辑与 AMP 观测提取
- [x] 编写 `mdp/observations.py`：实现 `policy` (1323 维)、`critic` (1352 维)、`amp` (30 维对齐动捕) 与 `elevation_map` (1275 维 3D 坐标高程图)。
- [x] 编写 `mdp/reward.py`：实现速度跟踪、姿态惩罚、足端触地时间与平滑惩罚。
- [x] 编写 `mdp/terminations.py`（摔倒、姿态翻转终止）与 `mdp/curriculum.py`（地形等级跃迁）。

### 阶段 4：Agent 配置与任务注册
- [x] 编写 `source/wmp/wmp/agent_cfg.py`，定义 `UnitreeA1WMPRunnerCfg`（联合配置 PPO、Dreamer 及 AMP 超参数）。
- [x] 在 `source/wmp/wmp/__init__.py` 中注册任务：
  - `Velocity-Rough-WMP-Train`
  - `Velocity-Rough-WMP-Play`
- [x] 适配 `scripts/rsl_rl/train.py` 与 `scripts/rsl_rl/play.py`，确保其直接调用本地 `WMPRunner`。

### 阶段 5：小规模冒烟测试与系统验证
- [x] 启动轻量环境测试：
  ```bash
  /home/brave/miniconda3/envs/isaaclab/bin/python scripts/rsl_rl/train.py --task Velocity-Rough-WMP-Train --num_envs 16 --headless --max_iterations 2
  ```
- [x] 关键验证指标全数通过：
  1. 动捕数据正常加载（4 motions, 30 维，10 万步预加载）；
  2. AMP 判别器 Loss、梯度惩罚与奖励正常计算；
  3. DreamerV3 隐状态前向推进与时序更新正常；
  4. PPO 策略梯度与 ActorCriticWMP 权重更新正常并成功保存 checkpoint。

### 阶段 6：推理回放与使用说明
- [x] 验证 `scripts/rsl_rl/play.py` 的加载与单步推断：
  ```bash
  /home/brave/miniconda3/envs/isaaclab/bin/python scripts/rsl_rl/play.py --task Velocity-Rough-WMP-Play --num_envs 2 --headless --num_steps 10 --checkpoint .../model_2.pt
  ```
  成功加载检查点并跑通完整的 World Model + PPO 时序推断循环。

