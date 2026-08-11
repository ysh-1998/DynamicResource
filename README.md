# DynamicResource — 大模型算力资源动态调度系统

基于流量预测与强化学习的端到端算力调度优化系统：先用序列模型预测大模型服务流量，再用PPO强化学习驱动GPU资源的动态扩缩容。


---

## 系统架构 / System Architecture

```
┌─────────────── 数据层 Data ───────────────┐    ┌──────── 模型层 Model ────────┐
│ resource_simulation.py                     │    │ model_traffic.py            │
│   四种流量模式仿真 → CSV                    │ →  │   AttentionLSTM / Improved   │
│   (daily/weekly/burst/steady)              │    │   Transformer / Hybrid /     │
└────────────────────────────────────────────┘    └──────────────┬───────────────┘
                                                                 │ train_traffic.py
                                                                 │   预训练 LSTM checkpoint
                                                                 ▼
┌─────────────── RL 调度层 RL Scheduling ────────────────────────────────────────┐
│ rl_env.py        PredictiveResourceEnv (gym.Env)                               │
│   262 维 obs = 256(LSTM 编码) + 6(即时状态) │ 7离散动作 │ 多目标加权奖励       │
│ rl_train.py      单卡两阶段PPO(SB3 model.learn)                              │
│ rl_train_ddp.py  多卡PPO(Worker采样 + 主进程更新 + Welford归一化聚合)        │
│ rl_evaluate.py   4策略对比评估(rule/rule_enhanced/hold/PPO) + 图表 + CSV      │
└────────────────────────────────────────────────────────────────────────────────┘
            │ run_rl_train_and_evaluate.sh        │ run_rl_train_and_evaluate_ddp.sh
            │ run_rl_evaluate.sh                  │ (SCRIPT_DIR 自动定位脚本所在目录)
            ▼
┌─────────────── 评估结果 Results ────────────────┐
│ rl_eval_results*/  CSV + 时序图 + 指标柱状图      │
└──────────────────────────────────────────────────┘
```

---

## 目录结构 / Directory Structure

| 文件File | 作用 / Role |
|---|---|
| **数据层 / Data** | |
| `resource_simulation.py` | 四种流量模式仿真数据生成器，输出QPS + token + 资源需求CSV / Traffic-pattern simulator → CSV |
| **模型层 / Model** | |
| `model_traffic.py` | AttentionLSTM（残差+LayerNorm+8头注意力）等4个模型定义 / Improved model definitions |
| `train_traffic.py` | 改进版训练脚本（4模型 + Accelerate DDP + 跨进程指标聚合）/ Main training script |
| **RL调度层 / RL** | |
| `rl_env.py` | RL环境：262维观测、7离散动作、多目标奖励；复用LSTM作状态编码器 / Gym env |
| `rl_train.py` | 单卡两阶段PPO训练（SB3）/ Single-GPU PPO trainer |
| `rl_train_ddp.py` | 多卡PPO训练（多进程采样 + 手写PPO更新）/ Multi-GPU PPO trainer |
| `rl_evaluate.py` | 4策略对比评估 + 可视化 + CSV / Policy evaluation |
| **Shell脚本 / Scripts** | |
| `run_rl_train_and_evaluate.sh` | 单卡训练+评测一体化（支持both/phase1/phase2 + --tag）/ Single-GPU pipeline |
| `run_rl_train_and_evaluate_ddp.sh` | 多卡训练+评测一体化 / Multi-GPU pipeline |
| `run_rl_evaluate.sh` | 独立评测（自动找最新模型 / 指定目录 / 指定模型路径）/ Standalone evaluation |
| `run_train_traffic.sh` | 流量预测模型训练（single单卡 / multi多卡，支持 --model --epochs --tag）/ Traffic predictor trainer |
| **配置 / Config** | |
| `requirements.txt` | Python依赖 / Dependencies |
| `.gitignore` | Git忽略规则 / Ignore rules |
| `training_results_lstm_*.png` | LSTM训练结果示例图 / Sample training plots |

---

## 核心特性 / Key Features

- **四种流量模式 / Four Traffic Patterns**：日常（9-18点高峰）、周（工作日1.0 / 周末0.6）、突发（2%概率2-8倍）、稳定（±8%波动）
- **模型架构 / Model Architecture**：双向LSTM + 8头注意力
- **编码器复用 / Encoder Reuse**：预训练LSTM作为RL状态编码器，262维观测（256编码 + 6即时状态）
- **PPO两阶段训练 / Two-Phase PPO**：Phase1冻结编码器训策略（lr 3e-4），Phase2解冻端到端微调（lr 1e-5）
- **多GPU训练 / Multi-GPU**：Worker进程并行采样 + 主进程集中PPO更新 + Welford归一化统计聚合
- **多目标奖励 / Multi-Objective Reward**：`R = 0.4·util + 0.3·sla + 0.25·cost + 0.1·change`

---

## 环境依赖 / Requirements

- Python 3.8+
- PyTorch 2.10.0+，CUDA 11.8+（GPU训练）
- 关键依赖：

```
torch==2.10.0
stable-baselines3==2.6.0
gymnasium==1.1.1
accelerate==1.14.0
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.9
tensorboard==2.21.0
```

完整依赖见 `requirements.txt`，安装：

```bash
pip install -r requirements.txt
```

> 注：`stable-baselines3` 与 `gymnasium` 是RL训练的必需依赖。

---

## 快速开始 / Quick Start

### 1. 生成仿真数据 / Generate simulation data

```bash
python resource_simulation.py
# 输出 / Output: simulation_data/*.csv
```

### 2. 训练流量预测模型 / Train traffic predictor

```bash
# 单卡 / Single GPU
bash run_train_traffic.sh single --model lstm --epochs 50
# 多卡 / Multi-GPU（GPU IDs 逗号分隔）
bash run_train_traffic.sh multi 0,1,2 --model lstm --epochs 50
# 模式 / Models: lstm | transformer | hybrid | ensemble（默认 lstm）
# 输出 / Output: models/<model>_traffic_<timestamp>.pth + training_results_<model>_<timestamp>.png
```

### 3. RL训练（单卡）/ RL training (single GPU)

```bash
bash run_rl_train_and_evaluate.sh both
# 模式 / Modes: both | phase1 | phase2
# 附加 / Optional: --tag <suffix>
```

### 4. RL训练（多卡）/ RL training (multi-GPU)

```bash
bash run_rl_train_and_evaluate_ddp.sh both
```

### 5. 独立评测 / Standalone evaluation

```bash
bash run_rl_evaluate.sh
# 支持三种调用 / Three usage modes:
#   无参数        → 自动找 rl_models/ 下最新训练目录
#   传训练目录     → 自动推断 best_model + vecnorm
#   手动指定       → --rl-model <path> --vecnorm <path>
```

---

## RL方案详解 / RL Approach

### MDP建模 / MDP Formulation

| 要素Element | 定义Definition |
|---|---|
| 状态S | 256（LSTM编码历史360步流量）+ 6（GPU数、计算利用率、显存利用率、延迟、冷却进度、副本数）|
| 动作A | 扩容+8/+16、缩容-8/-16、保持、切H100、切A100 |
| 奖励R | `0.4·util + 0.3·sla + 0.25·cost + 0.1·change` |
| 转移T | 1分钟/步，基于真实流量数据推进，动态重算指标 |
| Episode | 最大2000步，随机起点 |

### 两阶段训练 / Two-Phase Training

- **Phase 1**：冻结LSTM编码器，仅训策略/价值网络，lr=3e-4，100k步
- **Phase 2**：解冻编码器，端到端微调，lr=1e-5，400k步

### 多GPU架构 / Multi-GPU Architecture

每张GPU运行一个Worker进程并行采样，主进程通过 `param_queue` 广播新参数、`rollout_queue` 回收数据，沿env维拼接后做GAE + PPO更新；VecNormalize统计用Welford并行方差合并算法聚合各Worker增量，避免双重计数。

---

## 实验结果 / Results

ChatGLM_Daily数据集，100k+400k配置：

| 指标Metric | 规则基线Rule | 保守基线Hold | **RL (PPO)** |
|---|---|---|---|
| 平均奖励 | +0.0939 | -0.3854 | **+0.1185 (+26.2%)** |
| GPU利用率 | 77.49% | 81.68% | 75.22% |
| 延迟满足率 | 97.32% | 72.62% | 95.73% |
| 过载率 | 24.34% | 38.77% | **21.29%** |
| P95延迟(ms) | 1393.0 | 2500.2 | 1417.7 |
| 成本($/h) | 388.42 | 403.25 | 389.49 |
| 调整频率 | 48.66% | 0.00% | **12.38%** |
| 调整惩罚 | -0.4431 | +0.0000 | **-0.0697** |

**结论 / Conclusion**：RL综合奖励最高（+0.1185，较规则基线+26.2%），调整频率大幅下降（48.66% → 12.38%），过载率降低（24.34% → 21.29%），策略稳定性显著提升；成本与规则基线基本持平。

---

## 关键超参数 / Hyperparameters

| 超参数 | 值 |
|---|---|
| Phase 1 lr / steps | 3e-4 / 100,000 |
| Phase 2 lr / steps | 1e-5 / 400,000 |
| gamma / gae_lambda | 0.99 / 0.95 |
| n_steps / batch_size | 2048 / 256 |
| n_epochs / clip_range | 10 / 0.2 |
| ent_coef / vf_coef | 0.01 / 0.5 |
| Actor/Critic (Phase 1) | [256, 256, 128] ReLU |
| Actor/Critic (Phase 2) | [256, 128] ReLU |
| seq_len / max_episode | 360 / 2000 |
| n_envs | 4 |
