#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_env.py - 预测感知型强化学习环境

将预训练的 AttentionLSTM 作为状态编码器集成到 Gym 环境中，
使 RL Agent 能感知历史60分钟的流量趋势，从而做出更优的资源分配决策。

状态空间：256维时序语义编码 + 5维即时运营状态 = 261维
动作空间：7个离散动作（扩容/缩容/不变）
奖励函数：GPU利用率 × 0.40 + SLA满足度 × 0.30 + 成本惩罚 × 0.25 + 调整惩罚 × 0.10
SLA 基准：1500ms（ChatGLM Daily 场景 p50 延迟，低于基准正奖励，高于线性惩罚 clamp[-1,1]）
util 软区间：[0.65,0.85]=1.0，[0.55,0.65)∪(0.85,0.95]=0.5，其余陡惩罚
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import RobustScaler
from typing import Optional, Tuple, Dict, Any

from model_traffic import AttentionLSTM


# ─────────────────────────────────────────────
# 特征工程（与 train_traffic.py 保持完全一致）
# ─────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    从 DataFrame 提取 23 维特征向量序列，
    与 train_traffic.py / DataProcessor.extract_features_from_df 完全对齐。

    Args:
        df: 包含仿真数据的 DataFrame（单个 model_type 场景）

    Returns:
        features: (T, 23) float64 numpy array
    """
    features = []
    window_size = 10

    for idx in range(len(df)):
        row = df.iloc[idx]

        # ── 基础字段 ──────────────────────────────────────────────
        qps                = float(row['qps'])
        tokens_per_second  = float(row['effective_tokens_per_second'])
        memory_demand      = float(row.get('total_memory_gb',
                              row.get('memory_demand_gb',
                              row.get('memory_demand', 0.0))))
        gpu_demand         = float(row.get('required_gpus',
                              row.get('gpu_demand', 0.0)))
        gpu_utilization    = float(row['gpu_utilization'])
        memory_utilization = float(row['memory_utilization'])
        avg_input_tokens   = float(row.get('avg_input_tokens',
                              tokens_per_second / qps * 0.3 if qps > 0 else 100))
        avg_output_tokens  = float(row.get('avg_output_tokens',
                              tokens_per_second / qps * 0.7 if qps > 0 else 300))

        # ── 时间特征 ──────────────────────────────────────────────
        if 'timestamp' in df.columns:
            ts           = pd.to_datetime(row['timestamp'])
            hour         = ts.hour
            day_of_week  = ts.dayofweek
        else:
            hour         = int(row.get('hour', 12))
            day_of_week  = int(row.get('day_of_week', 3))

        is_peak_hour = 1 if (9 <= hour <= 17) else 0
        hour_sin     = np.sin(2 * np.pi * hour / 24)
        hour_cos     = np.cos(2 * np.pi * hour / 24)
        dow_sin      = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos      = np.cos(2 * np.pi * day_of_week / 7)
        time_features = [hour_sin, hour_cos, dow_sin, dow_cos, is_peak_hour]

        # ── 滑动窗口统计特征 ──────────────────────────────────────
        if idx >= window_size:
            recent_qps   = df['qps'].iloc[idx - window_size:idx].values
            recent_avg   = float(np.mean(recent_qps))
            recent_std   = float(np.std(recent_qps))
            recent_trend = float((qps - recent_qps[0]) / (recent_qps[0] + 1e-8))
        else:
            recent_avg   = qps
            recent_std   = 0.0
            recent_trend = 0.0

        # ── 场景 one-hot（训练集用 'combined'，对应 train_traffic 的模型+模式标签）
        # 此处为通用特征提取，4 位全 0 即可（推理时编码器不依赖这部分做资源决策）
        pattern_features = [0, 0, 0, 0]
        model_features   = [0, 0, 0, 0]

        # ── 组合 ──────────────────────────────────────────────────
        feature_vector = (
            [qps, tokens_per_second, memory_demand, gpu_demand,
             gpu_utilization, memory_utilization,
             avg_input_tokens, avg_output_tokens]
            + time_features                         # 5
            + [recent_avg, recent_std, recent_trend] # 3
            + pattern_features                       # 4
            + model_features                         # 4
        )
        # 交互特征（3）→ 总计 8+5+3+4+4+3 = 27 … 与 train_traffic 对齐
        feature_vector.extend([
            qps * gpu_utilization,
            memory_demand * memory_utilization,
            tokens_per_second / (avg_output_tokens + 1),
        ])

        features.append(feature_vector)

    return np.array(features, dtype=np.float64)


# ─────────────────────────────────────────────
# 预训练 LSTM 状态编码器
# ─────────────────────────────────────────────

class QPSAwareStateEncoder(nn.Module):
    """
    复用预训练 AttentionLSTM 的 LSTM + Attention 主干，
    将历史序列窗口 → 256 维时序语义表示，不经过原始的 QPS 回归头。

    参数
    ----
    checkpoint_path : str
        由 train_traffic.py 保存的 .pth 文件路径。
        checkpoint 必须包含：
            model_state_dict, feature_dim
    freeze : bool
        阶段1：True（冻结，只训练策略网络）
        阶段2：False（端到端微调）
    """

    ENCODER_OUT_DIM = 256  # hidden_dim(128) × 2(bidirectional)

    def __init__(self, checkpoint_path: str, freeze: bool = True):
        super().__init__()

        ckpt          = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        feature_dim   = ckpt['feature_dim']
        hidden_dim    = 128
        num_layers    = 3

        # 重建与 train_traffic.py 完全相同的结构
        self.backbone = AttentionLSTM(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=1,      # 原始头，加载权重用，forward 时不调用
            num_layers=num_layers,
            dropout=0.0,       # 推理时关闭 dropout
        )
        self.backbone.load_state_dict(ckpt['model_state_dict'])
        self.feature_dim = feature_dim

        if freeze:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()

        # 保存 scaler 用于特征标准化
        self.scaler: Optional[RobustScaler] = ckpt.get('feature_scaler', None)

    def freeze_encoder(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (batch, seq_len, feature_dim) 归一化后的特征序列

        Returns:
            encoded: (batch, 256) 时序语义向量
        """
        b = self.backbone

        # 输入投影
        x_proj   = b.input_projection(x_seq)             # (B, T, H)
        lstm_out, _ = b.lstm(x_proj)                      # (B, T, H*2)

        # 残差 + LayerNorm
        residual = b.residual_projection(x_seq)           # (B, T, H*2)
        lstm_out = b.layer_norm(lstm_out + residual)      # (B, T, H*2)

        # Multi-head Attention
        lstm_out_t = lstm_out.transpose(0, 1)             # (T, B, H*2)
        attn_out, _ = b.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        attn_out = attn_out.transpose(0, 1)               # (B, T, H*2)

        # 取最后时间步作为序列表示
        return attn_out[:, -1, :]                         # (B, 256)


# ─────────────────────────────────────────────
# 强化学习环境
# ─────────────────────────────────────────────

class PredictiveResourceEnv(gym.Env):
    """
    预测感知型 LLM 资源调度 RL 环境。

    状态：
        - 256 维：QPSAwareStateEncoder 对历史 seq_len 步的时序编码
        - 5   维：即时运营状态（当前GPU数、利用率、显存利用率、延迟、冷却进度）
        合计 261 维。

    动作（离散，7 个）：
        0: 扩容 +8  GPU（+1台服务器）
        1: 扩容 +16 GPU（+2台服务器）
        2: 缩容 -8  GPU（-1台服务器）
        3: 缩容 -16 GPU（-2台服务器）
        4: 保持不变
        5: 切换至 H100（高负载省显存）
        6: 切换至 A100（低负载省成本）

    奖励：
        R = 0.40 × util_reward
          + 0.30 × sla_reward
          + 0.25 × cost_penalty
          + 0.05 × change_penalty
          + overload_hard_penalty

    参数
    ----
    data_path     : combined_simulation_data.csv 路径
    checkpoint_path: 预训练 LSTM checkpoint 路径
    model_type    : 'GPT-4' | 'ChatGLM' | 'Claude' | 'LLaMA'
    gpu_type      : 初始 GPU 型号，'A100' 或 'H100'
    seq_len       : 历史窗口长度（分钟数），与训练时 sequence_length 一致
    cooldown_steps: 扩缩容冷却步数（防抖）
    device        : 'cpu' 或 'cuda'
    """

    metadata = {"render_modes": []}

    # 动作含义
    ACTION_DELTA = {0: 8, 1: 16, 2: -8, 3: -16, 4: 0, 5: 0, 6: 0}
    # GPU 单位成本（USD/h/GPU）
    GPU_COST = {'A100': 2.5, 'H100': 4.0}
    # GPU 显存（GB）
    GPU_MEMORY_GB = {'A100': 80, 'H100': 80}
    # 切换后的 GPU 型号
    GPU_SWITCH = {5: 'H100', 6: 'A100'}

    # 与 resource_simulation.py 完全对齐的模型配置
    MODEL_CONFIGS = {
        'GPT-4':   {'model_size_b': 1800, 'memory_per_billion_params': 2.5,
                    'batch_memory_factor': 1.5, 'gpu_efficiency': 0.65,
                    'min_gpus_per_replica': 32, 'prefill_compute_factor': 2.2,
                    'tokens_per_second_per_gpu': {'A100': 800,  'H100': 1500}},
        'ChatGLM': {'model_size_b': 130,  'memory_per_billion_params': 2.5,
                    'batch_memory_factor': 1.4, 'gpu_efficiency': 0.80,
                    'min_gpus_per_replica': 4,  'prefill_compute_factor': 1.8,
                    'tokens_per_second_per_gpu': {'A100': 2500, 'H100': 4000}},
        'Claude':  {'model_size_b': 400,  'memory_per_billion_params': 2.5,
                    'batch_memory_factor': 1.45,'gpu_efficiency': 0.75,
                    'min_gpus_per_replica': 8,  'prefill_compute_factor': 3.0,
                    'tokens_per_second_per_gpu': {'A100': 1800, 'H100': 3000}},
        'LLaMA':   {'model_size_b': 70,   'memory_per_billion_params': 2.5,
                    'batch_memory_factor': 1.3, 'gpu_efficiency': 0.85,
                    'min_gpus_per_replica': 2,  'prefill_compute_factor': 1.5,
                    'tokens_per_second_per_gpu': {'A100': 3000, 'H100': 5000}},
    }

    def __init__(
        self,
        data_path: str,
        checkpoint_path: str,
        model_type: str = 'GPT-4',
        gpu_type: str = 'A100',
        seq_len: int = 60,
        cooldown_steps: int = 5,
        max_episode_steps: int = 2000,
        device: str = 'cpu',
    ):
        super().__init__()

        self.seq_len        = seq_len
        self.cooldown_steps = cooldown_steps
        self.device         = torch.device(device)
        self.model_type     = model_type

        # ── 加载并过滤数据 ──────────────────────────────────────────
        df_all = pd.read_csv(data_path)
        if 'model_type' in df_all.columns:
            self.df = df_all[df_all['model_type'] == model_type].reset_index(drop=True)
        else:
            self.df = df_all.reset_index(drop=True)

        if len(self.df) < seq_len + 100:
            raise ValueError(
                f"数据量不足（{len(self.df)} 行），"
                f"至少需要 seq_len({seq_len}) + 100 行"
            )

        # ── 特征提取 + 标准化 ──────────────────────────────────────
        # 每个子进程独立初始化，用进程号区分日志避免重复刷屏
        pid = os.getpid()
        print(f"[Env pid={pid}] 提取特征（{len(self.df)} 行）...")
        raw_features = extract_features(self.df)        # (T, feat_dim)

        # ── 加载编码器（获取 scaler）─────────────────────────────────
        print(f"[Env pid={pid}] 加载预训练编码器: {checkpoint_path}")
        self.encoder = QPSAwareStateEncoder(checkpoint_path, freeze=True)
        self.encoder.eval()
        self.encoder.to(self.device)

        # 优先用 checkpoint 内置的 scaler，保证与训练时归一化完全一致
        if self.encoder.scaler is not None:
            scaler = self.encoder.scaler
            self.features_scaled = scaler.transform(raw_features).astype(np.float32)
            print(f"[Env pid={pid}] 使用 checkpoint 内置 RobustScaler")
        else:
            scaler = RobustScaler()
            self.features_scaled = scaler.fit_transform(raw_features).astype(np.float32)
            print(f"[Env pid={pid}] checkpoint 无内置 scaler，重新拟合 RobustScaler")

        self.feat_dim = raw_features.shape[1]

        # ── Gym 空间 ──────────────────────────────────────────────
        obs_dim = QPSAwareStateEncoder.ENCODER_OUT_DIM + 6  # 256 + 6 = 262
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        # ── 运行状态 ──────────────────────────────────────────────
        self.max_episode_steps = max_episode_steps
        self.current_step      = seq_len
        self._episode_start    = seq_len   # 记录每个 episode 的起始步，用于 truncated 判断
        self.current_gpus      = 0
        self.current_gpu_type  = gpu_type
        self.cooldown          = 0

        # ── 缓存当前模型配置（与 resource_simulation.py 完全对齐）──
        self._model_cfg = self.MODEL_CONFIGS[model_type]

    # ──────────────────────────────────────────────────────────────
    # Gym 接口
    # ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # 随机起始点（避免过拟合到固定序列）
        max_start = len(self.df) - 200
        self.current_step = int(self.np_random.integers(self.seq_len, max_start))

        # 记录本 episode 起始步（用于 truncated 判断）
        self._episode_start  = self.current_step
        # 用数据中的"规则分配"值作为初始 GPU 配置
        self.current_gpus    = max(8, int(self.df.iloc[self.current_step]['required_gpus']))
        self.cooldown        = 0

        obs = self._build_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # ── 执行动作 ──────────────────────────────────────────────
        if self.cooldown == 0:
            delta = self.ACTION_DELTA[action]
            if delta != 0:
                self.current_gpus = max(8, self.current_gpus + delta)
                self.cooldown = self.cooldown_steps

            # GPU 型号切换
            if action in self.GPU_SWITCH:
                new_type = self.GPU_SWITCH[action]
                if new_type != self.current_gpu_type:
                    self.current_gpu_type = new_type
                    self.cooldown = self.cooldown_steps
        else:
            self.cooldown -= 1

        # ── 推进时间 ──────────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated  = (self.current_step - self._episode_start) >= self.max_episode_steps

        row     = self.df.iloc[self.current_step]
        metrics = self._recompute_metrics(row)   # 基于当前 GPU 配置动态重算
        reward  = self._compute_reward(metrics, action)
        obs     = self._build_obs()

        info = {
            "gpu_utilization":    metrics['gpu_utilization'],
            "memory_utilization": metrics['memory_utilization'],
            "estimated_latency":  metrics['estimated_latency_ms'],
            "current_gpus":       self.current_gpus,
            "current_gpu_type":   self.current_gpu_type,
            "qps":                float(row['qps']),
            "hourly_cost":        metrics['hourly_cost'],
            "required_replicas":  metrics['required_replicas'],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        row     = self.df.iloc[self.current_step]
        metrics = self._recompute_metrics(row)
        print(
            f"Step {self.current_step:5d} | "
            f"QPS={row['qps']:7.1f} | "
            f"GPUs={self.current_gpus:4d} ({self.current_gpu_type}) | "
            f"Util={metrics['gpu_utilization']:.2%} | "
            f"MemUtil={metrics['memory_utilization']:.2%} | "
            f"Latency={metrics['estimated_latency_ms']:.0f}ms | "
            f"Replicas={metrics['required_replicas']} | "
            f"Cost=${metrics['hourly_cost']:.1f}/h"
        )

    # ──────────────────────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """拼接编码器输出（256维）和即时运营状态（5维）。"""
        encoded   = self._encode_history()          # (256,)
        instant   = self._instant_state()           # (5,)
        return np.concatenate([encoded, instant]).astype(np.float32)

    def _encode_history(self) -> np.ndarray:
        """对历史 seq_len 步特征序列做 LSTM 编码，返回 (256,)。"""
        start = max(0, self.current_step - self.seq_len)
        seq   = self.features_scaled[start:self.current_step]

        # 不足 seq_len 时在左侧补零
        if len(seq) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq), self.feat_dim), dtype=np.float32)
            seq = np.vstack([pad, seq])

        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, T, D)
        with torch.no_grad():
            encoded = self.encoder(x)                            # (1, 256)
        return encoded.squeeze(0).cpu().numpy()                  # (256,)

    def _recompute_metrics(self, row: pd.Series) -> dict:
        """
        基于 Agent 当前分配的 GPU 数，动态推导 t+1 时刻的运行指标。

        与 resource_simulation.py 的关键区别：
          - 仿真器方向：流量 → 计算最优 GPU 需求（含 redundancy_factor 冗余）
          - 本函数方向：给定实际 GPU 数 → 推导运行效果（利用率/延迟/成本）
          使用相同的物理公式（吞吐量、内存、延迟），但计算方向相反。

        具体差异：
          1. 副本数用整除 current_gpus // gpus_per_replica 反推，
             而仿真器用 ceil(吞吐需求) × redundancy_factor 正推，
             导致初始步（current_gpus=required_gpus）结果不完全一致。
          2. 不含冗余系数，直接反映 Agent 实际分配资源的运行状态。

        输入（来自仿真数据，反映真实流量负载，Agent 无法改变）：
            row['qps'], row['avg_input_tokens'], row['avg_output_tokens']

        输出（随 current_gpus / current_gpu_type 动态变化）：
            gpu_utilization, memory_utilization,
            estimated_latency_ms, hourly_cost, required_replicas
        """
        cfg      = self._model_cfg
        gpu_mem  = self.GPU_MEMORY_GB[self.current_gpu_type]
        gpu_cost = self.GPU_COST[self.current_gpu_type]

        qps               = float(row['qps'])
        avg_input_tokens  = float(row['avg_input_tokens'])
        avg_output_tokens = float(row['avg_output_tokens'])

        # ── 1. 吞吐量需求（与仿真器一致）────────────────────────────
        prefill_tokens    = qps * avg_input_tokens
        generation_tokens = qps * avg_output_tokens
        effective_tokens  = (prefill_tokens * cfg['prefill_compute_factor']
                             + generation_tokens)

        # ── 2. 内存需求 ───────────────────────────────────────────────
        model_memory         = cfg['model_size_b'] * cfg['memory_per_billion_params']
        concurrent_requests  = qps * 2
        kv_cache_total       = concurrent_requests * (avg_input_tokens + avg_output_tokens) * 0.001
        activation_memory    = model_memory * 0.2
        total_memory_replica = (model_memory + kv_cache_total + activation_memory) * cfg['batch_memory_factor']

        # ── 3. 单 GPU 吞吐（含效率折扣）────────────────────────────────
        tokens_per_gpu = cfg['tokens_per_second_per_gpu'][self.current_gpu_type] * cfg['gpu_efficiency']

        # ── 4. 每副本需要的 GPU 数（内存约束 vs 最小并行约束）──────────
        gpus_per_replica = max(
            cfg['min_gpus_per_replica'],
            int(np.ceil(total_memory_replica / gpu_mem)),
        )

        # ── 5. 当前分配的 GPU 数能支撑多少副本 ─────────────────────────
        #    注意：Agent 的 current_gpus 可能少于最优副本需求，
        #    超出部分会导致利用率 > 1（过载）
        actual_replicas   = max(1, self.current_gpus // gpus_per_replica)
        tokens_per_replica = gpus_per_replica * tokens_per_gpu

        # ── 6. GPU 利用率 ──────────────────────────────────────────────
        max_throughput    = actual_replicas * tokens_per_replica
        gpu_utilization   = float(min(effective_tokens / (max_throughput + 1e-8), 1.5))
        # 允许超过 1.0 以反映真实过载（奖励函数中会惩罚 > 0.95）

        # ── 7. 显存利用率 ──────────────────────────────────────────────
        actual_memory     = actual_replicas * total_memory_replica
        max_memory        = self.current_gpus * gpu_mem
        memory_utilization = float(min(actual_memory / (max_memory + 1e-8), 1.0))

        # ── 8. 延迟估算（队列深度由实际副本数决定）──────────────────────
        queue_depth           = concurrent_requests / actual_replicas
        estimated_latency_ms  = (50.0
                                 + (avg_input_tokens + avg_output_tokens) * 0.5
                                 + queue_depth * 10.0)

        # ── 9. 成本（当前实际 GPU 数 × 单价）──────────────────────────
        hourly_cost = self.current_gpus * gpu_cost

        return {
            'gpu_utilization':    gpu_utilization,
            'memory_utilization': memory_utilization,
            'estimated_latency_ms': estimated_latency_ms,
            'hourly_cost':        hourly_cost,
            'required_replicas':  actual_replicas,
            'effective_tokens':   effective_tokens,
        }

    def _instant_state(self) -> np.ndarray:
        """当前即时运营状态，6 维（基于动态重算结果）。"""
        row     = self.df.iloc[self.current_step]
        metrics = self._recompute_metrics(row)
        return np.array([
            self.current_gpus / 500.0,                          # GPU 数（归一化）
            min(metrics['gpu_utilization'], 1.0),               # 计算利用率（截断到1）
            metrics['memory_utilization'],                      # 显存利用率
            metrics['estimated_latency_ms'] / 2000.0,           # 延迟（归一化）
            self.cooldown / self.cooldown_steps,                # 冷却进度
            metrics['required_replicas'] / 50.0,                # 实际副本数（归一化）
        ], dtype=np.float32)

    def _compute_reward(self, metrics: dict, action: int) -> float:
        """
        多目标加权奖励函数（所有项均基于动态重算的 metrics）：
            R = 0.40 × util_reward
              + 0.30 × sla_reward
              + 0.25 × cost_penalty
              + 0.10 × change_penalty

        util_reward 设计（分段固定值，无魔法系数）：
            [0.70, 0.88]        → +1.0  最优区间
            [0.60, 0.70)∪(0.88,0.95] → +0.5  次优区间
            (0.95, ∞)           → -0.5  过载（合并了原 overload_penalty）
            [0, 0.60)           → (util-0.60)*5.0  线性下降，0.60→0，0.40→-1.0

        cost_penalty 设计：
            以数据中位成本 ~400 USD/h 为基准：
                cost_penalty = -(cost / 400.0 - 1.0) * 0.5
            中位成本时惩罚=0，成本越高惩罚线性增大，成本较低时有轻微正激励。

        change_penalty 设计：
            - 动作 4（不变）：0
            - 扩缩容动作（±8/±16 GPU）：-(delta/16)，最大 -1.0
            - GPU 型号切换（动作 5/6）：-0.5（固定）
        """
        util    = metrics['gpu_utilization']
        latency = metrics['estimated_latency_ms']
        cost    = metrics['hourly_cost']

        # ── 利用率奖励：全区间连续，无跳变 ─────────────────────────────
        # 最优区间 [0.70, 0.88]：满分 +1.0
        # 次优区间 [0.60, 0.70) ∪ (0.88, 0.95]：部分正奖励 +0.5
        # 过低区间 [0, 0.60)：线性下降，0.60→0，0.40→-1.0，提供向上的连续梯度
        # 过载区间 (0.95, ∞)：从 +0.5 开始线性下降（斜率 5.0），与次优区间平滑衔接，
        #   无跳变：util=0.95→+0.50，util=1.0→+0.25，util=1.1→-0.25，util=1.5→-2.25
        #   过载越严重惩罚越重，但不设悬崖，策略能感受到连续梯度信号
        if 0.70 <= util <= 0.88:
            util_reward = 1.0
        elif 0.60 <= util < 0.70 or 0.88 < util <= 0.95:
            util_reward = 0.5
        elif util > 0.95:
            util_reward = 0.5 - (util - 0.95) * 5.0   # 线性延伸：无跳变，util=1.0→+0.25
        else:                                           # util < 0.60
            util_reward = (util - 0.60) * 5.0          # 线性：0.60→0，0.40→-1.0

        # ── SLA 奖励：以 p50 延迟（约1500ms）为基准，低于基准有正奖励，高于线性惩罚 ──
        # 原阈值 500ms 达标率 0% → 改为 1200ms（p75）仍偏严，梯度信号不足。
        # 改为相对基准衡量：低于 SLA_BASELINE 奖励为正，高于则线性惩罚，clamp 到 [-1, 1]。
        SLA_BASELINE = 1500.0   # ChatGLM Daily 场景 p50 延迟（约1500ms）
        sla_reward = (SLA_BASELINE - latency) / SLA_BASELINE
        sla_reward = max(sla_reward, -1.0)

        # ── 成本惩罚（相对中位成本归一化）─────────────────────────────
        # 以数据中位成本 ~400 USD/h 为基准：中位成本时惩罚=0，成本翻倍时惩罚=-0.5
        # 比旧版 -cost*0.001 的量级（加权后约 -0.06）与其他奖励项更匹配
        cost_penalty = -(cost / 400.0 - 1.0) * 0.5

        # ── 频繁调整惩罚（按变化幅度比例，防止策略过度抖动）──────────
        # ACTION_DELTA: {0:+8, 1:+16, 2:-8, 3:-16, 4:0, 5:0(切H100), 6:0(切A100)}
        # 扩缩容：惩罚与 GPU 变化幅度成比例，最大变化 16 GPU → 惩罚 -1.0
        # GPU 型号切换（动作 5/6）：无 GPU 数量变化但有运维成本，固定 -0.5
        # 保持不变（动作 4）：无惩罚
        delta = abs(self.ACTION_DELTA.get(action, 0))
        if delta > 0:
            change_penalty = -(delta / 16.0)   # ±8 GPU → -0.5，±16 GPU → -1.0
        elif action in self.GPU_SWITCH:
            change_penalty = -0.5              # GPU 型号切换固定惩罚
        else:
            change_penalty = 0.0               # 动作 4，不变

        # 注：过载惩罚已合并到 util_reward（util>0.95 时固定 -0.5），不再单独设置 overload_penalty

        return float(
            0.40 * util_reward
            + 0.30 * sla_reward
            + 0.25 * cost_penalty
            + 0.10 * change_penalty
        )


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────

def _smoke_test():
    """不依赖真实文件的冒烟测试，验证环境接口正确性。"""
    import tempfile, json

    # ── 构造最小仿真数据 ──────────────────────────────────────────
    T = 300
    np.random.seed(0)
    ts = pd.date_range("2024-01-01", periods=T, freq="1min")
    df = pd.DataFrame({
        'timestamp':                 ts,
        'qps':                       np.random.uniform(50, 200, T),
        'avg_input_tokens':          np.random.uniform(200, 2000, T),
        'avg_output_tokens':         np.random.uniform(200, 3000, T),
        'effective_tokens_per_second': np.random.uniform(1e4, 5e5, T),
        'concurrent_requests':       np.random.uniform(10, 400, T),
        'total_memory_gb':           np.random.uniform(500, 5000, T),
        'required_gpus':             np.random.randint(32, 256, T),
        'required_servers':          np.random.randint(4, 32, T),
        'gpu_utilization':           np.random.uniform(0.3, 0.9, T),
        'memory_utilization':        np.random.uniform(0.3, 0.9, T),
        'estimated_latency_ms':      np.random.uniform(100, 1000, T),
        'model_type':                'GPT-4',
        'gpu_type':                  'H100',
    })

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        data_path = f.name
        df.to_csv(data_path, index=False)

    # ── 构造最小 checkpoint ──────────────────────────────────────
    feat_dim = 27  # extract_features 输出维度
    dummy_model = AttentionLSTM(
        input_dim=feat_dim, hidden_dim=128, output_dim=1, num_layers=3, dropout=0.0
    )
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        ckpt_path = f.name
        torch.save({
            'model_state_dict': dummy_model.state_dict(),
            'feature_dim':      feat_dim,
            'feature_scaler':   None,
            'use_log_transform': True,
            'config': {'model_type': 'lstm', 'sequence_length': 60},
        }, ckpt_path)

    # ── 运行环境 ──────────────────────────────────────────────────
    print("▶ 创建环境...")
    env = PredictiveResourceEnv(
        data_path=data_path,
        checkpoint_path=ckpt_path,
        model_type='GPT-4',
        seq_len=60,
        device='cpu',
    )

    obs, info = env.reset(seed=42)
    assert obs.shape == (262,), f"obs shape 错误: {obs.shape}"
    print(f"  reset OK | obs.shape={obs.shape}")

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated:
            break

    assert obs.shape == (262,)
    print(f"\n✅ 冒烟测试通过！10步累计奖励={total_reward:.3f}")

    # 清理临时文件
    os.unlink(data_path)
    os.unlink(ckpt_path)


if __name__ == '__main__':
    _smoke_test()
