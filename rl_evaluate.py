#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_evaluate.py - RL 策略评估与基线对比

支持四种策略的评估与对比：
    1. RL 策略      - 从训练好的 PPO 模型加载
    2. 规则基线     - 直接使用仿真数据中的 required_gpus（resource_simulation.py 的规则方法）
    3. 增强规则基线 - 规则基线 + GPU 型号自动切换（高负载→H100，低负载→A100）
    4. 保守基线     - 始终保持当前 GPU 数不变（action=4）

评估指标：
    - 平均 GPU 利用率
    - 平均显存利用率
    - SLA 满足率（延迟 < 1500ms 的时间步占比，与 rl_env SLA_BASELINE 对齐）
    - 平均每小时成本
    - 每 QPS 平均成本
    - 利用率过载率（util > 95%）
    - 平均奖励
    - 平均副本数
    - 频繁调整惩罚（change_penalty，与 rl_env._compute_reward 对齐）
        扩缩容：-(delta/16)，±8GPU→-0.5，±16GPU→-1.0
        GPU型号切换（动作5/6）：-0.5
        保持不变（动作4）：0.0
    - 调整动作频率（非 hold 动作占比）

环境状态空间（rl_env.py 最新版）：
    256 维时序语义编码（QPSAwareStateEncoder）
    +  6 维即时运营状态（GPU数/计算利用率/显存利用率/延迟/冷却进度/副本数）
    = 262 维

动作空间（7 个离散动作）：
    0: 扩容 +8  GPU（+1台服务器）
    1: 扩容 +16 GPU（+2台服务器）
    2: 缩容 -8  GPU（-1台服务器）
    3: 缩容 -16 GPU（-2台服务器）
    4: 保持不变
    5: 切换至 H100（高负载省显存）
    6: 切换至 A100（低负载省成本）

奖励函数（rl_env.py 最新版）：
    R = 0.40 × util_reward
      + 0.30 × sla_reward     (SLA_BASELINE=1500ms)
      + 0.25 × cost_penalty   (相对中位成本~400$/h 归一化)
      + 0.10 × change_penalty (按 GPU 变化幅度比例，最大-1.0)

    util_reward 分段：
      [0.70, 0.88] → +1.0，[0.60,0.70)∪(0.88,0.95] → +0.5
      >0.95 → -0.5（过载），<0.60 → (util-0.60)*5.0（线性下降）

用法：
    python rl_evaluate.py \\
        --data   simulation_data/combined_simulation_data.csv \\
        --ckpt   models/lstm_traffic_best.pth \\
        --rl-model rl_models/phase2/best/best_model \\
        --model-type GPT-4 \\
        --episodes 10
"""

import os
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from rl_env import PredictiveResourceEnv

plt.rcParams['axes.unicode_minus'] = False

# SLA 阈值与 rl_env._compute_reward 保持一致
SLA_THRESHOLD_MS = 1500.0

# 与 rl_env.ACTION_DELTA / GPU_SWITCH 保持一致
_ACTION_DELTA  = {0: 8, 1: 16, 2: -8, 3: -16, 4: 0, 5: 0, 6: 0}
_GPU_SWITCH    = {5, 6}


def _compute_change_penalty(action: int) -> float:
    """
    计算单步频繁调整惩罚，与 rl_env.PredictiveResourceEnv._compute_reward 完全对齐：

        - 扩缩容动作（0/1/2/3）：-(|delta| / 16.0)
            ±8  GPU（动作 0/2）→ -0.5
            ±16 GPU（动作 1/3）→ -1.0
        - GPU 型号切换（动作 5/6）：-0.5（固定）
        - 保持不变（动作 4）：0.0

    Args:
        action: 离散动作编号（0~6）

    Returns:
        change_penalty: float，取值范围 [-1.0, 0.0]
    """
    delta = abs(_ACTION_DELTA.get(action, 0))
    if delta > 0:
        return -(delta / 16.0)
    elif action in _GPU_SWITCH:
        return -0.5
    else:
        return 0.0


# ─────────────────────────────────────────────
# 评估单个 Episode
# ─────────────────────────────────────────────

def _rollout(
    env: PredictiveResourceEnv,
    policy,            # callable(obs) -> action，或字符串 'rule'/'hold'
    max_steps: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """
    运行一条完整轨迹，返回统计指标字典。

    policy 可以是：
        - PPO 模型对象（调用 model.predict(obs, deterministic=True)）
        - 'rule'：规则基线，直接使用数据中的 required_gpus 追赶调整
        - 'hold'：保守基线，始终 action=4（不变）

    动作空间（与 rl_env.ACTION_DELTA 完全对齐）：
        0: +8 GPU   1: +16 GPU   2: -8 GPU   3: -16 GPU
        4: 不变     5: 切换H100  6: 切换A100
    """
    obs, _ = env.reset(seed=seed)

    episode_rewards    = []
    gpu_utils          = []
    mem_utils          = []
    latencies          = []
    costs              = []
    qps_list           = []
    gpu_counts         = []
    replicas_list      = []
    sla_satisfied      = []
    overloaded         = []
    change_penalties   = []   # 每步频繁调整惩罚（与 rl_env._compute_reward 对齐）
    action_hist        = [0] * env.action_space.n   # 7 个动作

    for step in range(max_steps):
        # ── 选择动作 ──────────────────────────────────────────────
        if policy == 'hold':
            action = 4  # 永远不变

        elif policy == 'rule':
            # 规则基线：每步根据数据中 required_gpus 做追赶调整
            # ACTION_DELTA: {0:+8, 1:+16, 2:-8, 3:-16, 4:0, 5:H100, 6:A100}
            row         = env.df.iloc[env.current_step]
            target_gpus = int(row['required_gpus'])
            diff        = target_gpus - env.current_gpus
            if diff >= 16:
                action = 1   # +16 GPU
            elif diff >= 8:
                action = 0   # +8  GPU
            elif diff <= -16 and env.cooldown == 0:
                action = 3   # -16 GPU
            elif diff <= -8 and env.cooldown == 0:
                action = 2   # -8  GPU
            else:
                action = 4   # 保持

        elif policy == 'rule_enhanced':
            # 增强规则基线：在规则基线基础上叠加 GPU 型号切换逻辑
            # 高负载时切换至 H100（省显存、降延迟），低负载时切回 A100（省成本）
            row         = env.df.iloc[env.current_step]
            target_gpus = int(row['required_gpus'])
            diff        = target_gpus - env.current_gpus
            util        = float(row['gpu_utilization'])

            if util > 0.85 and env.current_gpu_type == 'A100' and env.cooldown == 0:
                action = 5  # 切换至 H100
            elif util < 0.40 and env.current_gpu_type == 'H100' and env.cooldown == 0:
                action = 6  # 切换回 A100
            elif diff >= 16:
                action = 1  # +16 GPU
            elif diff >= 8:
                action = 0  # +8  GPU
            elif diff <= -16 and env.cooldown == 0:
                action = 3  # -16 GPU
            elif diff <= -8 and env.cooldown == 0:
                action = 2  # -8  GPU
            else:
                action = 4  # 保持

        else:
            # PPO 模型：确定性推理
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)

        action_hist[action] += 1
        change_penalties.append(_compute_change_penalty(action))

        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        gpu_utils.append(info['gpu_utilization'])
        mem_utils.append(info['memory_utilization'])
        latencies.append(info['estimated_latency'])
        costs.append(info['hourly_cost'])
        qps_list.append(info['qps'])
        gpu_counts.append(info['current_gpus'])
        replicas_list.append(info['required_replicas'])
        # SLA 阈值与 rl_env.SLA_BASELINE 保持一致：1500ms
        sla_satisfied.append(1 if info['estimated_latency'] < SLA_THRESHOLD_MS else 0)
        overloaded.append(1 if info['gpu_utilization'] > 0.95 else 0)

        if terminated or truncated:
            break

    n = len(episode_rewards)
    # 调整动作频率：非「保持不变」（action≠4）的动作占比
    adjust_rate = float(np.mean([1 if p < 0 else 0 for p in change_penalties]))
    return {
        'steps':            n,
        'total_reward':     float(np.sum(episode_rewards)),
        'mean_reward':      float(np.mean(episode_rewards)),
        'mean_util':        float(np.mean(gpu_utils)),
        'std_util':         float(np.std(gpu_utils)),
        'mean_mem_util':    float(np.mean(mem_utils)),
        'p95_latency':      float(np.percentile(latencies, 95)),
        'mean_latency':     float(np.mean(latencies)),
        'mean_cost':        float(np.mean(costs)),
        'mean_qps':         float(np.mean(qps_list)),
        'cost_per_qps':     float(np.mean(costs)) / (float(np.mean(qps_list)) + 1e-8),
        'sla_rate':         float(np.mean(sla_satisfied)),
        'overload_rate':    float(np.mean(overloaded)),
        'mean_gpus':        float(np.mean(gpu_counts)),
        'mean_replicas':    float(np.mean(replicas_list)),
        # ── 频繁调整惩罚（与 rl_env._compute_reward change_penalty 对齐）──
        'mean_change_penalty':  float(np.mean(change_penalties)),  # 平均每步惩罚值
        'total_change_penalty': float(np.sum(change_penalties)),   # 全局累计惩罚
        'adjust_rate':          adjust_rate,                       # 调整动作频率
        'action_dist':      action_hist,
        # 时序数据（供绘图）
        '_ts_util':           gpu_utils,
        '_ts_mem_util':       mem_utils,
        '_ts_latency':        latencies,
        '_ts_cost':           costs,
        '_ts_gpus':           gpu_counts,
        '_ts_qps':            qps_list,
        '_ts_replicas':       replicas_list,
        '_ts_change_penalty': change_penalties,
    }


# ─────────────────────────────────────────────
# 多 Episode 评估
# ─────────────────────────────────────────────

def evaluate_policy(
    env: PredictiveResourceEnv,
    policy,
    n_episodes: int = 10,
    max_steps: int = 2000,
    policy_name: str = 'policy',
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """运行 n_episodes 条轨迹，汇总统计量。"""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\n{'─'*50}")
    logger.info(f"评估策略: {policy_name}  ({n_episodes} 个 episode)")

    episodes = []
    for ep in range(n_episodes):
        result = _rollout(env, policy, max_steps=max_steps, seed=ep * 1000)
        episodes.append(result)
        logger.info(
            f"  Ep {ep+1:2d}/{n_episodes} | "
            f"steps={result['steps']:4d} | "
            f"reward={result['mean_reward']:+.3f} | "
            f"util={result['mean_util']:.2%} | "
            f"mem={result['mean_mem_util']:.2%} | "
            f"SLA={result['sla_rate']:.2%} | "
            f"cost=${result['mean_cost']:.1f}/h | "
            f"replicas={result['mean_replicas']:.1f}"
        )

    # 汇总
    def _agg(key):
        return [ep[key] for ep in episodes]

    summary = {
        'policy':              policy_name,
        'episodes':            n_episodes,
        'mean_reward':         float(np.mean(_agg('mean_reward'))),
        'std_reward':          float(np.std(_agg('mean_reward'))),
        'mean_util':           float(np.mean(_agg('mean_util'))),
        'std_util':            float(np.std(_agg('std_util'))),
        'mean_mem_util':       float(np.mean(_agg('mean_mem_util'))),
        'p95_latency':         float(np.mean(_agg('p95_latency'))),
        'mean_latency':        float(np.mean(_agg('mean_latency'))),
        'mean_cost':           float(np.mean(_agg('mean_cost'))),
        'cost_per_qps':        float(np.mean(_agg('cost_per_qps'))),
        'sla_rate':            float(np.mean(_agg('sla_rate'))),
        'overload_rate':       float(np.mean(_agg('overload_rate'))),
        'mean_gpus':           float(np.mean(_agg('mean_gpus'))),
        'mean_replicas':       float(np.mean(_agg('mean_replicas'))),
        # ── 频繁调整惩罚汇总（与 rl_env._compute_reward change_penalty 对齐）──
        'mean_change_penalty': float(np.mean(_agg('mean_change_penalty'))),
        'total_change_penalty': float(np.mean(_agg('total_change_penalty'))),
        'adjust_rate':         float(np.mean(_agg('adjust_rate'))),
        '_episodes_raw':       episodes,   # 保留原始数据供绘图
    }

    logger.info(
        f"\n  汇总 [{policy_name}]:\n"
        f"    平均奖励:        {summary['mean_reward']:+.4f} ± {summary['std_reward']:.4f}\n"
        f"    GPU 利用率:      {summary['mean_util']:.2%} ± {summary['std_util']:.2%}\n"
        f"    显存利用率:      {summary['mean_mem_util']:.2%}\n"
        f"    SLA 满足率:      {summary['sla_rate']:.2%}  (阈值={SLA_THRESHOLD_MS:.0f}ms)\n"
        f"    过载率:          {summary['overload_rate']:.2%}\n"
        f"    平均延迟:        {summary['mean_latency']:.1f} ms (P95={summary['p95_latency']:.1f})\n"
        f"    平均成本:        ${summary['mean_cost']:.2f}/h\n"
        f"    成本/QPS:        ${summary['cost_per_qps']:.4f}\n"
        f"    平均 GPU 数:     {summary['mean_gpus']:.1f}\n"
        f"    平均副本数:      {summary['mean_replicas']:.1f}\n"
        f"    调整惩罚(均值):  {summary['mean_change_penalty']:+.4f}  "
        f"(越接近0越稳定，-1.0为最大惩罚)\n"
        f"    调整动作频率:    {summary['adjust_rate']:.2%}  "
        f"(非hold动作占比，越低越稳定)\n"
    )
    return summary


# ─────────────────────────────────────────────
# 打印对比表格
# ─────────────────────────────────────────────

def print_comparison_table(summaries: List[Dict], logger: logging.Logger):
    """将多个策略的评估结果并排打印为对比表格。"""
    metrics = [
        ('mean_reward',        '平均奖励',      '{:+.4f}'),
        ('mean_util',          'GPU 利用率',    '{:.2%}'),
        ('mean_mem_util',      '显存利用率',    '{:.2%}'),
        ('sla_rate',           'SLA 满足率',    '{:.2%}'),
        ('overload_rate',      '过载率',        '{:.2%}'),
        ('p95_latency',        'P95 延迟(ms)',  '{:.1f}'),
        ('mean_cost',          '成本($/h)',     '{:.2f}'),
        ('cost_per_qps',       '成本/QPS',     '{:.4f}'),
        ('mean_gpus',          '平均GPU数',     '{:.1f}'),
        ('mean_replicas',      '平均副本数',    '{:.1f}'),
        ('mean_change_penalty','调整惩罚均值',  '{:+.4f}'),
        ('adjust_rate',        '调整动作频率',  '{:.2%}'),
    ]

    names  = [s['policy'] for s in summaries]
    col_w  = max(14, max(len(n) for n in names) + 2)
    header = f"{'指标':<16}" + "".join(f"{n:>{col_w}}" for n in names)

    logger.info("\n" + "═" * len(header))
    logger.info("  策略对比汇总")
    logger.info("═" * len(header))
    logger.info(header)
    logger.info("─" * len(header))

    for key, label, fmt in metrics:
        row = f"{label:<16}"
        for s in summaries:
            row += f"{fmt.format(s[key]):>{col_w}}"
        logger.info(row)

    logger.info("═" * len(header))

    # 找出最优策略（按平均奖励）
    best = max(summaries, key=lambda s: s['mean_reward'])
    logger.info(f"\n🏆 最优策略: {best['policy']}  (mean_reward={best['mean_reward']:+.4f})")


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────

def plot_comparison(summaries: List[Dict], output_dir: str, logger: logging.Logger):
    """生成对比折线图和柱状图，保存到 output_dir。"""
    os.makedirs(output_dir, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    names  = [s['policy'] for s in summaries]

    # ── 图1：各策略单 episode 时序对比（取第1个 episode）──────────────
    # 展示 6 条时序曲线：GPU利用率 / 显存利用率 / 延迟 / 成本 / GPU数 / 副本数
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Strategy Comparison - Episode 0 Time Series', fontsize=14, fontweight='bold')

    ts_keys   = ['_ts_util', '_ts_mem_util', '_ts_latency', '_ts_cost', '_ts_gpus', '_ts_replicas']
    ts_labels = ['GPU Utilization', 'Memory Utilization', 'Latency (ms)',
                 'Hourly Cost ($)', 'GPU Count', 'Replica Count']
    ts_targets = [None, None, SLA_THRESHOLD_MS, None, None, None]   # SLA 参考线

    for col, (key, label, target) in enumerate(zip(ts_keys, ts_labels, ts_targets)):
        ax = axes[col // 3][col % 3]
        for i, s in enumerate(summaries):
            ep0 = s['_episodes_raw'][0]
            y   = ep0[key]
            ax.plot(y, label=s['policy'], color=colors[i % len(colors)],
                    alpha=0.8, linewidth=1.2)
        if target is not None:
            ax.axhline(target, color='red', linestyle='--', linewidth=1,
                       label=f'SLA={target:.0f}ms')
        ax.set_title(label)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(output_dir, 'timeseries_comparison.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  图表已保存: {out1}")

    # ── 图2：指标柱状图对比 ────────────────────────────────────────
    bar_metrics = [
        ('mean_reward',        '平均奖励'),
        ('mean_util',          'GPU 利用率'),
        ('mean_mem_util',      '显存利用率'),
        ('sla_rate',           'SLA 满足率'),
        ('overload_rate',      '过载率'),
        ('mean_cost',          '成本 ($/h)'),
        ('cost_per_qps',       '成本/QPS'),
        ('mean_replicas',      '平均副本数'),
        ('mean_change_penalty','调整惩罚均值'),
        ('adjust_rate',        '调整动作频率'),
    ]

    n_metrics = len(bar_metrics)
    n_cols = 5
    n_rows = (n_metrics + n_cols - 1) // n_cols   # 向上取整
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle('Strategy Comparison - Key Metrics', fontsize=14, fontweight='bold')

    for idx, (key, label) in enumerate(bar_metrics):
        ax   = axes[idx // n_cols][idx % n_cols]
        vals = [s[key] for s in summaries]
        bars = ax.bar(names, vals, color=colors[:len(names)], alpha=0.85, edgecolor='white')

        # 在柱顶标注数值
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9,
            )

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    # 隐藏多余子图（如果 n_metrics 不能整除 n_cols）
    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    out2 = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  图表已保存: {out2}")

    # ── 图3：动作分布对比 ──────────────────────────────────────────
    # 与 rl_env.ACTION_DELTA 完全对齐：
    #   0:+8GPU  1:+16GPU  2:-8GPU  3:-16GPU  4:不变  5:→H100  6:→A100
    action_labels = ['+8GPU', '+16GPU', '-8GPU', '-16GPU', '不变', '→H100', '→A100']
    fig, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 5))
    if len(summaries) == 1:
        axes = [axes]

    fig.suptitle('Action Distribution', fontsize=14, fontweight='bold')
    for i, s in enumerate(summaries):
        # 汇总所有 episode 的动作统计
        total_actions = [0] * env_action_n
        for ep in s['_episodes_raw']:
            for j, cnt in enumerate(ep['action_dist']):
                if j < env_action_n:
                    total_actions[j] += cnt
        total = sum(total_actions) + 1e-8
        fracs = [c / total for c in total_actions]
        axes[i].bar(action_labels[:env_action_n], fracs,
                    color=colors[i % len(colors)], alpha=0.85)
        axes[i].set_title(s['policy'])
        axes[i].set_ylabel('Fraction')
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out3 = os.path.join(output_dir, 'action_distribution.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  图表已保存: {out3}")


# ─────────────────────────────────────────────
# 保存评估结果为 CSV
# ─────────────────────────────────────────────

def save_results(summaries: List[Dict], output_dir: str, logger: logging.Logger):
    """将评估结果保存为 CSV，便于后续分析。"""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for s in summaries:
        row = {k: v for k, v in s.items()
               if not k.startswith('_') and k != 'action_dist'}
        rows.append(row)

    df  = pd.DataFrame(rows)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(output_dir, f'evaluation_results_{ts}.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    logger.info(f"\n📄 评估结果已保存: {out}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

# 全局变量，供 plot_comparison 使用
env_action_n: int = 7


def main():
    global env_action_n

    parser = argparse.ArgumentParser(
        description='RL 策略评估与规则基线对比',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data',       type=str,
                        default='simulation_data/combined_simulation_data.csv',
                        help='仿真数据路径')
    parser.add_argument('--ckpt',       type=str, required=True,
                        help='预训练 LSTM checkpoint 路径（.pth）')
    parser.add_argument('--rl-model',   type=str, default=None,
                        help='训练好的 PPO 模型路径（不含 .zip 后缀）')
    parser.add_argument('--vecnorm',    type=str, default=None,
                        help='VecNormalize 统计量路径（_vecnorm.pkl）')
    parser.add_argument('--model-type', type=str, default='GPT-4',
                        choices=['GPT-4', 'ChatGLM', 'Claude', 'LLaMA'])
    parser.add_argument('--gpu-type',   type=str, default='A100',
                        choices=['A100', 'H100'])
    parser.add_argument('--seq-len',    type=int, default=60)
    parser.add_argument('--episodes',   type=int, default=10,
                        help='每个策略评估的 episode 数')
    parser.add_argument('--max-steps',  type=int, default=2000,
                        help='每个 episode 最大步数')
    parser.add_argument('--output',     type=str, default='rl_eval_results',
                        help='评估结果输出目录')
    parser.add_argument('--device',     type=str, default='cuda')

    args = parser.parse_args()

    # ── 日志 ──────────────────────────────────────────────────────
    log_dir = os.path.join(args.output, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger  = logging.getLogger('rl_evaluate')
    logger.setLevel(logging.INFO)
    fmt     = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    fh      = logging.FileHandler(os.path.join(log_dir, f'evaluate_{ts}.log'))
    ch      = logging.StreamHandler()
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

    logger.info("═" * 60)
    logger.info("  RL 策略评估与对比")
    logger.info(f"  model_type={args.model_type}, gpu_type={args.gpu_type}")
    logger.info(f"  episodes={args.episodes}, max_steps={args.max_steps}")
    logger.info(f"  SLA 阈值={SLA_THRESHOLD_MS:.0f}ms（与 rl_env.SLA_BASELINE 对齐）")
    logger.info("═" * 60)

    # ── 创建环境（单进程，不需要 VecEnv）────────────────────────────
    # obs_dim = QPSAwareStateEncoder.ENCODER_OUT_DIM(256) + 6(即时状态) = 262
    logger.info("\n▶ 初始化评估环境...")
    env = PredictiveResourceEnv(
        data_path=args.data,
        checkpoint_path=args.ckpt,
        model_type=args.model_type,
        gpu_type=args.gpu_type,
        seq_len=args.seq_len,
        device=args.device,
    )
    env_action_n = env.action_space.n   # 7（与 rl_env.ACTION_DELTA 对齐）
    logger.info(f"  环境初始化完成 | obs_dim={env.observation_space.shape[0]}, "
                f"action_n={env_action_n}")

    summaries = []

    # ── 评估：规则基线 ────────────────────────────────────────────
    rule_summary = evaluate_policy(
        env, policy='rule',
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        policy_name='规则基线',
        logger=logger,
    )
    summaries.append(rule_summary)

    # ── 评估：增强规则基线 ────────────────────────────────────────
    rule_enh_summary = evaluate_policy(
        env, policy='rule_enhanced',
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        policy_name='增强规则基线',
        logger=logger,
    )
    summaries.append(rule_enh_summary)

    # ── 评估：保守基线（不变）────────────────────────────────────
    hold_summary = evaluate_policy(
        env, policy='hold',
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        policy_name='保守基线(不变)',
        logger=logger,
    )
    summaries.append(hold_summary)

    # ── 评估：RL 策略 ─────────────────────────────────────────────
    if args.rl_model is not None:
        if not os.path.exists(args.rl_model + '.zip'):
            logger.warning(f"RL 模型文件不存在: {args.rl_model}.zip，跳过 RL 策略评估")
        else:
            logger.info(f"\n▶ 加载 PPO 模型: {args.rl_model}")
            rl_model = PPO.load(args.rl_model, device=args.device)

            # 若有 VecNormalize 统计量，对 obs 做同样的归一化
            if args.vecnorm and os.path.exists(args.vecnorm):
                logger.info(f"  加载 VecNormalize: {args.vecnorm}")
                dummy_env = DummyVecEnv([lambda: env])
                vec_env   = VecNormalize.load(args.vecnorm, dummy_env)
                vec_env.training = False

                # 包装 predict：在外部做 obs 归一化（与训练时 VecNormalize 保持一致）
                class _NormPolicy:
                    def __init__(self, model, vec_norm):
                        self._m = model
                        self._v = vec_norm
                    def predict(self, obs, deterministic=True):
                        obs_n = self._v.normalize_obs(obs.reshape(1, -1)).flatten()
                        return self._m.predict(obs_n, deterministic=deterministic)

                rl_policy = _NormPolicy(rl_model, vec_env)
            else:
                rl_policy = rl_model

            rl_summary = evaluate_policy(
                env, policy=rl_policy,
                n_episodes=args.episodes,
                max_steps=args.max_steps,
                policy_name='RL策略(PPO)',
                logger=logger,
            )
            summaries.append(rl_summary)
    else:
        logger.info("\n⚠  未指定 --rl-model，跳过 RL 策略评估（仅对比基线）")

    # ── 对比表格 ──────────────────────────────────────────────────
    print_comparison_table(summaries, logger)

    # ── 可视化 ────────────────────────────────────────────────────
    logger.info("\n▶ 生成对比图表...")
    plot_comparison(summaries, output_dir=os.path.join(args.output, 'plots'), logger=logger)

    # ── 保存结果 ──────────────────────────────────────────────────
    save_results(summaries, output_dir=args.output, logger=logger)

    logger.info("\n✅ 评估完成！")


if __name__ == '__main__':
    main()
