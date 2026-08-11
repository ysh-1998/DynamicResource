#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_train_multigpu.py - 多卡两阶段 PPO 训练脚本

【多卡并行策略】
────────────────────────────────────────────────────────────────
每张 GPU 负责 n_envs_per_gpu 个子环境的 rollout（数据采集）：

  Worker 进程（每卡一个）：
    - 在本卡上运行 DummyVecEnv（n_envs_per_gpu 个 env 在进程内顺序执行）
    - 接收主进程广播的最新 policy 参数
    - 执行 n_steps 步动作采样，收集 rollout buffer（obs/action/reward/value/log_prob）
    - 将 RolloutData 通过 multiprocessing.Queue 发送给主进程

  主进程（cuda:0）：
    - 汇总所有 Worker 的 RolloutData，沿 env 维度拼接
    - 计算 GAE advantages & returns
    - 执行 PPO minibatch update（多个 epoch）
    - 将新参数广播给所有 Worker（通过各 Worker 的 param_queue）
    - 定期评估、保存 checkpoint

通信方案：
  参数广播：pickle 序列化 {state_dict, 全局 obs_rms/ret_rms} → mp.Queue（主进程 → Worker）
            Worker 收到后用广播的 obs_rms/ret_rms 覆盖本地统计，保证同一轮内所有
            Worker 使用同一份归一化尺度（等价于单卡多 env 共享一份 VecNormalize）。
  数据汇总：pickle 序列化 {RolloutData, 本轮增量 obs_stats/ret_stats} → mp.Queue（Worker → 主进程）
            增量统计（count 从 0 开始，只含本轮新采集数据）由主进程用 Welford 算法
            合并后累加到全局统计，避免重复计入广播时的"全局起点"。
  停止信号：mp.Event + None 哨兵值

【架构图】
  Main Process (cuda:0)
    │  ←── rollout buffer ──  Worker-0 (cuda:0, n_envs_per_gpu envs)
    │  ←── rollout buffer ──  Worker-1 (cuda:1, n_envs_per_gpu envs)
    │  ←── rollout buffer ──  Worker-N (cuda:N, n_envs_per_gpu envs)
    │
    ├── 合并 buffer (n_steps × n_gpus × n_envs_per_gpu 样本)
    ├── GAE 计算
    ├── PPO minibatch update（n_epochs 轮）
    └── 广播新参数 → 各 Worker

【用法示例】
  # 4 卡，每卡 4 个 env（共 16 env），Phase1+Phase2 完整训练
  python rl_train_multigpu.py \\
      --data   simulation_data/combined_simulation_data.csv \\
      --ckpt   models/lstm_traffic_best.pth \\
      --model-type GPT-4 \\
      --n-gpus 4 --n-envs-per-gpu 4 \\
      --phase1-steps 100000 --phase2-steps 200000 \\
      --output rl_models_multigpu/

  # 单卡调试（等价于 n-gpus=1）
  python rl_train_multigpu.py --data ... --ckpt ... --n-gpus 1 --n-envs-per-gpu 4

  # 只跑阶段 1
  python rl_train_multigpu.py --data ... --ckpt ... --phase2-steps 0
"""

import os
import sys
import time
import argparse
import logging
import pickle
from datetime import datetime
from typing import Optional, List, Dict, IO

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
import gymnasium

from rl_env import PredictiveResourceEnv


# ─────────────────────────────────────────────
# 日志工具（与单卡版保持一致）
# ─────────────────────────────────────────────

class _Tee:
    """同时将 stdout/stderr 写入文件，过滤 tqdm 进度条噪音行。"""

    import re as _re
    _ANSI_RE = _re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')
    _PROG_RE = _re.compile(r'^\s*[\d]+%|^\s*[|█▏▎▍▌▋▊▉━=>\s\[\]]+\s*$')

    def __init__(self, stream: IO, file_path: str):
        self._stream = stream
        self._file   = open(file_path, 'a', buffering=1, encoding='utf-8')

    @classmethod
    def _is_noise_line(cls, line: str) -> bool:
        cleaned = cls._ANSI_RE.sub('', line)
        stripped = cleaned.strip()
        if not stripped:
            return True
        if cls._PROG_RE.match(stripped):
            return True
        return False

    def write(self, data: str):
        self._stream.write(data)
        lines_to_write = []
        for segment in data.split('\r'):
            for line in segment.split('\n'):
                if not self._is_noise_line(line):
                    clean = self._ANSI_RE.sub('', line)
                    lines_to_write.append(clean)
        if lines_to_write:
            self._file.write('\n'.join(lines_to_write) + '\n')

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()

    def close(self):
        self._file.close()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _setup_logger(log_dir: str, tag: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = logging.getLogger(f"rl_train_mg.{tag}.{ts}")
    logger.setLevel(logging.INFO)
    fmt    = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                               datefmt='%Y-%m-%d %H:%M:%S')
    fh     = logging.FileHandler(os.path.join(log_dir, f'{tag}_{ts}.log'))
    ch     = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _redirect_stdout_stderr(log_dir: str, tag: str) -> str:
    """将 stdout/stderr tee 到 raw log 文件（保留 SB3 表格输出）。"""
    os.makedirs(log_dir, exist_ok=True)
    ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_log_path = os.path.join(log_dir, f'{tag}_raw_{ts}.log')
    sys.stdout   = _Tee(sys.stdout, raw_log_path)
    sys.stderr   = _Tee(sys.stderr, raw_log_path)
    return raw_log_path


# ─────────────────────────────────────────────
# 策略网络配置（与单卡版保持一致）
# ─────────────────────────────────────────────

PHASE1_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    activation_fn=torch.nn.ReLU,
)

PHASE2_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 128], vf=[256, 128]),
    activation_fn=torch.nn.ReLU,
)


# ─────────────────────────────────────────────
# 环境工厂（与单卡版保持一致）
# ─────────────────────────────────────────────

def _make_env_fn(data_path: str, ckpt_path: str, model_type: str,
                 gpu_type: str, seq_len: int, freeze: bool,
                 env_device: str = 'cpu'):
    """返回供 make_vec_env 使用的无参环境工厂函数。"""
    def _init():
        env = PredictiveResourceEnv(
            data_path=data_path,
            checkpoint_path=ckpt_path,
            model_type=model_type,
            gpu_type=gpu_type,
            seq_len=seq_len,
            device=env_device,
        )
        if freeze:
            env.encoder.freeze_encoder()
        else:
            env.encoder.unfreeze_encoder()
        return Monitor(env)
    return _init


# ─────────────────────────────────────────────
# Worker 参数容器（定义在 worker 函数前，保证 pickle 可用）
# ─────────────────────────────────────────────

class _WorkerArgs:
    """轻量级参数容器，供 Worker 进程使用（pickle 安全）。"""
    __slots__ = [
        'data', 'ckpt', 'model_type', 'gpu_type', 'seq_len',
        'n_envs_per_gpu', 'n_steps', 'batch_size', 'policy_kwargs',
    ]

    def __init__(self, data: str, ckpt: str, model_type: str, gpu_type: str,
                 seq_len: int, n_envs_per_gpu: int, n_steps: int, batch_size: int,
                 policy_kwargs: dict):
        self.data           = data
        self.ckpt           = ckpt
        self.model_type     = model_type
        self.gpu_type       = gpu_type
        self.seq_len        = seq_len
        self.n_envs_per_gpu = n_envs_per_gpu
        self.n_steps        = n_steps
        self.batch_size     = batch_size
        self.policy_kwargs  = policy_kwargs  # 与主进程 PPO 完全一致，避免网络结构不匹配


# ─────────────────────────────────────────────
# Rollout 数据容器（跨进程传输）
# ─────────────────────────────────────────────

class RolloutData:
    """
    一次 rollout 采集的原始数据（仅 numpy array + list，pickle 友好）。

    Fields
    ------
    observations   : (n_steps, n_envs, obs_dim)
    actions        : (n_steps, n_envs)
    rewards        : (n_steps, n_envs)
    episode_starts : (n_steps, n_envs)  当前步是否是新 episode 的开始（即上步是否 done）
    values         : (n_steps, n_envs)  V(s_t) 的估计值
    log_probs      : (n_steps, n_envs)  log π(a_t | s_t)
    dones          : (n_steps, n_envs)  当前步结束后是否 done（用于 GAE bootstrap）
    ep_infos       : List[dict]         本次 rollout 内完整结束的 episode 统计
                     每项包含 'r'（episode reward）和 'l'（episode length），
                     来自 SB3 Monitor wrapper 的 info['episode']。
    """
    __slots__ = [
        'observations', 'actions', 'rewards', 'episode_starts',
        'values', 'log_probs', 'dones', 'ep_infos',
    ]

    def __init__(
        self,
        observations:   np.ndarray,
        actions:        np.ndarray,
        rewards:        np.ndarray,
        episode_starts: np.ndarray,
        values:         np.ndarray,
        log_probs:      np.ndarray,
        dones:          np.ndarray,
        ep_infos:       list,
    ):
        self.observations   = observations
        self.actions        = actions
        self.rewards        = rewards
        self.episode_starts = episode_starts
        self.values         = values
        self.log_probs      = log_probs
        self.dones          = dones
        self.ep_infos       = ep_infos


# ─────────────────────────────────────────────
# VecNormalize 统计量聚合（多 Worker → 主进程）
# ─────────────────────────────────────────────

def _merge_running_stats(stats_list, shape):
    """将多份 (mean, var, count) 矩并行合并为一个 RunningMeanStd。

    使用 Welford 并行方差算法（与 SB3 RunningMeanStd.update_from_moments 一致）。
    每份 stats 之间数据互不相交（各 Worker 本轮各自新采集的增量数据），
    合并后即为"本轮所有 Worker 新增数据"的联合统计。
    """
    merged = RunningMeanStd(shape=shape)
    for mean, var, count in stats_list:
        count = float(count)
        if count <= 0:
            continue
        merged.update_from_moments(np.asarray(mean), np.asarray(var), count)
    return merged


def _sync_norm_stats(dummy_env, eval_env, payloads):
    """聚合各 Worker【本轮新增】的 VecNormalize 统计增量，累加到主进程的全局统计上，
    并同步回填 eval_env，供保存 _vecnorm.pkl 与训练期评估使用。

    【为什么要用增量而不是每轮重新聚合全量】
    每轮 rollout 开始前，主进程会把全局统计广播给所有 Worker，Worker 用它覆盖本地
    obs_rms/ret_rms 后再采样（见 worker_rollout 中 2.5 步），这样才能保证同一时刻
    所有 Worker 用的是同一份归一化尺度（等价于单卡多 env 共享一份 VecNormalize）。
    因此 Worker 上报的统计必须是"本轮新采集数据"的独立增量（count 从 0 开始统计），
    而不能用覆盖后持续 update 出来的累积 obs_rms，否则全局起点部分会被重复叠加
    （n_gpus 份 Worker 都各自把同一个全局起点算了一遍，导致 count 被放大 n_gpus 倍，
    使全局统计的置信度/方差被严重低估)。

    合并流程：
      1) 用 Welford 并行算法把本轮所有 Worker 的增量统计合并为"本轮联合增量"
      2) 再用 update_from_moments 把该增量累加到 dummy_env 已有的全局统计上
      3) 回填 eval_env，保持训练/评估观测归一化尺度一致
    """
    obs_stats = [p['obs_stats'] for p in payloads if p.get('obs_stats')]
    ret_stats = [p['ret_stats'] for p in payloads if p.get('ret_stats')]

    if obs_stats:
        round_obs_delta = _merge_running_stats(obs_stats, dummy_env.obs_rms.mean.shape)
        if round_obs_delta.count > 1e-4:
            dummy_env.obs_rms.combine(round_obs_delta)
        if eval_env is not None and getattr(eval_env, 'obs_rms', None) is not None:
            eval_env.obs_rms = dummy_env.obs_rms.copy()

    if ret_stats and getattr(dummy_env, 'ret_rms', None) is not None:
        round_ret_delta = _merge_running_stats(ret_stats, dummy_env.ret_rms.mean.shape)
        if round_ret_delta.count > 1e-4:
            dummy_env.ret_rms.combine(round_ret_delta)


# ─────────────────────────────────────────────
# Worker 进程：负责 rollout 数据采集
# ─────────────────────────────────────────────

def worker_rollout(
    rank:          int,
    world_size:    int,
    wargs:         _WorkerArgs,
    freeze:        bool,
    param_queue:   mp.Queue,   # 主进程 → Worker：序列化的 policy state_dict
    rollout_queue: mp.Queue,   # Worker → 主进程：序列化的 RolloutData
    ready_queue:   mp.Queue,   # Worker → 主进程：就绪信号（发 rank 值）
    stop_event:    mp.Event,   # 主进程 → Worker：停止信号
):
    """
    Worker 进程主函数。

    流程
    ----
    1. 初始化本卡的 DummyVecEnv（n_envs_per_gpu 个 env，在进程内顺序执行）
    2. 构建本卡的推理 PPO（仅做 forward，不反向传播）
    3. 发送就绪信号给主进程
    4. 主循环：
       a. 阻塞等待主进程的最新 policy 参数（param_queue.get）
       b. 加载参数到本卡 policy
       c. 执行 n_steps 步 rollout，收集 obs/action/reward/value/log_prob/done
       d. 将 RolloutData pickle 后放入 rollout_queue
       e. 循环至 stop_event 置位
    """
    # ── 设备选择 ─────────────────────────────────────────────────
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        device_id  = rank % n_cuda
        device     = torch.device(f'cuda:{device_id}')
        env_device = f'cuda:{device_id}'
    else:
        device     = torch.device('cpu')
        env_device = 'cpu'

    prefix = f"[Worker rank={rank} dev={device}]"
    print(f"{prefix} 进程启动 (pid={os.getpid()})", flush=True)

    # ── 构建本卡的向量化训练环境 ──────────────────────────────────
    # Worker 进程本身已是独立进程（spawn 启动），不能再创建子进程（daemon 限制），
    # 因此使用 DummyVecEnv（在当前进程内顺序运行多个 env）。
    # 进程级并行已由外层多个 Worker 实现，DummyVecEnv 完全满足需求。
    train_env = make_vec_env(
        env_id=_make_env_fn(
            wargs.data, wargs.ckpt, wargs.model_type, wargs.gpu_type,
            wargs.seq_len, freeze=freeze, env_device=env_device,
        ),
        n_envs=wargs.n_envs_per_gpu,
        vec_env_cls=DummyVecEnv,
    )
    # VecNormalize：对 obs 和 reward 做在线归一化
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
    )
    print(f"{prefix} 环境就绪，n_envs={wargs.n_envs_per_gpu}", flush=True)

    # ── 构建本卡推理 PPO（不做梯度更新，仅 forward 推理）────────────
    # policy_kwargs 由主进程通过 wargs 传入，保证与主进程模型网络结构完全一致，
    # 避免 Phase 2 时因 PHASE1/PHASE2_POLICY_KWARGS 不同导致 load_state_dict 失败。
    infer_model = PPO(
        policy='MlpPolicy',
        env=train_env,
        n_steps=wargs.n_steps,
        batch_size=max(wargs.batch_size // world_size, 64),
        learning_rate=3e-4,           # 推理用，lr 值无意义
        policy_kwargs=wargs.policy_kwargs,
        device=device,
        verbose=0,
    )
    print(f"{prefix} 推理 PPO 就绪（device={device}）", flush=True)

    # ── 通知主进程：本 Worker 已就绪 ─────────────────────────────
    ready_queue.put(rank)

    # ── 初始化 obs 和 done 状态 ───────────────────────────────────
    obs       = train_env.reset()                              # (E, obs_dim)
    last_done = np.zeros(wargs.n_envs_per_gpu, dtype=bool)    # 上一步是否 done

    # ── 主循环 ────────────────────────────────────────────────────
    while not stop_event.is_set():

        # 1. 等待主进程发送最新 policy 参数（阻塞，超时 120s）
        try:
            params_bytes = param_queue.get(timeout=120)
        except Exception:
            if stop_event.is_set():
                break
            print(f"{prefix} 等待参数超时，继续等待...", flush=True)
            continue

        # None 是停止哨兵
        if params_bytes is None:
            break

        # 2. 加载主进程广播的 policy 参数
        broadcast_payload = pickle.loads(params_bytes)
        state_dict = broadcast_payload['state_dict']
        # state_dict 的 value 是 CPU tensor，load_state_dict 时自动移到本卡
        infer_model.policy.load_state_dict(
            {k: v.to(device) for k, v in state_dict.items()},
            strict=True,
        )
        infer_model.policy.set_training_mode(False)

        # 2.5 用主进程聚合的全局 VecNormalize 统计覆盖本地统计。
        #     若各 Worker 各自独立累积 obs_rms/ret_rms，在训练早期（尤其
        #     n_envs_per_gpu 较小时）各卡统计噪声差异很大，导致同一套 policy
        #     参数在不同 Worker 上看到的"归一化观测尺度"不一致，混合后训练
        #     信号噪声偏大、PPO 更新不稳定（等价于单卡下 n_envs 个 env 共享
        #     同一份 VecNormalize，这里必须让所有 Worker 保持统计一致）。
        global_obs_stats = broadcast_payload.get('obs_stats')
        global_ret_stats = broadcast_payload.get('ret_stats')
        if global_obs_stats is not None:
            mean, var, count = global_obs_stats
            train_env.obs_rms.mean  = mean.copy()
            train_env.obs_rms.var   = var.copy()
            train_env.obs_rms.count = count
        if global_ret_stats is not None and train_env.ret_rms is not None:
            mean, var, count = global_ret_stats
            train_env.ret_rms.mean  = mean.copy()
            train_env.ret_rms.var   = var.copy()
            train_env.ret_rms.count = count

        # 3. 执行 n_steps 步 rollout
        all_obs            = []
        all_actions        = []
        all_rewards        = []
        all_episode_starts = []
        all_values         = []
        all_log_probs      = []
        all_dones          = []
        ep_infos           = []   # 本次 rollout 内完整结束的 episode 统计

        # 本轮【原始】(未归一化) obs / 折扣回报，用于计算本轮增量统计并上报主进程。
        # 注意：不能直接用 train_env.obs_rms（它已被 2.5 步覆盖为全局统计，
        # 循环中还会被 VecNormalize 内部持续 update），否则上报时会把"全局起点"
        # 重复计入，导致多个 Worker 的统计合并后 count 被放大、方差被低估。
        local_raw_obs   = []       # 每步的原始 obs（VecNormalize 归一化前）
        local_returns   = []       # 每步更新后的折扣回报 self.returns（与 SB3 ret_rms 定义一致）
        gamma           = train_env.gamma
        running_returns = np.zeros(wargs.n_envs_per_gpu, dtype=np.float64)

        with torch.no_grad():
            for _ in range(wargs.n_steps):
                # episode_start = 上一步结束后是否开始新 episode
                all_episode_starts.append(last_done.copy())
                all_obs.append(obs.copy())

                # policy forward
                obs_tensor = obs_as_tensor(obs, device)           # (E, D)
                actions, values, log_probs = infer_model.policy(obs_tensor)

                actions_np  = actions.cpu().numpy()               # (E,)
                values_np   = values.cpu().numpy().reshape(-1)    # (E,)
                log_probs_np = log_probs.cpu().numpy()            # (E,)

                all_actions.append(actions_np.copy())
                all_values.append(values_np.copy())
                all_log_probs.append(log_probs_np.copy())

                # 执行动作，从 info 里收集 Monitor 记录的完整 episode 统计
                new_obs, rewards, dones, infos = train_env.step(actions_np)
                all_rewards.append(rewards.copy())                # (E,)
                all_dones.append(dones.copy())                    # (E,)

                # 记录本步【原始】obs（VecNormalize.step_wait 会把归一化前的
                # obs 缓存在 old_obs），以及按 SB3 定义更新的折扣回报，
                # 用于之后统计本轮增量的 obs_rms / ret_rms。
                local_raw_obs.append(np.asarray(train_env.old_obs).copy())
                running_returns = running_returns * gamma + np.asarray(train_env.old_reward)
                local_returns.append(running_returns.copy())
                running_returns[dones] = 0.0

                # Monitor wrapper 在 episode 结束时向 info 写入 'episode' 字段
                # VecEnv 返回的 infos 是长度=n_envs 的 list of dict
                for info in infos:
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_infos.append({'r': ep_info['r'], 'l': ep_info['l']})

                obs       = new_obs
                last_done = dones

        # 4. 组装 RolloutData
        rollout_data = RolloutData(
            observations   = np.stack(all_obs,            axis=0),  # (T, E, D)
            actions        = np.stack(all_actions,        axis=0),  # (T, E)
            rewards        = np.stack(all_rewards,        axis=0),  # (T, E)
            episode_starts = np.stack(all_episode_starts, axis=0),  # (T, E)
            values         = np.stack(all_values,         axis=0),  # (T, E)
            log_probs      = np.stack(all_log_probs,      axis=0),  # (T, E)
            dones          = np.stack(all_dones,          axis=0),  # (T, E)
            ep_infos       = ep_infos,
        )

        # 5. 计算本轮【增量】VecNormalize 统计（count 从 0 开始，只统计本轮
        #    本 Worker 新采集的原始数据），发送给主进程用于累加到全局统计。
        #    主进程的 dummy_env/eval_env 从未接触真实数据，需靠各 Worker 的增量
        #    统计累加后回填，否则保存的 _vecnorm.pkl 与训练期评估会用默认（空）
        #    统计 → 分布失配（评估奖励远低于训练奖励）。
        raw_obs_arr = np.concatenate(local_raw_obs, axis=0)   # (T*E, D)
        ret_arr     = np.concatenate(local_returns, axis=0)   # (T*E,)

        obs_delta_mean, obs_delta_var = raw_obs_arr.mean(axis=0), raw_obs_arr.var(axis=0)
        obs_delta_count = raw_obs_arr.shape[0]

        payload = {
            'rollout_data': rollout_data,
            'obs_stats':    (obs_delta_mean, obs_delta_var, float(obs_delta_count)),
            'ret_stats':    None,
        }
        if train_env.ret_rms is not None:
            # ret_rms 的 shape 固定为 ()（标量），与 obs_rms 的向量 shape 不同，
            # 这里保持 mean/var 为 0-d 标量，避免与 dummy_env.ret_rms.mean.shape=() 不匹配。
            ret_delta_mean = np.asarray(ret_arr.mean())
            ret_delta_var  = np.asarray(ret_arr.var())
            payload['ret_stats'] = (ret_delta_mean, ret_delta_var, float(ret_arr.shape[0]))
        rollout_queue.put((rank, pickle.dumps(payload)))

    train_env.close()
    print(f"{prefix} Worker 退出", flush=True)


# ─────────────────────────────────────────────
# 合并多 Worker 的 RolloutData → SB3 RolloutBuffer
# ─────────────────────────────────────────────

def _build_rollout_buffer(
    rollout_data_list: List[RolloutData],
    model: PPO,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> RolloutBuffer:
    """
    将多个 Worker 的 RolloutData 沿 env 维度拼接，填入 SB3 RolloutBuffer，
    并用 GAE 计算 advantages 和 returns。

    合并策略
    --------
    - obs:    (T, E_0, D) ⊕ (T, E_1, D) ⊕ ... → (T, E_total, D)
    - 其余字段同理

    GAE bootstrap
    -------------
    使用最后一步（last_obs）估计 V(s_T)，与 SB3 标准 PPO 完全一致。
    """
    # ── 沿 env 轴（axis=1）拼接所有 Worker 的数据 ────────────────
    obs_all   = np.concatenate([rd.observations   for rd in rollout_data_list], axis=1)
    act_all   = np.concatenate([rd.actions        for rd in rollout_data_list], axis=1)
    rew_all   = np.concatenate([rd.rewards        for rd in rollout_data_list], axis=1)
    eps_all   = np.concatenate([rd.episode_starts for rd in rollout_data_list], axis=1)
    val_all   = np.concatenate([rd.values         for rd in rollout_data_list], axis=1)
    lp_all    = np.concatenate([rd.log_probs      for rd in rollout_data_list], axis=1)
    done_all  = np.concatenate([rd.dones          for rd in rollout_data_list], axis=1)

    n_steps, n_envs_total, _obs_dim = obs_all.shape

    # ── 创建 SB3 RolloutBuffer ───────────────────────────────────
    buffer = RolloutBuffer(
        buffer_size=n_steps,
        observation_space=model.observation_space,
        action_space=model.action_space,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=n_envs_total,
    )
    buffer.reset()

    # ── 逐时间步填入 buffer ──────────────────────────────────────
    for t in range(n_steps):
        val_tensor = torch.from_numpy(val_all[t]).float().to(device).unsqueeze(-1)  # (E, 1)
        lp_tensor  = torch.from_numpy(lp_all[t]).float().to(device)                  # (E,)

        buffer.add(
            obs           = obs_all[t],    # (E, D)  numpy
            action        = act_all[t],    # (E,)    numpy
            reward        = rew_all[t],    # (E,)    numpy
            episode_start = eps_all[t],    # (E,)    numpy bool
            value         = val_tensor,    # (E, 1)  Tensor
            log_prob      = lp_tensor,     # (E,)    Tensor
        )

    # ── 用最后一步 obs 估计 V(s_T)，用于 GAE bootstrap ──────────
    last_obs_tensor = torch.from_numpy(obs_all[-1]).float().to(device)   # (E, D)
    with torch.no_grad():
        last_values = model.policy.predict_values(last_obs_tensor)        # (E, 1)

    buffer.compute_returns_and_advantage(
        last_values=last_values,
        dones=done_all[-1],               # (E,) bool numpy
    )

    return buffer


# ─────────────────────────────────────────────
# SB3 风格训练统计表格输出
# ─────────────────────────────────────────────

def _log_sb3_style(
    logger:       logging.Logger,
    phase:        int,
    global_step:  int,
    fps:          float,
    t_elapsed:    float,
    ep_infos:     list,
    pg_loss:      float,
    value_loss:   float,
    entropy_loss: float,
    clip_frac:    float,
    approx_kl:    float,
):
    """
    以 SB3 标准格式打印 rollout / train / time 三段统计表格，
    与单卡 rl_train.py 的输出风格保持一致。

    SB3 标准格式示例：
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 1000         |
    |    ep_rew_mean          | -0.234       |
    | time/                   |              |
    |    fps                  | 671          |
    |    iterations           | 1            |
    |    time_elapsed         | 49           |
    |    total_timesteps      | 32768        |
    | train/                  |              |
    |    approx_kl            | 0.006        |
    |    clip_fraction        | 0.06         |
    |    entropy_loss         | -1.94        |
    |    n_updates            | 10           |
    |    policy_gradient_loss | -0.005       |
    |    value_loss           | 0.0558       |
    ------------------------------------------
    """
    # ── rollout 段：episode 统计（来自 Monitor，仅在有完整 episode 时输出）
    lines = []
    lines.append(f"{'':->50}")
    lines.append(f"| {'rollout/':<28}{'':>18} |")

    if ep_infos:
        ep_rew_mean = float(np.mean([e['r'] for e in ep_infos]))
        ep_len_mean = float(np.mean([e['l'] for e in ep_infos]))
        lines.append(f"|    {'ep_len_mean':<24} {ep_len_mean:<14.0f} |")
        lines.append(f"|    {'ep_rew_mean':<24} {ep_rew_mean:<14.3f} |")
    else:
        lines.append(f"|    {'ep_len_mean':<24} {'N/A':<14} |")
        lines.append(f"|    {'ep_rew_mean':<24} {'N/A':<14} |")

    # ── time 段
    lines.append(f"| {'time/':<28}{'':>18} |")
    lines.append(f"|    {'fps':<24} {fps:<14.0f} |")
    lines.append(f"|    {'phase':<24} {phase:<14} |")
    lines.append(f"|    {'time_elapsed':<24} {t_elapsed:<14.0f} |")
    lines.append(f"|    {'total_timesteps':<24} {global_step:<14,} |")

    # ── train 段
    lines.append(f"| {'train/':<28}{'':>18} |")
    lines.append(f"|    {'approx_kl':<24} {approx_kl:<14.5f} |")
    lines.append(f"|    {'clip_fraction':<24} {clip_frac:<14.3f} |")
    lines.append(f"|    {'entropy_loss':<24} {entropy_loss:<14.4f} |")
    lines.append(f"|    {'policy_gradient_loss':<24} {pg_loss:<14.5f} |")
    lines.append(f"|    {'value_loss':<24} {value_loss:<14.5f} |")
    lines.append(f"{'':->50}")

    for line in lines:
        logger.info(line)


# ─────────────────────────────────────────────
# Eval 结果表格输出（SB3 EvalCallback 风格）
# ─────────────────────────────────────────────

def _log_eval_result(
    logger:      logging.Logger,
    global_step: int,
    t_elapsed:   float,
    mean_reward: float,
    std_reward:  float,
    mean_ep_len: float,
    best_reward: float,
    is_new_best: bool,
    best_path:   Optional[str],
):
    """
    以 SB3 EvalCallback 风格打印评估结果，格式示例：

    --------------------------------------------------
    | eval/                   |                     |
    |    mean_ep_length       | 1933                |
    |    mean_reward          | -220.989            |
    |    best_mean_reward     | -220.989            |
    | time/                   |                     |
    |    total_timesteps      | 131,072             |
    |    time_elapsed         | 191                 |
    --------------------------------------------------
    New best mean reward!
    """
    lines = []
    lines.append(f"{'':->50}")
    lines.append(f"| {'eval/':<28}{'':>18} |")
    lines.append(f"|    {'mean_ep_length':<24} {mean_ep_len:<14.0f} |")
    lines.append(f"|    {'mean_reward':<24} {mean_reward:<14.3f} |")
    lines.append(f"|    {'best_mean_reward':<24} {best_reward:<14.3f} |")
    lines.append(f"| {'time/':<28}{'':>18} |")
    lines.append(f"|    {'total_timesteps':<24} {global_step:<14,} |")
    lines.append(f"|    {'time_elapsed':<24} {t_elapsed:<14.0f} |")
    lines.append(f"{'':->50}")

    for line in lines:
        logger.info(line)

    if is_new_best:
        logger.info("New best mean reward!")
        if best_path:
            logger.info(f"  ✨ 最优模型已保存: {best_path}")


# ─────────────────────────────────────────────
# PPO 手动更新（主进程执行，对应 SB3 PPO.train() 内部逻辑）
# ─────────────────────────────────────────────

def _run_ppo_update(
    model:         PPO,
    rollout_buffer: RolloutBuffer,
    n_epochs:      int,
    batch_size:    int,
    clip_range:    float,
    ent_coef:      float,
    vf_coef:       float,
    max_grad_norm: float,
) -> Dict[str, float]:
    """
    对 model.policy 执行一次完整的 PPO 梯度更新。

    直接操作 model.policy.optimizer，无需重复调用 model.learn()，
    与 SB3 内部 PPO.train() 逻辑等价。

    Returns
    -------
    stats : dict，包含 pg_loss / value_loss / entropy_loss / clip_frac / approx_kl
    """
    model.policy.set_training_mode(True)

    pg_losses, value_losses, entropy_losses, clip_fracs, approx_kls = [], [], [], [], []

    for _epoch in range(n_epochs):
        for batch in rollout_buffer.get(batch_size):
            # ── 取动作 ───────────────────────────────────────────
            actions = batch.actions
            if isinstance(model.action_space, gymnasium.spaces.Discrete):
                # 离散动作：SB3 存的是 float，evaluate_actions 需要 long
                actions = batch.actions.long().flatten()

            # ── policy forward ───────────────────────────────────
            values, log_prob, entropy = model.policy.evaluate_actions(
                batch.observations, actions
            )
            values = values.flatten()   # (B,)

            # ── normalize advantages ──────────────────────────────
            adv = batch.advantages
            if adv.shape[0] > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ── PPO surrogate loss ────────────────────────────────
            ratio = torch.exp(log_prob - batch.old_log_prob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # ── value loss ────────────────────────────────────────
            v_loss = torch.nn.functional.mse_loss(values, batch.returns)

            # ── entropy bonus ─────────────────────────────────────
            e_loss = -entropy.mean() if entropy is not None else torch.zeros(1, device=values.device).squeeze()

            # ── 总损失 ────────────────────────────────────────────
            loss = pg_loss + vf_coef * v_loss + ent_coef * e_loss

            # ── 反向传播 & 梯度裁剪 ───────────────────────────────
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.policy.parameters(), max_grad_norm
            )
            model.policy.optimizer.step()

            # ── 统计 ─────────────────────────────────────────────
            with torch.no_grad():
                clip_frac  = ((ratio - 1).abs() > clip_range).float().mean().item()
                approx_kl  = ((ratio - 1) - torch.log(ratio)).mean().item()

            pg_losses.append(pg_loss.item())
            value_losses.append(v_loss.item())
            entropy_losses.append(e_loss.item())
            clip_fracs.append(clip_frac)
            approx_kls.append(approx_kl)

    model.policy.set_training_mode(False)

    return {
        'pg_loss':      float(np.mean(pg_losses)),
        'value_loss':   float(np.mean(value_losses)),
        'entropy_loss': float(np.mean(entropy_losses)),
        'clip_frac':    float(np.mean(clip_fracs)),
        'approx_kl':    float(np.mean(approx_kls)),
    }


# ─────────────────────────────────────────────
# 策略评估
# ─────────────────────────────────────────────

def _evaluate_policy(
    model:           PPO,
    eval_env:        VecNormalize,
    n_eval_episodes: int,
    logger:          logging.Logger,
) -> tuple:
    """运行 n_eval_episodes 轮 deterministic 评估。

    Returns
    -------
    mean_r : float  平均 episode reward
    std_r  : float  episode reward 标准差
    mean_l : float  平均 episode 长度
    """
    model.policy.set_training_mode(False)
    all_rewards: List[float] = []
    all_lengths: List[int]   = []

    for _ep in range(n_eval_episodes):
        obs  = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps  = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _info = eval_env.step(action)
            done       = bool(done_arr[0])
            ep_reward += float(reward[0])
            ep_steps  += 1
            if ep_steps >= 2000:   # 防无限循环
                break
        all_rewards.append(ep_reward)
        all_lengths.append(ep_steps)

    mean_r = float(np.mean(all_rewards))
    std_r  = float(np.std(all_rewards))
    mean_l = float(np.mean(all_lengths))
    return mean_r, std_r, mean_l


# ─────────────────────────────────────────────
# 关闭 Worker 进程
# ─────────────────────────────────────────────

def _shutdown_workers(
    workers:      list,
    stop_event:   mp.Event,
    param_queues: list,
):
    """优雅地关闭所有 Worker 进程（先发停止信号，再 join，再 terminate）。"""
    stop_event.set()
    # 向每个 Worker 的 param_queue 发送 None 哨兵，解除阻塞的 get()
    for q in param_queues:
        try:
            q.put_nowait(None)
        except Exception:
            pass
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)


# ─────────────────────────────────────────────
# 主训练函数（Phase 1 & Phase 2 通用）
# ─────────────────────────────────────────────

def train_phase_multigpu(
    phase:                int,
    data_path:            str,
    ckpt_path:            str,
    output_dir:           str,
    model_type:           str,
    gpu_type:             str,
    seq_len:              int,
    n_gpus:               int,
    n_envs_per_gpu:       int,
    total_steps:          int,
    n_steps:              int    = 2048,
    batch_size:           int    = 512,
    n_epochs:             int    = 10,
    learning_rate:        float  = 3e-4,
    gamma:                float  = 0.99,
    gae_lambda:           float  = 0.95,
    clip_range:           float  = 0.2,
    ent_coef:             float  = 0.01,
    vf_coef:              float  = 0.5,
    max_grad_norm:        float  = 0.5,
    phase1_model_path:    Optional[str] = None,   # 仅 phase 2 使用
    logger:               Optional[logging.Logger] = None,
    eval_every_n_rollouts: int   = 4,
    n_eval_episodes:       int   = 5,
    save_freq_steps:       int   = 20_000,
) -> str:
    """
    多卡两阶段 PPO 训练通用函数。

    参数
    ----
    phase              : 1（冻结编码器）或 2（端到端微调）
    n_gpus             : 参与 rollout 的 GPU 数量
    n_envs_per_gpu     : 每张 GPU 负责的并行环境数
    total_steps        : 总环境步数（所有 env 合计，决定 rollout 轮数）
    n_steps            : 每轮 rollout 每个 env 采集的步数

    Returns
    -------
    model_path : 最终保存的模型路径（str）
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    freeze        = (phase == 1)
    phase_name    = f"phase{phase}"
    policy_kwargs = PHASE1_POLICY_KWARGS if freeze else PHASE2_POLICY_KWARGS

    n_envs_total        = n_gpus * n_envs_per_gpu
    samples_per_rollout = n_steps * n_envs_total          # 每轮 rollout 的总样本数
    n_rollouts          = max(1, total_steps // samples_per_rollout)

    # 主进程训练设备（优先 cuda:0）
    main_device = (torch.device('cuda:0') if torch.cuda.is_available()
                   else torch.device('cpu'))
    eval_env_dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logger.info("=" * 70)
    logger.info(f"▶ 阶段 {phase} 多卡 PPO 训练")
    logger.info(f"   n_gpus={n_gpus}  n_envs_per_gpu={n_envs_per_gpu}  "
                f"n_envs_total={n_envs_total}")
    logger.info(f"   n_steps={n_steps}  samples_per_rollout={samples_per_rollout:,}  "
                f"n_rollouts={n_rollouts}")
    logger.info(f"   total_steps={total_steps:,}  PPO device={main_device}")
    logger.info(f"   learning_rate={learning_rate}  batch_size={batch_size}  "
                f"n_epochs={n_epochs}")
    logger.info("=" * 70)

    phase_dir = os.path.join(output_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)

    # ── 主进程评估环境（DummyVecEnv，单进程，避免死锁）────────────
    eval_env = make_vec_env(
        env_id=_make_env_fn(data_path, ckpt_path, model_type, gpu_type,
                            seq_len, freeze=freeze, env_device=eval_env_dev),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    # ── 主进程虚拟训练环境（仅供 PPO 初始化和保存 VecNormalize 统计）
    dummy_env = make_vec_env(
        env_id=_make_env_fn(data_path, ckpt_path, model_type, gpu_type,
                            seq_len, freeze=freeze, env_device=eval_env_dev),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )
    dummy_env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    # ── 构建主进程 PPO 模型 ───────────────────────────────────────
    if phase == 1:
        model = PPO(
            policy='MlpPolicy',
            env=dummy_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(phase_dir, 'tb_logs'),
            device=main_device,
            verbose=0,
        )
    else:
        # Phase 2：从 Phase 1 权重继续，更换学习率
        model = PPO.load(
            phase1_model_path,
            env=dummy_env,
            device=main_device,
        )
        model.learning_rate = learning_rate
        model.lr_schedule   = lambda _: learning_rate
        logger.info(f"  Phase 2：从 {phase1_model_path} 加载 Phase 1 权重 ✓")

    logger.info(f"  PPO 参数量: {sum(p.numel() for p in model.policy.parameters()):,}")
    logger.info(f"  PPO 运行设备: {model.device}")

    # ── 启动 Worker 进程 ──────────────────────────────────────────
    # param_queues[i]：主进程 → Worker-i（发最新参数）
    # rollout_queue：  Worker-i → 主进程（发 rollout 数据）
    param_queues  = [mp.Queue(maxsize=1) for _ in range(n_gpus)]
    rollout_queue = mp.Queue(maxsize=n_gpus * 2)
    ready_queue   = mp.Queue()
    stop_event    = mp.Event()

    # _WorkerArgs 需要 pickle，须在主进程构造好再传给 Worker
    # policy_kwargs 直接取主进程 PPO 实际使用的配置，保证 Worker 推理模型结构一致
    wargs = _WorkerArgs(
        data           = data_path,
        ckpt           = ckpt_path,
        model_type     = model_type,
        gpu_type       = gpu_type,
        seq_len        = seq_len,
        n_envs_per_gpu = n_envs_per_gpu,
        n_steps        = n_steps,
        batch_size     = batch_size,
        policy_kwargs  = model.policy_kwargs,  # 从已构建的主进程 PPO 中取，Phase 1/2 自动正确
    )

    ctx     = mp.get_context('spawn')   # spawn：子进程可安全使用 CUDA
    workers = []

    for rank in range(n_gpus):
        p = ctx.Process(
            target=worker_rollout,
            args=(
                rank, n_gpus, wargs, freeze,
                param_queues[rank], rollout_queue,
                ready_queue, stop_event,
            ),
            daemon=False,  # 必须为 False：daemon 进程禁止创建子进程，而 DummyVecEnv 需要在进程内调用 env
        )
        p.start()
        workers.append(p)
        logger.info(f"  Worker rank={rank} 已启动 (pid={p.pid})")

    # ── 等待所有 Worker 就绪 ──────────────────────────────────────
    ready_cnt = 0
    while ready_cnt < n_gpus:
        try:
            rank_ready = ready_queue.get(timeout=300)
            ready_cnt += 1
            logger.info(f"  Worker rank={rank_ready} 就绪 ({ready_cnt}/{n_gpus})")
        except Exception:
            logger.error("Worker 就绪超时（300s），请检查环境初始化是否出错")
            _shutdown_workers(workers, stop_event, param_queues)
            eval_env.close()
            dummy_env.close()
            raise RuntimeError("Worker 初始化超时")

    logger.info(f"  ✓ 全部 {n_gpus} 个 Worker 就绪，开始训练主循环")

    # ── 训练主循环 ────────────────────────────────────────────────
    global_step   = 0
    best_reward   = -float('inf')
    best_dir      = os.path.join(phase_dir, 'best')
    ckpt_save_dir = os.path.join(phase_dir, 'checkpoints')
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    train_start = time.time()

    # tqdm 进度条：写到原始 stderr（绕过 _Tee 过滤），显示 rollout 级别进度
    pbar = tqdm(
        total=n_rollouts,
        desc=f"Phase {phase}",
        unit="rollout",
        file=sys.__stderr__,          # 直接写原始流，不被 _Tee 过滤
        dynamic_ncols=True,
        colour='green',
    )

    for rollout_idx in range(n_rollouts):
        t_rollout_start = time.time()

        # ── Step 1：将最新 policy 参数 + 全局 VecNormalize 统计广播给所有 Worker
        # 参数统一移到 CPU 再 pickle，节省内存，Worker 会自动移到本卡。
        # 同时把 dummy_env 当前累积的全局 obs_rms/ret_rms 一并广播，Worker 收到后
        # 会用它覆盖本地统计（见 worker_rollout 2.5 步），保证本轮采样时所有
        # Worker 用的是同一份归一化尺度，等价于单卡下 n_envs 个 env 共享一份
        # VecNormalize（首轮广播时 dummy_env 统计仍是初始默认值，属预期行为，
        # 从第 2 轮起即用上一轮聚合的真实全局统计）。
        obs_rms_g = dummy_env.obs_rms
        ret_rms_g = dummy_env.ret_rms
        params_bytes = pickle.dumps({
            'state_dict': {k: v.detach().cpu() for k, v in model.policy.state_dict().items()},
            'obs_stats':  (obs_rms_g.mean.copy(), obs_rms_g.var.copy(), float(obs_rms_g.count)),
            'ret_stats':  (ret_rms_g.mean.copy(), ret_rms_g.var.copy(), float(ret_rms_g.count))
                          if ret_rms_g is not None else None,
        })
        for q in param_queues:
            q.put(params_bytes)

        # ── Step 2：等待所有 Worker 完成 rollout ─────────────────
        received: Dict[int, dict] = {}
        while len(received) < n_gpus:
            try:
                rank_id, data_bytes = rollout_queue.get(timeout=600)
                received[rank_id] = pickle.loads(data_bytes)
            except Exception as e:
                logger.error(
                    f"等待 rollout 数据超时（已收到 {len(received)}/{n_gpus}）: {e}"
                )
                _shutdown_workers(workers, stop_event, param_queues)
                eval_env.close()
                dummy_env.close()
                raise RuntimeError("Rollout 收集超时，训练中止") from e

        # ── Step 2.5：聚合各 Worker 的 VecNormalize 统计 → 回填 dummy_env/eval_env
        #    保证训练期评估与最终保存的 _vecnorm.pkl 使用真实归一化统计。
        _sync_norm_stats(dummy_env, eval_env, [received[r] for r in range(n_gpus)])

        # ── Step 3：合并 RolloutData，计算 GAE ───────────────────
        ordered_data = [received[r]['rollout_data'] for r in range(n_gpus)]
        rollout_buffer = _build_rollout_buffer(
            rollout_data_list=ordered_data,
            model=model,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=main_device,
        )

        # ── Step 4：PPO 梯度更新 ──────────────────────────────────
        stats = _run_ppo_update(
            model=model,
            rollout_buffer=rollout_buffer,
            n_epochs=n_epochs,
            batch_size=batch_size,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )

        global_step     += samples_per_rollout
        t_rollout        = time.time() - t_rollout_start
        t_total_elapsed  = time.time() - train_start
        fps              = samples_per_rollout / max(t_rollout, 1e-6)

        # ── 汇总所有 Worker 的 episode 统计 ──────────────────────
        all_ep_infos = []
        for rd in ordered_data:
            all_ep_infos.extend(rd.ep_infos)

        # ── Step 5：以 SB3 风格打印训练统计表格 ──────────────────
        _log_sb3_style(
            logger         = logger,
            phase          = phase,
            global_step    = global_step,
            fps            = fps,
            t_elapsed      = t_total_elapsed,
            ep_infos       = all_ep_infos,
            pg_loss        = stats['pg_loss'],
            value_loss     = stats['value_loss'],
            entropy_loss   = stats['entropy_loss'],
            clip_frac      = stats['clip_frac'],
            approx_kl      = stats['approx_kl'],
        )

        # ── Step 6：定期评估（每 eval_every_n_rollouts 轮一次，或最后一轮强制评估）
        is_last_rollout = (rollout_idx + 1 == n_rollouts)
        if (rollout_idx + 1) % eval_every_n_rollouts == 0 or is_last_rollout:
            logger.info(
                f"  [Eval] {'最终评估' if is_last_rollout else '定期评估'} "
                f"(rollout {rollout_idx+1}/{n_rollouts}, step={global_step:,})"
            )
            mean_reward, std_reward, mean_ep_len = _evaluate_policy(
                model=model,
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                logger=logger,
            )
            is_new_best = mean_reward > best_reward
            if is_new_best:
                best_reward = mean_reward
                best_path   = os.path.join(best_dir, f'{phase_name}_best')
                model.save(best_path)
            # 评估结果表格（SB3 风格）
            _log_eval_result(
                logger       = logger,
                global_step  = global_step,
                t_elapsed    = t_total_elapsed,
                mean_reward  = mean_reward,
                std_reward   = std_reward,
                mean_ep_len  = mean_ep_len,
                best_reward  = best_reward,
                is_new_best  = is_new_best,
                best_path    = best_path if is_new_best else None,
            )

        # ── Step 7：定期保存 checkpoint ──────────────────────────
        # 判断：当前累计步数跨过下一个 save_freq_steps 整数倍时保存
        prev_step = global_step - samples_per_rollout
        if (global_step // save_freq_steps) > (prev_step // save_freq_steps):
            ckpt_path_out = os.path.join(
                ckpt_save_dir,
                f'ppo_{phase_name}_{global_step:010d}',
            )
            model.save(ckpt_path_out)
            logger.info(f"  💾 Checkpoint 已保存: {ckpt_path_out}")

        # ── Step 8：更新进度条 ────────────────────────────────────
        ep_rew_str = (
            f"{float(np.mean([e['r'] for e in all_ep_infos])):.2f}"
            if all_ep_infos else "N/A"
        )
        pbar.set_postfix(
            step=f"{global_step:,}",
            fps=f"{fps:.0f}",
            rew=ep_rew_str,
            v_loss=f"{stats['value_loss']:.4f}",
            kl=f"{stats['approx_kl']:.4f}",
        )
        pbar.update(1)

    pbar.close()

    # ── 训练结束，关闭 Worker ─────────────────────────────────────
    logger.info(f"阶段 {phase} 训练循环结束，关闭 Worker 进程...")
    _shutdown_workers(workers, stop_event, param_queues)

    # ── 保存最终模型与 VecNormalize 统计量 ───────────────────────
    ts         = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(phase_dir, f'ppo_{phase_name}_final_{ts}')
    model.save(model_path)
    dummy_env.save(model_path + '_vecnorm.pkl')

    eval_env.close()
    dummy_env.close()

    logger.info(f"✅ 阶段 {phase} 多卡训练完成！最终模型: {model_path}")
    return model_path


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='多卡两阶段 PPO 训练：预测感知型 LLM 资源调度',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据与模型路径 ────────────────────────────────────────────
    parser.add_argument(
        '--data', type=str,
        default='simulation_data/combined_simulation_data.csv',
        help='仿真数据路径（combined_simulation_data.csv）',
    )
    parser.add_argument(
        '--ckpt', type=str, required=True,
        help='预训练 LSTM checkpoint 路径（.pth）',
    )
    parser.add_argument(
        '--output', type=str, default='rl_models_multigpu',
        help='输出目录（模型、日志、checkpoint 均保存于此）',
    )

    # ── 场景参数 ──────────────────────────────────────────────────
    parser.add_argument(
        '--model-type', type=str, default='GPT-4',
        choices=['GPT-4', 'ChatGLM', 'Claude', 'LLaMA'],
        help='要训练的 LLM 场景',
    )
    parser.add_argument(
        '--load-pattern', type=str, default='Daily',
        choices=['Daily', 'Weekly', 'Burst', 'Steady'],
        help='负载模式（仅用于日志与输出目录命名）',
    )
    parser.add_argument(
        '--gpu-type', type=str, default='A100',
        choices=['A100', 'H100'],
        help='初始 GPU 型号',
    )
    parser.add_argument(
        '--seq-len', type=int, default=60,
        help='历史序列窗口长度（与 LSTM 训练时一致）',
    )

    # ── 多卡参数 ──────────────────────────────────────────────────
    parser.add_argument(
        '--n-gpus', type=int, default=None,
        help='使用的 GPU 数量（默认自动检测所有可用 GPU，最少 1）',
    )
    parser.add_argument(
        '--n-envs', type=int, default=4,
        help='总并行环境数（与单卡 --n-envs 含义一致，自动均分到每张 GPU）',
    )

    # ── 训练超参数（默认值与单卡 rl_train.py 完全一致）──────────
    parser.add_argument('--phase1-steps',  type=int,   default=100_000,
                        help='阶段 1 总训练步数（0=跳过阶段 1）')
    parser.add_argument('--phase2-steps',  type=int,   default=200_000,
                        help='阶段 2 总训练步数（0=跳过阶段 2）')
    parser.add_argument('--phase1-lr',     type=float, default=3e-4,
                        help='阶段 1 学习率')
    parser.add_argument('--phase2-lr',     type=float, default=1e-5,
                        help='阶段 2 学习率（端到端微调，通常更小）')
    parser.add_argument('--n-steps',       type=int,   default=2048,
                        help='每轮 rollout 每个 env 采集的步数')
    parser.add_argument('--batch-size',    type=int,   default=256,
                        help='PPO minibatch 大小（与单卡一致，默认 256）')
    parser.add_argument('--n-epochs',      type=int,   default=10,
                        help='每轮 rollout 后 PPO 更新的 epoch 数')
    parser.add_argument('--gamma',         type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--gae-lambda',    type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip-range',    type=float, default=0.2,
                        help='PPO clip 范围')
    parser.add_argument('--ent-coef',      type=float, default=0.01,
                        help='熵正则系数（鼓励探索）')
    parser.add_argument('--vf-coef',       type=float, default=0.5,
                        help='Value loss 系数')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='梯度裁剪最大范数')

    # ── 评估与保存 ────────────────────────────────────────────────
    parser.add_argument('--eval-every-n-rollouts', type=int, default=4,
                        help='每隔多少轮 rollout 评估一次策略（与单卡一致）')
    parser.add_argument('--n-eval-episodes',       type=int, default=5,
                        help='每次评估的 episode 数')
    parser.add_argument('--save-freq-steps',       type=int, default=20_000,
                        help='每隔多少环境步保存一次 checkpoint（与单卡一致）')

    # ── 断点续训 ──────────────────────────────────────────────────
    parser.add_argument('--phase1-model', type=str, default=None,
                        help='已有阶段 1 模型路径（跳过阶段 1 时指定）')

    args = parser.parse_args()

    # ── 路径校验 ──────────────────────────────────────────────────
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"仿真数据不存在: {args.data}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"LSTM checkpoint 不存在: {args.ckpt}")

    # ── GPU 数量处理 ──────────────────────────────────────────────
    n_available = torch.cuda.device_count()
    if args.n_gpus is None:
        args.n_gpus = max(1, n_available)
    else:
        args.n_gpus = min(args.n_gpus, max(1, n_available))

    if n_available == 0:
        print("⚠️  未检测到 GPU，使用 CPU 运行（仅调试用，性能极低）")
        args.n_gpus = 1

    # ── 将总 env 数均分到每张 GPU ─────────────────────────────────
    # 保证 n_envs 能被 n_gpus 整除；若不能整除则向上取整后重新计算总数
    args.n_envs_per_gpu = max(1, (args.n_envs + args.n_gpus - 1) // args.n_gpus)
    args.n_envs_total   = args.n_envs_per_gpu * args.n_gpus
    if args.n_envs_total != args.n_envs:
        print(f"⚠️  n_envs={args.n_envs} 不能被 n_gpus={args.n_gpus} 整除，"
              f"自动调整为 {args.n_envs_total}（{args.n_envs_per_gpu}/卡）")

    # ── 日志设置 ──────────────────────────────────────────────────
    log_dir = os.path.join(args.output, 'logs')
    log_tag = f"rl_mg_{args.model_type}_{args.load_pattern}"
    logger  = _setup_logger(log_dir, log_tag)
    raw_log = _redirect_stdout_stderr(log_dir, log_tag)

    logger.info(f"终端输出同步保存至: {raw_log}")
    logger.info("═" * 70)
    logger.info("  多卡 PPO 训练：预测感知型 LLM 资源调度")
    logger.info(f"  数据: {args.data}")
    logger.info(f"  LSTM ckpt: {args.ckpt}")
    logger.info(f"  场景: model_type={args.model_type}  "
                f"load_pattern={args.load_pattern}  gpu_type={args.gpu_type}")
    logger.info(f"  GPU: 可用 {n_available} 张，实际使用 {args.n_gpus} 张")
    logger.info(f"  总并行环境数: {args.n_envs_total} "
                f"（{args.n_gpus} GPU × {args.n_envs_per_gpu} envs/GPU）")
    logger.info("═" * 70)

    # ── 公共参数 dict（传给 train_phase_multigpu）────────────────
    # save_freq 与单卡对齐：单卡用 max(20_000 // n_envs, 1) 步/env × n_envs = 20_000 步
    # 多卡等效：同样以总环境步 20_000 为周期
    save_freq = max(args.save_freq_steps, args.n_steps * args.n_envs_total)

    common = dict(
        data_path            = args.data,
        ckpt_path            = args.ckpt,
        output_dir           = args.output,
        model_type           = args.model_type,
        gpu_type             = args.gpu_type,
        seq_len              = args.seq_len,
        n_gpus               = args.n_gpus,
        n_envs_per_gpu       = args.n_envs_per_gpu,
        n_steps              = args.n_steps,
        batch_size           = args.batch_size,
        n_epochs             = args.n_epochs,
        gamma                = args.gamma,
        gae_lambda           = args.gae_lambda,
        clip_range           = args.clip_range,
        ent_coef             = args.ent_coef,
        vf_coef              = args.vf_coef,
        max_grad_norm        = args.max_grad_norm,
        logger               = logger,
        eval_every_n_rollouts= args.eval_every_n_rollouts,
        n_eval_episodes      = args.n_eval_episodes,
        save_freq_steps      = save_freq,
    )

    phase1_model_path = args.phase1_model

    # ── 阶段 1 ────────────────────────────────────────────────────
    if args.phase1_steps > 0 and phase1_model_path is None:
        phase1_model_path = train_phase_multigpu(
            phase=1,
            total_steps=args.phase1_steps,
            learning_rate=args.phase1_lr,
            **common,
        )
    elif args.phase1_steps == 0:
        logger.info("⏭  跳过阶段 1（phase1-steps=0）")

    # ── 阶段 2 ────────────────────────────────────────────────────
    if args.phase2_steps > 0:
        if phase1_model_path is None:
            raise ValueError(
                "阶段 2 需要阶段 1 的模型，"
                "请提供 --phase1-model 或设置 --phase1-steps > 0"
            )
        phase2_model_path = train_phase_multigpu(
            phase=2,
            total_steps=args.phase2_steps,
            learning_rate=args.phase2_lr,
            phase1_model_path=phase1_model_path,
            **common,
        )
        logger.info(f"🎉 全部训练完成！最终模型: {phase2_model_path}")
    else:
        logger.info("⏭  跳过阶段 2（phase2-steps=0）")
        logger.info(f"🎉 训练完成！阶段 1 模型: {phase1_model_path}")


# ─────────────────────────────────────────────
# 程序入口（spawn 模式必须在此保护下）
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # spawn 模式：子进程重新初始化解释器，可安全使用 CUDA，
    # 必须在 if __name__ == '__main__' 保护块内设置，否则递归 spawn
    mp.set_start_method('spawn', force=True)
    main()
