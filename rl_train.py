#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_train.py - 两阶段 PPO 训练脚本

阶段 1（phase1）：冻结编码器，只训练 PPO Policy Net
    → 让策略先适应 256 维时序特征空间
    → 默认 100_000 步

阶段 2（phase2）：解冻编码器，端到端微调
    → 编码器适应资源调度任务的语义需求
    → 使用更小学习率（1e-5）
    → 默认 200_000 步

依赖：
    pip install stable-baselines3>=2.0 gymnasium torch

用法示例：
    # 两阶段完整训练（PPO Policy Net 自动跑 GPU）
    python rl_train.py \\
        --data   simulation_data/combined_simulation_data.csv \\
        --ckpt   models/lstm_traffic_best.pth \\
        --model-type GPT-4 \\
        --phase1-steps 100000 \\
        --phase2-steps 200000 \\
        --output rl_models/

    # 只跑阶段 1（调试用）
    python rl_train.py --data ... --ckpt ... --phase2-steps 0

设备说明：
    - PPO Policy Net 由 SB3 自动选择设备（有 GPU 就用 GPU），无需手动指定。
    - SubprocVecEnv 子进程中的 LSTM 编码器固定跑 CPU。
      原因：CUDA 不支持 fork 多进程共享上下文，强制用 cuda 会死锁或报错。
      推理量很小（seq_len=60 步前向），CPU 完全够用，不影响整体训练速度。
    - Phase 2 端到端微调时，编码器梯度由主进程 PPO（GPU）统一计算，
      子进程只负责采样（forward-only on CPU），架构完全正确。
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, IO

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from rl_env import PredictiveResourceEnv

# ─────────────────────────────────────────────
# 日志
# ─────────────────────────────────────────────

class _Tee:
    """同时将 stdout/stderr 写入文件，实现终端显示与文件保存并行。

    写入文件时自动过滤 tqdm 进度条行，保留 SB3 的 rollout/train/eval 统计表格。

    【过滤策略】
    tqdm 开启后，每次 write() 调用可能携带多行内容混杂在一起：
        - 真正的进度条行：含 \\r（tqdm 覆写刷新），去掉 ANSI 后只剩进度条符号
        - SB3 统计表格行：含实质文字/数字，去掉 ANSI 后有有意义的内容
        - 纯 ANSI 控制行：仅光标移动/隐藏/显示等，去掉后为空

    旧实现对整个 write() 数据块整体判断，只要数据中含 \\r 就丢弃全部内容，
    导致 tqdm 把 SB3 统计表格夹在进度条刷新里一起传入时，表格被整体误杀。

    新实现：先按 \\r 和 \\n 将数据拆成独立行，逐行判断是否为进度条噪音，
    只丢弃真正的进度条行，有实质内容的行（SB3 表格等）正常写入文件。
    """

    import re as _re
    # 匹配所有 ANSI 转义序列（颜色、光标移动、隐藏/显示光标等）
    _ANSI_RE = _re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')
    # tqdm 进度条特征：百分号、进度数字/斜杠、方括号填充字符（█ ▏▎▍▌▋▊▉ ━ = > 等）
    _PROG_RE = _re.compile(r'^\s*[\d]+%|^\s*[|█▏▎▍▌▋▊▉━=>\s\[\]]+\s*$')

    def __init__(self, stream: IO, file_path: str):
        self._stream = stream
        self._file   = open(file_path, 'a', buffering=1, encoding='utf-8')

    @classmethod
    def _is_noise_line(cls, line: str) -> bool:
        """
        判断单行是否为应丢弃的噪音（进度条或纯 ANSI 控制序列）。

        保留条件（任意一条满足即保留）：
          - 去掉 ANSI 后，含字母、中文、'|'、'$'、'+' 等实质字符
        丢弃条件：
          - 去掉 ANSI 后为空白（纯控制码行）
          - 去掉 ANSI 后匹配进度条特征模式（数字%、进度块填充等）
        """
        cleaned = cls._ANSI_RE.sub('', line)
        stripped = cleaned.strip()
        if not stripped:
            return True   # 纯 ANSI 控制行，丢弃
        if cls._PROG_RE.match(stripped):
            return True   # 进度条特征行，丢弃
        return False

    def write(self, data: str):
        self._stream.write(data)

        # 按 \r 拆分（tqdm 用 \r 覆写），再按 \n 拆分，逐行判断
        # 只将有实质内容的行写入文件，去掉残余 ANSI 码保证纯文本
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
        # 部分库（如 tqdm）会调用 fileno()，回退到原始流
        return self._stream.fileno()

    def close(self):
        self._file.close()

    # 透传其余属性（isatty 等）
    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _setup_logger(log_dir: str, tag: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger  = logging.getLogger(f"rl_train.{tag}.{ts}")
    logger.setLevel(logging.INFO)
    fmt     = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    fh      = logging.FileHandler(os.path.join(log_dir, f'{tag}_{ts}.log'))
    ch      = logging.StreamHandler()
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def _redirect_stdout_stderr(log_dir: str, tag: str) -> str:
    """将 stdout 和 stderr 同时 tee 到同一个 raw 日志文件（含 SB3 表格输出）。

    Returns:
        raw_log_path: tee 文件路径
    """
    os.makedirs(log_dir, exist_ok=True)
    ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_log_path = os.path.join(log_dir, f'{tag}_raw_{ts}.log')
    sys.stdout   = _Tee(sys.stdout, raw_log_path)
    sys.stderr   = _Tee(sys.stderr, raw_log_path)
    return raw_log_path


# ─────────────────────────────────────────────
# 辅助：批量环境工厂
# ─────────────────────────────────────────────

def _make_env_fn(data_path, ckpt_path, model_type, gpu_type, seq_len, freeze,
                 env_device: str = 'cpu'):
    """返回一个无参可调用对象（供 make_vec_env 使用）。

    【设备说明】
    - 使用 SubprocVecEnv(start_method='spawn') 时，子进程重新初始化解释器，
      可以安全使用 CUDA，env_device 可传 'cuda'，LSTM 推理也跑在 GPU 上。
    - 使用默认 fork 方式时，CUDA 上下文无法安全复制，env_device 须保持 'cpu'。
    PPO Policy Net 的设备由 SB3 独立管理（传入 device='auto' 时自动选 GPU）。
    """
    def _init():
        env = PredictiveResourceEnv(
            data_path=data_path,
            checkpoint_path=ckpt_path,
            model_type=model_type,
            gpu_type=gpu_type,
            seq_len=seq_len,
            device=env_device,
        )
        # 同步编码器冻结状态
        if freeze:
            env.encoder.freeze_encoder()
        else:
            env.encoder.unfreeze_encoder()
        return Monitor(env)
    return _init


# ─────────────────────────────────────────────
# 策略网络配置
# ─────────────────────────────────────────────

# 阶段 1：较大学习率，编码器冻结
PHASE1_POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256, 128],   # Actor
        vf=[256, 256, 128],   # Critic
    ),
    activation_fn=torch.nn.ReLU,
)

# 阶段 2：端到端微调，使用稍小网络避免过拟合
PHASE2_POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 128],
        vf=[256, 128],
    ),
    activation_fn=torch.nn.ReLU,
)


# ─────────────────────────────────────────────
# 阶段 1：冻结编码器训练
# ─────────────────────────────────────────────

def train_phase1(
    data_path: str,
    ckpt_path: str,
    output_dir: str,
    model_type: str = 'GPT-4',
    gpu_type: str = 'A100',
    seq_len: int = 60,
    n_envs: int = 4,
    total_steps: int = 100_000,
    learning_rate: float = 3e-4,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    阶段 1：冻结 LSTM 编码器，只训练 PPO 策略网络。
    PPO Policy Net 使用 device='auto'，SB3 会自动选择 GPU（若可用）。

    Returns:
        phase1_model_path: 保存的模型文件路径
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("▶ 阶段 1：冻结编码器，训练 PPO 策略网络")
    logger.info(f"   模型类型={model_type}, GPU={gpu_type}, 并行环境数={n_envs}")
    logger.info(f"   训练步数={total_steps:,}, 学习率={learning_rate}")
    logger.info("=" * 60)

    # 检测实际训练设备（SB3 auto 逻辑：有 cuda 用 cuda，否则 cpu）
    ppo_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("=" * 60)
    logger.info("▶ 阶段 1：冻结编码器，训练 PPO 策略网络")
    logger.info(f"   模型类型={model_type}, GPU={gpu_type}, 并行环境数={n_envs}")
    logger.info(f"   训练步数={total_steps:,}, 学习率={learning_rate}")
    logger.info(f"   PPO Policy 设备={ppo_device}  (env 内编码器推理固定 CPU)")
    logger.info("=" * 60)

    # ── 构建并行训练环境 ──────────────────────────────────────────
    # spawn 方式启动子进程：子进程重新初始化 Python 解释器，可安全使用 CUDA，
    # LSTM 编码器推理也能跑在 GPU 上，加速 obs 生成。
    env_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = make_vec_env(
        env_id=_make_env_fn(data_path, ckpt_path, model_type, gpu_type,
                            seq_len, freeze=True, env_device=env_device),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={'start_method': 'spawn'},  # spawn 允许子进程使用 CUDA
    )
    # 对 obs 和 reward 做在线归一化（不归一化 obs 里的 [0,1] 利用率也没关系，
    # 因为编码器输出的 256 维量级不一，归一化非常重要）
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    # ── 构建单个评估环境 ──────────────────────────────────────────
    # eval_env 用 DummyVecEnv（单进程），不需要 start_method
    eval_env = make_vec_env(
        env_id=_make_env_fn(data_path, ckpt_path, model_type, gpu_type,
                            seq_len, freeze=True, env_device=env_device),
        n_envs=1,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    os.makedirs(output_dir, exist_ok=True)
    phase1_dir = os.path.join(output_dir, 'phase1')
    os.makedirs(phase1_dir, exist_ok=True)

    # ── 回调 ─────────────────────────────────────────────────────
    # eval_freq 对齐到 rollout 长度（n_steps × n_envs）的整数倍，
    # 保证评估时 PPO 已完成 update，train/ 指标不会缺失。
    # SB3 的 eval_freq 单位是"每个子进程的步数"，所以直接用 n_steps 即可。
    rollout_steps = 2048        # 与 PPO 的 n_steps 保持一致
    eval_every_n_rollouts = 4   # 每 4 次 rollout（即 4×n_envs×n_steps 步）评估一次
    eval_freq = rollout_steps * eval_every_n_rollouts
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(phase1_dir, 'best'),
        log_path=os.path.join(phase1_dir, 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20_000 // n_envs, 1),
        save_path=os.path.join(phase1_dir, 'checkpoints'),
        name_prefix='ppo_phase1',
        verbose=1,
    )

    # ── PPO 模型（device='auto'：SB3 自动选 GPU）─────────────────
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # 熵正则：鼓励探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=PHASE1_POLICY_KWARGS,
        tensorboard_log=os.path.join(phase1_dir, 'tb_logs'),
        device='auto',         # 自动选择 GPU（若可用），不再硬编码 'cpu'
        verbose=1,
    )

    logger.info(f"PPO 模型参数量: {sum(p.numel() for p in model.policy.parameters()):,}")
    logger.info(f"PPO Policy 实际运行设备: {model.device}")

    # ── 训练 ─────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([eval_callback, checkpoint_callback]),
        progress_bar=True,
    )

    # ── 保存 ─────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(phase1_dir, f'ppo_phase1_final_{ts}')
    model.save(model_path)
    # 同时保存 VecNormalize 统计量（推理时需要）
    train_env.save(model_path + '_vecnorm.pkl')

    logger.info(f"✅ 阶段 1 完成，模型已保存: {model_path}")
    train_env.close()
    eval_env.close()
    return model_path


# ─────────────────────────────────────────────
# 阶段 2：解冻编码器端到端微调
# ─────────────────────────────────────────────

def train_phase2(
    phase1_model_path: str,
    data_path: str,
    ckpt_path: str,
    output_dir: str,
    model_type: str = 'GPT-4',
    gpu_type: str = 'A100',
    seq_len: int = 60,
    n_envs: int = 4,
    total_steps: int = 200_000,
    learning_rate: float = 1e-5,   # 端到端微调用小学习率
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    阶段 2：解冻编码器，从阶段 1 权重继续端到端微调。
    PPO Policy Net（含编码器梯度）使用 device='auto'，SB3 自动选 GPU。

    Returns:
        phase2_model_path: 保存的模型文件路径
    """
    ppo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("▶ 阶段 2：解冻编码器，端到端微调")
    logger.info(f"   从 {phase1_model_path} 加载阶段 1 权重")
    logger.info(f"   训练步数={total_steps:,}, 学习率={learning_rate}")
    logger.info(f"   PPO Policy 设备={ppo_device}  (env 内编码器推理固定 CPU)")
    logger.info("=" * 60)

    # ── 构建环境（编码器解冻）────────────────────────────────────
    env_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = make_vec_env(
        env_id=_make_env_fn(data_path, ckpt_path, model_type, gpu_type,
                            seq_len, freeze=False, env_device=env_device),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={'start_method': 'spawn'},
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    # eval_env 用 DummyVecEnv（单进程），不需要 start_method
    eval_env = make_vec_env(
        env_id=_make_env_fn(data_path, ckpt_path, model_type, gpu_type,
                            seq_len, freeze=False, env_device=env_device),
        n_envs=1,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    os.makedirs(output_dir, exist_ok=True)
    phase2_dir = os.path.join(output_dir, 'phase2')
    os.makedirs(phase2_dir, exist_ok=True)

    # ── 加载阶段 1 模型权重（device='auto' 自动放到 GPU）─────────
    model = PPO.load(
        phase1_model_path,
        env=train_env,
        device='auto',         # 自动选择 GPU（若可用）
    )
    # 更新学习率（端到端微调用更小的学习率）
    model.learning_rate = learning_rate
    model.lr_schedule   = lambda _: learning_rate

    logger.info(f"阶段 1 权重加载成功，实际运行设备: {model.device}")
    logger.info("开始端到端微调...")

    # ── 回调 ─────────────────────────────────────────────────────
    rollout_steps = 2048
    eval_every_n_rollouts = 4
    eval_freq = rollout_steps * eval_every_n_rollouts
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(phase2_dir, 'best'),
        log_path=os.path.join(phase2_dir, 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20_000 // n_envs, 1),
        save_path=os.path.join(phase2_dir, 'checkpoints'),
        name_prefix='ppo_phase2',
        verbose=1,
    )

    # ── 继续训练 ─────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([eval_callback, checkpoint_callback]),
        reset_num_timesteps=False,  # 保留阶段 1 的时间步计数
        progress_bar=True,
    )

    # ── 保存 ─────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(phase2_dir, f'ppo_phase2_final_{ts}')
    model.save(model_path)
    train_env.save(model_path + '_vecnorm.pkl')

    logger.info(f"✅ 阶段 2 完成，最终模型已保存: {model_path}")
    train_env.close()
    eval_env.close()
    return model_path


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='两阶段 PPO 训练：预测感知型 LLM 资源调度',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据与模型路径
    parser.add_argument('--data',       type=str,
                        default='simulation_data/combined_simulation_data.csv',
                        help='仿真数据路径（combined_simulation_data.csv）')
    parser.add_argument('--ckpt',       type=str, required=True,
                        help='预训练 LSTM checkpoint 路径（.pth）')
    parser.add_argument('--output',     type=str, default='rl_models',
                        help='输出目录')

    # 场景参数
    parser.add_argument('--model-type', type=str, default='GPT-4',
                        choices=['GPT-4', 'ChatGLM', 'Claude', 'LLaMA'],
                        help='要训练的模型场景')
    parser.add_argument('--load-pattern',  type=str, default='Daily',
                        choices=['Daily', 'Weekly', 'Burst', 'Steady'],
                        help='负载模式（仅用于日志和输出目录命名，不影响数据过滤）')
    parser.add_argument('--gpu-type',   type=str, default='A100',
                        choices=['A100', 'H100'],
                        help='初始 GPU 型号')
    parser.add_argument('--seq-len',    type=int, default=60,
                        help='历史序列窗口长度（与 LSTM 训练时一致）')

    # 训练参数
    parser.add_argument('--n-envs',          type=int,   default=4,
                        help='并行环境数')
    parser.add_argument('--phase1-steps',    type=int,   default=100_000,
                        help='阶段 1 训练步数（0=跳过）')
    parser.add_argument('--phase2-steps',    type=int,   default=200_000,
                        help='阶段 2 训练步数（0=跳过）')
    parser.add_argument('--phase1-lr',       type=float, default=3e-4,
                        help='阶段 1 学习率')
    parser.add_argument('--phase2-lr',       type=float, default=1e-5,
                        help='阶段 2 学习率（端到端微调）')

    # 从阶段 1 模型继续（仅跑阶段 2 时使用）
    parser.add_argument('--phase1-model',    type=str,   default=None,
                        help='已有的阶段 1 模型路径（跳过阶段 1 时使用）')

    args = parser.parse_args()

    # ── 路径校验 ──────────────────────────────────────────────────
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"仿真数据文件不存在: {args.data}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"LSTM checkpoint 不存在: {args.ckpt}")

    ppo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ── 日志 ──────────────────────────────────────────────────────
    log_dir = os.path.join(args.output, 'logs')
    log_tag = f"rl_train_{args.model_type}_{args.load_pattern}"
    logger  = _setup_logger(log_dir, log_tag)
    # 将 stdout/stderr（含 SB3 rollout/train/eval 表格）同步写入 raw log 文件
    raw_log = _redirect_stdout_stderr(log_dir, log_tag)
    logger.info(f"终端原始输出（rollout/train/eval 表格）同步保存至: {raw_log}")
    logger.info("═" * 60)
    logger.info("  预测感知型 LLM 资源调度 RL 训练")
    logger.info(f"  数据: {args.data}")
    logger.info(f"  LSTM ckpt: {args.ckpt}")
    logger.info(f"  场景: model_type={args.model_type}, load_pattern={args.load_pattern}, gpu_type={args.gpu_type}")
    logger.info(f"  PPO 训练设备: {ppo_device} "
          f"({'GPU 数量: ' + str(torch.cuda.device_count()) if ppo_device == 'cuda' else 'no GPU'})")
    logger.info(f"  env 内编码器推理设备: cpu (SubprocVecEnv 子进程固定)")
    logger.info("═" * 60)

    phase1_model_path = args.phase1_model

    # ── 阶段 1 ────────────────────────────────────────────────────
    if args.phase1_steps > 0 and phase1_model_path is None:
        phase1_model_path = train_phase1(
            data_path=args.data,
            ckpt_path=args.ckpt,
            output_dir=args.output,
            model_type=args.model_type,
            gpu_type=args.gpu_type,
            seq_len=args.seq_len,
            n_envs=args.n_envs,
            total_steps=args.phase1_steps,
            learning_rate=args.phase1_lr,
            logger=logger,
        )
    elif args.phase1_steps == 0:
        logger.info("⏭  跳过阶段 1（phase1-steps=0）")

    # ── 阶段 2 ────────────────────────────────────────────────────
    if args.phase2_steps > 0:
        if phase1_model_path is None:
            raise ValueError(
                "阶段 2 需要阶段 1 的模型，请提供 --phase1-model 或设置 --phase1-steps > 0"
            )
        phase2_model_path = train_phase2(
            phase1_model_path=phase1_model_path,
            data_path=args.data,
            ckpt_path=args.ckpt,
            output_dir=args.output,
            model_type=args.model_type,
            gpu_type=args.gpu_type,
            seq_len=args.seq_len,
            n_envs=args.n_envs,
            total_steps=args.phase2_steps,
            learning_rate=args.phase2_lr,
            logger=logger,
        )
        logger.info(f"🎉 训练全部完成！最终模型: {phase2_model_path}")
    else:
        logger.info("⏭  跳过阶段 2（phase2-steps=0）")
        logger.info(f"🎉 训练完成！阶段 1 模型: {phase1_model_path}")


if __name__ == '__main__':
    main()
