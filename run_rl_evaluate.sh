#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ChatGLM 资源调度 PPO 评测脚本
#
# 数据: ChatGLM_Daily.csv (10080 行, model_type=ChatGLM, gpu_type=A100)
# LSTM: lstm_traffic_20260709_120400.pth (feature_dim=27, sequence_length=360)
#
# 用法:
#   bash run_rl_evaluate.sh                        # 评测最新训练目录的 best_model
#   bash run_rl_evaluate.sh <rl_model_dir>         # 评测指定训练目录（自动推断路径）
#   bash run_rl_evaluate.sh <rl_model> <vecnorm>   # 手动指定模型和 vecnorm 路径
#
# 示例:
#   bash run_rl_evaluate.sh \
#       rl_models/ChatGLM_Daily_20260720_160523/phase2/best/best_model \
#       rl_models/ChatGLM_Daily_20260720_160523/phase2/ppo_phase2_final_20260720_164605_vecnorm.pkl
#   bash run_rl_evaluate.sh rl_models/ChatGLM_Daily_20260724_132815_rm_overload_penalty_rm_dou_penity_100k+600k
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── 激活环境 ─────────────────────────────────────────────────────────────────
source activate gpu

# ── 固定路径 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${SCRIPT_DIR}/simulation_data/ChatGLM_Daily.csv"
CKPT="${SCRIPT_DIR}/models/lstm_traffic_20260709_120400.pth"

cd "${SCRIPT_DIR}"

# ── 评测参数（与训练时保持一致）─────────────────────────────────────────────
MODEL_TYPE="ChatGLM"
GPU_TYPE="A100"
SEQ_LEN=360          # 以 checkpoint 内实际保存的 sequence_length 为准
EPISODES=10          # 每个策略评估的 episode 数
MAX_STEPS=2000       # 每个 episode 最大步数
DEVICE="cuda"        # 使用 GPU 加速推理（LSTM 136万参数，cuda 比 cpu 快 20~30%）

# ── 解析参数，推断 RL 模型路径和 vecnorm 路径 ─────────────────────────────────
ARG1="${1:-}"
ARG2="${2:-}"

if [ -z "${ARG1}" ]; then
    # ── 未传参数：自动找 rl_models/ 下最新的训练目录 ──────────────────────────
    LATEST_DIR=$(ls -dt "${SCRIPT_DIR}/rl_models/ChatGLM_Daily_"*/ 2>/dev/null | head -1)
    if [ -z "${LATEST_DIR}" ]; then
        echo "[ERROR] 未找到任何训练目录，请先运行 run_rl_train.sh 或手动指定模型路径"
        exit 1
    fi
    LATEST_DIR="${LATEST_DIR%/}"   # 去掉末尾 /
    RL_MODEL="${LATEST_DIR}/phase2/best/best_model"
    # vecnorm 用 phase2 下 final 保存的（best 目录无单独 vecnorm）
    VECNORM=$(ls "${LATEST_DIR}/phase2/"*"_vecnorm.pkl" 2>/dev/null | head -1)
    echo "[INFO] 自动选取最新训练目录: ${LATEST_DIR}"

elif [ -d "${ARG1}" ] || [[ "${ARG1}" == */ ]]; then
    # ── 传入训练目录：自动推断内部路径 ───────────────────────────────────────
    TRAIN_DIR="${ARG1%/}"
    RL_MODEL="${TRAIN_DIR}/phase2/best/best_model"
    VECNORM=$(ls "${TRAIN_DIR}/phase2/"*"_vecnorm.pkl" 2>/dev/null | head -1)
    echo "[INFO] 使用训练目录: ${TRAIN_DIR}"

else
    # ── 手动指定模型路径（和可选的 vecnorm 路径）─────────────────────────────
    RL_MODEL="${ARG1}"
    VECNORM="${ARG2}"
fi

# ── 校验模型文件 ──────────────────────────────────────────────────────────────
if [ ! -f "${RL_MODEL}.zip" ]; then
    echo "[ERROR] RL 模型文件不存在: ${RL_MODEL}.zip"
    exit 1
fi

# ── 输出目录（以模型所在目录的倒数第3级目录名 + 时间戳命名）─────────────────
# 例：ChatGLM_Daily_20260720_160523 → rl_eval_results/ChatGLM_Daily_20260720_160523_eval_<ts>
TRAIN_TAG=$(echo "${RL_MODEL}" | grep -oP 'ChatGLM_Daily_\d{8}_\d{6}' | head -1)
if [ -z "${TRAIN_TAG}" ]; then
    TRAIN_TAG="unknown"
fi
OUTPUT="${SCRIPT_DIR}/rl_eval_results/${TRAIN_TAG}_eval_$(date +%Y%m%d_%H%M%S)"

# ── 打印配置 ─────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  ChatGLM PPO 资源调度评测"
echo "  数据:      ${DATA}"
echo "  LSTM ckpt: ${CKPT}"
echo "  RL 模型:   ${RL_MODEL}.zip"
echo "  VecNorm:   ${VECNORM:-（不使用）}"
echo "  输出:      ${OUTPUT}"
echo "  episodes=${EPISODES}, max_steps=${MAX_STEPS}, device=${DEVICE}"
echo "================================================================"

# ── 运行评测 ─────────────────────────────────────────────────────────────────
VECNORM_ARG=""
if [ -n "${VECNORM}" ] && [ -f "${VECNORM}" ]; then
    VECNORM_ARG="--vecnorm ${VECNORM}"
else
    echo "[WARN] 未找到 vecnorm 文件，将跳过 obs 归一化（评测结果可能偏低）"
fi

python rl_evaluate.py \
    --data        "${DATA}"         \
    --ckpt        "${CKPT}"         \
    --rl-model    "${RL_MODEL}"     \
    ${VECNORM_ARG}                  \
    --model-type  "${MODEL_TYPE}"   \
    --gpu-type    "${GPU_TYPE}"     \
    --seq-len     "${SEQ_LEN}"      \
    --episodes    "${EPISODES}"     \
    --max-steps   "${MAX_STEPS}"    \
    --output      "${OUTPUT}"       \
    --device      "${DEVICE}"

echo "================================================================"
echo "  评测完成！结果目录: ${OUTPUT}"
echo "================================================================"
