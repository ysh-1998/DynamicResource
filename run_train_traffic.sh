#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# 流量预测模型（AttentionLSTM 等）训练脚本
#
# 训练 train_traffic.py，支持单卡 / 多卡两种模式：
#   single : 单卡训练（CUDA_VISIBLE_DEVICES 取 CUDA_VISIBLE_DEVICES 环境变量或 0）
#   multi  : 多卡训练（通过 --gpu-ids 传逗号分隔的 GPU ID，脚本内由 Accelerate 自动 DDP）
#
# 模型: lstm | transformer | hybrid | ensemble
# 数据: simulation_data/combined_simulation_data.csv（由 resource_simulation.py 生成）
# 产出: models/<model>_traffic_<timestamp>.pth + training_results_<model>_<timestamp>.png
#
# 用法:
#   bash run_train_traffic.sh single                                # 单卡默认 (lstm)
#   bash run_train_traffic.sh multi  0,1,2                          # 多卡 3 GPU
#   bash run_train_traffic.sh single --model lstm --epochs 50
#   bash run_train_traffic.sh multi  0,1 --model transformer --batch-size 64
#   bash run_train_traffic.sh single --tag baseline                 # 带自定义后缀（仅用于日志标识）
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── 激活环境 ─────────────────────────────────────────────────────────────────
source activate gpu

# ── 路径 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/simulation_data"
cd "${SCRIPT_DIR}"

# ── 默认训练参数 ─────────────────────────────────────────────────────────────
MODEL="lstm"
EPOCHS=100
BATCH_SIZE=32
LR=0.001
SEQ_LEN=360
TAG=""

# ── 解析参数 ─────────────────────────────────────────────────────────────────
# 参数布局：
#   $1 = mode（single | multi），缺省为 single
#   $2 = gpu_ids（仅 multi 模式需要，如 "0,1,2"）
#   其余 --model/--epochs/--batch-size/--lr/--sequence-length/--tag 可出现在任意位置
MODE="single"
GPU_IDS=""

if [[ $# -gt 0 ]]; then
    case "$1" in
        single|multi)
            MODE="$1"
            shift
            ;;
        *)
            ;;
    esac
fi

# multi 模式：第一个非选项参数视为 gpu_ids
if [ "${MODE}" = "multi" ] && [[ $# -gt 0 ]]; then
    case "$1" in
        --*)
            ;;
        *)
            GPU_IDS="$1"
            shift
            ;;
    esac
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --lr)
            LR="$2"; shift 2 ;;
        --sequence-length)
            SEQ_LEN="$2"; shift 2 ;;
        --tag)
            TAG="_${2}"; shift 2 ;;
        *)
            echo "[ERROR] 未知参数: $1"
            exit 1
            ;;
    esac
done

# ── 校验数据 ─────────────────────────────────────────────────────────────────
COMBINED_CSV="${DATA_DIR}/combined_simulation_data.csv"
if [ ! -f "${COMBINED_CSV}" ]; then
    echo "[ERROR] 找不到数据文件: ${COMBINED_CSV}"
    echo "        请先运行: python resource_simulation.py"
    exit 1
fi

# ── 组装 GPU 参数 ────────────────────────────────────────────────────────────
GPU_ARG=""
if [ "${MODE}" = "multi" ]; then
    if [ -z "${GPU_IDS}" ]; then
        echo "[ERROR] multi 模式需要指定 GPU IDs，例如: bash run_train_traffic.sh multi 0,1,2"
        exit 1
    fi
    GPU_ARG="--gpu-ids ${GPU_IDS}"
    GPU_DESC="多卡 GPU=${GPU_IDS}"
else
    # 单卡模式：尊重已有 CUDA_VISIBLE_DEVICES，否则默认 0
    if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
        export CUDA_VISIBLE_DEVICES=0
    fi
    GPU_IDS="${CUDA_VISIBLE_DEVICES}"
    GPU_ARG="--gpu-ids ${GPU_IDS}"
    GPU_DESC="单卡 GPU=${GPU_IDS}"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo "  流量预测模型训练"
echo "  模式:      ${MODE}"
echo "  ${GPU_DESC}"
echo "  模型:      ${MODEL}"
echo "  数据:      ${COMBINED_CSV}"
echo "  epochs:    ${EPOCHS}, batch_size: ${BATCH_SIZE}, lr: ${LR}, seq_len: ${SEQ_LEN}"
echo "  时间戳:    ${TIMESTAMP}${TAG}"
echo "================================================================"

# ════════════════════════════════════════════════════════════════════════════
# 训练
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> 开始训练  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "----------------------------------------------------------------"

python train_traffic.py \
    --model            "${MODEL}"    \
    --epochs           "${EPOCHS}"   \
    --batch-size       "${BATCH_SIZE}" \
    --lr               "${LR}"       \
    --sequence-length  "${SEQ_LEN}"  \
    ${GPU_ARG}

echo "----------------------------------------------------------------"
echo ">>> 训练完成  ($(date '+%Y-%m-%d %H:%M:%S'))"

echo ""
echo "================================================================"
echo "  ✅ 训练完成！"
echo "  模型:   ${SCRIPT_DIR}/models/${MODEL}_traffic_${TIMESTAMP}.pth"
echo "  训练图: ${SCRIPT_DIR}/training_results_${MODEL}_${TIMESTAMP}.png"
echo "================================================================"
