#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ChatGLM 资源调度 PPO 多卡训练 + 评测一体化脚本
#
# 按顺序完成：
#   1. 两阶段 PPO 多卡训练（与 run_rl_train_ddp.sh 参数完全一致）
#   2. 对训练产出的 best 模型进行自动评测（评测本身单卡即可，
#      与 run_rl_evaluate_ddp.sh 参数完全一致）
#
# 多卡策略:
#   N_ENVS 个 env 均分到所有可用 GPU 上做 rollout 采样，
#   主进程（cuda:0）汇总所有卡的 rollout buffer 后执行 PPO update。
#   训练配置（n_envs / batch_size / n_steps 等）与单卡 run_rl_train.sh 保持一致，
#   便于与单卡训练结果直接对比。
#
# 数据: ChatGLM_Daily.csv (10080 行, model_type=ChatGLM, gpu_type=A100)
# LSTM: lstm_traffic_20260709_120400.pth (feature_dim=27, sequence_length=360)
#
# 用法:
#   bash run_rl_train_and_evaluate_ddp.sh                                    # 完整训练 + 自动评测
#   bash run_rl_train_and_evaluate_ddp.sh phase1                             # 只跑阶段 1 + 评测
#   bash run_rl_train_and_evaluate_ddp.sh phase2 <phase1_model_path>         # 只跑阶段 2 + 评测
#   bash run_rl_train_and_evaluate_ddp.sh --tag <suffix>                     # 带自定义后缀
#   bash run_rl_train_and_evaluate_ddp.sh phase1 --tag <suffix>              # 组合使用
#   bash run_rl_train_and_evaluate_ddp.sh phase2 <phase1_model_path> --tag <suffix>
#   CUDA_VISIBLE_DEVICES=1,2,3,4 bash run_rl_train_and_evaluate_ddp.sh --tag rm_overload_penalty_rm_dou_penity_100k+600k_new
# --tag 说明:
#   为训练目录和评测目录追加自定义后缀，便于区分不同实验配置，例如:
#     --tag 0.2overload_penalty  →  ChatGLM_Daily_<ts>_0.2overload_penalty
#     --tag lr1e-4_n8            →  ChatGLM_Daily_<ts>_lr1e-4_n8
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── 激活环境 ─────────────────────────────────────────────────────────────────
source activate gpu

# ── 路径 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${SCRIPT_DIR}/simulation_data/ChatGLM_Daily.csv"
CKPT="${SCRIPT_DIR}/models/lstm_traffic_20260709_120400.pth"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "${SCRIPT_DIR}"

# ── 场景参数（与单卡 run_rl_train.sh 完全一致）────────────────────────────────
MODEL_TYPE="ChatGLM"
LOAD_PATTERN="Daily"
GPU_TYPE="A100"
SEQ_LEN=360          # 以 checkpoint 内实际保存的 sequence_length 为准

# ── 多卡参数 ──────────────────────────────────────────────────────────────────
N_GPUS=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
N_ENVS=4             # 总并行环境数（与单卡 N_ENVS 含义一致，自动均分到每卡）

# ── PPO 超参数（与单卡 run_rl_train.sh 完全一致）─────────────────────────────
PHASE1_STEPS=100000
PHASE2_STEPS=400000
PHASE1_LR=3e-4
PHASE2_LR=1e-5
N_STEPS=2048         # 每轮 rollout 每个 env 采集的步数
BATCH_SIZE=256       # PPO minibatch 大小（与单卡一致）
N_EPOCHS=10

# ── 评测参数（与 run_rl_evaluate_ddp.sh 完全一致）────────────────────────────
EPISODES=10
MAX_STEPS=2000
DEVICE="cuda"

# ── 解析参数（支持 --tag <suffix> 选项）─────────────────────────────────────
# 参数布局：
#   $1 = mode（both | phase1 | phase2），缺省为 both
#   $2 = phase1 模型路径（仅 phase2 模式需要）
#   --tag <suffix> 可出现在任意位置
MODE="both"
PHASE1_MODEL=""
DIR_SUFFIX=""
OUTPUT=""   # 待参数解析完成后再赋值

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            DIR_SUFFIX="_${2}"
            shift 2
            ;;
        phase1|phase2|both)
            MODE="$1"
            shift
            ;;
        *)
            # phase2 模式下的第一个非选项参数视为 phase1 模型路径
            if [ "${MODE}" = "phase2" ] && [ -z "${PHASE1_MODEL}" ]; then
                PHASE1_MODEL="$1"
            else
                echo "[ERROR] 未知参数: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# OUTPUT 在参数解析完成后赋值，确保 DIR_SUFFIX 已生效
OUTPUT="${SCRIPT_DIR}/rl_models_ddp/ChatGLM_Daily_${TIMESTAMP}${DIR_SUFFIX}"

echo "================================================================"
echo "  ChatGLM PPO 多卡训练 + 评测一体化流程"
echo "  训练模式    : ${MODE}"
echo "  数据        : ${DATA}"
echo "  CKPT        : ${CKPT}"
echo "  训练输出    : ${OUTPUT}"
echo "  目录后缀    : ${DIR_SUFFIX:-（无）}"
echo "  GPU 数量    : ${N_GPUS}"
echo "  总 env 数   : ${N_ENVS}  (每卡 $((( N_ENVS + N_GPUS - 1 ) / N_GPUS)) 个)"
echo "  load_pattern: ${LOAD_PATTERN}  seq_len=${SEQ_LEN}"
echo "  batch_size  : ${BATCH_SIZE}  n_steps=${N_STEPS}  n_epochs=${N_EPOCHS}"
echo "================================================================"

# ════════════════════════════════════════════════════════════════════════════
# 阶段一：PPO 多卡训练
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> [1/2] 开始 PPO 多卡训练  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "----------------------------------------------------------------"

if [ "${MODE}" = "phase1" ]; then
    # ── 只跑阶段 1 ────────────────────────────────────────────────
    python rl_train_ddp.py \
        --data           "${DATA}"         \
        --ckpt           "${CKPT}"         \
        --output         "${OUTPUT}"       \
        --model-type     "${MODEL_TYPE}"   \
        --load-pattern   "${LOAD_PATTERN}" \
        --gpu-type       "${GPU_TYPE}"     \
        --seq-len        "${SEQ_LEN}"      \
        --n-gpus         "${N_GPUS}"       \
        --n-envs         "${N_ENVS}"       \
        --phase1-steps   "${PHASE1_STEPS}" \
        --phase1-lr      "${PHASE1_LR}"    \
        --phase2-steps   0                 \
        --n-steps        "${N_STEPS}"      \
        --batch-size     "${BATCH_SIZE}"   \
        --n-epochs       "${N_EPOCHS}"

elif [ "${MODE}" = "phase2" ]; then
    # ── 只跑阶段 2（需要传入已有 phase1 模型路径）────────────────
    if [ -z "${PHASE1_MODEL}" ]; then
        echo "[ERROR] phase2 模式需要传入 phase1 模型路径，例如:"
        echo "  bash run_rl_train_and_evaluate_ddp.sh phase2 rl_models_ddp/.../phase1/ppo_phase1_final_xxx"
        exit 1
    fi
    python rl_train_ddp.py \
        --data           "${DATA}"          \
        --ckpt           "${CKPT}"          \
        --output         "${OUTPUT}"        \
        --model-type     "${MODEL_TYPE}"    \
        --load-pattern   "${LOAD_PATTERN}"  \
        --gpu-type       "${GPU_TYPE}"      \
        --seq-len        "${SEQ_LEN}"       \
        --n-gpus         "${N_GPUS}"        \
        --n-envs         "${N_ENVS}"        \
        --phase1-steps   0                  \
        --phase2-steps   "${PHASE2_STEPS}"  \
        --phase2-lr      "${PHASE2_LR}"     \
        --phase1-model   "${PHASE1_MODEL}"  \
        --n-steps        "${N_STEPS}"       \
        --batch-size     "${BATCH_SIZE}"    \
        --n-epochs       "${N_EPOCHS}"

else
    # ── 两阶段完整训练（默认）────────────────────────────────────
    python rl_train_ddp.py \
        --data           "${DATA}"          \
        --ckpt           "${CKPT}"          \
        --output         "${OUTPUT}"        \
        --model-type     "${MODEL_TYPE}"    \
        --load-pattern   "${LOAD_PATTERN}"  \
        --gpu-type       "${GPU_TYPE}"      \
        --seq-len        "${SEQ_LEN}"       \
        --n-gpus         "${N_GPUS}"        \
        --n-envs         "${N_ENVS}"        \
        --phase1-steps   "${PHASE1_STEPS}"  \
        --phase1-lr      "${PHASE1_LR}"     \
        --phase2-steps   "${PHASE2_STEPS}"  \
        --phase2-lr      "${PHASE2_LR}"     \
        --n-steps        "${N_STEPS}"       \
        --batch-size     "${BATCH_SIZE}"    \
        --n-epochs       "${N_EPOCHS}"
fi

echo "----------------------------------------------------------------"
echo ">>> [1/2] 训练完成  ($(date '+%Y-%m-%d %H:%M:%S'))"

# ════════════════════════════════════════════════════════════════════════════
# 阶段二：自动评测（单卡/CPU 推理即可，无需多卡）
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> [2/2] 开始自动评测  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "----------------------------------------------------------------"

# 推断 best 模型和 vecnorm 路径（与 run_rl_evaluate_ddp.sh 逻辑一致）
# 注意：多卡训练脚本保存的 best 模型文件名是 {phase_name}_best（如 phase2_best）
RL_MODEL="${OUTPUT}/phase2/best/phase2_best"
VECNORM=$(ls "${OUTPUT}/phase2/"*"_vecnorm.pkl" 2>/dev/null | head -1)

# 校验模型文件是否生成成功
if [ ! -f "${RL_MODEL}.zip" ]; then
    echo "[WARN] 未找到 best 模型: ${RL_MODEL}.zip"
    echo "       当前训练模式为 '${MODE}'，可能未产出 phase2 best 模型，跳过评测。"
    echo "       如需评测，请手动运行 run_rl_evaluate_ddp.sh 并指定模型路径。"
    exit 0
fi

# 评测结果输出目录（与训练目录使用相同后缀，便于对应关联）
TRAIN_TAG="ChatGLM_Daily_${TIMESTAMP}${DIR_SUFFIX}"
EVAL_OUTPUT="${SCRIPT_DIR}/rl_eval_results_ddp/${TRAIN_TAG}_eval_$(date +%Y%m%d_%H%M%S)"

echo "  RL 模型:  ${RL_MODEL}.zip"
echo "  VecNorm:  ${VECNORM:-（未找到，跳过 obs 归一化）}"
echo "  评测输出: ${EVAL_OUTPUT}"
echo "  episodes=${EPISODES}, max_steps=${MAX_STEPS}, device=${DEVICE}"

# 组装 vecnorm 参数
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
    --output      "${EVAL_OUTPUT}"  \
    --device      "${DEVICE}"

echo "----------------------------------------------------------------"
echo ">>> [2/2] 评测完成  ($(date '+%Y-%m-%d %H:%M:%S'))"

echo ""
echo "================================================================"
echo "  ✅ 全流程完成！"
echo "  训练产出: ${OUTPUT}"
echo "  评测结果: ${EVAL_OUTPUT}"
echo "================================================================"
