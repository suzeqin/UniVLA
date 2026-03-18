#!/usr/bin/env bash
set -euo pipefail
###############################################################################
# UniVLA-7B 微调脚本 - LabUtopia LeRobot V3.0 数据
#
# 使用方法：
#   cd /data1/rbc/UniVLA
#   bash vla-scripts/finetune_labutopia.sh
#
# 前置条件：
#   1. UniVLA 环境已安装（需要 prismatic, draccus, peft, decord, ema_pytorch）
#   2. 预训练 UniVLA-7B 权重: /data/rbc/Embodied_models/univla-7b
#   3. LAM Stage-2 检查点: latent_action_model/logs/task_centric_lam_stage2/epoch=0-step=200000.ckpt
#   4. LeRobot 数据集: /data1/rbc/lerobot/.cache/LabUtopia/Level3_TransportBeaker
#
# 硬件需求：
#   至少 2 × A100-80GB（或等效 GPU），batch_size=4 × grad_accum=2 × 2GPU = 有效 batch 16
###############################################################################

######################### 环境配置 #########################

# CUDA 配置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-12.8"}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-""}

# Conda 环境（根据实际环境名称修改）
CONDA_ROOT=${CONDA_ROOT:-/data/rbc/miniconda3}
CONDA_ENV=${CONDA_ENV:-"univla"}
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source ${CONDA_ROOT}/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
fi

# 禁用不必要的网络访问
export WANDB_MODE=${WANDB_MODE:-"offline"}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

######################### 项目路径 #########################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${PROJ_ROOT}

######################### 训练参数 #########################

# ===== 模型路径（按需修改）=====
# UniVLA-7B 预训练权重路径
VLA_PATH="${VLA_PATH:-/data/rbc/Embodied_models/univla-7b}"

# LAM Stage-2 检查点路径（相对于项目根目录）
LAM_PATH="${LAM_PATH:-latent_action_model/logs/task_centric_lam_stage2/epoch=0-step=200000.ckpt}"

# ===== 数据集配置 =====
# LeRobot 数据集根目录
DATASET_ROOT="${DATASET_ROOT:-/data1/rbc/lerobot/.cache}"
# 数据集仓库 ID（对应 DATASET_ROOT 下的子目录路径）
REPO_ID="${REPO_ID:-LabUtopia/Level3_TransportBeaker}"
# 使用的相机（多相机用逗号分隔，如 "camera_1_rgb,camera_2_rgb"）
CAMERA_NAMES="${CAMERA_NAMES:-camera_1_rgb}"

# ===== 训练超参 =====
BATCH_SIZE=${BATCH_SIZE:-4}            # 每 GPU batch size
MAX_STEPS=${MAX_STEPS:-20000}          # 总训练步数
SAVE_STEPS=${SAVE_STEPS:-5000}         # 检查点保存间隔
LR=${LR:-3.5e-4}                       # 学习率
GRAD_ACCUM=${GRAD_ACCUM:-2}            # 梯度累积步数
WINDOW_SIZE=${WINDOW_SIZE:-12}         # 动作预测窗口大小
ACTION_DIM=${ACTION_DIM:-8}            # 动作维度（7关节+1夹爪）
NUM_WORKERS=${NUM_WORKERS:-4}          # 数据加载线程数

# ===== LoRA 配置 =====
USE_LORA=${USE_LORA:-true}             # 是否使用 LoRA（推荐 true 节省显存）
LORA_RANK=${LORA_RANK:-32}             # LoRA 秩
FREEZE_VLA=${FREEZE_VLA:-false}        # 是否冻结 VLA 骨干（仅训练 ActionDecoder）

echo "============================================================"
echo "  UniVLA-7B LabUtopia 微调"
echo "============================================================"
echo "  VLA 权重      : ${VLA_PATH}"
echo "  LAM 权重      : ${LAM_PATH}"
echo "  数据集根目录  : ${DATASET_ROOT}"
echo "  数据集        : ${REPO_ID}"
echo "  相机          : ${CAMERA_NAMES}"
echo "  Batch size    : ${BATCH_SIZE} × accum ${GRAD_ACCUM}"
echo "  总步数        : ${MAX_STEPS}"
echo "  学习率        : ${LR}"
echo "  LoRA          : ${USE_LORA} (rank=${LORA_RANK})"
echo "  窗口大小      : ${WINDOW_SIZE}"
echo "  动作维度      : ${ACTION_DIM}"
echo "============================================================"

# 使用 accelerate launch 进行分布式训练
accelerate launch \
    --multi_gpu \
    --num_processes=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) \
    vla-scripts/finetune_labutopia.py \
    --vla_path="${VLA_PATH}" \
    --lam_path="${LAM_PATH}" \
    --dataset_root="${DATASET_ROOT}" \
    --repo_id="${REPO_ID}" \
    --camera_names="${CAMERA_NAMES}" \
    --batch_size=${BATCH_SIZE} \
    --max_steps=${MAX_STEPS} \
    --save_steps=${SAVE_STEPS} \
    --learning_rate=${LR} \
    --grad_accumulation_steps=${GRAD_ACCUM} \
    --window_size=${WINDOW_SIZE} \
    --action_dim=${ACTION_DIM} \
    --num_workers=${NUM_WORKERS} \
    --use_lora=${USE_LORA} \
    --lora_rank=${LORA_RANK} \
    --freeze_vla=${FREEZE_VLA} \
    --wandb_project="univla-finetune-labutopia"
