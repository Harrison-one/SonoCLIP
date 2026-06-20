#!/usr/bin/env bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -x

PARTITION=llm3
JOB_NAME=sonoclip——
EXP_NAME=log/sonoclip——
UL_DATA_ROOT=../ul_data/FetalP24
BASE_MODEL_PATH=../checkpoints/ViT-L-14-336px.pt
# 默认重新训练；改成 checkpoint 路径才会续训
# RESUME_PATH=log/sonoclip/checkpoints/iter_xxxxx.pth
RESUME_PATH="none"

GPUS=2
GPUS_PER_NODE=2
CPUS_PER_TASK=6
# 指定使用的 GPU：取消下面注释并改成你要的 ID，如 0,1,2 或 2,3,4
# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=2,3

export PREFETCH_FACTOR=1
export TRAIN_NUM_WORKERS=8
export TEST_NUM_WORKERS=2

if [ "${RESUME_PATH}" = "none" ]; then
  RESUME_ARGS=""
else
  RESUME_ARGS="--resume --resume_path ${RESUME_PATH}"
fi

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$(pwd)/..:$PYTHONPATH"

if command -v srun &>/dev/null; then
  srun -p ${PARTITION} \
      --job-name=${JOB_NAME} \
      --gres=gpu:${GPUS_PER_NODE} \
      --ntasks=${GPUS} \
      --ntasks-per-node=${GPUS_PER_NODE} \
      --cpus-per-task=${CPUS_PER_TASK} \
      --quotatype=spot \
      --async \
      --kill-on-bad-exit=1 \
      ${SRUN_ARGS} \
      python -u train/train_sonoclip.py --lr 1e-4 \
    --para_gamma 0.01 \
    --weight_decay 2e-2 \
    --warmup_length 800 \
    --log_scale 4.6052 \
    --lora_rank -1 \
    --common_pair 0.1 \
    ${RESUME_ARGS} \
    --amp \
    --epoch_num 10 \
    --subnum 1e7 \
    --ul_data_root "${UL_DATA_ROOT}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --exp_name "${EXP_NAME}"
else
  echo "srun not found, using torchrun (local multi-GPU)"
  NPROC=${NPROC:-${GPUS}}
  torchrun --nproc_per_node="$NPROC" train/train_sonoclip.py --lr 1e-4 \
    --para_gamma 0.01 \
    --weight_decay 2e-2 \
    --warmup_length 800 \
    --log_scale 4.6052 \
    --lora_rank -1 \
    --common_pair 0.1 \
    ${RESUME_ARGS} \
    --amp \
    --epoch_num 10 \
    --subnum 1e7 \
    --ul_data_root "${UL_DATA_ROOT}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --exp_name "${EXP_NAME}"
fi
