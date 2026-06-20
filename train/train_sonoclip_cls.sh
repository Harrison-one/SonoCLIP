#!/usr/bin/env bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -x

PARTITION=llm3
JOB_NAME=sonoclip_cls
EXP_NAME=log/sonoclip_cls
# Dataset paths. DATA_ROOT should contain class_name/images and class_name/masks.
# train/test txt format: stem<TAB>label_id<TAB>class_name
DATA_ROOT=../ul_data/FetalP6_cls_plane
TRAIN_TXT=../ul_data/FetalP6_cls_plane/train.txt
TEST_TXT=../ul_data/FetalP6_cls_plane/test.txt

# Model checkpoints, relative to this train/ directory after cd.
BASE_MODEL_PATH=../checkpoints/ViT-L-14-336px.pt
ALPHA_VISION_CKPT=../checkpoints/sonoclip_vision.pth

# Output directory. Checkpoints are saved under ${EXP_NAME}/ckpt.


# Switches.
USE_MASK=1       # 1: use real masks with common_pair mixing; 0: all-one masks only
COMMON_PAIR=0.1  # probability to replace real mask with all-one mask when USE_MASK=1
RESUME=0         # 1: resume from latest ${EXP_NAME}/ckpt/epoch_*.pth
AMP=0            # 1: mixed precision
NO_HI_RES=0      # 1: use 224px input instead of 336px
SUBNUM=none      # set to an integer for a debug subset

# Training hyperparameters.
LR=1e-3
WEIGHT_DECAY=1e-3
WARMUP_LENGTH=200
EPOCH_NUM=20
BATCH_SIZE=32
NUM_WORKERS_TRAIN=8
NUM_WORKERS_TEST=8
DEVICE=cuda

GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=6
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

EXTRA_ARGS=""
if [ "${USE_MASK}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --use_mask"
fi
if [ "${RESUME}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --resume"
fi
if [ "${AMP}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --amp"
fi
if [ "${NO_HI_RES}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --no_hi_res"
fi
if [ "${SUBNUM}" != "none" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --subnum ${SUBNUM}"
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
      python -u train/train_sonoclip_cls.py \
        --data_root "${DATA_ROOT}" \
        --train_txt "${TRAIN_TXT}" \
        --test_txt "${TEST_TXT}" \
        --base_model_path "${BASE_MODEL_PATH}" \
        --alpha_vision_ckpt "${ALPHA_VISION_CKPT}" \
        --exp_name "${EXP_NAME}" \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_length ${WARMUP_LENGTH} \
        --epoch_num ${EPOCH_NUM} \
        --batch_size ${BATCH_SIZE} \
        --common_pair ${COMMON_PAIR} \
        --num_workers_train ${NUM_WORKERS_TRAIN} \
        --num_workers_test ${NUM_WORKERS_TEST} \
        --device "${DEVICE}" \
        ${EXTRA_ARGS}
else
  python -u train/train_sonoclip_cls.py \
    --data_root "${DATA_ROOT}" \
    --train_txt "${TRAIN_TXT}" \
    --test_txt "${TEST_TXT}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --alpha_vision_ckpt "${ALPHA_VISION_CKPT}" \
    --exp_name "${EXP_NAME}" \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_length ${WARMUP_LENGTH} \
    --epoch_num ${EPOCH_NUM} \
    --batch_size ${BATCH_SIZE} \
    --common_pair ${COMMON_PAIR} \
    --num_workers_train ${NUM_WORKERS_TRAIN} \
    --num_workers_test ${NUM_WORKERS_TEST} \
    --device "${DEVICE}" \
    ${EXTRA_ARGS}
fi
