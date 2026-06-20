#!/usr/bin/env bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -x

PARTITION=llm3
JOB_NAME=sonoclip_seg_new_2
EXP_NAME=log/sonoclip_seg_new_2

# 数据地址集中在这里改
# DATA_ROOT 下应是 class_name/images 和 class_name/masks
# train/test txt 每行格式：stem<TAB>label_id<TAB>class_name
DATA_ROOT=../ul_data/FetalP5_cls_plane
TRAIN_TXT=../ul_data/FetalP5_cls_plane/train.txt
TEST_TXT=../ul_data/FetalP5_cls_plane/test.txt

# 模型初始化地址
BASE_MODEL_PATH=../checkpoints/ViT-L-14-336px.pt
VISION_CKPT=../checkpoints/sonoclip_vision.pth

# 默认重新训练；改成 yes 会从 EXP_NAME/ckpt 里最新的 epoch_*.pth 续训
RESUME=none

GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=6
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# 固定训练参数
LR=1e-3
WEIGHT_DECAY=1e-2
WARMUP_LENGTH=200
EPOCH_NUM=10
BATCH_SIZE=32
NUM_WORKERS_TRAIN=8
NUM_WORKERS_TEST=8
BCE_WEIGHT=0.5
SAVE_THRESHOLD=0.5
DECODER_MID_CHANNELS=512

if [ "${RESUME}" = "none" ]; then
  RESUME_ARGS=""
else
  RESUME_ARGS="--resume"
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
      python -u train/train_sonoclip_seg.py \
        --data_root "${DATA_ROOT}" \
        --train_txt "${TRAIN_TXT}" \
        --test_txt "${TEST_TXT}" \
        --base_model_path "${BASE_MODEL_PATH}" \
        --vision_ckpt "${VISION_CKPT}" \
        --exp_name "${EXP_NAME}" \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_length ${WARMUP_LENGTH} \
        --epoch_num ${EPOCH_NUM} \
        --batch_size ${BATCH_SIZE} \
        --num_workers_train ${NUM_WORKERS_TRAIN} \
        --num_workers_test ${NUM_WORKERS_TEST} \
        --bce_weight ${BCE_WEIGHT} \
        --save_threshold ${SAVE_THRESHOLD} \
        --decoder_mid_channels ${DECODER_MID_CHANNELS} \
        --amp \
        ${RESUME_ARGS}
else
  python -u train/train_sonoclip_seg.py \
    --data_root "${DATA_ROOT}" \
    --train_txt "${TRAIN_TXT}" \
    --test_txt "${TEST_TXT}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --vision_ckpt "${VISION_CKPT}" \
    --exp_name "${EXP_NAME}" \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_length ${WARMUP_LENGTH} \
    --epoch_num ${EPOCH_NUM} \
    --batch_size ${BATCH_SIZE} \
    --num_workers_train ${NUM_WORKERS_TRAIN} \
    --num_workers_test ${NUM_WORKERS_TEST} \
    --bce_weight ${BCE_WEIGHT} \
    --save_threshold ${SAVE_THRESHOLD} \
    --decoder_mid_channels ${DECODER_MID_CHANNELS} \
    --amp \
    ${RESUME_ARGS}
fi
