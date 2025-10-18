#!/bin/bash

# Simple Math Agent Training Script
# This script trains a simple math problem solving agent using AgentLightning

set -e

# Configuration
export N_GPUS=1
export BASE_MODEL=/root/autodl-tmp/llm/Qwen3-0.6B
export EXPERIMENT_NAME=simple_math_agent
export PROJECT_NAME=AgentLightningSimpleMath
export DATA_SIZE=100

# 设置Transformers为离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Create data directory and generate training data
echo "Generating training data..."
python generate_train_data.py

# Set data file paths
export TRAIN_DATA_FILE=~/data/rlhf/gsm8k/train.parquet
export VAL_DATA_FILE=~/data/rlhf/gsm8k/val.parquet

echo "Starting simple math agent training..."


# Run the training with local_files_only parameter
python -m agentlightning.verl \
    +ray_init.num_cpus=1 \
    algorithm.adv_estimator=ppo \
    data.train_batch_size=32 \
    data.train_files=${TRAIN_DATA_FILE} \
    data.val_files=${VAL_DATA_FILE} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.multi_turn.format=simple \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    +actor_rollout_ref.model.local_files_only=True \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.truncation='error' \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.rollout.name=openai \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    data.data_size=${DATA_SIZE} \
    "$@"

echo "Training completed!"