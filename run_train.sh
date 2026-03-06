#!/bin/bash
# ============================================================
# TableMind++ Training Script
# Stage 1: SFT warm-up  →  Stage 2: RFT with RAPO
# ============================================================

# ---- User-configurable variables ---------------------------
export BASE_MODEL=''          # Path to Qwen3-8B base/instruct model
export PROJECT_NAME=''        # W&B / logging project name
export EXPERIMENT_NAME=''     # W&B / logging experiment name
export CSV_FILE_PATH=''       # Path to directory containing CSV files for sandbox

# ---- Hyperparameters (paper settings) ----------------------
max_prompt_length=$((1024 * 12))     # 12 288 tokens
max_response_length=$((1024 * 7))    # 7 168 tokens

clip_ratio_low=0.2
clip_ratio_high=0.28

# ============================================================
# Stage 1: Supervised Fine-Tuning (SFT) warm-up
# ============================================================
# SFT is performed externally on 200 synthetic distilled samples for 1 epoch.
# Example using standard HuggingFace Trainer or LLaMA-Factory:
#
#   python -m llmtuner.train \
#       --model_name_or_path $BASE_MODEL \
#       --dataset sft_trajectories \
#       --num_train_epochs 1 \
#       --learning_rate 1e-6 \
#       --output_dir checkpoints/sft
#
# After SFT, set BASE_MODEL to the SFT checkpoint path before running Stage 2.
# ============================================================

echo "Starting Stage 2: Reinforcement Fine-Tuning (RAPO)..."

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=['data/train.parquet'] \
    data.val_files=['data/test.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.max_response_length_single_turn=2048 \
    data.use_default_tool_template=False \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.stop_token_ids=[] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n_repeat=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    tool.max_turns=3 \
    tool.tools=['python'] \
    tool.env=nous \
    tool.max_tool_response_length=128 $@
