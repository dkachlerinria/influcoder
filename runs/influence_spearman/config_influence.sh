#!/bin/bash
# Shared config for the influence-Spearman experiment.
# Sources runs/config.sh for TRAINING_MODEL, paths, LoRA defaults, etc.
# then layers experiment-specific knobs on top.

# Model for influence calculations
export INFLUENCE_MODEL="HuggingFaceTB/SmolLM2-1.7B"
#export INFLUENCE_MODEL="jhu-clsp/ettin-decoder-68m"
INFLUENCE_MODEL_SLUG=$(echo "${INFLUENCE_MODEL}" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')

# Proxy Models (progressively smaller approximations of the main model)
export PROXY_MODEL_1="HuggingFaceTB/SmolLM2-360M"
export PROXY_MODEL_2="HuggingFaceTB/SmolLM2-135M"

# Base Dataset and Task Config
export BENCHMARK="bbh"
export TRAIN_DATASET="Harvard-DCML/tulu-v2-197K-processed"
export ENCODER_MODEL="jhu-clsp/ettin-encoder-150m"
export ENCODER_MODEL_150M="jhu-clsp/ettin-encoder-150m"
export ENCODER_MODEL_68M="jhu-clsp/ettin-encoder-68m"

# Evaluation sizes
export END_INDEX=10000
export NUM_ANCHORS=100

# Training defaults for proxies
export BATCH_SIZE=1
export GRAD_ACC=8

# IProX sparsity — MUST match between train_iprox_proxy.sh and compute_iprox_scores.sh,
# otherwise the LinearSVD layer structure (which layers get replaced + their ranks)
# diverges between training and scoring → garbage gradients during scoring.
export IPROX_SPARSITY=0.1
export IPROX_N_TRAIN_P=1600   # dolly pool samples for IProX training
export IPROX_N_TRAIN_A=500    # BBH anchor samples for IProX training

# Projection dimensions
export GT_PROJ_DIM=65536
export LESS_PROJ_DIM=8192

# Fresh-LoRA seed (must be identical across GT and LESS for apples-to-apples)
export LORA_SEED=0

# Gradient accumulation before projection (tune down to save memory)
export PROJECT_INTERVAL=1

# Smaller LoRA for A30 memory
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.1
export LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# LoGRA settings (rank=8 matches paper default)
export LOGRA_RANK=8
export LOGRA_BATCH_SIZE=2          # main model (1.7B)
export LOGRA_BATCH_SIZE_PROXY1=4   # proxy1 (360M)
export LOGRA_BATCH_SIZE_PROXY2=8   # proxy2 (135M)
# Target all-linear layers (like LESS) for fair comparison, not just MLP
export LOGRA_ALL_LINEAR=1

# Cheaper Pareto-frontier variants
export LESS_SMALL_LORA_RANK=8    # was LORA_RANK=16; halves P_lora → ~2% FLOPs reduction
export LOGRA_SMALL_RANK=4        # was LOGRA_RANK=8; FIM inversion 64× cheaper

# Data shuffle seed (controls dolly/BBH split randomization in prepare_data.py)
export SHUFFLE_SEED=1

# After, 137, 0

# Random baseline seed
export RANDOM_SEED=0

# Sequence length used for analytic FLOPS accounting (training max_seq_length)
export FLOPS_SEQ_LEN=2048

# Influcoder settings
export INFLUCODER_PROJ_DIM=32768
export INFLUCODER_RUN_MODE="small"
export INFLUCODER_N_TRAIN_A=500         # BBH anchors for encoder training (start at NUM_ANCHORS)
export INFLUCODER_N_EVAL_A=100          # BBH anchors for encoder eval (start after train_anchors)
export INFLUCODER_N_TRAIN_P=1000        # dolly pool for encoder training (start at END_INDEX)
export INFLUCODER_N_EVAL_P=200          # dolly pool for encoder eval (start at END_INDEX+N_TRAIN_P)

# Output directory (model-scoped, completely separate from SFT paths)
export INFLUENCE_OUT="files/influence_models/${INFLUENCE_MODEL_SLUG}"

export INFLUCODER_DB_DIR="${INFLUENCE_OUT}/influcoder_db"
export INFLUCODER_ENCODER_DIR_150M="${INFLUENCE_OUT}/influcoder_encoder_150m"
export INFLUCODER_ENCODER_DIR_68M="${INFLUENCE_OUT}/influcoder_encoder_68m"
export INFLUCODER_ENCODER_DIR="${INFLUCODER_ENCODER_DIR_150M}"   # default alias
