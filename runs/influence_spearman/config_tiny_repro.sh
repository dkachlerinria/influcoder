#!/bin/bash
# Tiny end-to-end smoke config for reproducing InfluCoder (train encoder + Spearman
# eval against ground truth) in well under 15 minutes on a single GPU.
#
# Not meant to produce a meaningful Spearman number (dataset slices are far too
# small for that) -- this is purely a "does the pipeline still work" check.
#
# Usage:
#   bash runs/influence_spearman/run_all.sh runs/influence_spearman/config_tiny_repro.sh

# Small decoder (gradient source) -- already tiny, fast fwd+bwd on a single GPU.
export INFLUENCE_MODEL="HuggingFaceTB/SmolLM2-135M"
INFLUENCE_MODEL_SLUG=$(echo "${INFLUENCE_MODEL}" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')

export BENCHMARK="bbh"
export TRAIN_DATASET="Harvard-DCML/tulu-v2-197K-processed"   # unused locally (dolly/dolly_data.jsonl is read directly)

# Smallest encoder in the family.
export ENCODER_MODEL="jhu-clsp/ettin-encoder-68m"
export ENCODER_MODEL_150M="jhu-clsp/ettin-encoder-150m"
export ENCODER_MODEL_68M="jhu-clsp/ettin-encoder-68m"

# Spearman eval set (ground truth + influcoder scoring matrix shape = NUM_ANCHORS x END_INDEX)
export END_INDEX=40
export NUM_ANCHORS=15

export BATCH_SIZE=1
export GRAD_ACC=8

# Not used (no IProX in this trimmed repo) but prepare_data.py still slices these
# ranges out of dolly/BBH, so keep them small and nonzero.
export IPROX_SPARSITY=0.1
export IPROX_N_TRAIN_P=10
export IPROX_N_TRAIN_A=10

# Ground-truth projection -- lower than the paper default (65536) purely for speed.
export GT_PROJ_DIM=8192
export LESS_PROJ_DIM=8192

export LORA_SEED=0
export PROJECT_INTERVAL=1

# Small LoRA -- fewer trainable params -> faster fwd+bwd and projection.
export LORA_RANK=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.1
export LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

export SHUFFLE_SEED=1
export RANDOM_SEED=0
export FLOPS_SEQ_LEN=2048

# Influcoder settings -- small stocking pools, "quick" encoder run mode (2 epochs).
export INFLUCODER_PROJ_DIM=4096
export INFLUCODER_RUN_MODE="quick"
export INFLUCODER_N_TRAIN_A=30
export INFLUCODER_N_EVAL_A=15
export INFLUCODER_N_TRAIN_P=50
export INFLUCODER_N_EVAL_P=15

export INFLUENCE_OUT="files/influence_models/${INFLUENCE_MODEL_SLUG}_tiny_repro"

export INFLUCODER_DB_DIR="${INFLUENCE_OUT}/influcoder_db"
export INFLUCODER_ENCODER_DIR_150M="${INFLUENCE_OUT}/influcoder_encoder_150m"
export INFLUCODER_ENCODER_DIR_68M="${INFLUENCE_OUT}/influcoder_encoder_68m"
export INFLUCODER_ENCODER_DIR="${INFLUCODER_ENCODER_DIR_68M}"   # default alias -> smallest encoder
