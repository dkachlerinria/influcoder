#!/bin/bash
# Absolute-minimum config used only to check that every step of the pipeline
# runs end-to-end without error (model loads, LoRA attaches, gradients project,
# encoder trains for one tiny epoch, scores get written, run_experiment reads
# them back). Numbers are too small to mean anything statistically -- this is
# a smoke test, not an experiment. Should finish in a couple of minutes.
#
# Usage:
#   bash runs/influence_spearman/run_all.sh runs/influence_spearman/config_sanity.sh

# Pre-Ampere GPUs (e.g. P100) can't run triton/inductor or flash-attn; the
# ettin/ModernBERT encoder code paths hit both. Force eager everywhere.
export TORCHDYNAMO_DISABLE=1

export INFLUENCE_MODEL="HuggingFaceTB/SmolLM2-135M"
INFLUENCE_MODEL_SLUG=$(echo "${INFLUENCE_MODEL}" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')

export BENCHMARK="bbh"
export TRAIN_DATASET="Harvard-DCML/tulu-v2-197K-processed"

export ENCODER_MODEL="jhu-clsp/ettin-encoder-68m"
export ENCODER_MODEL_150M="jhu-clsp/ettin-encoder-150m"
export ENCODER_MODEL_68M="jhu-clsp/ettin-encoder-68m"

export END_INDEX=6
export NUM_ANCHORS=4

export BATCH_SIZE=1
export GRAD_ACC=8

export IPROX_SPARSITY=0.1
export IPROX_N_TRAIN_P=2
export IPROX_N_TRAIN_A=2

export GT_PROJ_DIM=2048
export LESS_PROJ_DIM=2048

export LORA_SEED=0
export PROJECT_INTERVAL=1

export LORA_RANK=4
export LORA_ALPHA=8
export LORA_DROPOUT=0.1
export LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

export SHUFFLE_SEED=1
export RANDOM_SEED=0
export FLOPS_SEQ_LEN=2048

export INFLUCODER_PROJ_DIM=1024
export INFLUCODER_RUN_MODE="tiny"
export INFLUCODER_N_TRAIN_A=4
export INFLUCODER_N_EVAL_A=2
export INFLUCODER_N_TRAIN_P=6
export INFLUCODER_N_EVAL_P=2

export INFLUENCE_OUT="files/influence_models/${INFLUENCE_MODEL_SLUG}_sanity"

export INFLUCODER_DB_DIR="${INFLUENCE_OUT}/influcoder_db"
export INFLUCODER_ENCODER_DIR_150M="${INFLUENCE_OUT}/influcoder_encoder_150m"
export INFLUCODER_ENCODER_DIR_68M="${INFLUENCE_OUT}/influcoder_encoder_68m"
export INFLUCODER_ENCODER_DIR="${INFLUCODER_ENCODER_DIR_68M}"
