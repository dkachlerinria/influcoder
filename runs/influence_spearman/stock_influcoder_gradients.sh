#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

SCRIPT_START=$SECONDS

DATA_DIR="${INFLUENCE_OUT}/data"
if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ ${DATA_DIR} not found. Run runs/influence_spearman/prepare_data.sh first."
    exit 1
fi

rm -rf "${INFLUCODER_DB_DIR}"
mkdir -p "${INFLUCODER_DB_DIR}"

# LoRA/projection args shared across splits — match compute_gradient_scores.py exactly
COMMON_ARGS=(
    --model_name           "${INFLUENCE_MODEL}"
    --proj_dim             "${INFLUCODER_PROJ_DIM}"
    --proj_seed            42
    --project_interval     "${PROJECT_INTERVAL}"
    --lora_target_modules  "${LORA_TARGET_MODULES}"
    --lora_rank            "${LORA_RANK}"
    --lora_alpha           "${LORA_ALPHA}"
    --lora_dropout         "${LORA_DROPOUT}"
    --lora_seed            "${LORA_SEED}"
    --output_dir           "${INFLUCODER_DB_DIR}"
)

# train_anchors: BBH from prepared influcoder_train_anchors
echo "Stocking train_anchors from data/influcoder_train_anchors/ ..."
python influcoder/gradient_stocking_EXACT.py \
    "${COMMON_ARGS[@]}" \
    --split                 train_anchors \
    --tokenized_input_path  "${DATA_DIR}/influcoder_train_anchors" \
    --inputs_json_path      "${DATA_DIR}/influcoder_train_anchors_inputs.json" \
    --output_name           train_anchors

# eval_anchors
echo "Stocking eval_anchors from data/influcoder_eval_anchors/ ..."
python influcoder/gradient_stocking_EXACT.py \
    "${COMMON_ARGS[@]}" \
    --split                 eval_anchors \
    --tokenized_input_path  "${DATA_DIR}/influcoder_eval_anchors" \
    --inputs_json_path      "${DATA_DIR}/influcoder_eval_anchors_inputs.json" \
    --output_name           eval_anchors

# train_pool
echo "Stocking pool from data/influcoder_train_pool/ ..."
python influcoder/gradient_stocking_EXACT.py \
    "${COMMON_ARGS[@]}" \
    --split                 pool \
    --tokenized_input_path  "${DATA_DIR}/influcoder_train_pool" \
    --inputs_json_path      "${DATA_DIR}/influcoder_train_pool_inputs.json" \
    --output_name           pool

# eval_pool
echo "Stocking eval_pool from data/influcoder_eval_pool/ ..."
python influcoder/gradient_stocking_EXACT.py \
    "${COMMON_ARGS[@]}" \
    --split                 eval_pool \
    --tokenized_input_path  "${DATA_DIR}/influcoder_eval_pool" \
    --inputs_json_path      "${DATA_DIR}/influcoder_eval_pool_inputs.json" \
    --output_name           eval_pool

echo "Gradient stocking complete. Files in: ${INFLUCODER_DB_DIR}"
ls -la "${INFLUCODER_DB_DIR}"
echo "⏱  Gradient stocking total time: $((SECONDS - SCRIPT_START))s"
