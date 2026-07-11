#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "$INFLUENCE_OUT"

python3 -m influence_eval.compute_gradient_scores \
    --model_name             "${INFLUENCE_MODEL}" \
    --save_dir               "${INFLUENCE_OUT}" \
    --out_name               "ground_truth" \
    --tokenized_train_path   "${INFLUENCE_OUT}/data/eval_pool" \
    --tokenized_anchor_path  "${INFLUENCE_OUT}/data/eval_anchors" \
    --proj_dim               "${GT_PROJ_DIM}" \
    --lora_target_modules    "${LORA_TARGET_MODULES}" \
    --lora_rank              "${LORA_RANK}" \
    --lora_alpha             "${LORA_ALPHA}" \
    --lora_dropout           "${LORA_DROPOUT}" \
    --lora_seed              "${LORA_SEED}" \
    --project_interval       "${PROJECT_INTERVAL}"
