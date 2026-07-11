#!/bin/bash
set -euo pipefail
# Single source of truth for all data files used by the pipeline.
# Run this FIRST before any other compute_*.sh / train_*.sh script.
#
# Usage:
#   bash runs/influence_spearman/prepare_data.sh
#   bash runs/influence_spearman/prepare_data.sh runs/influence_spearman/config_influence_tiny.sh

CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "${INFLUENCE_OUT}"

echo "🧹 Preparing data under ${INFLUENCE_OUT}/data/ ..."

python3 -m influence_eval.prepare_data \
    --dolly_path             "dolly/dolly_data.jsonl" \
    --gradient_model         "${INFLUENCE_MODEL}" \
    --output_dir             "${INFLUENCE_OUT}" \
    --max_seq_length         "${FLOPS_SEQ_LEN}" \
    --end_index              "${END_INDEX}" \
    --num_anchors            "${NUM_ANCHORS}" \
    --iprox_n_train_p        "${IPROX_N_TRAIN_P:-${NUM_ANCHORS}}" \
    --iprox_n_train_a        "${IPROX_N_TRAIN_A:-${NUM_ANCHORS}}" \
    --influcoder_n_train_p   "${INFLUCODER_N_TRAIN_P}" \
    --influcoder_n_train_a   "${INFLUCODER_N_TRAIN_A}" \
    --influcoder_n_eval_p    "${INFLUCODER_N_EVAL_P}" \
    --influcoder_n_eval_a    "${INFLUCODER_N_EVAL_A}" \
    --shuffle_seed           "${SHUFFLE_SEED:-42}"

echo "✅ Data ready in ${INFLUENCE_OUT}/data/"
