#!/bin/bash
set -euo pipefail
# Full InfluCoder pipeline: prepare data -> ground truth -> stock gradients ->
# train influence-encoder -> score -> compare against ground truth (Spearman).
#
# This is a trimmed-down copy of tis-ie's runs/influence_spearman/run_all.sh
# that only runs the ground-truth comparator + InfluCoder itself (the other
# baselines -- LESS, LoGRA, IProX, embeddings, TF-IDF, RDS+ -- live in the
# original repo and aren't reproduced here).
#
# Usage:
#   bash runs/influence_spearman/run_all.sh                                    # full config
#   bash runs/influence_spearman/run_all.sh runs/influence_spearman/config_sanity.sh
#   bash runs/influence_spearman/run_all.sh runs/influence_spearman/config_tiny_repro.sh
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

echo "=========================================="
echo "InfluCoder pipeline"
echo "  INFLUENCE_MODEL = ${INFLUENCE_MODEL}"
echo "  ENCODER_MODEL   = ${ENCODER_MODEL_68M}"
echo "  BENCHMARK       = ${BENCHMARK}"
echo "  NUM_ANCHORS     = ${NUM_ANCHORS}"
echo "  END_INDEX       = ${END_INDEX}"
echo "  GT_PROJ_DIM     = ${GT_PROJ_DIM}"
echo "  INFLUENCE_OUT   = ${INFLUENCE_OUT}"
echo "=========================================="

mkdir -p "$INFLUENCE_OUT"

bash runs/influence_spearman/prepare_data.sh "$CFG"                   || { echo "Data preparation failed"; exit 1; }
bash runs/influence_spearman/compute_ground_truth.sh "$CFG"           || { echo "Ground truth failed"; exit 1; }
bash runs/influence_spearman/stock_influcoder_gradients.sh "$CFG"     || { echo "Gradient stocking failed"; exit 1; }
bash runs/influence_spearman/train_influcoder_encoder_68m.sh "$CFG"   || { echo "Encoder training failed"; exit 1; }
bash runs/influence_spearman/compute_influcoder_68m_scores.sh "$CFG"  || { echo "Scoring failed"; exit 1; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FINAL RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -m influence_eval.run_experiment \
    --out_dir  "${INFLUENCE_OUT}" \
    --methods  influcoder_68m \
    --gt_name  ground_truth \
    --seq_len  "${FLOPS_SEQ_LEN}"

echo ""
echo "Done. Results in: ${INFLUENCE_OUT}/results.json"
