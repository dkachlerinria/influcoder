# InfluCoder

InfluCoder distills LoRA-gradient influence scores (LESS/TRAK-style) into a
cheap sentence-encoder: a small bi-encoder is trained to reproduce the ranking
that expensive per-sample gradient influence induces, so that scoring a
candidate at selection time is a single forward pass instead of a
forward+backward pass through the full target model.

This repo is a trimmed extraction of the `runs/influence_spearman/` pipeline
from a larger internal rebuttal repo (itself forked from the "Targeted
Instruction Selection" codebase). Only the pieces needed to (a) compute a
LoRA-gradient ground-truth influence matrix and (b) train + score InfluCoder
against it are included here — the other baselines that lived alongside it
(LESS, LoGRA, IProX, embeddings, TF-IDF, RDS+) are not reproduced in this repo.

## How it works

1. **`prepare_data.sh`** slices BBH (anchors/queries, `data/eval/bbh/`) and
   Dolly (candidate pool, `dolly/dolly_data.jsonl`) into disjoint index ranges
   shared by every downstream step, and tokenizes them.
2. **`compute_ground_truth.sh`** attaches a fresh (untrained) seeded LoRA
   adapter to the target decoder, computes per-sample SGD gradients on the
   LoRA params, projects them (TRAK Rademacher projector) to `GT_PROJ_DIM`,
   and takes cosine similarity between anchor and pool gradients → the
   ground-truth (num_anchors × num_train) influence matrix.
3. **`stock_influcoder_gradients.sh`** recomputes the *same* fresh-LoRA
   gradients (same model/seed/formatting) for a separate encoder-training
   split, but projects them with a cheaper sparse CountSketch projector to
   `INFLUCODER_PROJ_DIM`.
4. **`train_influcoder_encoder_68m.sh`** trains a `SentenceTransformer`
   bi-encoder (`jhu-clsp/ettin-encoder-68m` by default) with a listwise
   contrastive loss (Pearson + KL/MSE) to reproduce the CountSketch gradient
   similarities.
5. **`compute_influcoder_68m_scores.sh`** embeds the held-out eval anchors/pool
   with the trained encoder (forward pass only) → InfluCoder's score matrix.
6. **`run_experiment.py`** compares InfluCoder's score matrix against ground
   truth via Spearman correlation (per-anchor and aggregated) and reports
   FLOPs/timing/storage cost.

`runs/influence_spearman/run_all.sh` runs steps 1–6 end to end.

## Setup

```bash
conda create -n influcoder python=3.12 -c conda-forge
conda activate influcoder
pip install -r requirements.txt
```

BBH eval data (`data/eval/bbh/`) and Dolly (`dolly/dolly_data.jsonl`) are
already checked into this repo (small, ~3MB and ~14MB respectively) so no
download step is required. If you need to refresh them, see
`download_eval.sh`.

## Running

```bash
# 1. Sanity check -- do all the steps run at all (~1-2 min)?
bash runs/influence_spearman/run_all.sh runs/influence_spearman/config_sanity.sh

# 2. Tiny end-to-end reproduction -- small but non-degenerate sizes (~10-15 min)
bash runs/influence_spearman/run_all.sh runs/influence_spearman/config_tiny_repro.sh
```

Both configs use `HuggingFaceTB/SmolLM2-135M` as the gradient source and
`jhu-clsp/ettin-encoder-68m` as the encoder to keep runtime small on a single
GPU. Neither config is meant to reproduce paper-scale numbers — see
`runs/influence_spearman/config_influence.sh` for the full-scale reference
configuration (SmolLM2-1.7B decoder, `GT_PROJ_DIM=65536`, thousands of
anchors/pool samples) that the tiny configs are cut down from.

Results land in `${INFLUENCE_OUT}/results.json` (path printed at the end of
the run), with a markdown Spearman/FLOPs table printed to stdout.

## Known deviations from the original pipeline

See `KNOWN_ISSUES.txt` and `DEVIATIONS.md`.
