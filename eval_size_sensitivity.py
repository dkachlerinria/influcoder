#!/usr/bin/env python3
"""Does eval-set size affect InfluCoder's measured Spearman? A reviewer
flagged the original paper's 200x200 eval set as possibly too small/noisy;
this tests 100x100 vs 400x400 vs 800x800.

Trains ONE encoder ONCE (the "big" recipe: 500x1000 train, epochs=8,
hard_ratio=0, lr=5e-5, alpha=0.5), on data disjoint from an 800x800 eval
set. Ground truth (both this repo's own CountSketch and tis-ie's actual/
legacy GT) and the encoder's predicted scores are each computed ONCE at the
full 800x800 size; the 100x100 and 400x400 numbers are then read off as
NESTED SUBMATRICES of those same 800x800 score matrices, not separate
draws or separate training runs.

This is deliberate and exact, not an approximation: every score
gt[i,j] / pred[i,j] is a pairwise cosine between anchor i's and pool j's
own gradient/embedding, computed independently of how many other anchors
or pool candidates exist. Slicing the top-left NxN block of an 800x800
score matrix gives EXACTLY the same numbers a fresh NxN-only run would
produce -- so growing N here isolates the pure sample-size effect on the
Spearman estimate, with zero confound from re-training, re-sampling
anchors, or (for legacy GT) a different stochastic dropout draw.

Only InfluCoder is evaluated here (not the other baselines) -- this is a
statistical-power check on the eval set size itself, not another
cross-method comparison.

Usage:
    python eval_size_sensitivity.py
"""

import os

import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

import argparse
import json
import time
from pathlib import Path

from influcoder.data import disjoint_splits, load_bbh, load_dolly
from influcoder.encoder import distill, embed, load_encoder
from influcoder.gradients import GradientFeaturizer
from influcoder.legacy_gt import legacy_gt_scores
from influcoder.metrics import spearman_metrics

# The "big" preset recipe -- unchanged from run.py / sample_efficiency_sweep.py.
RECIPE = dict(epochs=8, lr=5e-5, hard_ratio=0.0, k_anchors=8, m_candidates=16, alpha=0.5)
N_TRAIN_A, N_TRAIN_P = 500, 1000
MAX_EVAL = 800          # largest eval size tested; 100/400 are prefixes of this
EVAL_SIZES = [100, 400, 800]
GRAD_MODEL = "HuggingFaceTB/SmolLM2-135M"
PROJ_DIM = 8192
GRAD_MAX_LEN = 1024
LORA_RANK = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs_out/eval_size_sensitivity")
    ap.add_argument("--legacy_proj_dim", type=int, default=65536)
    ap.add_argument("--legacy_lora_dropout", type=float, default=0.1)
    ap.add_argument("--skip_legacy", action="store_true",
                    help="CountSketch-only run (faster; skips the ~10min legacy-GT pass)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # -- data: eval front slice (MAX_EVAL), train the next slice (disjoint) --
    splits = disjoint_splits(
        load_bbh("data/eval/bbh", seed=42), load_dolly("dolly/dolly_data.jsonl", seed=42),
        MAX_EVAL, N_TRAIN_A, MAX_EVAL, N_TRAIN_P,
    )
    eval_anchors, eval_pool = splits["eval_anchors"], splits["eval_pool"]
    train_anchors, train_pool = splits["train_anchors"], splits["train_pool"]
    print(f"eval: {MAX_EVAL}x{MAX_EVAL} (100/400 read as prefixes of this)  "
          f"train: {N_TRAIN_A}x{N_TRAIN_P} (disjoint, starts right after eval)")

    # -- CountSketch GT + training targets, computed once at the full sizes --
    cs_cache = out_dir / "countsketch_gt_and_targets.pt"
    if cs_cache.exists():
        print(f"loading cached CountSketch GT + targets from {cs_cache}")
        cs = torch.load(cs_cache, weights_only=False)
        countsketch_gt_800, targets = cs["countsketch_gt_800"], cs["targets"]
    else:
        t0 = time.time()
        feat = GradientFeaturizer(GRAD_MODEL, lora_rank=LORA_RANK, lora_seed=args.seed,
                                  proj_dim=PROJ_DIM, max_len=GRAD_MAX_LEN)
        g_eval_a, _ = feat.features(eval_anchors, "CountSketch eval anchors (800)")
        g_eval_p, _ = feat.features(eval_pool, "CountSketch eval pool (800)")
        g_train_a, _ = feat.features(train_anchors, "CountSketch train anchors")
        g_train_p, _ = feat.features(train_pool, "CountSketch train pool")
        feat.close()
        countsketch_gt_800 = (g_eval_a @ g_eval_p.T).numpy()
        targets = g_train_a @ g_train_p.T
        print(f"CountSketch GT + targets: {time.time() - t0:.1f}s")
        torch.save({"countsketch_gt_800": countsketch_gt_800, "targets": targets}, cs_cache)

    # -- legacy (tis-ie actual) GT, computed once at the full 800x800 --------
    legacy_gt_800 = None
    legacy_cache = out_dir / "legacy_gt_800x800.pt"
    if not args.skip_legacy:
        if legacy_cache.exists():
            print(f"loading cached legacy GT from {legacy_cache}")
            legacy_gt_800 = torch.load(legacy_cache, weights_only=False)
        else:
            print(f"computing legacy (tis-ie) ground truth at {MAX_EVAL}x{MAX_EVAL} "
                  f"(proj_dim={args.legacy_proj_dim}, lora_dropout={args.legacy_lora_dropout})...")
            t0 = time.time()
            legacy_gt_800 = legacy_gt_scores(eval_anchors, eval_pool, GRAD_MODEL,
                                             proj_dim=args.legacy_proj_dim,
                                             lora_dropout=args.legacy_lora_dropout,
                                             lora_seed=args.seed, max_len=GRAD_MAX_LEN)
            print(f"legacy GT: {time.time() - t0:.1f}s")
            torch.save(legacy_gt_800, legacy_cache)

    # -- encoder: one training run, eval_spearman always over the FULL 800x800 --
    enc = load_encoder(args.encoder_model)
    eval_a_texts = [s.text for s in eval_anchors]
    eval_p_texts = [s.text for s in eval_pool]

    def eval_spearman():
        pred = embed(enc, eval_a_texts) @ embed(enc, eval_p_texts).T
        return spearman_metrics(pred, countsketch_gt_800)

    untrained_metrics_800 = eval_spearman()
    print(f"untrained encoder (800x800, CountSketch): "
          f"per-anchor rho={untrained_metrics_800['per_anchor_mean']:+.4f} "
          f"agg rho={untrained_metrics_800['aggregated']:+.4f}")
    pred_untrained_800 = embed(enc, eval_a_texts) @ embed(enc, eval_p_texts).T

    t0 = time.time()
    # NOTE: this worktree's influcoder/encoder.py doesn't expose distill()'s
    # alpha kwarg (that's an uncommitted change elsewhere) -- its hardcoded
    # default (0.5) is numerically identical to RECIPE["alpha"], so omitting
    # it here changes nothing.
    assert RECIPE["alpha"] == 0.5
    train_log = distill(enc, [s.text for s in train_anchors], [s.text for s in train_pool],
                        targets, epochs=RECIPE["epochs"], k_anchors=RECIPE["k_anchors"],
                        m_candidates=RECIPE["m_candidates"], lr=RECIPE["lr"],
                        hard_ratio=RECIPE["hard_ratio"],
                        seed=args.seed, epoch_eval=eval_spearman)
    print(f"encoder training: {time.time() - t0:.1f}s (best epoch {train_log['best_epoch'] + 1})")

    pred_trained_800 = embed(enc, eval_a_texts) @ embed(enc, eval_p_texts).T

    # -- slice all three score matrices at each eval size, compute metrics ---
    results = {"recipe": RECIPE, "n_train_a": N_TRAIN_A, "n_train_p": N_TRAIN_P,
              "eval_sizes": {}}
    print(f"\n{'N (eval size)':16s}{'untrained CS':>14s}{'trained CS':>12s}"
          + ("" if args.skip_legacy else f"{'untrained legacy':>18s}{'trained legacy':>16s}"))
    for n in EVAL_SIZES:
        cs_gt_n = countsketch_gt_800[:n, :n]
        pred_u_n = pred_untrained_800[:n, :n]
        pred_t_n = pred_trained_800[:n, :n]
        m_u_cs = spearman_metrics(pred_u_n, cs_gt_n)
        m_t_cs = spearman_metrics(pred_t_n, cs_gt_n)
        entry = {"untrained_countsketch": m_u_cs, "trained_countsketch": m_t_cs}

        line = f"{f'{n}x{n}':16s}{m_u_cs['aggregated']:>+14.4f}{m_t_cs['aggregated']:>+12.4f}"
        if not args.skip_legacy:
            lg_gt_n = legacy_gt_800[:n, :n]
            m_u_lg = spearman_metrics(pred_u_n, lg_gt_n)
            m_t_lg = spearman_metrics(pred_t_n, lg_gt_n)
            entry["untrained_legacy"] = m_u_lg
            entry["trained_legacy"] = m_t_lg
            line += f"{m_u_lg['aggregated']:>+18.4f}{m_t_lg['aggregated']:>+16.4f}"
        print(line)
        results["eval_sizes"][n] = entry

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
