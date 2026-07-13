#!/usr/bin/env python3
"""Baseline-vs-ground-truth comparison: LESS, LoGRA (raw+FIM), RDS+, TF-IDF,
and semantic embedding, scored against this repo's own gradient ground truth
(influcoder.gradients.GradientFeaturizer) on the same eval_anchors x
eval_pool split run.py uses -- no encoder training happens here, so there is
no train split.

LESS and semantic/embedding are thin wrappers around code this repo already
has (see influcoder/baselines.py's module docstring for why); each still
loads its own model independently, matching how tis-ie's separate
compute_*.py scripts each load their own model -- so the per-method timing
is directly comparable, at the cost of loading the same base model more than
once per run. LoGRA, RDS+, and TF-IDF are genuinely new, ported from the
tis-ie legacy pipeline.

Usage:
    python run_baselines.py --preset sanity
    python run_baselines.py --n_anchors 15 --n_pool 40 --methods less tfidf
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

from influcoder.baselines import (embedding_scores, less_scores, logra_scores,
                                  rdsplus_scores, tfidf_scores)
from influcoder.data import disjoint_splits, load_bbh, load_dolly
from influcoder.gradients import GradientFeaturizer, hardware_profile
from influcoder.metrics import spearman_metrics

# Eval-side sizes only. Matches run.py's "sanity"/"tiny" eval matrices
# (n_eval_a x n_eval_p) for direct comparability against those runs.
PRESETS = {
    "sanity": dict(n_anchors=6,  n_pool=12, proj_dim=2048, grad_max_len=512),
    "tiny":   dict(n_anchors=15, n_pool=40, proj_dim=8192, grad_max_len=1024),
}

ALL_METHODS = ["less", "logra", "rdsplus", "tfidf", "embedding"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="sanity")
    ap.add_argument("--n_anchors", type=int, default=None, help="override preset")
    ap.add_argument("--n_pool", type=int, default=None, help="override preset")
    ap.add_argument("--grad_model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--logra_rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="LoGRA / RDS+ batch size (tis-ie's own default: 1)")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--out_dir", default=None, help="default: runs_out/baselines_<preset>")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    n_anchors = args.n_anchors or cfg["n_anchors"]
    n_pool = args.n_pool or cfg["n_pool"]
    out_dir = Path(args.out_dir or f"runs_out/baselines_{args.preset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    dtype, attn = hardware_profile()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"hardware: {gpu} -> {dtype}, attn={attn}")

    splits = disjoint_splits(
        load_bbh("data/eval/bbh", seed=42),
        load_dolly("dolly/dolly_data.jsonl", seed=42),
        n_anchors, 0, n_pool, 0,
    )
    anchors, pool = splits["eval_anchors"], splits["eval_pool"]
    print(f"data: eval {n_anchors}x{n_pool}")

    # -- ground truth ---------------------------------------------------------
    t0 = time.time()
    feat = GradientFeaturizer(args.grad_model, lora_rank=args.lora_rank,
                              lora_seed=args.seed, proj_dim=cfg["proj_dim"],
                              max_len=cfg["grad_max_len"])
    g_a, _ = feat.features(anchors, "GT anchors")
    g_p, _ = feat.features(pool, "GT pool")
    feat.close()
    gt = (g_a @ g_p.T).numpy()
    print(f"ground truth: {time.time() - t0:.1f}s")

    results = {
        "preset": args.preset,
        "hardware": {"gpu": gpu, "dtype": str(dtype), "attn": attn},
        "config": {"n_anchors": n_anchors, "n_pool": n_pool, **cfg,
                   "grad_model": args.grad_model, "encoder_model": args.encoder_model,
                   "lora_rank": args.lora_rank, "logra_rank": args.logra_rank,
                   "seed": args.seed, "batch_size": args.batch_size},
        "methods": {},
    }

    def score(name, matrix, elapsed):
        m = spearman_metrics(matrix, gt)
        results["methods"][name] = {**m, "elapsed_s": round(elapsed, 2)}
        print(f"  {name:16s} per-anchor rho={m['per_anchor_mean']:+.4f}  "
              f"agg rho={m['aggregated']:+.4f}  ({elapsed:.1f}s)")

    if "less" in args.methods:
        t0 = time.time()
        pred = less_scores(anchors, pool, args.grad_model, lora_rank=args.lora_rank,
                           lora_seed=args.seed, max_len=cfg["grad_max_len"])
        score("less", pred, time.time() - t0)

    if "logra" in args.methods:
        t0 = time.time()
        raw, fim = logra_scores(anchors, pool, args.grad_model, rank=args.logra_rank,
                                batch_size=args.batch_size, max_len=cfg["grad_max_len"])
        elapsed = time.time() - t0
        score("logra_raw", raw, elapsed)
        score("logra_fim", fim, elapsed)

    if "rdsplus" in args.methods:
        t0 = time.time()
        pred = rdsplus_scores(anchors, pool, args.grad_model,
                              batch_size=args.batch_size, max_len=cfg["grad_max_len"])
        score("rdsplus", pred, time.time() - t0)

    if "tfidf" in args.methods:
        t0 = time.time()
        pred = tfidf_scores(anchors, pool)
        score("tfidf", pred, time.time() - t0)

    if "embedding" in args.methods:
        t0 = time.time()
        pred = embedding_scores(anchors, pool, encoder_model=args.encoder_model)
        score("embedding", pred, time.time() - t0)

    print("\n=== results ===")
    print(f"{'method':18s}{'per-anchor rho':>16s}{'agg rho':>10s}")
    for name, m in results["methods"].items():
        print(f"{name:18s}{m['per_anchor_mean']:>+16.4f}{m['aggregated']:>+10.4f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
