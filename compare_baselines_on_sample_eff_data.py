#!/usr/bin/env python3
"""Score LESS, LoGRA (raw+FIM), RDS+, TF-IDF, and semantic-embedding
baselines on EXACTLY the same 100x100 eval set -- and against the same two
ground truths (this repo's own CountSketch GT, and tis-ie's actual/legacy
GT) -- as sample_efficiency_sweep.py / sample_efficiency_sweep_legacy_gt.py.
Lets the InfluCoder-trained-encoder numbers reported there be sanity-checked
against a wider set of methods on identical data, the same comparison
already done against tis-ie's own ground truth in the influcoder-baselines
worktree, but now against InfluCoder's own sample-efficiency reference point.

Reuses cached artifacts rather than recomputing: grad_cache.pt (CountSketch
GT + eval texts) and legacy_gt_100x100.pt (legacy GT) must already exist --
run sample_efficiency_sweep_legacy_gt.py first if not.

Usage:
    python compare_baselines_on_sample_eff_data.py
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
from influcoder.metrics import spearman_metrics
from sample_efficiency_sweep import GRAD_MAX_LEN, GRAD_MODEL, N_EVAL_A, N_EVAL_P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="runs_out/sample_eff")
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="LoGRA / RDS+ batch size (tis-ie's own default: 1)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    grad_cache_path = out_dir / "grad_cache.pt"
    legacy_gt_path = out_dir / "legacy_gt_100x100.pt"
    if not grad_cache_path.exists() or not legacy_gt_path.exists():
        raise FileNotFoundError(
            f"Need {grad_cache_path} and {legacy_gt_path} -- run "
            f"sample_efficiency_sweep_legacy_gt.py first."
        )

    print(f"loading cached CountSketch GT + eval texts from {grad_cache_path}")
    grad_cache = torch.load(grad_cache_path, weights_only=False)
    countsketch_gt = (grad_cache["g_eval_a"] @ grad_cache["g_eval_p"].T).numpy()

    print(f"loading cached legacy GT from {legacy_gt_path}")
    legacy_gt = torch.load(legacy_gt_path, weights_only=False)

    # Reconstruct the SAME eval Samples: deterministic seed=42 load, front
    # slice by N_EVAL_A/N_EVAL_P -- identical regardless of train size, since
    # disjoint_splits always takes eval as a front slice. Verified against
    # the sweep's own cached texts rather than assumed.
    splits = disjoint_splits(
        load_bbh("data/eval/bbh", seed=42), load_dolly("dolly/dolly_data.jsonl", seed=42),
        N_EVAL_A, 0, N_EVAL_P, 0,
    )
    eval_anchors, eval_pool = splits["eval_anchors"], splits["eval_pool"]
    assert [s.text for s in eval_anchors] == grad_cache["eval_a_texts"], \
        "reconstructed eval anchors don't match the sweep's cache"
    assert [s.text for s in eval_pool] == grad_cache["eval_p_texts"], \
        "reconstructed eval pool doesn't match the sweep's cache"
    print(f"eval split: {N_EVAL_A}x{N_EVAL_P} (confirmed identical to the sweep's own cache)")

    gts = {"countsketch": countsketch_gt, "legacy": legacy_gt}
    results = {"methods": {}}

    def score(name, matrix, elapsed):
        m = {gt_name: spearman_metrics(matrix, gt) for gt_name, gt in gts.items()}
        results["methods"][name] = {**m, "elapsed_s": round(elapsed, 2)}
        cs, lg = m["countsketch"]["aggregated"], m["legacy"]["aggregated"]
        print(f"  {name:16s} agg rho: CS={cs:+.4f}  legacy={lg:+.4f}  ({elapsed:.1f}s)")

    print(f"\n{'method':18s}{'agg rho (CS)':>14s}{'agg rho (legacy)':>18s}")

    t0 = time.time()
    pred = less_scores(eval_anchors, eval_pool, GRAD_MODEL, max_len=GRAD_MAX_LEN)
    score("less", pred, time.time() - t0)

    t0 = time.time()
    raw, fim = logra_scores(eval_anchors, eval_pool, GRAD_MODEL, batch_size=args.batch_size,
                            max_len=GRAD_MAX_LEN)
    elapsed = time.time() - t0
    score("logra_raw", raw, elapsed)
    score("logra_fim", fim, elapsed)

    t0 = time.time()
    pred = rdsplus_scores(eval_anchors, eval_pool, GRAD_MODEL, batch_size=args.batch_size,
                          max_len=GRAD_MAX_LEN)
    score("rdsplus", pred, time.time() - t0)

    t0 = time.time()
    pred = tfidf_scores(eval_anchors, eval_pool)
    score("tfidf", pred, time.time() - t0)

    t0 = time.time()
    pred = embedding_scores(eval_anchors, eval_pool, encoder_model=args.encoder_model)
    score("embedding", pred, time.time() - t0)

    # For direct comparison: InfluCoder's own trained-encoder numbers from
    # the legacy-GT sample-eff re-run, if present.
    legacy_sweep_path = out_dir / "part_a_legacy_gt_results.json"
    if legacy_sweep_path.exists():
        sweep = json.loads(legacy_sweep_path.read_text())
        results["influcoder_sample_eff"] = sweep["size_trials"]
        print("\n(for reference) InfluCoder trained encoder, from the sample-eff sweep:")
        for t in sweep["size_trials"]:
            cs = t["trained"]["countsketch"]["aggregated"]
            lg = t["trained"]["legacy"]["aggregated"]
            print(f"  {t['n_train_a']}x{t['n_train_p']:<10} agg rho: CS={cs:+.4f}  legacy={lg:+.4f}")

    out_path = out_dir / "baselines_on_sample_eff_data.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
