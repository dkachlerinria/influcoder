#!/usr/bin/env python3
"""How does InfluCoder's Spearman scale with the size of its distillation set?

Sweeps the training-side size at a fixed 1:2 anchor:pool ratio, holding the eval
split and the ground truth *completely* fixed (the same 100x100 GT every other
baseline is scored on), and trains a fresh encoder at each size.

Two properties make the curve honest:

  * `disjoint_splits` takes FRONT slices, and the eval sizes are pinned, so a
    smaller training set is a strict prefix of every larger one. The sizes are
    therefore nested subsets -- moving along the x axis changes the amount of
    data and nothing else. No resampling noise is folded into the trend.
  * Gradient features are computed ONCE at the largest size and sliced, so every
    point sees byte-identical features for the samples it shares with its
    neighbours.

Competitor reference lines are horizontal because their cost does not depend on
this axis at all: LESS and LoGRA do no distillation, so a training-set size is
undefined for them -- they are a level to cross, not a curve to race.

    python -m baselines.scaling_sweep --sizes 25 50 100 200 350 500 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from baselines.common import CACHE_DIR, DEFAULT_GRAD_MODEL, ground_truth
from influcoder.encoder import distill, embed, load_encoder
from influcoder.gradients import GradientFeaturizer
from influcoder.metrics import spearman_metrics


def train_features(splits, cfg, grad_model, lora_rank, seed, data_seed=None):
    """Gradient features for the FULL train side, cached. Sliced per size.

    `data_seed` -- same meaning as `baselines.common.build_splits`'s: which
    shuffle produced the eval/train partition these `splits` came from. Not
    in the cache key unless it's a non-default value (see below) -- omitting
    it for the common case keeps every existing cache file's name unchanged."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_a, n_p = len(splits["train_anchors"]), len(splits["train_pool"])
    # Model name goes in the key: without it, a Qwen3-4B run at the same
    # size/rank/seed as a cached SmolLM2 run would silently load the wrong
    # model's gradients as "cache hit". Slug so "/" doesn't create a path.
    model_slug = grad_model.replace("/", "_")
    # Eval sizes go in the key too: disjoint_splits front-slices eval first,
    # so train_anchors starts at index n_eval_a. Two presets can share the
    # same (n_train_a, n_train_p) -- e.g. paper200x3 (n_eval_a=200) and fig1
    # (n_eval_a=400) both use 1500x3000 -- while covering completely
    # different underlying samples (offset 200 vs 400). Without n_eval_a/p
    # in the key, the second preset silently "cache hits" the first's
    # gradients against its OWN (mismatched) text samples -- this actually
    # happened once and produced a fully scrambled, near-zero-Spearman
    # training run before the bug was caught.
    #
    # Pool name goes in the key too (CONFIRMED real, not hypothetical: this is
    # exactly what corrupted the fig1_dolci InfluCoder checkpoints -- see
    # FINDINGS.md). "fig1" (dolly pool) and "fig1_dolci" (dolci_instruct pool)
    # are identical on every OTHER field of this key -- same model/train
    # size/rank/seed/eval size -- so without the pool name, whichever preset's
    # train_features() call ran second silently reused the first preset's
    # pool-side gradients. distill() then trained against a `targets` matrix
    # whose column j described a completely different text than the actual
    # pool_texts[j] it was embedding -- a scrambled label, not a hard one,
    # collapsing all three encoder sizes uniformly (they share one `targets`
    # matrix). ground_truth()'s GT cache key already included `pool` and was
    # never vulnerable to this; this key was the one place it was missing.
    pool_slug = cfg.get("pool", "dolly")
    # data_seed goes in the key for the SAME reason n_eval_a/pool did above:
    # a different data_seed means a different shuffle, so train_anchors/
    # train_pool at this exact (n_a, n_p, eval size) are DIFFERENT actual
    # samples -- without this, a second data_seed would silently "cache hit"
    # the first's gradients against mismatched text, the identical failure
    # mode already documented above for eval size and pool. Omitted from the
    # filename for the historical default (42 or unset) so every existing
    # cache file's name is unaffected.
    data_tag = "" if data_seed is None or data_seed == 42 else f"_data{data_seed}"
    cache = CACHE_DIR / (f"trainfeat_{model_slug}_{pool_slug}_{n_a}x{n_p}_r{lora_rank}_s{seed}"
                        f"_eval{cfg['n_eval_a']}x{cfg['n_eval_p']}{data_tag}.pt")
    if cache.exists():
        d = torch.load(cache)
        return d["a"], d["p"]
    feat = GradientFeaturizer(grad_model, lora_rank=lora_rank, lora_seed=seed,
                              proj_dim=cfg["proj_dim"], max_len=cfg["grad_max_len"])
    g_a, _ = feat.features(splits["train_anchors"], "grads train anchors")
    g_p, _ = feat.features(splits["train_pool"], "grads train pool")
    feat.close()
    torch.save({"a": g_a, "p": g_p}, cache)
    return g_a, g_p


def run_size(splits, gt, g_ta, g_tp, n_a, n_p, epochs, encoder_model, seed):
    """Train a fresh encoder on the first n_a anchors x n_p pool, eval on the GT."""
    targets = g_ta[:n_a] @ g_tp[:n_p].T
    enc = load_encoder(encoder_model)
    eval_a = [s.text for s in splits["eval_anchors"]]
    eval_p = [s.text for s in splits["eval_pool"]]

    def eval_spearman():
        return spearman_metrics(embed(enc, eval_a) @ embed(enc, eval_p).T, gt)

    baseline = eval_spearman()
    log = distill(enc, [s.text for s in splits["train_anchors"][:n_a]],
                  [s.text for s in splits["train_pool"][:n_p]],
                  targets, epochs=epochs, seed=seed, hard_ratio=0.0,
                  epoch_eval=eval_spearman)
    final = eval_spearman()
    del enc
    torch.cuda.empty_cache()
    return baseline, final, log["best_epoch"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="big")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[25, 50, 100, 200, 350, 500],
                    help="n_train_anchors; pool is 2x each (the 1:2 ratio)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None, help="default: preset epochs")
    ap.add_argument("--grad_model", default=DEFAULT_GRAD_MODEL)
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: baselines/out/<preset>/scaling.json")
    args = ap.parse_args()

    out = Path(args.out or f"baselines/out/{args.preset}/scaling.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    gt, splits, cfg = ground_truth(args.preset, seed=0, grad_model=args.grad_model,
                                   lora_rank=args.lora_rank)
    epochs = args.epochs or cfg["epochs"]
    max_a = max(args.sizes)
    if max_a > len(splits["train_anchors"]):
        raise SystemExit(f"--sizes max {max_a} > preset train anchors "
                         f"{len(splits['train_anchors'])}")
    print(f"GT {gt.shape[0]}x{gt.shape[1]} (fixed); sizes {args.sizes} @1:2; "
          f"seeds {args.seeds}; epochs {epochs}")

    # Feature seed is pinned to 0: the GT was built with lora_seed=0, so the
    # distillation targets must come from the same featurizer geometry. The
    # sweep's --seeds vary encoder init and batch sampling, not the features.
    g_ta, g_tp = train_features(splits, cfg, args.grad_model, args.lora_rank, 0)
    print(f"train features: anchors {tuple(g_ta.shape)}, pool {tuple(g_tp.shape)}")

    points = []
    for n_a in args.sizes:
        n_p = 2 * n_a
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            base, final, best_ep = run_size(splits, gt, g_ta, g_tp, n_a, n_p,
                                            epochs, args.encoder_model, seed)
            runs.append({"seed": seed, "best_epoch": best_ep,
                         "aggregated": final["aggregated"],
                         "per_anchor_mean": final["per_anchor_mean"],
                         "untrained_aggregated": base["aggregated"],
                         "untrained_per_anchor_mean": base["per_anchor_mean"]})
            print(f"  n_a={n_a:4d} n_p={n_p:4d} seed={seed}: "
                  f"agg {final['aggregated']:+.4f} "
                  f"per-anchor {final['per_anchor_mean']:+.4f} "
                  f"(best epoch {best_ep}, {time.time() - t0:.0f}s)")
        aggs = [r["aggregated"] for r in runs]
        pams = [r["per_anchor_mean"] for r in runs]
        points.append({
            "n_train_anchors": n_a, "n_train_pool": n_p, "n_train_pairs": n_a * n_p,
            "agg_mean": float(np.mean(aggs)), "agg_min": float(np.min(aggs)),
            "agg_max": float(np.max(aggs)), "agg_std": float(np.std(aggs)),
            "per_anchor_mean": float(np.mean(pams)),
            "per_anchor_min": float(np.min(pams)), "per_anchor_max": float(np.max(pams)),
            "untrained_agg_mean": float(np.mean([r["untrained_aggregated"] for r in runs])),
            "runs": runs,
        })
        print(f"  -> n_a={n_a}: agg {points[-1]['agg_mean']:+.4f} "
              f"[{points[-1]['agg_min']:+.4f}, {points[-1]['agg_max']:+.4f}]")

    payload = {"config": {"preset": args.preset, "gt_shape": list(gt.shape),
                          "epochs": epochs, "seeds": args.seeds,
                          "grad_model": args.grad_model,
                          "encoder_model": args.encoder_model,
                          "lora_rank": args.lora_rank, "ratio": "1:2"},
               "points": points}
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
