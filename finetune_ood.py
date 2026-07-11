#!/usr/bin/env python3
"""Cheap re-adaptation test: does a small amount of FineWeb-anchored
fine-tuning recover the OOD-pool ranking that eval_ood.py found degraded?

Hypothesis: the influence *mechanics* the encoder learned on Dolly aren't
gone on FineWeb, the encoder's embedding space is just calibrated to Dolly's
chat-formatted text. If so, a small continuation of training -- starting from
the already-Dolly-trained checkpoint, not the pretrained base -- on a tiny
disjoint slice of (BBH anchors x FineWeb pool) should re-adapt quickly,
because it only needs to re-calibrate, not learn ranking from scratch.

Data (all disjoint from both the "big" training run and eval_ood.py's eval):
  BBH:     eval_anchors  = bbh[0:100)      (same as eval_ood.py -- what we score against)
           train_anchors = bbh[100:200)    (new, small, for this fine-tune only)
  FineWeb: eval_pool     = fineweb[0:100)  (same as eval_ood.py -- what we score against)
           train_pool    = fineweb[100:300) (new, 200 samples, for this fine-tune only)

Usage:
    python finetune_ood.py --encoder_dir runs_out/big/encoder --epochs 6
"""

import os

import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from influcoder.data import disjoint_splits, load_bbh, load_fineweb
from influcoder.encoder import distill, embed, load_encoder
from influcoder.gradients import GradientFeaturizer, projection_fidelity
from influcoder.metrics import spearman_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder_dir", required=True,
                    help="Already-trained encoder to continue training from "
                         "(e.g. runs_out/big/encoder) -- NOT the pretrained base.")
    ap.add_argument("--grad_model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=8192)
    ap.add_argument("--grad_max_len", type=int, default=1024)
    ap.add_argument("--n_eval_a", type=int, default=100)
    ap.add_argument("--n_train_a", type=int, default=100)
    ap.add_argument("--n_eval_p", type=int, default=100)
    ap.add_argument("--n_train_p", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs_out/big_ft_fineweb")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timings, t0 = {}, time.time()
    bbh = load_bbh("data/eval/bbh", seed=42)
    fineweb = load_fineweb(seed=42)
    splits = disjoint_splits(bbh, fineweb, args.n_eval_a, args.n_train_a,
                             args.n_eval_p, args.n_train_p)
    print(f"eval:  {len(splits['eval_anchors'])} BBH anchors x {len(splits['eval_pool'])} FineWeb pool")
    print(f"train: {len(splits['train_anchors'])} BBH anchors x {len(splits['train_pool'])} FineWeb pool "
          f"(disjoint from eval and from the original Dolly training run)")
    timings["data"] = time.time() - t0

    t0 = time.time()
    feat = GradientFeaturizer(args.grad_model, lora_rank=args.lora_rank, lora_seed=args.seed,
                              proj_dim=args.proj_dim, max_len=args.grad_max_len)
    g_eval_a, _ = feat.features(splits["eval_anchors"], "grads eval anchors (BBH)")
    g_eval_p, _ = feat.features(splits["eval_pool"], "grads eval pool (FineWeb)")
    g_train_a, _ = feat.features(splits["train_anchors"], "grads ft-train anchors (BBH)")
    g_train_p, _ = feat.features(splits["train_pool"], "grads ft-train pool (FineWeb)")
    feat.close()
    timings["gradients"] = time.time() - t0

    gt = (g_eval_a @ g_eval_p.T).numpy()
    targets = g_train_a @ g_train_p.T

    eval_a_texts = [s.text for s in splits["eval_anchors"]]
    eval_p_texts = [s.text for s in splits["eval_pool"]]

    print(f"\nstarting from: {args.encoder_dir} (already Dolly-trained, NOT the pretrained base)")
    enc = load_encoder(args.encoder_dir)

    def eval_spearman():
        pred = embed(enc, eval_a_texts) @ embed(enc, eval_p_texts).T
        return spearman_metrics(pred, gt)

    t0 = time.time()
    before = eval_spearman()
    print(f"before fine-tune (Dolly-trained encoder, on this FineWeb eval slice): "
          f"per-anchor rho={before['per_anchor_mean']:+.4f} agg rho={before['aggregated']:+.4f}")
    timings["before_eval"] = time.time() - t0

    t0 = time.time()
    train_log = distill(enc, [s.text for s in splits["train_anchors"]],
                        [s.text for s in splits["train_pool"]],
                        targets, epochs=args.epochs, seed=args.seed,
                        hard_ratio=0.0, epoch_eval=eval_spearman)
    timings["finetune"] = time.time() - t0

    t0 = time.time()
    after = eval_spearman()
    timings["after_eval"] = time.time() - t0

    encoder_dir = out_dir / "encoder"
    enc.save(str(encoder_dir))

    print("\n=== FineWeb re-adaptation ===")
    print(f"{'':28s}{'per-anchor rho':>16s}{'agg rho':>10s}")
    print(f"{'before fine-tune':28s}{before['per_anchor_mean']:>+16.4f}{before['aggregated']:>+10.4f}")
    print(f"{'after fine-tune':28s}{after['per_anchor_mean']:>+16.4f}{after['aggregated']:>+10.4f}")
    print(f"(for reference, this encoder in-distribution on Dolly: see runs_out/big/results.json)")
    print("timings: " + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items())
          + f", total={sum(timings.values()):.1f}s")

    results = {
        "started_from": args.encoder_dir,
        "config": {**vars(args)},
        "before_finetune": before,
        "after_finetune": after,
        "best_epoch": train_log["best_epoch"],
        "epoch_metrics": train_log["epoch_metrics"],
        "encoder_dir": str(encoder_dir),
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "total_s": round(sum(timings.values()), 2),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
