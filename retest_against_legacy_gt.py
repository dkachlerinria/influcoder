#!/usr/bin/env python3
"""Re-evaluate a trained InfluCoder encoder against tis-ie's "actual" ground
truth (plain-SGD LoRA gradients, TRAK-Rademacher-projected -- see
influcoder/legacy_gt.py) instead of this repo's own CountSketch ground truth.

This is an external-validity check on the whole pipeline: the encoder is
trained to reproduce CountSketch-projected gradient cosines, and run.py
already reports how well it does at that. The question here is different --
does it (and does the CountSketch ground truth it was trained against) still
agree with the OLD repo's actual definition of ground truth, computed with a
completely different (TRAK Rademacher, dropout-noised) projection?

Usage:
    python run.py --preset tiny                                   # train + save an encoder first
    python retest_against_legacy_gt.py --preset tiny              # then re-test it
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
from influcoder.encoder import embed, load_encoder
from influcoder.gradients import GradientFeaturizer
from influcoder.legacy_gt import DEFAULT_TARGET_MODULES, legacy_gt_scores
from influcoder.metrics import spearman_metrics
from run import PRESETS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="sanity",
                    help="Must match the preset used to train the encoder being tested.")
    ap.add_argument("--encoder_dir", default=None, help="default: runs_out/<preset>/encoder")
    ap.add_argument("--encoder_model", default=None,
                    help="untrained-reference base model; default: read from results.json, "
                         "else jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--grad_model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--proj_dim", type=int, default=65536, help="tis-ie GT_PROJ_DIM default")
    ap.add_argument("--lora_dropout", type=float, default=0.1,
                    help="tis-ie stock value (stochastic GT); pass 0.0 for determinism")
    ap.add_argument("--seed", type=int, default=0, help="legacy GT's own LoRA seed")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    out_dir = Path(args.out_dir or f"runs_out/{args.preset}")
    encoder_dir = Path(args.encoder_dir or out_dir / "encoder")

    if not encoder_dir.exists():
        raise FileNotFoundError(
            f"{encoder_dir} not found -- run `python run.py --preset {args.preset}` "
            f"first to train and save an encoder."
        )

    prior_path = out_dir / "results.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    if prior is None:
        print(f"warning: {prior_path} not found -- can't show the CountSketch-GT "
              f"comparison numbers or infer encoder_model/lora_rank/seed from the "
              f"original run; using CLI defaults instead.")

    encoder_model = args.encoder_model or (prior["config"]["encoder_model"] if prior
                                           else "jhu-clsp/ettin-encoder-68m")
    gt_lora_rank = prior["config"]["lora_rank"] if prior else 8
    gt_seed = prior["config"]["seed"] if prior else args.seed

    splits = disjoint_splits(
        load_bbh("data/eval/bbh", seed=42),
        load_dolly("dolly/dolly_data.jsonl", seed=42),
        cfg["n_eval_a"], cfg["n_train_a"], cfg["n_eval_p"], cfg["n_train_p"],
    )
    eval_anchors, eval_pool = splits["eval_anchors"], splits["eval_pool"]
    a_texts = [s.text for s in eval_anchors]
    p_texts = [s.text for s in eval_pool]
    print(f"eval split: {cfg['n_eval_a']}x{cfg['n_eval_p']} (same split run.py used)")

    # -- this repo's own (CountSketch) ground truth, for direct agreement check -----
    print(f"\ncomputing this repo's own CountSketch ground truth "
          f"(lora_rank={gt_lora_rank}, seed={gt_seed}, proj_dim={cfg['proj_dim']})...")
    t0 = time.time()
    feat = GradientFeaturizer(args.grad_model, lora_rank=gt_lora_rank, lora_seed=gt_seed,
                              proj_dim=cfg["proj_dim"], max_len=cfg["grad_max_len"])
    g_a, _ = feat.features(eval_anchors, "CountSketch GT anchors")
    g_p, _ = feat.features(eval_pool, "CountSketch GT pool")
    feat.close()
    countsketch_gt = (g_a @ g_p.T).numpy()
    print(f"CountSketch GT: {time.time() - t0:.1f}s")

    # -- tis-ie's actual ground truth -------------------------------------------------
    print(f"\ncomputing tis-ie-style legacy ground truth "
          f"(proj_dim={args.proj_dim}, lora_rank=16, lora_dropout={args.lora_dropout})...")
    t0 = time.time()
    legacy_gt = legacy_gt_scores(eval_anchors, eval_pool, args.grad_model,
                                 proj_dim=args.proj_dim, lora_dropout=args.lora_dropout,
                                 lora_seed=args.seed, max_len=cfg["grad_max_len"])
    print(f"legacy GT: {time.time() - t0:.1f}s")

    gt_agreement = spearman_metrics(countsketch_gt, legacy_gt)
    print(f"\nCountSketch GT vs. legacy (tis-ie) GT: "
          f"per-anchor rho={gt_agreement['per_anchor_mean']:+.4f}  "
          f"agg rho={gt_agreement['aggregated']:+.4f}")

    # -- encoders vs. legacy ground truth ----------------------------------------------
    print(f"\nloading trained encoder from {encoder_dir}...")
    trained_enc = load_encoder(str(encoder_dir))
    trained_pred = embed(trained_enc, a_texts) @ embed(trained_enc, p_texts).T
    trained_vs_legacy = spearman_metrics(trained_pred, legacy_gt)

    print(f"loading untrained reference encoder ({encoder_model})...")
    untrained_enc = load_encoder(encoder_model)
    untrained_pred = embed(untrained_enc, a_texts) @ embed(untrained_enc, p_texts).T
    untrained_vs_legacy = spearman_metrics(untrained_pred, legacy_gt)

    print("\n=== re-test vs. tis-ie-style legacy ground truth ===")
    print(f"{'':24s}{'per-anchor rho':>16s}{'agg rho':>10s}")
    print(f"{'untrained encoder':24s}{untrained_vs_legacy['per_anchor_mean']:>+16.4f}"
          f"{untrained_vs_legacy['aggregated']:>+10.4f}")
    print(f"{'trained (influcoder)':24s}{trained_vs_legacy['per_anchor_mean']:>+16.4f}"
          f"{trained_vs_legacy['aggregated']:>+10.4f}")

    if prior:
        print("\n=== for comparison: same encoder vs. this repo's own CountSketch GT ===")
        b, t = prior["baseline"], prior["trained"]
        print(f"{'':24s}{'per-anchor rho':>16s}{'agg rho':>10s}")
        print(f"{'untrained encoder':24s}{b['per_anchor_mean']:>+16.4f}{b['aggregated']:>+10.4f}")
        print(f"{'trained (influcoder)':24s}{t['per_anchor_mean']:>+16.4f}{t['aggregated']:>+10.4f}")

    out = {
        "preset": args.preset,
        "encoder_dir": str(encoder_dir),
        "legacy_gt_config": {
            "proj_dim": args.proj_dim, "lora_rank": 16, "lora_alpha": 32,
            "lora_dropout": args.lora_dropout, "lora_target_modules": DEFAULT_TARGET_MODULES,
            "lora_seed": args.seed, "grad_model": args.grad_model,
            "note": "model.train() with dropout active -- stochastic, matches tis-ie",
        },
        "countsketch_gt_vs_legacy_gt": gt_agreement,
        "untrained_vs_legacy_gt": untrained_vs_legacy,
        "trained_vs_legacy_gt": trained_vs_legacy,
        "prior_vs_countsketch_gt": {"baseline": prior["baseline"], "trained": prior["trained"]}
                                   if prior else None,
    }
    out_path = out_dir / "legacy_gt_retest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
