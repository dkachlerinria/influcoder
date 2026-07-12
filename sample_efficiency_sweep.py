#!/usr/bin/env python3
"""Rebuttal experiment: how much do we lose by collecting fewer anchor/pool
samples for the main (BBH anchors x Dolly pool) encoder training run, and
can a small number of targeted changes claw some of it back?

Reference point: the "big" preset (500 train anchors x 1000 train pool,
100x100 eval) is the main run this whole investigation is calibrated
against -- trained per-anchor rho ~0.53, aggregated rho ~0.60 (runs_out/big).

Part A -- degradation curve. Nested subsets: shrink
(n_train_a, n_train_p) together, ratio fixed at 1:2, everything else
(epochs, hard_ratio, lr, k_anchors, m_candidates, alpha) held at the "big"
recipe. Gradients are computed ONCE over the full 500x1000 superset (plus
the fixed 100x100 eval set) and cached to disk; every trial just slices the
front of those tensors, so smaller-n trials are strict subsets of larger
ones -- a controlled shrink, not a resample.

Part B -- recovery. At one reduced size from part A's curve (picked after
looking at where the curve is degraded but not flat), try individually-cheap
changes already exposed by the pipeline (hard-negative mining, Pearson/KL
loss mix, epoch count) to see whether any claws back signal. Every trial's
delta dict has <=3 keys, same discipline as recalibrate_sweep.py -- this is
a constrained ablation of the existing method's own knobs, not new
machinery.

Usage:
    python sample_efficiency_sweep.py --part a
    python sample_efficiency_sweep.py --part b --b_na 125 --b_np 250
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

from influcoder.data import disjoint_splits, load_bbh, load_dolly
from influcoder.encoder import distill, embed, load_encoder
from influcoder.gradients import GradientFeaturizer
from influcoder.metrics import spearman_metrics

# The "big" preset recipe -- our ~0.6 rho reference run. Part A only ever
# changes n_train_a/n_train_p from this; part B trials change <=3 other keys.
BASELINE = dict(n_train_a=500, n_train_p=1000, epochs=8, lr=5e-5,
                hard_ratio=0.0, k_anchors=8, m_candidates=16, alpha=0.5)
N_EVAL_A, N_EVAL_P = 100, 100  # fixed, noise-resistant ruler -- never shrunk

# Ratio-preserving (1:2) shrink of the baseline, nested (nesting drops from
# the same shuffled superset, so trial n is a strict prefix of trial n-1).
SIZE_TRIALS = [
    (500, 1000), (250, 500), (125, 250), (62, 125), (30, 60), (16, 32), (8, 16),
]

GRAD_MODEL = "HuggingFaceTB/SmolLM2-135M"
PROJ_DIM = 8192
GRAD_MAX_LEN = 1024
LORA_RANK = 8
CACHE = Path("runs_out/sample_eff/grad_cache.pt")


def compute_or_load_gradients(seed: int):
    if CACHE.exists():
        print(f"loading cached gradients from {CACHE}")
        return torch.load(CACHE, weights_only=False)

    t0 = time.time()
    splits = disjoint_splits(
        load_bbh("data/eval/bbh", seed=42), load_dolly("dolly/dolly_data.jsonl", seed=42),
        N_EVAL_A, BASELINE["n_train_a"], N_EVAL_P, BASELINE["n_train_p"],
    )
    print(f"data: eval {N_EVAL_A}x{N_EVAL_P}, train superset "
          f"{BASELINE['n_train_a']}x{BASELINE['n_train_p']}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    feat = GradientFeaturizer(GRAD_MODEL, lora_rank=LORA_RANK, lora_seed=seed,
                              proj_dim=PROJ_DIM, max_len=GRAD_MAX_LEN)
    g_eval_a, _ = feat.features(splits["eval_anchors"], "grads eval anchors")
    g_eval_p, _ = feat.features(splits["eval_pool"], "grads eval pool")
    g_train_a, _ = feat.features(splits["train_anchors"], "grads train-superset anchors")
    g_train_p, _ = feat.features(splits["train_pool"], "grads train-superset pool")
    feat.close()
    print(f"gradients: {time.time()-t0:.1f}s")

    cache = {
        "g_eval_a": g_eval_a, "g_eval_p": g_eval_p,
        "g_train_a": g_train_a, "g_train_p": g_train_p,
        "eval_a_texts": [s.text for s in splits["eval_anchors"]],
        "eval_p_texts": [s.text for s in splits["eval_pool"]],
        "train_a_texts": [s.text for s in splits["train_anchors"]],
        "train_p_texts": [s.text for s in splits["train_pool"]],
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE)
    print(f"cached gradients to {CACHE} (reused by later --part b runs)")
    return cache


def run_trial(cache, gt, na, np_, cfg, seed, encoder_model):
    enc = load_encoder(encoder_model)
    g_a = cache["g_train_a"][:na]
    g_p = cache["g_train_p"][:np_]
    targets = g_a @ g_p.T
    train_a_texts = cache["train_a_texts"][:na]
    train_p_texts = cache["train_p_texts"][:np_]

    def eval_spearman():
        pred = embed(enc, cache["eval_a_texts"]) @ embed(enc, cache["eval_p_texts"]).T
        return spearman_metrics(pred, gt)

    baseline = eval_spearman()
    t0 = time.time()
    train_log = distill(enc, train_a_texts, train_p_texts, targets,
                        epochs=cfg["epochs"], k_anchors=cfg["k_anchors"],
                        m_candidates=cfg["m_candidates"], lr=cfg["lr"],
                        hard_ratio=cfg["hard_ratio"], alpha=cfg["alpha"],
                        seed=seed, epoch_eval=eval_spearman)
    final = eval_spearman()
    elapsed = time.time() - t0
    del enc
    torch.cuda.empty_cache()
    return {"baseline": baseline, "trained": final, "best_epoch": train_log["best_epoch"],
            "elapsed_s": round(elapsed, 1), "n_train_a": na, "n_train_p": np_, "config": cfg}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs_out/sample_eff")
    ap.add_argument("--b_na", type=int, help="part b: reduced n_train_a to intervene on")
    ap.add_argument("--b_np", type=int, help="part b: reduced n_train_p to intervene on")
    ap.add_argument("--only", nargs="*", default=None,
                    help="part b: run only these named trials (default: all)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = compute_or_load_gradients(args.seed)
    gt = (cache["g_eval_a"] @ cache["g_eval_p"].T).numpy()

    if args.part == "a":
        results = {"baseline_recipe": BASELINE, "size_trials": []}
        print(f"\n{'n_train_a x n_train_p':24s}{'per-anchor rho':>16s}{'agg rho':>10s}{'time(s)':>9s}")
        for na, np_ in SIZE_TRIALS:
            cfg = {**BASELINE, "n_train_a": na, "n_train_p": np_}
            r = run_trial(cache, gt, na, np_, cfg, args.seed, args.encoder_model)
            results["size_trials"].append(r)
            print(f"{f'{na}x{np_}':24s}{r['trained']['per_anchor_mean']:>+16.4f}"
                  f"{r['trained']['aggregated']:>+10.4f}{r['elapsed_s']:>9.0f}")
        with open(out_dir / "part_a_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {out_dir / 'part_a_results.json'}")

    else:
        if args.b_na is None or args.b_np is None:
            raise SystemExit("--part b requires --b_na and --b_np")
        reduced = dict(BASELINE, n_train_a=args.b_na, n_train_p=args.b_np)
        TRIALS_B = [
            ("reduced_baseline", {}),
            ("hard_negatives", dict(hard_ratio=0.5)),
            ("kl_weighted", dict(alpha=0.2)),
            ("more_epochs", dict(epochs=3 * BASELINE["epochs"])),
        ]
        if args.only:
            TRIALS_B = [(n, d) for n, d in TRIALS_B if n in args.only]
        for name, delta in TRIALS_B:
            assert len(delta) <= 3, f"trial {name} changes {len(delta)} settings (>3): {delta}"
            assert set(delta) <= set(BASELINE), f"trial {name} has unknown keys: {set(delta)-set(BASELINE)}"

        results = {"reduced_size": {"n_train_a": args.b_na, "n_train_p": args.b_np},
                  "reduced_baseline_recipe": reduced, "trials": {}}
        print(f"\npart B at {args.b_na}x{args.b_np}:")
        print(f"{'trial':20s}{'delta':>28s}{'per-anchor rho':>16s}{'agg rho':>10s}{'time(s)':>9s}")
        for name, delta in TRIALS_B:
            cfg = {**reduced, **delta}
            r = run_trial(cache, gt, args.b_na, args.b_np, cfg, args.seed, args.encoder_model)
            r["delta"] = delta
            results["trials"][name] = r
            print(f"{name:20s}{str(delta):>28s}{r['trained']['per_anchor_mean']:>+16.4f}"
                  f"{r['trained']['aggregated']:>+10.4f}{r['elapsed_s']:>9.0f}")
        out_path = out_dir / f"part_b_{args.b_na}x{args.b_np}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
