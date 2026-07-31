#!/usr/bin/env python3
"""EXP1 Part 3: GPU-time-vs-Samples-Processed Amortization.

At what point does InfluCoder's up-front setup cost (teacher-gradient
collection + distillation training) pay for itself against LESS's/LoGRA's
per-sample cost, in real cumulative GPU time?

"Process" means computing the per-sample representation ONLY (LESS: projected
gradient via `collect_grads`; LoGRA: per-sample [r,r] gradient factor via
`LoGra.encode`; InfluCoder: encoder embedding via `embed`) -- never the
anchor-vs-pool scoring matmul, which is out of scope (depends on query-set
size). Because this part only times wall-clock rather than scoring quality,
its code path necessarily looks different from Parts 1/2 (no `report()`/GT
comparison) -- but every model it loads goes through
`baselines.exp1.methods.load_less_model`/`load_logra_model` (same rank/attn/
block_size Part 1 uses) and its InfluCoder training goes through
`baselines.exp1.train.train_influcoder` (same epochs/hard_ratio/lr/seed Parts
1 and 2 use), so the PARAMETERS are identical even though the orchestration
isn't.

LESS/LoGRA are only measured at n=100/1000 (cost-prohibitive beyond that) and
extrapolated from there; InfluCoder is measured at n=100/1000/10000. Samples
come from `tasksource/dolci-instruct` -- the only one of this repo's three
pools with enough real, non-duplicated text to eventually cover 100K/1M (BBH
caps at 6511 total, Dolly at 15011).

Run the two halves as separate GPU jobs in parallel (no shared GPU-memory
contention) via --only:

    python -m baselines.exp1.part3 --only less_logra
    python -m baselines.exp1.part3 --only influcoder
    python -m baselines.exp1.part3 --only both   # sequential, one GPU
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from baselines.common import tokenized_dataset
from baselines.less.less_embeds import collect_grads
from influcoder.data import load_bbh, load_dolci_instruct
from influcoder.encoder import embed
from influcoder.gradients import GradientFeaturizer

from . import config as cfg
from . import methods, train

LESS_LOGRA_SIZES = [100, 1_000]
INFLUCODER_PROCESS_SIZES = [100, 1_000, 10_000]
N_TRAIN_A_SMALL = 250   # deliberately tiny exploratory set -- NOT config.N_TRAIN_A;
N_TRAIN_P_SMALL = 500   # this part times setup cost, doesn't measure quality,
                       # so a small, cheap-to-collect set is the point (see
                       # EXP1.md section 4.2.2)

OUT_LESS_LOGRA = Path("baselines/out") / cfg.PRESET / "exp1_part3_less_logra.json"
OUT_INFLUCODER = Path("baselines/out") / cfg.PRESET / "exp1_part3_influcoder.json"


def get_samples(n, seed):
    return load_dolci_instruct(seed=seed, max_docs=n)[:n]


def time_less_logra():
    samples_by_n = {n: get_samples(n, seed=7) for n in LESS_LOGRA_SIZES}
    print(f"loaded dolci-instruct samples: {[len(samples_by_n[n]) for n in LESS_LOGRA_SIZES]}\n")

    print(f"########## LESS {cfg.GT_MODEL} (r={cfg.LESS_RANK}, {cfg.ATTN}) ##########")
    t_load0 = time.perf_counter()
    tok, model = methods.load_less_model(cfg.GT_MODEL)
    torch.cuda.synchronize()
    less_load_s = time.perf_counter() - t_load0

    less_points = []
    for n in LESS_LOGRA_SIZES:
        dl = torch.utils.data.DataLoader(
            tokenized_dataset(tok, samples_by_n[n], cfg.MAX_LEN), batch_size=1, shuffle=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        collect_grads(dl, model, proj_dim=cfg.LESS_PROJ_DIM, adam_optimizer_state=None,
                     gradient_type="sgd", project_interval=cfg.LESS_PROJECT_INTERVAL,
                     block_size=cfg.LESS_BLOCK_SIZE)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        less_points.append({"n": n, "process_time_s": elapsed, "ms_per_sample": 1000 * elapsed / n})
        print(f"  LESS: n={n:6d}  {elapsed:8.2f}s  {1000 * elapsed / n:8.2f} ms/sample")
    del model
    torch.cuda.empty_cache()

    print(f"\n########## LoGRA Qwen/Qwen3-1.7B (r={cfg.LOGRA_RANK}, {cfg.ATTN}) ##########")
    t_load0 = time.perf_counter()
    logra = methods.load_logra_model("Qwen/Qwen3-1.7B")
    torch.cuda.synchronize()
    logra_load_s = time.perf_counter() - t_load0

    logra_points = []
    for n in LESS_LOGRA_SIZES:
        ds = tokenized_dataset(logra.tokenizer, samples_by_n[n], cfg.MAX_LEN)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logra.encode(ds, batch_size=1, is_test=False, show_progress_bar=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        logra_points.append({"n": n, "process_time_s": elapsed, "ms_per_sample": 1000 * elapsed / n})
        print(f"  LoGRA: n={n:6d}  {elapsed:8.2f}s  {1000 * elapsed / n:8.2f} ms/sample")
    del logra
    torch.cuda.empty_cache()

    OUT_LESS_LOGRA.parent.mkdir(parents=True, exist_ok=True)
    OUT_LESS_LOGRA.write_text(json.dumps({
        "config": {"max_len": cfg.MAX_LEN, "attn": cfg.ATTN, "sizes": LESS_LOGRA_SIZES,
                  "less_model": cfg.GT_MODEL, "less_rank": cfg.LESS_RANK,
                  "logra_model": "Qwen/Qwen3-1.7B", "logra_rank": cfg.LOGRA_RANK},
        "less": {"load_time_s": less_load_s, "points": less_points},
        "logra": {"load_time_s": logra_load_s, "points": logra_points},
    }, indent=2))
    print(f"\nwrote {OUT_LESS_LOGRA}")


def time_influcoder():
    encoder_model = cfg.ENCODER_MODELS["68m"]

    print(f"########## (A) collect data: {cfg.GT_MODEL} (r={cfg.GT_LORA_RANK}) "
          f"grad targets, {N_TRAIN_A_SMALL}x{N_TRAIN_P_SMALL} ##########")
    anchors = load_bbh("data/eval/bbh", seed=11)[:N_TRAIN_A_SMALL]
    pool = load_dolci_instruct(seed=11, max_docs=N_TRAIN_P_SMALL)[:N_TRAIN_P_SMALL]

    t0 = time.perf_counter()
    feat = GradientFeaturizer(cfg.GT_MODEL, lora_rank=cfg.GT_LORA_RANK, lora_seed=cfg.SEED,
                              proj_dim=cfg.INFLUCODER_PROJ_DIM, max_len=cfg.MAX_LEN)
    g_a, _ = feat.features(anchors, "targets: anchors")
    g_p, _ = feat.features(pool, "targets: pool")
    feat.close()
    torch.cuda.synchronize()
    collect_s = time.perf_counter() - t0
    print(f"  collect: {collect_s:.1f}s for {N_TRAIN_A_SMALL}+{N_TRAIN_P_SMALL}"
          f"={N_TRAIN_A_SMALL + N_TRAIN_P_SMALL} samples\n")
    targets = g_a @ g_p.T

    print(f"########## (B) train InfluCoder 68m, {cfg.INFLUCODER_EPOCHS} epochs, "
          f"{N_TRAIN_A_SMALL}x{N_TRAIN_P_SMALL} ##########")
    # No eval_anchor_texts/eval_pool_texts/gt passed -- this part times setup
    # cost, never quality, so no epoch_eval callback gets built at all.
    t0 = time.perf_counter()
    enc, _, _, _ = train.train_influcoder(
        encoder_model, [s.text for s in anchors], [s.text for s in pool], targets)
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    print(f"  train: {train_s:.1f}s\n")

    setup_s = collect_s + train_s
    print(f"total setup (A+B) = {setup_s:.1f}s ({collect_s:.1f}s collect + {train_s:.1f}s train)\n")

    print("########## (C) process: InfluCoder 68m encode() throughput ##########")
    embed(enc, ["warmup sample to eat the first-call CUDA kernel compile cost"])
    torch.cuda.synchronize()

    max_n = max(INFLUCODER_PROCESS_SIZES)
    texts = [s.text for s in load_dolci_instruct(seed=13, max_docs=max_n)[:max_n]]

    process_points = []
    for n in INFLUCODER_PROCESS_SIZES:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        embed(enc, texts[:n], batch_size=32)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        process_points.append({"n": n, "process_time_s": elapsed,
                               "ms_per_sample": 1000 * elapsed / n})
        print(f"  n={n:6d}  {elapsed:8.2f}s  {1000 * elapsed / n:8.3f} ms/sample")
    train.free(enc)

    OUT_INFLUCODER.parent.mkdir(parents=True, exist_ok=True)
    OUT_INFLUCODER.write_text(json.dumps({
        "config": {"grad_model": cfg.GT_MODEL, "gt_lora_rank": cfg.GT_LORA_RANK,
                  "encoder_model": encoder_model, "epochs": cfg.INFLUCODER_EPOCHS,
                  "hard_ratio": cfg.INFLUCODER_HARD_RATIO, "lr": cfg.INFLUCODER_LR,
                  "n_train_anchors": N_TRAIN_A_SMALL, "n_train_pool": N_TRAIN_P_SMALL},
        "collect_time_s": collect_s, "train_time_s": train_s,
        "setup_time_s": setup_s, "process_points": process_points,
    }, indent=2))
    print(f"\nwrote {OUT_INFLUCODER}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["less_logra", "influcoder", "both"], default="both")
    args = ap.parse_args()

    if args.only in ("less_logra", "both"):
        time_less_logra()
    if args.only in ("influcoder", "both"):
        time_influcoder()


if __name__ == "__main__":
    main()
