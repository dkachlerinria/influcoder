#!/usr/bin/env python3
"""Does InfluCoder's run-to-run instability come from hard-negative mining?

Motivation: the same Toxicity/Het config produced AUPRC 0.7370 and then 0.6023
on re-run -- large enough to flip the method's rank. Semantic (no training)
reproduced bit-identically across the same re-runs, isolating the
nondeterminism to InfluCoder's training path.

The mechanism is one line, influcoder/encoder.py's _sample_candidates():

    hard = hard[torch.randperm(hard.numel())[:n_hard]]

torch.randperm draws from the GLOBAL torch RNG, and torch.manual_seed is never
called anywhere in this path (verified by grep; it appears only in
gradients.py, an EXP1 code path). Everything ELSE in distill() is seeded --
anchor shuffling and the random negatives both go through
`rng = random.Random(seed)`. So the hard-negative subsample is the single
unseeded hole in an otherwise-deterministic pipeline.

Setting hard_ratio=0.0 takes the else-branch (`hard = torch.empty(0)`), so
torch.randperm is never reached and every candidate comes from the seeded
rng.sample(). Prediction: hard_ratio=0.0 is reproducible run-to-run; any
residual drift is GPU floating-point nondeterminism only, which should be far
smaller than the 0.135 AUPRC swing seen at hard_ratio=0.5.

Writes to its OWN results dir and summary file -- never touches the official
fig2 score files or JSONs.

Usage:
    python methods/exp_hard_ratio_stability.py --task het --hard-ratio 0.0 --rep 1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

OUT_DIR = Path("results_hardratio_stability")
SUMMARY = OUT_DIR / "summary.json"
GRAD_CACHE = OUT_DIR / "_gradcache"


def install_grad_cache(checkpoint: str):
    """Cache teacher gradients across reps.

    compute_gradient_features() is a pure function of (tokenized examples,
    model weights, proj_dim, proj_seed) -- no RNG beyond the seeded CountSketch
    -- and every rep here uses the SAME seeded pools and the SAME teacher
    checkpoint, so it recomputes a bit-identical tensor each time. That is
    ~2.5 min per rep of pure waste. Cache on a digest of the actual tokenized
    inputs (not just their count) so a cache hit cannot silently serve
    gradients computed for different data.
    """
    import torch
    import methods.influcoder.teacher_grads as tg

    original = tg.compute_gradient_features

    def cached(model, tokenized_examples, proj_dim, proj_seed=42, device="cuda", desc="grads"):
        h = hashlib.sha256()
        h.update(f"{checkpoint}|{desc}|{len(tokenized_examples)}|{proj_dim}|{proj_seed}".encode())
        for input_ids, labels in tokenized_examples:
            h.update(input_ids.cpu().numpy().tobytes())
            h.update(labels.cpu().numpy().tobytes())
        path = GRAD_CACHE / f"{h.hexdigest()[:32]}.pt"
        if path.exists():
            print(f"    [grad-cache HIT ] {desc} ({len(tokenized_examples)} ex) <- {path.name}")
            return torch.load(path)
        print(f"    [grad-cache MISS] {desc} ({len(tokenized_examples)} ex) -> computing")
        out = original(model, tokenized_examples, proj_dim, proj_seed=proj_seed,
                       device=device, desc=desc)
        GRAD_CACHE.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
        return out

    # run_influcoder() does `from methods.influcoder.teacher_grads import ...`
    # INSIDE the function body, so it resolves this attribute at call time --
    # patching the module attribute here does take effect.
    tg.compute_gradient_features = cached

TASK_MODULES = {
    "cf": "methods.run_fig2_counterfact",
    "het": "methods.run_fig2_toxicity",
    "hom": "methods.run_fig2_toxicity_hom",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_MODULES))
    ap.add_argument("--hard-ratio", type=float, required=True)
    ap.add_argument("--rep", type=int, required=True, help="repetition index (same config, fresh process)")
    args = ap.parse_args()

    mod = importlib.import_module(TASK_MODULES[args.task])

    # Redirect ALL output away from the official results/ tree. run_influcoder()
    # reads RESULTS_DIR at call time, so reassigning the module global is enough.
    tag = f"{args.task}_hard{args.hard_ratio:g}_rep{args.rep}"
    mod.RESULTS_DIR = OUT_DIR / tag
    mod.INFLUCODER_HARD_RATIO = args.hard_ratio

    print(f"=== hard-ratio stability: task={args.task} hard_ratio={args.hard_ratio} rep={args.rep} ===")
    print(f"    (SEED={mod.SEED}, encoder={mod.INFLUCODER_ENCODER_MODEL}, epochs={mod.INFLUCODER_EPOCHS})")
    install_grad_cache(mod.CHECKPOINT)

    t0 = time.perf_counter()
    wall_s, setup_s, score_path = mod.run_influcoder()
    total_s = time.perf_counter() - t0

    metrics = mod.evaluate(score_path)
    if isinstance(metrics, tuple):
        record = {"recall_at_50": metrics[0], "mrr": metrics[1]}
    else:
        record = {"auprc": metrics}

    record.update({
        "task": args.task,
        "hard_ratio": args.hard_ratio,
        "rep": args.rep,
        "wall_s": wall_s,
        "setup_s": setup_s,
        "total_s": total_s,
        "gpu": mod.detect_gpu_label(),
    })
    print(f"    -> {json.dumps({k: v for k, v in record.items() if k not in ('gpu',)})}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    summary[tag] = record
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"wrote {SUMMARY} (key={tag})")


if __name__ == "__main__":
    main()
