#!/usr/bin/env python3
"""Measure the embed() speedup (bf16 autocast + larger batch) in isolation.

Baseline to beat: Semantic on Toxicity/Het embedded the same 10,197 texts
(10,187 local train + 10 ref) in 422.4s on this exact A40, under the old
fp32 / batch-32 embed(). That number came from a full run_fig2_toxicity.py
--method semantic, so it is directly comparable to what this measures.

Reports per-batch-size wall time and words/sec so the bf16 gain and the batch
gain can be told apart. Writes nothing to results/ -- pure measurement.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "evaluation"))

import types  # noqa: E402
if "litgpt" not in sys.modules:
    for m in ("litgpt", "litgpt.lora", "litgpt.utils", "lightning"):
        sys.modules.setdefault(m, types.ModuleType(m))

import torch  # noqa: E402
from datamodules.load_data import get_dataset  # noqa: E402
from methods.influcoder import _bootstrap  # noqa: F401,E402
from influcoder.encoder import embed, load_encoder  # noqa: E402

ENCODER = "jhu-clsp/ettin-encoder-400m"
MAX_LEN = 1024
OLD_BASELINE_S = 422.4       # fp32 / batch 32, FULL 10197 texts, same A40
OLD_BASELINE_N = 10197
OLD_RATE = OLD_BASELINE_N / OLD_BASELINE_S  # texts/sec, the number we're actually comparing

N_SUBSAMPLE = 1000  # ~10% of the corpus -- enough to amortize batching effects,
                    # small enough this whole script runs in under a minute
BATCH_SIZES = [32, 128, 256]


def main():
    train, ref = get_dataset("Toxicity/Bias", "XSTest-response-Het")
    all_texts = [f"{d['prompt'].strip()}\n{d['response'].strip()}" for d in list(train) + list(ref)]
    # Same length-sorted order the real corpus has (dataset is Benign-first,
    # long-tail elsewhere) is fine here since sentence-transformers internally
    # length-sorts every call anyway -- input order doesn't affect throughput.
    texts = all_texts[:N_SUBSAMPLE]
    print(f"subsample={len(texts)} (of {len(all_texts)})  encoder={ENCODER}  max_seq_len={MAX_LEN}")
    print(f"baseline: {OLD_BASELINE_S:.1f}s for {OLD_BASELINE_N} texts "
          f"= {OLD_RATE:.1f} texts/s (fp32, batch=32)\n")

    enc = load_encoder(ENCODER, max_seq_len=MAX_LEN)
    print(f"encoder param dtype: {next(enc.parameters()).dtype}")

    # Warm up so the first timed config does not eat CUDA/kernel init.
    embed(enc, texts[:64], batch_size=32)
    torch.cuda.synchronize()

    for bs in BATCH_SIZES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            out = embed(enc, texts, batch_size=bs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rate = len(texts) / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  batch={bs:4d}  {dt:6.2f}s  {rate:7.1f} texts/s  "
                  f"({rate / OLD_RATE:.2f}x vs baseline rate)  "
                  f"peak_gpu_mem={peak:.1f}GB  out={out.shape} {out.dtype}")
        except torch.OutOfMemoryError:
            print(f"  batch={bs:4d}  OOM")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
