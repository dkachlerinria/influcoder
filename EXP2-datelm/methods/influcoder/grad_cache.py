"""Disk cache for InfluCoder's teacher gradients.

compute_gradient_features() is a pure function of (tokenized examples, teacher
weights, proj_dim, proj_seed) -- its only randomness is the seeded CountSketch.
Every InfluCoder run for a given task uses the SAME seeded pools and the SAME
published checkpoint, so it recomputes a bit-identical tensor every time. That
is several minutes of GPU work repeated on every run, every sweep, every
stability rep, forever.

This caches those tensors on disk keyed by a SHA-256 digest of the actual
tokenized input_ids/labels (not merely their count), plus checkpoint, proj_dim
and proj_seed -- so a cache hit can never silently serve gradients belonging to
different data, a different teacher, or a different projection.

Reported timings stay honest. The elapsed time of the original computation is
stored alongside each tensor, so a run that hits the cache can still report
what the work WOULD have cost from scratch. `stats["grad_s"]` accumulates that
true cost; `stats["grad_s_this_run"]` accumulates only what this process
actually spent. run_influcoder uses the difference to correct setup_s back to a
from-scratch number, so cached and uncached runs remain directly comparable.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path


def install_grad_cache(checkpoint: str, cache_dir: Path) -> dict:
    """Monkeypatch teacher_grads.compute_gradient_features with a cached
    version. Returns a mutable stats dict:

        grad_s           -- true from-scratch cost of all gradient work
        grad_s_this_run  -- seconds actually burned in this process
        hits / misses    -- cache outcome counts
    """
    import torch

    from . import teacher_grads as tg

    original = tg.compute_gradient_features
    stats = {"grad_s": 0.0, "grad_s_this_run": 0.0, "hits": 0, "misses": 0}

    def cached(model, tokenized_examples, proj_dim, proj_seed=42,
               device="cuda", desc="grads"):
        h = hashlib.sha256()
        h.update(f"{checkpoint}|{desc}|{len(tokenized_examples)}|{proj_dim}|{proj_seed}".encode())
        for input_ids, labels in tokenized_examples:
            h.update(input_ids.cpu().numpy().tobytes())
            h.update(labels.cpu().numpy().tobytes())
        path = Path(cache_dir) / f"{h.hexdigest()[:32]}.pt"

        if path.exists():
            blob = torch.load(path, weights_only=False)
            stats["hits"] += 1
            stats["grad_s"] += blob["elapsed_s"]
            print(f"    [grad-cache HIT ] {desc} ({len(tokenized_examples)} ex) "
                  f"-- saved {blob['elapsed_s']:.1f}s")
            return blob["features"]

        print(f"    [grad-cache MISS] {desc} ({len(tokenized_examples)} ex) -> computing")
        t0 = time.perf_counter()
        out = original(model, tokenized_examples, proj_dim, proj_seed=proj_seed,
                       device=device, desc=desc)
        elapsed = time.perf_counter() - t0
        stats["misses"] += 1
        stats["grad_s"] += elapsed
        stats["grad_s_this_run"] += elapsed
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"features": out, "elapsed_s": elapsed}, path)
        return out

    # run_influcoder() does `from methods.influcoder.teacher_grads import ...`
    # inside the function body, so the name resolves off this module attribute
    # at call time -- patching here takes effect.
    tg.compute_gradient_features = cached
    return stats
