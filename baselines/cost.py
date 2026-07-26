"""Cost accounting for baseline methods: FLOPs and wall-clock, per sample.

WHAT WE ARE MEASURING, AND WHY
------------------------------
Every method here reduces to the same two steps: featurize each sample into a
vector, then take cosines. The cosine step is O(A*P*D) and is negligible next to
featurizing with an LM, so the cost that actually separates these methods is
**what it takes to featurize one sample**. That is the number a practitioner
pays per candidate at selection time, so it is what we report:

    flops_per_sample = featurization FLOPs / (n_anchors + n_pool)
    time_per_sample  = featurization wall-clock / (n_anchors + n_pool)

Crucially this is *inference* cost. One-time costs -- distilling influcoder's
encoder, fitting a TF-IDF vocabulary, inverting LoGRA's FIM -- are amortized
over the pool and are not what a new candidate costs. That distinction is the
whole point of the comparison: influcoder's per-candidate cost is one encoder
forward pass, identical to the untrained `semantic` baseline, while LESS and
LoGRA pay a forward *and backward* through the target LM for every candidate.

Analytically, with model params P_m and a sample of L tokens:
    forward      ~ 2 * P_m * L      (one multiply + one add per param per token)
    backward     ~ 4 * P_m * L      (grads wrt both inputs and weights)
    forward+bwd  ~ 6 * P_m * L
so a gradient method costs ~3x a forward-only method on the *same* model, and
the LM-based methods cost ~(P_lm / P_enc) more than the encoder ones on top of
that. Rather than trust those closed forms, we MEASURE with FlopCounterMode,
which counts real matmul/SDPA FLOPs at true sequence lengths -- that
automatically captures attention terms and the fact that BBH anchors are far
longer than Dolly candidates.

WHERE MEASUREMENT IS BLIND (and we add analytic terms)
------------------------------------------------------
FlopCounterMode only hooks aten matmul-family ops. Two things it provably
cannot see, which we add by hand:
  * fast_jl's CudaProjector -- a custom CUDA kernel, invisible to the counter.
    A Rademacher projection of a P_lora-dim gradient to k dims costs ~2*P_lora*k
    per sample. (Only added when CudaProjector is actually used: TRAK's
    BasicProjector is a torch.mm and IS counted, so adding it there would double
    count.)
  * torch.linalg.pinv -- dispatches to aten._linalg_svd, which has no handler.
    SVD-based pinv of a k x k matrix is ~22*k^3 (Golub-Van Loan), once per FIM
    block.

Timing excludes model loading (a one-time setup cost, not a per-candidate one)
and synchronizes CUDA before stopping the clock, since kernel launches are async
and an unsynchronized timer would report launch time rather than compute time.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def flop_counter():
    """Counts real matmul/SDPA FLOPs performed inside the block.

    Yields an object with .total() -> int. Falls back to a zero-counter if
    FlopCounterMode is unavailable so cost accounting never breaks scoring.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        class _Null:
            def total(self):
                return 0
        yield _Null()
        return

    mode = FlopCounterMode(display=False)

    class _Counter:
        def total(self):
            return int(mode.get_total_flops())

    with mode:
        yield _Counter()


class CostMeter:
    """Separates one-time setup (model load) from per-sample featurization.

    A scorer calls `model_ready()` once its model is loaded; everything after
    that is charged as inference. `extra_flops` is for analytic terms the FLOP
    counter cannot observe (see module docstring).
    """

    def __init__(self):
        self._t0 = time.perf_counter()
        self.load_time_s = 0.0
        self.extra_flops = 0
        self.notes: list[str] = []

    def model_ready(self):
        self.load_time_s = time.perf_counter() - self._t0

    def add_flops(self, n: int, note: str = ""):
        self.extra_flops += int(n)
        if note:
            self.notes.append(note)


def projection_flops(n_params: int, proj_dim: int, n_samples: int) -> int:
    """Rademacher random projection R^n_params -> R^proj_dim, per sample.

    One multiply-accumulate per (param, output dim) pair => 2*n*k FLOPs.
    """
    return 2 * int(n_params) * int(proj_dim) * int(n_samples)


def pinv_flops(n_blocks: int, k: int) -> int:
    """SVD-based pseudo-inverse of `n_blocks` k x k matrices (~22k^3 each)."""
    return 22 * int(n_blocks) * (int(k) ** 3)


def summarize(measured_flops: int, meter: CostMeter, total_time_s: float,
              n_samples: int) -> dict:
    """Per-sample cost record for one method."""
    inference_time_s = max(total_time_s - meter.load_time_s, 0.0)
    total_flops = int(measured_flops) + int(meter.extra_flops)
    n = max(int(n_samples), 1)
    return {
        "flops_total": total_flops,
        "flops_measured": int(measured_flops),
        "flops_analytic": int(meter.extra_flops),
        "flops_per_sample": total_flops / n,
        "inference_time_s": inference_time_s,
        "load_time_s": meter.load_time_s,
        "time_per_sample_ms": 1000.0 * inference_time_s / n,
        "n_samples": n,
        "cost_notes": meter.notes,
    }


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
