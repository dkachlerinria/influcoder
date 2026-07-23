"""Random baseline: seeded uniform scores.

Faithful to influence_eval/compute_random_scores.py. The noise floor -- any
method below this in Spearman is anti-correlated with true influence.
"""

from __future__ import annotations

import torch


def score_random(splits, seed: int = 0, meter=None) -> torch.Tensor:
    if meter is not None:
        meter.model_ready()  # no model to load
    n_a = len(splits["eval_anchors"])
    n_p = len(splits["eval_pool"])
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n_a, n_p), generator=g, dtype=torch.float32)
