"""Shared harness for baseline methods.

Every baseline is scored on *exactly* the eval that `run.py` reports: the same
disjoint eval split (BBH anchors x Dolly pool) and the same ground-truth
gradient-influence cosine matrix produced by `influcoder.gradients`. A method's
job is only to produce an [n_eval_a, n_eval_p] score matrix over those same
samples; `spearman_metrics` then compares it to the shared GT, so every row in
the results table is apples-to-apples with the influcoder numbers.

The GT (and the eval split it is built from) is cached under baselines/cache/
keyed by the preset + gradient-model config, so repeated baseline runs do not
recompute it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Repo root on path so `influcoder.*` and `run.PRESETS` resolve when this file
# is imported from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run as _run  # noqa: E402  (single source of truth for preset sizes)
from influcoder.data import Sample, disjoint_splits, load_bbh, load_pool  # noqa: E402
from influcoder.gradients import GradientFeaturizer  # noqa: E402
from influcoder.metrics import spearman_metrics  # noqa: E402

PRESETS = _run.PRESETS
CACHE_DIR = _REPO_ROOT / "baselines" / "cache"
DEFAULT_GRAD_MODEL = "HuggingFaceTB/SmolLM2-135M"
DATA_SEED = 42  # matches run.py's load_bbh/load_dolly seed


# --------------------------------------------------------------------------- #
# Eval split + ground truth (mirrors run.py exactly)
# --------------------------------------------------------------------------- #
def build_splits(preset: str, seed: int = 0, data_seed: int | None = None):
    """The identical disjoint split run.py builds for this preset.

    `data_seed` controls WHICH samples land in eval vs. train (the shuffle
    `load_bbh`/`load_pool` do before disjoint_splits' front-slicing) --
    defaults to the historical fixed `DATA_SEED` (42) when not given, so
    every existing call site is byte-for-byte unaffected. Pass an explicit
    `data_seed` to get a genuinely different eval/train partition (e.g. for a
    multi-seed sweep that wants variance from data selection, not just model/
    training stochasticity) -- `seed` alone (LoRA/model randomness) does NOT
    change which samples are even in the split."""
    cfg = PRESETS[preset]
    data_seed = DATA_SEED if data_seed is None else data_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return cfg, disjoint_splits(
        load_bbh("data/eval/bbh", seed=data_seed),
        load_pool(cfg.get("pool", "dolly"), seed=data_seed),
        cfg["n_eval_a"], cfg["n_train_a"], cfg["n_eval_p"], cfg["n_train_p"],
    )


def _gt_key(preset: str, grad_model: str, lora_rank: int, seed: int, data_seed: int) -> str:
    cfg = PRESETS[preset]
    payload = json.dumps({
        "preset": preset, "grad_model": grad_model, "lora_rank": lora_rank,
        "seed": seed, "n_eval_a": cfg["n_eval_a"], "n_eval_p": cfg["n_eval_p"],
        "proj_dim": cfg["proj_dim"], "grad_max_len": cfg["grad_max_len"],
        "data_seed": data_seed, "pool": cfg.get("pool", "dolly"),
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def ground_truth(preset: str, seed: int = 0, grad_model: str = DEFAULT_GRAD_MODEL,
                 lora_rank: int = 8, data_seed: int | None = None):
    """[n_eval_a, n_eval_p] gradient-influence cosine GT, cached.

    Byte-for-byte the same matrix run.py uses as `gt` for this preset: gradient
    features of the eval anchors and eval pool under the same LoRA featurizer,
    cosine between them. `data_seed` -- see `build_splits`'s docstring --
    defaults to the historical fixed split (42) when not given.
    """
    cfg, splits = build_splits(preset, seed, data_seed)
    data_seed = DATA_SEED if data_seed is None else data_seed
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _gt_key(preset, grad_model, lora_rank, seed, data_seed)
    cache = CACHE_DIR / f"gt_{preset}_{key}.pt"
    if cache.exists():
        # stored as a plain tensor so torch.load's weights_only default is happy
        return torch.load(cache).numpy(), splits, cfg

    feat = GradientFeaturizer(grad_model, lora_rank=lora_rank, lora_seed=seed,
                              proj_dim=cfg["proj_dim"], max_len=cfg["grad_max_len"])
    g_eval_a, _ = feat.features(splits["eval_anchors"], "GT grads eval anchors")
    g_eval_p, _ = feat.features(splits["eval_pool"], "GT grads eval pool")
    feat.close()
    gt_t = g_eval_a @ g_eval_p.T
    torch.save(gt_t, cache)
    return gt_t.numpy(), splits, cfg


# --------------------------------------------------------------------------- #
# Tokenization consistent with the influcoder featurizer (loss on target only)
# --------------------------------------------------------------------------- #
def tokenize_sample(tokenizer, sample: Sample, max_len: int):
    """(input_ids, labels) with loss on target tokens only.

    Same scheme as GradientFeaturizer._encode: target tokenized first and capped
    at max_len//2, context tail-truncated to fit, labels -100 over the context.
    Keeps gradient-based baselines seeing byte-identical spans to the GT.
    """
    tgt = tokenizer(sample.target + (tokenizer.eos_token or ""),
                    add_special_tokens=False).input_ids
    tgt = tgt[: max_len // 2]
    ctx = tokenizer(sample.context, add_special_tokens=False).input_ids
    ctx = ctx[-(max_len - len(tgt)):]
    input_ids = ctx + tgt
    labels = [-100] * len(ctx) + tgt
    return input_ids, labels


class ListDataset:
    """Minimal dataset over pre-tokenized rows, no `datasets` dependency.

    Slicing returns a dict of lists (what LoGra.encode consumes); integer
    indexing returns a dict of tensors (what a torch DataLoader with
    batch_size=1 collates). Covers every baseline's access pattern.
    """

    def __init__(self, rows: list[tuple[list[int], list[int]]]):
        self.input_ids = [r[0] for r in rows]
        self.labels = [r[1] for r in rows]
        self.attention_mask = [[1] * len(r[0]) for r in rows]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return {
                "input_ids": self.input_ids[idx],
                "labels": self.labels[idx],
                "attention_mask": self.attention_mask[idx],
            }
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }


def tokenized_dataset(tokenizer, samples: list[Sample], max_len: int) -> ListDataset:
    return ListDataset([tokenize_sample(tokenizer, s, max_len) for s in samples])


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(name: str, scores, gt, baseline: dict | None = None) -> dict:
    """Print a row in the same shape as run.py's results table and return metrics."""
    if isinstance(scores, torch.Tensor):
        scores = scores.numpy()
    m = spearman_metrics(np.asarray(scores), np.asarray(gt))
    print("\n=== results ===")
    print(f"{'':24s}{'per-anchor rho':>16s}{'agg rho':>10s}")
    if baseline is not None:
        print(f"{'untrained encoder':24s}{baseline['per_anchor_mean']:>+16.4f}"
              f"{baseline['aggregated']:>+10.4f}")
    print(f"{name:24s}{m['per_anchor_mean']:>+16.4f}{m['aggregated']:>+10.4f}")
    return m
