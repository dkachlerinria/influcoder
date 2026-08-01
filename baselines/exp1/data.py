"""EXP1: shared data setup -- ground truth, eval slicing, train-side features.

Every part that needs the GT/eval split or the cached train-side gradient
features goes through these two functions, so "how the data is loaded and
sliced" can never quietly diverge between parts the way parameter constants
used to (see config.py's docstring).
"""
from __future__ import annotations

from baselines.common import ground_truth
from baselines.scaling_sweep import train_features

import os as _os
if _os.environ.get("EXP1_CONFIG") == "biggpu":
    from . import config_biggpu as cfg  # BIG_GPU_FINAL_EXP1 -- see that module's docstring
else:
    from . import config as cfg


def load_gt_and_splits(n_eval: int | None = None):
    """Full (cached) GT/splits/cfg for `config.PRESET`, optionally sliced to
    an `n_eval` x `n_eval` eval subset.

    Slicing takes FRONT prefixes of the already-cached full eval split, so a
    smaller `n_eval` is always a strict subset of a larger one (nested, not a
    fresh random draw) -- this is the same slicing Part 1's original 50x50
    peek relied on for its held-out-data guarantee (see EXP1.md section 5.2):
    the already-trained InfluCoder checkpoints were held out on the FULL
    eval split, so any front-slice of it is still guaranteed held-out data.

    Returns (gt, splits, full_cfg) where `splits` has `eval_anchors`/
    `eval_pool` sliced to n_eval (if given) and `train_anchors`/`train_pool`
    left untouched (those are sliced per-size by callers that need a training
    sweep, e.g. Part 2).
    """
    full_gt, full_splits, preset_cfg = ground_truth(
        cfg.PRESET, seed=cfg.SEED, grad_model=cfg.GT_MODEL, lora_rank=cfg.GT_LORA_RANK,
        data_seed=cfg.DATA_SEED)

    assert preset_cfg["grad_max_len"] == cfg.MAX_LEN == preset_cfg.get("encoder_max_len", cfg.MAX_LEN), (
        f"fig1_dolci's grad_max_len/encoder_max_len no longer both equal "
        f"config.MAX_LEN ({cfg.MAX_LEN}) -- the equal-sequence-length fairness "
        f"constraint would silently break. Do not remove this assertion.")

    if n_eval is None:
        return full_gt, full_splits, preset_cfg

    if n_eval > full_gt.shape[0] or n_eval > full_gt.shape[1]:
        raise ValueError(f"n_eval={n_eval} exceeds the cached eval split "
                         f"({full_gt.shape[0]}x{full_gt.shape[1]})")

    gt = full_gt[:n_eval, :n_eval]
    splits = dict(full_splits)
    splits["eval_anchors"] = full_splits["eval_anchors"][:n_eval]
    splits["eval_pool"] = full_splits["eval_pool"][:n_eval]
    return gt, splits, preset_cfg


def load_train_features(splits, preset_cfg):
    """Cached Qwen3-4B (r16) gradient features for the FULL train side.

    Thin wrapper around `baselines.scaling_sweep.train_features` pinned to
    `config.GT_MODEL`/`config.GT_LORA_RANK`/`config.SEED`/`config.DATA_SEED`
    -- the same cache file Part 1's checkpoint training and Part 2's sweep
    both hit (`trainfeat_Qwen_Qwen3-4B_<n_a>x<n_p>_r16_s0_eval400x400.pt`), so
    training features are computed exactly once regardless of which part
    asks first.
    """
    return train_features(splits, preset_cfg, cfg.GT_MODEL, cfg.GT_LORA_RANK, cfg.SEED,
                          data_seed=cfg.DATA_SEED)
