"""EXP1: the ONE InfluCoder training call, used by Parts 1, 2, and 3 alike.

Part 1 calls this once at `config.N_TRAIN_A`/`N_TRAIN_P` and saves the result
to disk (the real, scored checkpoint). Part 2 calls it once per swept size,
never saving. Part 3 calls it once at a small exploratory size, timing it
rather than scoring it. All three get the identical epochs/hard_ratio/lr/seed/
epoch-selection convention because all three go through this one function --
see config.py's docstring for why that wasn't true of the previous generation
of these scripts.
"""
from __future__ import annotations

import torch

from influcoder.encoder import distill, embed, load_encoder
from influcoder.metrics import spearman_metrics

import os as _os
if _os.environ.get("EXP1_CONFIG") == "biggpu":
    from . import config_biggpu as cfg  # BIG_GPU_FINAL_EXP1 -- see that module's docstring
else:
    from . import config as cfg


def train_influcoder(encoder_model: str, train_anchor_texts: list[str],
                     train_pool_texts: list[str], targets, *,
                     eval_anchor_texts: list[str] | None = None,
                     eval_pool_texts: list[str] | None = None, gt=None,
                     epochs: int | None = None,
                     hard_ratio: float | None = None, lr: float | None = None,
                     seed: int | None = None, encoder_max_len: int | None = None,
                     select_best_on: str | None = None,
                     restore_best: bool | None = None):
    """Load a fresh encoder and distill it against `targets`.

    Every keyword defaults to the canonical value in `config.py` -- callers
    only need to pass the training-set size (implicit in how many texts/
    targets are given) and, if they want a quality signal, `eval_anchor_texts`/
    `eval_pool_texts`/`gt`. Part 3 passes none of the eval_* args (it never
    scores quality, only times wall-clock), so no `epoch_eval` callback is
    built at all -- distill() just trains blind.

    The eval closure is built HERE, not by the caller, specifically because it
    needs to close over `enc` -- and `enc` doesn't exist until this function
    creates it. An earlier version of this function accepted a pre-built
    `epoch_eval` callback from the caller instead; that closure could only
    reference an `enc` that didn't exist yet in the caller's own scope, which
    would have raised NameError the moment distill() actually invoked it. Keep
    eval-closure construction in here.

    Returns (enc, log, final_metrics, untrained_metrics). `final_metrics` is
    always `log["epoch_metrics"][epochs - 1]` -- the LAST epoch's metrics, per
    the canonical epoch-selection convention. Whether `enc` itself (and
    therefore any checkpoint saved from it) also ends up at that same final
    epoch, rather than being silently restored to a different "best" epoch
    underneath the reported metric, is controlled by `restore_best`
    (defaults to `config.INFLUCODER_RESTORE_BEST_EPOCH` -- True preserves this
    config's original behavior, BIG_GPU_FINAL sets it False so the saved
    weights actually match the epoch-8 number being reported). `select_best_on`
    only picks WHICH per-epoch metric counts as "best" when restore_best=True;
    it's a no-op when restore_best=False. `untrained_metrics` is the same
    eval computed once before training starts (both are None if no eval_*
    args were given).
    """
    epochs = cfg.INFLUCODER_EPOCHS if epochs is None else epochs
    hard_ratio = cfg.INFLUCODER_HARD_RATIO if hard_ratio is None else hard_ratio
    lr = cfg.INFLUCODER_LR if lr is None else lr
    seed = cfg.INFLUCODER_SEED if seed is None else seed
    encoder_max_len = cfg.INFLUCODER_ENCODER_MAX_LEN if encoder_max_len is None else encoder_max_len
    select_best_on = cfg.INFLUCODER_SELECT_BEST_ON if select_best_on is None else select_best_on
    restore_best = cfg.INFLUCODER_RESTORE_BEST_EPOCH if restore_best is None else restore_best

    enc = load_encoder(encoder_model, max_seq_len=encoder_max_len)

    epoch_eval = None
    has_eval = eval_anchor_texts is not None and eval_pool_texts is not None and gt is not None
    if has_eval:
        def epoch_eval():
            return spearman_metrics(embed(enc, eval_anchor_texts) @ embed(enc, eval_pool_texts).T, gt)

    untrained_metrics = epoch_eval() if has_eval else None
    log = distill(enc, train_anchor_texts, train_pool_texts, targets,
                 epochs=epochs, hard_ratio=hard_ratio, lr=lr, seed=seed,
                 epoch_eval=epoch_eval, select_best_on=select_best_on,
                 restore_best=restore_best)
    final_metrics = log["epoch_metrics"][epochs - 1] if has_eval else None
    return enc, log, final_metrics, untrained_metrics


def free(enc):
    """Release an encoder's GPU memory between sweep points."""
    del enc
    torch.cuda.empty_cache()
