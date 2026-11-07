#!/usr/bin/env python3
"""Run InfluCoder as a DATE-LM application-task attribution method.

Parallels `methods/dattri.py`'s `attribute()` (same config format, same
train/ref dataset loading, same output convention: a JSON file at
`save_path/InfluCoder.pt` that `evaluation/evaluate_application.py` reads
unchanged), but InfluCoder isn't a `dattri` BaseAttributor -- it doesn't
score examples from per-sample gradients directly, it distills that
(expensive) signal into a cheap sentence encoder first:

  1. Compute projected per-sample gradient features of the *checkpoint's own
     trained LoRA parameters* -- the same gradient signal Grad_Sim/LESS/
     DataInf/EKFAC score from -- for the FULL reference set (small: tens of
     examples) and a SUBSET of the training set (`n_teacher_train`; the
     expensive part, one backward pass per example, so bounded).
  2. Train a small sentence encoder (default: jhu-clsp/ettin-encoder-68m) so
     that its embedding cosine reproduces that gradient-cosine matrix, via
     `influcoder.encoder.distill` -- unchanged from the parent repo's package.
  3. Embed the FULL train and ref sets with the trained encoder (a forward
     pass per example, not a backward pass -- this is the whole point) and
     take cosine similarity as the attribution score.

An optional held-out slice of training examples (`n_eval_train`, disjoint
from the distillation subset) gets its own gradient features computed too,
purely to report a Spearman fidelity number during training (how well the
distilled encoder's ranking matches the true gradient-cosine ranking on
examples it never saw) -- this does not affect the saved scores, which
always come from embedding the full train/ref sets in step 3.

Usage:
    python methods/influcoder_attribute.py --config configs/toxicity-bias-influcoder.yaml
    python methods/influcoder_attribute.py --config configs/factual-attribution-influcoder.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datamodules.load_data import get_dataset, prepare_chat_format  # noqa: E402

from methods.influcoder import _bootstrap  # noqa: F401,E402  (sys.path side effect, must run first)
from methods.influcoder.teacher_grads import (  # noqa: E402
    compute_gradient_features,
    free_teacher_model,
    load_teacher_model,
    sample_valid_indices,
    tokenize_chat_examples,
)
from influcoder.encoder import distill, embed, load_encoder  # noqa: E402
from influcoder.metrics import spearman_metrics  # noqa: E402

TOXICITY_BIAS_TASK = "Toxicity/Bias"
FACTUAL_TASKS = ("Counterfact", "ftrace")

DEFAULTS = dict(
    encoder_model="jhu-clsp/ettin-encoder-68m",
    encoder_max_len=1024,
    max_len=1024,  # teacher-gradient tokenization length, matches dattri.py's MessageDataset
    proj_dim=8192,  # matches this benchmark's own LESS proj_dim hyperparameter
    n_teacher_train=500,
    n_eval_train=250,
    epochs=8,
    hard_ratio=0.5,
    lr=5e-5,
    seed=0,
    select_best_on="aggregated",
)


def example_text(d: dict) -> str:
    return f"{d['prompt'].strip()}\n{d['response'].strip()}"


def free_encoder(enc):
    del enc
    torch.cuda.empty_cache()


def run(cfg: dict):
    cfg = {**DEFAULTS, **cfg}
    # YAML 1.1's float regex requires a decimal point, so bare-exponent forms
    # like "5e-5" (no ".") silently parse as a str, not a float -- PyYAML's
    # well-known gotcha. Cast defensively rather than relying on every config
    # author remembering to write "5.0e-5".
    cfg["lr"] = float(cfg["lr"])
    device = cfg.get("device", "cuda")
    seed = cfg["seed"]
    random.seed(seed)

    print(f"InfluCoder / DATE-LM | task={cfg['task']} subset={cfg['subset']} "
          f"| base_model={cfg['base_model_path']} checkpoint={cfg['checkpoint']}\n")

    train, ref = get_dataset(cfg["task"], cfg["subset"])
    train_chat = prepare_chat_format(train)
    ref_chat = prepare_chat_format(ref)
    n_train, n_ref = len(train), len(ref)
    print(f"train={n_train} ref={n_ref}\n")

    n_teacher_train = min(cfg["n_teacher_train"], n_train)
    n_eval_train = min(cfg["n_eval_train"], max(0, n_train - n_teacher_train))
    perm = list(range(n_train))
    random.Random(seed).shuffle(perm)

    # -- Step 1: teacher gradient features (checkpoint's own trained LoRA) --
    tokenizer, teacher = load_teacher_model(cfg["base_model_path"], cfg["checkpoint"], device=device)

    # Truncation at max_len can cut off an example's entire assistant span
    # (all -100 labels -> zero gradient row), which happens in practice on
    # this benchmark's longer prompt/response pairs. Sample only from
    # examples with a valid loss span rather than crashing on the first bad
    # one; teacher_idx/eval_idx are drawn disjoint from the same shuffled order.
    ref_idx = sample_valid_indices(ref_chat, tokenizer, cfg["max_len"], list(range(n_ref)), n_ref)
    if len(ref_idx) < n_ref:
        print(f"  [ref] skipped {n_ref - len(ref_idx)}/{n_ref} examples with no valid loss span at max_len={cfg['max_len']}")
    teacher_idx = sample_valid_indices(train_chat, tokenizer, cfg["max_len"], perm, n_teacher_train)
    eval_idx = sample_valid_indices(train_chat, tokenizer, cfg["max_len"], perm, n_eval_train, exclude=set(teacher_idx))

    print(f"########## teacher gradients: ref (n={len(ref_idx)}) ##########")
    ref_toks = tokenize_chat_examples([ref_chat[i] for i in ref_idx], tokenizer, cfg["max_len"])
    g_ref = compute_gradient_features(teacher, ref_toks, proj_dim=cfg["proj_dim"], device=device, desc="ref grads")

    print(f"\n########## teacher gradients: train subset (n={len(teacher_idx)}) ##########")
    teacher_toks = tokenize_chat_examples([train_chat[i] for i in teacher_idx], tokenizer, cfg["max_len"])
    g_train = compute_gradient_features(teacher, teacher_toks, proj_dim=cfg["proj_dim"], device=device,
                                        desc="train-subset grads")
    targets = g_ref @ g_train.T  # [n_ref, n_teacher_train] -- anchors=ref, pool=train subset

    eval_anchor_texts = eval_pool_texts = gt_eval = None
    if eval_idx:
        print(f"\n########## teacher gradients: held-out eval train subset (n={len(eval_idx)}) ##########")
        eval_toks = tokenize_chat_examples([train_chat[i] for i in eval_idx], tokenizer, cfg["max_len"])
        g_eval = compute_gradient_features(teacher, eval_toks, proj_dim=cfg["proj_dim"], device=device,
                                           desc="eval-subset grads")
        gt_eval = (g_ref @ g_eval.T).numpy()
        eval_anchor_texts = [example_text(ref[i]) for i in ref_idx]
        eval_pool_texts = [example_text(train[i]) for i in eval_idx]

    free_teacher_model(teacher)

    # -- Step 2: distill a cheap encoder from the teacher gradient targets --
    print(f"\n########## distilling {cfg['encoder_model']} ({cfg['epochs']} epochs) ##########")
    enc = load_encoder(cfg["encoder_model"], max_seq_len=cfg["encoder_max_len"])
    anchor_texts = [example_text(ref[i]) for i in ref_idx]
    pool_texts = [example_text(train[i]) for i in teacher_idx]

    epoch_eval = None
    if gt_eval is not None:
        def epoch_eval():
            pred = embed(enc, eval_anchor_texts) @ embed(enc, eval_pool_texts).T
            return spearman_metrics(pred, gt_eval)

    log = distill(enc, anchor_texts, pool_texts, targets, epochs=cfg["epochs"],
                 hard_ratio=cfg["hard_ratio"], lr=cfg["lr"], seed=seed,
                 epoch_eval=epoch_eval, select_best_on=cfg["select_best_on"])
    if gt_eval is not None and log["epoch_metrics"]:
        # log["epoch_metrics"][-1] is the LAST epoch, but distill() restores
        # enc to whichever epoch scored best on epoch_eval (log["best_epoch"])
        # -- that restored epoch is what actually generates the saved scores
        # below, so report ITS fidelity, not the (likely more overfit) last
        # epoch's.
        restored = log["epoch_metrics"][log["best_epoch"]]
        print(f"  restored (best-epoch={log['best_epoch'] + 1}/{cfg['epochs']}) fidelity vs held-out "
              f"teacher gradients: agg rho={restored['aggregated']:+.4f} "
              f"per-anchor rho={restored['per_anchor_mean']:+.4f}")

    # -- Step 3: embed the FULL train/ref sets and score --
    print(f"\n########## embedding full train (n={n_train}) + ref (n={n_ref}) ##########")
    t0 = time.perf_counter()
    all_train_emb = embed(enc, [example_text(d) for d in train])
    all_ref_emb = embed(enc, [example_text(d) for d in ref])
    embed_s = time.perf_counter() - t0
    scores = all_train_emb @ all_ref_emb.T  # [n_train, n_ref], matches dattri.py's attributor convention
    print(f"  embed+score wall time: {embed_s:.1f}s")
    free_encoder(enc)

    if cfg["task"] == TOXICITY_BIAS_TASK:
        final_scores = np.mean(scores, axis=1).tolist()
    elif cfg["task"] in FACTUAL_TASKS:
        final_scores = scores.T.tolist()
    else:
        raise NotImplementedError(f"Task '{cfg['task']}' not implemented.")

    save_path = Path(cfg["save_path"]) / "InfluCoder.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(final_scores, f)
    print(f"\nwrote {save_path}")


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run(config)
