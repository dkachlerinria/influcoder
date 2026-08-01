"""EXP1: canonical per-method wrappers -- every score_* call goes through here
so the rank/attn/max_len/block_size that gets passed can never quietly drift
from `config.py`'s single source of truth.

Part 1 uses the `run_*` functions directly (full anchor-vs-pool scoring).
Part 3 uses `load_less_model`/`load_logra_model`/`load_rdsplus_model` (the
same model-loading calls `run_less`/`run_logra`/`run_rdsplus` make
internally) to get identically-configured models for its process-only
timing, without duplicating the rank/attn/block_size constants at a second
call site.

`less_fingerprint`/`logra_fingerprint` + `score_less_cached`/
`score_logra_cached` implement cross-part score reuse: Part 1 already scores
every LESS/LoGRA model size at the full eval; Part 2's reference lines need
those exact same scores (same model, same eval slice, same rank/seed/etc) --
recomputing them is pure waste. See those functions' docstrings for the
matching rule.
"""
from __future__ import annotations

import json
from pathlib import Path

from baselines.less.model_utils import load_base_with_fresh_lora
from baselines.less.score import score_less
from baselines.logra.modeling_logra import LoGra
from baselines.logra.score import score_logra
from baselines.influcoder.score import score_influcoder
from baselines.rdsplus.score import score_rdsplus
from baselines.semantic.score import score_semantic
from baselines.tfidf.score import score_tfidf

import os as _os
if _os.environ.get("EXP1_CONFIG") == "biggpu":
    from . import config_biggpu as cfg  # BIG_GPU_FINAL_EXP1 -- see that module's docstring
else:
    from . import config as cfg


# --------------------------------------------------------------------------- #
# Full anchor-vs-pool scoring (Part 1; Part 2 uses train.py + these are not
# re-run since Part 2's own quality signal comes from its trained encoder).
# --------------------------------------------------------------------------- #
def run_less(splits, model_name: str, meter=None):
    return score_less(
        splits, model_name, proj_dim=cfg.LESS_PROJ_DIM, max_len=cfg.MAX_LEN,
        lora_rank=cfg.LESS_RANK, lora_alpha=cfg.LESS_LORA_ALPHA,
        lora_dropout=cfg.LESS_LORA_DROPOUT, lora_seed=cfg.SEED,
        project_interval=cfg.LESS_PROJECT_INTERVAL,
        block_size=cfg.LESS_BLOCK_SIZE,
        gradient_checkpointing=cfg.LESS_GRADIENT_CHECKPOINTING,
        attn_implementation=cfg.ATTN, meter=meter)


def run_logra(splits, model_name: str, meter=None):
    variants = score_logra(
        splits, model_name, lora_rank=cfg.LOGRA_RANK, mlp_only=cfg.LOGRA_MLP_ONLY,
        max_len=cfg.MAX_LEN, target_modules=cfg.LOGRA_TARGET_MODULES,
        seed=cfg.SEED, attn_implementation=cfg.ATTN, big_gpu=cfg.LOGRA_BIG_GPU,
        compute_fim=cfg.LOGRA_COMPUTE_FIM, meter=meter)
    return variants["logra_raw"]


def run_influcoder(splits, encoder_dir: str, meter=None):
    return score_influcoder(splits, encoder_dir, max_len=cfg.MAX_LEN,
                            attn_implementation=cfg.ATTN, meter=meter)


def run_untrained(splits, model_name: str, meter=None):
    return score_semantic(splits, model_name, max_len=cfg.MAX_LEN,
                          attn_implementation=cfg.ATTN, meter=meter)


def run_rdsplus(splits, meter=None):
    return score_rdsplus(splits, cfg.GT_MODEL, max_len=cfg.MAX_LEN,
                         attn_implementation=cfg.ATTN, meter=meter)


def run_tfidf(splits, meter=None):
    return score_tfidf(splits, meter=meter)


# --------------------------------------------------------------------------- #
# Model-loading only, for Part 3's process-only timing. Identical
# rank/attn/block_size to run_less/run_logra above -- deliberately factored
# out rather than duplicated, since this exact duplication (a new script
# re-hardcoding block_size=128 instead of the fixed 16) is what caused the
# Part 3 OOM regression documented in EXP1.md section 7.3.
# --------------------------------------------------------------------------- #
def load_less_model(model_name: str):
    """(tokenizer, model) for `model_name` under LESS's canonical config."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_base_with_fresh_lora(
        model_name=model_name, tokenizer=tok, lora_target_modules="all-linear",
        lora_rank=cfg.LESS_RANK, lora_alpha=cfg.LESS_LORA_ALPHA,
        lora_dropout=cfg.LESS_LORA_DROPOUT, seed=cfg.SEED,
        gradient_checkpointing=cfg.LESS_GRADIENT_CHECKPOINTING,
        attn_implementation=cfg.ATTN)
    return tok, model


def load_logra_model(model_name: str) -> LoGra:
    """A `LoGra` instance for `model_name` under LoGRA's canonical config."""
    logra = LoGra.from_pretrained(
        model_name=model_name, rank=cfg.LOGRA_RANK, mlp_only=cfg.LOGRA_MLP_ONLY,
        target_modules=cfg.LOGRA_TARGET_MODULES, attn_implementation=cfg.ATTN)
    if logra.tokenizer.pad_token is None:
        logra.tokenizer.pad_token = logra.tokenizer.eos_token
    return logra


def load_rdsplus_model(model_name: str):
    """(tokenizer, model) for `model_name` under RDS+'s canonical config --
    raw eval-mode forward pass, no LoRA (RDS+ scores the target model's own
    representations directly, see baselines/rdsplus/score.py)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation=cfg.ATTN)
    model.to("cuda").eval()
    return tok, model


# --------------------------------------------------------------------------- #
# Cross-part score reuse (Part 2's reference lines reusing Part 1's rows).
#
# Every value listed here can change the actual returned score. `big_gpu` and
# `compute_fim` are deliberately NOT included: `big_gpu` only changes batching
# strategy (verified-safe, same numbers, see EXP1_BIGGPU_FINAL.md), and
# `run_logra` always reads `variants["logra_raw"]` regardless of `compute_fim`
# -- so neither one ever changes what gets returned, only speed/memory.
#
# `part1.py` writes these exact dicts into its output's "config" section (so
# there's one source of truth for "what a LESS/LoGRA row's config actually
# was" -- part1.py can never write a fingerprint that this function doesn't
# then recognize), and any part.py wanting to reuse a Part 1 row instead of
# recomputing it compares against them here.
# --------------------------------------------------------------------------- #
def less_fingerprint(n_eval: int) -> dict:
    return {
        "n_eval": n_eval, "preset": cfg.PRESET, "gt_model": cfg.GT_MODEL,
        "gt_lora_rank": cfg.GT_LORA_RANK, "attn": cfg.ATTN, "max_len": cfg.MAX_LEN,
        "less_rank": cfg.LESS_RANK, "less_proj_dim": cfg.LESS_PROJ_DIM,
        "less_lora_alpha": cfg.LESS_LORA_ALPHA, "less_lora_dropout": cfg.LESS_LORA_DROPOUT,
        "less_project_interval": cfg.LESS_PROJECT_INTERVAL,
        "less_block_size": cfg.LESS_BLOCK_SIZE, "seed": cfg.SEED,
    }


def logra_fingerprint(n_eval: int) -> dict:
    return {
        "n_eval": n_eval, "preset": cfg.PRESET, "gt_model": cfg.GT_MODEL,
        "gt_lora_rank": cfg.GT_LORA_RANK, "attn": cfg.ATTN, "max_len": cfg.MAX_LEN,
        "logra_rank": cfg.LOGRA_RANK, "logra_mlp_only": cfg.LOGRA_MLP_ONLY,
        "logra_target_modules": cfg.LOGRA_TARGET_MODULES, "seed": cfg.SEED,
    }


def _part1_cached_aggregated(family: str, label: str, n_eval: int):
    """Return Part 1's `aggregated` score for `label` (e.g. "less_1.7B") if
    Part 1's output exists, was written under a config matching
    `less_fingerprint`/`logra_fingerprint` for the CURRENT config/n_eval, and
    actually scored `label`. Returns None on any mismatch -- missing file,
    different config, or the label just isn't in there -- so the caller can
    fall back to scoring it fresh exactly as before this existed."""
    part1_path = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "exp1_part1.json"
    if not part1_path.exists():
        return None
    try:
        p1 = json.loads(part1_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    fp = (less_fingerprint if family == "less" else logra_fingerprint)(n_eval)
    p1_cfg = p1.get("config", {})
    if any(p1_cfg.get(k) != v for k, v in fp.items()):
        return None
    row = p1.get("methods", {}).get(label)
    return None if row is None else row["aggregated"]


def score_less_cached(splits, model_name: str, label: str, gt, n_eval: int, meter=None):
    """`(aggregated_rho, reused)` for `label` (e.g. "less_4B") -- reuses Part
    1's row when its config matches (see `_part1_cached_aggregated`),
    otherwise runs `run_less` + scores it fresh, same as callers previously
    did unconditionally."""
    cached = _part1_cached_aggregated("less", label, n_eval)
    if cached is not None:
        return cached, True
    from influcoder.metrics import spearman_metrics
    scores = run_less(splits, model_name, meter=meter)
    return spearman_metrics(scores.numpy(), gt)["aggregated"], False


def score_logra_cached(splits, model_name: str, label: str, gt, n_eval: int, meter=None):
    """`(aggregated_rho, reused)` for `label` (e.g. "logra_1.7B") -- reuses
    Part 1's row when its config matches (see `_part1_cached_aggregated`),
    otherwise runs `run_logra` + scores it fresh, same as callers previously
    did unconditionally."""
    cached = _part1_cached_aggregated("logra", label, n_eval)
    if cached is not None:
        return cached, True
    from influcoder.metrics import spearman_metrics
    scores = run_logra(splits, model_name, meter=meter)
    return spearman_metrics(scores.numpy(), gt)["aggregated"], False
