"""EXP1: canonical per-method wrappers -- every score_* call goes through here
so the rank/attn/max_len/block_size that gets passed can never quietly drift
from `config.py`'s single source of truth.

Part 1 uses the `run_*` functions directly (full anchor-vs-pool scoring).
Part 3 uses `load_less_model`/`load_logra_model` (the same model-loading calls
`run_less`/`run_logra` make internally) to get identically-configured models
for its process-only timing, without duplicating the rank/attn/block_size
constants at a second call site.
"""
from __future__ import annotations

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
        lora_dropout=cfg.LESS_LORA_DROPOUT, project_interval=cfg.LESS_PROJECT_INTERVAL,
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
