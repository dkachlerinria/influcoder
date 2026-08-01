"""EXP1: the ONE canonical parameter set shared by Parts 1, 2, and 3.

Every part-script (`part1.py`/`part2.py`/`part3.py`) imports these values
instead of hardcoding its own local constants. This module exists specifically
because the previous (`.tuning_logs/`, scratch) generation of these scripts
each defined their own copies of GT model/rank, attn, max_len, ranks, and the
InfluCoder recipe -- and drifted (see EXP1.md section 4 for the full audit:
Part 2 silently used lr=1e-5 instead of the 5e-5 everyone else used, Part 1's
checkpoints used best-epoch selection while Part 2 used fixed-epoch-8, etc).
Centralizing these values here makes that class of bug structurally
impossible: change a value once, every part picks it up.

Where a part must genuinely diverge from this canonical set (e.g. Part 3 only
timing LESS-4B and LoGRA-1.7B, not the full proxy families), that's a
*scope* difference, not a parameter one -- the part still pulls its rank/attn/
max_len from here.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Ground truth / teacher model -- used for BOTH the eval GT (Parts 1, 2) and
# InfluCoder's distillation targets (Parts 1, 2, 3).
# --------------------------------------------------------------------------- #
GT_MODEL = "Qwen/Qwen3-4B"
GT_LORA_RANK = 16
SEED = 0

# Which shuffle produces the eval/train PARTITION itself (which BBH/pool
# samples land in eval vs. train) -- distinct from SEED above, which only
# controls LoRA/model randomness on top of a FIXED partition. None means "use
# the historical fixed split" (baselines.common.DATA_SEED, 42) -- every
# existing cache file/checkpoint stays valid. A multi-seed sweep wanting
# variance from data selection too (not just model/training stochasticity)
# sets this per seed, e.g. in a seed-specific config module.
DATA_SEED = None

# --------------------------------------------------------------------------- #
# Data / preset
# --------------------------------------------------------------------------- #
PRESET = "fig1_dolci"  # BBH anchors x tasksource/dolci-instruct pool

# Which config profile is active -- every output path (checkpoints, JSON
# results) is namespaced under this, specifically so switching EXP1_CONFIG
# can never silently reuse-or-skip an artifact produced under a DIFFERENT
# parameter set (rank, restore_best, etc.) at the same cfg.PRESET path.
# Previously all three parts' outputs lived at paths keyed only by PRESET --
# identical whether EXP1_CONFIG was set or not, so e.g. Part 1's checkpoint-
# reuse check or Part 2's resume-skip logic could silently treat a
# historical-config artifact as already-done under biggpu. Real risk, not
# hypothetical -- see EXP1_BIGGPU_FINAL.md's code-review pass.
PROFILE = "default"


def seed_dir(seed: int) -> str:
    """Path segment distinguishing a non-default SEED run, for a multi-seed
    sweep's checkpoints/outputs -- same rationale as PROFILE above, one level
    down. Empty for the canonical seed 0, so nothing about the
    already-established seed-0 paths changes (joining a Path with "" is a
    documented no-op); "seedN" otherwise, so multiple seeds' artifacts can
    never silently collide or reuse-skip each other the way un-namespaced
    PROFILE paths used to. Takes `seed` explicitly (not read from a module
    global) so it gives the right answer regardless of which config module's
    SEED is actually active."""
    return "" if seed == 0 else f"seed{seed}"


# Canonical eval size. Both Part 1 and Part 2 default to this (the full
# fig1_dolci eval) so their numbers are comparable; pass a smaller --n_eval on
# either part's CLI for a fast peek, but the DEFAULT is now the same for both
# parts (previously Part 1 defaulted to a 50x50 peek and Part 2 to the full
# 400x400 -- the single biggest source of cross-part number mismatches, see
# EXP1.md section 4.2.1).
N_EVAL = 400

# The "anchor point": Part 1's real, saved InfluCoder checkpoints are trained
# at this train-set size. Part 2's sweep MUST include this exact size (see
# part2.py's SIZES list) so that point can be read as "the same point in the
# graph as Part 1" per explicit instruction, rather than just a nearby size.
N_TRAIN_A = 1500
N_TRAIN_P = 3000  # 1:2 anchor:pool ratio, held throughout Part 2's sweep too

# --------------------------------------------------------------------------- #
# Fairness constraints shared by every method that has a model
# --------------------------------------------------------------------------- #
ATTN = "sdpa"      # NOT "eager" -- see EXP1.md section 3 for why (this is a
                   # timing/quality comparison, never wrapped in
                   # baselines.cost.flop_counter(), so the eager-for-FLOP-
                   # counting reason baked into several score_* defaults does
                   # not apply here; sdpa is also faster/lower-memory)
MAX_LEN = 1024     # grad_max_len == encoder_max_len in fig1_dolci; every
                   # method gets this same sequence-length cap

# --------------------------------------------------------------------------- #
# LESS
# --------------------------------------------------------------------------- #
LESS_RANK = 16              # uniform across LESS's 3 model sizes (NOT the
                           # paper default of 128 -- see EXP1.md section 10.1
                           # OOM saga for why)
LESS_PROJ_DIM = 8192
LESS_LORA_ALPHA = 512
LESS_LORA_DROPOUT = 0.0     # was 0.1 with the model left in train() mode --
                           # injected dropout noise into every LESS gradient,
                           # unlike GT/LoGRA which are both deterministic
                           # (eval mode, dropout=0); see EXP1.md's fix writeup
LESS_BLOCK_SIZE = 16        # NOT the collect_grads default of 128 -- also part
                           # of the OOM fix; MUST be passed explicitly at every
                           # call site, it is not a shared library default
LESS_GRADIENT_CHECKPOINTING = False  # sdpa makes this unnecessary (section 10.1)
LESS_PROJECT_INTERVAL = 8

LESS_MODEL_SIZES = {
    "4B": GT_MODEL,
    "1.7B": "Qwen/Qwen3-1.7B",
    "0.6B": "Qwen/Qwen3-0.6B",
}

# --------------------------------------------------------------------------- #
# LoGRA
# --------------------------------------------------------------------------- #
LOGRA_RANK = 8              # uniform across LoGRA's 3 model sizes (NOT the
                           # asymmetric r8(4B)/r32(proxies) scheme the OLDER
                           # figure1_table.py uses -- explicit user instruction
                           # this experiment, see EXP1.md section 3)
LOGRA_MLP_ONLY = True
LOGRA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
LOGRA_BIG_GPU = False       # score_logra's length-sorted-batching opt-in (see
                           # EXP1.md section 5.6) -- off here to preserve this
                           # config's original bs=1 numbers exactly; the
                           # BIG_GPU_FINAL config turns this on
LOGRA_COMPUTE_FIM = False   # score_logra computes an extra FIM-preconditioned
                           # score by default (compute_fim=True) that costs
                           # ~16x more at higher rank and that NOTHING in this
                           # repo actually uses -- run_logra() only ever reads
                           # variants["logra_raw"] (FINDINGS.md: "raw beats FIM
                           # everywhere"). False here so every ms/sample number
                           # this package reports reflects only the score
                           # that's actually used, not wasted computation.

LOGRA_MODEL_SIZES = {
    "4B": GT_MODEL,
    "1.7B": "Qwen/Qwen3-1.7B",
    "0.6B": "Qwen/Qwen3-0.6B",
}

# --------------------------------------------------------------------------- #
# InfluCoder -- the canonical distillation recipe, used identically by the
# checkpoints Part 1 scores, the sweep Part 2 runs, and the exploratory
# training Part 3 times.
# --------------------------------------------------------------------------- #
ENCODER_MODELS = {
    "68m": "jhu-clsp/ettin-encoder-68m",
    "150m": "jhu-clsp/ettin-encoder-150m",
}
INFLUCODER_PROJ_DIM = 65536   # teacher-gradient sketch dim (fig1_dolci preset)
INFLUCODER_EPOCHS = 8
INFLUCODER_HARD_RATIO = 0.0
INFLUCODER_LR = 5e-5          # distill()'s own default -- Part 2's scratch
                              # version silently overrode this to 1e-5 with no
                              # recorded rationale; canonicalized back to the
                              # recipe default here (see EXP1.md section 4.2.3)
INFLUCODER_SEED = 0
INFLUCODER_ENCODER_MAX_LEN = MAX_LEN

# Epoch-selection convention: report the FINAL epoch's metrics, not whichever
# epoch scored best on eval. This is Part 2's original (user-approved)
# correction -- "dont keep the best epoch, keep the same epoch (8)" -- now
# adopted as the canonical convention everywhere a quality number is reported,
# so Part 1's checkpoints and Part 2's sweep use the identical methodology
# (previously Part 1's checkpoints used best-epoch restore via
# train_fig1_encoders.py, a real, previously-undocumented-until-now
# discrepancy -- see EXP1.md section 4.2.4). `select_best_on` is still passed
# to distill() (it only controls what gets restored into the returned encoder
# object) but the reported metric is always epoch_metrics[-1].
INFLUCODER_SELECT_BEST_ON = "aggregated"

# Whether distill() actually RESTORES enc to that best epoch afterwards (True
# here, preserving this config's original behavior: the reported metric was
# already always epoch_metrics[-1], but the saved/returned `enc` itself was
# still silently restored to best-epoch underneath that). BIG_GPU_FINAL sets
# this False so the saved checkpoint's actual weights match the epoch-8
# metric being reported -- see distill()'s docstring in influcoder/encoder.py.
INFLUCODER_RESTORE_BEST_EPOCH = True

# --------------------------------------------------------------------------- #
# Part 3 scope (EXP1.md section 4.2.2/4.2.6): a deliberate, explicit scope
# reduction from the full LESS_MODEL_SIZES/LOGRA_MODEL_SIZES families and
# N_TRAIN_A/N_TRAIN_P above, NOT an oversight -- LESS/LoGRA are "waaaay too
# expensive" to time across every proxy size, and this part only needs SOME
# InfluCoder training-set size to measure setup-cost amortization, not the
# real one. Kept here (not hardcoded in part3.py) so BIG_GPU_FINAL can widen
# both without part3.py's code changing at all.
# --------------------------------------------------------------------------- #
PART3_LESS_MODELS = {"1.7B": "Qwen/Qwen3-1.7B"}
PART3_LOGRA_MODELS = {"1.7B": "Qwen/Qwen3-1.7B"}
PART3_N_TRAIN_A = 250
PART3_N_TRAIN_P = 500
