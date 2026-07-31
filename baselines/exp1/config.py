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

# --------------------------------------------------------------------------- #
# Data / preset
# --------------------------------------------------------------------------- #
PRESET = "fig1_dolci"  # BBH anchors x tasksource/dolci-instruct pool

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
