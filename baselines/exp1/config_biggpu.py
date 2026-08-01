"""BIG_GPU_FINAL_EXP1: the final-run parameter set for Parts 1, 2, and 3,
sized for a big GPU (developed against an H200 NVL, 143GB) instead of the
24GB/98GB cards `config.py`'s values were originally constrained by.

Import this INSTEAD of `config.py` by setting `EXP1_CONFIG=biggpu` in the
environment before running any part -- every consumer (`data.py`,
`methods.py`, `train.py`, `part1.py`/`part2.py`/`part3.py`) already does
`if os.environ.get("EXP1_CONFIG") == "biggpu": from . import config_biggpu as cfg`,
so nothing else needs to change:

    EXP1_CONFIG=biggpu python -m baselines.exp1.part1
    EXP1_CONFIG=biggpu python -m baselines.exp1.part2
    EXP1_CONFIG=biggpu python -m baselines.exp1.part3

This module inherits every value from `config.py` unchanged (`from .config
import *`) and then overrides ONLY the ones that genuinely change for the
big-GPU final run. See EXP1_BIGGPU_FINAL.md for the full rationale behind
each override -- summary here, detail there:

1. LESS_RANK: 16 -> 32 (matches LOGRA_RANK below, rather than the paper's 128
   default -- explicit instruction this session, keeps both gradient methods
   at the same rank and avoids running an untested rank=128 on this GPU with
   no prior data point anywhere in this repo).
2. LOGRA_RANK: 8 -> 32 (still UNIFORM across all 3 model sizes -- this stays a
   deliberate choice, isolating model size as the only confound, per
   config.py's own original rationale). Raised because uniform r8 was shown
   to rank-starve the 0.6B proxy (EXP1.md section 5.6: -0.13 at r8 vs +0.29 at
   r32) -- r8 was an OOM-driven compromise for the SAME reason LESS_RANK was,
   not a considered choice either.
3. LOGRA_BIG_GPU: False -> True. Verified-safe, real (not approximate)
   speedup (EXP1.md section 5.6) -- no reason not to use it for the final run.
4. INFLUCODER_RESTORE_BEST_EPOCH: True -> False. Explicit instruction this
   session: "we dont do best-epoch selection we just do 8 epochs." Every
   checkpoint Part 1 saves, every point Part 2 sweeps, and Part 3's
   exploratory training all now keep whatever the encoder looks like after
   the full 8-epoch budget -- no restoration to an earlier "best" epoch
   anywhere, matching what was already true of the REPORTED metric (which
   was always epoch_metrics[-1]) but previously wasn't true of the actual
   saved/returned encoder weights underneath that number.
5. PART3_N_TRAIN_A / PART3_N_TRAIN_P: widened from the deliberately-tiny
   250x500 exploratory set (EXP1.md section 4.2.2) to N_TRAIN_A/N_TRAIN_P
   (1500/3000) -- the SAME real training-set size Part 1's checkpoints and
   Part 2's largest sweep point use. This was already flagged as an open item
   in EXP1.md section 11 ("the amortization numbers should be re-measured at
   whatever the real experiment's actual InfluCoder training-set size ends up
   being") -- this run closes it.
6. N_EVAL: 400 -> 800. Explicit instruction: "full clean run at 800x800".
   `fig1_dolci`'s preset ceiling (run.py's PRESETS dict) was raised to match
   -- `data.load_gt_and_splits(n_eval=...)` can only slice DOWN from the
   cached full preset eval, never up, so N_EVAL here can never exceed that
   ceiling. Overridden here (not in config.py) since 800x800 is this profile's
   final-run size, not the smaller-GPU default's.

Part 3's model scope (`PART3_LESS_MODELS`/`PART3_LOGRA_MODELS`) is NOT widened
here, unlike an earlier draft of this file -- explicit instruction: start with
just LESS-4B and LoGRA-1.7B (config.py's existing scope), add more sizes
later once these are confirmed working. So those two stay inherited from
config.py unchanged, same as everything below.

Everything else (GT model/rank, N_TRAIN_A/N_TRAIN_P, ATTN, MAX_LEN,
LESS_PROJ_DIM/LORA_ALPHA/DROPOUT/BLOCK_SIZE/GRADIENT_CHECKPOINTING/
PROJECT_INTERVAL, LOGRA_MLP_ONLY/TARGET_MODULES, ENCODER_MODELS,
INFLUCODER_PROJ_DIM/EPOCHS/HARD_RATIO/LR/SEED/ENCODER_MAX_LEN, PART3_LESS_MODELS,
PART3_LOGRA_MODELS) is inherited from `config.py` unchanged -- listed in full
in EXP1_BIGGPU_FINAL.md's parameter table, not repeated here as a second
source of truth.
"""
from __future__ import annotations

from .config import *  # noqa: F401,F403 -- inherit everything, override below
from .config import N_TRAIN_A, N_TRAIN_P

# --------------------------------------------------------------------------- #
# 0. Output-path namespace -- see config.py's PROFILE docstring. Every
# checkpoint/JSON this profile writes lives under "biggpu", never colliding
# with a "default"-profile artifact at the same nominal path.
# --------------------------------------------------------------------------- #
PROFILE = "biggpu"

# --------------------------------------------------------------------------- #
# 1. LESS -- rank raised to match LoGRA's (not the paper's 128 default)
# --------------------------------------------------------------------------- #
LESS_RANK = 32   # was 16 (config.py) -- see module docstring point 1.

# --------------------------------------------------------------------------- #
# 2. LoGRA -- higher uniform rank (fixes 0.6B rank-starvation), batching on
# --------------------------------------------------------------------------- #
LOGRA_RANK = 32       # was 8 (config.py) -- see module docstring point 2.
LOGRA_BIG_GPU = True  # was False (config.py) -- see module docstring point 3.

# --------------------------------------------------------------------------- #
# 4. InfluCoder -- no best-epoch restoration anywhere, just 8 epochs, period
# --------------------------------------------------------------------------- #
INFLUCODER_RESTORE_BEST_EPOCH = False   # was True (config.py) -- see point 4.

# --------------------------------------------------------------------------- #
# 5. Part 3 -- real InfluCoder training-set size (model scope left as-is,
# see module docstring -- just LESS-4B/LoGRA-1.7B for now, matching config.py)
# --------------------------------------------------------------------------- #
PART3_N_TRAIN_A = N_TRAIN_A              # was 250 (config.py)
PART3_N_TRAIN_P = N_TRAIN_P              # was 500 (config.py)

# --------------------------------------------------------------------------- #
# 6. Eval size -- the final-run 800x800 (see module docstring point 6)
# --------------------------------------------------------------------------- #
N_EVAL = 800   # was 400 (config.py)
