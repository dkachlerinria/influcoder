# EXP1 BIG_GPU_FINAL: the final Figure 1 run

Status: parameters + code wiring done, NOT YET EXECUTED. Written for review before
launching any of Parts 1/2/3. Read `EXP1.md` first if you haven't — this doc only covers
what's NEW/DIFFERENT for the big-GPU final run; it assumes you already know the
methodology `EXP1.md` documents in full (what each part measures, the historical
parameter drift it fixed, the OOM sagas, etc).

## 1. What this is

A single, unified parameter set — `baselines/exp1/config_biggpu.py` — that Parts 1, 2, and
3 all import identically, sized for a big GPU (developed against an **NVIDIA H200 NVL,
143GB**, Grid5000 `lille`/`chicoree`) instead of the 24GB/98GB cards `config.py`'s values
were constrained by. This is meant to be the **final, reported** run — not another draft.

**The one constraint that matters most:** all three parts must use the *same* setup, so
they stay comparable to each other. This was already `config.py`'s whole reason for
existing (see its own docstring); `config_biggpu.py` extends that same discipline to the
big-GPU values rather than introducing a second, parallel source of truth.

## 2. How to select it

Nothing in `part1.py`/`part2.py`/`part3.py` changed structurally — they still all do
`from . import config as cfg` (now wrapped in an `EXP1_CONFIG` env-var check). Set the
env var to switch every part, and everything they import from (`data.py`, `methods.py`,
`train.py`), onto the big-GPU profile at once:

```bash
export EXP1_CONFIG=biggpu     # unset (or any other value) = the historical config.py
python -m baselines.exp1.part1
python -m baselines.exp1.part2
python -m baselines.exp1.part3
```

No part-script needed a single parameter hardcoded anywhere for this — that was the
point of the `baselines/exp1/` package already existing. What DID need code changes
(below, §4) were two things the existing package couldn't express yet: LoGRA batching
was never wired into it (only ever used from `.tuning_logs/` scratch scripts), and
"no best-epoch restoration" was only true of the *reported metric*, not the actual
saved/returned encoder weights underneath it.

## 3. Full parameter table

| Parameter | `config.py` (historical) | `config_biggpu.py` (final) | Changed? |
|---|---|---|---|
| GT model / rank | `Qwen/Qwen3-4B`, r16 | *(same)* | no |
| Preset | `fig1_dolci` | *(same)* | no |
| `N_EVAL` | 400 | *(same)* | no |
| `N_TRAIN_A` / `N_TRAIN_P` | 1500 / 3000 | *(same)* | no |
| `ATTN` | `sdpa` | *(same)* | no |
| `MAX_LEN` | 1024 | *(same)* | no |
| `LESS_RANK` | 16 | **32** (matches `LOGRA_RANK`, not the paper's 128 default) | **yes** — 16 was an OOM compromise on a 24GB card, not a methodology choice (EXP1.md §10.1); 32 keeps both gradient methods at the same rank and avoids an untested rank=128 |
| `LESS_PROJ_DIM` / `LORA_ALPHA` | 8192 / 512 | *(same)* | no |
| `LESS_LORA_DROPOUT` | 0.0 | *(same)* | no |
| `LESS_BLOCK_SIZE` | 16 | *(same, unverified at r128 — see §5)* | no (yet) |
| `LESS_GRADIENT_CHECKPOINTING` | False | *(same)* | no |
| `LOGRA_RANK` | 8 (uniform) | **32** (still uniform) | **yes** — r8 rank-starved the 0.6B proxy (EXP1.md §5.6: -0.13 at r8 → +0.29 at r32); staying *uniform* (not the asymmetric r8/r32 scheme) preserves the original "isolate model size, not rank" intent |
| `LOGRA_BIG_GPU` | False | **True** | **yes** — verified-safe, real (not approximate) speedup, no reason to leave off for the final run |
| `LOGRA_MLP_ONLY` / `TARGET_MODULES` | as before | *(same)* | no |
| `ENCODER_MODELS` | 68m, 150m | *(same)* | no |
| `INFLUCODER_PROJ_DIM` | 65536 | *(same)* | no |
| `INFLUCODER_EPOCHS` | 8 | *(same)* | no |
| `INFLUCODER_HARD_RATIO` | 0.0 | *(same)* | no |
| `INFLUCODER_LR` | 5e-5 | *(same)* | no |
| `INFLUCODER_SELECT_BEST_ON` | `"aggregated"` | *(same — moot, see next row)* | no |
| `INFLUCODER_RESTORE_BEST_EPOCH` | True | **False** | **yes** — explicit instruction: no best-epoch selection anywhere, just report/keep whatever 8 full epochs produces |
| `PART3_LESS_MODELS` | `{"4B": ...}` only | *(same — start narrow, widen later)* | no |
| `PART3_LOGRA_MODELS` | `{"1.7B": ...}` only | *(same — start narrow, widen later)* | no |
| `PART3_N_TRAIN_A` / `_P` | 250 / 500 (tiny, exploratory) | **1500 / 3000** (= real `N_TRAIN_A`/`N_TRAIN_P`) | **yes** — closes EXP1.md §11's own open item: "re-measure amortization at the real training-set size" |

Every unmarked row is **inherited unchanged** from `config.py` — `config_biggpu.py` does
`from .config import *` and only overrides the seven rows marked "yes."

## 4. Code changes made to wire this up

None of these change any existing call site's default behavior — every new parameter/flag
defaults to the historical value, same convention as the rest of this repo (EXP1.md §9).

- **`influcoder/encoder.py`**: `distill()` gained `restore_best: bool = True`. When False,
  skips the `enc.load_state_dict(best_state)` restoration entirely, so the returned/saved
  encoder actually reflects the final epoch's weights (previously, whichever epoch scored
  best on `epoch_eval` was ALWAYS restored into `enc`, regardless of what
  `select_best_on`/the reported metric said — the reported number and the actual model
  weights had quietly diverged).
- **`baselines/exp1/train.py`**: `train_influcoder()` gained `restore_best: bool | None =
  None`, defaulting to `cfg.INFLUCODER_RESTORE_BEST_EPOCH`, threaded into the `distill()`
  call above.
- **`baselines/exp1/config.py`**: added `LOGRA_BIG_GPU` (False), `INFLUCODER_RESTORE_BEST_EPOCH`
  (True), `PART3_LESS_MODELS`/`PART3_LOGRA_MODELS`/`PART3_N_TRAIN_A`/`PART3_N_TRAIN_P` —
  all set to whatever value preserves this config's exact historical behavior.
- **`baselines/exp1/methods.py`**: `run_logra()` now passes `big_gpu=cfg.LOGRA_BIG_GPU`
  to `score_logra()` (previously never threaded through at all in this package — LoGRA
  batching only ever existed in the separate `.tuning_logs/` scratch scripts).
- **`baselines/logra/score.py`**: renamed the internal `_encode_sorted` helper to
  `encode_sorted` (public) so Part 3 can reuse the same length-sorted-batching logic
  directly, without duplicating it.
- **`baselines/exp1/part3.py`**: `time_less_logra()` now loops over
  `cfg.PART3_LESS_MODELS`/`cfg.PART3_LOGRA_MODELS` (was: hardcoded to exactly one model
  each) and its LoGRA encode call now respects `cfg.LOGRA_BIG_GPU` via a local
  `_logra_encode()` helper — a simpler OOM-retry (halves batch size on the already-loaded
  model) than `score_logra`'s own (which reloads the model fresh per retry) since Part 3
  loads one `LoGra` instance and times multiple sizes against it sequentially. The
  `N_TRAIN_A_SMALL`/`N_TRAIN_P_SMALL` module constants were removed in favor of
  `cfg.PART3_N_TRAIN_A`/`cfg.PART3_N_TRAIN_P`.
- **`baselines/exp1/{data,methods,train,part1,part2,part3}.py`**: the shared
  `from . import config as cfg` line in all six files became a 4-line `EXP1_CONFIG`
  env-var check (§2) — the only change needed for every part/consumer to pick up
  `config_biggpu.py` uniformly.

## 5. Before launching the real run — open questions / smoke-test first

These are genuinely unverified, not glossed over:

1. **`LESS_RANK=32` has never been run in this repo before** (every LESS number so far
   is at rank=16), though it's a much smaller jump than the paper's rank=128 default
   would have been, and LoGRA already ran cleanly at rank=32 on a smaller (~98GB) card
   (EXP1.md §5.6). Low risk, but still worth a quick smoke-test (n=1-2 samples) before
   the full sweep, same as any first-time parameter combination.
2. **`fast_jl` (LESS's fast CUDA projector) needs a fresh build for this GPU.** The
   working `fast_jl` build from the EXP2 session was compiled with
   `TORCH_CUDA_ARCH_LIST="12.0+PTX"` for a Blackwell GPU (compute capability 12.0) — PTX
   forward-compatibility only works upward, so that build will NOT run on this H200
   (Hopper, compute capability 9.0). Without rebuilding it here (same recipe as EXP1.md
   §5.6 point 4, just with `"9.0+PTX"` instead), LESS will silently fall back to TRAK's
   slower `BasicProjector` — correct, just not the fastest available, and not
   "maximizing LESS's speed" per the standard this repo already holds itself to.
3. **`LESS_BLOCK_SIZE=16` was tuned for rank=16**, not rank=32 — `grad_dim` (trainable
   LoRA param count) scales with rank, so the memory math behind the original
   `block_size` choice doesn't automatically transfer, though EXP1.md §10.1's own
   tangential finding (block_size barely matters at rank=16 — 1.03x between 16 and 32)
   suggests this is unlikely to matter much at rank=32 either.
4. **LoGRA at rank=32 + `BIG_GPU` batching was already verified together in EXP1.md
   §5.6** (on a smaller, ~98GB Blackwell card, no less) — this combination is the one
   part of this parameter set with real prior evidence behind it on similarly-scaled
   hardware.
5. **LESS remains completely unbatched** (`score_less`'s DataLoader is hardcoded to
   `batch_size=1`) — this run does not attempt to fix that; it's a bigger lift than this
   pass covers (EXP1.md §11 open item, unchanged).

## 6. What this does NOT change

- Part 3's `LESS_LOGRA_SIZES = [100, 1_000]` and `INFLUCODER_PROCESS_SIZES = [100, 1_000,
  10_000]` (the actual timing checkpoints) are untouched — only which/how many
  model sizes get timed at those checkpoints changed (§3).
- No new InfluCoder training-set sizes were added to Part 2's sweep (`DEFAULT_SIZES` in
  `part2.py`) — still `[25, 50, 100, 250, 500, 750, 1000, N_TRAIN_A]`.
- The combined-figure plotting scripts (`plot_figure1_combined.py` and friends) are
  untouched — they read whatever JSON the parts write, regardless of which `EXP1_CONFIG`
  produced it. Re-running the plot after a `biggpu` run needs no changes there.
