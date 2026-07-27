# EXP1: InfluCoder vs. LESS/LoGRA/RDS+/TF-IDF — unified speed-vs-quality comparison

Status as of 2026-07-27, end of session. Written for a fresh agent picking this up cold.
Read this before touching any of the scripts in `.tuning_logs/` — several of them encode
fixes for bugs that are easy to reintroduce if you "clean up" or rewrite from scratch.

## 1. What this experiment actually is

This experiment produces a three-part Figure 1, each part answering a different question
about the same set of candidate-selection methods (InfluCoder, LESS, LoGRA, RDS+, TF-IDF):

- **Part 1 — Cost vs Quality.** For a fixed eval, how good is each method (aggregate
  Spearman rho vs. a gradient-influence ground truth) and how expensive is it
  (inference ms/sample)? A snapshot comparison.
- **Part 2 — InfluCoder Training-Sample Scaling.** Holding everything else fixed, how does
  InfluCoder's distillation quality improve as its training set grows, and how does that
  compare to LESS's/LoGRA's quality ceiling?
- **Part 3 — GPU-time-vs-Samples-Processed Amortization.** InfluCoder has a large
  up-front training cost that LESS/LoGRA don't; at what point (real GPU time, real
  sample count) does that up-front cost pay for itself?

Section 4 below is the important one to read first if you're trying to reconcile numbers
between parts: it lays out exactly which parameters are held constant across all three,
and which genuinely diverge (and why).

## 2. Where this sits relative to the rest of the repo

- `baselines/figure1_table.py` + `baselines/plot_figure1.py` are the **pre-existing**,
  checked-in unified table/plot for the ORIGINAL fig1/fig1_dolci methodology (400x400,
  asymmetric LoGRA proxy ranks r=32, LESS at paper-default rank=128, eager attention
  pinned for FLOP-counter compatibility). Do not confuse these with this session's work —
  none of Parts 1/2/3 below touch these files or their output.
- **Everything under `.tuning_logs/` is new, untracked, and preset-name-agnostic scratch
  work from this session** (gitignored — see repo convention). It intentionally does NOT
  touch `figure1_table.py`, `plot_figure1.py`, or any `score_*` function's *default*
  argument values — it only ADDS new optional parameters (see §9) so existing call sites
  elsewhere in the repo are unaffected. Only the JSON results and PNG/PDF figures under
  `baselines/out/fig1_dolci/` get committed; the scripts that produced them stay local.
- The GT (ground truth) throughout all three parts is: **Qwen/Qwen3-4B, LoRA rank 16**,
  on the `fig1_dolci` preset (BBH anchors x dolci-instruct pool, `grad_max_len=1024`,
  `encoder_max_len=1024` — these two already happen to be equal in this preset, which
  matters, see §3 constraint 2). This preset and its cached GT/train-features were
  already used earlier this session for a separate InfluCoder-training reproduction
  investigation (see `dolci-non-reproduction` memory / `FINDINGS.md`) — unrelated to this
  experiment except for sharing the same cached artifacts.
- **GPU access.** All GPU-requiring scripts in this doc were run on Grid5000's `rennes`
  site, `abacus11` cluster (24GB cards), reserved via e.g.
  `oarsub -q p3 -p "cluster='abacus11'" -l gpu=1,walltime=2:00:00 'sleep 7200'`, then
  running the script either via `oarsh $OAR_JOB_ID <cmd>` or, since `rennes`'s `/home` is
  a shared NFS (see §10.6), simply passing the script as the job's own command. This is
  infrastructure-specific to this session's environment — the actual reproduction
  commands given per-part below are the same regardless of how you get GPU access.

## 3. The methods being compared, and their current recipe

| Method | Code | Model size(s) | Rank | Notes |
|---|---|---|---|---|
| InfluCoder | `baselines/influcoder/score.py` | 68m, 150m encoders | n/a | trained via `train_fig1_encoders.py --preset fig1_dolci`, checkpoints at `runs_out/fig1_dolci/encoder_{68m,150m}` |
| untrained encoder | `baselines/semantic/score.py` | same 68m/150m arch | n/a | same architecture as InfluCoder pre-distillation, off-the-shelf `jhu-clsp/ettin-encoder-{68m,150m}` |
| LESS | `baselines/less/score.py` + `baselines/less/model_utils.py`, `less_embeds.py` | 4B, 1.7B, 0.6B (Qwen3 family) | **16** (uniform across sizes) | TRAK `BasicProjector` random projection of per-sample LoRA-SGD gradients |
| LoGRA | `baselines/logra/score.py` + `modeling_logra.py` | 4B, 1.7B, 0.6B | **8** (uniform across sizes) | custom autograd extracting compact per-sample `[r,r]` gradient factors, no dense projection |
| RDS+ | `baselines/rdsplus/score.py` | 4B only | n/a | forward-only, SGPT weighted-mean pooling of hidden states, no gradients |
| TF-IDF | `baselines/tfidf/score.py` | none | n/a | no model at all, ~0 cost |

**Why LESS uses rank=16, not its paper default of 128:** see §10.1 (OOM saga). This is a
real deviation from LESS's documented config, not just a memory-chunking change — label
it as such if this ever appears in a paper-facing table.

**Why LoGRA uses a *uniform* rank=8 across all three sizes, not the asymmetric r8(4B)/
r32(proxies) used in the older `figure1_table.py`:** explicit user instruction this
session — asymmetric rank confounds "smaller model" with "extra rank," uniform rank
isolates model size alone. This makes these LoGRA numbers **not directly comparable** to
the older `figure1_table.py`/`plot_figure1.py` output, which used r32 for the proxies.
This uniform-rank convention is one of the things held constant across all three parts —
see §4.

## 4. Shared setup and divergences across Parts 1, 2, and 3

The user's explicit constraint for this documentation pass: parameters must follow the
same setup across Parts 1/2/3 wherever that's meaningful, and any place they genuinely
diverge must be called out with a reason, not silently glossed over. Below is exactly
that — what's actually held constant (more than you might guess), and the handful of
places that genuinely differ.

### 4.1 What's held constant across all three parts

| Parameter | Value | Used in |
|---|---|---|
| GT / teacher model | `Qwen/Qwen3-4B`, LoRA rank **16** | Parts 1, 2, 3 — both as the eval ground truth AND as InfluCoder's distillation-target teacher |
| `attn_implementation` | `sdpa` | Parts 1, 2, 3, every method with a model |
| `max_len` (`grad_max_len` == `encoder_max_len`) | **1024** | Parts 1, 2, 3 |
| LESS rank | **16** | Parts 1 and 3 run LESS fresh at this rank; Part 2's LESS reference numbers are literally Part 1's r=16 rows (see divergence 4.2.4) |
| LESS `proj_dim` / `lora_alpha` / `block_size` | 8192 / 512 / **16** | Parts 1 and 3 (block_size=16 is itself a bug fix, see §10.1 and §7's Part 3 methodology) |
| LoGRA rank | **8** | Parts 1, 2 (fresh 400x400 recompute), and 3 — uniform across every LoGRA size tested anywhere in this experiment |
| InfluCoder encoder architecture | `jhu-clsp/ettin-encoder-68m` (Part 1 also tests 150m) | Parts 1, 2, 3 |
| InfluCoder teacher-gradient `proj_dim` | **65536** | Parts 1 (baked into the `fig1_dolci` preset used to train the real checkpoints), 2, 3 |
| InfluCoder `epochs` | **8** | Parts 1, 2, 3 |
| InfluCoder `hard_ratio` | **0.0** | Parts 1, 2, 3 |
| InfluCoder `lr` | **5e-5** (`distill()`'s own default, no override) | Parts 1 and 3. **Part 2 diverges here — see 4.2.3.** |
| Preset / candidate pool | `fig1_dolci` (BBH anchors x `tasksource/dolci-instruct` pool) | Parts 1, 2, 3 |

This is a substantially larger shared core than it might look like from reading each
part's section in isolation — the model families, ranks, attention implementation,
sequence length, and InfluCoder's core recipe (epochs/hard_ratio/proj_dim) are identical
everywhere. The real divergences are narrower and more specific than "everything's
different"; they're listed next.

### 4.2 Where they genuinely diverge, and why

1. **Eval size.** Part 1 evaluates on a **50x50 slice** of the cached 400x400
   `fig1_dolci` eval split (a deliberate "peek," fast to iterate on). Part 2 evaluates on
   the **full 400x400** eval (needed to keep noise down across an 8-point sweep — a
   50x50 slice's aggregate Spearman is visibly noisy at the low end, see §6's "read with
   caution" note). Part 3 **has no eval at all** — it only measures wall-clock time to
   compute representations, never scores anchor-vs-pool similarity, so "eval size" isn't
   a concept that applies to it. **Judgement call:** this is the single biggest source of
   any number mismatch between Part 1 and Part 2 for the same nominal method (e.g. Part
   1's `untrained_68m` = +0.2074 vs. Part 2's `untrained_agg` = +0.3171 — both are the
   same untrained 68m encoder, scored on different eval sizes, not a bug).
2. **InfluCoder training-set size.** Part 1's checkpoints are the *real*, fully-trained
   `fig1_dolci` recipe checkpoints: 1500 anchors x 3000 pool, trained once via
   `python -m baselines.train_fig1_encoders --preset fig1_dolci` and saved to
   `runs_out/fig1_dolci/encoder_{68m,150m}`. Part 2 sweeps training-set size itself
   (25 to 1500 anchors x 2x pool, fresh distillation from scratch at every point — it
   does NOT reuse Part 1's saved checkpoint). Part 3 uses a deliberately tiny **250
   anchors x 500 pool** exploratory set, per explicit instruction ("this is not for the
   final experiment, but just to see how the curve will look like") — its InfluCoder
   quality is never even measured, only its wall-clock cost. **Judgement call:** don't
   compare Part 3's InfluCoder to Part 1's/Part 2's on quality grounds at all; it's
   timing-only by design.
3. **InfluCoder learning rate.** Part 2's `distill()` call hardcodes `lr=1e-5`
   (`.tuning_logs/_exp1_part2_scaling.py` line with `distill(..., lr=1e-5)`), overriding
   the `5e-5` default that both Part 1's real checkpoints (via
   `train_fig1_encoders.py`, which never overrides `lr`) and Part 3's exploratory
   training (`.tuning_logs/_part3_influcoder_process.py`, which also never overrides
   `lr`) both use. **This override is unexplained** — no rationale for it was found
   anywhere in this session's transcript, and it was apparently introduced either by the
   user directly editing the file or by an earlier part of this same session before a
   context-compaction boundary. **Judgement call, newly surfaced by this documentation
   pass:** since Part 1 and Part 3 independently agree on `5e-5` and Part 2 is the lone
   outlier, treat Part 2's `lr=1e-5` as more likely an accidental leftover than a
   validated choice — worth asking the user directly, and worth re-running Part 2's
   sweep at `5e-5` at some point to see whether it changes the scaling curve's shape
   materially before this number appears in anything paper-facing.
4. **InfluCoder epoch-selection methodology (newly surfaced by this documentation
   pass).** Part 1's real checkpoints are saved by `train_fig1_encoders.py`, which calls
   `distill(..., select_best_on="aggregated")` and then does `enc.save()` on the
   returned encoder — per `distill()`'s own docstring, this means the **saved checkpoint
   is restored to whichever of the 8 epochs scored best on eval**, not necessarily epoch
   8. Part 2 explicitly does the opposite by design: after the "dont keep the best
   epoch, keep the same epoch (8)" correction (see §6), it reports
   `log["epoch_metrics"][epochs - 1]` — always the **last** epoch, regardless of whether
   an earlier epoch scored higher. Part 3's InfluCoder training doesn't pass an
   `epoch_eval` callback at all (no per-epoch eval happens), so this distinction is moot
   for Part 3. **Net effect:** Part 1's InfluCoder quality numbers and Part 2's scaling
   curve are not using an identical training methodology even where their recipe knobs
   nominally match — Part 1 reflects "best of 8 epochs," Part 2 reflects "exactly epoch
   8." Whether this changes the actual saved 68m/150m checkpoints' quality vs. their
   own epoch-8 metric was not checked this session (would require re-inspecting
   `train_results.json`'s `best_epoch` field per checkpoint, which does exist — see
   `baselines/out/fig1/train_results.json` — but cross-referencing it wasn't done as
   part of this pass).
5. **Part 2's LESS reference lines use Part 1's 50x50 numbers, not a fresh 400x400
   recompute.** Unlike LoGRA (which WAS recomputed fresh at 400x400 for Part 2, see
   §6's "Plot" subsection), LESS was not — Part 2's plot pulls `less_4B`/`less_1.7B`
   directly from `draft50x50_table.json`. This is explicitly flagged with an asterisk in
   both Part 2's write-up and the combined figure's Part 2 panel title
   ("*LESS refs: 50x50 eval, not 400x400"). **Judgement call:** treat these two
   reference lines as rough cross-references, not apples-to-apples numbers, until
   someone runs a `_logra_400x400_recompute.py`-style script for LESS too.
6. **Model-size scope in Part 3.** Parts 1 and 2 test the full LESS/LoGRA proxy
   families (4B/1.7B/0.6B). Part 3 only tests **LESS-4B and LoGRA-1.7B** — a deliberate
   scope reduction per the user's explicit Part 3 request (testing every proxy size's
   timing wasn't asked for, and LESS/LoGRA are "waaaay too expensive" to run broadly
   just for a timing curve). This is a scope difference, not a parameter inconsistency —
   the rank/attn/max_len conventions above still apply to whichever sizes Part 3 does
   test.

## 5. Part 1: Cost vs Quality

### 5.1 Question

For a fixed eval, how good is each method (aggregate Spearman rho vs. the Qwen3-4B
gradient-influence ground truth) and how expensive is it per candidate (inference
ms/sample)? One unified script scores **every** candidate-selection method on the
**same** eval split, under matched experimental conditions (same attention
implementation, same sequence length cap, same rank held constant *within* each method's
proxy family — see §4.1), and plots aggregate Spearman rho vs. inference ms/sample on
one chart.

This is currently a **50x50 "peek,"** not the final result — small eval on purpose, to
get directionally-correct numbers fast (see §4.2.1 and §9 for scaling this up).

### 5.2 Exact parameters

All from `.tuning_logs/_fig1_draft_50x50.py`'s module-level constants — this is the
authoritative source, cross-reference §4.1 for what's shared with the other parts:

```
GT_MODEL = "Qwen/Qwen3-4B"; GT_LORA_RANK = 16
PRESET = "fig1_dolci"
N_EVAL = 50           # 50x50 SLICE of the cached 400x400 fig1_dolci eval (see note below)
ATTN = "sdpa"         # uniform across every method
MAX_LEN = 1024        # uniform across every method (grad + encoder alike)
LESS_RANK = 16        # uniform across LESS's 3 model sizes
LOGRA_RANK = 8        # uniform across LoGRA's 3 model sizes
LESS: proj_dim=8192, lora_alpha=512, block_size=16, gradient_checkpointing=False
LoGRA: seed=0, mlp_only via LOGRA_TARGET_MODULES
InfluCoder/untrained: encoder_dir = runs_out/fig1_dolci/encoder_{68m,150m}
```

**Why a 50x50 *slice*, not a fresh `n_eval=50` preset:** the already-trained 68m/150m
encoders were held out on `fig1_dolci`'s full 400-anchor/400-pool eval set. Slicing
`eval_pool[:50]`/`eval_anchors[:50]` from that cached split guarantees the 50x50 subset
is still genuinely held-out data (a strict subset of what the checkpoints never trained
on). A fresh preset with `n_eval_a=50` would shift the train/eval boundary elsewhere in
the pipeline and risks leakage against the already-trained checkpoints. Since each GT
entry is an independent anchor-pool gradient cosine (no cross-sample normalization),
`gt[:50, :50]` is exactly the GT a fresh 50x50 computation would produce anyway.

### 5.3 How to reproduce

```bash
cd /home/dkachler/year1/rebut_ie/influcoder
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

# 1. (one-time, if runs_out/fig1_dolci/encoder_{68m,150m} don't already exist)
#    Train the real InfluCoder checkpoints this part scores. Needs a GPU.
.venv_h100/bin/python -m baselines.train_fig1_encoders --preset fig1_dolci

# 2. Run the unified table (needs a GPU -- loads Qwen3-4B/1.7B/0.6B + both encoders).
.venv_h100/bin/python .tuning_logs/_fig1_draft_50x50.py
# writes baselines/out/fig1_dolci/draft50x50_table.json

# 3. Plot (no GPU needed -- pure plotting from the JSON above).
.venv_h100/bin/python .tuning_logs/_plot_draft50x50.py
# (or: python3 .tuning_logs/_plot_draft50x50.py -- any matplotlib-having interpreter works,
# this step never touches torch/transformers)
# writes baselines/out/fig1_dolci/draft50x50_figure.{png,pdf}
```

Step 2 is idempotent w.r.t. step 1 (it just loads whatever checkpoint exists at
`runs_out/fig1_dolci/encoder_*`) but is NOT checkpoint-resumable itself — a kill mid-run
loses partial progress on whichever method was in flight; each method's row is fast
enough (tens of seconds to a few minutes) that this hasn't mattered in practice.

### 5.4 Results

From `baselines/out/fig1_dolci/draft50x50_table.json` (post all timing-hygiene fixes,
see §10.4):

| method | agg rho | per-anchor rho | ms/sample |
|---|---:|---:|---:|
| less_4B | +0.9781 | +0.9013 | 1244.73 |
| logra_4B (r8) | +0.9115 | +0.8041 | 364.52 |
| influcoder_68m | +0.7916 | +0.7444 | 9.99 |
| influcoder_150m | +0.7409 | +0.6892 | 41.78 |
| less_1.7B (r16) | +0.5602 | +0.5051 | 688.08 |
| logra_1.7B (r8) | +0.3546 | +0.1929 | 240.25 |
| untrained_68m | +0.2074 | +0.1243 | 5.75 |
| untrained_150m | +0.1292 | +0.1163 | 9.72 |
| tfidf | -0.2457 | -0.1674 | 0.83 |
| rdsplus | -0.1262 | -0.0666 | 112.42 |
| logra_0.6B (r8) | -0.1409 | -0.1096 | 233.50 |
| less_0.6B (r16) | -0.0897 | -0.0798 | 490.16 |

**Read with caution:** this is a 50x50 aggregate — high variance, especially for the
near-zero/negative rows (RDS+, TF-IDF, the 0.6B proxies). Don't treat the exact sign or
magnitude of those as settled; they're directionally "weak/collapsed," not precisely
quantified yet.

**Notable pattern:** both LESS and LoGRA proxies still carry real signal at 1.7B (+0.56,
+0.35) but collapse to ~0/negative at 0.6B — consistent across both gradient-method
families. InfluCoder's tiny trained encoders are extremely cost-competitive: 68m gets
+0.79 at 10ms/sample vs. LESS-4B's +0.98 at 1245ms/sample (~125x cheaper for ~80% of
the quality).

### 5.5 Plot design

`.tuning_logs/_plot_draft50x50.py` generates `baselines/out/fig1_dolci/draft50x50_figure.{png,pdf}`
as a **standalone** figure. Reuses this repo's validated categorical palette (`C_GRAD`
blue = LESS/LoGRA fwd+bwd methods, `C_FWD` orange = InfluCoder/untrained/RDS+
single-forward methods, `C_FREE` aqua = TF-IDF's horizontal reference line).

- **Linear x-axis** in the standalone version (NOT log — user explicitly asked for this
  originally, overriding `plot_figure1.py`'s log-x convention). **Note:** the *combined*
  Figure 1 (§8) uses a **sqrt x-axis** for this same panel instead, per a later,
  separate request specific to the combined layout — the standalone
  `draft50x50_figure.png` was NOT updated to match; the two figures currently use
  different x-axis transforms for the same data. See §8 and §11 for this open item.
- **No Pareto frontier.** Instead, a solid line connects each method's own proxy family
  in size order (4B → 1.7B → 0.6B) for LESS and for LoGRA separately — shows the
  within-family size/cost/quality trend directly.
- Marker shape distinguishes LESS (circle) from LoGRA (square) since both share the blue
  gradient-method color across 3 sizes each.
- Marker size scales with model size, consistently across both gradient methods AND the
  encoder sizes (150m > 68m; 4B > 1.7B > 0.6B).
- Filled = trained (InfluCoder, LESS, LoGRA, RDS+), hollow = untrained encoder only.

Just rerun `.venv_h100/bin/python .tuning_logs/_plot_draft50x50.py` from the repo root
after any table update — it's idempotent and fast (no GPU needed, pure plotting).

## 6. Part 2: InfluCoder Training-Sample Scaling

### 6.1 Question

How does InfluCoder's distillation quality scale with the number of training samples,
holding everything else (epochs, hard_ratio, encoder_max_len, encoder size=68m) at the
`fig1_dolci` recipe? Fixed 1:2 anchor:pool ratio throughout (matches `fig1_dolci`'s own
1500:3000 config). See §4.2.3 for the one place this part's recipe actually diverges
from Parts 1/3 (`lr=1e-5` instead of `5e-5`).

### 6.2 Exact parameters

From `.tuning_logs/_exp1_part2_scaling.py`:

```
GT_MODEL = "Qwen/Qwen3-4B"; GT_LORA_RANK = 16
PRESET = "fig1_dolci"
N_EVAL = 400                              # full eval, NOT the Part-1 50x50 slice (§4.2.1)
ENCODER_MODEL = "jhu-clsp/ettin-encoder-68m"
SIZES = [25, 50, 100, 250, 500, 750, 1000, 1500]   # n_train_anchors swept; n_train_pool = 2x
epochs=8, hard_ratio=0.0, encoder_max_len=1024     # from cfg, matches fig1_dolci preset
lr=1e-5                                    # ** diverges from the 5e-5 default -- see §4.2.3 **
select_best_on="aggregated"                # passed to distill(), but NOT what gets reported
                                            # (see epoch-selection note, §4.2.4 and below)
seed=0
```

**Important methodology correction made mid-session — read this before trusting any
older point:** the very first version of this sweep called
`distill(..., select_best_on="aggregated")` and reported
`log["epoch_metrics"][log["best_epoch"]]` — i.e. whichever epoch scored highest on the
eval callback, which is `distill()`'s own internal best-epoch-restore behavior (see
`influcoder/encoder.py`'s `distill()` docstring: it restores the encoder to its
best-scoring epoch before returning, specifically because eval Spearman is noisy at
small sample counts and "reliably peaks then degrades"). **The user explicitly rejected
this for this experiment**: "dont keep the best epoch, keep the same epoch (8)" — every
sweep point should reflect training for the FULL fixed epoch budget, not a cherry-picked
epoch, since the x-axis here is "how much data," not "how well can you early-stop." Fix:
`run_size()` now uses `final_idx = epochs - 1; final = log["epoch_metrics"][final_idx]`
instead of `log["epoch_metrics"][log["best_epoch"]]` — `epoch_metrics[-1]` is always the
metrics computed at the end of the LAST epoch (8), recorded by the `epoch_eval` callback
*before* `distill()`'s internal restoration runs, so no change to `influcoder/encoder.py`
itself was needed. `select_best_on="aggregated"` is still passed to `distill()` (harmless
— it only controls what gets restored into the returned `enc` object, which this script
`del enc`s immediately after anyway) but is no longer what gets *reported*. The three
already-completed points from the OLD (best-epoch) methodology were discarded and the
full sweep was rerun from scratch under the fixed-epoch-8 methodology — do not trust any
scaling number that isn't from a run using `final_idx`/`epoch_metrics[-1]`. **This is
exactly the methodology that Part 1's real checkpoints do NOT use** — see §4.2.4.

### 6.3 How to reproduce

```bash
cd /home/dkachler/year1/rebut_ie/influcoder
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

# 1. Main sweep (needs a GPU; reuses the cached Qwen3-4B train-side featurization from
#    Part 1's checkpoint training, trainfeat_Qwen_Qwen3-4B_1500x3000_r16_s0_eval400x400.pt
#    -- zero new expensive featurization for any size up to 1500 anchors).
.venv_h100/bin/python .tuning_logs/_exp1_part2_scaling.py
# writes baselines/out/fig1_dolci/scaling_68m_400x400.json
# safe to re-launch after a kill -- main() skips any n_a already present in the JSON.

# 2. (optional) fill in extra spot-check points without re-running the whole sweep:
.venv_h100/bin/python .tuning_logs/_exp1_part2_scaling_extra.py   # SIZES=[125,175,1500]
# writes baselines/out/fig1_dolci/scaling_68m_400x400_extra.json

# 3. Fresh LoGRA reference lines at the SAME 400x400 eval (needed since Part 1's LoGRA
#    numbers are on the 50x50 slice, not directly comparable to this part's 400x400 curve):
.venv_h100/bin/python .tuning_logs/_logra_400x400_recompute.py
# writes baselines/out/fig1_dolci/logra_400x400_r8.json

# 4. Plot (no GPU needed).
.venv_h100/bin/python .tuning_logs/plot_scaling_part2.py
# writes baselines/out/fig1_dolci/scaling_figure.{png,pdf}
```

If you add another training-set size, copy `_exp1_part2_scaling_extra.py`'s pattern
(a new `SIZES` list, same `run_size()` body) rather than editing `SIZES` in the main
script — this preserves the checkpoint-resume behavior described above.

**Going above 1500 anchors requires care:** the dolci-instruct pool is capped at 6000
total docs by `load_dolci_instruct`'s default `max_docs`, so `n_train_a` is capped near
2800 (at 1:2 ratio) without ALSO bumping `max_docs` — and bumping `max_docs` would
silently change the EVAL pool's identity too, since `random.Random(seed).shuffle()`'s
output depends on the full list length (Python's Fisher-Yates shuffle is NOT
prefix-stable across different list sizes). Do not just raise `max_docs` against the
existing GT cache without recomputing/rechecking the eval split still matches.

### 6.4 Results

`baselines/out/fig1_dolci/scaling_68m_400x400.json` + `_extra.json` combined, all
fixed-epoch-8, all on the full 400x400 eval. `untrained agg` is identical across every
row since it's the same untrained 68m encoder scored on the same eval before any
training:

| n_train_anchors | n_train_pool | n_train_total | untrained agg | agg (epoch 8) | per-anchor mean |
|---:|---:|---:|---:|---:|---:|
| 25 | 50 | 75 | +0.3171 | +0.4930 | +0.4524 |
| 50 | 100 | 150 | +0.3171 | +0.5330 | +0.4912 |
| 100 | 200 | 300 | +0.3171 | +0.6237 | +0.5785 |
| 125 | 250 | 375 | +0.3171 | +0.6433 | +0.5963 |
| 175 | 350 | 525 | +0.3171 | +0.7052 | +0.6516 |
| 250 | 500 | 750 | +0.3171 | +0.6990 | +0.6491 |
| 500 | 1000 | 1500 | +0.3171 | +0.7322 | +0.6830 |
| 750 | 1500 | 2250 | +0.3171 | +0.7450 | +0.6985 |
| 1000 | 2000 | 3000 | +0.3171 | +0.7509 | +0.7081 |
| 1500 | 3000 | 4500 | +0.3171 | +0.7676 | +0.7254 |

Monotonically increasing, diminishing returns past ~750-1000 anchors (last doubling,
750→1500, only buys +0.023 vs. the +0.10 gained going 100→250). Note this `untrained_agg`
(+0.3171) is a DIFFERENT number from Part 1's `untrained_68m` row (+0.2074) — expected
per §4.2.1 (different eval sizes), not a bug.

**Fresh LoGRA reference lines** (`baselines/out/fig1_dolci/logra_400x400_r8.json`,
same call pattern as Part 1's LoGRA rows — `score_logra(..., lora_rank=8, max_len=1024,
attn_implementation="sdpa")`, GT=Qwen3-4B rank16 — but on the FULL unsliced 400x400
splits instead of Part 1's 50x50 slice):

| row | agg rho (400x400, r8, sdpa) | vs. `table1.json`'s (unverified) value |
|---|---:|---:|
| logra_4B | +0.8942 | +0.9079 (close — within noise) |
| logra_1.7B (proxy) | +0.4309 | +0.5779 (notably different) |

The 1.7B row also cross-checks against an independent earlier-this-session run
(`baselines/out/fig1_dolci/logra_uniform_r8.json`'s `logra_proxy_1.7B_r8` = +0.4299) —
the two independent runs agree to within 0.001. **This plot uses these recomputed,
cross-verified values, NOT `table1.json`.** The InfluCoder curve crosses the 1.7B LoGRA
line almost immediately (already above it at the smallest tested size, n=75 total) and
does not reach the 4B LoGRA line anywhere in the tested range (max +0.7676 at n=4500 vs.
+0.8942).

**LESS reference lines** (added later, per explicit request) come directly from Part 1's
50x50 table — `less_4B` = +0.9781, `less_1.7B` = +0.5602 — **not recomputed at 400x400**.
See §4.2.5; treat these two as rough cross-references only.

**`table1.json`'s own internal inconsistency (its `influcoder_68m` = +0.0466 vs. this
session's +0.7676 at a comparable/larger training size) is still unexplained** — it
remains flagged in `FINDINGS.md` as the likely source of the `dolci-non-reproduction`
memory's still-open "~+0.78 vs ~+0.05" discrepancy.

### 6.5 Plot design

`.tuning_logs/plot_scaling_part2.py` generates `baselines/out/fig1_dolci/scaling_figure.{png,pdf}`
as a standalone figure: x-axis = InfluCoder training samples (linear), y-axis =
aggregate Spearman rho, InfluCoder's curve (orange, hollow marker at x=0 for the
untrained encoder) plus four horizontal reference lines (LoGRA 4B/1.7B, LESS 4B/1.7B,
each a different dash pattern). This axis choice (linear-x) matches the combined
figure's Part 2 panel — unlike Parts 1 and 3, Part 2's axis choice was never revised.

## 7. Part 3: GPU-time-vs-Samples-Processed Amortization

### 7.1 Question

At what point does using InfluCoder actually become cheaper than just running LESS/LoGRA
directly, in real cumulative GPU time? Explicit user request, verbatim intent: test
LoGRA 1.7B, LESS 4B, and InfluCoder 68m, timing how long each takes to "process" 100 /
1K / 10K samples, where **"process" means computing the per-sample representation
only** (LESS: TRAK-projected per-sample gradient via `collect_grads`; LoGRA: per-sample
`[r,r]` gradient factor via `LoGra.encode()`; InfluCoder: encoder embedding via
`embed()`) — explicitly excluding any anchor-vs-pool scoring matmul, which is out of
scope here (depends on query-set size, "a can of worms" for a separate experiment).

**Why dolci-instruct specifically (the pool all scripts draw samples from):** it's the
only one of this repo's three sample pools that can supply real, non-duplicated samples
at every scale this experiment needs. BBH is hard-capped at 6511 total examples; the
local Dolly file has only 15011 rows; `tasksource/dolci-instruct` streams from ~1.8M rows
across 8 parquet shards, comfortably covering 100 through 1,000,000+.

### 7.2 Exact parameters

From `.tuning_logs/_part3_less_logra_process.py` and `_part3_influcoder_process.py` —
cross-reference §4.1 for what's shared with Parts 1/2:

```
# LESS / LoGRA half
MAX_LEN = 1024; ATTN = "sdpa"; SIZES = [100, 1000]
LESS:  model="Qwen/Qwen3-4B",   proj_dim=8192, lora_rank=16, lora_alpha=512,
       lora_dropout=0.1, block_size=16, gradient_checkpointing=False
LoGRA: model="Qwen/Qwen3-1.7B", rank=8, mlp_only=True

# InfluCoder half
GRAD_MODEL = "Qwen/Qwen3-4B"; GT_LORA_RANK = 16; PROJ_DIM = 65536
GRAD_MAX_LEN = 1024; ENCODER_MAX_LEN = 1024
ENCODER_MODEL = "jhu-clsp/ettin-encoder-68m"
EPOCHS = 8; HARD_RATIO = 0.0
N_TRAIN_A = 250; N_TRAIN_P = 500     # deliberately small exploratory set, see §4.2.2
PROCESS_SIZES = [100, 1000, 10000]
# lr not overridden -- uses distill()'s 5e-5 default, matching Part 1, unlike Part 2
```

LESS and LoGRA were only run at n=100/1000 directly (cost-prohibitive beyond that, per
explicit instruction: "waaaay too expensive") and extrapolated from there; InfluCoder was
run directly at n=100/1000/10000, and further extrapolated to n=100,000/1,000,000 for
the combined figure (§8).

### 7.3 Bug hit and fixed: block_size regression

The LESS/LoGRA script initially OOM'd on its very first batch (n=100). Root cause: it
hardcoded `block_size=128` for `collect_grads`'s `BasicProjector`, silently ignoring the
exact fix already documented in §10.1 (this 24GB card requires `block_size=16` at
`lora_rank=16` — `block_size=128` was one of the three root causes of the original LESS
OOM saga, not a new bug, just the same old fix not being carried over into a new script).
Fixed by setting `block_size=16`; reran clean. **Lesson: any new script that calls
`collect_grads` directly needs this same fix applied by hand — it lives in each call
site, not in a shared default.**

### 7.4 How to reproduce

```bash
cd /home/dkachler/year1/rebut_ie/influcoder
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

# Both halves can run in parallel on separate GPU jobs (explicit user permission to
# parallelize across instances) -- shown here run sequentially for simplicity.

# LESS + LoGRA half (needs a GPU; loads Qwen3-4B then Qwen3-1.7B sequentially).
.venv_h100/bin/python .tuning_logs/_part3_less_logra_process.py
# writes baselines/out/fig1_dolci/part3_less_logra_process.json

# InfluCoder half (needs a GPU; does teacher-gradient collection + 8-epoch distill +
# encode-throughput timing, in that order).
.venv_h100/bin/python .tuning_logs/_part3_influcoder_process.py
# writes baselines/out/fig1_dolci/part3_influcoder_process.json

# Plot (no GPU needed).
.venv_h100/bin/python .tuning_logs/plot_part3.py
# writes baselines/out/fig1_dolci/part3_figure.{png,pdf}
```

### 7.5 Results

| method | model | setup/load time | n=100 | n=1000 | n=10000 |
|---|---|---:|---:|---:|---:|
| InfluCoder | 68m (250x500, epochs=8) | 459.58s (356.42s collect + 103.16s train) | 43.95 ms/sample* | 11.01 ms/sample | 10.81 ms/sample |
| LESS | 4B, r=16, sdpa | 74.06s (model load) | 1123.14 ms/sample | 1091.67 ms/sample | not run (extrapolated) |
| LoGRA | 1.7B, r=8, sdpa | 35.77s (model load) | 218.38 ms/sample | 217.63 ms/sample | not run (extrapolated) |

\* InfluCoder's n=100 figure is inflated by a batch_size=32 CUDA-kernel warmup artifact on
the first `embed()` call at that batch shape (same class of issue as §10.4's warmup
finding, but for a new kernel shape, not a cold GPU) — n=1000/10000 are consistent with
each other (~10.8-11.0 ms/sample) and are the trustworthy steady-state rate.

**Linearity check (the user explicitly asked to be checked on this): confirmed for all
three methods** — LESS varies only ~3% between n=100 and n=1000 (1123.14 -> 1091.67
ms/sample); LoGRA varies <0.4% (218.38 -> 217.63 ms/sample); InfluCoder's steady-state
(post-warmup) rate is effectively flat (11.01 -> 10.81 ms/sample, n=1000 -> n=10000).
Extrapolating LESS/LoGRA's measured per-sample rate further out is therefore sound.

InfluCoder's gradient collection (356.42s) is **~3.5x** more expensive than its
distillation training (103.16s) — the "expensive" part of standing up InfluCoder is
computing the teacher gradients it distills from, not the distillation step itself.

**Amortization points** (`.tuning_logs/plot_part3.py`, solving
`InfluCoder_samples(t) = other_samples(t)` from each method's own
setup/load-time-plus-constant-rate model):

- **InfluCoder overtakes LESS at ~463s of cumulative GPU time (~357 samples processed).**
- **InfluCoder overtakes LoGRA at ~482s of cumulative GPU time (~2049 samples processed).**

Both crossovers land within seconds to tens of seconds of InfluCoder's own setup
finishing (459.6s) — not a gradual catch-up. This is a direct consequence of the
per-sample speed gap being so large (InfluCoder ~101x faster than LESS, ~20x faster than
LoGRA at steady state) that once InfluCoder starts processing, it doesn't need to "catch
up" so much as immediately overtake.

Extrapolated further (used in the combined figure, §8): LESS would take ~10,917s (~3.03
hours) to reach 10,000 samples, ~109,241s (~30.3 hours) for 100,000, and ~1,091,744s
(~12.6 days) for 1,000,000. LoGRA: ~2,176s / ~21,798s / ~217,662s for the same three
sizes.

**Caveat — not a final number.** This is explicitly an exploratory run (250x500
distillation set, single measurement per size, no repeated trials for noise estimation)
meant only to confirm the shape of the curve exists and roughly where it crosses. Don't
quote the exact "463s" / "357 samples" figures as final results without re-running at the
real experiment's actual train size and with repeated trials.

### 7.6 Plot design

`.tuning_logs/plot_part3.py` generates `baselines/out/fig1_dolci/part3_figure.{png,pdf}`
as a standalone figure. X-axis = cumulative GPU time (seconds, **linear** in this
standalone version); Y-axis = cumulative samples processed (**log** scale, per explicit
request — a log y-axis can't render 0, so each curve is drawn as a flat segment at a
`FLOOR=0.5` visual floor during its setup/load phase, purely a plotting convention, not
a real value). Two shaded regions mark InfluCoder's setup phase split (light blue =
gradient collection, light orange = distillation training). LESS and LoGRA get their own
blue/purple (rather than sharing one color as in Parts 1/2, since here they appear as
separate curves that need to be told apart on the same axes). **Note:** the combined
figure's Part 3 panel (§8) uses a different x-axis (log, not linear) for the same data —
see §8 and §11.

## 8. Combined Figure 1

`.tuning_logs/plot_figure1_combined.py` concatenates all three parts into one row of
three square panels (`ax.set_box_aspect(1)` on each, regardless of each panel's own data
range), producing `baselines/out/fig1_dolci/figure1_combined.{png,pdf}`. This is a
**separate, later addition** — built after the three standalone figures already existed
— and its per-panel axis choices were iterated on independently, so **it does not always
match the corresponding standalone figure's design** (flagged per-panel below and in
§11).

### 8.1 How to reproduce

```bash
cd /home/dkachler/year1/rebut_ie/influcoder
export PYTHONPATH=$(pwd)
python3 .tuning_logs/plot_figure1_combined.py   # no GPU needed, pure plotting from
                                                  # the JSON files Parts 1/2/3 already wrote
# writes baselines/out/fig1_dolci/figure1_combined.{png,pdf}
```

Requires all of Part 1's `draft50x50_table.json`, Part 2's `scaling_68m_400x400*.json` +
`logra_400x400_r8.json`, and Part 3's `part3_*.json` to already exist — it does no GPU
work itself, only reads and re-plots.

### 8.2 Panel 1 (Cost vs Quality) — sqrt x-axis, NOT the standalone's linear

Per explicit request ("make the log of part 1 be less aggressive so that the LESS and
LoGRA values are more spread out"): a genuine log-base change (e.g. base 2 vs. base 10)
cannot actually redistribute spacing — `log_b(x)` is a pure affine rescaling of
`log(x)`, which cancels out once matplotlib fits the axis to its limits, so *every* log
base renders identically. A real "gentler than log" compression needs a different
function family. This panel uses **`ax.set_xscale("function", functions=(np.sqrt,
np.square))`** with manually placed ticks at `[10, 25, 50, 100, 250, 500, 1000]` ms. This
spreads the high-cost LESS/LoGRA cluster (233-1245ms) across ~60% of the panel width
(vs. ~25% under log10), at the cost of compressing the cheap InfluCoder/untrained
cluster (5.75-42ms) somewhat more than log10 would — an explicit, accepted trade-off.

### 8.3 Panel 2 (InfluCoder Scaling) — same as standalone, plus LESS reference lines

Same linear-x design as the standalone `scaling_figure.png`, with two additions per
explicit request: LESS 4B and LESS 1.7B horizontal reference lines, pulled from Part 1's
`draft50x50_table.json` (same numbers, same 50x50-eval caveat as §4.2.5 — the panel
title says so explicitly: "*LESS refs: 50x50 eval, not 400x400"). Y-axis limits were
padded slightly above the LESS 4B line (+0.98) so its label doesn't crowd the plot's top
edge.

### 8.4 Panel 3 (GPU-time vs Samples Processed) — log-log, with real/extrapolated markers

This panel went through the most iteration and ended up **log-x** (matching its
already-log y-axis) — this is a **change from the standalone `part3_figure.png`'s
linear-x** (§7.6). The reasoning, in order:

1. First request: add marker points at n=100/1K/10K/100K/1M samples per method (filled
  circle = actually measured at that exact n, hollow circle = extrapolated from the
  steady-state per-sample rate). This immediately breaks a fixed 0-900s linear window,
  since LESS's extrapolated 1M-sample point lands at **~1.09 million seconds (~12.6
  days)** of hypothetical GPU time.
2. Tried linear-x spanning the full 0-to-1.09M-second range: unusable — the crossing
  region (~470s) becomes <0.1% of the panel width, i.e. three nearly-vertical lines
  crushed against the left edge.
3. Tried sqrt-x (same transform as Panel 1, which handles x=0 natively unlike log):
  much better (~2% of panel width for the crossing region, a real improvement), and was
  the first fix attempt.
4. **Final choice, per explicit "revert to log" instruction: log-x**, `X_MIN=10` (a
  log axis can't include exactly 0, so the axis starts just below LoGRA's ~36s load
  time; each curve's pre-start "zero samples" segment is simply not drawn below its own
  load/setup time rather than being padded out to a fake x=0). Amortization crossing
  points are marked with star markers and small `"{t:.0f}s"` labels (463s vs LESS, 482s
  vs LoGRA).

Real vs. extrapolated, per method, at the five target sizes:
- **InfluCoder:** 100/1K/10K measured directly; 100K/1M extrapolated from the 10.81
  ms/sample (n=10000) steady-state rate.
- **LESS, LoGRA:** only 100/1K measured (per explicit instruction not to run further);
  10K/100K/1M all extrapolated from the n=1000 rate.

## 9. Code changes made this session (all additive, no default behavior changed)

- `baselines/less/model_utils.py`: `load_base_with_fresh_lora` gained
  `gradient_checkpointing: bool = False` and `attn_implementation: str = "eager"` params.
- `baselines/less/score.py`: `score_less` gained `block_size: int = 128`,
  `gradient_checkpointing: bool = False`, `attn_implementation: str = "eager"` params,
  threaded through to the calls above / `collect_grads`.
- `baselines/less/less_embeds.py`: `collect_grads` gained a `block_size: int = 128`
  parameter (previously **hardcoded** as a local variable inside the function — this was
  a real, separate bug source, see §10.1).
- `baselines/rdsplus/score.py`: `score_rdsplus` gained `attn_implementation: str =
  "eager"`.
- `baselines/semantic/score.py`, `baselines/influcoder/score.py`: gained
  `attn_implementation: str = "eager"`, AND both got a throwaway warm-up
  `model.encode(["warmup"], ...)` call inserted right after `model.to("cuda")` and
  before `meter.model_ready()` — this is a real bug fix, not a stylistic addition, see
  §10.4. **Do not remove this line** without understanding why it's there.
- `FINDINGS.md`: added bullets documenting the CUDA-warm-up-vs-InfluCoder finding
  (§10.4), the InfluCoder training-sample scaling results, the LoGRA r8 recompute, and
  the Part 3 amortization-curve findings.
- `G5K.md` / `G5K-CPU-AGENT.md`: updated to state that **multiple concurrent GPU jobs
  are explicitly allowed** (previously said "never hold more than one active GPU job") —
  explicit user instruction this session. See §10.3 for the related `oarstat -J` bug this
  uncovered.

None of these edits change any *existing* call site's behavior (every new parameter
defaults to the old hardcoded value).

## 10. Where we got stuck — read this before repeating any of it

### 10.1 LESS OOM saga (three rounds, do not redo the first two)

LESS at 4B, `eager` attention, rank=128 (paper default) OOMs on a 24GB card
(A5000). The debugging went through three distinct, separate root causes — **all three
had to be fixed together**, fixing only one was not enough at each stage:

1. **`BasicProjector`'s memory footprint.** TRAK's `BasicProjector.__init__` allocates
   `[grad_dim, block_size]` up front. At rank=128, `grad_dim` (trainable LoRA params,
   `lora_target_modules="all-linear"`) is ~157.5M, and `block_size` was **hardcoded to
   128** inside `collect_grads` (`baselines/less/less_embeds.py`) — a pure
   memory-chunking constant, NOT part of LESS's actual config (verified: it doesn't
   change the projection's mathematical output, only how many columns get materialized
   per GEMM chunk). This alone was a ~7-17GB allocation (estimate varies by
   fp32-vs-bf16 assumption — verify empirically before trusting a specific number).
   **User's explicit instruction:** reduce `lora_rank` (128→16, matching the GT
   featurizer's own rank), NOT `block_size`, as the primary fix — this IS a real
   deviation from LESS's paper config and must be labeled as such everywhere (`less_r16`
   / "LESS (r=16, reduced from paper default r=128)"). `block_size` was ALSO reduced
   (128→16) as a secondary, purely-mechanical fix once rank=16 alone still weren't
   enough (see next point) — the user approved this specifically after seeing rank=16
   alone didn't fully resolve things.
2. **Even at rank=16 + block_size=16, still OOM'd** — crashed consistently at
   sample 5/400, inside `eager_attention_forward`'s fp32 softmax, ~22.7-22.8GB used
   both before and after the block_size fix (i.e. block_size changed *nothing*
   measurable — proof it was never the dominant cost at rank=16). Root cause: `eager`
   attention materializes the full `[B,H,S,S]` attention matrix in fp32 and retains it
   for backward, with **no gradient checkpointing**, for a full 4B model — genuinely
   memory-hungry at `seq_len` near 1024 regardless of LoRA rank. Fixed by adding
   `gradient_checkpointing=True` support (trades an extra forward-recompute for much
   less peak memory; same loss/gradients, not a config change).
3. **Later in the session, discovered `sdpa` fixes this more cleanly than
   `gradient_checkpointing` does** — `sdpa` never materializes the full attention
   matrix in the first place, so it doesn't need the recompute trade-off at all. A
   direct empirical test (`_less_sdpa_test.py`, n=50/side) showed: `sdpa` no-checkpoint
   = 1241.52ms/sample @ 21.56GB peak vs. `eager`+checkpoint = 1487.21ms/sample @
   12.30GB peak — sdpa is **1.2x faster** despite using more memory, and comfortably
   fits in 24GB. **Once the fairness-constraint work started (§3), everything moved to
   `sdpa` with `gradient_checkpointing=False`** — the checkpointing code path still
   exists (parameter kept, defaults to False) but is no longer the active fix.

**A tangential finding surfaced by this saga:** a quick isolated test
(`_less_blocksize_test.py`) at block_size=16 vs 32 gave only a **1.03x** difference —
confirms the projection/chunking step is *not* where LESS's time goes; the dominant
cost is the full-model forward+backward, independent of the projector. This is why the
block_size regression in Part 3 (§7.3) was cheap to fix once spotted — it was never the
actual cost driver, just an OOM trigger.

### 10.2 "Is this a REAL speedup" — proxy-model speed methodology (earlier in session)

Before this unified experiment, a separate line of work (also in `.tuning_logs/`,
`_fair_speed.py` and friends) rigorously established that a smaller proxy model (1.7B
vs 4B) gives a genuine ~2x wall-clock speedup from fitting a deeper batch under a fixed
memory budget — NOT from any generic optimization that could also be applied to the 4B
model (the user was explicit and repeated about this distinction: "There should be no
changes we make that we also can't do on the full four b model"). This resulted in the
`logra-batching-not-dilution` memory (also flags that a prior `FINDINGS.md` claim about
batch-mean-loss "diluting" LoGRA's per-sample gradient was WRONG — the loss is actually
sum-reduced via `num_items_in_batch=1`). This finding is separate from and predates the
current unified-table experiment, but explains why proxy models are believed to be a
legitimate cost lever at all.

### 10.3 GPU discovery bug: `oarstat -J`'s `assigned_hostnames` field is always `None`

When surveying all of G5K for large idle GPUs (after the user allowed running multiple
concurrent GPU jobs), an initial script matched `oarstat -J`'s (JSON output)
`assigned_hostnames` field against `oarnodes -J`'s `network_address` to determine which
GPUs were occupied. **This field is always `None` in the JSON output** — even for jobs
we KNEW were `Running` at the time (verified by checking our own known job IDs
directly). **Fix:** use `assigned_resources` (a list of integer resource IDs) instead,
matched against each node entry's own `resource_id` field. `oarstat -fj <id>` (plain text
format, not `-J`) DOES populate `assigned_hostnames` correctly — the bug is specific to
the JSON output path.

### 10.4 The big one: InfluCoder measured ~17x slower than the untrained encoder — CUDA warm-up artifact, not real

Fully written up in `FINDINGS.md`'s "Instrument / measurement hygiene" section — read it
there for the canonical version. Summary for context:

- **Symptom:** first pass of the unified 50x50 table gave `influcoder_68m` = 99.75
  ms/sample vs. `untrained_68m` = 5.73 ms/sample (17x). `influcoder_150m` = 54.18 vs.
  `untrained_150m` = 9.69 ms/sample (5.6x). Both pairs are the SAME architecture
  (ModernBERT) — only the WEIGHTS differ (trained vs. off-the-shelf).
- **Ruled out (verified identical between the two, so NOT the cause):** dtype (both
  fp32), resolved `attn_implementation` (both `sdpa`), tokenized sequence length (both
  802 tokens for an identical test string).
- **Root cause, confirmed by a direct experiment:** reversed the scoring order
  (untrained first, influcoder second) in `_diag_influcoder_order.py` — the slowdown
  **flipped to whichever model ran first**. Whichever model of a given
  architecture/shape is scored FIRST in the process eats a one-time CUDA
  kernel/SDPA-path compilation cost, which `CostMeter.model_ready()` doesn't exclude
  (it only excludes model *loading* time, not first-forward-pass kernel warm-up).
- **Fix:** added a throwaway single-sample `model.encode(["warmup"], ...)` call
  immediately after `model.to("cuda")` and BEFORE `meter.model_ready()` in both
  `score_influcoder` and `score_semantic`.
- **Post-fix numbers:** `influcoder_68m` 99.75→**9.99** ms/sample (now only ~1.7x
  untrained's 5.75ms, plausible residual noise). `influcoder_150m` 54.18→**41.78**
  ms/sample — improved but **still ~4.3x** vs. untrained's 9.72ms — **this residual
  150m gap was NOT root-caused this session**, see §11.
- **General lesson, re-confirmed by Part 3 (§7.5):** ANY time a new script measures
  wall-clock timing on a first-of-its-kind call shape (new batch size, new kernel path),
  expect a warm-up artifact until proven otherwise by checking whether the rate is
  stable across repeated/larger calls.

### 10.5 GPU job walltime expired mid-session, not preempted

Even priority (`p3`) jobs expire at their requested walltime — this happened **twice**
this session (once for a LoGRA recompute job, once for Part 3's initial job pair) and is
not a preemption, just an expiry. `oarsh` returns `Cannot find cpuset file` once this
happens. **Lesson:** request a longer walltime up front for a session expected to run
several hours (this session moved to 2-4 hour walltimes for Part 3's jobs after the
first expiry), or expect to re-`oarsub` mid-session — reserving a fresh `gpu=1` job on
the same cluster/site is fast and was the fix used both times.

### 10.6 rennes NFS home is shared — no scp/cat-over-ssh needed for rennes-site jobs

Confirmed directly (byte-identical md5sum) that `/home` is the SAME filesystem between
the CPU-brain node and rennes GPU compute nodes (`abacus*.rennes...`). Just edit the
file locally (on the brain) and launch directly — no push step needed. This ONLY applies
within rennes — other sites (nancy, grenoble, sophia, etc.) have their own separate NFS
homes and DO need an explicit copy step if you ever run something there.

## 11. Explicitly NOT done yet / open next steps

- **Standalone-vs-combined figure axis mismatch (newly surfaced by this documentation
  pass).** `draft50x50_figure.png` (Part 1 standalone) uses linear-x; the combined
  figure's Panel 1 uses sqrt-x for the same data (§8.2). `part3_figure.png` (Part 3
  standalone) uses linear-x; the combined figure's Panel 3 uses log-x for the same data
  (§8.4). Nobody has asked for the standalone figures to be updated to match the
  combined one, so they haven't been — but if the standalone figures are ever used
  independently (not just as inputs to the combined one), be aware they currently tell
  the same story with different visual compression than the combined figure does.
- **Part 2's `lr=1e-5` is unexplained and disagrees with Parts 1 and 3** (§4.2.3) — the
  single highest-value thing to resolve before trusting Part 2's scaling curve in
  anything paper-facing. Ask the user directly, or re-run at `5e-5` and diff the curve.
- **Part 1's InfluCoder checkpoints use best-epoch selection; Part 2's sweep uses
  fixed-epoch-8** (§4.2.4) — whether this actually changes the saved 68m/150m
  checkpoints' quality vs. their own epoch-8 metric was not checked. Cross-reference
  `baselines/out/fig1/train_results.json`'s `best_epoch` field per checkpoint if this
  needs resolving.
- **Part 2's LESS reference lines are still on Part 1's 50x50 eval, not a fresh 400x400
  recompute** (§4.2.5) — same treatment LoGRA already got via
  `_logra_400x400_recompute.py` would resolve this; nobody has done the LESS equivalent.
- **Scale Part 1's eval past 50x50.** The design supports this directly:
  `_fig1_draft_50x50.py` slices `full_gt[:N_EVAL, :N_EVAL]` from the ALREADY-CACHED
  400x400 `fig1_dolci` GT — just bump `N_EVAL` up to 400, no need to touch anything else.
- **The 150m InfluCoder/untrained residual ~4.3x gap (§10.4) is unresolved.**
- **No FLOPs panel** — deliberately dropped this session (see `figure1_table.py`'s own
  `run_row(..., measure_flops=False)` default: FlopCounterMode roughly 5x's wall time for
  instrumentation alone and tells the same story as ms/sample anyway).
- **Part 3's exploratory 250x500 InfluCoder training set is not the final experiment**
  (§4.2.2) — the amortization numbers (§7.5) should be re-measured at whatever the real
  experiment's actual InfluCoder training-set size ends up being, with repeated trials
  for noise estimation, before being quoted as final.
