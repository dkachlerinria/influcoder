# EXP1: InfluCoder vs. LESS/LoGRA/RDS+/TF-IDF — unified speed-vs-quality comparison

Status as of 2026-07-26, end of session. Written for a fresh agent picking this up cold.
Read this before touching any of the scripts in `.tuning_logs/` — several of them encode
fixes for bugs that are easy to reintroduce if you "clean up" or rewrite from scratch.

## 0. What this experiment actually is

One unified script scores **every** candidate-selection method on the **same** eval split,
under **matched** experimental conditions (same attention implementation, same sequence
length cap, same rank held constant *within* each method's proxy family), and plots
**aggregate Spearman rho vs. inference ms/sample** on one chart. The point is a fair
cost/quality comparison — InfluCoder's tiny distilled encoders vs. LESS/LoGRA (gradient
methods, expensive but currently highest fidelity) vs. RDS+/TF-IDF (cheap baselines).

This is currently a **50x50 "peek"**, not the final result — small eval on purpose, to get
directionally-correct numbers fast. The design is meant to scale up cleanly (see §6).

## 1. Where this sits relative to the rest of the repo

- `baselines/figure1_table.py` + `baselines/plot_figure1.py` are the **pre-existing**,
  checked-in unified table/plot for the ORIGINAL fig1/fig1_dolci methodology (400x400,
  asymmetric LoGRA proxy ranks r=32, LESS at paper-default rank=128, eager attention
  pinned for FLOP-counter compatibility). Do not confuse these with this session's work.
- **Everything under `.tuning_logs/` is new, untracked, and preset-name-agnostic scratch
  work from this session.** It intentionally does NOT touch `figure1_table.py`,
  `plot_figure1.py`, or any `score_*` function's *default* argument values — it only
  ADDS new optional parameters (see §4) so existing call sites elsewhere in the repo are
  unaffected.
- The GT (ground truth) throughout is: **Qwen/Qwen3-4B, LoRA rank 16**, on the
  `fig1_dolci` preset (BBH anchors x dolci-instruct pool, `grad_max_len=1024`,
  `encoder_max_len=1024` — these two already happen to be equal in this preset, which
  matters, see §3 constraint 2). This preset and its cached GT/train-features were
  already used earlier this session for a separate InfluCoder-training reproduction
  investigation (see `dolci-non-reproduction` memory / `FINDINGS.md` /
  `INFLUCODER_FINDINGS.md`) — unrelated to this experiment except for sharing the same
  cached artifacts.

## 2. The methods being compared, and their current recipe

| Method | Code | Model size(s) | Rank | Notes |
|---|---|---|---|---|
| InfluCoder | `baselines/influcoder/score.py` | 68m, 150m encoders | n/a | trained via `train_fig1_encoders.py --preset fig1_dolci`, checkpoints at `runs_out/fig1_dolci/encoder_{68m,150m}` |
| untrained encoder | `baselines/semantic/score.py` | same 68m/150m arch | n/a | same architecture as InfluCoder pre-distillation, off-the-shelf `jhu-clsp/ettin-encoder-{68m,150m}` |
| LESS | `baselines/less/score.py` + `baselines/less/model_utils.py`, `less_embeds.py` | 4B, 1.7B, 0.6B (Qwen3 family) | **16** (uniform across sizes) | TRAK `BasicProjector` random projection of per-sample LoRA-SGD gradients |
| LoGRA | `baselines/logra/score.py` + `modeling_logra.py` | 4B, 1.7B, 0.6B | **8** (uniform across sizes) | custom autograd extracting compact per-sample `[r,r]` gradient factors, no dense projection |
| RDS+ | `baselines/rdsplus/score.py` | 4B only | n/a | forward-only, SGPT weighted-mean pooling of hidden states, no gradients |
| TF-IDF | `baselines/tfidf/score.py` | none | n/a | no model at all, ~0 cost |

**Why LESS uses rank=16, not its paper default of 128:** see §5.1 (OOM saga). This is a
real deviation from LESS's documented config, not just a memory-chunking change — label
it as such if this ever appears in a paper-facing table.

**Why LoGRA uses a *uniform* rank=8 across all three sizes, not the asymmetric r8(4B)/
r32(proxies) used in the older `figure1_table.py`:** explicit user instruction this
session — asymmetric rank confounds "smaller model" with "extra rank," uniform rank
isolates model size alone. This makes these LoGRA numbers **not directly comparable** to
the older `figure1_table.py`/`plot_figure1.py` output, which used r32 for the proxies.

## 3. The two fairness constraints (explicit user requirements) and how they're satisfied

1. **Attention implementation must be identical across every method that has one.**
   Chosen value: **`sdpa`** for everyone (not `eager`). Verified this session, method by
   method, that `sdpa` is passed through to the actual model-loading call with no
   silent override in between (see the trace in this session's final turns — every one
   of LESS/LoGRA/RDS+/InfluCoder/untrained forwards `attn_implementation` straight to
   its `AutoModelForCausalLM.from_pretrained(...)` / `SentenceTransformer(...,
   model_kwargs=...)` call). TF-IDF has no model, so this is moot for it, not a gap.
   - **Why `sdpa` and not `eager`:** several of these methods (LESS, RDS+, the encoder
     scorers) had `eager` HARDCODED with a stated reason — "so the FLOP counter can see
     the attention matmuls" (torch's FlopCounterMode can't handle SDPA on GQA models
     like Qwen3). But this experiment's `run_row()` calls `summarize(0, meter, ...)`
     directly — it **never wraps anything in `baselines.cost.flop_counter()`** — so that
     restriction doesn't apply here. `sdpa` is faster and more memory-efficient for
     every method that uses the Qwen3 LM family, so it was the natural uniform choice.
   - This required adding a new `attn_implementation` parameter (defaulting to the
     historical `"eager"`, so nothing else in the repo changes behavior) to:
     `baselines/less/model_utils.py::load_base_with_fresh_lora`,
     `baselines/less/score.py::score_less`, `baselines/rdsplus/score.py::score_rdsplus`,
     `baselines/semantic/score.py::score_semantic`,
     `baselines/influcoder/score.py::score_influcoder`. LoGRA already had this parameter
     (default `"sdpa"`) — no change needed there.
2. **Sequence length must be identical everywhere.** `MAX_LEN = 1024` is passed to every
   method. This is NOT a new value — it's simply `fig1_dolci`'s existing
   `grad_max_len` (LESS/LoGRA/RDS+) which happens to already equal its
   `encoder_max_len` (InfluCoder/untrained) — see `run.py`'s `PRESETS["fig1_dolci"]`.
   `_fig1_draft_50x50.py` has an assertion at startup that fails loudly if these two
   config values ever drift apart in the preset — **do not remove that assertion**, it's
   the only thing enforcing constraint 2 mechanically rather than by convention.

## 4. Code changes made this session (all additive, no default behavior changed)

- `baselines/less/model_utils.py`: `load_base_with_fresh_lora` gained
  `gradient_checkpointing: bool = False` and `attn_implementation: str = "eager"` params.
- `baselines/less/score.py`: `score_less` gained `block_size: int = 128`,
  `gradient_checkpointing: bool = False`, `attn_implementation: str = "eager"` params,
  threaded through to the calls above / `collect_grads`.
- `baselines/less/less_embeds.py`: `collect_grads` gained a `block_size: int = 128`
  parameter (previously **hardcoded** as a local variable inside the function — this was
  a real, separate bug source, see §5.1).
- `baselines/rdsplus/score.py`: `score_rdsplus` gained `attn_implementation: str =
  "eager"`.
- `baselines/semantic/score.py`, `baselines/influcoder/score.py`: gained
  `attn_implementation: str = "eager"`, AND both got a throwaway warm-up
  `model.encode(["warmup"], ...)` call inserted right after `model.to("cuda")` and
  before `meter.model_ready()` — this is a real bug fix, not a stylistic addition, see
  §5.4. **Do not remove this line** without understanding why it's there.
- `FINDINGS.md`: added a bullet under "Instrument / measurement hygiene" documenting the
  CUDA-warm-up-vs-InfluCoder finding (§5.4).
- `G5K.md` / `G5K-CPU-AGENT.md`: updated to state that **multiple concurrent GPU jobs
  are explicitly allowed** (previously said "never hold more than one active GPU job") —
  explicit user instruction this session. See §5.3 for the related `oarstat -J` bug this
  uncovered.

None of these edits change any *existing* call site's behavior (every new parameter
defaults to the old hardcoded value).

## 5. Where we got stuck — read this before repeating any of it

### 5.1 LESS OOM saga (three rounds, do not redo the first two)

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
   exists (parameter kept, defaults to False) but is no longer the active fix. Don't
   assume `gradient_checkpointing=True` is still needed anywhere in the current
   `_fig1_draft_50x50.py` pipeline — it explicitly sets it to `False`.

**A tangential finding surfaced by this saga:** a quick isolated test
(`_less_blocksize_test.py`) at block_size=16 vs 32 gave only a **1.03x** difference —
confirms the projection/chunking step is *not* where LESS's time goes; the dominant
cost is the full-model forward+backward, independent of the projector.

### 5.2 "Is this a REAL speedup" — proxy-model speed methodology (earlier in session)

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

### 5.3 GPU discovery bug: `oarstat -J`'s `assigned_hostnames` field is always `None`

When surveying all of G5K for large idle GPUs (after the user allowed running multiple
concurrent GPU jobs), an initial script matched `oarstat -J`'s (JSON output)
`assigned_hostnames` field against `oarnodes -J`'s `network_address` to determine which
GPUs were occupied. **This field is always `None` in the JSON output** — even for jobs
we KNEW were `Running` at the time (verified by checking our own known job IDs
directly). This made every single GPU look "free," which was wrong (cross-checked
against a job's own `scheduled_start` prediction, which implied real contention).
**Fix:** use `assigned_resources` (a list of integer resource IDs) instead, matched
against each node entry's own `resource_id` field. `oarstat -fj <id>` (plain text
format, not `-J`) DOES populate `assigned_hostnames` correctly — the bug is specific to
the JSON output path. If you write another multi-site GPU survey script, use
`assigned_resources`/`resource_id`, not `assigned_hostnames`, or re-verify this hasn't
silently changed.

### 5.4 The big one: InfluCoder measured ~17x slower than the untrained encoder — CUDA warm-up artifact, not real

This was the most confusing finding of the session and is fully written up in
`FINDINGS.md`'s "Instrument / measurement hygiene" section — read it there for the
canonical version. Summary for context:

- **Symptom:** first pass of the unified 50x50 table gave `influcoder_68m` = 99.75
  ms/sample vs. `untrained_68m` = 5.73 ms/sample (17x). `influcoder_150m` = 54.18 vs.
  `untrained_150m` = 9.69 ms/sample (5.6x). Both pairs are the SAME architecture
  (ModernBERT, verified identical `num_hidden_layers`/`hidden_size`/param count) —
  only the WEIGHTS differ (trained vs. off-the-shelf).
- **Ruled out (verified identical between the two, so NOT the cause):** dtype (both
  fp32), resolved `attn_implementation` (both `sdpa`), tokenized sequence length (both
  802 tokens for an identical test string).
- **Root cause, confirmed by a direct experiment:** reversed the scoring order
  (untrained first, influcoder second) in `_diag_influcoder_order.py` — the slowdown
  **flipped to whichever model ran first**. Whichever model of a given
  architecture/shape is scored FIRST in the process eats a one-time CUDA
  kernel/SDPA-path compilation cost. `CostMeter.model_ready()` only excludes model
  *loading* time from the measured `ms/sample` — it does NOT exclude first-forward-pass
  kernel warm-up, which happens on the first real `model.encode()` call, i.e. inside
  what gets counted as "inference time."
- **Fix:** added a throwaway single-sample `model.encode(["warmup"], ...)` call
  immediately after `model.to("cuda")` and BEFORE `meter.model_ready()` in both
  `score_influcoder` and `score_semantic`. This folds the warm-up cost into
  (excluded) load time instead of (measured) inference time.
- **Post-fix numbers:** `influcoder_68m` 99.75→**9.99** ms/sample (now only ~1.7x
  untrained's 5.75ms, plausible residual noise, not a real difference).
  `influcoder_150m` 54.18→**41.78** ms/sample — improved but **still ~4.3x** vs.
  untrained's 9.72ms. **This residual 150m gap was NOT root-caused this session** —
  possibilities not yet checked: the single-sample warmup call may not trigger the same
  kernel-selection path as the real encode's batch_size=32 (batch-size-specific kernel
  autotuning), or a tokenizer-side difference specific to the 150m checkpoint. If you
  pick this up: try a warmup call that matches real batch_size/seq_len shapes, or
  profile with `torch.profiler` to see where the extra 30ms/sample actually goes.
- **General lesson for whoever extends this table further:** ANY time two rows share
  the exact same underlying architecture (only weights differ), this warm-up
  confound can reappear. Right now only InfluCoder/untrained share an architecture
  (LESS's/LoGRA's three model sizes are each architecturally distinct — different
  hidden dims — so this specific confound does not apply between them, though a
  general "warm up before timing" habit is good practice anywhere).

### 5.5 GPU job walltime expired mid-session, not preempted

The rennes `p3` priority job used for most of this session's GPU work
(`OAR_JOB_ID=3943428`, `abacus11`) silently expired (hit its 2-hour `walltime`) partway
through — `oarsh` started returning `Cannot find cpuset file ... Cannot find cpuset
file /dev/cpuset//oar/dkachler_3943428/tasks`, which per `G5K-CPU-AGENT.md` §1 means
"preempted / job ended," but in this case it was a plain walltime expiry, not a
besteffort preemption (this WAS a `p3` priority job, not besteffort). **Lesson:** even
priority jobs expire at their requested walltime — request a longer walltime up front
for a session expected to run several hours, or expect to re-`oarsub` mid-session
(reserving a fresh `gpu=1` job on the same cluster/site is fast and was the fix used
here — see `oarsub -q p3 -p "cluster='abacus11'" -l gpu=1,walltime=2:00:00 'sleep
7200'`).

### 5.6 rennes NFS home is shared — no scp/cat-over-ssh needed for rennes-site jobs

Confirmed directly (byte-identical md5sum) that `/home` is the SAME filesystem between
the CPU-brain node (`parasilo-9.rennes...`) and rennes GPU compute nodes
(`abacus*.rennes...`). Earlier in the session, every script edit was manually pushed via
`cat > file` over a nested `ssh frennes "oarsh <node> '...'"` hop before this was
verified — **that push step is unnecessary for any job on a rennes-site node**. Just
edit the file locally (on the brain) and launch directly. This ONLY applies within
rennes — other sites (nancy, grenoble, sophia, etc.) have their own separate NFS homes
and DO need an explicit copy step if you ever run something there.

## 6. Current results (50x50 peek, all methods, sdpa + max_len=1024 uniform)

From `baselines/out/fig1_dolci/draft50x50_table.json` (post all fixes in §5.4):

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

## 7. Plots (Figure 1)

This experiment produces a three-part Figure 1:
- **Part 1: Cost vs Quality.** Aggregate Spearman rho vs inference ms/sample for all methods.
- **Part 2: InfluCoder Scaling.** Aggregate Spearman rho vs number of training anchors, showing how InfluCoder 68m improves with more data, with horizontal baselines for LoGRA.
- **Part 3: GPU-time-vs-samples-processed amortization curve.** At what cumulative GPU time does InfluCoder's fixed setup cost pay for itself against LESS's/LoGRA's per-sample cost?

### Part 1: Cost vs Quality
`.tuning_logs/_plot_draft50x50.py` generates `baselines/out/fig1_dolci/draft50x50_figure.{png,pdf}` from the JSON table above. Reuses this repo's validated categorical palette
(`C_GRAD` blue = LESS/LoGRA fwd+bwd methods, `C_FWD` orange = InfluCoder/untrained/RDS+
single-forward methods, `C_FREE` aqua = TF-IDF's horizontal reference line). Current
design (as of end of session, after several user-requested tweaks):
- **Linear x-axis** (NOT log — user explicitly asked for this, overriding
  `plot_figure1.py`'s log-x convention).
- **No Pareto frontier.** Instead, a solid line connects each method's own proxy family
  in size order (4B → 1.7B → 0.6B) for LESS and for LoGRA separately — shows the
  within-family size/cost/quality trend directly, which is more relevant to this
  experiment's actual question than a cross-method frontier.
- Marker shape distinguishes LESS (circle) from LoGRA (square) since both now share the
  blue gradient-method color across 3 sizes each (the older single-LESS-row script
  didn't need this since there was only one LESS point).
- Marker size scales with model size, consistently across both gradient methods AND the
  encoder sizes (150m > 68m; 4B > 1.7B > 0.6B) — "bigger marker = bigger underlying
  model" reads the same way everywhere on the plot.
- Filled = trained (InfluCoder, LESS, LoGRA, RDS+), hollow = untrained encoder only.

Just rerun `.venv_h100/bin/python .tuning_logs/_plot_draft50x50.py` from the repo root
after any table update — it's idempotent and fast (no GPU needed, pure plotting).

### Part 2: InfluCoder training-sample scaling

**Question:** how does InfluCoder's distillation quality scale with the number of
training samples, holding everything else (epochs, lr, hard_ratio, encoder_max_len,
encoder size=68m) at the exact `fig1_dolci` recipe from `train_fig1_encoders.py`? Fixed
1:2 anchor:pool ratio throughout (matches `fig1_dolci`'s own 1500:3000 config).

**Script:** `.tuning_logs/_exp1_part2_scaling.py` (main sweep, `SIZES = [25, 50, 100, 250,
500, 750, 1000, 1500]`) plus two follow-on copies that only exist to fill in extra points
without re-running the whole sweep: `_exp1_part2_scaling_extra.py` (`SIZES = [125, 175,
1500]`, writes `scaling_68m_400x400_extra.json`) and `_exp1_part2_scaling_1500.py` (a
single-point rerun, `scaling_68m_400x400_1500.json`, used as a spot-check). All three
share the identical `run_size()` body — if you add another size, copy the pattern rather
than editing `SIZES` in the main script and losing the checkpoint-resume behavior (each
script's `main()` skips any `n_a` already present in its own output JSON, so it's safe to
re-launch after a kill).

**Mechanism:** reuses `baselines.scaling_sweep.train_features()` for the cached
Qwen3-4B featurization (same cache file as Part 1,
`trainfeat_Qwen_Qwen3-4B_1500x3000_r16_s0_eval400x400.pt` — zero new GPU-expensive
featurization for any size up to 1500 anchors), then for each `n_a` slices
`targets = g_ta[:n_a] @ g_tp[:n_p].T` and calls `distill()` fresh on a newly-loaded 68m
encoder.

**Important correction made mid-session — read this before trusting any older point:**
the very first version of this sweep called `distill(..., select_best_on="aggregated")`
and reported `log["epoch_metrics"][log["best_epoch"]]` — i.e. whichever epoch scored
highest on the eval callback, which is `distill()`'s own internal best-epoch-restore
behavior (see `influcoder/encoder.py`'s `distill()` docstring: it restores the encoder to
its best-scoring epoch before returning, specifically because eval Spearman is noisy at
small sample counts and "reliably peaks then degrades"). **The user explicitly rejected
this for this experiment**: "dont keep the best epoch, keep the same epoch (8)" — every
sweep point should reflect training for the FULL fixed epoch budget, not a cherry-picked
epoch, since the x-axis here is "how much data," not "how well can you early-stop." Fix:
`run_size()` now uses `final_idx = epochs - 1; final = log["epoch_metrics"][final_idx]`
instead of `log["epoch_metrics"][log["best_epoch"]]` — `epoch_metrics[-1]` is always the
metrics computed at the end of the LAST epoch (8), recorded by the `epoch_eval` callback
*before* `distill()`'s internal restoration runs, so no change to `influcoder/encoder.py`
itself was needed. `select_best_on="aggregated"` is still passed to `distill()` (harmless
— it only controls what gets restored into the returned `enc` object, which these scripts
`del enc` immediately after anyway) but is no longer what gets *reported*.
The three already-completed points from the OLD (best-epoch) methodology were discarded
and the full sweep was rerun from scratch under the fixed-epoch-8 methodology — do not
trust any scaling number that isn't from a run using `final_idx`/`epoch_metrics[-1]`.

**A second change made at the same time, also on top of the original design:** `N_EVAL`
was bumped from 50 to **400** (full `fig1_dolci` eval, not the Part-1 50x50 slice) and
`lr` was explicitly pinned to `1e-5` in the `distill()` call (overriding its `5e-5`
default) — **this second change (lr=1e-5) was NOT an explicit user instruction observed
in this transcript**; it appears to have been made either by the user directly editing
the file, or by an earlier version of this same session before a context-compaction
boundary, and its rationale is not recorded anywhere. If you need to know why lr=1e-5 was
chosen over the `train_fig1_encoders.py` recipe's actual `5e-5` default, that reasoning is
lost — flag this to the user if it matters, don't assume it was validated the way the
epoch-8 fix was.

**Current results** (`baselines/out/fig1_dolci/scaling_68m_400x400.json` +
`_extra.json` combined, all fixed-epoch-8, all on the full 400x400 eval,
`untrained_agg` is identical across every row since it's the same untrained 68m encoder
scored on the same eval before any training):

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
750→1500, only buys +0.023 vs. the +0.10 gained going 100→250). Note this
`untrained_agg` (+0.3171) is a DIFFERENT number from Part 1's `draft50x50_table.json`
`untrained_68m` row (+0.2074) — expected, since Part 1's untrained number is on the 50x50
eval slice and this is the full 400x400 eval; don't treat this as a bug, they're
deliberately different eval sizes measuring the same thing.

**Plot:** `.tuning_logs/plot_scaling_part2.py` generates
`baselines/out/fig1_dolci/scaling_figure.{png,pdf}`. **RESOLVED** (was previously
pulling unverified reference lines from `table1.json` — see history below): the user
was asked whether to trust `table1.json`'s pre-existing LoGRA numbers or recompute fresh,
and explicitly chose to recompute. Fresh values now live in
`baselines/out/fig1_dolci/logra_400x400_r8.json`, produced by
`.tuning_logs/_logra_400x400_recompute.py` — the exact same call pattern as Part 1's
LoGRA rows (`score_logra(..., lora_rank=8, max_len=1024, attn_implementation="sdpa")`,
GT=Qwen3-4B rank16) but on the FULL unsliced 400x400 splits instead of Part 1's 50x50
slice:

| row | agg rho (400x400, r8, sdpa) | vs. table1.json's (unverified) value |
|---|---:|---:|
| logra_4B | +0.8942 | +0.9079 (close — within noise) |
| logra_1.7B (proxy) | +0.4309 | +0.5779 (notably different) |

The 1.7B row also cross-checks against an independent earlier-this-session run
(`baselines/out/fig1_dolci/logra_uniform_r8.json`'s `logra_proxy_1.7B_r8` = +0.4299,
computed via a near-identical script, `.tuning_logs/_logra_uniform_r8.py`, same rank/eval
size but relying on `score_logra`'s attn default rather than passing it explicitly) —
the two independent runs agree to within 0.001, which is exactly the kind of consistency
`table1.json` was lacking. **The `plot_scaling_part2.py` script and the current
`scaling_figure.png`/`.pdf` now use these recomputed, cross-verified values, NOT
`table1.json`.** The InfluCoder curve crosses the 1.7B LoGRA line almost immediately
(already above it at the smallest tested size, n=75 total) and does not reach the 4B
LoGRA line anywhere in the tested range (max +0.7676 at n=4500 vs. +0.8942).

**`table1.json`'s own internal inconsistency (its `influcoder_68m` = +0.0466 vs. this
session's +0.7676 at a comparable/larger training size) is still unexplained** — it
remains flagged in `FINDINGS.md` as the likely source of the `dolci-non-reproduction`
memory's still-open "~+0.78 vs ~+0.05" discrepancy. Nothing above resolves that; it only
means Part 2's own plot no longer depends on trusting that file.

### Part 3: GPU-time-vs-samples-processed amortization curve

**Question:** at what point does using InfluCoder actually become cheaper than just
running LESS/LoGRA directly, in real cumulative GPU time? Explicit user request, verbatim
intent: test LoGRA 1.7B, LESS 4B, and InfluCoder 68m, timing how long each takes to
"process" 100 / 1K / 10K samples, where **"process" means computing the per-sample
representation only** (LESS: TRAK-projected per-sample gradient via `collect_grads`;
LoGRA: per-sample `[r,r]` gradient factor via `LoGra.encode()`; InfluCoder: encoder
embedding via `embed()`) — explicitly excluding any anchor-vs-pool scoring matmul, which
is out of scope here (depends on query-set size, "a can of worms" for a separate
experiment). LESS/LoGRA were only run at n=100/1000 (cost-prohibitive beyond that, per
explicit instruction) and extrapolated to 10K from the measured per-sample rate; InfluCoder
was run at n=100/1000/10000 directly. InfluCoder's distillation set for this exploratory
run was 250 anchors x 500 pool (not the final experiment's real train size — "just to see
how the curve will look like").

**Scripts:** `.tuning_logs/_part3_less_logra_process.py` (LESS 4B r16 + LoGRA 1.7B r8,
both `sdpa`, `max_len=1024`) and `.tuning_logs/_part3_influcoder_process.py` (InfluCoder
68m). Run as two separate OAR jobs in parallel on the same GPU type (`abacus11`, 24GB
card) per explicit user permission to parallelize across instances.

**Why dolci-instruct specifically (the pool all three scripts draw samples from):** it's
the only one of this repo's three sample pools that can supply real, non-duplicated
samples at every scale this experiment needs. BBH is hard-capped at 6511 total examples;
the local Dolly file has only 15011 rows; `tasksource/dolci-instruct` streams from ~1.8M
rows across 8 parquet shards, comfortably covering 100 through 100K+.

**Bug hit and fixed: the LESS/LoGRA script initially OOM'd on its very first batch
(n=100).** Root cause: it hardcoded `block_size=128` for `collect_grads`'s
`BasicProjector`, silently ignoring the exact fix already documented in §5.1 above (this
24GB card requires `block_size=16` at `lora_rank=16` — `block_size=128` was one of the
three root causes of the original LESS OOM saga, not a new bug, just the same old fix not
being carried over into a new script). Fixed by setting `block_size=16`; reran clean.
**Lesson: any new script that calls `collect_grads` directly needs this same fix applied
by hand — it lives in each call site, not in a shared default.**

**Results** (`baselines/out/fig1_dolci/part3_influcoder_process.json` +
`part3_less_logra_process.json`):

| method | model | setup/load time | n=100 | n=1000 | n=10000 |
|---|---|---:|---:|---:|---:|
| InfluCoder | 68m (250x500, epochs=8) | 459.58s (356.42s collect + 103.16s train) | 43.95 ms/sample* | 11.01 ms/sample | 10.81 ms/sample |
| LESS | 4B, r=16, sdpa | 74.06s (model load) | 1123.14 ms/sample | 1091.67 ms/sample | not run (extrapolated) |
| LoGRA | 1.7B, r=8, sdpa | 35.77s (model load) | 218.38 ms/sample | 217.63 ms/sample | not run (extrapolated) |

\* InfluCoder's n=100 figure is inflated by a batch_size=32 CUDA-kernel warmup artifact on
the first `embed()` call at that batch shape (same class of issue as §5.4's warmup finding,
but for a new kernel shape, not a cold GPU) — n=1000/10000 are consistent with each other
(~10.8-11.0 ms/sample) and are the trustworthy steady-state rate.

**Linearity check (the user explicitly asked to be checked on this): confirmed for all
three methods** — LESS varies only ~3% between n=100 and n=1000 (1123.14 -> 1091.67
ms/sample); LoGRA varies <0.4% (218.38 -> 217.63 ms/sample); InfluCoder's steady-state
(post-warmup) rate is effectively flat (11.01 -> 10.81 ms/sample, n=1000 -> n=10000).
Extrapolating LESS/LoGRA's measured per-sample rate to 10K is therefore sound: LESS would
take ~10,917s (~3.03 hours) for 10K samples; LoGRA ~2,176s (~36.3 min).

**Amortization points** (`.tuning_logs/plot_part3.py`, solving
`InfluCoder_samples(t) = other_samples(t)` from each method's own
setup/load-time-plus-constant-rate model):

- **InfluCoder overtakes LESS at ~463s of cumulative GPU time (~357 samples processed).**
- **InfluCoder overtakes LoGRA at ~482s of cumulative GPU time (~2049 samples processed).**

Both crossovers land within seconds to tens of seconds of InfluCoder's own setup finishing
(459.6s) — not a gradual catch-up. This is a direct consequence of the per-sample speed
gap being so large (InfluCoder ~101x faster than LESS, ~20x faster than LoGRA at steady
state) that once InfluCoder starts processing, it doesn't need to "catch up" so much as
immediately overtake: LESS has only processed ~407 samples by the time InfluCoder's setup
finishes, and InfluCoder covers that same ground before LESS gets meaningfully further
ahead.

**Plot:** `.tuning_logs/plot_part3.py` generates `baselines/out/fig1_dolci/part3_figure.{png,pdf}`.
X-axis = cumulative GPU time (seconds, linear); Y-axis = cumulative samples processed
(log scale, per explicit request — a log y-axis can't render 0, so each curve is drawn as
a flat segment at a `FLOOR=0.5` visual floor during its setup/load phase, purely a
plotting convention and not a real value). Reuses `C_FWD` orange for InfluCoder; LESS and
LoGRA get their own blue/purple (previously both shared one `C_GRAD` blue in Parts 1/2,
which worked there since they never appeared as separate curves needing to be
told apart on the same axes — here they do).

**Caveat — not a final number.** This is explicitly an exploratory run (250x500
distillation set, single measurement per size, no repeated trials for noise estimation)
meant only to confirm the shape of the curve exists and roughly where it crosses. Don't
quote the exact "463s" / "357 samples" figures as final results without re-running at the
real experiment's actual train size and with repeated trials.

## 8. Explicitly NOT done yet / open next steps

- **Scale the eval past 50x50.** The design supports this directly: `_fig1_draft_50x50.py`
  slices `full_gt[:N_EVAL, :N_EVAL]` from the ALREADY-CACHED 400x400 `fig1_dolci` GT
  (built once via `ground_truth(...)`, cheap to refetch) — just bump `N_EVAL` up to
  400 for the full eval, no need to touch anything else. Runtime will grow
  ~linearly-ish with `N_EVAL^1` for the O(A+P) forward-pass-count methods and
  faster than that isn't expected for any of these (all are pairwise-cosine after
  independent per-sample featurization, no O(A*P) model cost).
- **The 150m InfluCoder/untrained residual ~4.3x gap (§5.4) is unresolved.**
- **No FLOPs panel** — deliberately dropped this session (see `figure1_table.py`'s own
  `run_row(..., measure_flops=False)` default and its docstring: FlopCounterMode
  roughly 5x's wall time for instrumentation alone and tells the same story as
  ms/sample anyway). If FLOPs are ever needed for a paper-facing table, that's a
  separate, deliberately deferred appendix experiment — don't casually turn
  `measure_flops` back on inside a script that's also trying to get a clean timing
  read, the two are in tension.
- **LESS/LoGRA proxy quality was measured via real Spearman rho in this experiment**
  (not just timing) — unlike an earlier, separate speed-only investigation this session
  (`_less_proxy_speed.py`) which explicitly did NOT compute quality. Don't confuse the
  two — the numbers in §6 ARE real quality numbers, not just speed extrapolations.
- **Multiple GPU jobs are allowed now** (see §5.3's context and the `G5K.md`/
  `G5K-CPU-AGENT.md` edits) — if scaling up to 400x400, consider splitting LESS/LoGRA's
  three model sizes each across separate concurrent GPU reservations to cut wall-clock,
  now that this is explicitly sanctioned. Nothing about the methodology requires
  sequential execution; it was only done sequentially this session because the 50x50
  peek was already fast enough not to bother.
- **`lr=1e-5` in the Part 2 sweep's `distill()` call is unexplained** (see §7's Part 2
  subsection) — it silently overrides `train_fig1_encoders.py`'s actual recipe default of
  `5e-5`, and no rationale for this specific override was found in this session's
  transcript. Worth asking the user directly if it matters for how these numbers get used.
