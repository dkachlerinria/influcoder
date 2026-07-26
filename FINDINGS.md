# Findings

_Editable digest of what we currently believe, blunt and falsifiable. New results append or amend a
line. This is the shareable summary; per-run detail (every tuning round, every bug writeup) is in
`INFLUCODER_FINDINGS.md`._

## Core (Figure 1, Dolly pool)

- **InfluCoder dominates the cheap end of the cost/quality Pareto frontier.** At identical inference
  cost, distillation roughly doubles the untrained-encoder baseline at every size (68m +0.39→+0.78,
  150m +0.33→+0.76, 400m +0.43→+0.81 agg ρ vs. Qwen3-4B-rank16 GT). LESS (+0.96, 358ms) and LoGRA r8/4B
  (+0.90, 144ms) still lead on raw fidelity but at 35-275x the cost.
- **LoGRA proxy quality does not transfer uniformly across model size.** At the same rank as the 4B
  anchor (r=8): 1.7B proxy +0.29, 0.6B proxy **-0.16** (worse than random). Raising proxy rank to 32
  (asymmetric vs. the 4B's r=8) rescues the 1.7B proxy (+0.29→+0.51, plateaus by r16) but **not** the
  0.6B proxy (stays pinned near zero, -0.03 to +0.03, at every rank tested) — its failure is the 0.6B
  model's gradient geometry not correlating with the 4B's influence signal, not an adapter-capacity
  problem. FIM-preconditioned LoGRA is much worse than raw at every rank/proxy — use raw.
- **LoGRA rank barely moves inference cost.** The pinv/FIM term is ~0.02% of total FLOPs; the dominant
  cost is the rank-independent frozen-backbone forward+backward. An apparent 38% rank-4-vs-8 timing gap
  was a same-process OS-page-cache-warmup artifact, not a rank effect (isolated compute-only comparison:
  ~5%, noise-level).
- **Batching LoGRA is a real but unsafe lever — rejected for the default methodology.** Where it fits
  (proxies only; the 4B model already OOMs at batch=1, 21.9GB/46GB), batch=4 buys 7-18% speedup, but
  `modeling_logra.py` computes one batch-*mean* loss before backward, so a sample's "per-sample gradient"
  at batch>1 is diluted by whatever else shares its batch — verified real score drift up to **0.126**
  (1.7B) / **0.064** (0.6B), far past noise. `batch_size=1` stays the correctness standard; a batch=4
  override exists only as an explicit, flagged, user-requested variant (`update_proxy_rows_batched.py`).
- **SDPA attention is a free, verified win — adopted as default.** vs. eager: -30-35% memory, -10-28%
  time at batch=1, zero batching needed. Verified safe (eager-vs-sdpa Δagg = +0.0046, noise-level) before
  adopting. Also reveals a real, if narrow, size-proportional speed ordering (144/130/126ms across
  4B/1.7B/0.6B) that eager's memory-bound overhead had been masking.

## Instrument / measurement hygiene

- **LoGRA is unseeded upstream** (`kaiming_uniform_`, no seed on `logix_lora_A`/`_C`). Single draws drift
  ±0.03-0.14 depending on rank/variant, and at 4B/r8 specifically a 5-draw spread was [0.68, 0.92] (mean
  +0.8231, std 0.087) — **use a multi-seed mean as any target, never a single draw.**
- **Two silent-corruption bugs found and fixed this session, both classic "looks like a bad result, is
  actually a bug" traps:**
  1. GT/train-feature cache key was missing eval size — `fig1` (n_eval_a=400) silently cache-hit
     `paper200x3`'s cached gradients (same train size, different eval size), training an encoder against
     targets for the wrong underlying samples. Symptom: eval Spearman whipsawing despite smoothly falling
     loss. Fixed by adding eval size to the cache key.
  2. `load_encoder()` hardcoded a 512-token cap. Fine for Dolly (median ~120 encoder-text tokens), silently
     truncated dolci-instruct's much longer text (median ~400, p90 ~1000) — 24% of samples had <50% of
     their answer visible to the encoder, 5.6% had none, while the GT gradient target saw far more of the
     answer via a target-prioritized truncation scheme. Fixed via a per-preset `encoder_max_len` (1024 for
     dolci-instruct, matching `grad_max_len`; Dolly untouched at 512).
  3. (Smaller) `torch.tensor(rng.sample(remaining, 0))` defaults to float32 when `n_random=0`
     (`hard_ratio=1.0`), silently producing a float index tensor that crashes later at the actual
     indexing site — fixed with explicit `dtype=torch.long`.
- **General lesson:** sequential same-process model reloads across configs confound wall-clock timing via
  OS page-cache warmup — a clean cost comparison needs fresh processes or an explicit cache-warm step.
- **InfluCoder vs. untrained-encoder timing is confounded by CUDA kernel/SDPA-path warm-up, not just OS
  page cache.** `influcoder_*` and `untrained_*` are the SAME architecture (only weights differ), so
  whichever one is scored FIRST in a process eats the first-call CUDA kernel/SDPA-path compilation cost
  inside its measured *inference* time (`CostMeter.model_ready()` only excludes model loading, not
  first-forward-pass warm-up) — e.g. one draft run measured influcoder_68m at 99.75ms/sample vs.
  untrained_68m at 5.73ms (17x), which flipped to untrained being the slow one when scoring order was
  reversed. Fixed by adding a one-sample throwaway `model.encode(...)` call right after `model.to("cuda")`
  and before `meter.model_ready()` in both `baselines/semantic/score.py` and
  `baselines/influcoder/score.py` — folds the warm-up cost into (excluded) load time instead of (measured)
  inference time. Post-fix: influcoder_68m dropped to ~10ms/sample, much closer to untrained's 5.75ms.
  A residual ~4x gap remained at 150m even after the fix (smaller than before, not fully explained —
  possibly warm-up specific to the real batch shape (32) vs. the single-sample warmup call, or tokenizer-
  side) — not yet root-caused.

## InfluCoder training-tuning (13 rounds on Qwen3-4B GT, condensed)

**Reference config at pause: 400m encoder, 1500x3000 train (`paper200x3`), lr=5e-5, epochs=8,
hard_ratio=0, alpha=0.3 → agg ~+0.796-0.798.** Gap to the LoGRA r8/4B target (+0.8231, 5-draw mean):
**~-0.025 to -0.027**, the closest result reached. Paused (not abandoned) to build Figure 1. Below is
every axis that was actually swept, split into what helped, what didn't, and what values to start from.

### What works / good starting values

- **Train size 1000x2000 to 1500x3000** (`paper200x2`/`paper200x3`) is the single biggest lever found —
  500x1000→1000x2000 alone was +0.052, more than 10x every other individual lever. 1000x2000 and
  1500x3000 are statistically tied once seed noise is accounted for (2-seed means: +0.7659 vs. +0.7667,
  std ~0.01) — either is a safe default; **don't bother chasing 2000x4000+** (see below).
- **Hard-negative mining (`hard_ratio`) genuinely helps, but only once there's enough data.** It *hurt*
  at 500x1000 (both flat 0.5 and a 0.0→0.5 curriculum ramp lost to hard_ratio=0), but at 1000x2000+ it's
  a real, repeatable +0.005 to +0.02 win. **Rule of thumb: hard_ratio ≈ 0.5-0.6 at 1000x2000-1500x3000,
  climbing toward ~0.75 at 2000x4000** — the right ratio scales up with pool size, because a bigger
  pool needs more hard fraction before the negatives are diluted back down to "all easy."
- **Bigger encoders (150m, 400m) are viable and can win — but only with enough data.** At 1500x3000, 400m
  with **lr=5e-5 (the same LR that's default for 68m!) and no hard mining** hit **+0.7960**, the best
  single InfluCoder config found this session; extending to 16 epochs nudged it to +0.8017 (judged
  within noise, not worth the 2x compute — see below). 68m remains the safe default at smaller data
  scales (≤500x1000) or when compute is the binding constraint (68m trains ~10-40x faster than 400m for
  a result in the same ballpark once data is adequate).
- **Alpha (listwise-KL weight vs. global-Pearson weight) has a small, real, direction-consistent optimum
  around 0.2-0.3** (default is 0.5). Effect size is small (~0.01-0.02, close to the seed-noise floor) but
  the *direction* — lower alpha (more listwise KL, less Pearson) is better — replicated across both 68m
  and 400m, and across a full 0.0→0.7 sweep (0.0 and 1.0-ish extremes are worse than the 0.2-0.3 interior,
  so it's a real interior optimum, not a monotonic "always lower" trend).
- **Defaults for everything else are already good**: `lr=5e-5`, `weight_decay=0.01`, `max_grad_norm=1.0`,
  `temperature=0.05`, `m_candidates=16`, `k_anchors=8`, `grad_accum_steps=1`, linear LR decay, `epochs=8`
  — every deviation tried from these (see below) was flat-to-negative. If you're setting up a new sweep,
  start here rather than re-testing these axes from scratch.

### What doesn't work (tested and rejected — don't re-run these)

- **`grad_accum_steps` 1→4**: no benefit, tested on 68m (-0.018) *and* 400m (-0.031). Smooths the loss
  curve visually but never improves the eval number. Dead lever regardless of encoder size.
- **Bigger batch composition (`m_candidates` 16→24/32, `k_anchors` 8→12/16)**: consistently worse.
  m_candidates=24 was the single worst regression found on that axis (-0.044); m_candidates=32 recovered
  partway but still lost to 16; k_anchors=12/16 lost by -0.03 to -0.033. More candidates/anchors per
  step dilutes the loss with more easy negatives — the opposite of what "more compute per step" should
  buy. Confirmed on both 68m and 400m.
- **Cosine LR schedule**: worse than linear decay (-0.018) at this epoch budget. Not revisited.
- **LR off 5e-5 in either direction**: 2e-5 (-0.02), 1e-4 (worse, though "still rising at epoch 8" so not
  fully conclusive), and for 400m specifically 3e-5 (-0.01) and 2e-5 (-0.023, though this was *before* the
  data-scale explanation was found — see below). 5e-5 wins every head-to-head run.
- **Temperature off 0.05 in either direction**: 0.02 (sharper, -0.005) and 0.10 (softer, -0.01) both
  lose. Local optimum confirmed, not pursued further.
- **`weight_decay=0`** (-0.016) and **`max_grad_norm=0.5`** (-0.009): both worse than the defaults.
- **More epochs almost never helps, and can actively hurt.** 68m: 8→16 epochs at 500x1000 bought
  *nothing* (peaks by epoch 4-5 regardless, then just overfits from the same-ish checkpoint) — epoch
  budget is not the bottleneck for 68m. 400m: 8→16 epochs at 1500x3000 gained +0.0057, judged noise-sized
  and **not worth 2x the compute** — standardized on 8 epochs.
- **Pure hard negatives (`hard_ratio=1.0`) is a hard cliff, not a graceful falloff.** Collapses to +0.474
  — far below hard_ratio=0.75's +0.769 at the same pool size. With every candidate near-maximally similar
  to the anchor, there's no easy contrast left for the ranking loss to calibrate against and the signal
  degenerates. Never use ratio=1.0 at any pool size.
- **More data alone, without proportionally more hard mining, regresses.** 2000x4000 (`big4x`) at
  hard_ratio=0 scored *worse* (+0.750) than 1000x2000 at hard_ratio=0 (+0.765) — bigger pool dilutes
  random negatives faster than it adds signal. Fixed by raising hard_ratio in step (see above), but
  data size alone past ~1500x3000 is not a free lunch.
- **The original "150m/400m are worse than 68m" conclusion was real at small data but wrong in general** —
  don't cite it standalone. At 500x1000/no-hard-mining both bigger encoders lost (150m -0.042, 400m
  collapsed within 1 epoch, -0.095 by final epoch). A learning-rate-mismatch theory was tested (lower LR
  for 400m) and *did* fix the instability (lr=1e-5 → smooth curve, +0.7385) but still didn't make it
  competitive; only adding data (1500x3000) actually closed the gap, and at that point the **original**
  LR (5e-5) turned out best after all (+0.796 beats lr=2e-5's +0.7735) — the two-step "wrong theory, then
  right one" is worth remembering before re-deriving it.
- **Ensembling (multi-seed score averaging) reduces variance, it does not raise the ceiling.** 3-seed
  average (+0.7781) beat the solo mean (+0.7683, seeds 0.7643/0.7582/0.7825) by only +0.0098, and did
  *not* beat the single best individual seed (+0.7825). Use it only if you specifically need a
  lower-variance estimate, not as a way to systematically beat your best single run.
- **LoGRA's FIM-preconditioned variant is much worse than raw at every rank and every proxy tested**
  (e.g. 1.7B r16: raw +0.50 vs. fim +0.08) — never the safe default choice it sounds like; use raw.
- **Batching LoGRA (`batch_size`>1)** gives a real 7-18% speedup where it fits, but silently corrupts the
  per-sample gradient (verified score drift up to 0.126) — see Core section. Not a free win, don't adopt
  it without explicitly flagging the tradeoff.

## Time / compute savers

- **Cache GT + train-side gradient features once, reuse across every encoder size.** Only the encoder
  differs across the 68m/150m/400m sweep — `train_fig1_encoders.py` pays the Qwen3-4B featurization cost
  (the expensive part, ~15min) exactly once and shares it. Once cached, a same-size hyperparameter run at
  500x1000/8 epochs is ~60s. Only *changing train size* pays the featurization cost again — batch same-size
  sweeps together.
- **Skip FLOPs measurement unless you specifically need a camera-ready appendix.** `FlopCounterMode`
  roughly 5x's wall-clock just for instrumentation, and every time it was checked it told the exact same
  ranking story ms/sample already tells. `figure1_table.run_row` defaults to `measure_flops=False` for
  this reason — leave it off.
- **Don't over-rank the LoGRA proxies past r16.** 1.7B plateaus by r16 (r16 +0.5032 vs. r32 +0.5109 — a
  wash); 0.6B never benefits from rank at all (stuck near zero r8 through r32). Raising rank costs nothing
  in *speed* (rank is ~0.02% of FLOPs) but there's no quality reason to go past r16 for 1.7B and no reason
  to raise rank for 0.6B at all.
- **Don't re-draw more LoGRA seeds for the target.** The existing 5-draw estimate (mean +0.8231, std
  0.087) already clears the "~3 seeds for a variance reading" bar; further seeds are not planned unless
  something about the target itself looks wrong. Put compute into InfluCoder training instead.
- **Don't retest grad_accum, wider batch composition (m_candidates/k_anchors), or weight_decay/grad_clip
  deviations** — all confirmed dead levers (see above) on two different encoder sizes each. Skip them in
  future sweeps rather than re-verifying.
- **Beware same-process sequential reloads when timing multiple configs** — an apparent 38% rank-4-vs-8
  LoGRA timing gap was actually ~95s of OS-page-cache-cold-load I/O on the *first* config only; a clean
  comparison needs either fresh processes per config or an explicit cache-warming pass before timing.
- **SDPA + batch_size=1 is now the default and doesn't need re-verifying per run** — already confirmed
  safe (Δagg = +0.0046 vs. eager) and adopted; re-timing already-adopted rows isn't necessary unless the
  underlying model/rank changes.

## Pool swap: Dolly → tasksource/dolci-instruct (everything else held fixed)

- **Headline: InfluCoder's distillation gain essentially vanishes on dolci-instruct, and the truncation
  bug above is NOT the explanation.** On Dolly, distillation adds +0.35 to +0.40 agg ρ over the untrained
  baseline at every size. On dolci-instruct (post-fix, 1024-token cap): trained sits *below* untrained at
  every size (68m +0.32 untrained vs. +0.05 trained; 150m +0.28 vs. +0.12; 400m +0.21 vs. +0.07). Fixing
  the truncation bug nearly doubled every untrained score (confirming it was real) but did not rescue
  trained scores — they stayed flat or got worse.
- **Root cause is overfitting, not truncation:** all three encoder sizes show eval agg ρ peaking at
  epoch 3/8 then degrading every subsequent epoch, despite training loss falling smoothly to near-zero —
  reproduces identically at both the buggy (512) and fixed (1024) max_len, ruling out truncation as the
  driver. Plausible cause (not investigated further): dolci-instruct is a heterogeneous SFT-mix (math,
  crystallography, moderation, multilingual in one flat prompt/answer pool) vs. Dolly's more uniform
  open-domain instruction shape — the bi-encoder may need different epoch selection or train-side
  sampling to generalize across that heterogeneity.
- **Secondary pool-swap effects:** both LoGRA proxies transfer *better* on dolci-instruct than Dolly
  (0.6B +0.03→+0.30, 1.7B +0.51→+0.58). TF-IDF goes **negative** (+0.30→-0.09) — lexical overlap is a much
  weaker influence signal on this heterogeneous pool. RDS+ drops (+0.31→+0.10).

## InfluCoder training-sample scaling (EXP1 figure, Part 2)

- **Distillation quality scales smoothly with training-sample count, fixed 1:2
  anchor:pool ratio, same recipe as `train_fig1_encoders.py`'s `fig1_dolci` config (68m
  encoder, epochs=8, hard_ratio=0, encoder_max_len=1024).** On the full 400x400 eval:
  agg ρ rises from +0.49 (n=75 total) to +0.77 (n=4500 total, `fig1_dolci`'s own full
  train size), monotonically, with diminishing returns past ~750-1000 anchors (the last
  doubling, 750→1500 anchors, buys only +0.023 vs. +0.10 for 100→250). Untrained-encoder
  baseline on this same 400x400 eval is +0.3171 (a different number from the 50x50-slice
  untrained_68m of +0.2074 in the Part-1 table above — expected, different eval size, not
  a bug). Full point table in `EXP1.md` §7's Part 2 subsection.
- **Report the fixed last epoch, not the internally-selected best epoch, when the x-axis
  is "how much data."** `distill()`'s own best-epoch restoration (`select_best_on`) exists
  because eval Spearman is noisy and peaks-then-degrades at small sample counts — good for
  reporting a single config's best achievable score, wrong for a scaling curve, where an
  early best-epoch pick at small n confounds "how much data helps" with "how much
  early-stopping helps." Fix used: read `log["epoch_metrics"][-1]` (the metrics recorded
  at the end of the final epoch, before `distill()`'s internal restore runs) instead of
  `log["epoch_metrics"][log["best_epoch"]]` — no change to `distill()` itself needed, this
  is purely how the caller reads its return value.
- **Open, unresolved discrepancy:** `baselines/out/fig1_dolci/table1.json` (a pre-existing,
  already-committed file from an earlier project stage, NOT from this session) has
  `influcoder_68m` aggregated = **+0.0466** — inconsistent with this session's scaling
  sweep's n_a=1500 result of **+0.7676** for what should be a comparable/larger training
  config on dolci-instruct. This is very likely the same result behind the
  `dolci-non-reproduction` memory ("InfluCoder hit ~+0.78 where the docs record a ~+0.05
  collapse, on identical code") — i.e. `table1.json` may be the actual source of that
  older collapse finding, produced under some different (unrecorded) condition. Don't
  trust `table1.json`'s other rows (e.g. its LoGRA numbers) as automatically comparable to
  this session's pipeline just because they're at the same 400x400 eval size — the
  internal inconsistency suggests the run that produced the whole file differs from the
  current known-good code path in some unidentified way.
- **LoGRA r8 recomputed fresh at 400x400 (sdpa, max_len=1024) rather than trusting
  `table1.json`, per explicit user instruction.** `logra_4B` = +0.8942 (vs.
  `table1.json`'s unverified +0.9079 — close, within noise); `logra_1.7B` proxy =
  +0.4309 (vs. `table1.json`'s +0.5779 — notably different, another reason not to have
  trusted that file blindly). The 1.7B number cross-checks to within 0.001 against an
  independent earlier-this-session run (`logra_uniform_r8.json`'s `logra_proxy_1.7B_r8` =
  +0.4299), which is the kind of agreement `table1.json` conspicuously lacked. Script:
  `.tuning_logs/_logra_400x400_recompute.py`; output:
  `baselines/out/fig1_dolci/logra_400x400_r8.json`. The Part-2 scaling curve crosses the
  1.7B LoGRA line almost immediately (already above it at the smallest tested size,
  n=75) and doesn't reach the 4B line anywhere in the tested range (max +0.7676 at
  n=4500 vs. +0.8942).

## Why this session's full-vs-proxy LoGRA gap looks smaller than the original paper's

Two distinct, independently-evidenced mechanisms, either of which could explain a remembered
"300→200→100"-shaped taper not reproducing cleanly here:

1. **Quality axis — a deliberate methodology change closed part of the gap.** At a uniform rank (r=8,
   matching the anchor), the proxy quality cliff is stark: 4B +0.90 → 1.7B +0.29 → 0.6B **-0.16**. This
   session adopted an *asymmetric* rank (proxies at r=32, anchor stays r=8) specifically to stop
   penalizing smaller proxies twice — with that change, the taper becomes 4B +0.91 → 1.7B +0.58 → 0.6B
   +0.30 (dolci) — much smoother, but partly *because* the comparison got kinder to the proxies, not
   because the underlying capability gap disappeared.
2. **Cost axis — batch_size=1 is overhead-dominated, not FLOPs-dominated, in this size range.** ms/sample
   across 4B/1.7B/0.6B compresses to a ~1.2-1.5x spread (144/130/126ms under SDPA), not 3x+, because at
   batch=1 the 4B model is already pinned near the GPU memory ceiling and none of the three models can
   batch to amortize fixed per-sample overhead (kernel launch, autograd graph build/teardown). A paper
   timing setup where FLOPs actually dominate wall-clock (larger batches, aggregate-pass timing, or
   different hardware) would show a much more size-proportional, graded cost curve.
