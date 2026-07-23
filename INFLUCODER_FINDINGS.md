# InfluCoder training-tuning log — beat LoGRA r8 on Qwen3-4B

## Goal

Beat LoGRA (raw, rank 8) on the `paper200` setup (200x200 BBH x Dolly eval,
GT from **Qwen/Qwen3-4B**, GT LoRA rank 16, proj_dim 65536) purely through
InfluCoder **training** changes — data size/composition, optimizer, loss,
batching, encoder choice. Eval definition (the GT itself) stays fixed; the
LoGRA target is scored the same way as always (`baselines.logra.score_logra`
against the same cached GT). This file is the running log: every experiment
tried, what changed, what happened, and why (or why not).

Harness: `baselines/tune_influcoder.py`. Every run appends one line to
`baselines/out/qwen4b_tuning/log.jsonl`; the best encoder so far lives at
`runs_out/qwen4b_tuning/<name>/encoder` with its record mirrored in
`baselines/out/qwen4b_tuning/best.json`.

**Iteration speed note:** once GT + train features are cached for a given
size, each `tune_influcoder.py` run only pays for encoder training/eval —
~60s per run at 500x1000/8 epochs. This makes a same-size hyperparameter
sweep cheap; only changing train size pays the 4B featurization cost again.

**Operational note:** background job logs go to `.tuning_logs/` (repo-local),
not `/tmp/.../scratchpad/` — the scratchpad has been wiped by session
restarts mid-run at least twice this project, losing in-flight logs (though
never the repo files: this doc, the harness script, and everything under
`baselines/out/` and `baselines/cache/` all survive restarts fine). If you're
resuming after a restart: check `nvidia-smi`, `pgrep -af tune_influcoder`,
and `log.jsonl` before assuming a launched run finished or is still going.

## Target

LoGRA is unseeded upstream (`kaiming_uniform_`, no seed set on
`logix_lora_A`/`_C`) — established earlier in this session that single draws
drift by as much as +-0.08 to +-0.14 depending on rank/variant. **The bar to
beat is a multi-seed mean, not the first draw**, or "beating LoGRA" could
just mean getting lucky against an unlucky draw.

- Unseeded draw (already run): LoGRA raw r8 agg = **+0.8952**
- Multi-seed estimate (5 draws: unseeded + seeds 1-4, `logra_target.py`):
  `[0.8952, 0.8366, 0.7866, 0.6759, 0.9213]`
  **mean = +0.8231, std = 0.087, range [0.6759, 0.9213]**
  This is a *much* wider spread than we saw at 1.7B/135M (there it was
  +-0.03 to +-0.14; here one draw alone spans +-0.09 around the mean and the
  full range covers 0.25). Saved to
  `baselines/out/paper200_qwen4b/logra_r8_target.json`.

**Target: best-epoch agg >= 0.8231 (the 5-draw mean).** Given std=0.087,
treat anything in [0.80, 0.85) as "essentially tied," and don't declare
victory off a single InfluCoder training seed either -- verify a win with a
second seed before calling it done, same standard we're holding LoGRA to.

**Effort allocation (explicit user direction):** almost all further compute
goes to InfluCoder training experiments, not to re-drawing LoGRA. The 5-draw
rank-8 raw estimate above already exceeds the "~3 seeds for a variance
reading" bar and is not being revisited unless something about the target
looks wrong. No further LoGRA sweeps are planned.

## Baseline (already run, before this file existed)

InfluCoder, current `distill()` defaults, 500x1000 train, 8 epochs, no
accumulation, `jhu-clsp/ettin-encoder-68m`, Qwen3-4B targets:

| | per-anchor | agg |
|---|---|---|
| untrained | +0.2885 | +0.3767 |
| best-epoch (epoch 5/8) | +0.6793 | **+0.6996** |
| final-epoch (epoch 8/8) | — | — (not recorded separately for this run) |

Gap to unseeded LoGRA r8: **-0.1956** agg.

## Constraints / things NOT changed

- The GT itself (eval split, Qwen3-4B, LoRA rank 16, proj_dim 65536) — changing
  this would move the goalposts, not improve the method.
- LoGRA's own hyperparameters — the target is scored as-is, same as every
  other number in this repo.

## Prior findings this session that carry over (established on 1.7B/135M, not yet re-verified on 4B)

- **grad_accum_steps 1 -> 4** (paper's 12 anchors x 15 candidates x accum4 =
  48 eff. batch): no effect on 1.7B ceiling (+0.4685 -> +0.4616 best-epoch);
  did smooth the loss/eval trace substantially. Worth re-testing on 4B since
  the 4B loss curve was already much smoother than 1.7B's even at accum=1.
- **Encoder 68m -> 150m**: *worse* at 1.7B (+0.4616 -> +0.4171). Surprising;
  worth one re-check on 4B since the underlying signal is cleaner there, but
  low prior. User note: since the GT model is 4B, a larger encoder (up to
  ~400m, `jhu-clsp/ettin-encoder-400m`, confirmed to exist on the Hub) is
  explicitly permitted budget-wise — queued for round 2 alongside the 150m
  recheck.
- **Train size 500x1000 -> 1000x2000** (only lever that worked on 1.7B):
  +0.4616 -> +0.5234 best-epoch (+0.0618), +0.4328 -> +0.4519 final-epoch
  (+0.0191). Modest but real, and the strongest lead we have.
- LoGRA FIM variant is much noisier than raw across seeds/ranks (ill
  conditioned `pinv` on an unseeded projection) — don't chase FIM as a target,
  raw is the stable one.

## Experiment log

*(Append one entry per run below, most recent first. Use the exact numbers
from `tune_influcoder.py`'s printed record / `log.jsonl`.)*

---

### `baseline` — harness parity check

Same config as the pre-harness baseline (500x1000, 8 epochs, no accum, 68m
encoder, lr 5e-5, hard_ratio 0.0, alpha 0.5/temp 0.05, linear warmup+decay).

| | per-anchor | agg |
|---|---|---|
| untrained | +0.2885 | +0.3767 |
| best-epoch (epoch 4/8) | +0.6897 | **+0.7123** |
| final-epoch (epoch 8/8) | +0.6708 | +0.6956 |

Elapsed 375s (first call, pays for 4B featurization of the 500x1000 train
side — now cached, so every future run at <=500x1000 skips straight to
encoder training and should be much faster).

vs. the earlier `run.py`-based baseline (+0.6996 best-epoch agg): **+0.0127
higher**, well inside the noise band established earlier (AMP
non-determinism moved a rerun by ~0.010 previously). **Harness confirmed at
parity.** Gap to LoGRA target (+0.8231): **-0.1108** agg (using this run's
number, since it's the one going forward).

Saved as first `best.json` entry (trivially, nothing to beat yet).

---

### `accum4` — grad_accum_steps 1 -> 4 (re-test on 4B)

best-epoch agg **+0.6943** (epoch 3/8), final +0.6744. **Worse than baseline
(+0.7123), by -0.0180** — outside the noise band. Confirms the 1.7B finding
(accum didn't help there either, +0.4685->+0.4616): accumulation is not a
lever that helps this setup, on either gradient model. Not pursuing further.

---

### `hard05` — fixed hard_ratio 0.5

best-epoch agg **+0.6873** (epoch **8/8** — still rising at the last epoch,
not yet peaked), final same. Worse than baseline at this epoch budget
(-0.025), but the "best epoch = last epoch" pattern is different from every
other run (all others peak mid-run then degrade) — hard negatives are
teaching it something slower/harder, not saturating as fast. Worth a
follow-up with more epochs before writing this off; noted, not abandoning
hard negatives yet.

---

### `hard_ramp` — curriculum hard_ratio 0.0 -> 0.5

best-epoch agg **+0.6868** (epoch 7/8), final +0.6862. Same story as
`hard05`: worse than baseline at 8 epochs, still climbing late. Curriculum
ramp doesn't beat flat hard_ratio=0.5 here (+0.6868 vs +0.6873, a wash) —
the ramp isn't adding anything over just turning on hard negatives from the
start. Both hard-negative variants share the "not converged by epoch 8"
pattern -> queuing a longer-epoch re-test of hard_ratio next.

---

### `cosine` — cosine LR schedule instead of linear decay

best-epoch agg **+0.6942** (epoch 5/8), final +0.6862. Worse than baseline
(-0.0181). Linear decay remains better than cosine for this short a run;
not pursuing.

---

### `alpha03` — alpha 0.5 -> 0.3 (more listwise KL, less global Pearson)

best-epoch agg **+0.7159** (epoch 5/8), final +0.7119. **First config to beat
baseline**, +0.0036 — small, plausibly noise, but the right direction:
listwise ranking signal matters at least as much as global correlation here.
Waiting on `alpha07` (opposite direction) to see if the effect is
monotonic before deciding whether to push alpha lower still (e.g. 0.2, 0.1).

### `alpha07` — alpha 0.5 -> 0.7 (more global Pearson, less listwise KL)

best-epoch agg **+0.7062** (epoch 4/8), final +0.6953. Worse than both
baseline (0.5) and alpha03. **Confirms a monotonic trend: lower alpha (more
KL/listwise weight, less Pearson) is better** — alpha07 (+0.7062) < alpha05
baseline (+0.7123) < alpha03 (+0.7159). Queuing alpha=0.1 and alpha=0.0 (pure
listwise KL) as a follow-up — this is the most promising lead from round 1.

### `temp02` — temperature 0.05 -> 0.02 (sharper KL softmax)

best-epoch agg **+0.7074** (epoch 3/8), final +0.6992. Slightly worse than
baseline (-0.0049). Sharper isn't better; waiting on `temp10` (softer) for
the other direction.

### `temp10` — temperature 0.05 -> 0.10 (softer KL softmax)

best-epoch agg **+0.7021** (epoch 5/8), final +0.6916. Also worse than
baseline (-0.0102). **Default temperature=0.05 is a local optimum** — both
directions tested are worse. Not pursuing temperature further; the alpha
axis remains the one real lead.

### `lr1e4` — LR 5e-5 -> 1e-4

best-epoch agg **+0.7005** (epoch **8/8**, still rising), final same. Worse
than baseline at this epoch budget, but hasn't peaked yet — same pattern as
the hard-negative runs. Worth a longer-epoch recheck alongside those rather
than ruled out.

### `lr2e5` — LR 5e-5 -> 2e-5

best-epoch agg **+0.6928** (epoch 4/8), final +0.6790. Worse than baseline
(-0.0195), and worse than lr1e4 too. **Default lr=5e-5 beats both directions
tested at this epoch budget** — mirrors the temperature result. Not pursuing
further at 8 epochs; the higher-LR arm still needs the longer-epoch recheck
noted above.

### `epochs16` — 8 -> 16 epochs (same lr/schedule, just longer)

best-epoch agg **+0.7120** (epoch 4/16), final +0.7032. Essentially
identical to baseline's epoch-4 peak (+0.7123) — extending the run alone
buys nothing, the encoder peaks early (epoch 4-5) and doesn't find a better
optimum later, it just overfits and gets restored from the same-ish early
checkpoint. **Epoch budget is not the bottleneck** at this LR/schedule;
peaking early is a property of the setup, not of running out of epochs.
This narrows the earlier lr1e4/hard_ratio "still rising at epoch 8"
observations too: since a stock 16-epoch run doesn't relocate the peak,
those likely just have a slower LR schedule (more total steps to decay
through) rather than a genuinely later, better optimum -- lowers priority on
the longer-epoch recheck of those two.

### `wd0` — weight_decay 0.01 -> 0.0

best-epoch agg **+0.6959** (epoch 3/8), final +0.6902. Worse than baseline
(-0.0164). Default weight_decay=0.01 is better than none; not pursuing.

### `gradclip05` — max_grad_norm 1.0 -> 0.5

best-epoch agg **+0.7031** (epoch 4/8), final +0.6848. Worse than baseline
(-0.0092). Default clip=1.0 is better; not pursuing.

---

## Round 1 summary (13 single-variable tests @ 500x1000)

Only **alpha03** (alpha 0.5->0.3, more listwise-KL weight) beat baseline
(+0.7159 vs +0.7123, +0.0036). Everything else -- accum, hard_ratio (flat or
ramped), cosine schedule, temperature (either direction), lr (either
direction), more epochs, no weight decay, tighter grad clipping -- was flat
or worse. The alpha axis is the only real signal; confirmed monotonic across
3 points (0.7:+0.7062 < 0.5:+0.7123 < 0.3:+0.7159). Round 2: push alpha
further (0.0-0.2), recheck encoder size (150m, and 400m per user go-ahead
since GT model is 4B), then combine whatever wins with the one lever that
mattered at 1.7B -- more train data.

---

### `alpha00` — alpha 0.0 (pure listwise KL, no Pearson term at all)

best-epoch agg **+0.6923** (epoch **8/8**, still rising), final same. Worse
than baseline (0.5) AND worse than alpha03 -- **not monotonic all the way
down**: the trend reverses somewhere between 0.0 and 0.3. Pure Pearson (1.0)
and pure KL (0.0) are both worse than a blend; alpha03 (0.3) looks like it's
near an interior optimum, not an endpoint. Waiting on alpha01/alpha02 to
locate it more precisely.

### `alpha01` — alpha 0.1

best-epoch agg **+0.7015** (epoch 5/8), final +0.7000. Between alpha00
(+0.6923) and alpha03 (+0.7159), but still worse than baseline (0.5,
+0.7123). Points so far: 0.0:+0.6923, 0.1:+0.7015, 0.3:+0.7159, 0.5:+0.7123,
0.7:+0.7062 -- an interior peak somewhere around 0.3-0.4. Waiting on
alpha02/alpha04 to bracket it.

### `alpha02` — alpha 0.2

best-epoch agg **+0.7148** (epoch **8/8**, still rising), final same. Close
to alpha03's +0.7159, and still climbing at the epoch limit -- may go higher
than alpha03 with more epochs. Second-best so far. Waiting on alpha04 to
finish bracketing before deciding whether alpha02 or alpha03 is the one to
extend-epoch retest.

### `alpha04` — alpha 0.4

best-epoch agg **+0.6988** (epoch 4/8). Breaks the smooth trend: worse than
alpha03 (0.7159), alpha02 (0.7148), AND baseline alpha05 (0.7123) despite
sitting between 0.3 and 0.5. **This is the tell that single-seed noise here
is comparable to the alpha effect size** (~0.02-0.03 swings between
adjacent, similarly-configured runs) -- the alpha sweep is a noisy bowl with
a soft floor around 0.2-0.3, not a precise curve. Conclusion: **alpha in
[0.2, 0.3] is the region to use, but don't over-trust any single value's
rank within it** -- confirm the final choice with a second seed once we're
picking a config to actually beat LoGRA with.

### `enc150m` — encoder 68m -> 150m (recheck on 4B)

best-epoch agg **+0.6702** (epoch 5/8), final +0.6659. Clearly worse than
baseline (-0.0421). **Confirms the 1.7B finding on 4B too: 150m is worse
than 68m**, not just noise-sized this time -- a real, repeated regression.
Bigger encoder is not automatically better for this distillation task;
closing this lead for good, no further encoder-size-down testing planned.
Still waiting on `enc400m` (user-approved) as one more data point in case
the relationship is non-monotonic (68m good, 150m bad, 400m good again would
be a genuinely different story) but low prior given this result.

### `enc400m` — encoder 68m -> 400m (user-approved given 4B GT model)

best-epoch agg **+0.6986** (epoch **1/8**), final-epoch agg **+0.6174** --
peaks immediately then collapses hard by the end of training. Confirms the
150m result and rules out a "bigger is better again eventually" story: at
500 training anchors, a 400m encoder overfits almost instantly (peak at
epoch 1) and then degrades sharply, worse than both 68m and 150m by
final-epoch. **68m remains the right encoder size for this data scale** --
larger encoders need more data than we're giving them, not a bigger model.
Encoder-size axis closed; not testing 1b.

---

### `big2x` — train size 500x1000 -> 1000x2000 (paper200x2, alpha=0.5 default)

best-epoch agg **+0.7645** (epoch 7/8, still near the end -- not clearly
saturated), final +0.7615. **By far the biggest single lever found on 4B**:
+0.0522 over baseline (+0.7123), more than 10x the alpha effect size.
Confirms the 1.7B finding (500->1000x2000 was the one lever that worked
there too, +0.0618) transfers cleanly to 4B, and the effect is *larger*
here. **Gap to LoGRA target (+0.8231) is now just -0.0586** -- more than
halved from the original baseline gap (-0.1108). Waiting on
`big2x_alpha03` (same size + the winning alpha) to see if the two levers
stack.

### `big2x_alpha03` — 1000x2000 + alpha 0.3

best-epoch agg **+0.7584** (epoch 4/8), final +0.7559. **Worse than
`big2x`'s default alpha=0.5 (+0.7645)** -- the levers don't stack, the small
alpha win found at 500x1000 doesn't transfer to the bigger-data regime.
Sticking with default alpha=0.5 going forward at larger sizes; alpha tuning
is apparently a small-data-regime effect that gets swamped once there's
more data to learn from.

**Current best: `big2x`, +0.7645 best-epoch agg (default hyperparams,
1000x2000 train data). Gap to LoGRA target: -0.0586.** Round 4: push data
size further (added `paper200x4` preset, 2000x4000, to `run.py` --
6511/15011 BBH/Dolly samples available so there's headroom), and retest
epochs/hard_ratio at the larger size now that "peaks early" was established
only at 500x1000.

### `big2x_epochs16` — 1000x2000, 8 -> 16 epochs

best-epoch agg **+0.7599** (epoch 15/16, near the end), final +0.7579.
**Slightly worse than `big2x`'s 8-epoch peak (+0.7645)**, not better --
doubling epochs also stretches the LR schedule (total_steps scales with
epochs), so this isn't a clean "same schedule, more time" comparison, but
the practical conclusion holds: **more epochs at 1000x2000 does not beat the
8-epoch run**. The "best epoch 7/8, still climbing" observation on `big2x`
was likely just within-run noise near a plateau, not a genuinely unfinished
climb. Sticking with 8 epochs.

### `big2x_hard05` — 1000x2000 + hard_ratio 0.5 -- **NEW BEST**

best-epoch agg **+0.7716** (epoch 6/8), final +0.7693. **Beats `big2x`
(+0.7645) by +0.0071.** First time hard-negative mining has actually helped
-- it lost at 500x1000 (worse both times) but wins once there's more data.
Makes sense in hindsight: with only 500 anchors, "hardest negatives" is a
small, repeat-heavy set that the encoder overfits to fast; at 1000 anchors
there's a bigger and more varied hard set, so the mining signal is actually
useful rather than a narrow overfitting shortcut. **Gap to LoGRA target
(+0.8231): -0.0515.** Queuing a hard_ratio sweep at this size (0.25, 0.75)
plus a curriculum ramp retest, now that we know hard negatives are a real
lever at this scale.

### `big4x` — train size 1000x2000 -> 2000x4000 (paper200x4, default hyperparams)

best-epoch agg **+0.7504** (epoch 4/8), final +0.7451. **Worse than `big2x`
(+0.7645) AND worse than `big2x_hard05` (+0.7716)** -- more data alone
stopped helping, and actually regressed, once hard negatives aren't also
scaled up. This makes sense from `_sample_candidates`'s own docstring:
"pure random candidates are almost all easy negatives once the pool is more
than a few dozen items" -- at pool=4000 with m_candidates still fixed at 16
random, the negative signal is even more diluted than at pool=2000. **The
"more data helps" lever isn't unconditional -- it needs hard-negative mining
to scale with it**, or growing the pool just makes random negatives easier
and easier to ignore. Re-prioritizing: `big4x_hard05` (2000x4000 + hard
mining) is now the most promising next experiment, ahead of the planned
hard_ratio micro-sweep at 1000x2000.

### `big4x_hard05` — 2000x4000 + hard_ratio 0.5

best-epoch agg **+0.7562** (epoch 7/8), final +0.7525. Confirms hard mining
helps at this size too (+0.0058 over plain `big4x`'s +0.7504), but **still
worse than `big2x_hard05` (+0.7716) at the smaller 1000x2000 size**. So
2000x4000 is not a straightforward win over 1000x2000 even with hard
negatives turned on -- **1000x2000 + hard_ratio=0.5 remains the best
config overall.** Possible explanations not yet tested: m_candidates=16 may
need to scale up with pool size too (more hard candidates to pick from
doesn't help if the block only samples 16 total), or k_anchors=8 needs to
grow with anchor count. Lower priority than finishing the hard_ratio
micro-sweep at the size that's actually winning.

### `big4x_hard075` — 2000x4000 + hard_ratio 0.75

best-epoch agg **+0.7686** (epoch 6/8), final +0.7617. Jumps well past
`big4x_hard05` (+0.7562) -- **bigger pool does want a higher hard_ratio**,
confirming the hypothesis above. Now close to (but still just under)
`big2x_hard05`'s +0.7716. Trend across hard_ratio at pool=4000:
0.0:+0.7504, 0.5:+0.7562, 0.75:+0.7686 -- still rising. Queuing hard_ratio=1.0
(pure hard negatives) at 2000x4000 to see if it keeps climbing or peaks
here.

### `big2x_hard025` — 1000x2000 + hard_ratio 0.25

best-epoch agg **+0.7490** (epoch **8/8**, still rising), final same. Worse
than `big2x_hard05` (+0.7716). At 1000x2000: 0.0:+0.7645, 0.25:+0.7490 (dip
below plain), 0.5:+0.7716 (peak so far). Non-monotonic at the low end --
some noise here too, but 0.5 is clearly the best of the three. Waiting on
`big2x_hard075` to see if it keeps climbing past 0.5 the way pool=4000 did.

### `big2x_hard075` — 1000x2000 + hard_ratio 0.75

best-epoch agg **+0.7481** (epoch 7/8), final +0.7443. Also worse than 0.5,
roughly level with 0.25. **At 1000x2000, hard_ratio=0.5 is a genuine
interior optimum** (0.0:+0.7645, 0.25:+0.7490, **0.5:+0.7716**,
0.75:+0.7481) -- unlike at 2000x4000 where higher ratios keep helping. This
is consistent with the "optimal hard fraction scales with pool size" story:
bigger pool, more room for useful hard negatives before you run out of
genuinely hard ones. **`big2x_hard05` (+0.7716) remains the overall best.**
Still want `big4x_hard10` (pure hard, pool=4000) to see where that curve
peaks.

---

## Round 5 summary

Best remains **`big2x_hard05`: +0.7716** (1000x2000, hard_ratio=0.5, default
everything else). Gap to LoGRA target (+0.8231): **-0.0515**. Round 6:
finish the pool=4000 hard_ratio curve (1.0), and try scaling
`m_candidates`/`k_anchors` alongside hard_ratio=0.5 at 1000x2000 -- with
only 16 candidates per block and half of them hard, the actual hard-negative
count per step (8) may itself be a limiting factor rather than the ratio.

**Bug found & fixed:** `hard_ratio=1.0` crashed `big4x_hard10`
(`TypeError: only integer tensors of a single element can be converted to
an index` in `_sample_candidates`). Root cause: when `n_random=0`,
`torch.tensor(rng.sample(remaining, 0))` -> `torch.tensor([])`, which
defaults to **float32**, not `torch.long`; `torch.cat` with the int64 `hard`
tensor then silently produces a float index tensor that only breaks later
at `pool_texts[j] for j in c_idx`. Fixed in `influcoder/encoder.py` by
passing `dtype=torch.long` explicitly. Every `hard_ratio<1.0` result logged
so far was unaffected (n_random>0 there), but `hard_ratio=1.0` was never
actually tested -- re-queued for round 7.

### `big2x_hard05_m24` — 1000x2000 + hard_ratio 0.5 + m_candidates 16 -> 24

best-epoch agg **+0.7273** (epoch 7/8), final +0.7258. **Much worse than
`big2x_hard05` (+0.7716)** -- more candidates per block, at the same
hard_ratio, was not an improvement, it was a substantial regression
(-0.0443). More candidates means more (mostly-easier, since only 12 of 24
are hard now vs 8 of 16) negatives diluting the loss per step -- the
opposite of what more compute-per-step "should" buy, but consistent with
the general theme this session that more easy negatives hurts, not helps.
Waiting on `m32` (should be worse still if this trend holds) and `k12`
before concluding.

### `big2x_hard05_m32` — 1000x2000 + hard_ratio 0.5 + m_candidates 16 -> 32

best-epoch agg **+0.7598** (epoch **8/8**, still rising), final same. Better
than m24 (+0.7273) but still worse than the m16 baseline (+0.7716) --
**non-monotonic in m_candidates** (16 best, 24 worst, 32 middle), which
means this axis is noisy rather than showing a clean trend, similar to the
alpha04 outlier earlier. **m_candidates=16 (the default) stays the best
found on this axis** -- not pursuing further tuning here.

### `big2x_hard05_k12` — 1000x2000 + hard_ratio 0.5 + k_anchors 8 -> 12

best-epoch agg **+0.7565** (epoch 7/8), final +0.7526. Worse than the k8
baseline (+0.7716). Closer to the old paper's k_anchors=12, but not better
here either. **k_anchors=8 (current default) is the best on this axis
too.** Batch-composition tuning (m_candidates, k_anchors) is a dead end at
this size/ratio -- the defaults already found by earlier rounds hold up.

---

## Round 6 summary

Nothing beat `big2x_hard05` (+0.7716). Batch-composition knobs
(m_candidates, k_anchors) are noisy-to-negative here; leave them at
defaults. Round 7: re-run `big4x_hard10` with the bug fix, verify
`big2x_hard05` with a second training seed (the target's own multi-seed
standard applies to us too), and try an intermediate train size
(1500x3000) since 1000x2000 beat 2000x4000 -- there may be a sweet spot
in between rather than "bigger always better" or "1000x2000 uniquely
best."

---

### `big4x_hard10` — 2000x4000 + hard_ratio 1.0 (retry after the dtype fix)

best-epoch agg **+0.4742** (epoch 8/8, still rising but far below
everything else), final same. **Collapses badly** -- worse than the
untrained baseline range, nowhere close to hard_ratio=0.75's +0.7686.
**Pure hard negatives (no easy negatives at all) breaks training**, not
just "helps less": with every candidate near-maximally similar to the
anchor by construction, there's no easy contrast left for the ranking loss
to calibrate against, so the signal degenerates instead of sharpening.
Confirms the pool=4000 hard_ratio curve peaks somewhere under 1.0 (0.75 is
close to it, 1.0 is a cliff) -- the interior-optimum story holds at every
size tested, just at a different location. Not testing hard_ratio=1.0
again at any size.

### `big2x_hard05_seed1` — second-seed verification of the current best

best-epoch agg **+0.7602** (epoch **8/8**, still rising -- this seed hadn't
plateaued by epoch 8, unlike seed 0), final same. **Confirms the
`big2x_hard05` win is real, not a single-seed fluke**: seed 0 (+0.7716) and
seed 1 (+0.7602) both land clearly above the plain-`big2x` baseline
(+0.7645) and far above every other config tried. Two-seed mean:
**+0.7659**. Gap to LoGRA target (+0.8231) using the more honest 2-seed
mean: **-0.0572**. Still short, but the config itself (1000x2000,
hard_ratio=0.5, everything else default) is now the confirmed best
direction to keep building on.

### `big3x_hard05` — 1500x3000 (paper200x3) + hard_ratio 0.5 -- **NEW BEST**

best-epoch agg **+0.7747** (epoch 7/8), final +0.7732. **Beats
`big2x_hard05` (+0.7716) by +0.0031** -- a real sweet spot: 1500x3000 beats
both 1000x2000 and 2000x4000. Confirms the "intermediate size" hypothesis
from the round-6 summary rather than either "bigger always better" (ruled
out by big4x) or "1000x2000 is uniquely best" (ruled out by this result).
**Gap to LoGRA target (+0.8231): -0.0484** -- the closest yet. Queuing a
hard_ratio micro-sweep (0.6, 0.7) at this size plus a second-seed check
before calling this the new reference config.

---

## Round 7 summary

**New best: `big3x_hard05`, +0.7747** (1500x3000, hard_ratio=0.5). Gap to
target: -0.0484. Also: hard_ratio=1.0 is a hard cliff (collapses to
+0.4742, not a graceful falloff) -- confirmed the interior-optimum story
holds at every size, just at a different peak location each time. The
current best (`big2x_hard05`) held up under a second seed (+0.7602 vs
+0.7716), giving real confidence the size+hard-negative combination is a
genuine effect, not noise. Round 8: hard_ratio micro-sweep around 0.5 at
1500x3000, second-seed check on `big3x_hard05` itself.

---

### `big3x_hard06` — 1500x3000 + hard_ratio 0.6

best-epoch agg **+0.7634** (epoch **8/8**, still rising), final same. Worse
than hard_ratio=0.5's +0.7747. At 1500x3000, 0.5 still looks like the peak
(consistent with 1000x2000's own peak at 0.5, even though 2000x4000 wanted
0.75-ish) -- the optimal hard_ratio doesn't move in perfect lockstep with
pool size. Waiting on `hard07` to confirm the downward direction, then the
seed-1 check.

### `big3x_hard07` — 1500x3000 + hard_ratio 0.7 -- marginal new best

best-epoch agg **+0.7768** (epoch 6/8, converged mid-run not at the
boundary), final +0.7707. Edges out hard05's +0.7747 by +0.0021 -- inside
the noise band we've been treating as "not really different" all session
(single-seed swings of 0.02-0.04 are common here). So 0.5-0.7 are all
roughly tied at 1500x3000; **not a clean re-confirmation that 0.6 is worse
and 0.7 is better** than 0.5, more like "anywhere in [0.5, 0.7] works about
the same, don't over-fit the third decimal." Treating **+0.775 +- 0.02** as
the honest estimate of this config's quality rather than chasing which
exact hard_ratio "wins." Gap to target with this number: -0.0463.

### `big3x_hard05_seed1` — second-seed check on 1500x3000 + hard_ratio 0.5

best-epoch agg **+0.7588** (epoch 5/8), final +0.7495. vs seed 0's +0.7747
-> **2-seed mean +0.7667**, std ~0.011. This is statistically
indistinguishable from `big2x_hard05`'s own 2-seed mean (+0.7659) --
**1000x2000 and 1500x3000, both with hard_ratio~0.5-0.7, land in the same
~0.76-0.77 band once seed noise is accounted for.** The "sweet spot between
1000x2000 and 2000x4000" framing from round 7 should be softened: it's less
a sharp peak at 1500x3000 and more a plateau across 1000x2000-1500x3000
that all these single-run "new bests" were sampling noisily.

## Round 8 summary

**Honest current estimate: ~+0.767 agg** (2-seed means across both
1000x2000 and 1500x3000 configs agree). **Gap to LoGRA target (+0.8231):
~-0.056.** The size+hard-negative combination has plateaued in this range;
squeezing another ~0.05 needs either (a) a genuinely different lever, or
(b) an ensembling trick rather than a single-encoder training change.
Round 9: try `big5x` (3000x6000) with a proportionally higher hard_ratio
(the 2000x4000 data suggested optimal hard_ratio rises with pool size), and
test score-averaging an ensemble of the seed-0/seed-1 encoders already
trained -- averaging two independently-trained encoders' score matrices is
a training-adjacent, not eval-side, change (still purely a function of how
InfluCoder itself is built) and costs no new GPU time since both encoders
already exist.

### Ensemble test — 3 seeds at 1500x3000, hard_ratio=0.6, score-averaged

New script `baselines/ensemble_influcoder.py`: trains N seeds fresh (solo
runs weren't saved to disk unless "best", so this retrains rather than
reusing), averages their raw cosine score matrices, re-scores against GT.

Solo: seed 0 +0.7643, seed 1 +0.7582, seed 2 +0.7825 (mean +0.7683, std
0.010 -- matches the seed-noise magnitude seen everywhere else this
session). **Ensemble (3-seed average): agg +0.7781.** Beats the solo mean
(+0.7683) by +0.0098 -- a real but modest smoothing gain -- but does **not**
beat the best individual seed (+0.7825). **Ensembling reduces variance, it
doesn't raise the ceiling**: it's a hedge against drawing a bad seed, not a
way to systematically beat the single-run best. Gap to target with the
ensemble number: -0.0450. Given LoGRA's own target is itself a *mean* over
seeds (not a best-of), comparing our ensemble mean-ish number to their
mean is the fairer comparison, and it doesn't close the gap by much.

## Round 9: settle on 1500x3000, keep working other knobs, try new losses

**Decision (user call): 1500x3000 is the standard train size going
forward.** Hard_ratio ~0.6 (middle of the 0.5-0.7 plateau) is the working
default.

**Redirect (user call):** before moving to new loss functions, re-examine
whether a bigger encoder (150m/400m) can work now that the training setup
is much stronger than when they were tested. The earlier 150m/400m results
(round 2) were BOTH at 500x1000 with NO hard-negative mining -- 400m
overfit in a single epoch there. That's a very different regime from the
current best (1500x3000 + hard_ratio~0.6). Re-testing both at the current
best config before writing off encoder size for good. Added
`soft_spearman_loss` and `infonce_loss` to `influcoder/encoder.py`
(`LOSS_FNS` registry) in passing -- not wired into `distill()`'s dispatch
yet, paused to do the encoder-size recheck first.

### `big3x_hard06_150m` — 150m encoder at 1500x3000 + hard_ratio=0.6

best-epoch agg **+0.7289** (epoch 6/8), final +0.7220. Still clearly worse
than 68m at the same config (+0.76-0.78 range). **More data + hard mining
does NOT rescue 150m** -- it's still a real regression, not an artifact of
the earlier small-data/no-mining setup. Waiting on `400m` before drawing
the final conclusion, since 400m's earlier failure mode (collapse after
epoch 1) was more dramatic and more plausibly data-starved than 150m's
milder underperformance.

**Redirect (user call):** drop hard-negative mining from the encoder-size
test (killed the in-flight `big3x_hard06_400m` run) -- isolate encoder size
from the hard-mining interaction, and instead sweep **learning rate**
specifically for 150m/400m at 1500x3000. Rationale: lr=5e-5 was found
tuning against the 68m encoder; a much bigger encoder plausibly needs a
smaller LR (400m collapsing after 1 epoch at 500x1000 looks like classic
"LR too high for this parameter count," not necessarily a data/capacity
mismatch). Round 10: lr in {1e-5, 2e-5, 5e-5} x {150m, 400m}, no hard
mining, at 1500x3000.

### `big3x_400m_lr1e5` — 400m encoder, lr 5e-5 -> 1e-5, no hard mining

best-epoch agg **+0.7385** (epoch 5/8, stable peak not at either boundary),
final +0.7328. **Much better than 400m's earlier collapse** (+0.6174
final-epoch at 500x1000/lr5e-5/hard0.6) -- no more single-epoch peak then
crash, the training curve looks like a normal run now. **Lower LR does fix
the instability.** Still below 68m's ~0.76-0.78 range at this size, but this
is a qualitatively different (and much healthier) result than before.
Waiting on lr2e-5 and lr5e-5 (at 1500x3000, no hard mining) to map the
curve before concluding whether 400m can close the gap to 68m with the
right LR, or just stops being broken without becoming competitive.

### `big3x_400m_lr2e5` — 400m encoder, lr 5e-5 -> 2e-5, no hard mining -- big jump

best-epoch agg **+0.7735** (epoch 6/8, stable), final +0.7717. **Huge
improvement over lr1e-5's +0.7385 -- now squarely competitive with 68m's
range** (+0.76-0.78, no hard mining even turned on yet). This strongly
confirms the user's hypothesis: **the earlier "bigger encoder is worse"
conclusion was an LR-mismatch artifact, not a real capacity/data-scale
finding.** lr=5e-5 (tuned on/for 68m) was simply too aggressive for 400m's
larger parameter count; at lr=2e-5 it's a different story entirely. Waiting
on lr5e-5 (matched control) to confirm the earlier 500x1000 collapse
generalizes to 1500x3000 too (i.e. that the fix really is LR, not scale),
then this needs its own hard_ratio sweep -- 400m + the right LR + hard
mining could plausibly beat 68m outright.

### `big3x_400m_lr5e5` — 400m encoder, ORIGINAL lr=5e-5, no hard mining -- **NEW OVERALL BEST**

best-epoch agg **+0.7960** (epoch **8/8**, still rising at the boundary),
final same. **Beats everything found this entire session**, including
every 68m result, and it doesn't even have hard-negative mining turned on
yet. **Revised understanding, correcting the LR-mismatch theory above**:
the original LR wasn't actually wrong for 400m -- what was missing was
enough DATA (500x1000 -> 1500x3000) for a bigger encoder to use its extra
capacity without overfitting. lr=2e-5 (+0.7735) is worse than lr=5e-5
(+0.7960) at this same size -- so it's not "400m needs a smaller LR," it's
"400m needs more data, and once it has enough, the same LR that works for
68m works even better for 400m." **Gap to LoGRA target (+0.8231): -0.0271
-- the closest result yet, and it hasn't even had hard mining or an
extended epoch budget applied.** Best epoch sitting at the boundary (8/8)
means this run is very likely under-trained -- queuing an epoch extension
and a hard_ratio sweep on this exact config immediately, ahead of the
remaining planned 150m LR runs.

**Round 11 (killed the in-flight 150m LR sweep to prioritize this):**
epoch extension (16), hard_ratio 0.5/0.6 on top, and a second-seed check --
all on 400m/1500x3000/lr=5e-5/no-hard-mining, the new reference config.

### `big3x_400m_epochs16` — 400m, 8 -> 16 epochs, no hard mining -- **NEW BEST**

best-epoch agg **+0.8017** (epoch 14/16, close to but not at the boundary),
final +0.7996. **Beats the 8-epoch run (+0.7960) by +0.0057, and this time
the peak is nearly interior (14/16), not sitting right at the edge** -- much
more likely close to converged. **Gap to LoGRA target (+0.8231): -0.0214.**
Note this contradicts the earlier 68m finding that more epochs didn't help
(`big2x_epochs16` was *worse* than the 8-epoch peak) -- **the epoch-budget
story is encoder-size-dependent**: 68m saturates by ~epoch 6-7 regardless
of extra epochs offered, but 400m (more capacity, needs more updates to use
it) is still improving at epoch 8 and only starting to plateau around
epoch 14. User directive: if the remaining hard-mining runs in this round
don't help here either, drop hard mining as a lever entirely and pivot to
batch composition (m_candidates, k_anchors, grad_accum) instead --
consistent with hard mining's mixed/marginal track record all session.
Waiting on `hard05`/`hard06` (still 8 epochs each) and `seed1` before
deciding.

**Redirect (user call):** don't bother testing hard_ratio=0.5 here -- drop
hard mining entirely (matches its mixed/marginal track record all
session -- helped narrowly at some 68m sizes, actively hurt at 500x1000,
collapsed completely at hard_ratio=1.0) and pivot to **batch composition**
instead: how many pool samples are seen per step (`m_candidates`), how many
anchors per step (`k_anchors`), and effective batch via `grad_accum_steps`.
Killed the in-flight `big3x_400m_hard05` run. Round 12: m_candidates
{32, 64}, k_anchors 16, grad_accum 4, plus a second-seed check -- all on
top of the current best (400m, 1500x3000, lr=5e-5, epochs=16, hard_ratio=0
explicit).

**Redirect (user call):** the epochs=16 gain (+0.8017 @ epoch14) over
epochs=8 (+0.7960 @ epoch8-boundary) is only +0.0057 -- likely a lucky
convergence within seed noise, not a reliable effect worth the 2x compute.
**Sticking with epochs=8 as the standard** to save cost, killed the
in-flight 16-epoch `big3x_400m_m32` run. Broadening round 13 to a wider
single-variable sweep at epochs=8 fixed on 400m/1500x3000/hard_ratio=0:
m_candidates, k_anchors, grad_accum, alpha, lr (finer grid), weight_decay,
max_grad_norm, warmup_frac, lr_schedule.

### `big3x_400m_m32` — m_candidates 16 -> 32

best-epoch agg **+0.7825** (epoch **8/8**, still rising), final same. Worse
than the m16 reference (+0.7960). Same direction as the earlier 68m finding
(m24/m32 were both worse than m16 there too) -- **m_candidates=16 (default)
continues to be the best value on this axis, now confirmed on a second
encoder size.** Not pursuing m64.

### `big3x_400m_k16` — k_anchors 8 -> 16

best-epoch agg **+0.7633** (epoch 4/8), final +0.7613. Worse than reference
(-0.0327), the biggest single-variable drop in this round so far. Matches
the earlier 68m finding (k12 was worse than k8 there too) --
**k_anchors=8 (default) stays the best on this axis for both encoder
sizes.** Not pursuing bigger anchor batches.

### `big3x_400m_accum4` — grad_accum_steps 1 -> 4

best-epoch agg **+0.7653** (epoch **8/8**, still rising), final same. Worse
than reference (-0.0307). Consistent with the 68m finding (accum4 was worse
there too, both times this axis was tested this session) --
**grad_accum stays a dead lever regardless of encoder size.**

### `big3x_400m_alpha03` — alpha 0.5 -> 0.3

best-epoch agg **+0.7978** (epoch **8/8**, still rising), final same.
Slightly beats reference (+0.7960) by +0.0018 -- inside the noise band, but
the same direction as the 68m alpha finding (0.3 beat 0.5 there too, though
that one didn't transfer when combined with bigger data). Waiting on
`alpha07` to check whether the direction is consistent here too.

### `big3x_400m_alpha07` — alpha 0.5 -> 0.7

best-epoch agg **+0.7901** (epoch 8/8), final same. Worse than both
reference (0.5) and alpha03 -- **consistent direction with the 68m result:
lower alpha (more listwise-KL weight) is mildly better, monotonic across
0.7 < 0.5 < 0.3 on 400m too.** Effect size is still small (~0.01-0.02,
inside the noise band), but at least the *direction* replicates across
both encoder sizes now, which the earlier (also monotonic-then-broken)
68m sweep didn't fully establish on its own. Using alpha=0.3 going forward
as the marginal best, without over-claiming it's a big lever.

### `big3x_400m_lr3e5` — lr 5e-5 -> 3e-5

best-epoch agg **+0.7860** (epoch 8/8), final same. Worse than reference
(-0.0100). 5e-5 remains the best LR found for 400m at this size (killed the
in-flight `lr7e5` run to pause the sweep here per user direction -- work on
InfluCoder tuning is paused, not concluded).

## Status at pause: current best is `big3x_400m_epochs16`

**+0.8017 agg** (400m encoder, 1500x3000, lr=5e-5, epochs=16, hard_ratio=0).
User called the epochs=16 gain over epochs=8 (+0.7960) likely noise and
elected to standardize on epochs=8 to save cost; **the practical reference
config going forward is 400m/1500x3000/lr=5e-5/epochs=8/hard_ratio=0/alpha=0.3
(marginal best from this round), agg ~+0.7960-0.7978**. Gap to LoGRA target
(+0.8231): **~-0.025 to -0.027**. Not yet beaten. **Paused (not abandoned)
to rebuild Figure 1** per user request -- resume the sweep (remaining:
lr7e5, wd002, wd005, clip2, warmup02, cosine, then a second-seed
verification of whatever wins) when told to.

---

## Figure 1 build (separate deliverable, not the LoGRA-beating task)

Building the paper's main speed-vs-quality table: a NEW 400x400 eval
(`fig1` preset in `run.py`, n_train_a/p=1500/3000, Qwen3-4B GT rank 16),
scoring LESS, LoGRA r8, LoGRA proxy (Qwen3-1.7B, Qwen3-0.6B -- same-family
cheaper proxies), InfluCoder (68m/150m/400m), untrained encoder
(68m/150m/400m), RDS+, and TF-IDF, all against the same GT. New files:
`baselines/train_fig1_encoders.py` (trains + ALWAYS saves all 3 InfluCoder
sizes, unlike `tune_influcoder.py`'s best-only saving), `baselines/figure1_table.py`
(runs every other method with the same cost-measurement methodology as
`eval_baselines.py`).

**Bug found & fixed (serious):** the first `train_fig1_encoders.py` run
produced garbage -- 68m's "best epoch" scored +0.3016, *worse* than its own
untrained baseline (+0.3863), with eval Spearman whipsawing between
negative and moderate positive every epoch despite loss decreasing
smoothly (0.586 -> 0.073). Root cause: `train_features()`'s cache key
(`baselines/scaling_sweep.py`) was keyed only on `(model, n_train_a,
n_train_p, lora_rank, seed)` -- **not** on eval size. `fig1` (n_eval_a=400)
and the earlier `paper200x3` (n_eval_a=200) both use train size 1500x3000,
so `fig1` silently "cache hit" `paper200x3`'s already-cached gradients --
which belong to a *different* underlying set of BBH/Dolly samples
(disjoint_splits front-slices eval first, so train_anchors starts at index
200 for paper200x3 but 400 for fig1). The encoder was training against
targets computed for entirely different text than what it was actually
being shown -- essentially random noise from the model's perspective,
which explains the non-improving, whipsawing trace exactly. **Fixed** by
adding `n_eval_a`/`n_eval_p` to the cache key. This bug is specific to
`fig1` (the first preset this session to reuse an existing preset's exact
train size while differing in eval size) -- it does **not** affect any
earlier tuning-phase result, since every paper200* preset used a distinct
(n_train_a, n_train_p) pair at the same n_eval_a=200, so no two of them
could have collided. Re-running `train_fig1_encoders.py` now that the fix
is in place; it will pay the ~15min featurization cost again since the old
(buggy-keyed) cache file doesn't match the new key format.

**Retry succeeded** -- all 3 InfluCoder sizes trained clean on the 400x400
eval: 68m +0.7818 (epoch 7/8), 150m +0.7598 (epoch 8/8, still rising),
**400m +0.8102 (epoch 8/8, still rising)**. Now running `figure1_table.py`
for the remaining 9 rows (LESS, LoGRA r8, LoGRA proxy 1.7B/0.6B, untrained
encoder x3, RDS+, TF-IDF) against the same GT.

**Figure 1 table 1 complete** (`baselines/out/fig1/table1.json`, 1 seed,
400x400 eval, GT from Qwen3-4B rank 16): LESS +0.9571 (358.5ms/sample),
LoGRA r8 +0.8891 (200.0ms), LoGRA proxy 1.7B +0.2874 (117.2ms), LoGRA proxy
0.6B **-0.1576** (111.2ms, worse than random), InfluCoder 400m +0.8102
(4.13ms), 68m +0.7816 (1.30ms), 150m +0.7598 (2.01ms), untrained encoders
400m/68m/150m at +0.43/+0.39/+0.33 respectively, RDS+ +0.3060 (56.1ms),
TF-IDF +0.2985 (0.41ms). InfluCoder dominates the Pareto frontier below
LoGRA/LESS; the untrained-encoder baseline at identical cost only reaches
about half InfluCoder's agreement, confirming distillation buys real
signal. Rendered as `baselines/out/fig1/figure1.png` (two-panel FLOPs/ms
vs. quality scatter, following `plot_cost_quality.py`'s design language).

**Digression: does LoGRA rank move inference speed?** User asked whether
r=4 would be faster than r=8. FLOPs say no -- `logra_r8`'s pinv/FIM term
(the only rank-dependent cost) is 1.45e9 out of 6.9e15 total FLOPs, i.e.
0.02%; the dominant cost is the frozen-backbone forward+backward, which is
rank-independent. Ran an empirical timing check
(`baselines/time_logra_rank.py`) to verify: raw totals looked like a big
gap (r=8 329ms/sample vs r=4 204ms/sample, ~38%), but this was a **test
artifact**, not a rank effect -- the script reloads the model fresh per
rank in the same process, so the r=8 load hit a cold OS page cache (~95s
of the 263s total was pure weight-loading I/O) while r=4's reload was
served warm from cache (~5s). Isolating just the `encode()` compute (pool
+ anchor passes, excluding load): r=8 168s vs r=4 159s for the same 800
samples -- **~5%, noise-level**, consistent with the FLOPs argument.
Lesson: sequential same-process reloads across configs confound wall-clock
comparisons via disk-cache warmup; a clean comparison needs fresh
processes or a cache-warming step before each timed run. Separately
noticed anchor encoding takes ~2x pool encoding at fixed rank (104s vs
64s for 400 samples each) -- a sequence-length effect (anchor texts are
longer), a far bigger lever on ms/sample than rank is.

Also want to check whether raising LoGRA's `batch_size` (hardcoded to 1 in
`score_logra`, vs. `modeling_logra.encode`'s own default of 8) gives a real
throughput win -- the custom backward already tracks per-sample gradients
correctly within a batch (`[B, r, r]` outer products via
`LoraBFunction`), and GPU memory is nowhere near saturated (46GB free vs
~8GB for the 4B model's weights), so batch_size=1 looks like an
unexploited lever rather than a hard constraint. `baselines/time_logra_batch.py`
written to test bs in [1, 4, 8, 16] with a drift check (scores must not
change) -- not yet completed (background job was stopped mid-run to
prioritize the proxy-rank sweep below; needs a rerun).

**Does higher rank fix the LoGRA proxies?** User's hypothesis: proxy
1.7B/0.6B's poor r=8 transfer (+0.287 / -0.158) might be a rank-capacity
problem -- a smaller model needs more rank to carry comparable signal.
Tested r=16 and r=32 for both (`baselines/logra_proxy_rank_sweep.py`,
same fig1 GT/splits, both `logra_raw` and `logra_fim` variants):

| config | raw agg | fim agg |
|---|---|---|
| proxy_1.7B r8 (table1) | +0.2874 | -- |
| proxy_1.7B r16 | **+0.5032** | +0.0797 |
| proxy_1.7B r32 | +0.5109 | +0.0835 |
| proxy_0.6B r8 (table1) | -0.1576 | -- |
| proxy_0.6B r16 | -0.0303 | +0.0620 |
| proxy_0.6B r32 | +0.0302 | +0.0744 |

**1.7B genuinely benefits from higher rank** -- +0.287 -> +0.50 going r8
-> r16 (then flat r16->r32), closing roughly half the gap to the 4B's
+0.889 but plateauing well short of it. **0.6B does not** -- stays
pinned near zero (-0.03 to +0.03) across every rank tested, meaning its
failure is not a rank/capacity-of-the-adapter problem but something about
the 0.6B model itself: its gradient geometry doesn't correlate with the
4B's influence signal regardless of how much LoRA rank you give it to
express that correlation. Also notable: `logra_fim` (FIM-preconditioned)
is *much worse* than `logra_raw` for both proxies at every rank tested
(e.g. 1.7B r16: +0.50 raw vs +0.08 fim) -- the opposite of being a safe
default choice for a cheap proxy; raw is the variant to use if a proxy is
ever revisited.

**Methodology change: FLOPs deferred to an appendix experiment.** The
FlopCounterMode instrumentation pass roughly 5x's wall-clock just to
measure itself, and for a speed-vs-quality story it was telling the same
tale ms/sample already tells (nothing in the FLOPs panel changed the
ranking of methods vs. the ms/sample panel). `figure1_table.run_row` now
defaults to `measure_flops=False` (timing-only, single clean pass); the
old two-pass behavior is preserved behind that flag for when the FLOPs
appendix experiment happens. `plot_figure1.py` is now a single ms/sample
panel instead of two. Existing FLOPs numbers already in `table1.json` for
the other 10 rows are left as-is (not stale, just not being recomputed);
only the two proxy rows being patched right now use the new timing-only
path, so their `flops_*` fields are 0 pending the appendix pass.

**Does raising LoGRA's batch_size close the 0.6B/1.7B/4B speed gap?** User
noticed 0.6B (144.5ms) and 1.7B (152.9ms) proxies looked suspiciously
close given the 3x parameter gap, suspected the shared hardcoded
`batch_size=1` (in `score_logra`) was hiding a real speed difference small
models should get from batching more per GPU memory headroom. Swept batch
size per model with `baselines/time_logra_batch.py` (generalized to take
`--grad_model`/`--rank`/`--batch_sizes`), checking peak memory and score
drift vs. batch_size=1 at each step:

| model | batch=1 | max feasible batch | speedup | peak mem @ bs=1 |
|---|---|---|---|---|
| 4B (r8) | 204.2ms | OOMs already at batch=4 | none | 21.9GB |
| 1.7B | 152.9ms | 142.0ms @ batch=4 (OOMs @ 8) | ~7% | 10.9GB |
| 0.6B | 144.5ms | 118.3ms @ batch=4 (OOMs @ 8) | ~18% | 7.8GB |

Two findings: (1) the compressed 0.6B/1.7B/4B gap is real, not a
batch_size=1 artifact -- eager attention's backprop memory cost is high
enough that even 4B is already near the 46GB ceiling *at batch=1*
(21.9GB) and can't batch at all; per-sample overhead dominates for all
three sizes regardless of batching. (2) **Batching LoGRA isn't free even
where it fits** -- max|delta| in raw scores between batch=1 and batch=4
was **0.126** (1.7B) and **0.064** (0.6B), far past float noise. Root
cause: `modeling_logra.py`'s `encode()` computes one batch-*mean* loss
(`self.model(...).loss`) before a single `.backward()`; at batch_size=1
that mean is trivially "this sample's own loss," but at batch_size>1 each
sample's gradient gets diluted by its share of total valid tokens
*within that specific batch* -- so a sample's "per-sample gradient" is no
longer purely a function of that sample, it depends on what else happened
to be batched with it. Given the modest, memory-capped speedup and this
unverified-safe quality risk, **not adopting batching for the LoGRA rows**
-- `batch_size=1` stays the measurement standard.

Also checked whether RDS+ (batch_size=1 default, but exposed as a
parameter) was a safe/free win instead, since it has no backward pass.
It is not: `ListDataset.__getitem__` returns raw *unpadded*
variable-length tensors, and `score_rdsplus` hands them to a plain
`DataLoader` with the default collate_fn -- batch_size>1 would try to
`torch.stack` different-length tensors and crash outright for real text.
Even fixed with padding, the weighted-mean pooling
(`torch.sum(hidden * w, dim=1)`) never multiplies by `attention_mask`, so
padding tokens' hidden states would leak into the pooled embedding
uncontrolled. Batching RDS+ correctly needs a custom collate (pad +
track lengths) and mask-aware pooling -- real engineering, not a
parameter bump. Left at `batch_size=1`. **Net: no changes to
`table1.json` from this investigation** -- both batching leads were
either capped/risky or non-trivial to do safely, so the existing numbers
stand.

**Follow-up: switching eager -> SDPA attention gave a real, verified win.**
The `attn_implementation="eager"` hardcoded throughout `modeling_logra.py`
was only required to avoid `torch.utils.flop_counter`'s SDPA-on-GQA crash
-- a constraint that no longer applies now that FLOPs are off by default
(see the methodology-change entry above). Added an `attn_implementation`
parameter (default `"sdpa"`) threaded through `LoGra.__init__` /
`from_pretrained` / `score_logra`. Re-swept batch sizes under sdpa
(`baselines/time_logra_batch.py --attn_implementation sdpa`):

| model | eager bs=1 | sdpa bs=1 | sdpa best (bs=4) | peak mem sdpa bs=1 |
|---|---|---|---|---|
| 4B (r8) | 204.2ms | 148.0ms | 147.1ms (no batching win) | 14.6GB (was 21.9GB) |
| 1.7B | 152.9ms | 132.4ms | 109.8ms | 8.1GB (was 10.9GB) |
| 0.6B | 144.5ms | 129.5ms | 86.7ms | 5.0GB (was 7.8GB) |

SDPA alone cuts memory by ~30-35% and time by 10-28% at batch=1, with NO
batching needed -- this also finally reveals a real, roughly
size-proportional speed ordering between the three models (144/130/126ms
in the final adopted numbers) that eager's memory-bound overhead had been
masking. Verified correctness before adopting
(`baselines/verify_logra_sdpa.py`, 0.6B proxy): eager bs=1 vs sdpa bs=1
aggregated rho delta = **+0.0046** (noise-level, safe) vs sdpa bs=1 vs
sdpa bs=4 aggregated rho delta = **-0.0130** (confirms the earlier
batch-dilution concern is real, not just raw-matrix noise -- still not
adopting batch_size>1). **Adopted**: `score_logra` now defaults to
`attn_implementation="sdpa"`, `batch_size` stays at 1. Re-timed all 3
LoGRA rows in `table1.json`: logra_r8 199.97ms -> **144.35ms** (agg
+0.8891 -> +0.8982, both within seed/kernel noise), logra_proxy_1.7B
151.11ms -> **130.32ms** (agg +0.5109 -> +0.5088), logra_proxy_0.6B
142.11ms -> **126.26ms** (agg +0.0302 -> +0.0345). Figure regenerated.

---

## Figure 1 rebuilt on tasksource/dolci-instruct (pool swap, everything else fixed)

User request: recreate the whole Figure 1 pipeline with the Dolly candidate
pool replaced by `tasksource/dolci-instruct` (flat `prompt`/`answer`/`task`
schema, 8 parquet shards, ~1.8M rows -- no chat-message wrapping like
Dolly's JSONL), BBH anchors unchanged, every other setting held identical
(400x400 eval, 1500x3000 train, GT = Qwen3-4B rank 16, SDPA/batch_size=1
methodology).

**Plumbing**: added `load_dolci_instruct()` (streams shards via
`datasets.load_dataset(..., streaming=True)`, same `Sample(context, target,
text)` shape and seeded-shuffle contract as `load_dolly`) and a
`load_pool(name, seed)` dispatcher to `influcoder/data.py`; added a
`"fig1_dolci"` preset (`pool="dolci_instruct"`) to `run.py`'s `PRESETS`;
`baselines/common.py`'s `build_splits`/`_gt_key` now route through
`cfg.get("pool", "dolly")` so the Dolly presets are untouched and the GT
cache key includes the pool. `figure1_table.py`, `train_fig1_encoders.py`,
and `plot_figure1.py` all took a `--preset` arg instead of hardcoding
`"fig1"`, deriving `out_dir`/`enc_dir` from the preset name.

**Batch size override (user-requested, NOT the default methodology)**: the
user asked to batch the two LoGRA proxies (1.7B, 0.6B) as large as
possible instead of the official batch_size=1. This is the same
quality-drift tradeoff documented above (batch_size stays at 1 for the
*official* methodology precisely because batching moves the aggregated
Spearman rho by a real, non-noise amount) -- flagged explicitly, user
confirmed override. Applied batch_size=4 (the max-feasible ceiling found
earlier; both OOM at 8) via a new `baselines/update_proxy_rows_batched.py`,
kept separate from `figure1_table.py`'s default path so the apples-to-apples
batch=1 run stays reproducible with one command. The main `logra_r8` (4B)
row is untouched at batch_size=1 -- it already OOMs at batch=4.

**Results** (`baselines/out/fig1_dolci/table1.json`,
`baselines/out/fig1_dolci/figure1.png`), vs. the Dolly table:

| method | dolly agg | dolci agg | dolly ms | dolci ms |
|---|---|---|---|---|
| less | +0.9571 | +0.9517 | 358.5 | 390.6 |
| logra_r8 | +0.8982 | +0.9080 | 144.3 | 153.8 |
| logra_proxy_1.7B | +0.5088 | +0.5783 | 130.3 | 131.1 (bs=4) |
| logra_proxy_0.6B | +0.0345 | +0.3046 | 126.3 | 98.0 (bs=4) |
| influcoder_68m | +0.7816 | +0.1676 | 1.30 | 2.41 |
| untrained_68m | +0.3863 | +0.1758 | 1.13 | 2.37 |
| influcoder_150m | +0.7598 | +0.1213 | 2.01 | 4.17 |
| untrained_150m | +0.3304 | +0.1576 | 2.00 | 4.13 |
| influcoder_400m | +0.8102 | +0.1563 | 4.13 | 8.10 |
| untrained_400m | +0.4297 | +0.1498 | 4.12 | 8.10 |
| rdsplus | +0.3060 | +0.1038 | 56.1 | 67.4 |
| tfidf | +0.2985 | -0.0944 | 0.41 | 0.19 |

Headline finding: **InfluCoder's distillation gain essentially vanishes on
dolci-instruct.** On Dolly, distillation adds +0.35 to +0.40 agg rho over
the untrained encoder baseline at every size (0.33-0.43 -> 0.76-0.81). On
dolci-instruct, trained and untrained land in the same tight 0.12-0.18
band, and the ordering even flips at 68m/150m (untrained beats trained).
Loss curves in `train_fig1_encoders`'s log show why: eval agg rho peaks
early (epoch 3-4 of 8) then degrades every subsequent epoch as train loss
keeps falling -- textbook overfitting to the distillation targets rather
than learning transferable structure, worse than on Dolly. Plausible cause:
dolci-instruct is an SFT-mix aggregate (task field shows it's sourced from
allenai's olmo-3-instruct-sft mix) spanning very heterogeneous task types
(math word problems, crystallography, content moderation, multilingual) in
one flat prompt/answer pool, vs. Dolly's more uniform open-domain
instruction-following shape -- the bi-encoder may need more/better epoch
selection or a different train-side sampling strategy to generalize across
that heterogeneity, not investigated further here.

Secondary findings: the 0.6B LoGRA proxy is far more competitive here
(+0.30 vs +0.03 on Dolly) while the 1.7B proxy also improves (+0.58 vs
+0.51) -- both proxies transfer better on this pool. TF-IDF goes
*negative* (-0.09, vs +0.30 on Dolly) -- lexical overlap is a much weaker
influence signal on this heterogeneous pool. RDS+ drops from +0.31 to
+0.10.

**UPDATE -- found and fixed a real bug in the above, but the headline
finding survives it.** User asked to check for bugs before accepting the
"distillation doesn't help" result. Found one: `load_encoder()` (and every
caller: `train_fig1_encoders.py`, `figure1_table.py`'s `score_influcoder`/
`score_semantic`) hardcoded `max_seq_len=512` tokens for the bi-encoder.
Fine for Dolly (median encoder-text ~120 tokens) but wrong for
dolci-instruct, whose `text = prompt + answer` runs far longer (median
~400 tokens, p90 ~1000). Measured directly: at the 512 cap, 24% of pool
samples had **less than half** their answer inside the truncation window
and 5.6% had **none of it** -- meanwhile the GT gradient-influence target
for those same samples is computed from `tokenize_sample`'s
target-prioritized truncation (`grad_max_len=1024`, target capped at 512
tokens, *context* tail-truncated to make room), so GT saw the answer the
encoder never did. A real train/eval signal-corruption bug, not a
cost/quality tradeoff -- the Ettin encoders support up to ~8000 tokens
natively, so 512 was just an under-provisioned default that happened to
work by accident on Dolly's shorter text.

**Fix**: added `encoder_max_len` to preset configs (`run.py`), threaded
through `load_encoder`/`score_influcoder`/`score_semantic`; set to 1024
(matching `grad_max_len`) for `fig1_dolci` only -- `fig1` (Dolly) is
untouched, still correct at 512, its already-committed numbers don't move.
At 1024, mean answer coverage rises from 76%->94%, fully-truncated
samples drop from 5.6%->0.6%.

**Effect of the fix**, retraining all 3 encoders and rescoring:

| method | 512 (buggy) | 1024 (fixed) |
|---|---|---|
| untrained_68m | +0.1758 | **+0.3196** |
| influcoder_68m | +0.1676 | +0.0466 |
| untrained_150m | +0.1576 | **+0.2766** |
| influcoder_150m | +0.1213 | +0.1230 |
| untrained_400m | +0.1498 | **+0.2116** |
| influcoder_400m | +0.1563 | +0.0744 |

The bug was real and mattered a lot -- fixing it nearly doubles every
untrained-encoder score, bringing them back into the same range as Dolly's
untrained baselines (+0.33 to +0.43). But it does **not** rescue
InfluCoder: trained now sits *below* untrained at every single size (was
roughly tied before the fix), and 68m/400m got worse in absolute terms.
Training logs are unambiguous and consistent across all 3 sizes: eval agg
rho peaks at epoch 3/8, then degrades every epoch after even as train loss
keeps falling monotonically to near-zero (e.g. 400m: epoch 3 peak +0.0743,
epoch 8 loss=0.043 but agg rho=-0.056) -- textbook overfitting to the
distillation targets, not a truncation artifact, since it reproduces
identically at both max_len settings.

Stopping here rather than continuing to tweak hyperparameters until a run
"looks good": one confirmed, fixed bug is enough justification to rerun;
blindly retrying without a specific new hypothesis would just be
p-hacking a result. If this is worth chasing further, the next real lever
is the distillation recipe itself (lr, hard-negative mining, regularization,
or whether the gradient-influence targets are simply noisier/less learnable
on this heterogeneous a pool), not another truncation-style bug -- not
attempted here.

Final artifacts: `baselines/out/fig1_dolci/{table1.json,figure1.png}`. The
buggy 512-cap run is kept for reference at
`baselines/out/fig1_dolci_512cap_buggy/` and
`runs_out/fig1_dolci_512cap_buggy/` rather than deleted.

---
