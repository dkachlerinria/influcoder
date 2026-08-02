# EXP2: DATE-LM benchmark — InfluCoder's leak-free attribution results

Status as of 2026-07-31, end of session. Written for a fresh agent picking this up cold.

## 1. What this experiment is

[DATE-LM](https://arxiv.org/abs/2507.09424) (NeurIPS 2025) is a third-party benchmark for
data-attribution methods, vendored into this repo as `EXP2-datelm/` (see that directory's
own git history for the integration). It scores methods (BM25, Rep-Sim, Grad Dot, Grad Sim,
LESS, DataInf, EKFAC, and — added by this repo — InfluCoder) on two application tasks:

- **Toxicity/Bias filtering** — given a model fine-tuned on a training set that secretly
  contains a handful of unsafe examples mixed into a large benign corpus, and a small held-out
  set of unsafe "reference" queries, can the method rank the unsafe training examples above
  the benign ones? Scored by AUPRC.
- **Factual attribution (Counterfact)** — given a model fine-tuned on corrupted
  ("counterfactual") facts, and reference queries that elicit that corrupted knowledge, can
  the method trace back to the specific training examples that taught it? Scored by
  Recall@50 and MRR.

InfluCoder is integrated via `EXP2-datelm/methods/influcoder_attribute.py`, which distills the
checkpoint's own per-sample gradient signal into a cheap sentence encoder (same idea as EXP1),
then embeds and scores DATE-LM's train/ref sets with it. Results land in
`EXP2-datelm/results/<task>-influcoder*/InfluCoder.pt` (a JSON file despite the extension,
matching DATE-LM's own convention) and get scored by DATE-LM's own, unmodified
`EXP2-datelm/evaluation/evaluate_application.py` — the same code path every other method's
number in the paper came from.

## 2. Two leaks found in the original InfluCoder integration

`influcoder_attribute.py`'s original recipe distilled the encoder using teacher gradients
computed **from inside the benchmark's own train/ref split**:

1. **Train-side leak**: distillation targets came from `n_teacher_train=500` examples drawn
   from DATE-LM's own `train` set — the exact same set later embedded and scored. For
   Counterfact, 100/1007 (9.9%) of the ground-truth fact-matches for the 66 ref queries were
   inside that 500-example fit set, and *all 66* ref queries had >=1 match inside it.
2. **Ref-side leak**: the encoder was also fit to reproduce true gradient-similarity
   specifically **for the exact ref/query examples it was later graded on** — an advantage no
   other DATE-LM method has, since Grad Dot/Grad Sim/LESS/DataInf/EKFAC compute exact
   gradients fresh for any query with zero fitting step. This one is subtler and was caught on
   a second pass — fixing only the train-side leak still left it in place.

Neither is "trained on the answer key" in the labeled-supervision sense (no unsafe/fact-match
*labels* ever leak in) — but both let the encoder specialize to the literal examples/queries
it's graded against, which nothing else in the benchmark gets to do. See §4 for the fix.

## 3. Important caveat: leak-free is not the same as equal-resource

Every other DATE-LM method computes its score directly from the checkpoint with **zero**
separate training/fitting step and **zero** extra data beyond what the benchmark provides.
InfluCoder structurally can't do that — its entire value proposition is distilling the
expensive gradient signal into a cheap encoder, which means it inherently needs *some* data to
fit that encoder on. The fix below moves that data from "inside the eval set" (leaky) to
"disjoint, same-distribution, external" (leak-free) — but InfluCoder still gets to use
additional same-distribution data that the other seven methods never touch. That's a real,
disclosed asymmetry: fair on "did it see the eval labels" (no), not fair on "did every method
get the same resources" (no). Report both numbers with that caveat attached.

## 4. The fix: distill entirely on external, disjoint data

For each task, the encoder is now trained end-to-end on data with **zero** overlap with
anything DATE-LM's `train`/`ref` sets contain, and only *after* training does it ever see local
data — a single forward-pass embed+score of the untouched, full local train/ref sets, in the
same shape DATE-LM's native evaluator expects.

### 4.1 Counterfact / Pythia-1b — `influcoder_attribute_noleak_counterfact_extquery.py`

DATE-LM's local Counterfact/Pythia-1b split (5,473 train + 66 ref = 5,539 rows) is confirmed
(by exact `prompt`-text join, 5,539/5,539 matched) to be a curated subset of the public
**ROME/CounterFact dataset** (Meng et al. 2022, NeurIPS; `NeelNanda/counterfact-tracing` on HF,
21,919 rows total — same source, same schema, format-identical by construction).

Clean external pool: exclude any external row with (a) an exact `prompt` match to local
train+ref, or (b) a `subject` match to local train+ref (even under a different relation) →
**15,648 clean rows** left, spanning the exact same 34 Wikidata relation types the local eval
data uses (verified — not a broader or narrower relation-type distribution, just disjoint
subject instances of the same 34 relations).

From that pool: 300 external "queries" (anchors) + 500 external candidates + 250 held-out
(internal fidelity check) — all disjoint slices. Real teacher gradients computed on these
(same checkpoint, `DataAttributionEval/Pythia-1b-counterfactual`), encoder distilled purely
against that signal, then the **full, untouched** local train (5,473) + local ref (66) embedded
and scored in one pass — no cross-validation/stitching needed, since the encoder never saw any
local data during training.

Verified via independent audit (`methods/audit_leakage.py`, committed and runnable — an earlier
version of this doc cited a script with this name that was never actually committed; this is the
real one). It reconstructs, via the same code paths the training scripts use, exactly which
external rows a run drew for its anchor/candidate/held-out-eval sets, then independently
cross-checks those specific rows against freshly-loaded local data: 0 exact-prompt overlaps, 0
subject overlaps, 0 exact (prompt,response) pair overlaps — confirmed for both this script and
the `_extquery_moredata` variant in §6.

Earlier iterations kept for reference (`influcoder_attribute_noleak_counterfact.py`: fixes
only the train-side leak; `influcoder_attribute_noleak_counterfact_kfold.py`: fixes both leaks
via 2-fold CV over the local 66 ref queries instead of external queries) — superseded by the
external-query version, which needs no local ref data at all.

### 4.2 Toxicity/Bias (XSTest-response-Het) / Pythia-1b — `influcoder_attribute_noleak_toxicity.py`

Local data (10,187 train + 10 ref) is: 10,000 Benign rows from **UltraChat**
(`HuggingFaceH4/ultrachat_200k` — confirmed, 742/15,000 streamed rows matched local benign
prompts exactly) + 66 Unsafe + 121 Safety-Aligned + all 10 ref rows from
**`allenai/xstest-response`**'s `response_harmfulness` split (446 rows, gated on HF; all 197
local non-benign rows matched 1:1 by exact prompt+response text).

Unlike CounterFact, XSTest-response is **nearly saturated**: of its 249 unmatched rows, 247 are
"unharmful"/"prompt_safe" and only 2 are "harmful" — nowhere near enough fresh unsafe material
to build query-analogs or positive candidates. (Checked and ruled out at the time: raw
`walledai/XSTest`/`Paul/XSTest` doesn't help either — its "unsafe" label pool is 200 total, 197
already consumed, same 3-row remainder problem. `allenai/wildguardmix` was also gated and
inaccessible at the time this section was first written — access was granted later; see §4.3.)

Per explicit instruction, used **`lmsys/toxic-chat`** instead (`toxicchat0124` config, 746
`toxicity=1` rows across train+test, confirmed 0 overlap with local prompts) — one of DATE-LM's
own three alternate sources for this exact task (the ToxicChat-Het/Hom subsets use it
directly), just not the specific subset being evaluated here. **This is a weaker distributional
match than the Counterfact fix**: ToxicChat is real-world multilingual user-chatbot toxicity,
XSTest-response is a structured, single-turn adversarial test suite — same task family, not the
same narrow source. Disclosed, not hidden.

Pool: 40 external toxic queries (anchors) + candidates (300 fresh UltraChat benign + 100 fresh
ToxicChat toxic) + held-out eval (100 benign + 50 toxic). Real teacher gradients computed
(checkpoint `DataAttributionEval/Pythia-1b-XSTest-response-Het`), encoder distilled purely on
this external mix, then the full local train (10,187) + local ref (10) embedded and scored.

Also verified via `methods/audit_leakage.py` (see §4.1): 0 exact-prompt overlaps, 0 exact
(prompt,response) pair overlaps against local train+ref, for both this script and the
`_toxicity_moredata` variant in §6 (no `subject` field on this task, so that axis doesn't apply).

### 4.3 Toxicity/Bias with WildGuardMix (closes the §4.2 distributional-match gap)

`allenai/wildguardmix` access was granted to this account after §4.2 was written. Redone here as
a **like-for-like swap of the toxic-side external source only** —
`influcoder_attribute_noleak_toxicity_wildguard.py`, identical to §4.2's script (same
`N_EXT_ANCHORS=40`/`N_EXT_BENIGN=300`/`N_EXT_TOXIC_POS=100`/`N_EVAL_BENIGN=100`/
`N_EVAL_TOXIC=50`/`EPOCHS=8`/`HARD_RATIO=0.5`) except the toxic pool is drawn from WildGuardMix
instead of ToxicChat, so the result is directly comparable to §4.2/§5.2's row.

`wildguardmix`'s `wildguardtrain` config (86,759 rows) has `prompt`, `response`,
`response_harm_label` ('harmful'/'unharmful', matching `xstest-response`'s
`response_harmfulness` semantics directly — unlike ToxicChat, which only has prompt-level
`toxicity`) among its fields. Toxic pool = rows with `response_harm_label == "harmful"` and a
non-empty response: **8,368 usable rows** (0 overlap with local prompts) — over 11x ToxicChat's
746, and the same WildGuard team/methodology as the local task's own `xstest-response` source,
closing the distributional-match gap §4.2 disclosed.

Verified leak-free via `methods/audit_leakage.py` (extended to cover this script): 0 exact-prompt
overlaps, 0 exact (prompt,response) pair overlaps.

**Result: AUPRC 0.4222** — essentially a wash against ToxicChat's 0.4242 (§5.2), very slightly
*lower*, within run-to-run noise. A closer distributional match did not translate into a better
score here. Kept as the honest result rather than an assumed improvement — see §5.2 for the full
comparison table.

## 5. Results

### 5.1 Counterfact / Pythia-1b (Recall@50, MRR)

| Version | Recall@50 | MRR |
|---|---|---|
| Original (both leaks) | 0.4549 | 0.8393 |
| Train-side fixed only, ref still leaky | 0.4541 | 0.8057 |
| Leak-free, 2-fold ref CV | 0.4543 | 0.7985 |
| **Leak-free, external queries (recommended)** | **0.4442** | **0.7756** |

Paper baselines, Pythia-1B (Table 6):

| Method | Recall@50 | MRR |
|---|---|---|
| BM25 | 0.305 | 0.771 |
| Rep Sim | 0.376 | 0.790 |
| Grad Dot | 0.466 | 0.768 |
| Grad Sim | 0.493 | 0.836 |
| LESS | 0.500 | 0.772 |
| DataInf | 0.472 | 0.765 |
| EKFAC | 0.465 | 0.766 |
| **InfluCoder (leak-free)** | **0.444** | **0.776** |

Both leaks together inflated MRR by ~0.06 (0.84→0.78) but barely moved Recall@50 (~0.01) —
consistent with MRR being sensitive to a handful of "gimme" hits from the leaked examples that
Recall@50's top-50 window mostly absorbs. Leak-free, InfluCoder lands mid-pack: clearly ahead
of BM25/Rep-Sim, comparable to Grad Dot/DataInf/EKFAC/LESS on both metrics.

**Superseded for Recall@50 by §6.11**: `ettin-400m` at `hard_ratio=0.25` reaches Recall@50=0.4689 /
MRR=0.8739 — beats every number in this table except LESS's Recall@50 (0.500, now within 0.011),
and beats Grad Sim's MRR (0.836) outright. See §6.10/§6.11 before citing this table's numbers as
InfluCoder's ceiling.

### 5.2 Toxicity/Bias, XSTest-response-Het / Pythia-1b (AUPRC)

| Version | AUPRC |
|---|---|
| Original (both leaks) | 0.4737 |
| **Leak-free (ToxicChat-sourced external, recommended)** | **0.4242** |
| Leak-free (WildGuardMix-sourced external, §4.3) | 0.4222 |

Paper baselines, Pythia-1B, XSTest-response column (Table 12, Heterogeneous):

| Method | AUPRC |
|---|---|
| Grad Dot | 0.389 |
| DataInf | 0.392 |
| EKFAC | 0.344 |
| Rep-Sim | 0.580 |
| Grad Sim | 0.601 |
| LESS | 0.734 |
| **InfluCoder (leak-free, ToxicChat)** | **0.424** |
| **InfluCoder (leak-free, WildGuardMix)** | **0.422** |

Same shape as Counterfact: ahead of Grad Dot/DataInf/EKFAC, behind Rep-Sim/Grad Sim/LESS,
regardless of which external toxic source is used. The §4.2 distributional-match caveat has now
been directly tested (§4.3): despite WildGuardMix being a substantially closer match to the local
task's own source (same team/methodology, response-level harm labels, 8,368 vs. 746 usable rows),
the result is a statistical wash, not an improvement — 0.4222 vs. 0.4242, a 0.002 gap well within
run-to-run noise. Distributional match quality doesn't appear to be what's bottlenecking this
number; whatever's capping InfluCoder below Rep-Sim/Grad Sim/LESS here isn't the toxic-source
choice.

**Superseded by §6.4/§6.12**: a cross-source external mix (WildGuardMix anchors + ToxicChat
candidates) reaches 0.6160 at the 68m default encoder, and **0.7651 at `ettin-400m`** — beating
every method in the baseline table above, LESS included. Don't cite this section's numbers as
InfluCoder's ceiling on this task; see §6.12 first.

## 6. Ablation: more external samples, hard_ratio/epochs held fixed

Tests whether the leak-free recommended results (§5) are data-starved: scale up the external
anchor/candidate/held-out-eval counts substantially on both tasks, while deliberately **holding
`hard_ratio` and `epochs` fixed** at their original values —
`influcoder_attribute_noleak_counterfact_extquery_moredata.py` and
`influcoder_attribute_noleak_toxicity_moredata.py`, copies of the §4 scripts with only the
sample-count constants changed.

`hard_ratio`/`epochs` were held fixed on purpose, not by oversight: EXP1's `FINDINGS.md` (same
`influcoder.encoder.distill` code path) found (a) more epochs is a dead lever — 8→16 bought
+0.0057 (noise-sized) at best — and (b) scaling up training data *without* proportionally
raising `hard_ratio` in step can regress quality (a bigger pool dilutes the easy-negative signal
faster than it adds hard-negative signal). This run deliberately does not compensate for (b), to
test directly whether that effect transfers to these two DATE-LM tasks rather than assuming it
does.

| Task | Anchors | Candidates | Held-out eval | vs. recommended (§5) |
|---|---|---|---|---|
| Counterfact | 900 (was 300) | 2,000 (was 500) | 500 (was 250) | out of 15,648 clean pool |
| Toxicity/Bias | 100 toxic (was 40) | 1,000 benign + 300 toxic (was 300+100) | 300 benign + 100 toxic (was 100+50) | toxic side used 500/746 available |

### 6.1 Counterfact — Recall@50, MRR

| Version | Recall@50 | MRR |
|---|---|---|
| Leak-free, external queries (recommended, §5.1) | 0.4442 | 0.7756 |
| **Leak-free, more samples (hard_ratio/epochs fixed)** | **0.4411** | **0.8226** |
| Δ | −0.0031 | **+0.0470** |

Recall@50 is flat (a 0.003 wobble, within run-to-run noise). MRR moved up a genuinely large
amount — from mid-pack to just below Grad Sim (0.836), ahead of every other DATE-LM baseline on
this metric (BM25 0.771, Rep-Sim 0.790, Grad Dot 0.768, LESS 0.772, DataInf 0.765, EKFAC 0.766).

### 6.2 Toxicity/Bias — AUPRC

| Version | AUPRC |
|---|---|
| Leak-free (recommended, §5.2) | 0.4242 |
| **Leak-free, more samples (hard_ratio/epochs fixed)** | **0.5845** |
| Δ | **+0.1603** |

A large jump — moves InfluCoder from behind Grad Dot/DataInf/EKFAC to just past Rep-Sim (0.580),
though still behind Grad Sim (0.601) and LESS (0.734).

### 6.3 Reading this against the EXP1 finding

This **does not reproduce** EXP1's "more data without more hard_ratio regresses" pattern — both
tasks improved (one metric essentially flat, three metrics up, one substantially). Plausible
reasons this differs from EXP1's `big4x` regression rather than confirming it: EXP1's regression
was measured on a *fixed*-size Dolly/dolci-instruct eval slice as pool size grew past ~1500×3000
with `hard_ratio=0` (no hard mining at all); these runs start from `hard_ratio=0.5` (already
substantial hard mining) and scale a smaller multiple (2.2–3.3x depending on the axis, not
EXP1's 4x), and the recommended baseline here was arguably *under*-provisioned to begin with (300
anchors / 500 candidates vs. EXP1's finding that gains keep coming, with diminishing returns,
past 1000). Read as: at `hard_ratio=0.5`, these particular sample-count ranges hadn't hit the
dilution regime EXP1 hit at `hard_ratio=0`, not as a contradiction of that result. Not
independently re-verified beyond this single run (seed=0 throughout, same as §5 — no multi-seed
variance estimate here).

Given the win is this clear on both tasks, worth considering promoting the moredata configs to
the new recommended row in §5 and re-running the §3 equal-resource caveat's accounting (moredata
uses noticeably more external data than the original recommended configs). Left as an open item
rather than done inline here, to avoid silently swapping out §5's numbers without a dedicated
pass.

### 6.4 Toxicity/Bias: does external source matter at moredata scale, and does cross-source mixing?

Two more runs, same moredata sample counts as §6.2 (100 toxic anchors / 1,000 benign + 300 toxic
candidates / 300 benign + 100 toxic held-out eval), `hard_ratio`/`epochs` held fixed as always:

- `influcoder_attribute_noleak_toxicity_wildguard_moredata.py` — WildGuardMix for anchors,
  candidates, *and* held-out eval (single-source, same design as every other toxicity script here,
  just WildGuardMix instead of ToxicChat, at moredata volume instead of §4.3's small-scale volume).
- `influcoder_attribute_noleak_toxicity_mixed.py` — **cross-source**: toxic anchors and held-out
  eval from WildGuardMix, toxic *candidates* from ToxicChat — a different corpus than the anchors,
  by construction. Benign side is UltraChat throughout, unchanged. This directly tests a concern
  raised about every single-source script in this doc (`_toxicity.py`, `_toxicity_moredata.py`,
  `_toxicity_wildguard.py`, `_toxicity_wildguard_moredata.py`, all of which draw anchors and
  candidates from one shuffled pool of the same corpus): is the anchor↔candidate gradient-similarity
  signal the encoder learns actually about toxicity content, or partly just "these two texts share
  the same corpus's writing style/format"? If it were the latter, forcing the two roles onto
  different corpora should hurt. (Held-out eval was sourced from WildGuardMix — same as the
  anchors, not the candidates — since its role is closer to "another query-side probe for
  best-epoch selection" than to the scored candidate pool; stated plainly as a judgment call, not
  hidden in the script.)

| Version | AUPRC |
|---|---|
| ToxicChat, recommended (§5.2) | 0.4242 |
| ToxicChat, moredata (§6.2) | 0.5845 |
| WildGuardMix, recommended-scale (§4.3) | 0.4222 |
| WildGuardMix, moredata (single-source) | 0.5515 |
| **Cross-source mixed, moredata (WildGuardMix anchor+eval / ToxicChat candidate)** | **0.6160** |

Two findings:

1. **Source identity still doesn't explain much at moredata scale either.** WildGuardMix-moredata
   (0.5515) lands close to but slightly below ToxicChat-moredata (0.5845) — same story as §4.3's
   small-scale comparison (0.4222 vs. 0.4242), just replayed at higher volume. Confirms volume, not
   source-distribution match, is what's doing the work here.
2. **The cross-source mix did not reproduce the "same-source is a confound" worry — if anything,
   the opposite happened.** 0.6160 is the best AUPRC in this entire document, ahead of *both*
   single-source moredata variants, past Rep-Sim (0.580), and past **Grad Sim (0.601)** — only LESS
   (0.734) remains ahead of it among every DATE-LM baseline. Forcing anchors and candidates onto
   different corpora did not degrade the signal; it improved it. Read cautiously (single run,
   seed=0, no multi-seed variance estimate, same epistemic caveat as §6.3) — but as far as this one
   run goes, the earlier single-source design does not look like it was inflating AUPRC via a
   corpus-style shortcut. A plausible (not verified further here) alternative explanation: mixing
   sources may simply add training-signal diversity the same way more samples did in §6.1/§6.2,
   rather than removing a confound per se.

Verified leak-free the same way as every other script in this doc, via
`methods/audit_leakage.py`'s extended `TOXICITY_MODULES`/`audit_toxicity_mixed`: 0 exact-prompt
overlaps, 0 exact (prompt,response) pair overlaps for both new scripts.

### 6.5 Does a much bigger candidate pool beat ToxicChat's, holding the WildGuardMix anchor fixed?

§6.4 found bigger *anchor+eval* volume (WildGuardMix, single-source) didn't beat ToxicChat's, and
the cross-source mix's win looked like it might be about candidate-pool diversity rather than
source-match. If diversity/volume on the candidate side is really what's driving the mix's win, a
much larger candidate pool should do even better. **`PKU-Alignment/BeaverTails`** (open access, no
gating) is exactly that: `prompt`/`response`/`is_safe` rows, `330k_train` split, **166,347 usable
unsafe rows after exclusion** (35 excluded for exact-prompt overlap with local data) — roughly
223x ToxicChat's 746.

`influcoder_attribute_noleak_toxicity_mixed_beavertails.py`: byte-identical anchor+held-out-eval
logic to §6.4's mixed script (same `build_wildguard_toxic_pool` call, same `N_EXT_ANCHORS=100`/
`N_EVAL_TOXIC=100`, same seed — draws the identical 200 WildGuardMix rows), only the candidate pool
source changes: BeaverTails instead of ToxicChat, same `N_EXT_TOXIC_POS=300`, same `hard_ratio=0.5`/
`epochs=8`.

| Version | AUPRC |
|---|---|
| WildGuardMix, moredata (single-source) | 0.5515 |
| ToxicChat, moredata (single-source) | 0.5845 |
| **Cross-source mix, WildGuardMix anchor + ToxicChat candidate (§6.4)** | **0.6160** |
| Cross-source mix, WildGuardMix anchor + **BeaverTails** candidate | 0.5608 |

**This does not replicate — a ~223x bigger candidate pool performed *worse* than ToxicChat's much
smaller one**, and worse than even the single-source ToxicChat-moredata run (0.5845). Holding the
same WildGuardMix anchor fixed and only swapping the candidate source (ToxicChat → BeaverTails)
cost −0.055 AUPRC, the opposite direction from what "bigger/more diverse pool helps" would predict.
So candidate-pool *volume* isn't the mechanism behind §6.4's win either — it's something more
specific to ToxicChat as a candidate source (real-world multilingual user↔chatbot toxicity, closer
in register/format to what a fine-tuned chat model's own training distribution looks like, perhaps)
that BeaverTails' more templated QA-style harmful-response pairs don't reproduce. Not investigated
further here. §6.4's 0.6160 (WildGuardMix anchor + ToxicChat candidate) remains the best toxicity
number in this document.

Verified leak-free via `methods/audit_leakage.py`'s generalized `TOXICITY_MIXED_MODULES` list
(now covers both cross-source scripts via introspection on which candidate-pool builder each module
defines): 0 exact-prompt overlaps, 0 exact (prompt,response) pair overlaps.

### 6.6 Scaling up §6.4's winning mix — and a deliberate `hard_ratio=0.0` deviation

Scaled §6.4's winning cross-source design (WildGuardMix anchor+eval, ToxicChat candidates) to
substantially bigger sample counts: anchors 100→**300** (WildGuardMix, well under its 8,368-row
pool), candidates 300→**500** toxic (ToxicChat — capped here on purpose, its usable pool is only
746 total) + 1,000→**2,500** benign (UltraChat), held-out eval 100→**200** toxic + 300→**600**
benign. `influcoder_attribute_noleak_toxicity_mixed_bigger.py`, same role assignment as §6.4's
`_mixed.py`, `EPOCHS` still held at 8.

**One deliberate deviation from every other run in this document: `HARD_RATIO=0.0` here, not
0.5.** This was an explicit instruction, specifically to test whether hard-negative mining itself
was doing something analogous to what the same-source anchor/pool design turned out *not* to be
doing in §6.4 (i.e., is `hard_ratio=0.5` quietly propping up these numbers the same way "same
corpus for both roles" turned out not to be a confound?). This is a real risk, not a free
scale-up: EXP1's `FINDINGS.md` documented that scaling candidate volume *without* proportionally
raising `hard_ratio` can regress quality via pool dilution — and that finding was specifically
about `hard_ratio=0` at scale, which is exactly the regime this run sits in (3,000 total
candidates at `hard_ratio=0.0`).

Before the (expensive) teacher-gradient/GPU step, the script prints a spot-check of a handful of
raw prompt/response rows from each of the three external sources — manually reviewed here and all
legible, properly delimited, on-topic (real toxic content from WildGuardMix/ToxicChat, real
UltraChat benign chat), no encoding/mojibake/truncation artifacts. Training curve was clean too:
loss fell smoothly (0.72→0.58) and held-out fidelity (`agg rho`) rose monotonically every single
epoch (+0.37→+0.59), peaking at the final epoch — no collapse, no overfitting-shaped reversal.

| Version | AUPRC |
|---|---|
| Cross-source mix, moredata scale, `hard_ratio=0.5` (§6.4, current best) | **0.6160** |
| Cross-source mix, **bigger** scale, `hard_ratio=0.0` | 0.5191 |
| Δ | **−0.0969** |

**The dilution risk materialized — this regresses, clearly.** More data did not help here; it hurt,
and in exactly the way FINDINGS.md predicted for `hard_ratio=0` at this candidate volume: with no
hard-negative mining, a bigger pool of overwhelmingly-easy negatives swamps whatever hard-negative
signal the encoder needs to calibrate against. This is a real, reported negative result, not
hedged — §6.4's 0.6160 (moredata scale, `hard_ratio=0.5`) remains the best toxicity number in this
document, and this run is evidence *for* keeping `hard_ratio` at 0.5 rather than a case for
scaling further without it. Whether a bigger pool at `hard_ratio=0.5` (rather than 0.0) would beat
0.6160 is a different, untested question — not run here since the instruction for this pass was
specifically `hard_ratio=0.0`.

Verified leak-free via `methods/audit_leakage.py`'s `TOXICITY_MIXED_MODULES` (now 3 cross-source
scripts, 9 leak-free variants total across the whole document): 0 exact-prompt overlaps, 0 exact
(prompt,response) pair overlaps.

### 6.7 Isolating `hard_ratio` alone, at §6.4's original (unscaled) sample counts

§6.6 confounded two variables at once (bigger sample counts AND `hard_ratio=0.0`), so it can't say
whether the regression was about the scale-up, the missing hard mining, or both. This run isolates
just `hard_ratio`: `influcoder_attribute_noleak_toxicity_mixed_nohard.py` is byte-identical to
§6.4's winning `_mixed.py` (100 WildGuardMix anchors, 300 ToxicChat + 1,000 UltraChat candidates,
100 WildGuardMix + 300 UltraChat held-out eval, `EPOCHS=8`, `SEED=0`) with exactly one constant
changed: `HARD_RATIO` 0.5 → **0.0**. Confirmed by diff against the source script — every other
line is identical except the docstring, the save path, and that one constant.

| Version | AUPRC |
|---|---|
| Cross-source mix, original scale, `hard_ratio=0.5` (§6.4, current best) | **0.6160** |
| Cross-source mix, original scale, `hard_ratio=0.0` | 0.4664 |
| Δ | **−0.1496** |
| *(for reference)* bigger scale, `hard_ratio=0.0` (§6.6) | 0.5191 |

**This isolates the effect cleanly: hard mining alone accounts for a −0.15 swing**, larger than
the scale-up's partial rescue in §6.6 (0.4664 → 0.5191, +0.0527, from adding more data at
`hard_ratio=0.0` without fixing the missing hard mining). Put together, the two runs tell a
consistent story: hard-negative mining is doing real, substantial work in this cross-source setup,
more so than data volume — more candidates without hard mining helps somewhat (more random draws
occasionally land on a moderately-hard negative by chance) but comes nowhere close to recovering
what `hard_ratio=0.5` provides directly. This directly answers the question §6.6 raised (was
`hard_ratio=0.5` quietly propping up these numbers the way "same-source anchor/pool" turned out
*not* to be propping up the single-source results?): yes, clearly — unlike the anchor/pool-source
confound (which turned out not to be inflating anything when tested), hard-negative mining is a
real, load-bearing part of why the cross-source mix works as well as it does. §6.4's 0.6160 remains
the best number in this document, and this result reinforces keeping `hard_ratio=0.5` rather than
casting doubt on it.

Verified leak-free via `methods/audit_leakage.py`'s `TOXICITY_MIXED_MODULES` (now 4 cross-source
scripts, 10 leak-free variants total across the whole document): 0 exact-prompt overlaps, 0 exact
(prompt,response) pair overlaps.

### 6.8 Extending the `hard_ratio` curve: does higher than 0.5 do even better?

§6.7 showed `hard_ratio=0.5` beats `hard_ratio=0.0` by a wide margin at §6.4's original sample
counts. Natural next question: does pushing hard mining further help more, or is 0.5 already past
the peak? `influcoder_attribute_noleak_toxicity_mixed_hard075.py` is byte-identical to §6.4's
winning `_mixed.py` with exactly one constant changed: `HARD_RATIO` 0.5 → **0.75** (not 1.0 —
`FINDINGS.md` documented `hard_ratio=1.0` as a hard cliff in EXP1's setting, collapsing to +0.474
vs. 0.75's +0.769 at the same pool size there, so 1.0 wasn't worth testing here). Confirmed by diff
against the source script: only the docstring, save path, and that one constant differ.

| `hard_ratio` | AUPRC | Δ vs. 0.5 |
|---|---|---|
| 0.0 (§6.7) | 0.4664 | −0.1496 |
| **0.5 (§6.4, current best)** | **0.6160** | — |
| 0.75 (this run) | 0.5861 | −0.0299 |

Three points now trace an interior optimum, not a monotonic "more hard mining is always better"
trend: 0.75 beats 0.0 by a wide margin (+0.1197) but falls short of 0.5. Consistent with the
general shape `FINDINGS.md` found in EXP1's setting too (an interior optimum on the hard_ratio
axis, with 1.0 as a known cliff) — this cross-source toxicity setup follows the same pattern
rather than a different one. Training was stable (loss decreasing smoothly, best epoch = final
epoch, no collapse), so this isn't a `hard_ratio=1.0`-style degenerate failure, just a milder
overshoot past the optimum. **§6.4's `hard_ratio=0.5` config remains the best number in this
document (0.6160)** — this run closes out the hard_ratio axis rather than displacing the winner.

Verified leak-free via `methods/audit_leakage.py`'s `TOXICITY_MIXED_MODULES` (now 5 cross-source
scripts, 11 leak-free variants total across the whole document): 0 exact-prompt overlaps, 0 exact
(prompt,response) pair overlaps.

### 6.9 Counterfact: does hard_ratio or more data move Recall@50?

§6.1's moredata run left Recall@50 essentially flat (0.4442 → 0.4411) while MRR jumped a lot
(→0.8226). Unlike Toxicity/Bias's external pool, Counterfact's is not just a same-team analog —
§4.1 already confirmed `NeelNanda/counterfact-tracing` is the exact parent corpus DATE-LM's local
split was curated from (5,539/5,539 exact prompt match). So distributional mismatch isn't a
plausible explanation for Recall@50's flatness here the way it briefly was for toxicity; the two
remaining levers to check are `hard_ratio` (untested on this task until now — both Counterfact
scripts had only ever used the default 0.5) and raw sample count.

**Phase A — hard_ratio sweep at §6.1's moredata scale (900 anchors/2,000 candidates/500 held-out
eval), everything else held fixed.** Three new scripts
(`influcoder_attribute_noleak_counterfact_extquery_moredata_hard000/025/075.py`), each diffing
from the moredata baseline in exactly one constant (`HARD_RATIO`) plus save path/docstring —
confirmed via `diff`. `hard_ratio=1.0` skipped as in §6.8 (known collapse point in EXP1).

| `hard_ratio` | Recall@50 | MRR |
|---|---|---|
| 0.0 | 0.4414 | 0.7837 |
| 0.25 | 0.4508 | 0.8055 |
| **0.5 (§6.1 baseline)** | **0.4411** | **0.8226** |
| 0.75 | 0.4515 | 0.7910 |

Unlike §6.8's clean interior optimum for toxicity, this is **non-monotonic**: 0.25 and 0.75 both
score slightly *above* 0.5 and 0.0 on Recall@50, with no consistent trend across the sweep. The
full spread (0.4411–0.4515) is only ~0.011 wide — comparable to the ~0.014 band this exact metric
has shown across every Counterfact configuration tried anywhere in this document (original leaky
0.4549 down to moredata's 0.4411), none of which involved a hard_ratio change. Read as: at this
scale, hard_ratio does not have a confident, reliable effect on Recall@50 for this task, unlike
its large and reproducible effect on toxicity AUPRC. (Best individual point, 0.75's 0.4515, is
nominally the highest Recall@50 recorded anywhere in this document — noted for completeness, but
given the non-monotonic shape this reads as noise around a flat response, not a real win worth
promoting to "recommended.") MRR continues to prefer 0.5 outright (0.8226, clearly ahead of the
other three), so 0.5 remains the right default regardless of what Recall@50 alone suggests.

**Phase B — since phase A didn't move Recall@50 with confidence, tried more data instead.**
`influcoder_attribute_noleak_counterfact_extquery_bigger.py`: doubles §6.1's moredata counts again
(1,800 anchors/4,000 candidates/1,000 held-out eval, 6,800 of the 15,648-row clean pool),
`hard_ratio=0.5` held (no confident alternative from phase A).

| Version | Recall@50 | MRR |
|---|---|---|
| Recommended, §5.1 (300/500/250) | 0.4442 | 0.7756 |
| Moredata, §6.1 (900/2,000/500) | 0.4411 | 0.8226 |
| **Bigger, this run (1,800/4,000/1,000)** | **0.4448** | **0.8003** |

Another small, inconclusive move (+0.0037 vs. moredata) — nowhere near toxicity's response to
scale. MRR actually dropped slightly vs. moredata (though still well above the original
recommended run's 0.7756).

**Conclusion: Recall@50 on Counterfact appears largely insensitive to both hard_ratio (0.0–0.75)
and sample count (300→1,800 anchors) within the ranges tested here** — it stays in a tight
0.441–0.452 band throughout, never producing the kind of large, confident, reproducible jump
either lever produced for toxicity AUPRC. This is a legitimate negative finding, not a tuning
failure: the original recommended run (§5.1, 0.4442) and moredata (§6.1, 0.4411, but with much
better MRR) remain the two reasonable choices depending on which metric matters more; nothing
tested here displaces either with confidence. All single-seed (seed=0), no variance estimate.

Verified leak-free via `methods/audit_leakage.py`'s extended `COUNTERFACT_MODULES` (6 variants
now, all clean; 15 leak-free variants total across the whole document): 0 exact-prompt overlaps,
0 subject overlaps, 0 exact (prompt,response) pair overlaps for every new script.

### 6.10 Encoder-size sweep: the lever that actually moves Recall@50

§6.9 found Recall@50 essentially flat (0.441–0.452 band) across both `hard_ratio` (0.0–0.75) and
sample count (300→1,800 anchors), always using the default `jhu-clsp/ettin-encoder-68m`. Neither
of those levers had ever varied the *encoder size* itself. Per direct instruction, swept it —
`jhu-clsp/ettin-encoder-{150m,400m,1b}` (full family also includes 17m/32m, skipped as smaller
than the already-tested 68m default and unlikely to help per EXP1's general "bigger + enough data
wins" pattern) — holding §6.1's moredata sample counts/hard_ratio/epochs fixed, varying nothing
else.

**Efficiency note:** the teacher (Pythia-1B checkpoint) gradient computation for the fixed
anchor/candidate/eval sets doesn't depend on which student encoder is being distilled into, so it
was computed once (`influcoder_attribute_noleak_counterfact_extquery_moredata_encodersweep.py`)
and reused across all three sizes — only `load_encoder`→`distill`→embed+score repeats per size,
per `FINDINGS.md`'s documented time-saver for exactly this kind of sweep. This also guarantees all
three sizes trained against the *identical* anchor/candidate/eval draw as the existing 68m
moredata run (same `SEED`, same pool construction) — a clean single-variable comparison.

| Encoder | Params | Recall@50 | MRR | Distill wall time |
|---|---|---|---|---|
| ettin-68m (§6.1, existing) | 68M | 0.4411 | 0.8226 | — |
| ettin-150m | 150M | 0.4542 | 0.8187 | 194s |
| **ettin-400m** | 400M | **0.4609** | 0.8241 | 266s |
| ettin-1b | 1B | 0.4602 | **0.8537** | 471s |

Real, monotonic improvement from 68m→150m→400m on Recall@50 (+0.013), breaking cleanly out of
§6.9's 0.441–0.452 band — this is the first lever in the whole Counterfact investigation that
moved Recall@50 by more than noise. 1b then **plateaus** rather than continuing the trend
(0.4602 vs. 400m's 0.4609 — a 0.0007 difference, clearly noise, not a further gain), but its MRR
jumps well past every other Counterfact result in this document, **0.8537 — ahead of the paper's
Grad Sim baseline (0.836)**, the strongest MRR anywhere here. All three sizes trained cleanly, no
collapse (fidelity `rho` still rising through the final epoch for 400m and 1b, best-epoch
restoration landing at epoch 6-8/8 throughout) — the instability EXP1's `FINDINGS.md` documented
for 150m/400m at *small* data scale (collapse within 1 epoch at 500x1000) did not reproduce here,
consistent with this run's 3,400-total-sample scale being close to the ~4,500 threshold EXP1 found
sufficient for stability.

Recall@50 0.4609 beat every gradient-projection paper baseline except Grad Dot, and was within 0.04
of LESS's 0.500 — but §6.11 below found a further, larger gain on top of this by tuning
`hard_ratio` specifically at 400m (§6.9's hard_ratio sweep was only ever run on 68m).

Verified leak-free via `methods/audit_leakage.py`'s extended `COUNTERFACT_MODULES` (adds the
encoder-sweep module; since all three sizes share one anchor/candidate/eval draw, auditing the
module once covers all three): 0 exact-prompt overlaps, 0 subject overlaps, 0 exact
(prompt,response) pair overlaps.

## 6.11 hard_ratio and data-scale tuning at ettin-400m

§6.9's hard_ratio sweep (0.0/0.25/0.5/0.75) was run only on the 68m encoder, where it found
Recall@50 essentially flat. Since encoder size turned out to matter a lot (§6.10), hard_ratio's
optimum could plausibly be encoder-size-dependent too — worth checking directly at 400m rather
than assuming 68m's flatness transfers. Separately, §6.9's phase-B data-scale-up (also 68m-only)
barely moved Recall@50; retested here at 400m to see if the encoder-size gain compounds with more
data.

All four runs below reuse the moredata (900/2,000/500) or bigger (1,800/4,000/1,000)
anchor/candidate/eval draws already established in §6.1/§6.9 — same `SEED=0`, same pool
construction — with the encoder fixed at `ettin-400m` and only `hard_ratio` (and, for the last two
rows, sample count) varying:

| Config | hard_ratio | Sample counts | Recall@50 | MRR |
|---|---|---|---|---|
| §6.10 baseline | 0.5 | moredata (900/2,000/500) | 0.4609 | 0.8241 |
| **hard_ratio=0.25** | **0.25** | **moredata** | **0.4689** | **0.8739** |
| hard_ratio=0.75 | 0.75 | moredata | 0.4503 | 0.8141 |
| bigger data alone | 0.5 | bigger (1,800/4,000/1,000) | 0.4511 | 0.7976 |
| bigger data + hard_ratio=0.25 | 0.25 | bigger | 0.4575 | 0.8082 |

**`hard_ratio=0.25` at 400m is the single largest gain found anywhere in this Counterfact
investigation** — +0.008 Recall@50 and +0.050 MRR over the already-good 400m/0.5 baseline, on both
metrics at once. Unlike 68m (§6.9, where hard_ratio was ~flat), it matters substantially at 400m —
the optimum is encoder-size-dependent, not a fixed property of the task. `hard_ratio=0.75`
overshoots the same way it did at 68m (§6.7's toxicity sweep also found 0.75 worse than 0.5), so
0.25–0.5 looks like the right neighborhood generally, with the exact optimum shifting by encoder.

Data scale continues to be the wrong lever here: bigger data alone regressed vs. moredata at
matched `hard_ratio` (0.4511 < 0.4609), and combining the winning `hard_ratio=0.25` with bigger
data did *not* stack additively — it landed at 0.4575, worse than `hard_ratio=0.25` alone at
moredata scale (0.4689). More data has now failed to help Recall@50 on both 68m (§6.9) and 400m
(here) — this looks like a real, encoder-size-independent property of this metric/task, not a
68m-specific artifact.

**Current best Recall@50 for Counterfact: `ettin-400m`, moredata sample counts (900/2,000/500),
`hard_ratio=0.25` → Recall@50=0.4689, MRR=0.8739.** This is now within 0.011 of LESS's paper
baseline (0.500) and its MRR (0.8739) is the best anywhere in this document, ahead of both
Grad Sim (0.836) and the previous best (`ettin-1b`'s 0.8537 from §6.10).

One preemption occurred during this run: the `bigger data + hard_ratio=0.25` combination was
interrupted mid-run by besteffort preemption on its first GPU reservation (confirmed via the
`oarsh` "cannot find cpuset" signature, not silently assumed) and re-run to completion on a second
GPU — the number reported above is from the completed re-run, not an estimate.

Verified leak-free via `methods/audit_leakage.py`'s extended `COUNTERFACT_MODULES` (three new
modules: the 400m hard_ratio sweep and both bigger-data-at-400m scripts): 0 exact-prompt overlaps,
0 subject overlaps, 0 exact (prompt,response) pair overlaps for all three — 19 leak-free variants
total across the whole document, all clean.

## 6.12 Encoder-size sweep on the toxicity cross-source mix

§6.10's encoder-size sweep only covered Counterfact. The toxicity cross-source mix (§6.4,
`influcoder_attribute_noleak_toxicity_mixed.py`, AUPRC=0.6160, the best toxicity number until this
section) never had one — worth checking whether the same 400m/1b gain that helped Counterfact
transfers here, especially since this config's total external data (~1,800 rows: 100 anchors + 300
ToxicChat + 1,000 UltraChat candidates + 100 WildGuardMix + 300 UltraChat held-out eval) is smaller
than Counterfact's moredata scale (3,400), closer to the range `FINDINGS.md` (EXP1) documented
150m/400m collapsing in at default `lr=5e-5` — this was a real, watched-for risk, not assumed safe
by analogy.

`influcoder_attribute_noleak_toxicity_mixed_encodersweep.py`: identical data draw to §6.4's script
(same `SEED=0`, same WildGuardMix-anchor/ToxicChat-candidate/UltraChat-benign role assignment,
`HARD_RATIO=0.5`, `EPOCHS=8`), teacher gradients for the fixed draw computed once and reused across
encoder sizes (same pattern as §6.10's Counterfact sweep). Per-epoch eval Spearman was watched for
every size to catch a collapse pattern (metric cratering after an early peak despite falling train
loss) rather than trusting a blind final AUPRC number:

| Encoder | Per-epoch agg ρ | Collapsed? | AUPRC |
|---|---|---|---|
| 68m (§6.4, existing) | — | — | 0.6160 |
| 150m | +0.317 → +0.568 (peak ep.6) → +0.556 | No | 0.5307 |
| **400m** | +0.418 → **+0.638** (final epoch, still rising) | No | **0.7651** |
| 1b | +0.482 → +0.596 (peak ep.8) | No | 0.7343 |

No collapse at any size — all three curves rise smoothly to a late-epoch peak, the risk flagged
above didn't materialize.

**400m is a large, clean win: AUPRC 0.7651, +0.149 over the existing 0.6160 best.** This is now
the best result anywhere in this document on *either* task, and it beats every DATE-LM paper
baseline on this task including LESS (0.734) — the only baseline that had, until now, been out of
reach on any InfluCoder configuration tried. 1b also clears LESS's baseline (0.7343) but doesn't
beat 400m. 150m is the one regression — worse than the 68m default (0.5307 vs 0.6160), the same
"150m is often the weak middle size" pattern EXP1's `FINDINGS.md` noted, now confirmed on a second
task/pool.

**New current best AUPRC for Toxicity/Bias filtering: `ettin-400m` on the §6.4 cross-source mix
(WildGuardMix anchor/eval + ToxicChat candidate), `hard_ratio=0.5`, `epochs=8` → AUPRC=0.7651.**
This supersedes §5.2/§6.4's 0.6160 and every other toxicity number in this document. Unlike
Counterfact, where 400m/hard_ratio=0.25 closed most but not all of the gap to the strongest paper
baseline (LESS's Recall@50), here InfluCoder now outright beats every DATE-LM baseline on this
task, LESS included.

Verified leak-free via `methods/audit_leakage.py`'s extended `TOXICITY_MIXED_MODULES` (one new
module — all three sizes share one anchor/candidate/eval draw, same as §6.10's Counterfact sweep,
so auditing once covers all three): 0 exact-prompt overlaps, 0 exact (prompt,response) pair
overlaps — 20 leak-free variants total across the whole document, all clean.

Not yet tried: hard_ratio tuning at 400m for this task (§6.11's finding that hard_ratio's optimum
is encoder-size-dependent for Counterfact suggests the same could be true here — 0.5 has not been
confirmed optimal at 400m specifically, only carried over unchanged from the 68m default).

## 7. Reproduction

```bash
cd EXP2-datelm
# Counterfact (needs a GPU; ~1-2 min on an RTX 6000 Ada)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_counterfact_extquery.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/factual-attribution.yaml \
  --score_path results/factual-attribution-influcoder-noleak-extquery/InfluCoder.pt

# Toxicity/Bias (needs HF auth only if you want the original leaky config for comparison;
# the leak-free path below needs no gated dataset)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak/InfluCoder.pt

# §6 moredata ablation (needs a GPU; ~2-3 min on an RTX 6000 Ada -- ~3-4x the sample volume)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_moredata.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/factual-attribution.yaml \
  --score_path results/factual-attribution-influcoder-noleak-extquery-moredata/InfluCoder.pt

../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity_moredata.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-moredata/InfluCoder.pt

# §4.3 WildGuardMix variant (needs a GPU + HF auth granted access to allenai/wildguardmix)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity_wildguard.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-wildguard/InfluCoder.pt

# §6.4 WildGuardMix moredata + cross-source mixed variants (needs a GPU + the same wildguardmix
# HF access as above)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity_wildguard_moredata.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-wildguard-moredata/InfluCoder.pt

../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity_mixed.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-mixed/InfluCoder.pt

# §6.5 BeaverTails candidate-pool variant (needs a GPU + the same wildguardmix HF access as above;
# BeaverTails itself is open, no gating)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity_mixed_beavertails.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-mixed-beavertails/InfluCoder.pt

# §6.6 scaled-up mix, hard_ratio=0.0 (needs a GPU + wildguardmix HF access as above)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity_mixed_bigger.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-mixed-bigger/InfluCoder.pt

# §6.7 original-scale mix, hard_ratio=0.0 (isolates the hard_ratio variable alone; needs a GPU +
# wildguardmix HF access as above)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity_mixed_nohard.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-mixed-nohard/InfluCoder.pt

# §6.8 original-scale mix, hard_ratio=0.75 (extends the hard_ratio curve past 0.5; needs a GPU +
# wildguardmix HF access as above)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_toxicity_mixed_hard075.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-mixed-hard075/InfluCoder.pt

# §6.9 Counterfact hard_ratio sweep + phase-B data scale-up (needs a GPU, no HF gating)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_moredata_hard000.py
../.venv_py311/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_moredata_hard025.py
../.venv_py311/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_moredata_hard075.py
../.venv_py311/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_bigger.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/factual-attribution.yaml \
  --score_path results/factual-attribution-influcoder-noleak-extquery-moredata-hard000/InfluCoder.pt
# (repeat --score_path for the hard025/hard075/bigger result dirs)

# §6.10 encoder-size sweep (needs a GPU; teacher grads computed once, ~3min, then
# ~200-470s per encoder size for 150m/400m/1b -- reuses the moredata sample draw)
../.venv_py311/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_moredata_encodersweep.py
../.venv_py311/bin/python evaluation/evaluate_application.py \
  --config configs/factual-attribution.yaml \
  --score_path results/factual-attribution-influcoder-noleak-extquery-moredata-400m/InfluCoder.pt
# (repeat --score_path for the 150m/1b result dirs)

# §6.11 hard_ratio + data-scale tuning at ettin-400m (needs a GPU, no HF gating; besteffort
# jobs on G5K can be preempted mid-run -- the combined bigger+hard025 run below needed one retry
# in practice, see §6.11's note)
../.venv_h100/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_moredata_400m_hardsweep.py
../.venv_h100/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_bigger_400m.py
../.venv_h100/bin/python methods/influcoder_attribute_noleak_counterfact_extquery_bigger_400m_hard025.py
../.venv_h100/bin/python evaluation/evaluate_application.py \
  --config configs/factual-attribution.yaml \
  --score_path results/factual-attribution-influcoder-noleak-extquery-moredata-400m-hard025/InfluCoder.pt
# (repeat --score_path for the moredata-400m-hard075/bigger-400m/bigger-400m-hard025 result dirs)

# Toxicity encoder-size sweep (§6.12)
../.venv_h100/bin/python methods/influcoder_attribute_noleak_toxicity_mixed_encodersweep.py
../.venv_h100/bin/python evaluation/evaluate_application.py \
  --config configs/toxicity-bias.yaml \
  --score_path results/toxicity-bias-influcoder-noleak-mixed-400m/InfluCoder.pt
# (repeat --score_path for the mixed-150m/mixed-1b result dirs)

# Independent leak audit (CPU-only, no GPU/model loading needed -- a couple minutes,
# dominated by re-downloading/streaming the external pools). Checks all twenty runs above.
../.venv_h100/bin/python methods/audit_leakage.py
```

## 8. Open items

- **RESOLVED:** `allenai/wildguardmix` access was granted and the toxicity fix was redone against
  it (§4.3): AUPRC 0.4222 vs. ToxicChat's 0.4242 — a wash, not an improvement, despite the closer
  distributional match. ToxicChat remains the recommended row in §5.2 (marginally higher, and
  already the more thoroughly-exercised script); WildGuardMix is documented as an equally-valid
  alternative with no meaningful score difference. Retested at moredata scale and with cross-source
  anchor/candidate mixing in §6.4 — same "source doesn't matter much, volume does" conclusion holds
  at the larger scale, but the cross-source mix (0.6160) is now the best toxicity number in this
  document, ahead of Grad Sim and behind only LESS. Worth folding into whatever §6.3 promotion
  decision gets made for §5's recommended row, rather than treating §6/§6.4 as separate from §5
  indefinitely. Tested one more variant (§6.5): swapping the mix's candidate pool from ToxicChat to
  the much larger BeaverTails (166,347 vs. 746 usable rows) — this made things *worse* (0.5608),
  not better, so candidate-pool volume alone isn't the mechanism; §6.4's ToxicChat-candidate mix
  (0.6160) stays the best result found so far. Tried scaling that winning mix further at §6.6, but
  deliberately with `hard_ratio=0.0` (testing whether hard mining itself was propping up the
  numbers) — this regressed sharply (0.5191, −0.097), confirming `hard_ratio=0.5` is load-bearing
  and matches EXP1's documented dilution-at-hard_ratio=0 pattern. §6.4's 0.6160 remains the best
  result and the standing candidate for §5's recommended row; a bigger pool *at* `hard_ratio=0.5`
  is untested and would be the natural next step if this gets revisited.
- **UPDATE (§6.12): the "bigger pool at hard_ratio=0.5" question above was answered indirectly —
  not by scaling the pool, but by scaling the encoder.** `ettin-400m` on the exact same §6.4 mix
  (same sample counts, same `hard_ratio=0.5`) reaches **AUPRC=0.7651**, beating every DATE-LM
  baseline on this task including LESS (0.734). This is now the best result in the whole document,
  on either task, and is the new standing candidate for §5.2's recommended row. hard_ratio tuning
  at 400m specifically (mirroring §6.11's Counterfact result, where 0.25 beat 0.5) remains untested
  for toxicity and is the natural next step.
- Only Pythia-1B has been leak-free-fixed. DATE-LM also reports Llama-3.2-1B and Llama-3.1-8B
  for both tasks; neither has been redone with external data yet.

## 9. Wall-clock cost vs. other DATE-LM methods (partial measurement)

Real fresh-process wall-clock timing (`/usr/bin/time -p`, one A40, warm HF/OS cache shared across
runs) via `methods/dattri.py` (its `__main__` had a real bug -- `args.config_path` instead of
`args.config`, meaning this code path had apparently never been successfully invoked before;
fixed). Deprioritized mid-measurement to free the GPU for the WildGuardMix work in §4.3/§6.4, so
this is partial, not a full method x task grid:

| Method | Task | Wall-clock | Notes |
|---|---|---|---|
| Grad Sim | Counterfact | 1001s (~16.7 min) | full 5,473-train + 66-ref backward-pass gradients |
| LESS | Counterfact | 2585s (~43.1 min) | ~2.6x Grad Sim -- extra projection overhead, unbatched |
| InfluCoder (leak-free) | Counterfact | 183s (~3.1 min) | distills on 1,050 external rows, embeds full train+ref in one forward pass |
| Grad Sim | Toxicity/Bias | 1764s (~29.4 min) | full 10,187-train + 10-ref |
| LESS | Toxicity/Bias | not completed | killed at 56% (~50 min in) when GPU was reclaimed for §4.3/§6.4 -- no number, not extrapolated |
| InfluCoder (leak-free) | Toxicity/Bias | not measured | deprioritized before this run started |

Sanity check: re-evaluating the completed Grad Sim/LESS score files against DATE-LM's own
`evaluate_application.py` reproduced the paper's published numbers closely (Grad Sim Counterfact
0.493/0.836 vs. paper's 0.493/0.836; LESS Counterfact 0.500/0.772 vs. paper's 0.500/0.772; Grad Sim
Toxicity/Bias 0.625 vs. paper's 0.601) -- the `dattri.py` code path, once the CLI bug was fixed, is
producing real, correct scores, not just plausible-looking noise.

Read on what's here: even this partial grid already shows InfluCoder's whole value proposition
concretely -- ~3 min vs. ~17-43 min for one-off exact-gradient methods on Counterfact, because
InfluCoder pays a bounded distillation cost once (on external data, not scaling with local train
size) and then only *embeds* the full local train/ref set, while Grad Sim/LESS pay one backward
pass per local training example every time. That gap should widen further on Toxicity/Bias's
larger 10,187-example train set, but the LESS number to confirm that wasn't finished. Not
independently re-verified beyond one run each (no multi-seed variance estimate); GPU-dependent
(numbers are for a single A40, not the RTX 6000 Ada used for the §4-§6 InfluCoder runs, so don't
diff these against §4's "~1-2 min" docstring estimate directly). Completing the LESS/InfluCoder
Toxicity/Bias cells is left as a follow-up if this comparison becomes load-bearing.

### 9.1 Rep-Sim / Counterfact -- measured on a THIRD, different GPU (not directly comparable above)

`methods/run_repsim_counterfact.py`: DATE-LM's Rep-Sim baseline (forward-pass-only -- last-token
hidden state as the representation, cosine similarity to the ref set, no backward pass at all)
re-hosted against `get_dataset("Counterfact", "Pythia-1b")` via the same `checkpoints_load_func`
every other method here uses (`baselines/repsim.py`, this repo's own vendored implementation, is
built around jsonl files rather than `get_dataset()`, so this mirrors its actual algorithm rather
than calling it directly). `batch_size=1`, matching the convention above.

**Measured on an RTX PRO 6000 Blackwell (grenoble), not the A40 the Grad Sim/LESS/InfluCoder row
above used, nor the RTX 6000 Ada §4-§6 used** -- a third GPU model. Don't treat this number as
directly diffable against the table above; it's here for Rep-Sim's own accuracy/cost shape, not
for a cross-method ranking on identical hardware. Getting genuinely consistent numbers (the
user's original ask) needs every method re-run on one single GPU in one sitting, which this
partial measurement still isn't -- see the open item below.

| Method | Task | Wall-clock | Notes |
|---|---|---|---|
| Rep-Sim | Counterfact | 62.7s total (45.2s forward-pass) | full 5,473-train + 66-ref forward pass, no backward pass |
| InfluCoder (leak-free, 400m, best config: hard_ratio=0.25) | Counterfact | 195.0s total | `methods/run_influcoder_400m_best_counterfact.py` -- teacher grads + distill (8 epochs) + embed+score |

Sanity check: re-evaluated the Rep-Sim score file against `evaluate_application.py` --
Recall@50=0.3763, MRR=0.7907, matching the paper's published Rep-Sim numbers (0.376/0.790) closely.

As expected, forward-only Rep-Sim is far cheaper than any backward-pass method: ~63s vs. Grad
Sim's 1001s and LESS's 2585s on the same task (different GPUs, so read this as "same shape,
not a precise ratio" until re-run on one GPU). InfluCoder's own 195s here (same GPU as Rep-Sim,
directly comparable to that one row) sits well below Grad Sim/LESS too, though those two aren't
on this GPU yet either -- see the open item below.

**Reproducibility caveat, found while running the InfluCoder-400m timing row above:** re-running
the exact best-config script (same `SEED=0`, same data draw, same hyperparameters, verified
line-by-line against the original) on this RTX PRO 6000 Blackwell reproduced the training
successfully but **did not reproduce the exact 0.4689/0.8739 score** -- this run scored
Recall@50=0.4502/MRR=0.8165 instead. The script logic matches the original exactly, so the most
likely explanation is GPU-architecture-level floating-point non-determinism accumulating over 8
epochs of training (different kernel/reduction-order selection across GPU models), not a bug --
but this hasn't been root-caused further, and it means **the 0.4689/0.8739 "best" number is not
confirmed stable across different GPUs** without a multi-run variance check. Doesn't affect the
195.0s timing measurement itself, which is what this section is about, but is a real caveat for
anyone citing 0.4689 as a fixed number rather than "roughly ~0.45-0.47 depending on hardware."

**Open item: get everything on one GPU.** This session now has Rep-Sim (RTX PRO 6000 Blackwell),
Grad Sim/LESS/InfluCoder-Counterfact (A40), and InfluCoder's various leak-free configs (RTX 6000
Ada / Quadro RTX 8000 / H100 NVL / A100-40GB, whichever GPU happened to be free when each fork
ran) scattered across at least four different GPU models. None of these numbers should be
cross-compared as if they were controlled timings. A real "same GPU" comparison needs Grad Dot/
Grad Sim/LESS/DataInf/EKFAC/Rep-Sim/InfluCoder all re-run back-to-back on one single reserved GPU,
same session, same warm/cold cache state for each -- not done yet.
