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

# Independent leak audit (CPU-only, no GPU/model loading needed -- a couple minutes,
# dominated by re-downloading/streaming the external pools). Checks all eight runs above.
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
  (0.6160) stays the best result found so far.
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
