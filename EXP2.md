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

# Independent leak audit (CPU-only, no GPU/model loading needed -- a couple minutes,
# dominated by re-downloading/streaming the external pools). Checks all five runs above.
../.venv_h100/bin/python methods/audit_leakage.py
```

## 8. Open items

- **RESOLVED:** `allenai/wildguardmix` access was granted and the toxicity fix was redone against
  it (§4.3): AUPRC 0.4222 vs. ToxicChat's 0.4242 — a wash, not an improvement, despite the closer
  distributional match. ToxicChat remains the recommended row in §5.2 (marginally higher, and
  already the more thoroughly-exercised script); WildGuardMix is documented as an equally-valid
  alternative with no meaningful score difference.
- Only Pythia-1B has been leak-free-fixed. DATE-LM also reports Llama-3.2-1B and Llama-3.1-8B
  for both tasks; neither has been redone with external data yet.
