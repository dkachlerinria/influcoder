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

Verified via independent audit (`sample: audit_leakage.py`, reconstructs the exact rows a
completed run used and cross-checks them against local data): 0 exact-prompt overlaps, 0
subject overlaps, 0 exact (prompt,response) pair overlaps.

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
to build query-analogs or positive candidates. (Checked and ruled out: `allenai/wildguardmix`,
the same WildGuard team's larger 37,976-row prompt+response set, would have been the closest
format/distribution match, but access is gated and not granted to this account; going back to
raw `walledai/XSTest`/`Paul/XSTest` doesn't help either — its "unsafe" label pool is 200 total,
197 already consumed, same 3-row remainder problem.)

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
| **Leak-free (ToxicChat-sourced external)** | **0.4242** |

Paper baselines, Pythia-1B, XSTest-response column (Table 12, Heterogeneous):

| Method | AUPRC |
|---|---|
| Grad Dot | 0.389 |
| DataInf | 0.392 |
| EKFAC | 0.344 |
| Rep-Sim | 0.580 |
| Grad Sim | 0.601 |
| LESS | 0.734 |
| **InfluCoder (leak-free)** | **0.424** |

Same shape as Counterfact: ahead of Grad Dot/DataInf/EKFAC, behind Rep-Sim/Grad Sim/LESS.
Remember the §4.2 caveat — this used a different-source external pool (ToxicChat, not
XSTest-response), so it's a slightly less exact distributional match than the Counterfact fix.

## 6. Reproduction

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
```

## 7. Open items

- Toxicity/Bias's external pool (ToxicChat) is a weaker distributional match than Counterfact's
  (different source dataset, not just disjoint rows of the same one). If `allenai/wildguardmix`
  access is ever granted, redoing the toxicity fix against that source instead would close this
  gap — same team/methodology as `xstest-response`, ~85x more prompt+response rows.
- Only Pythia-1B has been leak-free-fixed. DATE-LM also reports Llama-3.2-1B and Llama-3.1-8B
  for both tasks; neither has been redone with external data yet.
