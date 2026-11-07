# REFURB: restructuring of the InfluCoder repo

Branch `refurb` on the private `review` remote (nothing pushed to `origin`). Base: `d84c4ac`. LM28, 2026-11-07.

## 1. Template

The README follows **[AirRep](https://github.com/sunnweiwei/AirRep)**, as the PI asked. It has a badge row, a
one-paragraph summary, **Paper/Authors** lines, a teaser figure, and a Quick Start with install, a one-line run and a
Python scoring snippet. Then come the EXP1 pipeline as numbered steps (`### 1) ... 4)`), each with its command and an
"Output format" JSON block, a short second-experiment section (DATE-LM), the layout, and a citation.

**[LESS](https://github.com/princeton-nlp/LESS)** guided the repo layout: one package with the method, baselines next to
it, one script per experiment, and no internal notes at the top.

## 2. Layout

```
README.md  requirements.txt  run.py
influcoder/          the method (unchanged except one data path)
baselines/           LESS, LoGRA, RDS+, semantic, TF-IDF, random + shared harness
experiments/exp1/    EXP1 / Figure 1, Parts 1-3, collect/plot, alpha ablation, storage   (was baselines/exp1/)
experiments/datelm/  DATE-LM integration                                                (was EXP2-datelm/)
scripts/             quickstart.sh, exp1_figure1.sh, exp1_ablations.sh, datelm.sh        (new, thin wrappers)
data/                eval/bbh/, dolly/                                                  (dolly/ was top level)
results/             EXP1 seeds 0-2, alpha ablation, storage                            (was baselines/out/)
```

Top level went from 22 entries (13 internal .md files, loose scripts, launchers) to 9, plus this REFURB.md. The tracked tree went from
625 to 428 files.

## 3. Old -> new

| old | new |
|---|---|
| `baselines/exp1/*` (module `baselines.exp1`) | `experiments/exp1/*` (module `experiments.exp1`) |
| `baselines/out/fig1_dolci/biggpu/seed{0,1,2}/`, `.../default/alpha_ablation.json`, `addendum_storage_{1k,n100}.json` | same names under `results/` |
| `EXP2-datelm/*` | `experiments/datelm/*` |
| `dolly/dolly_data.jsonl` | `data/dolly/dolly_data.jsonl` |

Command changes: `python -m baselines.exp1.partN` -> `python -m experiments.exp1.partN`; `cd EXP2-datelm` ->
`cd experiments/datelm`.

**Removed from the branch.** All of these are still in git history at `d84c4ac`, nothing is lost. They are listed so
you can put any back.

| removed | why |
|---|---|
| 12 internal notes (`G5K*.md`, `COORDINATION.md`, `CONTINUE.md`, `TODO.md`, `GOALS.md`, `EXP1*.md`, `EXP2.md`, `FINDINGS.md`, `INFLUCODER_FINDINGS.md`) + 2 launchers | lab notebooks and cluster notes with stale numbers |
| Dolly-pilot scripts (`figure1_table`, `plot_*`, `rank_sweep`, `logra_proxy_rank_sweep`, `train_fig1_encoders`, `update_proxy_rows*`, `eval_ood`, `finetune_ood`) and their results (`fig1/`, `paper200*/`, `big/`) | earlier Dolly-pool setting, not EXP1 |
| diagnostics (`tune_influcoder`, `ensemble_influcoder`, `time_logra_*`, `verify_logra_sdpa`) | not behind any paper number |
| `results/fig1_dolci/*` except `biggpu/seed{0,1,2}` and `default/alpha_ablation.json` (drafts, 50x50 tests, archives, vercors18 runs, `512cap_buggy`, `qwen4b_tuning`, superseded `addendum_storage.json`) | superseded |
| DATE-LM: 35 exploration score dirs (`factual-attribution-*`, `toxicity-bias-*`, 148 MB), `timing_summary.json`, 22 `influcoder_attribute_*` variants, `run_influcoder_400m_best_counterfact`, `run_repsim_counterfact`, `run_timed_baseline`, `time_*_subset`, `audit_leakage`, `exp_*`, `bench_embed` | one-off variants. Kept `influcoder_attribute.py`, because `run-application.sh` calls it. |

## 4. Functionality unchanged: evidence

The harness is in `_lm28/golden/` (untracked). It checks clean `git archive` snapshots of old and new commits.

- **Static check, every file.** Of the 428 kept files, 398 are byte-identical and 19 differ only by the declared
  module/output-path substitutions (`all_subs.py`, e.g. `baselines.exp1` -> `experiments.exp1`, `baselines/out` ->
  `results`). The rest are new docs or scripts, or doc-only edits. There is exactly **one changed code line**:
  `experiments/datelm/methods/influcoder/_bootstrap.py`, `parents[3]` -> `parents[4]`, because the directory moved one
  level deeper. It still resolves to the repo root, which I checked.
- **No dangling imports.** Every kept module of `influcoder`, `baselines`, `experiments.exp1` and `run` imports
  cleanly. All 203 .py files compile.
- **Regenerated outputs are byte-identical, old vs final (`d84c4ac` vs `ee93636`):**
  - EXP1 `collect_results` + `plot_figure1` for seeds 0/1/2: `EXP1_results.json`, `figure1.png`, `figure1.pdf`.
    9/9 files identical.
  - `make_final_jsons.py` -> 3 DATE-LM final JSONs: identical.
  - Re-scoring all committed DATE-LM score files (at `ada7c81`): identical.
- **GPU, same card (2080 Ti), `ada7c81` vs `d84c4ac`:**
  - `run.py --preset sanity` and four baselines: identical.
  - `run.py --preset tiny`: gradients, untrained baseline and first-epoch loss identical. The trained metric is within
    the old-vs-old run-to-run spread, which comes from the non-deterministic SDPA backward the repo documents.
  - The code these touch has not changed since `ada7c81` except comments.

## 5. Reproduction (earlier tonight, before the PI said no reruns)

| result | paper / committed | reproduced | notes |
|---|---|---|---|
| EXP1 Part 1, seed 0, InfluCoder-68m Spearman | 0.767 | **0.778** (150m: 0.783 vs 0.770) | original code, RTX PRO 6000, 85 min. LESS-4B 0.972 vs 0.977, LoGRA-4B 0.925 vs 0.928. TF-IDF moved -0.093 -> -0.061, so the eval data or ground truth differs slightly; likely suspect is the HF dataset revision. Pinning the dataset revision and library versions is advisable. |
| DATE-LM Counterfact, R@50 / MRR | 0.460 / 0.830 | 0.457 / 0.874, 0.452 / 0.829, 0.456 / 0.826 | 3 runs on RTX PRO 6000 (original code once, refurb code twice) |
| DATE-LM toxicity Het / Hom AUPRC | 0.646 / 0.714 | blocked | gated HF dataset `allenai/wildguardmix`: needs the PI's token |

Queued equivalence runs (EXP1 seeds with the new code, A40 Counterfact) were cancelled when the PI said reruns are not
needed.

**Provenance gap (relevant before any data release).** The committed DATE-LM score files for InfluCoder (all 3 tasks)
and Counterfact BM25 do not reproduce their JSON numbers when re-scored (e.g. Het InfluCoder 0.537 vs 0.646 reported).
Git history shows the score files are from before the retune (`740f9c3`). The retuned numbers were committed later
(`77f2925`) without their score files. Every other method's score file matches its JSON exactly.

## 6. Open, for the PI

- README TODOs: paper link, author list (badge + BibTeX), and a LICENSE (none exists; DATE-LM's upstream license
  should be vendored too).
- `requirements.txt` says `transformers>=4.48`, but EXP1 needs `>=4.51` (Qwen3) plus `traker`. The README tells the
  user; I left the file unchanged to stay behaviour-neutral.
- Optional further trims: the vendored DATE-LM pre-training/fine-tuning pipeline we never run (`train/`,
  `minimal_multitask/`, `data_processing/`, `methods/{mates,edu_classifier}`), and the 58 MB of DATE-LM score files,
  which could move to a data release.
- Publish from a fresh squashed history: the current history is 134 MB and has cluster hostnames in author emails.
