# REFURB: first-shot restructuring of the InfluCoder repo

Branch `refurb` (private review remote `review`; nothing pushed to `origin`). Base: `d84c4ac` (current public `main` + local EXP1 addendum commits).
Author: LM28, 2026-11-07. Status: first shot for the PI's critique. **EXPLORATORY, NOT REVIEWED.**

## 1. Template and why

**Template: [princeton-nlp/LESS](https://github.com/princeton-nlp/LESS)** (Xia et al., ICML 2024). It is the most-used
gradient-based data-selection repo, and InfluCoder's closest relative: InfluCoder distills exactly the LESS/LoGRA kind of
gradient-influence signal, and LESS is one of our baselines. Readers in this area know its shape:

- a short README: paper link, install, data, a step-by-step "how to reproduce", citation;
- one importable package with the method;
- one shell script per experiment step (`less/scripts/...`);
- no internal notes at the top level.

DATE-LM's repo was the alternative. It is a benchmark rather than a method repo, so I kept it as a vendored
subproject (`experiments/datelm/`) and did not use it as the model for the whole repo.

## 2. New layout

```
README.md              LESS-style: paper, install, data, quickstart, per-result reproduction table, layout, citation
requirements.txt       unchanged
run.py                 unchanged: end-to-end InfluCoder pipeline (BBH x Dolly presets)
influcoder/            the method (unchanged except one data path)
baselines/             competitor scorers only: logra/ less/ rdsplus/ semantic/ tfidf/ random/ influcoder/ + common.py,
                       cost.py, eval_baselines.py, scaling_sweep.py (kept here: experiments.agreement.data imports it)
experiments/
  agreement/           EXP1 = Figure 1 (Parts 1-3), alpha ablation, storage addendum      (was baselines/exp1/)
  datelm/              DATE-LM integration, Counterfact + toxicity                       (was EXP2-datelm/)
  dolly_pilot/         earlier Dolly-pool experiments: rank sweep, scaling, OOD           (was baselines/*.py, *_ood.py)
  diagnostics/         timing/tuning helpers, not behind any paper number                 (was baselines/*.py)
scripts/               quickstart.sh, agreement_fig1.sh, agreement_ablations.sh, datelm.sh  (new, thin wrappers)
data/                  eval/bbh/ (unchanged), dolly/                                       (was dolly/)
results/               every committed agreement result                                   (was baselines/out/)
internal/              notes + one-off launchers, untouched, pending your decision (section 5)
```

Top level before: 13 internal .md files, 2 loose scripts, 2 launchers, `dolly/`, `EXP2-datelm/`, and so on (22 entries).
Top level after: `README.md requirements.txt run.py influcoder/ baselines/ experiments/ scripts/ data/ results/ internal/`.
`internal/` goes away once you decide on section 5.

Commits (each one small and checked):

| commit | step |
|---|---|
| `bf7dc72` | 1: notes and launchers -> `internal/` (no code touched) |
| `261b3c6` | 2: `EXP2-datelm/` -> `datelm/` |
| `49f35fa` | 3: `dolly/` -> `data/dolly/` |
| `ada7c81` | 4: `baselines/exp1` -> `experiments/agreement`; pilot and diagnostic scripts out of `baselines/`; `baselines/out` -> `results/` |
| `a156b3a` | 5: `datelm/` -> `experiments/datelm/` |
| `566a529` | 6: README + `scripts/` |

## 3. Old -> new mapping

The full machine-readable map is in `_lm28/golden/mapping.tsv` (untracked; I can commit it if useful).

| old | new |
|---|---|
| `baselines/exp1/*` | `experiments/agreement/*` (module `baselines.exp1` -> `experiments.agreement`) |
| `baselines/out/*` | `results/*` |
| `EXP2-datelm/*` | `experiments/datelm/*` |
| `dolly/dolly_data.jsonl` | `data/dolly/dolly_data.jsonl` |
| `baselines/{figure1_table,plot_figure1,plot_cost_quality,plot_paper200,plot_scaling,rank_sweep,logra_proxy_rank_sweep,train_fig1_encoders,update_proxy_rows,update_proxy_rows_batched}.py` | `experiments/dolly_pilot/` |
| `eval_ood.py`, `finetune_ood.py` | `experiments/dolly_pilot/` (now run as `python -m experiments.dolly_pilot.eval_ood`) |
| `baselines/{tune_influcoder,ensemble_influcoder,time_logra_batch,time_logra_rank,verify_logra_sdpa}.py` | `experiments/diagnostics/` |
| `CONTINUE.md COORDINATION.md EXP1*.md EXP2.md FINDINGS.md G5K*.md GOALS.md INFLUCODER_FINDINGS.md TODO.md` | `internal/notes/` |
| `launcher_alpha.sh`, `EXP2-datelm/_baseline_launcher.sh` | `internal/launchers/` |

Command changes for users: `python -m baselines.exp1.partN` -> `python -m experiments.agreement.partN`. Outputs move
from `baselines/out/...` to `results/...`. `cd EXP2-datelm` becomes `cd experiments/datelm`.

## 4. Equivalence evidence (behaviour unchanged)

Harness: `_lm28/golden/` (untracked). It runs each entry point on a clean `git archive` snapshot of a commit and diffs
the outputs, ignoring only timing, host and path fields.

**4.1 Static: every changed byte is accounted for.** `verify_step.py d84c4ac 566a529` checks every tracked file against
the old->new map:

- 587 files are byte-identical.
- 37 files equal `subs(old)`, where `subs` is the declared list of module and output path substitutions (`all_subs.py`).
- 4 files are new: empty `__init__.py` files and `internal/README.md`.
- 0 old files are missing.
- The only remaining edits are docstrings, `.gitignore` paths, and one real code line:
  `experiments/datelm/methods/influcoder/_bootstrap.py`: `parents[3]` -> `parents[4]`, because the directory moved one
  level deeper. It still resolves to the repo root, which I checked.
- All 253 .py files compile.

**4.2 Regenerated artefacts are byte-identical** (CPU, old `d84c4ac` vs new `ada7c81`):

| entry point | outputs | result |
|---|---|---|
| `experiments.agreement.collect_results` + `plot_figure1`, `EXP1_CONFIG=biggpu`, seeds 0/1/2 | `EXP1_results.json`, `figure1.{png,pdf}` x3 | **identical** (9/9 files) |
| Dolly-pilot plots (`plot_paper200`, `plot_cost_quality`, `plot_scaling`, `plot_figure1`) | 8 png/pdf | **identical** |
| `make_final_jsons.py` | 3 `exp2-*-final.json` | **identical** |
| re-scoring all 61 committed DATE-LM score files with DATE-LM's own evaluator | Recall@50 / MRR / AUPRC | **identical** |

Side check: re-running `collect_results` on the original code reproduces the committed `EXP1_results.json` exactly
(all 443 numeric fields). Only `generated_at` and the source paths differ.

**4.3 GPU runs, same card (2080 Ti, nancy graffiti), 2 reps each of old and new:**

| run | old vs new |
|---|---|
| `run.py --preset sanity` | identical metrics. Losses differ about 1e-7, the same as old vs old. |
| `baselines.eval_baselines --method tfidf random semantic rdsplus --preset sanity` | identical |
| `run.py --preset tiny` (8 epochs) | gradients, untrained baseline and epoch-1 loss identical. Trained per-anchor rho: old 0.336 / 0.317, new 0.338 / 0.343 (agg 0.351 / 0.352 vs 0.350 / 0.349). Within the run-to-run spread of the same code, which comes from the non-deterministic SDPA backward documented in the repo. |

Counterfact InfluCoder, old vs new code on the same card class: see section 6.

## 5. Bloat: proposals (nothing deleted; your call)

| item | size | proposal |
|---|---|---|
| `internal/notes/G5K.md`, `G5K-CPU-AGENT.md`, `COORDINATION.md`, `CONTINUE.md`, `TODO.md`, `GOALS.md` | 60 KB | **drop** (cluster and agent operations; they mention internal infrastructure) |
| `internal/notes/EXP1.md`, `EXP2.md`, `FINDINGS.md`, `INFLUCODER_FINDINGS.md`, `EXP1_BIGGPU_FINAL.md` | 230 KB | **drop** from the release. They are lab notebooks with stale numbers: EXP1.md quotes the 50x50 runs, and EXP1_BIGGPU_FINAL says "not yet executed". Anything a reader needs is in the README and configs. |
| `internal/notes/EXP1-ADDENDUM-STORAGE.md` | 8 KB | **keep** as `docs/storage.md` if the appendix cites it, otherwise drop |
| `internal/launchers/*` | <1 KB | **drop** (hard-coded old paths and venvs; replaced by `scripts/`) |
| `experiments/datelm/results/factual-attribution-*` and `toxicity-bias-*` (35 dirs of exploration score files) | 148 MB | **drop** (pre-final InfluCoder variants; no paper number uses them) |
| `experiments/datelm/results/fig2-*/*.pt` (score files behind the DATE-LM table) | 58 MB | **move to a data release** (HF dataset or GitHub release asset), keep the `fig2_*.json` and `exp2-*-final.json` in git. But see 6.3: the InfluCoder and Counterfact-BM25 files there are *not* the ones behind the reported numbers. |
| `experiments/datelm/results_influcoder_sweep/summary.json` | 10 KB | keep (evidence for the chosen lr / hard_ratio) |
| `experiments/datelm/{assets,checkpoints,tokenizer}` (upstream logo pngs, tokenizer configs) | 11 MB | **drop** `assets/` (logos). Keep `checkpoints/` and `tokenizer/` only if the upstream litgpt path needs them; ours does not. |
| `experiments/datelm/{train,minimal_multitask,data_processing,dattri/benchmark}`, `methods/{mates,edu_classifier,bm25/*_litgpt*}` (upstream pre-training / fine-tuning pipeline we never run) | 0.6 MB | **drop**, or keep upstream byte-identical. Recommendation: drop and point to upstream with a commit hash. |
| `experiments/datelm/methods/influcoder_attribute*.py` (23 variants), `run_influcoder_400m_best_counterfact.py`, `run_repsim_counterfact.py`, `run_timed_baseline.py`, `time_*_subset.py`, `audit_leakage.py`, `exp_*.py`, `bench_embed.py` | 0.3 MB | **drop** the variants. Keep `audit_leakage.py` and the two `exp_*` scripts in `experiments/diagnostics/` if the stability claim stays in the paper. |
| `results/fig1_dolci/biggpu/_*_archive/`, `*_test50x50*`, `results/fig1_dolci_512cap_buggy/`, `results/big/`, `results/qwen4b_tuning/` | 2.5 MB | **drop** (superseded, buggy or tuning runs) |
| `results/fig1/`, `results/paper200*/` (Dolly pilot) | 0.6 MB | keep only if the paper shows the Dolly rank sweep or scaling; otherwise drop together with `experiments/dolly_pilot/` |
| `results/addendum_storage.json` | small | drop (superseded by `_1k` and `_n100`) |
| `data/dolly/dolly_data.jsonl` | 14 MB | **replace by a download** (`databricks/databricks-dolly-15k` + the existing seeded shuffle) only if the loader can be shown to give identical samples; otherwise keep |
| `experiments/diagnostics/` | small | drop from the release |
| git history (134 MB, about 40 commits with cluster hostnames in author emails) | | **publish from a fresh squashed history** of the final tree, not this branch |
| no `LICENSE` at the top; DATE-LM upstream license not vendored | | **add** a LICENSE (your choice) and the upstream DATE-LM license file |

## 6. Reproduction

PI priority: EXP1 first, DATE-LM second.

### 6.1 EXP1 / Figure 1 (Spearman of InfluCoder vs the Qwen3-4B gradient-influence ground truth)

The "~0.8 aggregated" is `results/fig1_dolci/biggpu/seed{0,1,2}/exp1_part1.json` -> `methods.influcoder_68m.aggregated`
= 0.767 / 0.816 / 0.783 (mean 0.789). The 150m encoder gives 0.770 / 0.803 / 0.791. In the same files LoGRA-4B gives
0.928–0.950 and LESS-4B r32 0.966–0.977, both computed against the same ground truth (LoGRA is a competitor here, not
the target).

**Seed 0 reproduces:** InfluCoder-68m 0.778 vs 0.767 committed (+0.010), 150m 0.783 vs 0.770. Full row-by-row
comparison, same config (the `config` block is identical to the committed one), torch 2.7.1 / transformers 4.51.3 /
sentence-transformers 4.1.0, 85 min on one RTX PRO 6000:

| method | committed | reproduced | diff |
|---|---|---|---|
| influcoder_68m | 0.767 | 0.778 | +0.010 |
| influcoder_150m | 0.770 | 0.783 | +0.013 |
| untrained_68m | 0.306 | 0.250 | -0.056 |
| untrained_150m | 0.205 | 0.161 | -0.044 |
| less_4B_r32 / r8 | 0.977 / 0.968 | 0.972 / 0.962 | -0.004 / -0.006 |
| logra_4B | 0.928 | 0.925 | -0.003 |
| less_1.7B_r8 / logra_1.7B | 0.668 / 0.597 | 0.644 / 0.560 | -0.024 / -0.037 |
| less_0.6B_r32 / logra_0.6B | 0.253 / 0.273 | 0.200 / 0.304 | -0.053 / +0.031 |
| rdsplus | 0.039 | 0.041 | +0.002 |
| tfidf | -0.093 | -0.061 | +0.031 |

The headline (InfluCoder ~0.78, LESS-4B / LoGRA-4B above 0.92, the cheap proxies far below) holds. The weak rows move
by 0.03–0.06. TF-IDF uses no model and still moved by +0.031, so the ground truth or the eval samples themselves differ
slightly from the original run, not just the trained methods. Suspects, unverified: a newer revision of the HF dataset
`tasksource/dolci-instruct`, a different library stack (exact original versions are not recorded), and Qwen3-4B bf16
gradients on a different card. LESS also ran without `fast_jl` (TRAK `BasicProjector`). Pinning the dataset revision
and the library versions is a cheap fix before release.

| run | code | card | InfluCoder-68m agg | paper / committed |
|---|---|---|---|---|
| Part 1, seed 0, 800x800 | original `d84c4ac` | RTX PRO 6000 (granite) | **0.778** (150m: 0.783) | 0.767 (150m: 0.770) |
| Part 1, seed 0 | refurb `566a529` | granite | *running (job 7012139)*: old-vs-new on the same card | 0.767 |
| Part 1, seeds 1, 2 | refurb | granite | *running (jobs 7012140/41)* | 0.816 / 0.783 |

### 6.2 DATE-LM (lower priority)

| task | metric | paper | reproduced | code | card |
|---|---|---|---|---|---|
| Counterfact, Pythia-1B | Recall@50 / MRR | 0.460 / 0.830 | 0.457 / 0.874 | original | RTX PRO 6000 (granite), torch 2.7.1 |
| Counterfact | | | 0.452 / 0.829 and 0.456 / 0.826 (2 reps) | refurb `566a529` | granite |
| Counterfact | | | *queued* (first try died on a Triton error in the torch-2.6 venv; rerun with torch 2.7.1) | original | A40 (nancy gruss) |
| Counterfact | | | *queued* | refurb | A40 (nancy gruss) |
| XSTest Het / Hom | AUPRC | 0.646 / 0.714 | **blocked** | | |

On granite, InfluCoder Counterfact is not bit-stable across reps (Recall@50 0.452–0.457, MRR 0.826–0.874 over 3 runs), although the repo documents it as bit-identical on A40. Old and new code fall in the same range.

**Toxicity is blocked.** The paper runner builds its toxic pool from `allenai/wildguardmix`, a gated HF dataset. No
token exists on the cluster, so both runs stop after 10 s. This needs you to accept the dataset terms and provide a
token path; I did not look for workarounds.

### 6.3 Provenance gap found while re-scoring (important for the release)

DATE-LM's own evaluator re-scores every committed score file in `experiments/datelm/results/fig2-*/`. The results match
the recorded JSON numbers exactly for every method except:

| file | JSON (reported) | re-scored committed file |
|---|---|---|
| Counterfact InfluCoder | R@50 0.4604 / MRR 0.8301 | 0.4671 / 0.8158 |
| Het InfluCoder | AUPRC 0.6456 | 0.5365 |
| Hom InfluCoder | AUPRC 0.7140 | 0.7743 |
| Counterfact BM25 | R@50 0.3490 / MRR 0.5829 | 0.3484 / 0.5594 |

Cause, from git history: the InfluCoder score files were last committed in `740f9c3` (08-04 02:26), before the retune.
The retuned numbers were committed in `77f2925` (08-04 04:41, "InfluCoder retuned") without their score files. Likewise
the BM25 file is from the earlier RTX 6000 Ada run, while the JSON holds the later A40 rerun. The reported numbers are
not wrong as such: they are single runs within the documented 3-rep spread. But the files that should back them are not
in the repo. Proposal: regenerate the four score files on an A40 with the shipped config before any data release, or
release only the JSONs.

## 7. Not done / open

- Toxicity reproduction (blocked, section 6.2).
- EXP1 Parts 2–3 not rerun (hours each). Part 1: seed 0 done (original code); seed 0 refurb and seeds 1–2 refurb running.
- LESS without `fast_jl` falls back to TRAK's `BasicProjector` (different random projection), so LESS rows will not be
  bit-identical on a machine without the CUDA kernel.
- Qwen3 needs `transformers>=4.51`, while `requirements.txt` says `>=4.48`. The README says so; I did not change
  requirements, to stay behaviour-neutral.
- Paper link, author list in the BibTeX, overview figure: TODO in the README.
- No deletion done. Everything in section 5 waits for your call.
- Exact library versions used for the paper are not recorded anywhere in the repo. My runs used torch 2.6/2.7.1,
  transformers 4.48.3 (DATE-LM) / 4.51.3 (EXP1); `pip freeze` for each run is in `_lm28/runs/<job>/`.
