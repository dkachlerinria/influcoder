# EXP2 — file guide

How the EXP2 benchmark is laid out: which file produces which number, how they
link together, and where every hyperparameter lives.

**One sentence:** three `run_fig2_*.py` scripts (one per task) each hold every
parameter for every method at the top of the file, dispatch one method per
invocation via `--method`, write scores to `results/`, and accumulate metrics
into a per-task JSON that `make_final_jsons.py` filters into the three
`exp2-*-final.json` headline files.

---

## 1. The results

| file | task | metrics |
|---|---|---|
| `exp2-counter-final.json` | Counterfact | Recall@50, MRR |
| `exp2-het-final.json` | Toxicity/Bias — XSTest-response-**Het** | AUPRC |
| `exp2-hom-final.json` | Toxicity/Bias — XSTest-response-**Hom** | AUPRC |

These hold only the **7 kept methods**: `bm25`, `repsim`, `gradsim`, `less`,
`datainf`, `semantic`, `influcoder`. They are *generated*, never hand-edited —
regenerate at any time with:

```bash
python methods/make_final_jsons.py
```

It reads the three full result files below and filters them by the `KEEP` list
at the top of that script. `graddot` and `ekfac` were run but dropped from the
kept set; their numbers survive in the full files.

**Full results** (all methods, including dropped ones):

```
results/fig2_counterfact.json
results/fig2_toxicity.json          # Het
results/fig2_toxicity_hom.json      # Hom
```

Each is a flat `{method_name: record}` map. A record always carries:

| field | meaning |
|---|---|
| `task` | `"Counterfact"` or `"Toxicity/Bias"` |
| `wall_s` | **the comparable cost number** — see §4 |
| `recall_at_50`, `mrr` *or* `auprc` | the accuracy metric for that task |
| `gpu` | auto-detected model + hostname the run executed on |
| `setup_s` | *(InfluCoder / Rep-Sim / Semantic only)* one-time cost, excluded from `wall_s` |
| `setup_note` | what that setup covers |

InfluCoder records additionally carry `encoder`, `epochs`, `lr`, `hard_ratio`
and a `config_note` recording the measured run-to-run spread — so the reported
value is not mistaken for exact. See §6.

**Raw scores** (what the metrics were computed from):

```
results/fig2-<task>-<method>/<Method>.pt
   e.g. results/fig2-toxicity-hom-less/LESS.pt
```

Despite the `.pt` extension these are **JSON**, matching upstream DATE-LM's
convention. Counterfact stores a `[n_ref][n_train]` matrix; the toxicity tasks
store a flat `[n_train]` vector (mean over the reference set), which is what
`AUPRC()` expects.

---

## 2. The scripts that produce them

```
methods/run_fig2_counterfact.py     # Counterfact
methods/run_fig2_toxicity.py        # Toxicity/Bias, Het
methods/run_fig2_toxicity_hom.py    # Toxicity/Bias, Hom
```

One method per invocation, so a preempted GPU only loses that method:

```bash
export PYTHONPATH="_stubs:.:$PYTHONPATH"
python methods/run_fig2_toxicity.py --method less
```

Valid `--method`: `bm25 repsim graddot gradsim less datainf ekfac influcoder semantic`

Each script:
1. runs the method → writes `results/fig2-<task>-<method>/<Method>.pt`
2. scores it with **DATE-LM's own evaluation functions**, imported not
   reimplemented (`get_fact_indices_counterfact` + `evaluate_fact`, or
   `get_unsafe_indices` + `AUPRC`, from `evaluation/evaluate_application.py`)
3. merges the record into the task's summary JSON under a file lock

The three scripts are near-identical; the Hom one differs from Het only in
`SUBSET`, `CHECKPOINT`, and output paths.

---

## 3. Where the parameters are — the important bit

**Every parameter lives in one block at the top of each `run_fig2_*.py`**
(roughly lines 100–160). Nothing is hidden in a config file, an env var, or a
shell script. To find what produced a number, open the script for that task and
read the block.

```python
TASK / SUBSET / BASE_MODEL / CHECKPOINT   # what is being attributed
MAX_LEN = 1024                            # tokenizer truncation
SEED = 0

DATTRI_METHOD_NAME    # our key -> dattri's name, e.g. "less" -> "LESS"
DATTRI_EXTRA_PARAMS   # per-method hyperparameters (see below)
REPSIM_BATCH_SIZE = 1
BM25_STOPWORDS = "en"

INFLUCODER_*          # ~12 constants: encoder, pool sizes, epochs, lr, hard_ratio
```

`DATTRI_EXTRA_PARAMS` matches upstream DATE-LM's own YAML defaults exactly
(`configs/factual-attribution.yaml`, `configs/toxicity-bias.yaml`):

```python
"less":    {"proj_dim": 8192}
"datainf": {"regularization": 1e-5, "fim_estimate_data_ratio": 1.0}
"ekfac":   {"damping": 1e-7}
"graddot": {}, "gradsim": {}          # no hyperparameters
```

Two independent ways to confirm what a committed number used:

- **the script** — the constants above, as committed alongside the results
- **the record** — InfluCoder records embed `encoder`/`epochs`/`lr`/`hard_ratio`
  directly, so the JSON is self-describing without cross-referencing anything

---

## 4. What `wall_s` means (and the one asymmetry that remains)

`wall_s` is the **comparable per-query cost**. All results were measured on a
single GPU model (**NVIDIA A40**, Rennes) precisely so this column can be
compared across rows — check the `gpu` field to confirm.

Three methods report an additional `setup_s`, excluded from `wall_s`:

| method | setup covers | why excluded |
|---|---|---|
| `influcoder` | pool building + teacher gradients + encoder load + distillation | one-time, amortizes over future queries |
| `semantic` | `get_dataset` + encoder load | so its `wall_s` brackets the *same* region as InfluCoder's |
| `repsim` | checkpoint load + `get_dataset` + dataset construction | same convention |

Semantic is InfluCoder's control — identical encoder and texts, only the
weights differ — so their `wall_s` values are directly comparable and land
within ~0.5% of each other.

**Remaining asymmetry, stated plainly:** the five dattri-backed methods
(`graddot`, `gradsim`, `less`, `datainf`, `ekfac`) load the checkpoint *inside*
upstream's `attribute()`, which we keep byte-identical, so their `wall_s`
carries ~5s of model load that the three above exclude. At their 340–6400s
scale that is under 1.5%.

`setup_s` is reported as **cost-from-scratch** even when teacher gradients were
served from cache, so it can exceed a run's actual elapsed time. That is
deliberate — it keeps cached and uncached runs comparable.

---

## 5. Fidelity to upstream DATE-LM

Verified by diffing against a fresh clone of `DataAttributionEval/DATE-LM`:

- `dattri/` — **byte-identical** (all five gradient-based attributors)
- `datamodules/` — **byte-identical** (the data path every method consumes)
- `methods/model_utils.py` — **byte-identical** (checkpoint loading)
- hyperparameters — match upstream's own YAML defaults

Three one-line deviations, all in CLI/plumbing that our scripts bypass, all
fixing upstream bugs: `args.config_path`→`args.config` (upstream reads an
attribute its argparse never defines), the task-name check in
`evaluate_application.py` (upstream compares against *subset* names, so its
toxicity branch can never fire with its own config), and wiring `--score_path`
into the config (defined but never read).

**Known exceptions:** BM25 is *not* upstream's implementation (different
library, tokenization, and indexed corpus). Rep-Sim is a re-host — verified
line-by-line equivalent to `methods/repsim/repsim_utils_huggingface.py`.
InfluCoder and Semantic are ours by definition.

---

## 6. InfluCoder: reproducibility

InfluCoder is the only method with a training step, so it is the only one whose
accuracy is not bit-reproducible. Cause: the memory-efficient SDPA attention
**backward** is non-deterministic by design, and that compounds over 8 epochs.

Ruled out as causes, each with direct evidence in
`results_influcoder_sweep/summary.json`:

| hypothesis | outcome |
|---|---|
| best-epoch selection | `restore_best=False` didn't help (0.687 vs 0.537) |
| unseeded `torch.randperm` | `hard_ratio=0.0` bypasses it, still varied (0.496 vs 0.419) |
| RNG seeding + cudnn flags | unchanged (0.727 vs 0.651) |
| deterministic MATH attention | correct in principle, **OOMs** at `seq_len=1024` — the seq×seq attention matrices dominate and do *not* shrink with a smaller encoder |

**What actually fixed it: lowering the learning rate.** At fixed
`hard_ratio=0.25`, 2-rep spreads were **0.075 / 0.018 / 0.0007** at lr =
5e-5 / 2e-5 / 1e-5. Hard mining was *not* the culprit — removing it entirely was
worse on both stability and accuracy.

Shipped config: `ettin-encoder-400m`, `epochs=8`, `lr=2e-5`, `hard_ratio=0.25`.

Reported values are a **single run (rep 2 of 3), not a mean**. Measured 3-rep
spread at this config: Counterfact **0.000** (bit-identical — its ~7-word texts
are too short for the nondeterminism to accumulate), Het **0.041**, Hom **0.021**.

The other six methods have no training step and reproduce to ~0.002 even across
different GPU models.

---

## 7. Supporting tools

| file | purpose |
|---|---|
| `methods/make_final_jsons.py` | regenerates the three `exp2-*-final.json` |
| `methods/exp_influcoder_sweep.py` | hyperparameter sweep; writes to its own dir, never touches official results |
| `methods/exp_hard_ratio_stability.py` | the hard-mining / stability probe |
| `methods/bench_embed.py` | isolated `embed()` throughput benchmark |
| `methods/influcoder/grad_cache.py` | caches teacher gradients (see below) |
| `results_influcoder_sweep/summary.json` | all 28 sweep runs — the evidence for §6 |

**Gradient cache.** Teacher gradients depend only on the seeded pools and the
fixed checkpoint, never on any training hyperparameter, so they are computed
once and reused. Cached under `results/_grad_cache/`, keyed by a SHA-256 digest
of the actual tokenized `input_ids`/`labels` plus checkpoint/`proj_dim`/
`proj_seed` — a hit cannot serve gradients belonging to different data. Saves
~2–4 min per run. The tensors are gitignored (regenerable); delete the
directory to force recomputation.

---

## 8. Reproducing a number

```bash
cd EXP2-datelm
export PYTHONPATH="_stubs:.:$PYTHONPATH"

python methods/run_fig2_toxicity.py --method less        # -> results/fig2_toxicity.json
python methods/make_final_jsons.py                       # -> exp2-het-final.json
```

The parameters used are the constants at the top of that script, as committed.
For InfluCoder, expect the spread quoted in §6 rather than an exact match;
every other method should reproduce to ~0.002.
