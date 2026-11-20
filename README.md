# InfluCoder

[![Paper](https://img.shields.io/badge/Paper-AACL--IJCNLP%202026-b31b1b.svg)](TODO-paper-link)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](requirements.txt)

InfluCoder is a fast, representation-based data attribution method. Gradient-based methods such as LESS and LoGRA
need a forward and a backward pass through the target LM for every candidate. InfluCoder distills the LM's
gradient-influence rankings into a small text encoder, so scoring a candidate costs a single encoder forward pass.
With a 68M encoder it reaches a Spearman correlation of about 0.8 with Qwen3-4B gradient influence, at about 2 ms per
candidate.

**Paper:** [Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution](TODO-paper-link) (AACL-IJCNLP 2026)
**Authors:** Dimitri Kachler, Damien Sileo, Pascal Denis

<img width="800" alt="Figure 1: agreement with gradient influence vs. inference cost" src="results/fig1_dolci/biggpu/seed0/figure1.png">

## Quick Start

```bash
pip install -r requirements.txt
pip install "transformers>=4.51" traker    # for EXP1 (Qwen3 teachers, LESS baseline)
```

A minimal end-to-end run, about one minute on one GPU: SmolLM2-135M teacher, `ettin-encoder-68m` student, BBH anchors
× Dolly candidates.

```bash
python run.py --preset tiny
```

Score candidates with the trained encoder:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("runs_out/tiny/encoder")      # written by run.py

queries    = ["Q: Is the following sentence plausible? ..."]   # e.g. BBH anchors
candidates = ["Write a haiku about autumn.\nLeaves drift ...",  # e.g. instruction-tuning examples
              "What is the capital of France?\nParis."]

q = encoder.encode(queries, normalize_embeddings=True)
c = encoder.encode(candidates, normalize_embeddings=True)
scores = q @ c.T            # [n_queries, n_candidates]: higher = more influential
```

## Reproduce EXP1 (Figure 1)

Every method is scored on the same task: rank 800 dolci-instruct candidates for each of 800 BBH anchors. The score
is the Spearman correlation with the ground truth, Qwen3-4B gradient influence (rank-16 LoRA gradients, cosine),
plotted against inference cost. All parameters are in `experiments/exp1/config.py`; the paper setting overrides them
in `experiments/exp1/config_biggpu.py`. Run from the repository root on one large GPU (tested on a 96 GB RTX PRO 6000).

Select the paper setting and a seed (the paper uses seeds 0, 1, 2):

```bash
export EXP1_CONFIG=biggpu EXP1_SEED=0 EXP1_INFLUCODER_SEED=0 EXP1_DATA_SEED=0
```

Or run all steps for all three seeds with `bash scripts/exp1_figure1.sh`. Outputs go to
`results/fig1_dolci/biggpu/seed<k>/`; the committed files there are the paper's numbers.

### 1) Score every method (quality vs. cost)

```bash
python -m experiments.exp1.part1
```

This computes the Qwen3-4B ground truth (cached after the first run) and distills InfluCoder (68M and 150M encoders,
saved under `runs_out/`). It then scores InfluCoder, the untrained encoders, LESS and LoGRA (0.6B / 1.7B / 4B
teachers), RDS+ and TF-IDF. Takes about 1.5 h per seed.

Output format (`exp1_part1.json`):

```json
{
  "config":  {"n_eval": 800, "gt_model": "Qwen/Qwen3-4B", "gt_lora_rank": 16, "influcoder_epochs": 8, ...},
  "methods": {
    "influcoder_68m": {"aggregated": 0.767,   // Spearman vs. ground truth, all anchor-candidate pairs
                       "per_anchor_mean": 0.741,
                       "time_per_sample_ms": 1.8, "flops_per_sample": ...},
    "less_4B_r32": {...}, "logra_4B": {...}, "rdsplus": {...}, "tfidf": {...}, ...
  }
}
```

### 2) Scale the number of distillation examples

```bash
python -m experiments.exp1.part2
```

Output format (`exp1_part2.json`):

```json
{
  "points": [{"n_train_anchors": 25, "n_train_pool": 50, "untrained_agg": 0.306, "agg": 0.585, ...}, ...],
  "reference_lines": {...}      // LESS / LoGRA on the same eval
}
```

### 3) Measure amortized cost

```bash
python -m experiments.exp1.part3
```

Output format (`exp1_part3_influcoder.json`; the LESS/LoGRA counterpart is `exp1_part3_less_logra.json`):

```json
{
  "setup_time_s": 771,      // one-time: teacher gradients (593 s) + distillation (178 s)
  "process_points": [{"n": 100, "process_time_s": 0.50, "ms_per_sample": 5.0}, ...]
}
```

### 4) Collect and plot

```bash
python -m experiments.exp1.collect_results     # -> EXP1_results.json
python -m experiments.exp1.plot_figure1        # -> figure1.{pdf,png}
```

Paper numbers (seed 0, `results/fig1_dolci/biggpu/seed0/exp1_part1.json`):

| Method | Spearman | ms / candidate |
|---|---|---|
| **InfluCoder (68M encoder)** | **0.77** | **~2** |
| Untrained encoder (68M) | 0.31 | ~2 |
| LESS (Qwen3-4B, rank 32) | 0.98 | ~176 |
| LoGRA (Qwen3-4B) | 0.93 | ~61 |
| LESS / LoGRA with the 0.6B proxy | 0.25 / 0.27 | ~114 / ~22 |

Seeds 1 and 2 give 0.82 and 0.78 for InfluCoder (68M). Distillation is not bit-reproducible on GPU (the SDPA attention
backward is non-deterministic), so expect about ±0.02 between runs.

Ablations: `python -m experiments.exp1.part1_alpha_ablation` (loss weight α; summary in
`experiments/exp1/ALPHA_ABLATION.md`) and `bash scripts/exp1_ablations.sh` (index storage per 1K examples).

## DATE-LM Applications

Factual attribution (Counterfact) and toxicity filtering (XSTest-response) on the
[DATE-LM](https://github.com/DataAttributionEval/DATE-LM) benchmark with Pythia-1B:

```bash
bash scripts/datelm.sh        # results -> experiments/datelm/exp2-{counter,het,hom}-final.json
```

The toxicity tasks need access to the gated HF dataset `allenai/wildguardmix`. Details:
[`experiments/datelm/EXP2-GUIDE.md`](experiments/datelm/EXP2-GUIDE.md).

The per-example score files behind the reported numbers are in `experiments/datelm/results/`, except InfluCoder (all
three tasks) and Counterfact BM25: those will come with the data release.

## Repository Layout

```
influcoder/          the method: LoRA gradient features + CountSketch, encoder distillation, metrics, data
run.py               end-to-end pipeline (Quick Start presets)
baselines/           LESS, LoGRA, RDS+, semantic, TF-IDF, random, scored on the same eval
experiments/exp1/    EXP1 / Figure 1 and its ablations
experiments/datelm/  DATE-LM integration (vendored upstream code + our runners)
scripts/             one shell script per experiment
data/                BBH anchors, Dolly candidates (dolci-instruct is downloaded from the HF Hub)
results/             result files behind the paper's numbers and figures
```

`baselines/less/` and `baselines/logra/` adapt the [LESS](https://github.com/princeton-nlp/LESS) and LoGRA reference
implementations; `experiments/datelm/` vendors DATE-LM.

## License

MIT, see [LICENSE](LICENSE). This covers our code. The vendored DATE-LM code in `experiments/datelm/` stays under its
authors' terms (see [`experiments/datelm/NOTICE`](experiments/datelm/NOTICE)).

## Citation

```bibtex
@inproceedings{influcoder2026,
  title     = {Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution},
  author    = {Kachler, Dimitri and Sileo, Damien and Denis, Pascal},
  booktitle = {Proceedings of AACL-IJCNLP 2026},
  year      = {2026}
}
```
