# InfluCoder

Code for **"Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution"** (AACL-IJCNLP 2026).
[Paper](TODO-link) · [Citation](#citation)

Gradient-based data attribution (LESS, LoGRA, …) needs a forward *and* backward pass through the target LM for
every candidate example. InfluCoder distills those gradient-influence rankings into a small text bi-encoder:
at selection time, scoring a candidate costs one encoder forward pass.

<!-- TODO: overview figure, e.g. results/fig1_dolci/biggpu/seed0/figure1.png -->

**Method in one paragraph.** Attach a fresh, seeded LoRA adapter to the target LM. Each example's loss gradient on
the adapter, sketched to `proj_dim` dimensions with a seeded CountSketch, is its influence feature;
influence(anchor, candidate) is the cosine of the two features. On a train split of anchors × candidates we compute
that matrix and train a sentence encoder to reproduce it from text (Pearson + listwise-KL loss). On a disjoint eval
split we report the Spearman rank agreement of the encoder's scores with the gradient ground truth.

## Installation

```bash
git clone <this repo> && cd influcoder
pip install -r requirements.txt
# Figure 1 experiments (Qwen3 teachers, LESS baseline) additionally need:
pip install "transformers>=4.51" traker      # optional: fast_jl for the CUDA TRAK projector
```

Tested with Python 3.11 and PyTorch 2.6–2.7 on A40 / RTX PRO 6000 GPUs. Pre-Ampere GPUs fall back to fp32 + eager
attention automatically.

## Data

| What | Where | Source |
|---|---|---|
| BBH anchors | `data/eval/bbh/` (checked in) | BIG-Bench Hard |
| Dolly candidates | `data/dolly/dolly_data.jsonl` (checked in) | databricks-dolly-15k |
| dolci-instruct candidates (Figure 1) | downloaded on first use | `tasksource/dolci-instruct` (HF) |
| DATE-LM tasks and checkpoints | downloaded on first use | `DataAttributionEval/*` (HF) |

## Quickstart

```bash
bash scripts/quickstart.sh          # = python run.py --preset sanity && python run.py --preset tiny
```

`run.py` computes the gradient ground truth with a small teacher (SmolLM2-135M by default), distills it into
`ettin-encoder-68m`, and writes `runs_out/<preset>/results.json` (untrained vs. trained Spearman) and the trained
encoder, loadable with `SentenceTransformer("runs_out/<preset>/encoder")`.

## Reproducing the paper

| Paper result | Command | Output (committed copy) |
|---|---|---|
| **Figure 1**: agreement with gradient influence (Qwen3-4B ground truth) vs. cost; InfluCoder vs. LESS, LoGRA, RDS+, TF-IDF, untrained encoder; 3 seeds | `bash scripts/agreement_fig1.sh` | `results/fig1_dolci/biggpu/seed{0,1,2}/` |
| Distillation-size scaling and amortization (Fig. 1 panels) | included above (Parts 2, 3) | same directory |
| Loss-weight (α) ablation | `bash scripts/agreement_ablations.sh` | `results/fig1_dolci/default/alpha_ablation.json` |
| Index storage per 1K examples | same script | `results/addendum_storage_{1k,n100}.json` |
| **DATE-LM** factual attribution (Counterfact, Pythia-1B) and toxicity filtering (XSTest-response Het/Hom) | `bash scripts/datelm.sh` | `experiments/datelm/exp2-{counter,het,hom}-final.json` |

All hyperparameters live in one place per experiment: `experiments/agreement/config.py` (+ `config_biggpu.py`, the
paper setting) and the constant block at the top of each `experiments/datelm/methods/run_fig2_*.py`.

InfluCoder's distillation is not bit-reproducible on GPU (non-deterministic SDPA backward); expect a spread of
about ±0.02 in aggregated Spearman / AUPRC across runs. The training-free methods reproduce to ~1e-3.

## Repository layout

```
influcoder/            the method: data loading, LoRA gradient featurizer + CountSketch, bi-encoder distillation, metrics
run.py                 end-to-end InfluCoder pipeline (presets sanity / tiny / push / big / paper200)
baselines/             competitor scorers on the same eval: LoGRA, LESS, RDS+, semantic, TF-IDF, random
                       + shared harness (common.py: splits, cached ground truth, Spearman; eval_baselines.py: CLI)
experiments/
  agreement/           Figure 1 and appendix ablations (Parts 1-3, alpha ablation, storage)
  datelm/              DATE-LM benchmark integration (vendored upstream code + our run_fig2_*.py runners)
  dolly_pilot/         earlier Dolly-pool experiments (rank sweep, scaling, OOD)
  diagnostics/         timing / tuning helpers, not needed for the paper numbers
scripts/               one shell script per paper result
data/                  BBH anchors and Dolly candidates
results/               committed result files behind the paper's numbers and figures
```

## Acknowledgements

`experiments/datelm/` vendors the [DATE-LM](https://github.com/DataAttributionEval/DATE-LM) benchmark;
see its own README (TODO: confirm upstream license before release). `baselines/logra/modeling_logra.py` and `baselines/less/less_embeds.py` are adapted
from the LoGRA and LESS reference implementations.

## Citation

```bibtex
@inproceedings{influcoder2026,
  title     = {Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution},
  author    = {TODO},
  booktitle = {Proceedings of AACL-IJCNLP 2026},
  year      = {2026}
}
```
