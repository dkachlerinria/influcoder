# InfluCoder

Distill gradient-influence rankings into a small text bi-encoder, so that at
selection time scoring a candidate is one encoder forward pass instead of a
forward+backward through the target LM.

**Method.** Attach a fresh, seeded LoRA adapter to a target LM. Each sample's
loss gradient on the adapter, sketched to `proj_dim` dims with a seeded
CountSketch, is its *influence feature*; influence(anchor, candidate) =
cosine of their features. On a **train** split of anchors x pool we compute
that matrix and train a sentence encoder to reproduce it from text
(Pearson + listwise-KL loss). On a **disjoint eval** split we compute the same
gradient matrix as ground truth and report Spearman rank agreement of the
encoder's scores against it — versus the untrained encoder as baseline.

Anchors come from BBH (`data/eval/bbh/`, checked in), candidates from Dolly
(`dolly/dolly_data.jsonl`, checked in). No downloads, no intermediate files.

## Layout

```
run.py                    pipeline runner: presets, orchestration, results.json
influcoder/data.py        BBH + Dolly loading, disjoint splits, text views
influcoder/gradients.py   LoRA featurizer, CountSketch, projection-fidelity check
influcoder/encoder.py     bi-encoder distillation (Pearson+KL loss, training loop)
influcoder/metrics.py     Spearman metrics
```

## Setup & run

```bash
pip install -r requirements.txt

python run.py --preset sanity   # smallest end-to-end check + projection fidelity
python run.py --preset tiny     # small but non-degenerate reproduction
python run.py --preset push     # bigger train side, hard-negative mining
python run.py --preset big      # bigger eval+train, no mining, fewer epochs
```

Defaults: `SmolLM2-135M` as gradient source, `ettin-encoder-68m` as encoder
(both configurable via `--grad_model` / `--encoder_model`). Each run writes to
`runs_out/<preset>/` (gitignored -- these are outputs, not source):

- `results.json` -- config, baseline/trained Spearman, per-epoch metrics, timings
- `encoder/` -- the trained encoder (best-checkpoint-restored), loadable directly:
  ```python
  from sentence_transformers import SentenceTransformer
  enc = SentenceTransformer("runs_out/<preset>/encoder")
  ```

The sanity preset also runs a **projection fidelity check**: exact pairwise
gradient cosines vs. sketched cosines on held full gradients — the numeric
proof that the featurizer core is correct, independent of any reference
implementation.

## Hardware

The code picks dtype/attention from the GPU's compute capability:

| GPU | What runs | Expected speed |
|---|---|---|
| Ampere or newer (A100, H100, A40, RTX 30/40/50xx) | bf16 + SDPA/flash | ~20–60 ms per gradient sample (135M model) |
| Older (P100, V100) | fp32 + eager fallback | ~1 s per gradient sample — works, but slow |

Per-sample gradient extraction is the only real cost and it is embarrassingly
parallel across samples; `torch.func` vmap'd per-sample gradients or sharding
across GPUs are the known next steps if scale demands it.

## Correctness anchor

The original full pipeline this was rewritten from is preserved at the git tag
`legacy-pipeline` (and in the upstream `tis-ie` repo). It is a reference
oracle for cross-checking outputs if results ever look wrong — not a template.
