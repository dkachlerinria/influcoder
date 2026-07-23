# Baselines

Competitor influence/selection methods ported from the old `tis-ie` repo, each
scored on **exactly** the eval `run.py` reports: the same disjoint BBH×Dolly
split and the same gradient-influence cosine ground truth (`influcoder.gradients`).
A method only has to produce an `[n_eval_a, n_eval_p]` score matrix over those
same samples; `baselines/common.py` builds/caches the GT and computes Spearman,
so every baseline row is directly comparable to the influcoder numbers.

## Layout

```
common.py            shared harness: eval split + cached GT + tokenization + Spearman + reporting
eval_baselines.py    CLI runner (--method ... --preset ...)
logra/               LoGRA  — per-sample LoRA-B gradients + FIM preconditioning (modeling_logra.py copied verbatim)
less/                LESS   — LoRA SGD gradients + TRAK random projection + cosine (less_embeds.py copied ~verbatim)
semantic/            sentence-transformer embedding cosine (== influcoder's untrained-encoder row)
rdsplus/             RDS+   — SGPT weighted-mean pooling of last hidden state, cosine
tfidf/               TF-IDF lexical cosine
random/              seeded uniform noise floor
cache/               cached ground-truth matrices (gt_<preset>_<hash>.pt)
```

## Run

```bash
python -m baselines.eval_baselines --method logra --preset big
python -m baselines.eval_baselines --method less semantic rdsplus tfidf random --preset big
```

## Faithfulness notes

* **Same eval as the table.** The old repo and the new influcoder repo shuffle
  BBH/Dolly differently, so results are only comparable if the *new* repo's data
  pipeline defines the split. The harness therefore builds the eval split + GT
  with `influcoder.data` / `influcoder.gradients` and swaps in only each method's
  scorer.
* **Same gradient geometry.** Gradient-based baselines (LoGRA, LESS, RDS+)
  tokenize each Sample with the identical loss-on-target masking as the GT
  featurizer (`common.tokenize_sample`), and LoGRA LoRA-adapts the same module
  set (`q,k,v,o,gate,up,down`) the GT featurizer uses (`all-linear`) — matching
  the old `config_influence.sh`. Using LoGRA's own `mlp_only=True` default
  instead restricts to MLP layers and understates it.
* **Method internals copied verbatim** where possible: `logra/modeling_logra.py`
  is byte-identical to the old repo; `less/less_embeds.py` keeps `collect_grads`
  verbatim (only unused/removed top-level imports were trimmed — see the note in
  that file).

## Extra dependencies

* **LESS** needs `traker` (TRAK projectors). For the fast CUDA projector it also
  wants `fast_jl`; without it, LESS falls back to `BasicProjector` (pure torch,
  same result, slower). Install: `pip install traker` (+ `pip install fast_jl`
  for the CUDA kernel).
