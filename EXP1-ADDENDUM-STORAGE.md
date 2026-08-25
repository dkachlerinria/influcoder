# EXP1 addendum — representation storage cost

How much disk does each method's per-example representation need, at EXP1's
ranking-estimation setup? EXP1's existing plots cover ranking quality and
wall-clock; this covers the third axis — the size of the index you have to keep
around once scoring is done.

Produced by `baselines/exp1/addendum_storage.py`. The committed run is
**n = 1,000 dolci-instruct examples**, written to
`baselines/out/addendum_storage.json` (tracked -- `baselines/out/` is
deliberately not gitignored, same as EXP1's other tables).

The JSON is the source of truth for every number quoted below. Each row carries
its analytic width and, where the method could be measured on the run host, a
`measured` block with the real tensor shape/dtype, in-memory `nbytes`, and the
on-disk size of a `torch.save` artifact. Rows whose method needs a GPU
(per-sample gradient extraction for LESS/LoGra, and the target-model forward
pass for RDS+ at this scale) carry a `skipped` reason instead -- their analytic
widths are still exact, since they are fixed by config and architecture rather
than measured.

## 1. What is actually stored

All four methods reduce a training example to **one fixed-length vector** and
score by cosine. They differ only in how wide that vector is:

| method | per-example vector | width set by |
|---|---|---|
| InfluCoder | student-encoder sentence embedding | encoder hidden size |
| RDS+ | weighted-mean last hidden state of the target model | target hidden size |
| LESS | TRAK random projection of the LoRA gradient | `proj_dim` (a free choice) |
| LoGra | per-module flattened `[r, r]` LoRA-B gradient blocks, concatenated | `n_modules × r²` |

**The width does not depend on example length.** Every method pools or projects
to a fixed dimension, so bytes-per-example is a constant and the 1K figure is an
exact multiple of the 10-example one — the small run is a proof of concept, not
a sample estimate to be extrapolated with error bars.

**InfluCoder needs no training for this measurement.** Distillation changes the
encoder's weights, never its output width, so an untrained `ettin-encoder-*`
produces a byte-for-byte identically sized artifact. No gradient collection, no
distillation.

## 2. Setup

Inherited from EXP1 ranking estimation (`baselines/exp1/config.py`,
`config_biggpu.py`), pool = `tasksource/dolci-instruct`:

```
target / GT model   Qwen/Qwen3-4B        # RDS+ scores with this; LESS and LoGra
                                          # also run 1.7B / 0.6B proxies
LESS proj_dim       8192                 # cfg.LESS_PROJ_DIM
LoGra rank          32                   # cfg.LOGRA_RANK, biggpu final run (8 historically)
LoGra modules       q,k,v,o,gate,up,down # cfg.LOGRA_TARGET_MODULES -> 7 per layer
encoders            ettin-encoder-68m / -150m
max_len             1024
```

## 3. Results — 1,000 dolci examples

| Method | dim | dtype | B/example | n=10 | **n=1,000** | vs InfluCoder-68m |
|---|---:|---|---:|---:|---:|---:|
| **InfluCoder (68m)** | 512 | float32 | 2,048 | 20.0 KiB | **2.0 MiB** | 1.0× |
| InfluCoder (150m) | 768 | float32 | 3,072 | 30.0 KiB | **2.9 MiB** | 1.5× |
| RDS+ | 2,560 | bfloat16 | 5,120 | 50.0 KiB | **4.9 MiB** | 2.5× |
| LESS | 8,192 | float16 | 16,384 | 160.0 KiB | **15.6 MiB** | 8.0× |
| LoGra (r=8) | 16,128 | bfloat16 | 32,256 | 315.0 KiB | **30.8 MiB** | 15.8× |
| **LoGra (r=32)** | 258,048 | bfloat16 | 516,096 | 4.9 MiB | **492.2 MiB** | 252.0× |

r=32 is the rank the final (biggpu) run used; r=8 is the historical `config.py`
profile, included because the gap between them is the single largest effect in
the table.

## 4. Where each width comes from

- **InfluCoder** — `ettin-encoder-68m` hidden size 512 (`config.json`).
  `influcoder/encoder.py:embed()` calls `enc.encode(..., convert_to_numpy=True)`,
  which returns **float32**.
- **RDS+** — Qwen3-4B hidden size 2560. `baselines/rdsplus/score.py:weighted_mean_embeds`
  keeps the model's bf16 dtype all the way through `.cpu()`.
- **LESS** — `cfg.LESS_PROJ_DIM = 8192`. `baselines/less/less_embeds.py:_project`
  casts to `torch.float16` before the TRAK projector.
- **LoGra** — `baselines/logra/modeling_logra.py:step()` concatenates each
  LoRA-B module's `[B, r, r]` gradient flattened to `[B, r²]`. Qwen3-4B has
  **252** target modules (36 layers × 7), verified by counting `nn.Linear`
  modules with matching suffixes on a meta-device model — so 252 × 32² = 258,048.

## 5. How each one scales

This matters more than the absolute numbers:

- **InfluCoder is the only method whose index size is independent of the model
  being attributed.** It is set by the student encoder alone. Attributing a 70B
  checkpoint instead of a 4B one leaves it at 2.0 MiB/1K.
- **RDS+** scales with target hidden size — linear in model width.
- **LESS** is independent of both target size *and* LoRA rank; `proj_dim` is a
  free knob. It is the only method here whose storage is a deliberate choice
  rather than a consequence of the architecture.
- **LoGra scales with target depth × rank².** Going r=8 → r=32 is 16× the
  storage. This is why it lands at ~0.5 GiB per 1K examples at the final run's
  rank, and it gets worse with deeper targets.

## 6. Fairness notes

- **InfluCoder's number is the pessimistic one.** It is the only method in the
  table stored at 4 bytes/dim; the other three are already at 2. Storing the
  embedding at fp16 — a lossless change for a cosine-ranked index in practice,
  though not verified here — would halve it to 1.0 MiB/1K and double every ratio
  in the last column.
- **LESS's 8.0× is a config artifact, not a property of the method.** At
  `proj_dim=2048` it would sit at 3.9 MiB/1K. EXP1 already measured what that
  costs in quality (`config_biggpu.py`: agg rho +0.9144 at proj_dim=2048 vs
  +0.9766 at 8192, a −0.062 drop) — so the storage is buying real accuracy, but
  the trade is available.
- These are raw tensor sizes. No quantization or compression is applied to any
  method; applying it uniformly would shift all rows, not reorder them.

## 7. Reproduce

```bash
# the committed run: 1,000 dolci examples -> baselines/out/addendum_storage.json
python -m baselines.exp1.addendum_storage --n 1000 --mode real --target Qwen/Qwen3-4B

# analytic only -- widths from model configs, no weights downloaded, seconds to run
python -m baselines.exp1.addendum_storage --n 1000

# quick proof of concept
python -m baselines.exp1.addendum_storage --n 10 --mode real --target Qwen/Qwen3-0.6B
```

On a GPU node the same command fills in the `measured` blocks for RDS+ (and,
with the gradient methods wired in, LESS/LoGra) instead of recording a `skipped`
reason. `--force_cpu_target` runs the target-model methods on CPU anyway; it is
tractable only at small `--n`.
