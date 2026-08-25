# EXP1 addendum — representation storage cost

How much disk does each method's per-example representation need, at EXP1's
ranking-estimation setup? EXP1's existing plots cover ranking quality and
wall-clock; this covers the third axis — the size of the index you keep once
scoring is done.

Produced by `baselines/exp1/addendum_storage.py`. Committed runs:

| file | n | schema |
|---|---|---|
| `baselines/out/addendum_storage_1k.json` | 1,000 | `native` + `normalized` |
| `baselines/out/addendum_storage_n100.json` | 100 | `native` + `normalized` |
| `baselines/out/addendum_storage.json` | 1,000 | **superseded** — native-only, flat blocks, predates dtype normalization |

## 1. What is actually stored

All four methods reduce a training example to **one fixed-length vector** and
score by cosine. They differ only in how wide that vector is:

| method | per-example vector | width set by |
|---|---|---|
| InfluCoder | student-encoder sentence embedding | encoder hidden size |
| RDS+ | weighted-mean last hidden state of the target model | target hidden size |
| LESS | TRAK random projection of the LoRA gradient | `proj_dim` (a free choice) |
| LoGra | per-module flattened `[r, r]` LoRA-B gradient blocks, concatenated | `n_modules × r²` |

LESS "embeds" in exactly the sense that matters here: it projects each example
down to a smaller space, and the width of that space is its embedding size.

**The width does not depend on example length.** Every method pools or projects
to a fixed dimension, so bytes-per-example is a constant. This was verified by
running at two sample counts (100 and 1,000) and getting identical widths.

**InfluCoder needs no training for this measurement.** Distillation changes the
encoder's weights, never its output width, so an untrained `ettin-encoder-*`
produces a byte-for-byte identically sized artifact.

## 2. Setup

From `baselines/exp1/config.py` / `config_biggpu.py`, pool =
`tasksource/dolci-instruct`:

```
target / GT model   Qwen/Qwen3-4B        # RDS+ scores with this; LESS and LoGra
                                          # also run 1.7B / 0.6B proxies
LESS proj_dim       8192                 # cfg.LESS_PROJ_DIM
LoGra rank          32                   # cfg.LOGRA_RANK, biggpu (8 historically)
LoGra modules       q,k,v,o,gate,up,down # cfg.LOGRA_TARGET_MODULES -> 7 per layer
encoders            ettin-encoder-68m / -150m
max_len             1024
```

Measured on an NVIDIA RTX A5000 (`abacus11-1.rennes.grid5000.fr`).

## 3. The dtype trap

**The methods do not store at a common dtype, and three of them do not store at
the dtype their source reads like.** Measured directly:

| method | looks like | actually stores | why |
|---|---|---|---|
| InfluCoder | float32 | **float32** | `embed()` uses `convert_to_numpy=True` |
| RDS+ | bfloat16 (model dtype) | **float32** | `rdsplus/score.py` builds position weights with `torch.arange(...)` (fp32), so `hidden * w` type-promotes |
| LESS | float16 | **bfloat16** | `_project()` casts its *input* to fp16, but the TRAK projector is constructed with `dtype=next(model.parameters()).dtype` and emits at the model dtype |
| LoGra | bfloat16 (model dtype) | **float32** | `modeling_logra.step()`'s per-sample gradients accumulate in fp32 |

Left uncorrected this confounds the comparison: LESS sits at 2 B/dim while
every other method is at 4, so LESS's index looks half as large for a reason
that has nothing to do with the method. Note the direction — it *understated*
InfluCoder's advantage rather than inflating it.

So every row is measured twice. `native` is what the implementation really
produces; `normalized` casts everything to a common `--store_dtype` (default
float32). **The normalized numbers are the comparable ones**: with dtype fixed,
the only thing that can make one index larger is the width of the vector.

## 4. Results — dtype normalized (float32, 4 B/dim)

| Method | dim | B/example | n=100 | **n=1,000** | vs InfluCoder-68m |
|---|---:|---:|---:|---:|---:|
| **InfluCoder (68m)** | 512 | 2,048 | 200.0 KiB | **2.0 MiB** | 1.0× |
| InfluCoder (150m) | 768 | 3,072 | 300.0 KiB | **2.9 MiB** | 1.5× |
| RDS+ | 2,560 | 10,240 | 1,000.0 KiB | **9.8 MiB** | 5.0× |
| LESS | 8,192 | 32,768 | 3.1 MiB | **31.2 MiB** | 16.0× |
| LoGra (r=8) | 16,128 | 64,512 | 6.2 MiB | **61.5 MiB** | 31.5× |
| **LoGra (r=32)** | 258,048 | 1,032,192 | 98.4 MiB | **984.4 MiB** | 504.0× |

Every ratio is exactly the width ratio, which is the point: dtype is held
constant, so nothing else can be doing the work. Storing at fp16 instead would
halve every row and leave the ratios unchanged.

r=32 is the rank the final (biggpu) run uses; r=8 is the historical `config.py`
profile, included because the gap between them is the largest single effect
in the table.

### Native dtypes (what each implementation actually writes)

| Method | native dtype | B/example | n=1,000 |
|---|---|---:|---:|
| InfluCoder (68m) | float32 | 2,048 | 2.0 MiB |
| InfluCoder (150m) | float32 | 3,072 | 2.9 MiB |
| RDS+ | float32 | 10,240 | 9.8 MiB |
| LESS | bfloat16 | 16,384 | 15.6 MiB |
| LoGra (r=8) | float32 | 64,512 | 61.5 MiB |
| LoGra (r=32) | float32 | 1,032,192 | 984.4 MiB |

## 5. Where each width comes from

- **InfluCoder** — `ettin-encoder-68m` hidden size 512 (`config.json`).
- **RDS+** — Qwen3-4B hidden size 2560.
- **LESS** — `cfg.LESS_PROJ_DIM = 8192`.
- **LoGra** — `modeling_logra.py:step()` concatenates each LoRA-B module's
  `[B, r, r]` gradient flattened to `[B, r²]`. Qwen3-4B has **252** target
  modules (36 layers × 7), so 252 × 32² = 258,048.

## 6. How each one scales

This matters more than the absolute numbers:

- **InfluCoder is the only method whose index size is independent of the model
  being attributed.** It is set by the student encoder alone. Attributing a 70B
  checkpoint instead of a 4B one leaves it at 2.0 MiB/1K.
- **RDS+** scales with target hidden size — linear in model width.
- **LESS** is independent of both target size *and* LoRA rank; `proj_dim` is a
  free knob, the only storage in this table that is a deliberate choice rather
  than a consequence of the architecture.
- **LoGra scales with target depth × rank².** r=8 → r=32 is 16× the storage,
  which is why it reaches ~1 GiB per 1K examples at the final run's rank.

## 7. Caveats

- **LESS's 16× is a config artifact, not a property of the method.** At
  `proj_dim=2048` it would sit at 7.8 MiB/1K. EXP1 already priced what that
  costs in quality (`config_biggpu.py`: agg rho +0.9144 at proj_dim=2048 vs
  +0.9766 at 8192, −0.062) — the storage buys real accuracy, but the trade
  exists.
- Raw tensor sizes. No quantization or compression anywhere; applying it
  uniformly shifts all rows without reordering them.
- The JSON also records `on_disk`, the real size of a `torch.save` artifact
  (tensor bytes plus pickle overhead), alongside in-memory `nbytes`.

## 8. Reproduce

```bash
# committed runs
python -m baselines.exp1.addendum_storage --n 1000 --mode real --target Qwen/Qwen3-4B
python -m baselines.exp1.addendum_storage --n 100  --mode real --target Qwen/Qwen3-4B

# analytic only -- widths from model configs, no weights downloaded, seconds to run
python -m baselines.exp1.addendum_storage --n 1000

# a different common dtype for the headline comparison
python -m baselines.exp1.addendum_storage --n 100 --mode real --store_dtype float16
```

`--mode real` needs a GPU: LESS's TRAK projector and LoGra's custom per-sample
backward are GPU-only in this codebase. `--mode analytic` derives every width
from model configs (LoGra's module count via a meta-device model) and needs
nothing.
