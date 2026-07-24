# Findings

_Editable digest of what we currently believe, blunt and falsifiable. New results append or amend a
line. This is the shareable summary; per-run detail (every tuning round, every bug writeup) is in
`INFLUCODER_FINDINGS.md`._

## Core (Figure 1, Dolly pool)

- **InfluCoder dominates the cheap end of the cost/quality Pareto frontier.** At identical inference
  cost, distillation roughly doubles the untrained-encoder baseline at every size (68m +0.39→+0.78,
  150m +0.33→+0.76, 400m +0.43→+0.81 agg ρ vs. Qwen3-4B-rank16 GT). LESS (+0.96, 358ms) and LoGRA r8/4B
  (+0.90, 144ms) still lead on raw fidelity but at 35-275x the cost.
- **LoGRA proxy quality does not transfer uniformly across model size.** At the same rank as the 4B
  anchor (r=8): 1.7B proxy +0.29, 0.6B proxy **-0.16** (worse than random). Raising proxy rank to 32
  (asymmetric vs. the 4B's r=8) rescues the 1.7B proxy (+0.29→+0.51, plateaus by r16) but **not** the
  0.6B proxy (stays pinned near zero, -0.03 to +0.03, at every rank tested) — its failure is the 0.6B
  model's gradient geometry not correlating with the 4B's influence signal, not an adapter-capacity
  problem. FIM-preconditioned LoGRA is much worse than raw at every rank/proxy — use raw.
- **LoGRA rank barely moves inference cost.** The pinv/FIM term is ~0.02% of total FLOPs; the dominant
  cost is the rank-independent frozen-backbone forward+backward. An apparent 38% rank-4-vs-8 timing gap
  was a same-process OS-page-cache-warmup artifact, not a rank effect (isolated compute-only comparison:
  ~5%, noise-level).
- **Batching LoGRA is a real but unsafe lever — rejected for the default methodology.** Where it fits
  (proxies only; the 4B model already OOMs at batch=1, 21.9GB/46GB), batch=4 buys 7-18% speedup, but
  `modeling_logra.py` computes one batch-*mean* loss before backward, so a sample's "per-sample gradient"
  at batch>1 is diluted by whatever else shares its batch — verified real score drift up to **0.126**
  (1.7B) / **0.064** (0.6B), far past noise. `batch_size=1` stays the correctness standard; a batch=4
  override exists only as an explicit, flagged, user-requested variant (`update_proxy_rows_batched.py`).
- **SDPA attention is a free, verified win — adopted as default.** vs. eager: -30-35% memory, -10-28%
  time at batch=1, zero batching needed. Verified safe (eager-vs-sdpa Δagg = +0.0046, noise-level) before
  adopting. Also reveals a real, if narrow, size-proportional speed ordering (144/130/126ms across
  4B/1.7B/0.6B) that eager's memory-bound overhead had been masking.

## Instrument / measurement hygiene

- **LoGRA is unseeded upstream** (`kaiming_uniform_`, no seed on `logix_lora_A`/`_C`). Single draws drift
  ±0.03-0.14 depending on rank/variant, and at 4B/r8 specifically a 5-draw spread was [0.68, 0.92] (mean
  +0.8231, std 0.087) — **use a multi-seed mean as any target, never a single draw.**
- **Two silent-corruption bugs found and fixed this session, both classic "looks like a bad result, is
  actually a bug" traps:**
  1. GT/train-feature cache key was missing eval size — `fig1` (n_eval_a=400) silently cache-hit
     `paper200x3`'s cached gradients (same train size, different eval size), training an encoder against
     targets for the wrong underlying samples. Symptom: eval Spearman whipsawing despite smoothly falling
     loss. Fixed by adding eval size to the cache key.
  2. `load_encoder()` hardcoded a 512-token cap. Fine for Dolly (median ~120 encoder-text tokens), silently
     truncated dolci-instruct's much longer text (median ~400, p90 ~1000) — 24% of samples had <50% of
     their answer visible to the encoder, 5.6% had none, while the GT gradient target saw far more of the
     answer via a target-prioritized truncation scheme. Fixed via a per-preset `encoder_max_len` (1024 for
     dolci-instruct, matching `grad_max_len`; Dolly untouched at 512).
  3. (Smaller) `torch.tensor(rng.sample(remaining, 0))` defaults to float32 when `n_random=0`
     (`hard_ratio=1.0`), silently producing a float index tensor that crashes later at the actual
     indexing site — fixed with explicit `dtype=torch.long`.
- **General lesson:** sequential same-process model reloads across configs confound wall-clock timing via
  OS page-cache warmup — a clean cost comparison needs fresh processes or an explicit cache-warm step.

## InfluCoder training-tuning (13 rounds on Qwen3-4B GT, condensed)

- **The one big lever is train size, and it needs hard-negative mining to scale with it.** 500x1000 →
  1000x2000 was the single largest single-variable gain found (+0.052, >10x the next-best lever).
  1000x2000 and 1500x3000 land in the same ~0.76-0.77 band once seed noise is accounted for (2-seed
  means agree); 2000x4000 *regresses* unless hard_ratio also scales up (random negatives get diluted as
  pool size grows past ~a few dozen candidates).
- **Hard-negative mining has an interior optimum that shifts with pool size**, and is a hard cliff at
  the top: ratio=1.0 (pure hard negatives, no easy contrast left to calibrate against) collapses training
  to +0.47, far below every other config tested. At 1000x2000/1500x3000 the peak is ~0.5-0.7; at 2000x4000
  it needs ~0.75+.
- **Encoder size "bigger is worse" was a data-starvation artifact, not a real capacity finding.** At
  500x1000/no hard mining, 150m/400m were both worse than 68m (400m collapsed within 1 epoch). An LR-
  mismatch theory was tested and refuted (lower LR "fixed" 400m's instability but didn't make it
  competitive); more data did — at 1500x3000, 400m becomes the single best config found all session
  (+0.802 at 8 epochs, +0.802→+0.8017 clean gain from 16 epochs, ruled noise-sized and not adopted).
- **Everything else tested was flat-to-negative and not pursued:** grad_accum (1→4), cosine LR schedule,
  weight_decay=0, tighter grad-clip, m_candidates/k_anchors scaled up, and most single-variable LR/temp
  sweeps around the found optima. Alpha (listwise-KL vs. Pearson blend) has a small (~0.01-0.02, noise-
  adjacent) interior optimum around 0.2-0.3, replicated in direction (not magnitude) across 68m and 400m.
- **Ensembling reduces variance, it doesn't raise the ceiling.** 3-seed score-averaging beat the solo mean
  by +0.0098 but did not beat the single best individual seed.
- **Reference config at pause: 400m encoder, 1500x3000 train, lr=5e-5, epochs=8, hard_ratio=0,
  alpha=0.3 → agg ~+0.796-0.798.** Gap to the LoGRA r8/4B target (+0.8231, 5-draw mean): **~-0.025 to
  -0.027**, the closest result reached. Paused (not abandoned) to build Figure 1.

## Pool swap: Dolly → tasksource/dolci-instruct (everything else held fixed)

- **Headline: InfluCoder's distillation gain essentially vanishes on dolci-instruct, and the truncation
  bug above is NOT the explanation.** On Dolly, distillation adds +0.35 to +0.40 agg ρ over the untrained
  baseline at every size. On dolci-instruct (post-fix, 1024-token cap): trained sits *below* untrained at
  every size (68m +0.32 untrained vs. +0.05 trained; 150m +0.28 vs. +0.12; 400m +0.21 vs. +0.07). Fixing
  the truncation bug nearly doubled every untrained score (confirming it was real) but did not rescue
  trained scores — they stayed flat or got worse.
- **Root cause is overfitting, not truncation:** all three encoder sizes show eval agg ρ peaking at
  epoch 3/8 then degrading every subsequent epoch, despite training loss falling smoothly to near-zero —
  reproduces identically at both the buggy (512) and fixed (1024) max_len, ruling out truncation as the
  driver. Plausible cause (not investigated further): dolci-instruct is a heterogeneous SFT-mix (math,
  crystallography, moderation, multilingual in one flat prompt/answer pool) vs. Dolly's more uniform
  open-domain instruction shape — the bi-encoder may need different epoch selection or train-side
  sampling to generalize across that heterogeneity.
- **Secondary pool-swap effects:** both LoGRA proxies transfer *better* on dolci-instruct than Dolly
  (0.6B +0.03→+0.30, 1.7B +0.51→+0.58). TF-IDF goes **negative** (+0.30→-0.09) — lexical overlap is a much
  weaker influence signal on this heterogeneous pool. RDS+ drops (+0.31→+0.10).

## Why this session's full-vs-proxy LoGRA gap looks smaller than the original paper's

Two distinct, independently-evidenced mechanisms, either of which could explain a remembered
"300→200→100"-shaped taper not reproducing cleanly here:

1. **Quality axis — a deliberate methodology change closed part of the gap.** At a uniform rank (r=8,
   matching the anchor), the proxy quality cliff is stark: 4B +0.90 → 1.7B +0.29 → 0.6B **-0.16**. This
   session adopted an *asymmetric* rank (proxies at r=32, anchor stays r=8) specifically to stop
   penalizing smaller proxies twice — with that change, the taper becomes 4B +0.91 → 1.7B +0.58 → 0.6B
   +0.30 (dolci) — much smoother, but partly *because* the comparison got kinder to the proxies, not
   because the underlying capability gap disappeared.
2. **Cost axis — batch_size=1 is overhead-dominated, not FLOPs-dominated, in this size range.** ms/sample
   across 4B/1.7B/0.6B compresses to a ~1.2-1.5x spread (144/130/126ms under SDPA), not 3x+, because at
   batch=1 the 4B model is already pinned near the GPU memory ceiling and none of the three models can
   batch to amortize fixed per-sample overhead (kernel launch, autograd graph build/teardown). A paper
   timing setup where FLOPs actually dominate wall-clock (larger batches, aggregate-pass timing, or
   different hardware) would show a much more size-proportional, graded cost curve.
