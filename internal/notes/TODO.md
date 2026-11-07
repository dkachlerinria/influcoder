# TODO

_Light, text-only. Claim an item by appending `[claim <who>]` before you start; check it off `- [x]`
when done and add a one-line result. Per-run detail for anything referenced here lives in
`INFLUCODER_FINDINGS.md`._

## In flight / paused

- [ ] Resume InfluCoder tuning sweep (paused to build Figure 1) — remaining: lr7e5, wd002, wd005, clip2,
      warmup02, cosine schedule at the 400m/1500x3000/lr=5e-5/hard_ratio=0 reference config, then a
      second-seed verification of whatever wins. Current best: agg ~+0.796-0.798, gap to LoGRA r8/4B
      target (+0.8231, 5-draw mean) ~-0.025 to -0.027.
- [ ] Explain InfluCoder's overfitting collapse on dolci-instruct (eval ρ peaks epoch 3/8 then degrades
      at every encoder size, reproduces at both max_len settings — not a truncation artifact). Next real
      lever: the distillation recipe itself (lr, hard-negative mining, regularization) or whether
      dolci-instruct's task heterogeneity makes the gradient-influence targets inherently noisier/less
      learnable — not another silent-bug hunt.
- [ ] FLOPs appendix experiment (deferred all session) — `FlopCounterMode` 5x's wall-clock and told the
      same ranking story as ms/sample every time it was checked; only worth doing for a camera-ready
      appendix, not blocking anything now.

## Queued / next

- [ ] Rerun the LoGRA proxy rank sweep (r8/r16/r32) on dolci-instruct — currently only measured on Dolly;
      the "asymmetric rank rescues 1.7B but not 0.6B" finding needs a second pool to generalize.
- [ ] Build a uniform-rank (r=8 for every LoGRA row, no asymmetric bump) Figure-1 variant alongside the
      current adopted one, so a reader can directly compare the "fair capacity" framing vs. the "adopted/
      generous-to-proxies" framing — this is the most direct data-backed answer to "why is the full-vs-
      proxy gap smaller here than in the original paper."
- [ ] Investigate whether a task-type-stratified train/eval split or per-task-type epoch selection helps
      InfluCoder generalize across dolci-instruct's heterogeneous SFT mix.

## Task-design / repo follow-ups

- [ ] Push the local-only commits (69-file checkpoint + 2 dolci-instruct commits) to `origin/main` —
      not done yet, only committed locally.

## Done

- [x] Figure 1 (Dolly pool) — LESS/LoGRA(r8,4B)/LoGRA proxies(1.7B,0.6B)/InfluCoder(68/150/400m)/untrained
      encoders/RDS+/TF-IDF, all vs. Qwen3-4B rank-16 GT, SDPA + batch_size=1 methodology.
- [x] LoGRA proxy rank sweep (Dolly) — 1.7B genuinely benefits from higher rank (plateaus by r16); 0.6B
      does not (stuck near zero at every rank tested); FIM variant much worse than raw everywhere.
- [x] LoGRA batching investigation — real (7-18%) but unsafe speedup (batch-mean loss dilutes per-sample
      gradient, up to 0.126 score drift); rejected for the default methodology, batch_size=1 stays
      standard; batch=4 kept only as an explicit flagged override for the two proxy rows.
- [x] SDPA adoption — verified safe (Δagg = +0.0046 vs. eager), -30-35% memory/-10-28% time at batch=1;
      adopted as the new default attention implementation, all LoGRA rows re-timed.
- [x] Figure 1 rebuilt on dolci-instruct pool — found and fixed a real encoder `max_seq_len` truncation
      bug (512→1024 tokens); confirmed "distillation gain vanishes on dolci-instruct" survives the fix
      (not a truncation artifact).
- [x] GT/train-feature cache-key collision bug — `fig1` was silently reusing `paper200x3`'s cached
      gradients (same train size, unkeyed eval size) → fixed by adding eval size to the cache key.
- [x] `hard_ratio=1.0` crash bug — `torch.tensor([])` defaulting to float32 corrupting an index tensor →
      fixed with explicit `dtype=torch.long`.
- [x] Saved/committed the InfluCoder codebase (69 files + 2 follow-up commits, local only so far).
