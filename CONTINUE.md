# Continue

_A "pick up here" snapshot for the next session. State only — the reasoning and evidence live in
`FINDINGS.md` (what we believe), `GOALS.md` (why we're doing this), `TODO.md` (what's queued)._

## Where things stand

Figure 1 (cost vs. quality Pareto vs. Qwen3-4B-rank16 GT) is built and believed-correct on two pools:
**Dolly** and **tasksource/dolci-instruct**. Both runs use the same adopted default methodology:
`attn_implementation=sdpa`, `batch_size=1` for every gradient method (LESS/LoGRA), asymmetric LoGRA
proxy rank (anchor r=8, proxies r=32). Two silent-corruption bugs (a GT/train-feature cache-key
collision, and a 512-token encoder truncation cap that was silently starving dolci-instruct's longer
samples) were found and fixed this session — both are believed fully resolved, not worked around.

InfluCoder (the distilled bi-encoder surrogate) training was paused mid-sweep, not abandoned, to go
build Figure 1 on the second pool. Best result so far: 400m encoder / 1500x3000 train / lr=5e-5 /
hard_ratio=0 → agg ρ ≈ **+0.796-0.798**, ~0.025-0.027 short of the LoGRA r8/4B 5-draw mean target
(+0.8231). 13 tuning rounds' worth of what worked/didn't is condensed in `FINDINGS.md`.

The codebase (69 files, 2 follow-up commits, the dolci-instruct pool support, SDPA adoption, and the
new FINDINGS/GOALS/TODO digest) is all **committed and pushed** to `origin/main` at
`https://github.com/dkachlerinria/influcoder`. Nothing outstanding locally as of this writing.

## What's true right now (headline claims, see FINDINGS.md for evidence)

- InfluCoder distillation roughly doubles the untrained-encoder baseline on Dolly, but **that gain
  essentially vanishes on dolci-instruct** — trained sits at or below untrained at every encoder size,
  root-caused to overfitting (eval ρ peaks epoch 3/8, then degrades), not the truncation bug.
- LoGRA proxy quality does not transfer uniformly by size: 1.7B genuinely benefits from extra rank
  (plateaus by r16), 0.6B never does (its gradient geometry just doesn't correlate with the 4B target).
- The full-vs-proxy LoGRA gap looking smaller here than in the original paper has two independent,
  evidenced causes: (1) the asymmetric-rank methodology choice was deliberately kinder to proxies on
  the quality axis, and (2) at the enforced `batch_size=1` standard, wall-clock is overhead-dominated
  rather than FLOPs-dominated in this size range, compressing what would otherwise be a more
  parameter-proportional cost curve down to a ~1.2-1.5x spread instead of ~3x.

## Immediate next step

Resume the InfluCoder tuning sweep at the 400m/1500x3000/lr=5e-5/hard_ratio=0 reference config —
remaining untested axes are lr7e5, wd002, wd005, clip2, warmup02, cosine schedule, then a second-seed
check on whatever wins. Full queue, in priority order, is in `TODO.md`.

## Gotchas for whoever picks this up

- Don't re-test anything in FINDINGS.md's "What doesn't work" list (grad_accum, wider batch
  composition, weight_decay/grad_clip/temperature deviations, hard_ratio=1.0, more epochs, cosine
  schedule) — all confirmed dead on two encoder sizes each.
- LoGRA is unseeded upstream; always use a multi-seed mean as a target, never a single draw.
- FLOPs measurement is deferred on purpose (5x wall-clock cost, same ranking story as ms/sample) —
  don't turn it back on except for a camera-ready appendix pass.
- Sequential same-process model reloads across configs confound wall-clock timing via OS page-cache
  warmup — time fresh processes, or warm the cache explicitly first.
