# Goals

_Editable, high-level. The north star. Keep short; details live in `INFLUCODER_FINDINGS.md` and
`FINDINGS.md`._

## North star

Characterize the **cost-vs-quality tradeoff of gradient-influence data-selection methods** (LESS, LoGRA)
against cheap surrogates — distilled bi-encoders (InfluCoder) and smaller same-family LoGRA proxies — for
selecting training data by influence on a target task (BBH), and find out **whether a cheap surrogate can
match full-fidelity gradient influence at a fraction of the cost**, honestly (multi-seed, bug-checked, no
p-hacking) and across more than one candidate data pool.

## Working goals

1. **Maintain Figure 1** (cost vs. quality Pareto, every method against the same GT) as the main
   deliverable, reproducible across candidate pools via `--preset` (`fig1` = Dolly, `fig1_dolci` =
   dolci-instruct).
2. **Push InfluCoder to match or beat LoGRA r8/4B** on the `paper200` setup through training changes only
   (data size/composition, optimizer, loss, batching, encoder) — the GT eval definition and LoGRA's own
   hyperparameters stay fixed; the bar is a multi-seed LoGRA mean, not a single lucky draw.
3. **Test whether findings are pool-specific or general.** The dolci-instruct swap is the first check of
   this — does "distillation beats untrained-encoder baseline" hold outside Dolly? (Current answer: no —
   see FINDINGS.md.) Extend this check (LoGRA proxy rank sweep, uniform-vs-asymmetric rank tables) to
   dolci-instruct before generalizing any Dolly-only conclusion.
4. **Keep it honest.** Seed-verify surprising wins before calling them real; root-cause a concrete bug
   before rerunning anything, and stop once that bug is fixed rather than grinding hyperparameters hoping
   for a better-looking number. Document rejected levers, not just winners, so nobody re-treads a dead
   end.
5. **Keep cost measurement apples-to-apples.** `batch_size=1` and `attn_implementation=sdpa` are the
   default for every gradient method unless a deviation is explicitly requested and flagged (with its
   known correctness tradeoff stated, not silently absorbed into "the" methodology). FLOPs measurement is
   deferred to an appendix pass — ms/sample already tells the same ranking story at ~1/5th the cost to
   measure.

## Non-goals

- Chasing a "beat LoGRA" result via blind retries/hyperparameter grinding without a specific new
  hypothesis or a confirmed bug — that's p-hacking, not a finding.
- Silently folding a user-requested methodology override (e.g. batch>1 for LoGRA proxies) into the
  default/official numbers without flagging the tradeoff it was chosen despite.
- Moving the goalposts by changing the GT eval definition or LoGRA's own hyperparameters to make a
  surrogate look better.
