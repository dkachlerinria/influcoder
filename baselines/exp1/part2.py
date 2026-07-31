#!/usr/bin/env python3
"""EXP1 Part 2: InfluCoder Training-Sample Scaling.

How does InfluCoder's distillation quality scale with the number of training
samples, holding everything else fixed? Reuses `baselines.exp1.train`'s
canonical training call (same epochs/hard_ratio/lr/seed/epoch-selection
convention Part 1's checkpoints use -- previously these had drifted, see
EXP1.md section 4), and evaluates on the SAME eval slice Part 1 defaults to
(config.N_EVAL), so LESS/LoGRA reference lines can be recomputed FRESH here
(via `baselines.exp1.methods`) instead of borrowed from a different-eval-size
Part 1 run.

`SIZES` includes `config.N_TRAIN_A` (1500) -- the exact size Part 1's real
checkpoint is trained at -- so that sweep point is directly comparable to
(not just near) Part 1's own `influcoder_68m` row. If `baselines/out/<preset>/
exp1_part1.json` already exists, this script cross-checks that point against
it and prints the delta as a sanity check: since both now share the identical
training methodology and eval, they should agree closely (any gap indicates
GPU kernel non-determinism, not a methodology difference).

    python -m baselines.exp1.part2
    python -m baselines.exp1.part2 --sizes 25 100 500 1500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from influcoder.metrics import spearman_metrics

from . import config as cfg
from . import data, methods, train

OUT = Path("baselines/out") / cfg.PRESET / "exp1_part2.json"
PART1_OUT = Path("baselines/out") / cfg.PRESET / "exp1_part1.json"
DEFAULT_SIZES = [25, 50, 100, 250, 500, 750, 1000, cfg.N_TRAIN_A]


def run_size(splits, gt, eval_a, eval_p, g_ta, g_tp, n_a, n_p, encoder_model):
    targets = g_ta[:n_a] @ g_tp[:n_p].T
    train_a = [s.text for s in splits["train_anchors"][:n_a]]
    train_p = [s.text for s in splits["train_pool"][:n_p]]

    enc, log, final, untrained = train.train_influcoder(
        encoder_model, train_a, train_p, targets,
        eval_anchor_texts=eval_a, eval_pool_texts=eval_p, gt=gt)
    train.free(enc)
    return untrained, final


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_eval", type=int, default=cfg.N_EVAL)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                    help=f"n_train_anchors swept, pool is 2x each "
                        f"(default includes {cfg.N_TRAIN_A} = Part 1's own point)")
    ap.add_argument("--encoder_model", default=cfg.ENCODER_MODELS["68m"])
    args = ap.parse_args()

    if cfg.N_TRAIN_A not in args.sizes:
        print(f"WARNING: --sizes does not include {cfg.N_TRAIN_A} (Part 1's anchor "
              f"point) -- this sweep will not have a point directly comparable to "
              f"Part 1's influcoder_68m row.")

    gt, splits, preset_cfg = data.load_gt_and_splits(n_eval=args.n_eval)
    eval_a = [s.text for s in splits["eval_anchors"]]
    eval_p = [s.text for s in splits["eval_pool"]]

    max_a = max(args.sizes)
    if max_a > len(splits["train_anchors"]):
        raise SystemExit(f"size {max_a} > preset train anchors "
                         f"{len(splits['train_anchors'])}")

    g_ta, g_tp = data.load_train_features(splits, preset_cfg)
    print(f"EXP1 Part 2 | {args.n_eval}x{args.n_eval} eval | encoder={args.encoder_model} "
          f"| epochs={cfg.INFLUCODER_EPOCHS} hard_ratio={cfg.INFLUCODER_HARD_RATIO} "
          f"lr={cfg.INFLUCODER_LR} | sizes={args.sizes}\n")

    points = []
    if OUT.exists():
        points = json.loads(OUT.read_text()).get("points", [])
    done_sizes = {p["n_train_anchors"] for p in points}

    for n_a in args.sizes:
        if n_a in done_sizes:
            print(f"\nSkipping n_a={n_a}, already in JSON.")
            continue
        n_p = 2 * n_a
        print(f"\n########## n_a={n_a} n_p={n_p} (total={n_a + n_p}) ##########")
        t0 = time.time()
        untrained, final = run_size(splits, gt, eval_a, eval_p, g_ta, g_tp,
                                    n_a, n_p, args.encoder_model)
        elapsed = time.time() - t0
        pt = {"n_train_anchors": n_a, "n_train_pool": n_p,
             "n_train_total": n_a + n_p, "untrained_agg": untrained["aggregated"],
             "agg": final["aggregated"], "per_anchor_mean": final["per_anchor_mean"],
             "elapsed_s": round(elapsed, 1)}
        points.append(pt)
        print(f"-> n_a={n_a:5d} n_p={n_p:5d} total={n_a + n_p:6d}: "
              f"untrained {untrained['aggregated']:+.4f} -> "
              f"agg {final['aggregated']:+.4f} (final epoch {cfg.INFLUCODER_EPOCHS}, "
              f"{elapsed:.0f}s)")

        if n_a == cfg.N_TRAIN_A and PART1_OUT.exists():
            p1 = json.loads(PART1_OUT.read_text())
            row = p1["methods"].get(f"influcoder_{args.encoder_model.split('-')[-1]}")
            if row is not None:
                delta = final["aggregated"] - row["aggregated"]
                print(f"   [sanity check] Part 1's matching row: agg={row['aggregated']:+.4f} "
                      f"(this run: {final['aggregated']:+.4f}, delta={delta:+.4f}) -- "
                      f"same n_train/eval/recipe now, so this should be small")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "config": {"preset": cfg.PRESET, "n_eval": args.n_eval,
                      "encoder_model": args.encoder_model, "gt_model": cfg.GT_MODEL,
                      "gt_lora_rank": cfg.GT_LORA_RANK, "epochs": cfg.INFLUCODER_EPOCHS,
                      "hard_ratio": cfg.INFLUCODER_HARD_RATIO, "lr": cfg.INFLUCODER_LR,
                      "seed": cfg.INFLUCODER_SEED, "ratio": "1:2",
                      "anchor_point_n_train_a": cfg.N_TRAIN_A},
            "points": points,
        }, indent=2))

    # Fresh LESS/LoGRA reference lines, SAME eval slice as this sweep -- fixes
    # the previous version's mismatched-eval-size reference lines (EXP1.md
    # 4.2.5). Skipped if already computed at this exact n_eval (these take
    # several minutes each; don't redo them on every incremental --sizes resume).
    payload = json.loads(OUT.read_text())
    if payload.get("reference_lines_n_eval") == args.n_eval and "reference_lines" in payload:
        print(f"\nLESS/LoGRA reference lines already computed at n_eval={args.n_eval}, skipping.")
    else:
        print("\n########## LESS/LoGRA reference lines (same eval as this sweep) ##########")
        refs = {}
        for label, model_name in [("less_4B", cfg.GT_MODEL), ("less_1.7B", "Qwen/Qwen3-1.7B")]:
            scores = methods.run_less(splits, model_name)
            agg = spearman_metrics(scores.numpy(), gt)["aggregated"]
            refs[label] = agg
            print(f"  {label}: {agg:+.4f}")
        for label, model_name in [("logra_4B", cfg.GT_MODEL), ("logra_1.7B", "Qwen/Qwen3-1.7B")]:
            scores = methods.run_logra(splits, model_name)
            agg = spearman_metrics(scores.numpy(), gt)["aggregated"]
            refs[label] = agg
            print(f"  {label}: {agg:+.4f}")

        payload["reference_lines"] = refs
        payload["reference_lines_n_eval"] = args.n_eval
        OUT.write_text(json.dumps(payload, indent=2))

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
