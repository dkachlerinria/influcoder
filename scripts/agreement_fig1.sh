#!/bin/bash
# Figure 1 (agreement with gradient influence): Qwen3-4B rank-16 LoRA gradient-cosine ground truth,
# BBH anchors x dolci-instruct pool, 800x800 eval, 3 seeds.
#   Part 1: quality vs. inference cost of every method      -> results/fig1_dolci/biggpu/seed*/exp1_part1.json
#   Part 2: InfluCoder quality vs. #distillation samples     -> .../exp1_part2.json
#   Part 3: amortized GPU time vs. #samples processed        -> .../exp1_part3_*.json
#   then:   merge + plot                                     -> .../EXP1_results.json, figure1.{pdf,png}
# Needs one large GPU (>= 80 GB recommended). Run from the repository root.
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
for s in $SEEDS; do
  export EXP1_CONFIG=biggpu EXP1_SEED=$s EXP1_INFLUCODER_SEED=$s EXP1_DATA_SEED=$s
  python -m experiments.agreement.part1
  python -m experiments.agreement.part2
  python -m experiments.agreement.part3
  python -m experiments.agreement.collect_results
  python -m experiments.agreement.plot_figure1
done
