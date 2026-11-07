#!/bin/bash
# Appendix: loss-weight (alpha) ablation and index-storage comparison. Run from the repository root.
set -euo pipefail
python -m experiments.agreement.part1_alpha_ablation   # default profile (as in the original launcher) -> results/fig1_dolci/default/alpha_ablation.json
python -m experiments.agreement.addendum_storage --n 1000 --mode real --target Qwen/Qwen3-4B --out results/addendum_storage_1k.json
python -m experiments.agreement.addendum_storage --n 100  --mode real --target Qwen/Qwen3-4B --out results/addendum_storage_n100.json
