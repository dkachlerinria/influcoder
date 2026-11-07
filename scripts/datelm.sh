#!/bin/bash
# DATE-LM applications (Pythia-1B): Counterfact factual attribution (Recall@50, MRR) and
# XSTest-response toxicity filtering, heterogeneous / homogeneous (AUPRC).
# The toxicity runs need access to the gated HF dataset allenai/wildguardmix (accept its terms, then `hf auth login`).
# One method per call; results accumulate in experiments/datelm/results/fig2_*.json.
set -euo pipefail
cd experiments/datelm
METHODS=${METHODS:-"influcoder semantic bm25 repsim gradsim less datainf"}
for m in $METHODS; do
  python methods/run_fig2_counterfact.py   --method $m
  python methods/run_fig2_toxicity.py      --method $m
  python methods/run_fig2_toxicity_hom.py  --method $m
done
python methods/make_final_jsons.py      # -> exp2-{counter,het,hom}-final.json
