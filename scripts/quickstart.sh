#!/bin/bash
# Smallest end-to-end InfluCoder run (SmolLM2-135M teacher, ettin-68m encoder, BBH x Dolly), ~1 min on one GPU.
set -euo pipefail
python run.py --preset sanity     # includes the CountSketch projection-fidelity check
python run.py --preset tiny       # small but non-degenerate distillation
