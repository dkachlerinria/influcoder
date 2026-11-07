#!/bin/bash
cd /home/dkachler/year1/rebut_ie/influcoder
source .venv_h100/bin/activate
PYTHONUNBUFFERED=1 python -m baselines.exp1.part1_alpha_ablation
