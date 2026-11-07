#!/bin/bash
cd /home/dkachler/year1/rebut_ie/influcoder/EXP2-datelm
export PYTHONPATH="/home/dkachler/year1/rebut_ie/influcoder/EXP2-datelm/_stubs:$PYTHONPATH"
LOG=/home/dkachler/year1/rebut_ie/influcoder/EXP2-datelm/_baseline_run.log
for M in Grad_Dot Grad_Sim DataInf EKFAC; do
  echo "########## START $M ##########" >> "$LOG"
  { /usr/bin/time -p ../.venv_py311/bin/python methods/run_timed_baseline.py --method "$M" ; } >> "$LOG" 2>&1
  echo "########## END $M ##########" >> "$LOG"
done
echo "===ALL_COMPLETE===" >> "$LOG"
