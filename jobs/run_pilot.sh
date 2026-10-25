#!/bin/bash
# Pilot (24 runs). Afterwards, set optim.epochs and seeds in configs/clean_*.yaml
# from results/pilot/ (trace = where training loss plateaus; runtime_s).
# Usage: bash jobs/run_pilot.sh [gpu]
source "$(dirname "$0")/_job_setup.sh"
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
python -m mdgap.train --config configs/pilot.yaml $DRY_FLAG 2>&1 | tee -a logs/pilot.log
