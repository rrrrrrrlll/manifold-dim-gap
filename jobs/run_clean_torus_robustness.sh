#!/bin/bash
# Robustness checks of the clean-manifold result (torus, MSE): noise 0.05 and k = 30 (280 runs).
# Usage: bash jobs/run_clean_torus_robustness.sh [gpu]
source "$(dirname "$0")/_job_setup.sh"
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
for cfg in configs/clean_torus_noise.yaml configs/clean_torus_k30.yaml; do
    python -m mdgap.train --config "$cfg" $DRY_FLAG 2>&1 | tee -a logs/clean_torus_robustness.log
done
