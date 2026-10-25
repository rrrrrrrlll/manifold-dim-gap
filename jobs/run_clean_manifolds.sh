#!/bin/bash
# Clean-manifold main grid for ONE manifold (both losses, 980 runs).
# Run the two manifolds at the same time on two GPUs:
#   bash jobs/run_clean_manifolds.sh torus 0
#   bash jobs/run_clean_manifolds.sh sphere 1
source "$(dirname "$0")/_job_setup.sh"
MANIFOLD=${1:?usage: run_clean_manifolds.sh <torus|sphere> [gpu]}
export CUDA_VISIBLE_DEVICES=${2:-${CUDA_VISIBLE_DEVICES:-0}}
python -m mdgap.train --config configs/clean_manifolds.yaml --manifold "$MANIFOLD" $DRY_FLAG 2>&1 | tee -a "logs/clean_manifolds_${MANIFOLD}.log"
