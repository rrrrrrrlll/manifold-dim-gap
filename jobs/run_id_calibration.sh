#!/bin/bash
# ID-estimator calibration at known m (CPU only, no GPU needed).
# Usage: bash jobs/run_id_calibration.sh
source "$(dirname "$0")/_job_setup.sh"
if [ -n "$DRY_RUN" ]; then echo "would run: python -m mdgap.calibrate_id --config configs/clean_manifolds.yaml"; exit 0; fi
python -m mdgap.calibrate_id --config configs/clean_manifolds.yaml 2>&1 | tee -a logs/id_calibration.log
