# Sourced by run_pilot.sh and run_id_calibration.sh: conda env, repo root, logs/, and a
# DRY_RUN-aware flag. Run every job inside tmux/screen (or nohup) so closing the terminal
# does not stop it. Every python call skips runs already in its results CSV, so a killed
# job is resumed by re-launching the same command. DRY_RUN=1 lists what would run.
# (Experiment A has its own script, jobs/run_experiment_a.sh.)
set -e
CONDA_ENV=${CONDA_ENV:-image_gnn}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# src/ on the path, so the jobs work even without `pip install -e .`
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
DRY_FLAG=""
if [ -n "${DRY_RUN:-}" ]; then DRY_FLAG="--dry-run"; fi
