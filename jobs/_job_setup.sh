# Sourced by the job scripts: conda env, repo root, logs/, and a DRY_RUN-aware runner.
# Run every job inside tmux/screen (or nohup) so closing the terminal does not stop it.
# Every python call skips runs already in its results CSV, so a killed job is resumed
# by re-launching the same command. DRY_RUN=1 lists what would run without running it.
set -e
CONDA_ENV=${CONDA_ENV:-mdgap}
if [ -z "$DRY_RUN" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# src/ on the path, so the jobs work even without `pip install -e .`
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
DRY_FLAG=""
[ -n "$DRY_RUN" ] && DRY_FLAG="--dry-run"
