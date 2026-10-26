#!/bin/bash
# Experiment A (clean manifolds) on the GPU server: check the environment, launch
# the runs in tmux, watch progress, pack the results for download.
# See docs/running_experiment_a.md. Usage, from the repo folder on the server:
#
#   bash jobs/run_experiment_a.sh check         # env imports, CUDA, run counts, one timed GPU run
#   bash jobs/run_experiment_a.sh launch 0 1    # main grid + robustness checks on GPUs 0 and 1
#   bash jobs/run_experiment_a.sh status        # runs done per results file, last log lines
#   bash jobs/run_experiment_a.sh pack          # results + logs -> results_experiment_a.tar.gz
#   bash jobs/run_experiment_a.sh followups 0 1 # exploratory follow-ups (configs/followup_*.yaml)
#
# Everything stays inside the repo folder: results/, logs/ and the packed archive.
# Setting (optional): CONDA_ENV = conda env with torch + PyG (default image_gnn, the
# old repo's env, so no new env is created).
set -euo pipefail

SCRIPT=$(readlink -f "$0")
MDGAP_DIR=$(cd "$(dirname "$SCRIPT")/.." && pwd)   # the repo folder this script sits in
CONDA_ENV=${CONDA_ENV:-image_gnn}
export CONDA_ENV
export PYTHONWARNINGS=ignore   # hide torch's deprecation warnings in check/status output

die() { echo "error: $*" >&2; exit 1; }

activate_env() {
    command -v conda >/dev/null || die "conda not on PATH (run this from your normal login shell)"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || die "conda env '$CONDA_ENV' not found; set CONDA_ENV=<name>"
}

cmd_check() {
    cd "$MDGAP_DIR"
    activate_env
    export PYTHONPATH="$PWD/src"
    python - <<'EOF'
import importlib, sys
missing = []
for mod in ("torch", "torch_geometric", "numpy", "scipy", "pandas", "sklearn", "yaml"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:16s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        missing.append(f"{mod} ({e.__class__.__name__}: {e})")
if missing:
    sys.exit("missing in this env: " + "; ".join(missing) + "\ninstall with: pip install torch-geometric scipy pandas scikit-learn PyYAML")
import torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False: runs would silently use the CPU. Check the driver / torch CUDA build.")
print(f"  CUDA OK: {torch.cuda.device_count()} visible GPU(s), torch built for CUDA {torch.version.cuda}")
EOF
    echo "runs planned (skips rows already in results/):"
    for cfg in clean_manifolds clean_torus_noise clean_torus_k30; do
        echo "  $cfg: $(python -m mdgap.train --config "configs/$cfg.yaml" --dry-run | tail -1)"
    done
    echo "one full-length run (torus, m=2, N=1000, 2000 epochs) on GPU ${1:-0}, results not saved:"
    CUDA_VISIBLE_DEVICES=${1:-0} python - <<'EOF'
import yaml, torch
from mdgap.train import run_one
cfg = yaml.safe_load(open("configs/clean_manifolds.yaml"))
row = run_one(cfg, "torus", 2, 1000, 0, "mse", torch.device("cuda"))
print(f"  device {row['device']}, {row['runtime_s']:.1f} s, gap {row['gen_gap_loss']:.4f}, "
      f"test acc {row['test_acc']:.1f}%")
EOF
}

train() {   # config, log name, extra args: one mdgap.train call, output also appended to logs/<log name>.log
    local cfg=$1 log=$2; shift 2
    python -m mdgap.train --config "configs/$cfg.yaml" "$@" 2>&1 | tee -a "logs/$log.log"
}

cmd__run() {   # internal: what each tmux session runs
    cd "$MDGAP_DIR"
    activate_env
    export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="$2"
    mkdir -p logs
    local rc=0
    case "$1" in
        torus)  { train clean_manifolds clean_manifolds_torus --manifold torus \
                  && train clean_torus_noise clean_torus_robustness \
                  && train clean_torus_k30 clean_torus_robustness; } || rc=$? ;;
        sphere) train clean_manifolds clean_manifolds_sphere --manifold sphere || rc=$? ;;
        followup_a)  # larger N (slowest: kNN graphs of 32k-64k points), then the wider model
                { train followup_large_n followup_large_n \
                  && train followup_capacity_h128 followup_capacity_h128; } || rc=$? ;;
        followup_b)  # noise sweep
                { train followup_noise_0.005 followup_noise \
                  && train followup_noise_0.01 followup_noise \
                  && train followup_noise_0.02 followup_noise; } || rc=$? ;;
    esac
    echo; echo "[mdgap_$1 finished with exit code $rc; if it is not 0, run the same launch or followups command again to resume]"
    exec bash
}

launch_session() {   # manifold, gpu
    local name="mdgap_$1"
    if tmux has-session -t "$name" 2>/dev/null; then
        echo "tmux session $name already exists; attach with: tmux attach -t $name"
        return
    fi
    # pass PATH and settings explicitly: an already running tmux server does not see this shell's variables
    tmux new-session -d -s "$name" "env PATH='$PATH' CONDA_ENV='$CONDA_ENV' bash '$SCRIPT' _run $1 $2"
    echo "started tmux session $name on GPU $2"
}

cmd_launch() {
    cd "$MDGAP_DIR"
    command -v tmux >/dev/null || die "tmux not found"
    local g0=${1:-0} g1=${2:-${1:-0}}
    # One manifold per GPU (both losses, 980 runs each); the
    # robustness checks (280 runs) follow the torus grid on the first GPU.
    launch_session torus "$g0"
    launch_session sphere "$g1"
    echo "progress: bash jobs/run_experiment_a.sh status    watch live: tmux attach -t mdgap_torus  (leave: Ctrl-b, then d)"
    echo "if a session stops early, run this launch again: finished runs are skipped."
}

cmd_followups() {
    cd "$MDGAP_DIR"
    command -v tmux >/dev/null || die "tmux not found"
    local f
    for f in followup_large_n followup_capacity_h128 followup_noise_0.005 followup_noise_0.01 followup_noise_0.02; do
        [ -f "configs/$f.yaml" ] || die "configs/$f.yaml missing: upload it first (docs/running_experiment_a.md)"
    done
    local g0=${1:-0} g1=${2:-${1:-0}}
    # session A: larger N (60 runs) then hidden 128 (210 runs); session B: noise sweep (420 runs)
    launch_session followup_a "$g0"
    launch_session followup_b "$g1"
    echo "progress: bash jobs/run_experiment_a.sh status"
}

cmd_status() {
    cd "$MDGAP_DIR"
    local entry f total n
    for entry in clean_manifolds/torus_mse:490 clean_manifolds/torus_ce:490 clean_manifolds/sphere_mse:490 \
                 clean_manifolds/sphere_ce:490 clean_torus_noise/torus_mse:140 clean_torus_k30/torus_mse:140 \
                 followup_large_n/torus_mse:60 followup_capacity_h128/torus_mse:210 followup_noise_0.005/torus_mse:140 \
                 followup_noise_0.01/torus_mse:140 followup_noise_0.02/torus_mse:140; do
        f="results/${entry%:*}.csv"; total=${entry#*:}
        [[ $f == results/followup_* && ! -f $f ]] && continue   # follow-ups not started: not listed
        n=0; [ -f "$f" ] && n=$(( $(wc -l < "$f") - 1 ))
        printf "  %-46s %4d / %d runs\n" "$f" "$n" "$total"
    done
    for log in logs/*.log; do
        [ -f "$log" ] || continue
        echo "--- $log (last run lines)"
        grep -E '^(torus|sphere) m=' "$log" | tail -2 || true
    done
    tmux ls 2>/dev/null | grep mdgap || echo "  (no mdgap tmux sessions running)"
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    fi
}

cmd_pack() {
    cd "$MDGAP_DIR"
    local parts=()
    for d in results/clean_* results/followup_* logs; do
        [ -d "$d" ] && parts+=("$d")
    done
    [ ${#parts[@]} -gt 0 ] || die "no results yet"
    tar -czf results_experiment_a.tar.gz "${parts[@]}"
    echo "packed ${parts[*]} into $MDGAP_DIR/results_experiment_a.tar.gz ($(du -h results_experiment_a.tar.gz | cut -f1))"
}

case "${1:-}" in
    check)  shift; cmd_check "$@" ;;
    launch) shift; cmd_launch "$@" ;;
    status) cmd_status ;;
    pack)   cmd_pack ;;
    followups) shift; cmd_followups "$@" ;;
    _run)   shift; cmd__run "$@" ;;
    *) sed -n '2,14p' "$0"; exit 1 ;;
esac
