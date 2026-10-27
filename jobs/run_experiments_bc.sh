#!/bin/bash
# Experiments B and C2 (real image embeddings) on the GPU server: check, launch in tmux,
# watch progress, pack the results. Plan: docs/analysis_plan_bc.md.
# Usage, from the repo folder on the server:
#
#   bash jobs/run_experiments_bc.sh check         # env, CUDA, data files, run counts, two timed GPU runs
#   bash jobs/run_experiments_bc.sh launch 0 1    # experiment B on GPU 0, experiment C2 on GPU 1
#   bash jobs/run_experiments_bc.sh status        # runs done per results file, last log lines
#   bash jobs/run_experiments_bc.sh pack          # results + logs -> results_experiments_bc.tar.gz
#
# Everything stays inside the repo folder: data/real/, results/, logs/ and the packed archive.
# Setting (optional): CONDA_ENV = conda env with torch + PyG (default image_gnn).
set -euo pipefail

SCRIPT=$(readlink -f "$0")
MDGAP_DIR=$(cd "$(dirname "$SCRIPT")/.." && pwd)   # the repo folder this script sits in
CONDA_ENV=${CONDA_ENV:-image_gnn}
export CONDA_ENV
export PYTHONWARNINGS=ignore   # hide torch's deprecation warnings in check/status output

B_FILES="mnist_pca128.npy mnist_y.npy cifar10_pca128.npy cifar10_y.npy"
C2_FILES="mnist_x.npy mnist_y.npy fmnist_x.npy fmnist_y.npy pathmnist_x.npy pathmnist_y.npy cifar10_x.npy cifar10_y.npy fer2013_x.npy fer2013_y.npy celeba_x.npy celeba_gb_y.npy celeba_sb_y.npy"

die() { echo "error: $*" >&2; exit 1; }

activate_env() {
    command -v conda >/dev/null || die "conda not on PATH (run this from your normal login shell)"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || die "conda env '$CONDA_ENV' not found; set CONDA_ENV=<name>"
}

missing_files() {   # list of file names -> those not in data/real
    local f out=""
    for f in $1; do [ -f "data/real/$f" ] || out="$out $f"; done
    echo "$out"
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
    sys.exit("missing in this env: " + "; ".join(missing))
import torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False: runs would silently use the CPU.")
print(f"  CUDA OK: {torch.cuda.device_count()} visible GPU(s), torch built for CUDA {torch.version.cuda}")
EOF
    local mb mc
    mb=$(missing_files "$B_FILES"); mc=$(missing_files "$C2_FILES")
    [ -z "$mb" ] && echo "  data for B: OK" || echo "  data for B missing:$mb"
    [ -z "$mc" ] && echo "  data for C2: OK" || echo "  data for C2 missing:$mc"
    echo "runs planned (skips rows already in results/):"
    echo "  B  (real_pca_width): $(python -m mdgap.train_real --config configs/real_pca_width.yaml --dry-run | tail -1)"
    echo "  C2 (real_full):      $(python -m mdgap.train_real --config configs/real_full.yaml --dry-run | tail -1)"
    [ -z "$mb" ] && [ -z "$mc" ] || die "upload the missing data files (data/real/) first"
    echo "two full-length runs on GPU ${1:-0} (N = 1,000, 2,000 epochs), results not saved:"
    CUDA_VISIBLE_DEVICES=${1:-0} python - <<'EOF'
import yaml, torch
from mdgap.real_data import EmbeddingPool
from mdgap.train_real import run_one
for cfg_path, dataset, features, width in (("configs/real_pca_width.yaml", "mnist", "pca128", 16),
                                           ("configs/real_full.yaml", "cifar10", "full", None)):
    cfg = yaml.safe_load(open(cfg_path))
    pool = EmbeddingPool(dataset, features, width, "data/real")
    row = run_one(cfg, pool, 1000, 0, "mse", torch.device("cuda"))
    print(f"  {dataset} ({pool.dim}-d): device {row['device']}, {row['runtime_s']:.1f} s, "
          f"gap {row['gen_gap_loss']:.4f}, test acc {row['test_acc']:.1f}%")
EOF
}

train() {   # config, log name: one mdgap.train_real call, output also appended to logs/<log name>.log
    python -m mdgap.train_real --config "configs/$1.yaml" 2>&1 | tee -a "logs/$2.log"
}

cmd__run() {   # internal: what each tmux session runs
    cd "$MDGAP_DIR"
    activate_env
    export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="$2"
    mkdir -p logs
    local rc=0
    case "$1" in
        b)  train real_pca_width real_pca_width || rc=$? ;;
        c2) train real_full real_full || rc=$? ;;
    esac
    echo; echo "[mdgap_$1 finished with exit code $rc; if it is not 0, run the same launch command again to resume]"
    exec bash
}

launch_session() {   # name, gpu
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
    [ -z "$(missing_files "$B_FILES $C2_FILES")" ] || die "data files missing in data/real: run check"
    local g0=${1:-0} g1=${2:-${1:-0}}
    launch_session b "$g0"    # experiment B: 700 runs
    launch_session c2 "$g1"   # experiment C2: 490 runs (CIFAR-10's 4096-d graphs are the slow part)
    echo "progress: bash jobs/run_experiments_bc.sh status    watch live: tmux attach -t mdgap_b  (leave: Ctrl-b, then d)"
    echo "if a session stops early, run this launch again: finished runs are skipped."
}

cmd_status() {
    cd "$MDGAP_DIR"
    local entry f total n
    for entry in real_pca_width/mnist_mse:350 real_pca_width/cifar10_mse:350 \
                 real_full/mnist_mse:70 real_full/fmnist_mse:70 real_full/pathmnist_mse:70 real_full/cifar10_mse:70 \
                 real_full/fer2013_mse:70 real_full/celeba_gb_mse:70 real_full/celeba_sb_mse:70; do
        f="results/${entry%:*}.csv"; total=${entry#*:}
        n=0; [ -f "$f" ] && n=$(( $(wc -l < "$f") - 1 ))
        printf "  %-40s %4d / %d runs\n" "$f" "$n" "$total"
    done
    for log in logs/real_pca_width.log logs/real_full.log; do
        [ -f "$log" ] || continue
        echo "--- $log (last run lines)"
        grep -E '^[a-z0-9_]+ width=' "$log" | tail -2 || true
    done
    tmux ls 2>/dev/null | grep -E 'mdgap_(b|c2):' || echo "  (no mdgap_b / mdgap_c2 tmux sessions running)"
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    fi
}

cmd_pack() {
    cd "$MDGAP_DIR"
    local parts=()
    for d in results/real_pca_width results/real_full logs/real_pca_width.log logs/real_full.log; do
        [ -e "$d" ] && parts+=("$d")
    done
    [ ${#parts[@]} -gt 0 ] || die "no results yet"
    tar -czf results_experiments_bc.tar.gz "${parts[@]}"
    echo "packed ${parts[*]} into $MDGAP_DIR/results_experiments_bc.tar.gz ($(du -h results_experiments_bc.tar.gz | cut -f1))"
}

case "${1:-}" in
    check)  shift; cmd_check "$@" ;;
    launch) shift; cmd_launch "$@" ;;
    status) cmd_status ;;
    pack)   cmd_pack ;;
    _run)   shift; cmd__run "$@" ;;
    *) sed -n '2,12p' "$0"; exit 1 ;;
esac
