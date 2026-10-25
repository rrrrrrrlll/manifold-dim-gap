"""
One-time copy of the few files this repo needs from image-gnn-id-estimation-2.
Run it from anywhere; it only reads the old repo and writes into this repo's data/.

Exported now (needed to compare with the 7 real datasets and for the "real data on the
synthetic curve" figure):
  results/gap_curve/gap_curve_matched_n/*.csv        unregularized GNN gap curves (7 datasets)
  results/gap_curve/gap_curve_matched_n_proj/*.csv   same for GNNProj
  results/id_estimation_runs/id_estimation/{mle,twonn,corrint}/*.csv   data-ID estimates

Not exported yet (see docs/data_from_old_repo.md): image embeddings for the PCA-width
and re-run experiments, and the VAE checkpoint for the VAE-decoded-images rung -- which exact files to take still has to be confirmed.

Writes data/old_repo/<same relative path> and data/old_repo/MANIFEST.csv
(source path, size, SHA-256, old-repo commit), so every copied file is traceable.

    python scripts/export_from_old_repo.py --old-repo ../image-gnn-id-est-2
"""
import argparse
import csv
import hashlib
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PATTERNS = [
    "results/gap_curve/gap_curve_matched_n/*.csv",
    "results/gap_curve/gap_curve_matched_n_proj/*.csv",
    "results/id_estimation_runs/id_estimation/mle/*.csv",
    "results/id_estimation_runs/id_estimation/twonn/*.csv",
    "results/id_estimation_runs/id_estimation/corrint/*.csv",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-repo", required=True)
    p.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "data" / "old_repo"))
    args = p.parse_args()

    old, dest = Path(args.old_repo).resolve(), Path(args.dest)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=old, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    rows = []
    for pattern in PATTERNS:
        files = sorted(old.glob(pattern))
        if not files:
            raise SystemExit(f"nothing matches {pattern} under {old}")
        for src in files:
            rel = src.relative_to(old)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            rows.append({"path": rel.as_posix(), "bytes": src.stat().st_size, "sha256": sha256(src),
                         "old_repo_commit": commit, "copied_at": datetime.now().isoformat(timespec="seconds")})
    with open(dest / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"copied {len(rows)} files ({sum(r['bytes'] for r in rows) / 1e6:.2f} MB) to {dest}")


if __name__ == "__main__":
    main()
