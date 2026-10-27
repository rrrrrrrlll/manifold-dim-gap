"""
One-time preparation of the real embeddings for experiments B and C (docs/analysis_plan_bc.md).

Reads, from the old repo, the features its GNN was trained on:
  data/<Dataset>Embeddings/<dir>/<name>_smsl_train_full_embeddings.npy
(every image of the dataset: training split, then test split; last two columns = label, split flag)
and writes into data/real/ of this repo:
  <features>_x.npy       float32 features (all columns but the last two)
  <dataset>_y.npy        int64 labels re-indexed to 0 .. C-1
  <features>_pca128.npy  float32 scores on the top 128 principal components (MNIST, CIFAR-10 only;
                         PCA fitted on the whole pool, no labels; component signs fixed so that the
                         largest-magnitude loading is positive)
  MANIFEST.csv           source file, its SHA-256, the old repo's commit, outputs and their SHA-256
  summary.txt            pool sizes, widths, classes, duplicate rows, PCA explained variance

Works in chunks (needs about 1 GB of memory for CIFAR-10's 4096-d features).

    python scripts/prepare_real_embeddings.py --old-repo ../image-gnn-id-est-2
"""
import argparse
import csv
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

SOURCES = {   # feature set -> (embedding folder, sub folder, file stem, [(dataset, label source sub folder)])
    "mnist": ("MNISTEmbeddings", "smsl_embeddings", "mnist", [("mnist", "smsl_embeddings")]),
    "fmnist": ("FMNISTEmbeddings", "smsl_embeddings", "fmnist", [("fmnist", "smsl_embeddings")]),
    "pathmnist": ("PathMNISTEmbeddings", "smsl_embeddings", "pathmnist", [("pathmnist", "smsl_embeddings")]),
    "cifar10": ("CIFAR10Embeddings", "smsl_embeddings", "cifar10", [("cifar10", "smsl_embeddings")]),
    "fer2013": ("FER2013Embeddings", "smsl_embeddings", "fer2013", [("fer2013", "smsl_embeddings")]),
    "celeba": ("CELEBAEmbeddings", "gb_smsl_embeddings", "celeba",
               [("celeba_gb", "gb_smsl_embeddings"), ("celeba_sb", "sb_smsl_embeddings")]),
}
PCA_SETS = ("mnist", "cifar10")
PCA_WIDTH = 128
CHUNK = 4000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def source_path(old, folder, sub, stem):
    return old / "data" / folder / sub / f"{stem}_smsl_train_full_embeddings.npy"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-repo", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "real"))
    p.add_argument("--only", help="comma-separated feature sets (default: all)")
    args = p.parse_args(argv)
    old, out = Path(args.old_repo).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        old_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=old, capture_output=True, text=True).stdout.strip()
    except Exception:
        old_commit = ""
    sets = args.only.split(",") if args.only else list(SOURCES)
    manifest, summary = [], [f"prepared {datetime.now().isoformat(timespec='seconds')} from {old} (commit {old_commit})", ""]

    for fset in sets:
        folder, sub, stem, label_sets = SOURCES[fset]
        src = source_path(old, folder, sub, stem)
        a = np.load(src, mmap_mode="r")
        n, d = a.shape[0], a.shape[1] - 2
        x_path = out / f"{fset}_x.npy"
        x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n, d))
        row_hashes = set()
        dup = 0
        for i in range(0, n, CHUNK):
            block = np.asarray(a[i:i + CHUNK, :d], dtype=np.float32)
            x[i:i + CHUNK] = block
            for r in block:
                h = hashlib.blake2b(r.tobytes(), digest_size=16).digest()
                dup += h in row_hashes
                row_hashes.add(h)
        x.flush()
        del x, row_hashes
        manifest.append({"output": x_path.name, "source": str(src.relative_to(old)), "source_sha256": sha256(src),
                         "old_repo_commit": old_commit, "output_sha256": sha256(x_path)})
        summary.append(f"{fset}: pool {n}, width {d}, duplicate rows {dup}")

        for dataset, lsub in label_sets:
            lsrc = source_path(old, folder, lsub, stem)
            b = np.load(lsrc, mmap_mode="r")
            if b.shape != a.shape:
                raise SystemExit(f"{dataset}: label source {lsrc} has shape {b.shape}, features {a.shape}")
            if lsrc != src:   # CelebA sb: the features must be the gb features
                for i in range(0, n, CHUNK):
                    if not np.array_equal(np.asarray(a[i:i + CHUNK, :d]), np.asarray(b[i:i + CHUNK, :d])):
                        raise SystemExit(f"{dataset}: features differ from {src}")
            raw = np.asarray(b[:, d])
            classes, y = np.unique(raw, return_inverse=True)
            y_path = out / f"{dataset}_y.npy"
            np.save(y_path, y.astype(np.int64))
            manifest.append({"output": y_path.name, "source": str(lsrc.relative_to(old)), "source_sha256": sha256(lsrc),
                             "old_repo_commit": old_commit, "output_sha256": sha256(y_path)})
            counts = np.bincount(y)
            summary.append(f"  {dataset}: {len(classes)} classes {classes.tolist()} -> 0..{len(classes) - 1}, "
                           f"counts {counts.tolist()}")

        if fset in PCA_SETS:
            xm = np.load(x_path, mmap_mode="r")
            mean = np.zeros(d)
            for i in range(0, n, CHUNK):
                mean += np.asarray(xm[i:i + CHUNK], dtype=np.float64).sum(0)
            mean /= n
            cov = np.zeros((d, d))
            for i in range(0, n, CHUNK):
                c = np.asarray(xm[i:i + CHUNK], dtype=np.float64) - mean
                cov += c.T @ c
            cov /= n - 1
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals, evecs = evals[order], evecs[:, order]
            width = min(PCA_WIDTH, d)
            comps = evecs[:, :width]
            signs = np.sign(comps[np.abs(comps).argmax(axis=0), np.arange(width)])
            comps = comps * signs
            pca_path = out / f"{fset}_pca{PCA_WIDTH}.npy"
            z = np.lib.format.open_memmap(pca_path, mode="w+", dtype=np.float32, shape=(n, width))
            for i in range(0, n, CHUNK):
                z[i:i + CHUNK] = (np.asarray(xm[i:i + CHUNK], dtype=np.float64) - mean) @ comps
            z.flush()
            del z, xm
            evr = np.cumsum(evals) / evals.sum()
            manifest.append({"output": pca_path.name, "source": x_path.name, "source_sha256": sha256(x_path),
                             "old_repo_commit": old_commit, "output_sha256": sha256(pca_path)})
            summary.append("  PCA cumulative explained variance: " +
                           ", ".join(f"r={r}: {evr[r - 1]:.3f}" for r in (4, 8, 16, 32, 64, 128) if r <= d))
        print("\n".join(s for s in summary if s.startswith(fset) or s.startswith("  ")), flush=True)

    mpath = out / "MANIFEST.csv"
    rows = []
    if mpath.exists():   # keep entries of feature sets not redone in this call
        with open(mpath, newline="") as f:
            keep = {r["output"] for r in manifest}
            rows = [r for r in csv.DictReader(f) if r["output"] not in keep]
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["output", "source", "source_sha256", "old_repo_commit", "output_sha256"])
        w.writeheader()
        w.writerows(rows + manifest)
    with open(out / "summary.txt", "a") as f:
        f.write("\n".join(summary) + "\n\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
