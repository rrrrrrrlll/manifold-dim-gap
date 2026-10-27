"""
Dimension estimates of the real embeddings used in experiments B and C (docs/analysis_plan_bc.md).

For every feature set in the config(s) (and every PCA width for experiment B): draws 16,000 pool
points with a fixed seed, removes exact duplicate rows, and runs MLE (k = 20), TwoNN and the
correlation integral over the same sample-size sweep as the synthetic calibration
(mdgap.calibrate_id). Writes results/id_real/<method>/<label>.csv, label = <features set>_<width>
(e.g. mnist_pca4, cifar10_full, celeba_full). Labels already present are skipped. CPU only; needs
scikit-dimension.

    python -m mdgap.estimate_real_id --config configs/real_pca_width.yaml configs/real_full.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from mdgap.id_estimation import deduplicate_embeddings, run_id_scale_sweep, write_id_results
from mdgap.real_data import DATASETS, EmbeddingPool

REPO_ROOT = Path(__file__).resolve().parents[2]


def id_label(dataset, features, width):
    fset = DATASETS[dataset][0]
    return f"{fset}_{'full' if features == 'full' else f'pca{width}'}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--n", type=int, default=16000, help="pool points per estimate (largest training-graph size)")
    p.add_argument("--fractions", default="0.0625,0.125,0.25,0.5,1.0")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345, help="sample seed, distinct from training seeds")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "real"))
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "id_real"))
    args = p.parse_args(argv)
    out = Path(args.out)
    fractions = tuple(float(f) for f in args.fractions.split(","))

    done = set()
    for cfg_path in args.config:
        cfg = yaml.safe_load(open(cfg_path))
        for dataset in cfg["datasets"]:
            for width in cfg.get("widths") or [None]:
                label = id_label(dataset, cfg["features"], width)
                if label in done:          # both CelebA tasks share their features
                    continue
                done.add(label)
                if all((out / meth / f"{label}.csv").exists() for meth in ("mle", "twonn", "corrint")):
                    print(f"skip {label} (already done)")
                    continue
                pool = EmbeddingPool(dataset, cfg["features"], width, args.data_dir)
                rng = np.random.default_rng([args.seed, pool.code])   # same images for every width
                idx = np.sort(rng.choice(pool.size, size=min(args.n, pool.size), replace=False))
                x, dropped = deduplicate_embeddings(pool.x[idx].astype(np.float64))
                print(f"{label}: {len(x)} points ({dropped} duplicates dropped), {pool.dim} dims", flush=True)
                records = run_id_scale_sweep(x, size_fractions=fractions, num_repeats=args.repeats,
                                             random_state=args.seed, label=label, true_d=None)
                write_id_results(records, out, label)


if __name__ == "__main__":
    main()
