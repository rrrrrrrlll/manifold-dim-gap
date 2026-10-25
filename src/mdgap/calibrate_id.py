"""
Estimator calibration at known dimension: run MLE, TwoNN and the
correlation integral on the same manifolds the GNNs are trained on, at each m, over
several sample sizes. Shows how far the estimators are from the true m in this
setting (ambient dimension 128); comparisons with real datasets, whose m can only
be estimated, rely on it.

Writes results/id_calibration/<method>/<manifold>_m<m>.csv (one file per method,
same format as the old repo's results/id_estimation). Labels already present are skipped.

    python -m mdgap.calibrate_id --config configs/clean_manifolds.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from mdgap.id_estimation import run_id_scale_sweep, write_id_results
from mdgap.manifolds import sample_manifold

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="a grid config: manifolds, m, ambient_dim, embed_seed, noise_sigma")
    p.add_argument("--n", type=int, default=16000, help="points per manifold (largest training-graph size)")
    p.add_argument("--fractions", default="0.0625,0.125,0.25,0.5,1.0")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345, help="sample seed, distinct from training seeds")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "id_calibration"))
    args = p.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    fractions = tuple(float(f) for f in args.fractions.split(","))
    for manifold in cfg["manifolds"]:
        for m in cfg["m"]:
            label = f"{manifold}_m{m}" + (f"_noise{cfg['noise_sigma']:g}" if cfg["noise_sigma"] else "")
            if all((out / meth / f"{label}.csv").exists() for meth in ("mle", "twonn", "corrint")):
                print(f"skip {label} (already done)")
                continue
            x = sample_manifold(manifold, m, args.n, seed=args.seed, stream=0, ambient_dim=cfg["ambient_dim"],
                                embed_seed=cfg["embed_seed"], noise_sigma=cfg["noise_sigma"]).x.astype(np.float64)
            records = run_id_scale_sweep(x, size_fractions=fractions, num_repeats=args.repeats,
                                         random_state=args.seed, label=label, true_d=m)
            write_id_results(records, out, label)


if __name__ == "__main__":
    main()
