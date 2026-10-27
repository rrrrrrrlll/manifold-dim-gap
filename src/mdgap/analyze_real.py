"""
Analysis of experiments B, C1 and C2 as fixed in docs/analysis_plan_bc.md (all descriptive).

Inputs (each part is skipped, with a note, if its inputs are missing):
  B   results/real_pca_width/<dataset>_mse.csv                    (mdgap.train_real)
  C1  data/old_repo/results/gap_curve/gap_curve_matched_n/*.csv   (old repo, cross-entropy gap)
  C2  results/real_full/<dataset>_mse.csv                         (mdgap.train_real)
  dimension estimates: results/id_real/<method>/<label>.csv (mdgap.estimate_real_id) and, for
  reference, the old repo's estimates data/old_repo/results/id_estimation_runs/id_estimation/
  experiment A, for the comparison figure: results/clean_manifolds/torus_mse_analysis/beta_by_m.csv
  and results/id_calibration/mle/torus_m<m>.csv

Writes results/real_analysis/: summary.md, beta_b.csv, beta_c1.csv, beta_c2.csv, id_estimates.csv,
beta_vs_id.png.

    python -m mdgap.analyze_real
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from mdgap.analyze_decay_exponent import betas_from_cells
from mdgap.real_data import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("mle", "twonn", "corrint")
C_DATASETS = ["mnist", "fmnist", "pathmnist", "cifar10", "fer2013", "celeba_gb", "celeba_sb"]


# ---------- statistics ----------

def beta_table(df, group, ncol="n", gap="gen_gap_loss", n_boot=2000, seed=0):
    """beta_hat per group value (fit of log mean gap on log N) with a seed-bootstrap 95% interval."""
    groups = list(dict.fromkeys(df[group]))
    ns = sorted(df[ncol].unique())
    cells = {(g, n): v[gap].to_numpy(float) for (g, n), v in df.groupby([group, ncol])}
    means = {k: v.mean() for k, v in cells.items()}
    bh = betas_from_cells(means, np.array(groups, dtype=object), ns, "logn")
    rng = np.random.default_rng(seed)
    boot = np.array([betas_from_cells({k: rng.choice(v, len(v)).mean() for k, v in cells.items()},
                                      np.array(groups, dtype=object), ns, "logn") for _ in range(n_boot)])
    rows = []
    for i, g in enumerate(groups):
        excl = [n for n in ns if (g, n) in means and means[(g, n)] <= 0]
        b = boot[:, i][np.isfinite(boot[:, i])]
        lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) > 10 else (np.nan, np.nan))
        rows.append({group: g, "beta_hat": bh[i], "beta_lo95": lo, "beta_hi95": hi,
                     "sizes_used": sum(1 for n in ns if (g, n) in means and means[(g, n)] > 0),
                     "excluded_sizes": " ".join(str(n) for n in excl)})
    return pd.DataFrame(rows), boot


def spearman_exact(x, y):
    """Spearman rho and its exact two-sided permutation p-value (small n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, n
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rho = np.corrcoef(rx, ry)[0, 1]
    perms = [np.corrcoef(rx, ry[list(p)])[0, 1] for p in itertools.permutations(range(n))]
    p = float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))
    return float(rho), p, n


# ---------- inputs ----------

def read_id(root, label, method):
    f = root / method / f"{label}.csv"
    if not f.exists():
        return np.nan
    d = pd.read_csv(f)
    d = d[d.status == "ok"] if "status" in d else d
    full = d[d.fraction == d.fraction.max()]
    return float(full.estimated_id.mean()) if len(full) else np.nan


def id_estimates(results):
    rows = []
    for label in sorted({p.stem for p in (results / "id_real").glob("*/*.csv")}):
        rows.append({"label": label, **{m: read_id(results / "id_real", label, m) for m in METHODS}})
    return pd.DataFrame(rows, columns=["label", *METHODS])


def full_label(dataset):
    return f"{DATASETS[dataset][0]}_full"


def fmt(v, d=3):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def md_table(df, cols, digits=3):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rec = r._asdict()
        out.append("| " + " | ".join(fmt(rec[c], digits) if isinstance(rec[c], float) else str(rec[c]) for c in cols) + " |")
    return out


# ---------- parts ----------

def part_b(results, ids, lines, n_boot):
    tables = []
    for dataset in ("mnist", "cifar10"):
        f = results / "real_pca_width" / f"{dataset}_mse.csv"
        if not f.exists():
            lines += [f"- {dataset}: no results yet ({f.relative_to(REPO_ROOT)})"]
            continue
        d = pd.read_csv(f)
        d["width"] = d["width"].astype(int)
        t, boot = beta_table(d, "width", n_boot=n_boot)
        fset = DATASETS[dataset][0]
        for m in METHODS:
            t[f"id_{m}"] = [ids.set_index("label")[m].get(f"{fset}_pca{w}", np.nan) if len(ids) else np.nan for w in t.width]
        big = d[d.n == d.n.max()].groupby("width")
        t["test_acc_maxN"] = t.width.map(big.test_acc.mean())
        t["mean_gap_maxN"] = t.width.map(big.gen_gap_loss.mean())
        t["runs"] = t.width.map(d.groupby("width").size())
        t["not_plateaued_frac"] = t.width.map(d.groupby("width").plateau_rel_change.apply(lambda v: float((v.abs() > 0.02).mean())))
        t.insert(0, "dataset", dataset)
        tables.append(t)
        lines += [f"### {dataset}", ""] + md_table(t, ["width", "beta_hat", "beta_lo95", "beta_hi95", "id_mle", "id_twonn",
                                                     "id_corrint", "test_acc_maxN", "mean_gap_maxN", "runs",
                                                     "not_plateaued_frac"], digits=4) + [""]
        for m in METHODS:
            rho, p, k = spearman_exact(t.beta_hat, t[f"id_{m}"])
            lines.append(f"- Spearman rho(beta_hat, {m} dimension) over {k} widths: {fmt(rho)} (exact two-sided p {fmt(p)})")
        x = t["id_mle"].to_numpy(float)
        if np.isfinite(x).sum() >= 2:
            ok = np.isfinite(x)
            slope = np.polyfit(x[ok], t.beta_hat.to_numpy(float)[ok], 1)[0]
            bs = [np.polyfit(x[ok], b[ok], 1)[0] for b in boot if np.isfinite(b[ok]).all()]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append(f"- slope of beta_hat on the MLE dimension: {slope:+.4f} per dimension (95% bootstrap {lo:+.4f} to {hi:+.4f})")
        lines.append("")
    return pd.concat(tables) if tables else None


def part_c(df, label, ids, old_ids, lines, n_boot, ncol="n"):
    t, _ = beta_table(df, "dataset", ncol=ncol, n_boot=n_boot)
    for m in METHODS:
        t[f"id_{m}"] = [ids.set_index("label")[m].get(full_label(ds), np.nan) if len(ids) else np.nan for ds in t.dataset]
        t[f"old_id_{m}"] = [old_ids.get((ds, m), np.nan) for ds in t.dataset]
    lines += md_table(t, ["dataset", "beta_hat", "beta_lo95", "beta_hi95", "sizes_used", "excluded_sizes",
                          "id_mle", "id_twonn", "id_corrint", "old_id_mle"]) + [""]
    for prefix, what in (("id_", "dimension estimated on the features used"), ("old_id_", "old repo's estimate (other vectors)")):
        for m in METHODS:
            rho, p, k = spearman_exact(t.beta_hat, t[f"{prefix}{m}"])
            lines.append(f"- {label}: Spearman rho(beta_hat, {m}; {what}) over {k} datasets: {fmt(rho)} (exact two-sided p {fmt(p)})")
    lines.append("")
    return t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(REPO_ROOT / "results"))
    p.add_argument("--old", default=str(REPO_ROOT / "data" / "old_repo" / "results"))
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    results, old = Path(args.results), Path(args.old)
    out = results / "real_analysis"
    out.mkdir(parents=True, exist_ok=True)

    ids = id_estimates(results)
    ids.to_csv(out / "id_estimates.csv", index=False)
    old_ids = {(ds, m): read_id(old / "id_estimation_runs" / "id_estimation", ds, m) for ds in C_DATASETS for m in METHODS}
    lines = ["# Experiments B and C: analysis (docs/analysis_plan_bc.md; descriptive)", "",
             "Dimension estimates: MLE / TwoNN / correlation integral at the largest sample size, "
             "`results/id_real` (none yet)" if not len(ids) else
             "Dimension estimates: MLE / TwoNN / correlation integral at the largest sample size, `results/id_real`.", ""]

    lines += ["## Experiment B: PCA width", "", "Prediction: rho < 0 (higher dimension, slower decay).", ""]
    tb = part_b(results, ids, lines, args.n_boot)
    if tb is not None:
        tb.to_csv(out / "beta_b.csv", index=False)

    lines += ["## Experiment C1: old gap curves (cross-entropy gap, induced-subgraph protocol)", ""]
    files = sorted((old / "gap_curve" / "gap_curve_matched_n").glob("*.csv"))
    tc1 = None
    if files:
        d1 = pd.concat([pd.read_csv(f) for f in files])
        tc1 = part_c(d1, "C1", ids, old_ids, lines, args.n_boot, ncol="n_actual")
        tc1.to_csv(out / "beta_c1.csv", index=False)
    else:
        lines += ["- old gap curves not found", ""]

    lines += ["## Experiment C2: the 7 datasets rerun (MSE gap, per-sample graphs)", ""]
    f2 = [results / "real_full" / f"{ds}_mse.csv" for ds in C_DATASETS]
    tc2 = None
    if any(f.exists() for f in f2):
        d2 = pd.concat([pd.read_csv(f) for f in f2 if f.exists()])
        missing = [ds for ds, f in zip(C_DATASETS, f2) if not f.exists()]
        if missing:
            lines += [f"- no results yet for: {', '.join(missing)}", ""]
        tc2 = part_c(d2, "C2", ids, old_ids, lines, args.n_boot)
        conv = d2.groupby("dataset").plateau_rel_change.apply(lambda v: float((v.abs() > 0.02).mean()))
        lines += ["- share of runs whose training loss still changed by > 2% over the last 10% of epochs: " +
                  ", ".join(f"{k} {v:.2f}" for k, v in conv.items()), ""]
        tc2.to_csv(out / "beta_c2.csv", index=False)
    else:
        lines += ["- no results yet (results/real_full/)", ""]

    if tc1 is not None and tc2 is not None:
        both = tc1[["dataset", "beta_hat"]].merge(tc2[["dataset", "beta_hat"]], on="dataset", suffixes=("_c1", "_c2"))
        lines += ["## C1 vs C2 (same datasets, old vs new protocol)", ""] + md_table(both, ["dataset", "beta_hat_c1", "beta_hat_c2"]) + [""]

    # figure: beta vs MLE dimension, with experiment A's torus at its own MLE estimate
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    a_beta = results / "clean_manifolds" / "torus_mse_analysis" / "beta_by_m.csv"
    if a_beta.exists():
        b = pd.read_csv(a_beta)
        idm = [read_id(results / "id_calibration", f"torus_m{m}", "mle") for m in b.m]
        ax.plot(idm, b.beta_hat, "-o", color="#1f5fa8", lw=2, ms=4, label="A: torus (m = 2 ... 24)")
    for t, name, c, mk in ((tb[tb.dataset == "mnist"] if tb is not None else None, "B: MNIST, PCA widths", "#d9822b", "s"),
                           (tb[tb.dataset == "cifar10"] if tb is not None else None, "B: CIFAR-10, PCA widths", "#8a5cc2", "^"),
                           (tc2, "C2: 7 datasets", "#2e8b57", "D")):
        if t is not None and len(t) and t.id_mle.notna().any():
            ax.errorbar(t.id_mle, t.beta_hat, yerr=[t.beta_hat - t.beta_lo95, t.beta_hi95 - t.beta_hat], fmt=mk + ("-" if "B:" in name else ""),
                        color=c, ms=5, lw=1.5, capsize=2, label=name)
    ax.set_xlabel("estimated dimension (MLE, 16,000 points)")
    ax.set_ylabel("decay exponent of the MSE loss gap")
    ax.grid(color="#e3e6ea", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "beta_vs_id.png", dpi=170)

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
