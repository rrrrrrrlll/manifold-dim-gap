"""
Primary analysis of the synthetic-manifold grids, exactly as fixed in docs/analysis_plan.md.

For one (manifold, loss) results file:
  1. gap of a run = mean test loss over the fresh graphs minus the training loss
     (column gen_gap_loss);
  2. average over seeds within each cell (m, N); cells with a mean gap <= 0 are
     excluded from the fit and listed;
  3. for each m, least-squares slope of log(mean gap) on log N  ->  beta_hat(m) = -slope
     (needs >= 3 usable sizes, otherwise beta_hat(m) is missing);
  4. weighted least-squares slope b1 of beta_hat(m) on m, weights = 1 / bootstrap
     variance of each beta_hat(m);
  5. uncertainty: resample seeds within every cell (B replicates) and repeat 2-4.

Decision rule (H1), with SE = bootstrap SD of b1 * sqrt(s/(s-1)), s = fewest seeds in any
cell used, and t = 95th percentile of Student's t with (number of m with a beta_hat - 2) df:
  * support                : one-sided 95% upper bound b1 + t * SE < 0;
  * evidence of no effect  : 90% interval (b1 +- t * SE) * (m_max - m_min) inside +-margin, where
                             margin = 0.25 * (1/(m_min+4) - 1/(m_max+4))  (0.033 for m = 2..24);
  * otherwise              : inconclusive.
Why not the plain percentile bound: in the pilot-based null simulation it claimed support
in 9-10% of runs instead of 5%, because resampling s seeds understates the variance of a cell
mean by (s-1)/s and only ~7 values of m enter the slope (docs/analysis_plan.md, Power).
Also reported, not used for the decision: the percentile bound, and a bound from the scatter
of beta_hat(m) around the fitted line (residual SE), which also reflects curvature in m.
Also reports H2 (beta_hat(m) - 1/(m+4), descriptive) and H3 (fit beta = a/(m+c), exploratory).

    python -m mdgap.analyze_decay_exponent --results results/clean_manifolds/torus_mse.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def x_transform(n, kind):
    n = np.asarray(n, float)
    if kind == "logn":
        return np.log(n)
    if kind == "n_over_logn":
        return np.log(n / np.log(n))
    raise ValueError(kind)


def betas_from_cells(cell_means, ms, ns, xkind, min_sizes=3):
    """cell_means: dict (m, n) -> mean gap. Returns array of beta_hat per m (nan if < min_sizes usable)."""
    out = np.full(len(ms), np.nan)
    for i, m in enumerate(ms):
        pts = [(n, cell_means[(m, n)]) for n in ns if (m, n) in cell_means and cell_means[(m, n)] > 0]
        if len(pts) >= min_sizes:
            x = x_transform([p[0] for p in pts], xkind)
            y = np.log([p[1] for p in pts])
            out[i] = -np.polyfit(x, y, 1)[0]
    return out


def wls_slope(x, y, w):
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    xm = np.sum(w * x) / np.sum(w)
    ym = np.sum(w * y) / np.sum(w)
    return np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)


def analyze(df, n_boot=2000, xkind="logn", seed=0):
    df = df.copy()
    ms = np.array(sorted(df.m.unique()))
    ns = sorted(df.n.unique())
    groups = {key: g.gen_gap_loss.to_numpy(float) for key, g in df.groupby(["m", "n"])}
    point_means = {k: v.mean() for k, v in groups.items()}
    beta_hat = betas_from_cells(point_means, ms, ns, xkind)

    rng = np.random.default_rng(seed)
    boot_betas = np.empty((n_boot, len(ms)))
    for b in range(n_boot):
        means = {k: rng.choice(v, size=len(v), replace=True).mean() for k, v in groups.items()}
        boot_betas[b] = betas_from_cells(means, ms, ns, xkind)
    with np.errstate(all="ignore"):
        var = np.nanvar(boot_betas, axis=0, ddof=1) if np.isfinite(boot_betas).any() else np.full(len(ms), np.nan)
    weights = 1.0 / var

    b1_hat = wls_slope(ms.astype(float), beta_hat, weights)
    b1_boot = np.array([wls_slope(ms.astype(float), bb, weights) for bb in boot_betas])
    b1_boot = b1_boot[np.isfinite(b1_boot)]

    m_min, m_max = ms.min(), ms.max()
    margin = 0.25 * (1 / (m_min + 4) - 1 / (m_max + 4))
    # seeds in the cells that enter the fit (positive mean gap, at an m with a beta_hat)
    fit_m = set(ms[np.isfinite(beta_hat)])
    used = [len(v) for k, v in groups.items() if k[0] in fit_m and point_means[k] > 0]
    s_min = min(used) if used else 0
    df_t = int(np.isfinite(beta_hat).sum()) - 2
    se_boot = se = t_crit = upper95 = lo90 = hi90 = upper95_pct = upper95_res = np.nan
    if len(b1_boot) < 2 or not np.isfinite(b1_hat) or s_min < 2 or df_t < 1:
        decision = ("not estimable: need beta_hat at >= 3 values of m, each from >= 3 sizes N "
                    "with a positive mean gap, and >= 2 seeds per cell")
    else:
        se_boot = float(np.std(b1_boot, ddof=1))
        se = se_boot * np.sqrt(s_min / (s_min - 1))
        t_crit = float(stats.t.ppf(0.95, df_t))
        upper95 = b1_hat + t_crit * se
        lo90, hi90 = (b1_hat - t_crit * se) * (m_max - m_min), (b1_hat + t_crit * se) * (m_max - m_min)
        upper95_pct = float(np.percentile(b1_boot, 95))
        # sensitivity: SE from the weighted scatter of beta_hat around the fitted line
        ok = np.isfinite(beta_hat) & np.isfinite(weights) & (weights > 0)
        x, y, w = ms[ok].astype(float), beta_hat[ok], weights[ok]
        xm = np.sum(w * x) / np.sum(w)
        resid = y - (np.sum(w * y) / np.sum(w) + b1_hat * (x - xm))
        se_res = np.sqrt(np.sum(w * resid ** 2) / df_t / np.sum(w * (x - xm) ** 2))
        upper95_res = b1_hat + t_crit * se_res
    if not np.isfinite(upper95):
        pass
    elif upper95 < 0:
        decision = "support for H1 (one-sided 95% upper bound of b1, b1 + t * SE, < 0)"
    elif lo90 > -margin and hi90 < margin:
        decision = f"evidence of no dependence (90% interval of the total change inside +-{margin:.3f})"
    else:
        decision = "inconclusive"

    with np.errstate(all="ignore"):
        lo_b, hi_b = (np.nanpercentile(boot_betas, q, axis=0) if np.isfinite(boot_betas).any() else np.full(len(ms), np.nan)
                      for q in (2.5, 97.5))
    per_m = pd.DataFrame({
        "m": ms, "beta_hat": beta_hat,
        "beta_lo95": lo_b,
        "beta_hi95": hi_b,
        "bound_rate_1_over_m_plus_4": 1 / (ms + 4),
    })
    per_m["beta_minus_bound_rate"] = per_m.beta_hat - per_m.bound_rate_1_over_m_plus_4
    cells = pd.DataFrame([{"m": m, "n": n, "n_seeds": len(groups[(m, n)]), "mean_gap": point_means[(m, n)],
                           "excluded_nonpositive": point_means[(m, n)] <= 0} for (m, n) in sorted(groups)])

    # H3 (exploratory): beta = a / (m + c)
    h3 = {"a": np.nan, "c": np.nan}
    try:
        from scipy.optimize import curve_fit
        ok = np.isfinite(beta_hat)
        (a, c), _ = curve_fit(lambda m, a, c: a / (m + c), ms[ok].astype(float), beta_hat[ok], p0=(1.0, 4.0),
                              bounds=([0, -1.9], [np.inf, 1e3]), maxfev=10000)
        h3 = {"a": a, "c": c}
    except Exception as e:
        h3["error"] = str(e)

    summary = {"b1_hat": b1_hat, "b1_upper95_one_sided": upper95,
               "b1_se_boot": se_boot, "b1_se": se, "seeds_min": s_min, "t_df": df_t, "t_crit": t_crit,
               "b1_upper95_percentile": upper95_pct, "b1_upper95_residual_se": upper95_res,
               "total_change_lo90": lo90, "total_change_hi90": hi90, "margin": margin,
               "n_boot_valid": len(b1_boot), "decision": decision, "x_transform": xkind,
               "cells_excluded": int(cells.excluded_nonpositive.sum()), **{f"h3_{k}": v for k, v in h3.items()}}
    return summary, per_m, cells


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="results CSV for one (manifold, loss)")
    p.add_argument("--out", help="output folder (default: <results stem>_analysis next to the CSV)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--x", choices=["logn", "n_over_logn"], default="logn")
    args = p.parse_args(argv)

    res = Path(args.results)
    df = pd.read_csv(res)
    out = Path(args.out) if args.out else res.parent / f"{res.stem}_analysis{'' if args.x == 'logn' else '_' + args.x}"
    out.mkdir(parents=True, exist_ok=True)

    summary, per_m, cells = analyze(df, n_boot=args.n_boot, xkind=args.x)
    per_m.to_csv(out / "beta_by_m.csv", index=False)
    cells.to_csv(out / "cells.csv", index=False)

    lines = [f"# Decay-exponent analysis: {res.name}", "",
             f"Seeds per cell: {cells.n_seeds.min()}-{cells.n_seeds.max()}; bootstrap replicates: {summary['n_boot_valid']}; "
             f"x = {args.x}.", "",
             f"**Decision (H1): {summary['decision']}.**", "",
             f"- b1 (slope of beta_hat on m) = {summary['b1_hat']:+.5f}, SE {summary['b1_se']:.5f} "
             f"(bootstrap SD {summary['b1_se_boot']:.5f} x sqrt(s/(s-1)), s = {summary['seeds_min']}), "
             f"t({summary['t_df']}) = {summary['t_crit']:.3f}; one-sided 95% upper bound {summary['b1_upper95_one_sided']:+.5f}",
             f"- not used for the decision: percentile upper bound {summary['b1_upper95_percentile']:+.5f}; "
             f"residual-SE upper bound {summary['b1_upper95_residual_se']:+.5f}",
             f"- total change over m: 90% interval [{summary['total_change_lo90']:+.4f}, {summary['total_change_hi90']:+.4f}], "
             f"equivalence margin +-{summary['margin']:.4f}",
             f"- cells excluded (mean gap <= 0): {summary['cells_excluded']}",
             f"- H3 (exploratory) beta = a/(m+c): a = {summary.get('h3_a', np.nan):.3f}, c = {summary.get('h3_c', np.nan):.2f}", "",
             "| m | beta_hat | 95% interval | 1/(m+4) | beta_hat - 1/(m+4) |", "|---|---|---|---|---|"]
    for r in per_m.itertuples():
        lines.append(f"| {r.m} | {r.beta_hat:.4f} | [{r.beta_lo95:.4f}, {r.beta_hi95:.4f}] | {r.bound_rate_1_over_m_plus_4:.4f} | {r.beta_minus_bound_rate:+.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.errorbar(per_m.m, per_m.beta_hat, yerr=[per_m.beta_hat - per_m.beta_lo95, per_m.beta_hi95 - per_m.beta_hat],
                fmt="o-", color="#2a78d6", capsize=3, label="fitted decay exponent (95% bootstrap)")
    ax.plot(per_m.m, per_m.bound_rate_1_over_m_plus_4, "--", color="#888", label="bound's rate 1/(m+4)")
    ax.set_xlabel("manifold dimension m")
    ax.set_ylabel("decay exponent of the loss gap")
    ax.grid(color="#e3e6ea", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "beta_vs_m.png", dpi=160)
    print("\n".join(lines[:9]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
