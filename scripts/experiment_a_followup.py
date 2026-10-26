"""
Follow-up analyses of experiment A, beyond the pre-specified test in
mdgap.analyze_decay_exponent (run that first: the beta_vs_m figure reads its beta_by_m.csv).

Prints: power-law fit quality and local decay exponents per m; gap, accuracy, loss and graph
tables; seed spread; the robustness contrasts beta(4) - beta(16) with bootstrap CIs (the b1 test
needs >= 3 values of m, the robustness arms have 2); the accuracy-gap analysis; one-sided
p-values for Holm; the H3 fits. Writes the figures in docs/figures/experiment_a/.

    python scripts/experiment_a_followup.py > results/experiment_a_followup.txt
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from mdgap.analyze_decay_exponent import analyze, betas_from_cells
R = "results/"
tm = pd.read_csv(R+"clean_manifolds/torus_mse.csv"); sm = pd.read_csv(R+"clean_manifolds/sphere_mse.csv")
tc = pd.read_csv(R+"clean_manifolds/torus_ce.csv"); sc = pd.read_csv(R+"clean_manifolds/sphere_ce.csv")
nz = pd.read_csv(R+"clean_torus_noise/torus_mse.csv"); k30 = pd.read_csv(R+"clean_torus_k30/torus_mse.csv")
pd.set_option("display.width", 200)

print("== (a) log-log fit quality per m (cell means): R2, slope small N (250-2000) vs large N (2000-16000)")
for name, d in (("torus_mse", tm), ("sphere_mse", sm), ("torus_ce", tc), ("sphere_ce", sc)):
    g = d.groupby(["m", "n"]).gen_gap_loss.mean().reset_index()
    rows = []
    for m, h in g.groupby("m"):
        x, y = np.log(h.n.values), np.log(h.gen_gap_loss.values)
        b, a = np.polyfit(x, y, 1); r2 = 1 - np.sum((y - (a + b*x))**2) / np.sum((y - y.mean())**2)
        lo = h.n <= 2000; hi = h.n >= 2000
        rows.append((m, -b, r2, -np.polyfit(x[lo], y[lo], 1)[0], -np.polyfit(x[hi], y[hi], 1)[0]))
    print(name); print(pd.DataFrame(rows, columns=["m", "beta_all", "R2", "beta_N<=2000", "beta_N>=2000"]).round(3).to_string(index=False))

print("\n== (b) mean gap by cell, torus MSE (rows m, cols N)")
print(tm.pivot_table(index="m", columns="n", values="gen_gap_loss").to_string(float_format=lambda v: f"{v:.5f}"))
print("\n== test accuracy (%) torus MSE"); print(tm.pivot_table(index="m", columns="n", values="test_acc").round(1).to_string())
print("\n== train accuracy (%) torus MSE"); print(tm.pivot_table(index="m", columns="n", values="train_acc").round(1).to_string())
print("\n== train / test loss, torus MSE, N=250 and 16000")
print(tm[tm.n.isin([250, 16000])].groupby(["m", "n"])[["train_loss", "test_loss"]].mean().round(4).unstack().to_string())

print("\n== (c) graph stats by m (torus MSE, N=1000 and 16000)")
print(tm[tm.n.isin([1000, 16000])].groupby(["m", "n"])[["mean_degree_train", "isolated_frac_train", "sigma_train"]].mean().round(4).unstack().to_string())
print("noise arm vs clean, sigma (median kNN distance), N=1000 / 16000:")
for nm, d in (("clean", tm[tm.m.isin([4, 16])]), ("noise0.05", nz), ("k30", k30)):
    print(" ", nm, d[d.n.isin([1000, 16000])].groupby(["m", "n"]).sigma_train.mean().round(4).to_dict())
print("noise arm accuracy:"); print(nz.pivot_table(index="m", columns="n", values="test_acc").round(1).to_string())
print("noise arm mean gap:"); print(nz.pivot_table(index="m", columns="n", values="gen_gap_loss").to_string(float_format=lambda v: f"{v:.4f}"))

print("\n== (d) seed SD of log gap per cell (torus MSE): ")
sd = tm[tm.gen_gap_loss > 0].groupby(["m", "n"]).gen_gap_loss.apply(lambda v: np.std(np.log(v), ddof=1)).unstack()
print(sd.round(3).to_string())

print("\n== (e) robustness arms: beta(4) - beta(16) with bootstrap 95% CI")
def dbeta(d, B=2000, seed=0):
    groups = {k: g.gen_gap_loss.to_numpy() for k, g in d.groupby(["m", "n"])}
    ms = np.array([4, 16]); ns = sorted(d.n.unique())
    pm = {k: v.mean() for k, v in groups.items()}
    b = betas_from_cells(pm, ms, ns, "logn"); rng = np.random.default_rng(seed); bb = []
    for _ in range(B):
        mm = {k: rng.choice(v, len(v)).mean() for k, v in groups.items()}
        bb.append(betas_from_cells(mm, ms, ns, "logn"))
    bb = np.array(bb); diff = bb[:, 0] - bb[:, 1]
    return b, b[0] - b[1], np.percentile(diff, [2.5, 97.5]), np.std(diff, ddof=1)
for nm, d in (("clean torus (same m, k=10)", tm[tm.m.isin([4, 16])]), ("noise 0.05", nz), ("k=30", k30)):
    b, dd, ci, s = dbeta(d)
    print(f"  {nm:28s} beta(4)={b[0]:.3f} beta(16)={b[1]:.3f}  diff={dd:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

print("\n== (f) accuracy gap (train acc - test acc, pp) through the same analysis")
for nm, d in (("torus_mse runs", tm), ("sphere_mse runs", sm)):
    dd = d.copy(); dd["gen_gap_loss"] = dd.gen_gap_acc
    s_, pm_, cells_ = analyze(dd, n_boot=2000)
    print(f"  {nm}: {s_['decision']}; b1={s_['b1_hat']:+.4f} upper={s_['b1_upper95_one_sided']:+.4f}; excluded cells {s_['cells_excluded']}")
    print("   beta_hat:", dict(zip(pm_.m, pm_.beta_hat.round(3))))
print("acc gap by cell, torus mse:"); print(tm.pivot_table(index="m", columns="n", values="gen_gap_acc").round(2).to_string())

print("\n== (g) one-sided p-values (t, df=5) and Holm over the estimable secondary analyses")
res = {}
for nm, d, kw in (("torus_mse (primary)", tm, {}), ("sphere_mse", sm, {}), ("torus_ce", tc, {}), ("sphere_ce", sc, {}),
                  ("torus_mse n/logn", tm, {"xkind": "n_over_logn"})):
    s_, _, _ = analyze(d, n_boot=2000, **kw)
    res[nm] = stats.t.cdf(s_["b1_hat"] / s_["b1_se"], s_["t_df"]); print(f"  {nm:22s} t={s_['b1_hat']/s_['b1_se']:7.1f}  p={res[nm]:.2e}")

print("\n== (h) H3: beta = a/(m+c) free c vs c fixed at 4")
from scipy.optimize import curve_fit
s_, pm_, _ = analyze(tm, n_boot=200)
m_ = pm_.m.values.astype(float); y_ = pm_.beta_hat.values
(a, c), _ = curve_fit(lambda m, a, c: a/(m+c), m_, y_, p0=(1, 4)); r_free = y_ - a/(m_+c)
a4, _ = curve_fit(lambda m, a: a/(m+4), m_, y_, p0=(1,)); r4 = y_ - a4[0]/(m_+4)
lin = np.polyfit(m_, y_, 1); rl = y_ - np.polyval(lin, m_)
print(f"  free: a={a:.2f} c={c:.2f} rms={np.sqrt(np.mean(r_free**2)):.4f};  c=4: a={a4[0]:.2f} rms={np.sqrt(np.mean(r4**2)):.4f};  linear: slope={lin[0]:.4f} rms={np.sqrt(np.mean(rl**2)):.4f}")
print("  ratio beta(2)/beta(24): observed %.2f, (m+4) form %.2f" % (y_[0]/y_[-1], 28/6))

# ---- local decay exponents ----
print("\n== (i) local decay exponents")
g = tm.groupby(["m","n"]).gen_gap_loss.mean().unstack()
ns = np.array(g.columns, float)
print("local slope -dlog g/dlog N between consecutive N (torus MSE):")
loc = pd.DataFrame({f"{int(a)}-{int(b)}": -(np.log(g[b])-np.log(g[a]))/np.log(b/a) for a,b in zip(g.columns[:-1], g.columns[1:])})
print(loc.round(2).to_string())
print("\nlocal slope at matched gap level (interpolated in log gap):")
mid = {m: (np.sqrt(g.loc[m].values[:-1]*g.loc[m].values[1:]), loc.loc[m].values) for m in g.index}
for lev in (0.06, 0.03, 0.015, 0.008):
    row=[]
    for m,(gm,sl) in mid.items():
        x=np.log(gm[::-1]); y=sl[::-1]
        row.append(f"{m}:{np.interp(np.log(lev),x,y):.2f}" if x.min()<=np.log(lev)<=x.max() else f"{m}:  - ")
    print(f"  gap {lev}: "+"  ".join(row))

# ---- figures ----
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "docs/figures/experiment_a/"
__import__("pathlib").Path(OUT).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e6ea", "grid.linewidth": 0.6, "axes.edgecolor": "#888", "legend.frameon": False})
MS = [2, 4, 6, 8, 12, 16, 24]
cmap = plt.get_cmap("Blues"); col = {m: cmap(0.35 + 0.6 * i / 6) for i, m in enumerate(MS)}
tm = pd.read_csv("results/clean_manifolds/torus_mse.csv"); sm = pd.read_csv("results/clean_manifolds/sphere_mse.csv")
nz = pd.read_csv("results/clean_torus_noise/torus_mse.csv"); k30 = pd.read_csv("results/clean_torus_k30/torus_mse.csv")

# 1. gap vs N
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (name, d) in zip(axes, (("Torus $T^m$", tm), ("Sphere $S^m$", sm))):
    g = d.groupby(["m", "n"]).gen_gap_loss.agg(["mean", "sem"]).reset_index()
    for m in MS:
        h = g[g.m == m]
        ax.errorbar(h.n, h["mean"], yerr=h["sem"], color=col[m], lw=2, marker="o", ms=4, capsize=0, label=f"m = {m}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{name}, MSE loss gap", fontsize=10)
    ax.set_xlabel("graph size N (log scale)")
axes[0].set_ylabel("generalization gap (test - train loss)")
axes[1].legend(loc="lower left", fontsize=8, ncol=2)
fig.suptitle("Loss gap vs graph size, mean over 10 seeds (bars: standard error)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "gap_vs_n.png", dpi=170); plt.close(fig)

# 2. beta vs m
fig, ax = plt.subplots(figsize=(5.4, 3.8))
for name, path, c, mk in (("torus", "results/clean_manifolds/torus_mse_analysis/beta_by_m.csv", "#1f5fa8", "o"),
                          ("sphere", "results/clean_manifolds/sphere_mse_analysis/beta_by_m.csv", "#d9822b", "s")):
    b = pd.read_csv(path)
    ax.errorbar(b.m, b.beta_hat, yerr=[b.beta_hat - b.beta_lo95, b.beta_hi95 - b.beta_hat], color=c, marker=mk, ms=5, lw=2, capsize=3,
                label=f"{name}: fitted decay exponent (95% CI)")
mm = np.linspace(2, 24, 100)
ax.plot(mm, 1 / (mm + 4), "--", color="#666", lw=1.5, label="bound's rate 1/(m+4)")
ax.set_xlabel("manifold dimension m"); ax.set_ylabel("decay exponent of the MSE loss gap")
ax.set_ylim(0, 1.45); ax.legend(fontsize=8, loc="upper right")
ax.set_title("Fitted exponent over N = 250-16,000 falls with m", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "beta_vs_m.png", dpi=170); plt.close(fig)

# 3. local slope
fig, ax = plt.subplots(figsize=(5.4, 3.8))
g = tm.groupby(["m", "n"]).gen_gap_loss.mean().unstack(); ns = np.array(g.columns, float)
mid = np.sqrt(ns[:-1] * ns[1:])
for m in MS:
    sl = -np.diff(np.log(g.loc[m].values)) / np.diff(np.log(ns))
    ax.plot(mid, sl, color=col[m], lw=2, marker="o", ms=4, label=f"m = {m}")
ax.set_xscale("log"); ax.set_xlabel("graph size N (log scale, midpoint of each doubling)")
ax.set_ylabel("local decay exponent  -d log gap / d log N"); ax.legend(fontsize=8, ncol=2, loc="upper left")
ax.set_title("Torus, MSE: the decay steepens with N; higher m lags behind", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "local_slope_vs_n.png", dpi=170); plt.close(fig)

# 4. robustness
def dbeta(d, B=2000, seed=0):
    groups = {k: v.gen_gap_loss.to_numpy() for k, v in d.groupby(["m", "n"])}; ns_ = sorted(d.n.unique()); ms_ = np.array([4, 16])
    b = betas_from_cells({k: v.mean() for k, v in groups.items()}, ms_, ns_, "logn"); rng = np.random.default_rng(seed)
    bb = np.array([betas_from_cells({k: rng.choice(v, len(v)).mean() for k, v in groups.items()}, ms_, ns_, "logn") for _ in range(B)])
    return b[0] - b[1], np.percentile(bb[:, 0] - bb[:, 1], [2.5, 97.5])
arms = (("clean torus, k = 10", tm[tm.m.isin([4, 16])]), ("denser graph, k = 30", k30), ("noise sigma = 0.05", nz))
fig, ax = plt.subplots(figsize=(6.4, 2.7))
for i, (name, d) in enumerate(arms):
    v, ci = dbeta(d)
    ax.errorbar(v, i, xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color="#1f5fa8", ms=6, lw=2, capsize=3)
    ax.text(v, i - 0.28, f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", ha="center", fontsize=7.5, color="#333")
ax.axvline(0, color="#888", lw=1); ax.axvline(1/8 - 1/20, color="#888", lw=1, ls="--")
ax.text(1/8 - 1/20 + 0.005, 2.55, "change implied by the bound's rate, 1/8 - 1/20", fontsize=7, color="#666", va="center")
ax.set_yticks(range(3)); ax.set_yticklabels([a for a, _ in arms]); ax.set_ylim(-0.6, 2.8); ax.invert_yaxis()
ax.set_xlabel("decay exponent at m = 4 minus at m = 16 (95% bootstrap CI)"); ax.set_xlim(-0.02, 0.42)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT + "robustness_contrast.png", dpi=170); plt.close(fig)
print("ok")
