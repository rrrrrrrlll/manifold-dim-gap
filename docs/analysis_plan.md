# Analysis plan: data-manifold dimension and the decay of the generalization gap

**Status: frozen.** The freeze takes effect with the commit that contains this file and the
final `configs/clean_*.yaml`. From then on, changes go in the dated "Deviations" section at
the end, never by editing the plan.

- Frozen on: 2026-10-25
- Epochs (full-batch updates), from the pilot trace: **2000**  -> `optim.epochs` in `configs/clean_*.yaml`
- Seeds per cell, from the pilot's seed-to-seed spread and the power simulation: **10**  -> `seeds`

Source of this plan: "Design of experiments: data-manifold dimension and the generalization gap"
(sections 1, 3 and 5). `src/mdgap/analyze_decay_exponent.py` implements the primary analysis below.

## Question and hypotheses

Write the empirical loss gap as roughly C * N^(-beta(m)) over the measured range of N.

- **H1 (confirmatory, the main test):** beta(m) decreases as m grows, all else fixed.
- **H2 (consistency check only):** beta(m) >= 1/(m+4) at every m. Descriptive: an O(.) bound with
  unknown constants cannot be refuted on a finite range of N.
- **H3 (exploratory):** beta(m) is close to a/(m+c); report c.

## Clean-manifold design

Torus T^m (primary) and sphere S^m (secondary), m in {2, 4, 6, 8, 12, 16, 24}, fixed isometric
map into R^128, no noise; N in {250, 500, 1000, 2000, 4000, 8000, 16000}; labels = quadrant of
the first two intrinsic coordinates (4 balanced classes); kNN graph on the sampled points
(k = 10, union-symmetrized, w = exp(-d^2/sigma^2), sigma = median kNN distance); GNN as in the
old repo (2 GCN layers, hidden 32, dropout 0.5); full-batch Adam, lr 0.01, fixed epochs, no
early stopping; test risk = mean loss over 5 fresh N-point graphs; losses MSE on one-hot
(primary) and cross-entropy (secondary). Robustness arms (torus, MSE, m in {4, 16}):
noise sigma = 0.05; k = 30.

## Primary analysis (torus, MSE loss gap)

1. Gap of a run = mean test loss over the fresh graphs - final training loss (both in eval mode).
2. Average over seeds in each (m, N) cell. Cells with mean gap <= 0 are excluded and listed.
3. For each m, least-squares slope of log(mean gap) on log N; beta_hat(m) = -slope
   (requires >= 3 usable sizes).
4. Weighted least-squares slope b1 of beta_hat(m) on m; weights = 1 / bootstrap variance of beta_hat(m).
5. Uncertainty: 2,000 bootstrap replicates resampling seeds within every cell, repeating 2-4.
   SE(b1) = bootstrap SD of b1 x sqrt(s/(s-1)), where s is the fewest seeds in any cell that
   enters the fit (s = 10 by design); the factor undoes the bootstrap's (s-1)/s shrinkage of
   the variance of a cell mean. t = 95th percentile of Student's t with (number of m with a
   beta_hat) - 2 degrees of freedom (5 by design, t = 2.015).

**Decision rule:**

- Support for H1: one-sided 95% upper bound b1 + t * SE(b1) < 0.
- Evidence of no dependence: the 90% interval (b1 +- t * SE(b1)) * (24 - 2) lies inside +-0.033
  (a quarter of the change the bound's exponent implies between m = 2 and m = 24:
  1/6 - 1/28 = 0.131).
- Otherwise: inconclusive, reported as such.
- Reported alongside, not used for the decision: the percentile bootstrap bound (95th
  percentile of the bootstrap b1) and a bound with SE from the weighted scatter of beta_hat(m)
  around the fitted line (residual SE, which also grows if beta_hat is curved in m).

Why this rule and not the percentile bootstrap bound: see Power below.

## Secondary analyses

Same b1 test, Holm-corrected across this set: sphere; cross-entropy gap (its gap was still
growing at 2,000 epochs in the pilot, so its beta_hat describes this training length only and
is reported with that caveat); accuracy gap;
noise check; k = 30 check; log(N / log N) instead of log N (`--x n_over_logn`); each rung of
the realism ladder. Plus H2 (table of beta_hat(m) - 1/(m+4)) and H3 (fit of a/(m+c)), both descriptive.

## Realism ladder (after the clean-manifold primary analysis)

Each rung adds one property of real embeddings, at m in {2, 6, 12, 24}, same N grid, MSE:

- **Smooth random map:** the torus pushed through a fixed random two-layer tanh map into R^128
  (curvature, uneven density, anisotropy).
- **Off-manifold noise:** the smooth random map plus Gaussian noise, sigma = 0.05 and 0.2 x the
  clean data's per-coordinate standard deviation.
- **One component per class:** four independently drawn smooth random maps, one per class,
  shifted apart so accuracy at N = 1,000 is 70-90%.
- **Mixed-dimension components (exploratory):** as above, half the classes at m and half at 2m.
- **VAE-decoded images:** the MNIST VAE decoder with k free latent coordinates, decoded and
  re-encoded to the 128-d embedding.

Labels always come from latent coordinates. The dimension is checked on every rung (Jacobian
rank at 1,000 points, offline estimators); if the VAE-decoded estimates sit clearly below k,
the estimate is used as m. If the clean manifolds show no trend, run only the smooth random map
and the VAE-decoded images. Summary figure: the clean-manifold and VAE-decoded beta_hat(m)
curves with the 7 real datasets overlaid at their estimated m (bars spanning the three
estimators).

Estimator calibration (`results/id_calibration`, 16,000 points, MLE / TwoNN / correlation
integral): within 10% of the true m up to m = 6 on both manifolds; above that the torus is
overestimated (m = 8: 9.1 / 8.4 / 9.6; m = 24: 28.6 / 29.7 / 27.8) and the sphere
underestimated (m = 24: 19.2 / 20.2 / 18.6), and estimates fall with fewer points (MLE at
1/16 of the points, mean of 5 draws, m = 24: torus 25.6, sphere 17.1). So in the overlay figure the synthetic
curves are also placed at their estimated dimension, with the same estimator and the same
number of points as the real datasets, next to the curve at the true m.

## Power

Before freezing: simulate b1 under the bound-implied change (0.131 over m = 2-24) using the
pilot's seed-to-seed spread of log-gaps; raise the seed count if power < 80%.
`src/mdgap/summarize_pilot.py` does this: run gaps are drawn as log g = a - beta(m) log N + e
with beta(m) = 1/(m+4) and e ~ N(0, sd^2), sd taken from the pilot (one fresh graph, so
conservative) and interpolated over m and log N; each simulated grid goes through the real
primary analysis. It also reports the false-"support" rate when beta does not depend on m.

Pilot result (`results/pilot/summary/power.csv`, 1,000 simulations per row, run with
`--n-sims 1000`; sd of log gap 0.39 / 0.32 at m = 2, N = 250 / 16,000 and 0.15 / 0.18 at m = 24):

| seeds per cell | power (bound-implied change) | false "support" (no change) | "no dependence" when there is none |
|---|---|---|---|
| 5 | 0.86 | 0.05 | 0.00 |
| 10 | 0.99 | 0.04 | 0.00 |
| 15 | 1.00 | 0.04 | 0.00 |
| 20 | 1.00 | 0.05 | 0.05 |

Seeds: 10 (power 0.99, false support 0.04). Consequence to keep in mind: with 10 seeds the
"evidence of no dependence" outcome is practically unreachable (0 of 1,000 null simulations),
because the 90% interval of the total change (about +-0.046) is wider than the +-0.033 margin;
even at 20 seeds it is reached in only 5% of null simulations.
If H1 is not supported, the result will read "inconclusive", not "no dependence".

Why the decision rule uses t and the corrected SE: the first version of the rule (support if
the 95th percentile of the bootstrap b1 < 0) claimed support in 9-15% of 100 null simulations
per seed count. A check with 2,000 simulations per setting (same noise model, same grid)
gave 10% / 9% / 9% at 5 / 10 / 15 seeds; there the bootstrap SD of b1 was 18% / 9% / 7%
below its simulated SD. The rule above gave 4.4% / 4.4% / 4.9% in that check, with power
0.89 / 0.99 / 1.00. The rule was changed before the freeze, so it is not a deviation.
Limit of this check: it assumes the simulation's noise model (log-normal, sd interpolated
from four pilot cells, log gap linear in log N).

Choosing the epoch count: the pilot traces the training loss and the loss gap on a fresh
graph every 50 epochs. Record which epoch count was chosen and why (training loss levelled
off; how much the gap still moves after that), because the gap depends on training length.

Chosen: 2,000 epochs. MSE: the training loss levels off (later training improves it by < 2%)
between epoch 650 and 1,900 across the 12 MSE runs; between epochs 1,000 and 2,000 the mean
gap moves by 4% (m = 2, N = 250), 10% (m = 2, N = 16,000), 0.3% (m = 24, N = 250) and 11%
(m = 24, N = 16,000); the seed spread and the power above are measured at 2,000.
Cross-entropy: the gap keeps growing through epoch 2,000 (m = 24, N = 250: 7.5 at epoch 1,000,
13.1 at 2,000; N = 16,000: 0.74 and 1.28) while test accuracy stays far below training
accuracy (m = 24, N = 250: 64% vs 99% at the end). It stays in the grid as a secondary
analysis with that caveat.
Measured time on the CPU where the pilot ran: 11.3 s (N = 250) and 157.5 s (N = 16,000) per
run (mean over both losses and both m); interpolating log-log in N between these two sizes
gives about 16 h per loss for the main grid on that machine (an estimate, not measured).

## Pilot provenance

The pilot rows (`results/pilot/*.csv`) have an empty `code_commit` because the repository had
no commits when they were run. They were produced by the `src/mdgap/train.py`, `manifolds.py`,
`graphs.py` and `models.py` of the first commit (all four were last modified before the pilot's
result files were written);
after the pilot only `analyze_decay_exponent.py` (decision rule), the tests, the configs and
the documentation changed.

## Deviations

_(none yet)_
