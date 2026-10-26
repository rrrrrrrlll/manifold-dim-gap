Subject: Experiment A results + a question about harder manifolds

Hi [Name],

Quick update on the manifold-dimension direction. Everything (code, the plan I froze before the runs, results write-up) is here: https://github.com/rrrrrrrlll/manifold-dim-gap

Short version:
- I trained the GCN on kNN graphs sampled from a flat torus and a sphere of dimension m = 2 to 24, with N = 250 to 16k and 10 seeds (2,240 runs), and measured how fast the train-test loss gap shrinks with N.
- The decay exponent drops clearly with m (torus: 0.92 at m = 2, 0.33 at m = 24), so the pre-specified test supports the dependence on m that the bound suggests. Same on the sphere, for the cross-entropy and accuracy gaps, and with k = 30.
- But the bound's rate 1/(m+4) is far too slow, the gap isn't a clean power law (higher m mostly means the fast decay starts later, at least up to 16k), and off-manifold noise almost erases the effect.

To check the analysis:
- docs/experiment_a_results.md: results, figures and caveats (start here): https://github.com/rrrrrrrlll/manifold-dim-gap/blob/main/docs/experiment_a_results.md
- docs/analysis_plan.md: the plan as frozen before the runs, with dated deviations at the end
- src/mdgap/analyze_decay_exponent.py (the pre-specified test) and scripts/experiment_a_followup.py (everything else) produce every number in the write-up

Next I'm running three quick follow-ups (larger N, a wider model, a noise sweep).

Questions:
1. Is "slower decay at higher m over practical graph sizes" enough for the paper, or should I go after the asymptotic rate (e.g. with a radius graph that shrinks with N)?
2. A concern I haven't raised yet: the torus and sphere are very clean compared with real embeddings, which are curved, noisy and uneven, so I'm not sure they say much about real data on their own. Two things I'd like your view on:
   a. Is there a standard way to generate harder manifolds whose intrinsic dimension is still known and fixed? From what I found, the options are benchmark manifolds used to test ID estimators, random nonlinear maps of a low-dimensional latent (the hidden manifold model), and images generated from a fixed number of free latents (links below). Is one of these what you'd use, or something else?
   b. Apart from harder manifolds, how would you argue that results on simple manifolds are still meaningful for the paper?

Papers on harder manifolds of known dimension:
- Benchmark manifolds for testing ID estimators: Hein & Audibert (ICML 2005), https://www.ml.uni-saarland.de/Publications/HeiAud2005.pdf ; Campadelli et al. (2015), https://doi.org/10.1155/2015/759567 ; Tempczyk et al. (ICLR 2026), https://mlanthology.org/iclr/2026/tempczyk2026iclr-we/ ; implemented in scikit-dimension, https://scikit-dimension.readthedocs.io/en/latest/skdim.datasets.BenchmarkManifolds.html
- Random nonlinear map of a low-dimensional latent: hidden manifold model (Goldt et al., PRX 2020), https://arxiv.org/abs/1909.11500
- Images from a fixed number of free latents: Pope et al. (ICLR 2021), https://arxiv.org/abs/2104.08894

Thanks,
Roy
