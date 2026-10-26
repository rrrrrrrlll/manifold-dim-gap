# manifold-dim-gap

Does the generalization gap of a GNN on a manifold-sampled graph shrink more slowly
with graph size N when the data manifold has a higher dimension m, as the bound

    GA = O( (log(N/delta)/N)^(1/(m+4)) + (log N / N)^(1/(m+4)) )

(Wang, Cervino & Ribeiro, arXiv:2409.05191, as stated in Deberaldini Netto, Wang & Ruiz,
arXiv:2409.13063) suggests? This repo trains GNNs on synthetic manifolds of known
dimension: first clean manifolds (flat torus, round sphere), the theorem's own setting,
then progressively messier ones that move toward real image embeddings (the "realism
ladder"). The design and the pre-specified analysis are in `docs/analysis_plan.md`.

## Files

| File | What it does |
|---|---|
| `src/mdgap/manifolds.py` | Samples the flat torus and the round sphere at any dimension m, labels each point (4 classes) from its intrinsic coordinates, and places it in R^128 with a fixed distance-preserving map |
| `src/mdgap/graphs.py` | Builds the kNN graph on a set of sampled points (k nearest neighbors, symmetrized, Gaussian edge weights with sigma = median neighbor distance) |
| `src/mdgap/models.py` | The GNN classifier (`GNNModel` and its building blocks), copied from the old repo |
| `src/mdgap/train.py` | Runs a whole grid from a config: for each (manifold, m, N, seed, loss) it trains the GNN full-batch, evaluates it on 5 fresh graphs, and appends one row to `results/<grid>/<manifold>_<loss>.csv`; re-running skips finished rows. With `trace_every` set, it also records training and fresh-graph test loss during training (`_trace.csv`) |
| `src/mdgap/analyze_decay_exponent.py` | The pre-specified analysis: fits how fast the loss gap decays with N at each m, tests whether that rate falls as m grows, and writes a summary, tables and a figure |
| `src/mdgap/summarize_pilot.py` | Reads the pilot and reports, per run, when the training loss levels off and when the gap settles; the seed-to-seed spread of the gap; simulated power of the main analysis for several seed counts; time per run; plus loss-by-epoch plots |
| `src/mdgap/calibrate_id.py` | Runs the three ID estimators on the same manifolds at known m, to measure their bias in this setting |
| `src/mdgap/id_estimation.py` | MLE / TwoNN / correlation-integral estimators over several sample sizes, copied from the old repo |
| `configs/pilot.yaml` | Short pilot: 2 values of m, 2 sizes, 3 seeds, 2,000 epochs with training and test-loss traces, to choose the epoch and seed counts |
| `configs/clean_manifolds.yaml` | Main grid on the torus and sphere: 7 values of m x 7 sizes x 10 seeds x 2 losses, 2,000 epochs |
| `configs/clean_torus_noise.yaml` | Robustness check: torus with Gaussian noise off the manifold (sigma = 0.05) |
| `configs/clean_torus_k30.yaml` | Robustness check: torus with a denser graph (k = 30) |
| `configs/followup_large_n.yaml` | Exploratory follow-up: torus at N = 32,000 and 64,000 (m = 4, 12, 24) |
| `configs/followup_capacity_h128.yaml` | Exploratory follow-up: 4x wider GNN (hidden 128) at m = 2, 8, 24 |
| `configs/followup_noise_*.yaml` | Exploratory follow-up: noise sweep (sigma = 0.005, 0.01, 0.02) at m = 4, 16 |
| `jobs/run_pilot.sh` | Runs the pilot on one GPU |
| `jobs/run_id_calibration.sh` | Runs the estimator calibration (CPU) |
| `jobs/_job_setup.sh` | Shared by the two scripts above: activates the conda env, puts `src/` on the path, handles `DRY_RUN=1` |
| `jobs/run_experiment_a.sh` | Experiment A on the GPU server: checks the env and CUDA, launches the main grid and the robustness checks in tmux, reports progress, packs the results for download |
| `scripts/experiment_a_followup.py` | Analyses of experiment A beyond the pre-specified test (robustness contrasts, accuracy gap, p-values for Holm, fit diagnostics) and the report figures in `docs/figures/experiment_a/` |
| `scripts/export_from_old_repo.py` | One-time copy of the old repo's result tables this repo needs, with a checksum manifest |
| `tests/test_pipeline.py` | Fast CPU checks of the sampler, graph builder, model wiring, training runner and ID estimators |
| `docs/analysis_plan.md` | Hypotheses, design, decision rule and secondary analyses, frozen before the main grid; later changes go in its Deviations section |
| `docs/running_experiment_a.md` | Step-by-step guide: upload the needed files to the server, run experiment A, download and analyse the results |
| `docs/experiment_a_results.md` | Results of experiment A: the pre-specified test, secondary analyses, robustness, caveats and next steps |
| `docs/data_from_old_repo.md` | Which old data each later experiment needs, and what is still to be confirmed |

`data/`, `results/` and `logs/` are not in git.

## Setup (cluster)

Any conda env with PyTorch (CUDA build), PyTorch Geometric, scipy, pandas, scikit-learn and PyYAML
works; on the GPU server the old repo's env `image_gnn` is used, which is the default of the job
scripts (`CONDA_ENV=<name>` to use another). To make a new one instead:

```bash
conda create -n mdgap python=3.11 -y && conda activate mdgap
# install PyTorch for the cluster's CUDA version from https://pytorch.org, then:
pip install -r requirements.txt
pip install -e .
pytest -q                      # ~10 s on CPU
```

Training is full-batch, so pyg-lib / torch-sparse (needed by NeighborLoader in the old repo) are not required.

## Run order

1. **Pilot:** `bash jobs/run_pilot.sh 0` (24 runs).
2. **Estimator calibration (CPU):** `bash jobs/run_id_calibration.sh`.
3. **Freeze the plan:** run `python -m mdgap.summarize_pilot --pilot results/pilot --n-sims 1000` (about an hour on a 2-core machine; the default 100 simulations are faster but too few to check a 5% error rate) and read `results/pilot/summary/summary.md`; choose the epoch and seed counts; set `optim.epochs` and `seeds` in all `configs/clean_*.yaml`; fill in and date `docs/analysis_plan.md`; commit. Chosen on 2026-10-25: 2,000 epochs, 10 seeds.
4. **Experiment A, main grid (1,960 runs) and robustness checks (280 runs):** follow `docs/running_experiment_a.md` (upload, then `bash jobs/run_experiment_a.sh check` and `bash jobs/run_experiment_a.sh launch 0 1` on the server).
5. **Analysis:** `python -m mdgap.analyze_decay_exponent --results results/clean_manifolds/torus_mse.csv` (primary); the same for the other result files (secondary). Then `python scripts/experiment_a_followup.py > results/experiment_a_followup.txt`; results in `docs/experiment_a_results.md`.
6. **Exploratory follow-ups:** `bash jobs/run_experiment_a.sh followups 0 1` on the server (see `docs/running_experiment_a.md`).

Pilot and calibration jobs: prefix with `DRY_RUN=1` to list the runs without running them.

## Provenance

Copied from [image-gnn-id-estimation-2](https://github.com/rrrrrrrlll/image-gnn-id-estimation-2) at commit `d6fb71b`:

- `src/mdgap/models.py`: `GNNModel` (`src/scripts/models/classifier.py`), `GNNBasicBlock` (`src/scripts/models/gnns.py`), `WeightedSAGEConv` (`src/scripts/models/encoders.py`); class bodies verbatim, imports merged.
- `src/mdgap/id_estimation.py`: `src/scripts/id_estimation.py`, verbatim.

Old results used by later analyses are copied by `scripts/export_from_old_repo.py`, which records each file's checksum and the old repo's commit in `data/old_repo/MANIFEST.csv`. See `docs/data_from_old_repo.md`.
