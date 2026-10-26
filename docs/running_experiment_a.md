# Running experiment A on the GPU server

Experiment A = the clean-manifold grid (torus and sphere, m in {2, 4, 6, 8, 12, 16, 24}, 7 sizes N,
10 seeds, MSE and cross-entropy: 1,960 runs) plus two robustness checks on the torus (noise 0.05 and
k = 30: 140 runs each). Settings are fixed in `docs/analysis_plan.md`.

Everything on the server lives in one folder, `~/manifold-dim-gap`:

```
~/manifold-dim-gap/
  src/mdgap/            the 5 Python files the runs need
  configs/              clean_manifolds.yaml, clean_torus_noise.yaml, clean_torus_k30.yaml
  jobs/run_experiment_a.sh
  results/              written by the runs (one CSV row per run)
  logs/                 one line per run
```

Replace `you@server` below with what you normally type after `ssh`.

## 1. Upload (PowerShell on your PC)

```powershell
cd C:\Users\ruoqu\Python_projects\image_gnn\manifold-dim-gap

tar -czf mdgap_upload.tar.gz src/mdgap/__init__.py src/mdgap/manifolds.py src/mdgap/graphs.py src/mdgap/models.py src/mdgap/train.py configs/clean_manifolds.yaml configs/clean_torus_noise.yaml configs/clean_torus_k30.yaml jobs/run_experiment_a.sh

scp mdgap_upload.tar.gz you@server:

ssh you@server "mkdir -p manifold-dim-gap && tar -xzf mdgap_upload.tar.gz -C manifold-dim-gap && rm mdgap_upload.tar.gz"

Remove-Item mdgap_upload.tar.gz
```

The archive (about 12 KB) passes through your home folder on the server and is deleted right after
unpacking. If you change code later, repeat these commands; files are overwritten, results kept.

## 2. Check the server (once)

```bash
ssh you@server
cd ~/manifold-dim-gap
bash jobs/run_experiment_a.sh check
```

It uses the conda env `image_gnn` (the old repo's env, so nothing new is installed). For another
env, put its name in front: `CONDA_ENV=myenv bash jobs/run_experiment_a.sh check`.

You should see the package versions, `CUDA OK: ... visible GPU(s)`, `planned 1960 runs`,
`planned 140 runs` twice, and the time of one full run on GPU 0. If it says CUDA is not
available, stop: the runs would quietly train on the CPU.

## 3. Launch

Check with `nvidia-smi` which GPUs you may use, then (example: GPUs 0 and 1):

```bash
bash jobs/run_experiment_a.sh launch 0 1
```

This starts two tmux sessions and returns; you can log out.

- `mdgap_torus` on the first GPU: torus grid (980 runs), then both robustness checks (280 runs).
- `mdgap_sphere` on the second GPU: sphere grid (980 runs).
- With one GPU: `launch 0` puts both on GPU 0.

## 4. Watch progress

```bash
cd ~/manifold-dim-gap && bash jobs/run_experiment_a.sh status
```

Shows runs done per results file (out of 490 or 140), the last finished runs with their times,
which sessions are alive, and GPU use. To watch live: `tmux attach -t mdgap_torus`; leave with
Ctrl-b, then d (the runs keep going).

When a session ends it prints `[mdgap_torus finished with exit code N ...]` and stays open.

- Exit code 0: done. Close it with `tmux kill-session -t mdgap_torus`.
- Anything else, or the server rebooted: resume with
  ```bash
  tmux kill-session -t mdgap_torus     # only if the old session still exists
  bash jobs/run_experiment_a.sh launch 0 1
  ```
  Finished runs are skipped; the interrupted run is redone from scratch.

## 5. Download the results (PowerShell on your PC)

Works while runs are still going, too (partial results).

```powershell
cd C:\Users\ruoqu\Python_projects\image_gnn\manifold-dim-gap

ssh you@server "cd manifold-dim-gap && bash jobs/run_experiment_a.sh pack"

scp you@server:manifold-dim-gap/results_experiment_a.tar.gz .

tar -xzf results_experiment_a.tar.gz

Remove-Item results_experiment_a.tar.gz
```

This fills `results\clean_manifolds\`, `results\clean_torus_noise\`, `results\clean_torus_k30\` and
`logs\` in your repo (all git-ignored).

## 6. Analyse (on your PC, in the env where you ran the pilot)

```powershell
cd C:\Users\ruoqu\Python_projects\image_gnn\manifold-dim-gap
$env:PYTHONPATH = "src"

# primary test (H1): torus, MSE loss gap
python -m mdgap.analyze_decay_exponent --results results/clean_manifolds/torus_mse.csv

# secondary analyses
python -m mdgap.analyze_decay_exponent --results results/clean_manifolds/sphere_mse.csv
python -m mdgap.analyze_decay_exponent --results results/clean_manifolds/torus_ce.csv
python -m mdgap.analyze_decay_exponent --results results/clean_manifolds/sphere_ce.csv
python -m mdgap.analyze_decay_exponent --results results/clean_torus_noise/torus_mse.csv
python -m mdgap.analyze_decay_exponent --results results/clean_torus_k30/torus_mse.csv
python -m mdgap.analyze_decay_exponent --results results/clean_manifolds/torus_mse.csv --x n_over_logn
```

Each writes `<name>_analysis/` next to its CSV with `summary.md` (the decision), `beta_by_m.csv`,
`cells.csv` and `beta_vs_m.png`. Then the robustness contrasts, the accuracy-gap analysis, the
p-values for Holm's correction, the fit diagnostics and the report figures:

```powershell
python scripts/experiment_a_followup.py > results/experiment_a_followup.txt
```

The results are written up in `docs/experiment_a_results.md`.

## 7. Exploratory follow-ups (after the main grid)

Three short follow-ups, described in `docs/analysis_plan.md` (Deviations, 2026-10-26) and
`docs/experiment_a_results.md` (Next steps): larger N, a 4x wider model, and a noise sweep.

Upload the five follow-up configs (PowerShell on your PC; the run script is uploaded again
because it gained the `followups` command):

```powershell
cd C:\Users\ruoqu\Python_projects\image_gnn\manifold-dim-gap

tar -czf mdgap_upload.tar.gz jobs/run_experiment_a.sh configs/followup_large_n.yaml configs/followup_capacity_h128.yaml configs/followup_noise_0.005.yaml configs/followup_noise_0.01.yaml configs/followup_noise_0.02.yaml

scp mdgap_upload.tar.gz you@server:

ssh you@server "tar -xzf mdgap_upload.tar.gz -C manifold-dim-gap && rm mdgap_upload.tar.gz"

Remove-Item mdgap_upload.tar.gz
```

On the server:

```bash
cd ~/manifold-dim-gap
bash jobs/run_experiment_a.sh followups 0 1
```

- `mdgap_followup_a` on the first GPU: larger N (60 runs), then hidden 128 (210 runs). The larger-N
  runs are the slow ones: building one kNN graph of 64,000 points took about 20 s on a 2-core
  machine, and each run builds 6 graphs.
- `mdgap_followup_b` on the second GPU: the noise sweep (3 x 140 runs).

`status`, resuming and downloading work as above (`pack` includes the follow-up folders).

The ID estimates for the noise sweep run on your PC (CPU, needs `scikit-dimension`), in the
repo folder:

```powershell
$env:PYTHONPATH = "src"
python -m mdgap.calibrate_id --config configs/clean_torus_noise.yaml
python -m mdgap.calibrate_id --config configs/followup_noise_0.005.yaml
python -m mdgap.calibrate_id --config configs/followup_noise_0.01.yaml
python -m mdgap.calibrate_id --config configs/followup_noise_0.02.yaml
```

They write `results/id_calibration/<method>/torus_m4_noise<sigma>.csv` and `..._m16_...`, next to
the clean estimates from the calibration.

## Disk use on the server

About 60 KB of code, roughly 1 MB of results and logs at the end, no model files, and the conda env
you already have.

## If something goes wrong

- `conda not on PATH`: run the command from your normal login shell (where `conda` works).
- `missing in this env: ...`: the env lacks a package. Installing into `image_gnn` also changes the
  old repo's env, so check with your PI's setup first.
- `$'\r': command not found`: the script got Windows line endings (opened and saved on Windows).
  Fix with `sed -i 's/\r//' jobs/run_experiment_a.sh`.
