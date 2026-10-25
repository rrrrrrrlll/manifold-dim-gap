# Data this repo uses from the old repo

None of it is in git. The clean manifolds and the synthetic rungs of the realism ladder
(smooth random map, off-manifold noise, one component per class) need no external data.

| Needed for | Files (old repo) | Size | Status |
|---|---|---|---|
| Comparing with the 7 real datasets (existing gap curves; "real data on the synthetic curve" figure) | `results/gap_curve/gap_curve_matched_n{,_proj}/*.csv`, `results/id_estimation_runs/id_estimation/{mle,twonn,corrint}/*.csv` | ~0.2 MB | Copied: `python scripts/export_from_old_repo.py --old-repo ../image-gnn-id-est-2` |
| VAE-decoded images rung | one MNIST VAE checkpoint + its encoder/decoder code | 15-60 MB | **To confirm:** which checkpoint in `src/scripts/checkpoints/mnist/` produced the 128-d embeddings the graphs use |
| PCA-width experiment on real embeddings (MNIST, CIFAR-10) | their embeddings, or only their PCA projections computed once in the old repo | ~75 MB (MNIST); CIFAR-10 ~2 GB full or ~30 MB as projections | **To confirm:** `*_smsl_*` vs `*_smsl_*_full_*` embedding files |
| Re-running the 7 real datasets with per-sample graphs (optional) | embeddings of all 7 datasets | ~2.8 GB full, much less as projections | Same question as above |

Rule: anything copied in is listed in `data/old_repo/MANIFEST.csv` with its SHA-256 and the old repo's commit.
