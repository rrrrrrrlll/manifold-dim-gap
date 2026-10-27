# Data this repo uses from the old repo

None of it is in git. The clean manifolds and the synthetic rungs of the realism ladder
(smooth random map, off-manifold noise, one component per class) need no external data.

| Needed for | Files (old repo) | Size | Status |
|---|---|---|---|
| Comparing with the 7 real datasets (existing gap curves; "real data on the synthetic curve" figure) | `results/gap_curve/gap_curve_matched_n{,_proj}/*.csv`, `results/id_estimation_runs/id_estimation/{mle,twonn,corrint}/*.csv` | ~0.2 MB | Copied: `python scripts/export_from_old_repo.py --old-repo ../image-gnn-id-est-2` |
| VAE-decoded images rung | one MNIST VAE checkpoint + its encoder/decoder code | 15-60 MB | **To confirm:** which checkpoint in `src/scripts/checkpoints/mnist/` produced the 128-d `mnist_smsl_*` embeddings the graphs use (they differ from `mnist_train_embeddings.npy`) |
| Experiments B and C2 (PCA width; the 7 datasets rerun) | `data/<Dataset>Embeddings/**/<name>_smsl_train_full_embeddings.npy`: the features the old GNN used (`config/knn/*.yaml` builds the graphs from `*_smsl_*`), every image (train split, then test split; last two columns label and split flag) | 1.3 GB as float32 in `data/real/` | Copied: `python scripts/prepare_real_embeddings.py --old-repo ../image-gnn-id-est-2` (checksums in `data/real/MANIFEST.csv`). Checked 2026-10-26: `_full` = train rows followed by test rows; the `*_train_embeddings.npy` files the old ID estimates used are different vectors (MNIST: same width, different values; Fashion-MNIST 128 vs 256 and CIFAR-10 364 vs 4096 dimensions) |

Rule: anything copied in is listed in `data/old_repo/MANIFEST.csv` with its SHA-256 and the old repo's commit.
