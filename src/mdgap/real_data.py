"""
Real image embeddings as a node pool (experiments B and C; docs/analysis_plan_bc.md).

Files (written by scripts/prepare_real_embeddings.py into data/real/):
  <features>_x.npy       float32 (n_pool, d)   the old GNN's features, all images
  <features>_pca128.npy  float32 (n_pool, 128) top-128 principal-component scores (MNIST, CIFAR-10)
  <dataset>_y.npy        int64   (n_pool,)     labels 0 .. C-1

A dataset is a label set on a feature set; both CelebA tasks share the "celeba" features.

Sampling: the training graph is N images drawn without replacement from the pool; test graph t is N
images drawn without replacement from the images NOT in the training sample. The draw depends only
on (seed, feature set, N), so every PCA width and both CelebA tasks see the same images.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# dataset -> (feature set, label file stem)
DATASETS = {
    "mnist": ("mnist", "mnist"),
    "fmnist": ("fmnist", "fmnist"),
    "pathmnist": ("pathmnist", "pathmnist"),
    "cifar10": ("cifar10", "cifar10"),
    "fer2013": ("fer2013", "fer2013"),
    "celeba_gb": ("celeba", "celeba_gb"),
    "celeba_sb": ("celeba", "celeba_sb"),
}
FEATURE_CODES = {"mnist": 101, "fmnist": 102, "pathmnist": 103, "cifar10": 104, "fer2013": 105, "celeba": 106}


@dataclass
class Sample:
    x: np.ndarray   # (n, d) float32 features as given (the graph is built on these)
    y: np.ndarray   # (n,) int64 labels
    idx: np.ndarray  # (n,) pool indices


class EmbeddingPool:
    """
    One dataset at one width. features = "full" or "pca128"; width = None (all columns of the file)
    or r (first r columns, only for pca128).
    """

    def __init__(self, dataset, features="full", width=None, data_dir="data/real"):
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}")
        fset, ystem = DATASETS[dataset]
        data_dir = Path(data_dir)
        suffix = {"full": "_x.npy", "pca128": "_pca128.npy"}[features]
        x = np.load(data_dir / f"{fset}{suffix}", mmap_mode="r")
        if width is not None:
            if features != "pca128":
                raise ValueError("a width is only defined for the pca128 features")
            if not 1 <= width <= x.shape[1]:
                raise ValueError(f"width {width} outside 1..{x.shape[1]}")
            x = x[:, :width]
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.load(data_dir / f"{ystem}_y.npy").astype(np.int64)
        if len(self.y) != len(self.x):
            raise ValueError(f"{dataset}: {len(self.x)} feature rows but {len(self.y)} labels")
        self.dataset, self.features, self.width, self.code = dataset, features, width, FEATURE_CODES[fset]
        self.num_classes = int(self.y.max()) + 1
        # pool mean and SD per coordinate, in chunks (CIFAR-10's pool is 1 GB as float32)
        s, ss = np.zeros(self.dim), np.zeros(self.dim)
        for i in range(0, self.size, 8192):
            block = self.x[i:i + 8192].astype(np.float64)
            s += block.sum(0)
            ss += (block ** 2).sum(0)
        mean = s / self.size
        var = np.maximum(ss / self.size - mean ** 2, 0.0)
        self.mean = mean.astype(np.float32)
        self.sd = np.where(var > 0, np.sqrt(var), 1.0).astype(np.float32)

    @property
    def size(self):
        return len(self.y)

    @property
    def dim(self):
        return self.x.shape[1]

    def indices(self, n, seed, stream=0):
        """Pool indices of the training sample (stream 0) or of test graph `stream` >= 1."""
        if 2 * n > self.size:
            raise ValueError(f"{self.dataset}: N = {n} needs a pool of at least {2 * n}, have {self.size}")
        perm = np.random.default_rng([seed, self.code, n, 0]).permutation(self.size)
        if stream == 0:
            return np.sort(perm[:n])
        rest = perm[n:]
        return np.sort(np.random.default_rng([seed, self.code, n, stream]).choice(rest, size=n, replace=False))

    def sample(self, n, seed, stream=0):
        idx = self.indices(n, seed, stream)
        return Sample(x=self.x[idx], y=self.y[idx], idx=idx)

    def standardize(self, x):
        """Model input: per-coordinate standardization with the pool's mean and SD."""
        return (x - self.mean) / self.sd
