"""
Geometric graph built on the sampled points themselves (the setting of the
generalization bound): symmetrized kNN graph with Gaussian edge weights.

Rule (identical for every manifold, m and N):
  * k nearest neighbors of each point (excluding itself), Euclidean distance;
  * symmetrize by union (i~j if j is among i's k neighbors or vice versa);
  * weight w_ij = exp(-d_ij^2 / sigma^2), sigma = median distance over all kNN pairs.

Note: the image pipeline in image-gnn-id-estimation-2 used exp(-d / 5^2) with
k = 100 and a median-weight threshold on one graph over all train+test nodes; this
is a different, per-sample rule chosen to match the theorem's setting.
"""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def knn_graph(x, k=10):
    """
    x: (n, d) array. Returns (edge_index LongTensor (2, E), edge_weight FloatTensor (E,),
    stats dict with sigma, mean_degree and isolated-node fraction).
    """
    n = x.shape[0]
    if n <= k:
        raise ValueError(f"need more than k={k} points, got {n}")
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(x)
    dist, idx = nn_.kneighbors(x)
    dist, idx = dist[:, 1:], idx[:, 1:]          # drop the point itself
    sigma = float(np.median(dist))
    if sigma <= 0:
        raise ValueError("median kNN distance is 0 (duplicate points?)")

    src = np.repeat(np.arange(n), k)
    dst = idx.reshape(-1)
    d = dist.reshape(-1)
    # union symmetrization: keep each unordered pair once, then add both directions
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    key = a.astype(np.int64) * n + b
    key, first = np.unique(key, return_index=True)
    a, b, d = a[first], b[first], d[first]
    w = np.exp(-(d ** 2) / sigma ** 2)

    edge_index = torch.from_numpy(np.stack([np.concatenate([a, b]), np.concatenate([b, a])])).long()
    edge_weight = torch.from_numpy(np.concatenate([w, w])).float()
    deg = np.bincount(edge_index[0].numpy(), minlength=n)
    stats = {"sigma": sigma, "mean_degree": float(deg.mean()), "isolated_frac": float((deg == 0).mean())}
    return edge_index, edge_weight, stats
