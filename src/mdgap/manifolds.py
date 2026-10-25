"""
Synthetic manifolds of known intrinsic dimension m, with class labels defined on
intrinsic (latent) coordinates, placed in a fixed ambient space R^D by a fixed
random linear isometry.

Clean manifolds, the theorem's own setting (docs/analysis_plan.md):
  * torus  -- flat torus T^m: angles theta ~ U[0, 2pi)^m, embedded as
              (cos theta_1, sin theta_1, ..., cos theta_m, sin theta_m) / sqrt(m),
              so every point has unit norm. Labels: quadrant of (theta_1, theta_2).
              Because the label depends only on two angles that are independent
              of the others, the label distribution and the decision boundary
              are exactly the same at every m.
  * sphere -- round sphere S^m in R^(m+1): normalized Gaussian vectors (uniform on
              the sphere). Labels: quadrant of (x_1, x_2). Classes are balanced
              by symmetry, but coordinates concentrate as m grows, so the task is
              not identical across m (secondary manifold for that reason).

Both are compact, smooth and sampled from their uniform measure, as the
generalization bound assumes. The ambient map is x_amb = Q z with Q (D x d_z)
having orthonormal columns, so distances (and therefore the kNN graph) are those
of the manifold's own embedding in R^(d_z); only the ambient width D is common
across m.

Randomness is fully determined by integer seeds: the ambient map depends only on
(embed_seed, manifold, m, D); a sample depends on (seed, manifold, m, n, stream),
where stream 0 is the training graph and stream t >= 1 the t-th fresh test graph.
"""
from dataclasses import dataclass

import numpy as np

MANIFOLD_CODES = {"torus": 1, "sphere": 2}
NUM_CLASSES = 4


def intrinsic_ambient_width(manifold, m):
    """Width d_z of the manifold's own (pre-rotation) embedding."""
    if manifold == "torus":
        return 2 * m
    if manifold == "sphere":
        return m + 1
    raise ValueError(f"unknown manifold {manifold!r}")


def ambient_map(manifold, m, ambient_dim, embed_seed=0):
    """Fixed (ambient_dim x d_z) matrix with orthonormal columns."""
    d_z = intrinsic_ambient_width(manifold, m)
    if d_z > ambient_dim:
        raise ValueError(f"{manifold} with m={m} needs ambient_dim >= {d_z}")
    rng = np.random.default_rng([embed_seed, MANIFOLD_CODES[manifold], m, ambient_dim])
    q, r = np.linalg.qr(rng.standard_normal((ambient_dim, d_z)))
    # fix the sign ambiguity of QR so the map is a deterministic function of the seed
    q = q * np.sign(np.diag(r))
    return q


@dataclass
class Sample:
    x: np.ndarray        # (n, ambient_dim) float32 node features
    y: np.ndarray        # (n,) int64 labels in {0, .., 3}
    latent: np.ndarray   # (n, m) intrinsic coordinates (angles for the torus, sphere coords for the sphere)


def sample_manifold(manifold, m, n, seed, stream=0, ambient_dim=128, embed_seed=0, noise_sigma=0.0):
    """
    Draw n points from the manifold's uniform measure, label them, and place them in
    R^ambient_dim. noise_sigma adds isotropic Gaussian noise in the ambient space, in
    the same units as the unit-norm manifold.
    """
    if m < 2:
        raise ValueError("labels use two intrinsic coordinates, so m >= 2")
    rng = np.random.default_rng([seed, MANIFOLD_CODES[manifold], m, n, stream])

    if manifold == "torus":
        theta = rng.uniform(0.0, 2 * np.pi, size=(n, m))
        z = np.empty((n, 2 * m))
        z[:, 0::2] = np.cos(theta)
        z[:, 1::2] = np.sin(theta)
        z /= np.sqrt(m)
        y = 2 * (theta[:, 0] >= np.pi) + (theta[:, 1] >= np.pi)
        latent = theta
    elif manifold == "sphere":
        g = rng.standard_normal((n, m + 1))
        z = g / np.linalg.norm(g, axis=1, keepdims=True)
        y = 2 * (z[:, 0] >= 0) + (z[:, 1] >= 0)
        latent = z
    else:
        raise ValueError(f"unknown manifold {manifold!r}")

    x = z @ ambient_map(manifold, m, ambient_dim, embed_seed).T
    if noise_sigma > 0:
        x = x + noise_sigma * rng.standard_normal(x.shape)
    return Sample(x=x.astype(np.float32), y=y.astype(np.int64), latent=latent)
