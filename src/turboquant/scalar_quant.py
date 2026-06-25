"""Per-coordinate optimal scalar quantization (Lloyd-Max).

After rotation, every coordinate of a unit vector shares the *same* marginal
distribution (by spherical symmetry it is the same for all coordinates, and for a
random unit vector it is the Beta-derived density with variance ``1/d``, which is
near-Gaussian for large d). So we design **one** scalar quantizer for that marginal and
reuse it on every coordinate - the source of TurboQuant's "zero per-block overhead".

``fit_lloyd_max`` is the classic Lloyd-Max iteration (a.k.a. 1-D k-means / Lloyd's
algorithm): alternately (1) set decision boundaries to the midpoints between adjacent
reconstruction points, and (2) set each reconstruction point to the conditional mean of
its cell. It converges to the minimum-MSE fixed-rate scalar quantizer for the sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit_vector_coordinate_samples(d: int, n_vectors: int, *, seed=None, rng=None) -> np.ndarray:
    """Sample the marginal distribution of a *single* coordinate of a random unit vector.

    Draws ``n_vectors`` uniform points on the unit sphere in R^d and returns all of
    their coordinates flattened (``n_vectors * d`` i.i.d.-marginal samples; coordinates
    are identically distributed, so this is a cheap way to get many marginal samples).
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    g = rng.standard_normal((n_vectors, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g.ravel()


def fit_lloyd_max(
    samples: np.ndarray,
    n_levels: int,
    *,
    n_iter: int = 200,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a minimum-MSE ``n_levels``-point scalar quantizer to 1-D ``samples``.

    Returns ``(centroids, boundaries)`` with ``centroids`` sorted ascending
    (length ``n_levels``) and ``boundaries`` the ``n_levels - 1`` midpoints used as
    decision thresholds.
    """
    s = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    if s.size < n_levels:
        raise ValueError("need at least n_levels samples to fit")

    # Initialise reconstruction points at evenly spaced sample quantiles.
    probs = (np.arange(n_levels) + 0.5) / n_levels
    centroids = np.quantile(s, probs)

    for _ in range(n_iter):
        boundaries = 0.5 * (centroids[1:] + centroids[:-1])
        # Sample i falls in cell `idx[i]`; s is sorted so cells are contiguous.
        idx = np.searchsorted(boundaries, s)
        new = centroids.copy()
        # Conditional mean of each non-empty cell via segment boundaries on sorted s.
        edges = np.searchsorted(idx, np.arange(n_levels + 1))
        for k in range(n_levels):
            lo, hi = edges[k], edges[k + 1]
            if hi > lo:
                new[k] = s[lo:hi].mean()
        shift = np.abs(new - centroids).max()
        centroids = new
        if shift < tol:
            break

    boundaries = 0.5 * (centroids[1:] + centroids[:-1])
    return centroids, boundaries


@dataclass
class LloydMaxQuantizer:
    """A fixed-rate Lloyd-Max scalar quantizer (``bits`` bits, ``2**bits`` levels)."""

    bits: int
    centroids: np.ndarray  # shape (2**bits,), sorted ascending
    boundaries: np.ndarray  # shape (2**bits - 1,)

    @classmethod
    def fit(cls, samples: np.ndarray, bits: int, **kwargs) -> "LloydMaxQuantizer":
        centroids, boundaries = fit_lloyd_max(samples, 1 << bits, **kwargs)
        return cls(bits=bits, centroids=centroids, boundaries=boundaries)

    @classmethod
    def for_dimension(cls, d: int, bits: int, *, n_vectors: int = 20000, seed: int = 0,
                      **kwargs) -> "LloydMaxQuantizer":
        """Fit the optimal quantizer for the rotated-coordinate marginal of dim ``d``."""
        samples = unit_vector_coordinate_samples(d, n_vectors, seed=seed)
        return cls.fit(samples, bits, **kwargs)

    @property
    def n_levels(self) -> int:
        return 1 << self.bits

    def quantize(self, y: np.ndarray) -> np.ndarray:
        """Map values to codebook indices (smallest int dtype that fits the levels)."""
        idx = np.searchsorted(self.boundaries, y)
        dtype = np.uint8 if self.bits <= 8 else np.uint16
        return idx.astype(dtype)

    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        """Map codebook indices back to reconstruction values."""
        return self.centroids[idx]

    def reconstruct(self, y: np.ndarray) -> np.ndarray:
        """Quantize then dequantize (the rounding map ``Q(y)``)."""
        return self.dequantize(self.quantize(y))
