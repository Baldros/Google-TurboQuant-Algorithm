"""TurboQuant Stage 1 (MSE) quantizer.

Pipeline per vector ``x``:

    norm = ||x||
    u    = x / norm                      # unit vector
    y    = R u                           # random orthogonal rotation
    idx  = LloydMax.quantize(y)          # one shared codebook for every coordinate
    -- store (idx, norm) --
    u_hat = R^{-1} LloydMax.dequantize(idx)
    x_hat = norm * u_hat

The only per-vector side information is the single scalar ``norm`` - there is no stored
scale/zero-point *per coordinate or per block*, which is the whole point. Stage 2 (the
optional QJL residual sketch) is added in a later phase; this module is Stage 1 only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotation import random_orthogonal, apply_rotation, apply_inverse_rotation
from .scalar_quant import LloydMaxQuantizer


@dataclass
class QuantizedVectors:
    """Compressed representation: one index array + one norm per input vector."""

    indices: np.ndarray  # shape (n, d), small uint
    norms: np.ndarray    # shape (n,)


class TurboQuantMSE:
    """Stage-1 (minimum-MSE) TurboQuant quantizer for a fixed dimension and bit-rate."""

    def __init__(
        self,
        d: int,
        bits: int,
        *,
        seed: int = 0,
        codebook: LloydMaxQuantizer | None = None,
        rotation: np.ndarray | None = None,
        n_fit_vectors: int = 20000,
    ):
        self.d = d
        self.bits = bits
        self.rotation = (
            rotation if rotation is not None else random_orthogonal(d, seed=seed)
        )
        self.codebook = (
            codebook
            if codebook is not None
            else LloydMaxQuantizer.for_dimension(d, bits, n_vectors=n_fit_vectors, seed=seed)
        )

    def quantize(self, x: np.ndarray) -> QuantizedVectors:
        x = np.asarray(x, dtype=np.float64)
        norms = np.linalg.norm(x, axis=-1)
        # Guard against zero vectors (their direction is undefined; store norm 0).
        safe = np.where(norms == 0.0, 1.0, norms)
        u = x / safe[..., None]
        y = apply_rotation(u, self.rotation)
        idx = self.codebook.quantize(y)
        return QuantizedVectors(indices=idx, norms=norms)

    def dequantize(self, q: QuantizedVectors) -> np.ndarray:
        y_hat = self.codebook.dequantize(q.indices)
        u_hat = apply_inverse_rotation(y_hat, self.rotation)
        return u_hat * q.norms[..., None]

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Convenience: ``dequantize(quantize(x))``."""
        return self.dequantize(self.quantize(x))
