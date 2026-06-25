"""Distortion metrics and the paper's distortion bound.

The figure of merit for Stage 1 is the *normalized* mean squared error
``E[ ||x - x_hat||^2 / ||x||^2 ]``. For unit vectors the denominator is 1, so this is
just the average squared reconstruction error.

TurboQuant's guarantee is that this distortion stays within a small constant factor
(~2.7x) of the information-theoretic floor ``2^{-2b}`` for ``b`` bits per coordinate -
i.e. ``distortion <= ~2.7 * 2^{-2b}``. :func:`paper_distortion_bound` returns that
upper bound, which the Phase-0 test asserts we stay under.
"""

from __future__ import annotations

import numpy as np


def normalized_distortion(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Mean over the batch of ``||x - x_hat||^2 / ||x||^2``."""
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    num = np.sum((x - x_hat) ** 2, axis=-1)
    den = np.sum(x ** 2, axis=-1)
    return float(np.mean(num / den))


def mean_squared_error(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Mean over the batch of ``||x - x_hat||^2`` (un-normalized)."""
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    return float(np.mean(np.sum((x - x_hat) ** 2, axis=-1)))


def paper_distortion_bound(bits: int, factor: float = 2.7) -> float:
    """Upper bound on TurboQuant's normalized distortion at ``bits`` bits/coordinate.

    ``factor * 2^{-2*bits}``; ``factor ~= 2.7`` is the near-optimality constant the
    paper proves the rotation + optimal scalar quantizer achieves.
    """
    return factor * 2.0 ** (-2.0 * bits)
