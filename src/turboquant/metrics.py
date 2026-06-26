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


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    """Mean cosine similarity between corresponding last-axis vectors of ``a`` and ``b``.

    Both are ``(..., d)``; cosine is taken over the last axis per row and averaged over all
    leading positions. This is the Phase-3 attention-output fidelity: how close the
    compressed-KV attention output is to the fp32 reference, per query position.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return float(np.mean(num / den))


def kl_divergence_rows(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    """Mean ``KL(p || q)`` over rows; ``p, q`` are ``(..., n)`` distributions on the last axis.

    Used on attention weight rows: how much the compressed-KV softmax distribution diverges
    from the reference. Terms with ``p == 0`` contribute nothing (the ``0 log 0`` convention),
    so masked (future) positions are handled correctly.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    kl = np.sum(np.where(p > 0, p * (np.log(p + eps) - np.log(q + eps)), 0.0), axis=-1)
    return float(np.mean(kl))


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Mean recall@k of approximate search against ground-truth neighbours.

    For each query, recall@k is ``|found[:k] ∩ truth[:k]| / k`` - the fraction of the
    true k nearest neighbours that the approximate search recovered in its own top-k.
    Averaged over all queries. ``found`` and ``truth`` are ``(nq, >=k)`` index arrays
    (``truth`` is typically the dataset's precomputed ground-truth neighbour list).
    """
    found = np.asarray(found)
    truth = np.asarray(truth)
    nq = found.shape[0]
    hits = 0
    for i in range(nq):
        hits += np.intersect1d(found[i, :k], truth[i, :k]).size
    return hits / (nq * k)
