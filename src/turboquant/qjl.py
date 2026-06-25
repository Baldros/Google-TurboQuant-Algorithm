"""QJL - the Quantized Johnson-Lindenstrauss sign sketch (Stage 2).

Stage 1 (``quantizers.TurboQuantMSE``) minimises *reconstruction* error, but its
inner-product estimate ``<q, x_hat>`` is **biased**. For applications that only need
inner products - attention scores, maximum-inner-product search - QJL provides an
**unbiased** estimate from a 1-bit-per-coordinate sketch.

The sketch
----------
Fix a Gaussian projection ``S ~ N(0,1)^{m x d}`` (shared and fixed, *not* stored per
vector). For a stored vector ``x`` keep only::

    signs = sign(S x)        # m sign bits
    norm  = ||x||            # one scalar

The asymmetric estimator
------------------------
Keep the *query* ``q`` in full precision (this is the "asymmetric" part - only the
stored side is quantized). Then ``<q, x>`` is estimated by::

    <q, x> ~= ||x|| * sqrt(pi/2) / m * <S q, sign(S x)>

Why it is unbiased. For one row ``s ~ N(0, I_d)`` the pair ``(a, b) = (s.x, s.q)`` is
jointly Gaussian with ``Var(a) = ||x||^2`` and ``Cov(a, b) = <q, x>``. A standard
identity for jointly-Gaussian zero-mean variables gives

    E[ sign(a) * b ] = sqrt(2/pi) * Cov(a, b) / sqrt(Var(a))
                     = sqrt(2/pi) * <q, x> / ||x|| .

Multiplying by ``||x|| * sqrt(pi/2)`` cancels everything except ``<q, x>``; averaging
``m`` rows leaves the mean unchanged and divides the variance by ``m``. The
``sqrt(pi/2)/m`` factor is exactly that de-bias constant (the ``E[|g|] = sqrt(2/pi)``
correction for a standard Gaussian ``g``).

Composition with Stage 1 (the full TurboQuant Stage-1+2 estimator) just applies this to
the Stage-1 *residual* ``r = u - u_hat`` and adds the exact Stage-1 contribution::

    <q, x> ~= <q, x_hat_stage1> + ||r|| * sqrt(pi/2)/m * <S q, sign(S r)>

so this module's :meth:`QJL.estimate` is the reusable building block; the residual
composition lives in the quantizer layer. **On its own, applied to a raw vector ``x``,
:meth:`QJL.estimate` is an unbiased estimator of ``<q, x>`` - which is what Phase 1
verifies.**

Convention: vectors are rows; a batch ``x`` has shape ``(..., d)`` and the projection
acts on the last axis. The sketch's leading shape mirrors the input's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# E[|g|] = sqrt(2/pi) for standard-Gaussian g; its reciprocal de-biases the sign sketch.
_DEBIAS = np.sqrt(np.pi / 2.0)


def _as_rng(seed=None, rng=None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def gaussian_projection(d: int, m: int, *, seed=None, rng=None) -> np.ndarray:
    """Return a fixed ``(m, d)`` Gaussian projection ``S ~ N(0, 1)``.

    Shared across every vector and never stored per-vector. ``m`` is the number of
    sign bits kept per vector (the paper's default is ``m = d``).
    """
    rng = _as_rng(seed, rng)
    return rng.standard_normal((m, d))


def _sign(a: np.ndarray) -> np.ndarray:
    """1-bit sign as ``int8`` in ``{-1, +1}`` (mapping the measure-zero 0 case to +1)."""
    return np.where(a >= 0.0, np.int8(1), np.int8(-1))


@dataclass
class QJLSketch:
    """A QJL sketch: the 1-bit sign code plus the single stored norm.

    ``signs`` has shape ``(..., m)`` with values in ``{-1, +1}`` (``int8`` - it packs to
    ``m`` bits in Phase 5); ``norms`` has shape ``(...)``.
    """

    signs: np.ndarray
    norms: np.ndarray


class QJL:
    """QJL sign-sketch encoder + asymmetric unbiased inner-product estimator."""

    def __init__(
        self,
        d: int,
        m: int,
        *,
        seed: int = 0,
        projection: np.ndarray | None = None,
    ):
        self.d = d
        self.m = m
        self.projection = (
            projection if projection is not None else gaussian_projection(d, m, seed=seed)
        )

    # -- encode -------------------------------------------------------------- #
    def sketch(self, x: np.ndarray) -> QJLSketch:
        """Encode ``x`` (shape ``(..., d)``) into its sign code and stored norm."""
        x = np.asarray(x, dtype=np.float64)
        norms = np.linalg.norm(x, axis=-1)
        signs = _sign(x @ self.projection.T)
        return QJLSketch(signs=signs, norms=norms)

    def project_query(self, q: np.ndarray) -> np.ndarray:
        """Project a full-precision query: ``S q`` (shape ``(..., m)``)."""
        q = np.asarray(q, dtype=np.float64)
        return q @ self.projection.T

    # -- estimate ------------------------------------------------------------ #
    def estimate(self, q: np.ndarray, sketch: QJLSketch) -> np.ndarray:
        """Unbiased estimate of ``<q, x>`` for paired/broadcast ``q`` and ``sketch``.

        ``q`` has shape ``(..., d)`` and ``sketch`` must broadcast against it
        (same leading shape), giving one estimate per row.
        """
        sq = self.project_query(q)  # (..., m)
        raw = np.sum(sq * sketch.signs, axis=-1)  # <S q, sign(S x)>
        return sketch.norms * _DEBIAS / self.m * raw

    def estimate_scores(self, q: np.ndarray, sketch: QJLSketch) -> np.ndarray:
        """All-pairs estimate: ``q`` is ``(nq, d)``, ``sketch`` covers ``nk`` vectors.

        Returns the ``(nq, nk)`` matrix of estimated inner products - the attention /
        MIPS score matrix. (``sketch.signs`` is ``(nk, m)``, ``sketch.norms`` is ``(nk,)``.)
        """
        sq = self.project_query(q)  # (nq, m)
        raw = sq @ sketch.signs.T.astype(np.float64)  # (nq, nk)
        return _DEBIAS / self.m * raw * sketch.norms[None, :]


def qjl_estimator_variance(q: np.ndarray, x: np.ndarray, m: int) -> float:
    """Closed-form variance of :meth:`QJL.estimate` for a single ``(q, x)`` pair.

    ``Var = (1/m) * [ (pi/2) * ||q||^2 ||x||^2 - <q, x>^2 ]`` - always positive (since
    ``pi/2 > 1`` and Cauchy-Schwarz bounds ``<q,x>^2 <= ||q||^2 ||x||^2``), and shrinks
    like ``1/m``. This is what Phase 1's variance test checks the empirics against.
    """
    q = np.asarray(q, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    qn2 = float(q @ q)
    xn2 = float(x @ x)
    ip = float(q @ x)
    return (np.pi / 2.0 * qn2 * xn2 - ip * ip) / m
