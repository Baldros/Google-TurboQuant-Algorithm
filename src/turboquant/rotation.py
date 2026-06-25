"""Random orthogonal rotations.

TurboQuant's core trick: rotate every vector by a *fixed, data-independent* random
orthogonal matrix so its coordinates become statistically predictable (a known Beta
marginal that is near-Gaussian for large d). A fixed per-coordinate quantizer can then
be designed once, offline, with no per-vector or per-block side information.

Two rotations are provided:

* ``random_orthogonal`` - a dense Haar-random orthogonal matrix (O(d^2) to apply).
  This is the *oracle*: simplest to reason about and exactly uniform on the sphere.
* ``randomized_hadamard`` - a randomized Walsh-Hadamard transform (O(d log d) to
  apply, no d-by-d matrix stored). This is the *production* rotation; it only
  approximates Haar but is what makes the method fast. Requires d to be a power of 2.

Convention: vectors are rows. A batch ``x`` has shape ``(..., d)`` and the rotation
acts on the last axis.
"""

from __future__ import annotations

import numpy as np


def _as_rng(seed=None, rng=None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Dense Haar-random orthogonal matrix (the oracle rotation)
# --------------------------------------------------------------------------- #
def random_orthogonal(d: int, *, seed=None, rng=None) -> np.ndarray:
    """Return a ``(d, d)`` Haar-distributed orthogonal matrix.

    Built from the QR decomposition of a Gaussian matrix, with the sign
    correction ``Q[:, i] *= sign(R[i, i])`` that makes the result *exactly*
    Haar-uniform over O(d) (without it, QR has a sign bias and is not Haar).
    """
    rng = _as_rng(seed, rng)
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    # Remove the sign ambiguity of QR so Q is genuinely Haar-distributed.
    q *= np.sign(np.diag(r))
    return q


def apply_rotation(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate row-vectors ``x`` (shape ``(..., d)``) by ``q``: ``y = x @ q.T``."""
    return x @ q.T


def apply_inverse_rotation(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Inverse of :func:`apply_rotation`. Since ``q`` is orthogonal, ``x = y @ q``."""
    return y @ q


# --------------------------------------------------------------------------- #
# Randomized Walsh-Hadamard transform (the fast production rotation)
# --------------------------------------------------------------------------- #
def fast_hadamard_transform(x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Fast Walsh-Hadamard transform along the last axis.

    The last dimension ``d`` must be a power of two. With ``normalize=True`` the
    transform is orthogonal (it equals ``H / sqrt(d)`` for the {+-1} Hadamard
    matrix ``H``) and is its own inverse.
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    if d < 1 or (d & (d - 1)) != 0:
        raise ValueError(f"last dimension must be a power of two, got {d}")
    batch = x.shape[:-1]
    a = x.reshape(-1, d).copy()
    h = 1
    while h < d:
        a = a.reshape(-1, d // (2 * h), 2, h)
        top = a[:, :, 0, :].copy()
        bot = a[:, :, 1, :].copy()
        a[:, :, 0, :] = top + bot
        a[:, :, 1, :] = top - bot
        a = a.reshape(-1, d)
        h *= 2
    if normalize:
        a /= np.sqrt(d)
    return a.reshape(*batch, d)


def random_signs(d: int, *, seed=None, rng=None) -> np.ndarray:
    """Return a length-``d`` vector of random +-1 signs (the diagonal of D)."""
    rng = _as_rng(seed, rng)
    return rng.choice(np.array([-1.0, 1.0]), size=d)


def randomized_hadamard(
    x: np.ndarray, signs: np.ndarray, *, normalize: bool = True
) -> np.ndarray:
    """Apply the randomized Hadamard rotation ``R x = H D x`` (D = diag(signs))."""
    return fast_hadamard_transform(x * signs, normalize=normalize)


def inverse_randomized_hadamard(
    y: np.ndarray, signs: np.ndarray, *, normalize: bool = True
) -> np.ndarray:
    """Inverse of :func:`randomized_hadamard` (H and D are each their own inverse)."""
    return signs * fast_hadamard_transform(y, normalize=normalize)
