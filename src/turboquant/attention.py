"""Phase 3 - attention fidelity under TurboQuant KV-cache compression.

Pure-numpy scaled-dot-product attention plus the two **key-scoring** estimators we
compare head-to-head on a real model's K/V:

* **MSE keys** (:func:`mse_key_scores`): score ``<q, k>`` by ``<q, k_hat>`` using the
  Stage-1 reconstruction ``k_hat`` - *biased* but *low variance*.
* **Prod keys** (:func:`prod_key_scores`): the full TurboQuant-Prod estimate - Stage-1 at
  ``b-1`` bits plus a 1-bit QJL residual sketch - *unbiased* but *higher variance*. This is
  the exact same estimator as ``search.TurboQuantProdIndex``, here batched over heads.

**Values** are always Stage-1-reconstructed in both variants (the attention output is a
weighted *average of value vectors*, which needs reconstructions, not inner products). So
when we hold ``V_hat`` fixed and swap only the key-score path, any fidelity difference
isolates exactly the *"MSE-only vs MSE+QJL"* question on real attention - the KV-cache echo
of the Phase-2 search finding. At matched budget (Prod's ``b-1`` index bits + ``m=d`` sign
bits = ``b*d`` = MSE's ``b`` bits) the question is whether the unbiased-but-noisy QJL key
estimate helps or hurts the softmax.

All functions act on the last two axes ``(T, d)``; any number of leading axes (layers,
heads, batch) broadcast through numpy ``matmul``.
"""

from __future__ import annotations

import numpy as np

from .qjl import QJL, _DEBIAS
from .quantizers import TurboQuantMSE


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along ``axis`` (returns float64)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_additive_mask(tq: int, tk: int) -> np.ndarray:
    """``(tq, tk)`` additive mask: ``0`` where query may attend, ``-inf`` for future keys.

    The last ``tq`` queries align with the last ``tq`` keys (offset ``tk - tq``), i.e.
    standard causal self-attention: query at full-axis position ``i`` attends to keys
    ``0..i``. For square self-attention (``tq == tk``) this is the usual lower-triangular
    mask.
    """
    offset = tk - tq
    qi = np.arange(tq)[:, None] + offset
    kj = np.arange(tk)[None, :]
    return np.where(kj <= qi, 0.0, -np.inf)


def attention(Q, K, V, *, scale: float | None = None, causal: bool = True):
    """Reference scaled-dot-product attention. Returns ``(out, weights)``.

    ``Q,K,V`` are ``(..., T*, d)``; ``out`` is ``(..., Tq, d)`` and ``weights`` is
    ``(..., Tq, Tk)``. ``scale`` defaults to ``1/sqrt(d)``.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if scale is None:
        scale = 1.0 / np.sqrt(Q.shape[-1])
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale
    return attention_from_prescaled(scores, V, causal=causal)


def attention_from_scores(scores, V, *, scale: float, causal: bool = True):
    """Finish attention from *raw* (unscaled) approximate ``<q,k>`` scores.

    Multiplies by ``scale``, applies the causal mask, softmaxes, and averages ``V``.
    Returns ``(out, weights)``.
    """
    return attention_from_prescaled(np.asarray(scores, dtype=np.float64) * scale, V,
                                    causal=causal)


def attention_from_prescaled(scores, V, *, causal: bool = True):
    """Finish attention from already-scaled scores. Returns ``(out, weights)``."""
    scores = np.asarray(scores, dtype=np.float64)
    if causal:
        scores = scores + causal_additive_mask(scores.shape[-2], scores.shape[-1])
    w = softmax(scores, axis=-1)
    out = np.matmul(w, np.asarray(V, dtype=np.float64))
    return out, w


def mse_key_scores(Q, K, quant: TurboQuantMSE) -> np.ndarray:
    """Raw (unscaled) ``<q, k_hat>`` for Stage-1 MSE-reconstructed keys, batched."""
    k_hat = quant.reconstruct(K)
    return np.matmul(np.asarray(Q, dtype=np.float64), np.swapaxes(k_hat, -1, -2))


def prod_key_scores(Q, K, quant: TurboQuantMSE, qjl: QJL) -> np.ndarray:
    """Raw (unscaled) TurboQuant-Prod estimate of ``<q, k>``, batched over leading axes.

    ``<q,k> ~= <q, k_hat> + (sqrt(pi/2)/m) * ||r|| ||k|| * <S q, sign(S r)>`` with the
    unit-space residual ``r = k/||k|| - k_hat/||k||`` - identical to
    ``search.TurboQuantProdIndex``, just batched. ``quant`` carries the Stage-1 (``b-1``)
    bits; ``qjl`` carries the ``m`` residual sign bits.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    qv = quant.quantize(K)
    k_hat = quant.dequantize(qv)
    norms = np.linalg.norm(K, axis=-1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    r = K / safe - k_hat / safe
    rk = qjl.sketch(r)
    rnorms = rk.norms * safe[..., 0]                       # ||r|| * ||k||
    stage1 = np.matmul(Q, np.swapaxes(k_hat, -1, -2))      # (..., Tq, Tk)
    sq = np.matmul(Q, qjl.projection.T)                    # (..., Tq, m)
    raw = np.matmul(sq, np.swapaxes(rk.signs.astype(np.float64), -1, -2))  # (..., Tq, Tk)
    qjl_term = (_DEBIAS / qjl.m) * raw * rnorms[..., None, :]
    return stage1 + qjl_term
