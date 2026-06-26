"""Phase 2 - vector search (maximum inner-product search) over quantized vectors.

This module isolates **quantizer quality**: every index here does an *exhaustive*
(brute-force) scan of the whole database, scoring each stored vector with the
approximate (quantized) inner product and returning the top-k. There is deliberately
no ANN structure (IVF / HNSW / graph) - that would mix in a second, orthogonal source
of error. The fair comparison point is FAISS ``IndexPQ``, which is *also* an exhaustive
ADC (asymmetric distance computation) scan; the only thing that differs is how a vector
is compressed. So "recall@k vs PQ at matched bit-rate" measures exactly one thing: is
TurboQuant's data-*oblivious* scalar quantization as good as PQ's *learned* codebooks?

Three encoders, all scored exhaustively:

* :class:`Stage1Index` - TurboQuant Stage 1 (rotation + per-coordinate Lloyd-Max).
  Scores ``<q, x_hat>`` against the reconstruction ``x_hat``. Data-oblivious; the direct
  competitor to PQ at ``b`` bits/coordinate.
* :class:`QJLIndex` - the Stage-2 QJL sign sketch on its own. Scores with the unbiased
  1-bit estimator ``||x|| sqrt(pi/2)/m <S q, sign(S x)>`` (1 bit/coordinate at ``m = d``).
* :class:`TurboQuantProdIndex` - the full Stage-1+2 estimator (``TurboQuantProd``): the
  exact Stage-1 score *plus* QJL applied to the Stage-1 residual ``r = u - u_hat``.

For unit-normalized vectors (cosine / "angular" search, e.g. GloVe) MIPS is equivalent to
nearest-neighbour, so these indices reproduce angular-NN recall directly.

Memory note: the in-RAM scan holds reconstructions / sign codes as ``float32`` for fast
BLAS matmuls. That is a *scan-time* convenience, not the compression claim - the advertised
bit-rate is the packed code (``b*d`` index bits, or ``m`` sign bits), measured in Phase 5.
"""

from __future__ import annotations

import numpy as np

from .qjl import QJL, _DEBIAS
from .quantizers import TurboQuantMSE

_DEFAULT_CHUNK = 100_000  # rows processed per build chunk (bounds peak memory)


def _topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the top-``k`` scores per row, ranked best-first.

    ``scores`` is ``(nq, n_db)``; the result is ``(nq, k)`` int64. Uses an
    ``argpartition`` (O(n_db)) to find the k candidates, then sorts only those k.
    """
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


class _FlatIndex:
    """Base class: exhaustive top-k driven by a subclass ``_score_block``."""

    def search(self, queries: np.ndarray, k: int, *, qblock: int = 128) -> np.ndarray:
        """Return ``(nq, k)`` int64 indices of the approximate top-k for each query.

        Queries are processed in blocks of ``qblock`` so the dense score matrix
        ``(qblock, n_db)`` never materialises for all queries at once.
        """
        queries = np.asarray(queries, dtype=np.float32)
        nq = queries.shape[0]
        out = np.empty((nq, min(k, self._n)), dtype=np.int64)
        for s in range(0, nq, qblock):
            e = min(s + qblock, nq)
            scores = self._score_block(queries[s:e])
            out[s:e] = _topk_from_scores(scores, k)
        return out

    def _score_block(self, qb: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class Stage1Index(_FlatIndex):
    """Exhaustive MIPS using TurboQuant Stage-1 reconstructions ``x_hat``.

    Score is the plain inner product ``<q, x_hat>``. This is the data-oblivious
    counterpart to FAISS PQ at ``b`` bits per coordinate (``b*d`` code bits per vector).
    """

    def __init__(
        self,
        d: int,
        bits: int,
        *,
        seed: int = 0,
        quantizer: TurboQuantMSE | None = None,
    ):
        self.d = d
        self.bits = bits
        self.quantizer = quantizer or TurboQuantMSE(d, bits, seed=seed)
        self._recon: np.ndarray | None = None
        self._n = 0

    def add(self, db: np.ndarray, *, chunk: int = _DEFAULT_CHUNK) -> "Stage1Index":
        db = np.asarray(db, dtype=np.float64)
        n = db.shape[0]
        self._recon = np.empty((n, self.d), dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            qv = self.quantizer.quantize(db[s:e])
            self._recon[s:e] = self.quantizer.dequantize(qv).astype(np.float32)
        self._n = n
        return self

    def _score_block(self, qb: np.ndarray) -> np.ndarray:
        return qb @ self._recon.T


class QJLIndex(_FlatIndex):
    """Exhaustive MIPS using the QJL sign sketch and its unbiased estimator.

    Stores ``m`` sign bits + one norm per vector (``m = d`` -> 1 bit/coordinate) and scores
    ``<q, x> ~= ||x|| sqrt(pi/2)/m <S q, sign(S x)>``. Query stays full precision (asymmetric).
    """

    def __init__(self, d: int, m: int, *, seed: int = 0, qjl: QJL | None = None):
        self.d = d
        self.m = m
        self.qjl = qjl or QJL(d, m, seed=seed)
        self._S = self.qjl.projection.astype(np.float32)  # (m, d)
        self._signs: np.ndarray | None = None  # (n, m) float32 in {-1, +1}
        self._norms: np.ndarray | None = None
        self._n = 0

    def add(self, db: np.ndarray, *, chunk: int = _DEFAULT_CHUNK) -> "QJLIndex":
        db = np.asarray(db, dtype=np.float64)
        n = db.shape[0]
        self._signs = np.empty((n, self.m), dtype=np.float32)
        self._norms = np.empty(n, dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sk = self.qjl.sketch(db[s:e])
            self._signs[s:e] = sk.signs.astype(np.float32)
            self._norms[s:e] = sk.norms.astype(np.float32)
        self._n = n
        return self

    def _score_block(self, qb: np.ndarray) -> np.ndarray:
        sq = qb @ self._S.T                       # (b, m)
        raw = sq @ self._signs.T                  # (b, n)  = <S q, sign(S x)>
        return (_DEBIAS / self.m) * raw * self._norms[None, :]


class TurboQuantProdIndex(_FlatIndex):
    """Exhaustive MIPS using the full Stage-1+2 estimator (``TurboQuantProd``).

    Score = exact Stage-1 term ``<q, x_hat>`` plus QJL applied to the Stage-1 residual
    ``r = u - u_hat`` (in the original, un-rotated unit-vector space)::

        <q, x> ~= <q, x_hat> + ||r|| sqrt(pi/2)/m <S q, sign(S r)>

    Costs ``b*d`` index bits + ``m`` sign bits per vector. The residual sketch de-biases the
    inner product that Stage-1 reconstruction alone leaves biased.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        m: int,
        *,
        seed: int = 0,
        quantizer: TurboQuantMSE | None = None,
        qjl: QJL | None = None,
    ):
        self.d = d
        self.bits = bits
        self.m = m
        self.quantizer = quantizer or TurboQuantMSE(d, bits, seed=seed)
        self.qjl = qjl or QJL(d, m, seed=seed + 1)
        self._S = self.qjl.projection.astype(np.float32)
        self._recon: np.ndarray | None = None       # x_hat, (n, d) float32
        self._rsigns: np.ndarray | None = None       # sign(S r), (n, m) float32
        self._rnorms: np.ndarray | None = None       # ||r||, (n,) float32
        self._n = 0

    def add(self, db: np.ndarray, *, chunk: int = _DEFAULT_CHUNK) -> "TurboQuantProdIndex":
        db = np.asarray(db, dtype=np.float64)
        n = db.shape[0]
        self._recon = np.empty((n, self.d), dtype=np.float32)
        self._rsigns = np.empty((n, self.m), dtype=np.float32)
        self._rnorms = np.empty(n, dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            block = db[s:e]
            qv = self.quantizer.quantize(block)
            x_hat = self.quantizer.dequantize(qv)        # (be, d)
            self._recon[s:e] = x_hat.astype(np.float32)
            # Residual of the *direction*: u - u_hat. For unit-norm input ||x||=1 so
            # x_hat == u_hat; in general we de-norm to compare directions.
            norms = np.linalg.norm(block, axis=-1, keepdims=True)
            safe = np.where(norms == 0.0, 1.0, norms)
            r = block / safe - x_hat / safe
            rk = self.qjl.sketch(r)
            self._rsigns[s:e] = rk.signs.astype(np.float32)
            self._rnorms[s:e] = rk.norms.astype(np.float32) * safe[:, 0].astype(np.float32)
        self._n = n
        return self

    def _score_block(self, qb: np.ndarray) -> np.ndarray:
        stage1 = qb @ self._recon.T                       # (b, n)
        sq = qb @ self._S.T                               # (b, m)
        raw = sq @ self._rsigns.T                          # (b, n)
        qjl = (_DEBIAS / self.m) * raw * self._rnorms[None, :]
        return stage1 + qjl


def exact_search(db: np.ndarray, queries: np.ndarray, k: int, *, qblock: int = 128) -> np.ndarray:
    """Brute-force exact MIPS top-k (the recall ceiling / ground-truth oracle)."""
    db = np.asarray(db, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    nq = queries.shape[0]
    out = np.empty((nq, min(k, db.shape[0])), dtype=np.int64)
    for s in range(0, nq, qblock):
        e = min(s + qblock, nq)
        out[s:e] = _topk_from_scores(queries[s:e] @ db.T, k)
    return out
