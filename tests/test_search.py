"""Phase 2 - quantized vector search (MIPS) recall, on small synthetic data.

These tests are **offline and fast** (no GloVe download, no FAISS): they pin the search
API and the recall *properties* the big scoreboard relies on -

  * ``recall_at_k`` counts intersections correctly;
  * ``exact_search`` really returns the true top-k;
  * Stage-1 search recall *rises with the bit-rate* and is high at 4 bits;
  * the QJL sign-sketch index ranks far above chance;
  * the full Stage-1+2 (``TurboQuantProd``) index is no worse than Stage-1 alone.

The end-to-end "recall@k >= FAISS PQ at matched bit-rate" claim is exercised by
``scripts/run_phase2.py`` on real GloVe-200; here we just guard the building blocks.
"""

import numpy as np
import pytest

from turboquant import (
    QJLIndex,
    Stage1Index,
    TurboQuantProdIndex,
    exact_search,
    recall_at_k,
)


def _unit_db(n, d, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #
def test_recall_perfect_and_zero():
    truth = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(truth.copy(), truth, k=3) == 1.0
    disjoint = np.array([[7, 8, 9], [10, 11, 12]])
    assert recall_at_k(disjoint, truth, k=3) == 0.0


def test_recall_partial_and_order_insensitive():
    truth = np.array([[1, 2, 3, 4]])
    found = np.array([[3, 1, 99, 98]])  # 2 of the true top-4 recovered, order irrelevant
    assert recall_at_k(found, truth, k=4) == pytest.approx(0.5)
    # k slices both arrays: only the first 2 truth ids count at k=2.
    assert recall_at_k(np.array([[2, 1]]), np.array([[1, 2, 3, 4]]), k=2) == 1.0


# --------------------------------------------------------------------------- #
# exact_search is the ground-truth oracle
# --------------------------------------------------------------------------- #
def test_exact_search_matches_bruteforce():
    db = _unit_db(500, 16, seed=1)
    q = _unit_db(20, 16, seed=2)
    k = 5
    got = exact_search(db, q, k)
    # Independent brute force.
    full = q @ db.T
    want = np.argsort(-full, axis=1)[:, :k]
    assert got.shape == (20, k)
    assert got.dtype == np.int64
    assert np.array_equal(got, want)


def test_exact_search_recall_is_one_against_itself():
    db = _unit_db(800, 24, seed=3)
    q = _unit_db(100, 24, seed=4)
    truth = exact_search(db, q, k=10)
    assert recall_at_k(truth, truth, k=10) == 1.0


# --------------------------------------------------------------------------- #
# Stage-1 search quality
# --------------------------------------------------------------------------- #
def test_stage1_recall_increases_with_bits():
    d, n, nq, k = 32, 3000, 200, 10
    db = _unit_db(n, d, seed=10)
    q = _unit_db(nq, d, seed=11)
    truth = exact_search(db, q, k)

    recalls = {}
    for bits in (1, 2, 4):
        idx = Stage1Index(d, bits, seed=0).add(db)
        found = idx.search(q, k)
        recalls[bits] = recall_at_k(found, truth, k)

    # Monotone in the bit-rate, and 4-bit recovers most of the true neighbours.
    assert recalls[1] < recalls[2] < recalls[4]
    assert recalls[4] > 0.75
    # All far above random chance (k / n).
    assert recalls[1] > 5 * (k / n)


def test_search_returns_valid_indices():
    d, n, k = 16, 400, 8
    db = _unit_db(n, d, seed=20)
    q = _unit_db(15, d, seed=21)
    found = Stage1Index(d, 3, seed=0).add(db).search(q, k)
    assert found.shape == (15, k)
    assert found.dtype == np.int64
    assert found.min() >= 0 and found.max() < n


# --------------------------------------------------------------------------- #
# QJL sign-sketch search and the full Stage-1+2 product estimator
# --------------------------------------------------------------------------- #
def test_qjl_index_beats_chance_by_a_wide_margin():
    d, n, nq, k = 32, 3000, 200, 10
    db = _unit_db(n, d, seed=30)
    q = _unit_db(nq, d, seed=31)
    truth = exact_search(db, q, k)

    found = QJLIndex(d, m=4 * d, seed=0).add(db).search(q, k)
    recall = recall_at_k(found, truth, k)
    assert recall > 0.30             # a 1-bit sketch still ranks well (~0.39 here)
    assert recall > 20 * (k / n)     # and dramatically above chance


def test_prod_index_no_worse_than_stage1_alone():
    d, n, nq, k = 32, 3000, 200, 10
    db = _unit_db(n, d, seed=40)
    q = _unit_db(nq, d, seed=41)
    truth = exact_search(db, q, k)

    s1 = Stage1Index(d, 2, seed=0).add(db)
    prod = TurboQuantProdIndex(d, bits=2, m=4 * d, seed=0).add(db)
    r_s1 = recall_at_k(s1.search(q, k), truth, k)
    r_prod = recall_at_k(prod.search(q, k), truth, k)

    # The residual sketch de-biases the inner product, so it should not hurt ranking.
    assert r_prod >= r_s1 - 0.02


def test_prod_diagonal_estimate_is_unbiased_enough_to_rank():
    # A self-recall sanity check: querying with the database vectors themselves must
    # put each vector at (or very near) rank 0 under the product estimator.
    d, n, k = 24, 1500, 1
    db = _unit_db(n, d, seed=50)
    prod = TurboQuantProdIndex(d, bits=3, m=4 * d, seed=0).add(db)
    found = prod.search(db[:200], k)
    self_hits = np.mean(found[:, 0] == np.arange(200))
    assert self_hits > 0.9
