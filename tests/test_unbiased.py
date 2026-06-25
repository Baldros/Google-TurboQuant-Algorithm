"""Phase 1 - QJL is an unbiased inner-product estimator with 1/m variance.

Definition of done (docs/03): the estimate of ``<q, x>`` is **unbiased** (mean error
-> 0 over many seeds, a `hypothesis` property test) and its variance shrinks like ``1/m``.

We test *properties*, statistically, not a single point value:
  * over many independent projections the mean estimate converges to the true inner
    product (within a few standard errors);
  * the empirical variance matches the closed form and halves/quarters as ``m`` grows;
  * the sketch really is 1-bit and the stored norm is exact;
  * the all-pairs score matrix agrees with the paired estimate.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from turboquant import QJL, gaussian_projection, qjl_estimator_variance


def _rand_vectors(seed, d, scale_q=1.0, scale_x=1.0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(d) * scale_q
    x = rng.standard_normal(d) * scale_x
    return q, x


# --------------------------------------------------------------------------- #
# Unbiasedness
# --------------------------------------------------------------------------- #
def _mean_estimate_over_seeds(q, x, m, n_seeds, base_seed=0):
    """Average the QJL estimate of <q, x> over ``n_seeds`` independent projections."""
    d = q.shape[0]
    ests = np.empty(n_seeds)
    for i in range(n_seeds):
        qjl = QJL(d, m, seed=base_seed + i)
        ests[i] = qjl.estimate(q, qjl.sketch(x))
    return ests


def test_estimate_unbiased_over_many_seeds():
    # The literal DoD: mean error ~= 0 over many independent random projections.
    q, x = _rand_vectors(seed=1, d=24)
    truth = float(q @ x)
    m, n_seeds = 64, 600

    ests = _mean_estimate_over_seeds(q, x, m, n_seeds)
    mean = ests.mean()
    se = ests.std(ddof=1) / np.sqrt(n_seeds)  # standard error of the mean

    # Within 4 standard errors of the truth (a ~1-in-16k false-failure rate).
    assert abs(mean - truth) <= 4.0 * se, (
        f"mean estimate {mean:.4f} not within 4 SE ({4*se:.4f}) of truth {truth:.4f}"
    )


def test_debias_factor_is_necessary():
    # Sanity that the sqrt(pi/2)/m de-bias matters: dropping it gives a *biased* result
    # (off by exactly the sqrt(pi/2) factor on average), so the corrected one is closer.
    q, x = _rand_vectors(seed=2, d=24)
    truth = float(q @ x)
    m, n_seeds = 64, 600

    d = q.shape[0]
    corrected = np.empty(n_seeds)
    naive = np.empty(n_seeds)
    for i in range(n_seeds):
        qjl = QJL(d, m, seed=100 + i)
        sk = qjl.sketch(x)
        corrected[i] = qjl.estimate(q, sk)
        # naive: same sketch but without the ||x|| * sqrt(pi/2) calibration
        naive[i] = np.sum(qjl.project_query(q) * sk.signs) / m

    assert abs(corrected.mean() - truth) < abs(naive.mean() - truth)


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    d=st.sampled_from([8, 16, 32]),
    log2m=st.integers(4, 7),
)
def test_estimate_unbiased_property(seed, d, log2m):
    # For arbitrary q, x: one estimate that averages a large M of rows must land within
    # a few standard deviations (closed-form) of the truth - a consistency/unbiasedness
    # check across many random instances.
    q, x = _rand_vectors(seed=seed, d=d)
    truth = float(q @ x)
    m = 1 << (log2m + 6)  # 1024 .. 8192 rows -> tight variance

    qjl = QJL(d, m, seed=seed)
    est = qjl.estimate(q, qjl.sketch(x))
    sd = np.sqrt(qjl_estimator_variance(q, x, m))

    assert abs(est - truth) <= 6.0 * sd + 1e-9


# --------------------------------------------------------------------------- #
# Variance behaviour
# --------------------------------------------------------------------------- #
def test_empirical_variance_matches_closed_form():
    q, x = _rand_vectors(seed=3, d=24)
    m, n_seeds = 32, 4000

    ests = _mean_estimate_over_seeds(q, x, m, n_seeds, base_seed=10_000)
    emp_var = ests.var(ddof=1)
    closed = qjl_estimator_variance(q, x, m)

    assert emp_var == pytest.approx(closed, rel=0.12), (
        f"empirical var {emp_var:.5f} vs closed form {closed:.5f}"
    )


def test_variance_scales_like_one_over_m():
    # Doubling m should roughly halve the variance; closed form says exactly 1/m.
    q, x = _rand_vectors(seed=4, d=24)
    n_seeds = 4000

    var_m = _mean_estimate_over_seeds(q, x, 16, n_seeds, base_seed=1).var(ddof=1)
    var_2m = _mean_estimate_over_seeds(q, x, 32, n_seeds, base_seed=1).var(ddof=1)
    var_4m = _mean_estimate_over_seeds(q, x, 64, n_seeds, base_seed=1).var(ddof=1)

    # Ratios should be ~2 each step; allow generous Monte-Carlo slack.
    assert var_m / var_2m == pytest.approx(2.0, rel=0.25)
    assert var_2m / var_4m == pytest.approx(2.0, rel=0.25)


def test_closed_form_variance_is_positive():
    # pi/2 > 1 and Cauchy-Schwarz guarantee a positive variance for any q, x, m.
    rng = np.random.default_rng(5)
    for _ in range(50):
        d = int(rng.integers(2, 64))
        q = rng.standard_normal(d)
        x = rng.standard_normal(d)
        assert qjl_estimator_variance(q, x, m=8) > 0.0


# --------------------------------------------------------------------------- #
# Encoding / API invariants
# --------------------------------------------------------------------------- #
def test_sketch_is_one_bit_and_norm_exact():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((100, 32))
    qjl = QJL(32, 48, seed=0)
    sk = qjl.sketch(x)

    assert sk.signs.shape == (100, 48)
    assert sk.signs.dtype == np.int8
    assert set(np.unique(sk.signs)).issubset({-1, 1})
    # The only stored scalar is the exact norm.
    assert np.allclose(sk.norms, np.linalg.norm(x, axis=1))


def test_sign_sketch_is_scale_invariant():
    # sign(S (c*x)) == sign(S x) for c > 0: the direction is sketched, scale is the norm.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((20, 16))
    qjl = QJL(16, 40, seed=0)
    a = qjl.sketch(x).signs
    b = qjl.sketch(3.5 * x).signs
    assert np.array_equal(a, b)


def test_estimate_scores_matches_paired_estimate():
    # The diagonal of the all-pairs matrix must equal the paired estimate.
    rng = np.random.default_rng(8)
    d, n, m = 16, 12, 64
    qs = rng.standard_normal((n, d))
    xs = rng.standard_normal((n, d))
    qjl = QJL(d, m, seed=0)
    sk = qjl.sketch(xs)

    paired = qjl.estimate(qs, sk)
    scores = qjl.estimate_scores(qs, sk)
    assert np.allclose(np.diag(scores), paired)


def test_projection_reproducible_from_seed():
    assert np.array_equal(
        gaussian_projection(32, 64, seed=42), gaussian_projection(32, 64, seed=42)
    )
    assert not np.array_equal(
        gaussian_projection(32, 64, seed=1), gaussian_projection(32, 64, seed=2)
    )
