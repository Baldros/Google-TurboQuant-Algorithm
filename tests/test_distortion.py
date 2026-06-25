"""Phase 0 - the headline reproduction.

Definition of done (docs/03): quantize random unit vectors at b = 1,2,3,4 bits and
show the measured normalized MSE is **below the paper's upper bound** (~2.7 * 2^{-2b})
and within the near-optimality factor. The 3-bit point is the one called out in the
docs: d=128 -> ~0.034, comfortably under the ~0.043 bound.

We also check the measured distortion matches the known optimal Lloyd-Max distortion of
a unit-variance Gaussian (the large-d limit of the rotated-coordinate marginal), and
that the fast randomized-Hadamard rotation reaches the same distortion as the dense
Haar oracle.
"""

import numpy as np
import pytest

from turboquant import (
    TurboQuantMSE,
    LloydMaxQuantizer,
    normalized_distortion,
    paper_distortion_bound,
    random_signs,
    randomized_hadamard,
    inverse_randomized_hadamard,
)

D = 128
N_EVAL = 50000

# Optimal fixed-rate Lloyd-Max MSE for a unit-variance Gaussian (Max, 1960). Because
# the rotated coordinate has variance 1/d and the distortion is summed over d
# coordinates, the *normalized* distortion equals these constants, independent of d.
GAUSSIAN_LLOYD_MAX = {1: 0.3634, 2: 0.1175, 3: 0.03454, 4: 0.009497}


def _unit_vectors(n, d, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def eval_vectors():
    return _unit_vectors(N_EVAL, D, seed=123)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_distortion_below_paper_bound(bits, eval_vectors):
    tq = TurboQuantMSE(D, bits, seed=0)
    x_hat = tq.reconstruct(eval_vectors)
    dist = normalized_distortion(eval_vectors, x_hat)
    bound = paper_distortion_bound(bits)

    # The DoD: stay under the paper's distortion bound.
    assert dist <= bound, f"b={bits}: distortion {dist:.4f} exceeds bound {bound:.4f}"

    # And match the known optimal scalar-quantizer distortion (sanity on correctness).
    expected = GAUSSIAN_LLOYD_MAX[bits]
    assert dist == pytest.approx(expected, rel=0.12), (
        f"b={bits}: distortion {dist:.4f} far from optimal {expected:.4f}"
    )


def test_three_bit_headline_number(eval_vectors):
    # The specific number quoted in the docs: d=128, 3-bit -> ~0.034 (bound ~0.043).
    tq = TurboQuantMSE(D, 3, seed=0)
    dist = normalized_distortion(eval_vectors, tq.reconstruct(eval_vectors))
    assert 0.030 <= dist <= 0.040
    assert dist < paper_distortion_bound(3)


def test_distortion_decreases_with_bits(eval_vectors):
    dists = []
    for bits in (1, 2, 3, 4):
        tq = TurboQuantMSE(D, bits, seed=0)
        dists.append(normalized_distortion(eval_vectors, tq.reconstruct(eval_vectors)))
    assert all(b < a for a, b in zip(dists, dists[1:])), dists


def test_distortion_within_optimality_factor(eval_vectors):
    # The information-theoretic floor is ~2^{-2b}; we should be within ~2.7x of it.
    for bits in (1, 2, 3, 4):
        tq = TurboQuantMSE(D, bits, seed=0)
        dist = normalized_distortion(eval_vectors, tq.reconstruct(eval_vectors))
        floor = 2.0 ** (-2.0 * bits)
        assert dist / floor <= 2.7


def test_hadamard_rotation_matches_haar_distortion(eval_vectors):
    # The fast O(d log d) rotation should achieve essentially the same distortion as
    # the dense Haar oracle, using the same per-coordinate codebook.
    bits = 3
    codebook = LloydMaxQuantizer.for_dimension(D, bits, seed=0)

    haar = TurboQuantMSE(D, bits, seed=0, codebook=codebook)
    haar_dist = normalized_distortion(eval_vectors, haar.reconstruct(eval_vectors))

    signs = random_signs(D, seed=0)
    y = randomized_hadamard(eval_vectors, signs)
    y_hat = codebook.reconstruct(y)
    x_hat = inverse_randomized_hadamard(y_hat, signs)
    hadamard_dist = normalized_distortion(eval_vectors, x_hat)

    assert hadamard_dist == pytest.approx(haar_dist, rel=0.05)
    assert hadamard_dist <= paper_distortion_bound(bits)


def test_norm_is_preserved_through_quantization():
    # Only the direction is quantized; the stored scalar norm is exact.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1000, D)) * rng.uniform(0.1, 10.0, (1000, 1))
    tq = TurboQuantMSE(D, 4, seed=0)
    q = tq.quantize(x)
    assert np.allclose(q.norms, np.linalg.norm(x, axis=1))
