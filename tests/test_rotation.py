"""Phase 0 - rotation invariants.

These test *properties* that must hold exactly (orthogonality, norm/inner-product
preservation, exact invertibility), not just a metric. A rotation bug here would
silently corrupt every downstream distortion number, so we pin it down first.
"""

import numpy as np
import pytest

from turboquant import (
    random_orthogonal,
    apply_rotation,
    apply_inverse_rotation,
    fast_hadamard_transform,
    random_signs,
    randomized_hadamard,
    inverse_randomized_hadamard,
)


@pytest.mark.parametrize("d", [2, 8, 64, 128])
def test_haar_matrix_is_orthogonal(d):
    q = random_orthogonal(d, seed=0)
    assert np.allclose(q @ q.T, np.eye(d), atol=1e-10)
    assert np.allclose(q.T @ q, np.eye(d), atol=1e-10)
    # Orthogonal => determinant is +-1.
    assert np.isclose(abs(np.linalg.det(q)), 1.0, atol=1e-8)


def test_haar_preserves_norm_and_inner_product():
    rng = np.random.default_rng(1)
    q = random_orthogonal(64, seed=2)
    x = rng.standard_normal((100, 64))
    y = rng.standard_normal((100, 64))
    xr = apply_rotation(x, q)
    yr = apply_rotation(y, q)
    # Norms preserved.
    assert np.allclose(np.linalg.norm(xr, axis=1), np.linalg.norm(x, axis=1))
    # Inner products preserved (rotation is an isometry).
    assert np.allclose(np.sum(xr * yr, axis=1), np.sum(x * y, axis=1))


def test_haar_round_trip():
    q = random_orthogonal(32, seed=3)
    x = np.random.default_rng(4).standard_normal((10, 32))
    assert np.allclose(apply_inverse_rotation(apply_rotation(x, q), q), x)


def test_haar_is_reproducible_from_seed():
    assert np.array_equal(random_orthogonal(16, seed=7), random_orthogonal(16, seed=7))


def test_fwht_matches_dense_hadamard():
    # For small d, compare the fast transform to an explicit Hadamard matmul.
    d = 8
    # Sylvester construction of the (unnormalized) Hadamard matrix.
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    x = np.random.default_rng(5).standard_normal((7, d))
    fast = fast_hadamard_transform(x, normalize=False)
    assert np.allclose(fast, x @ h.T)


@pytest.mark.parametrize("d", [2, 16, 128])
def test_randomized_hadamard_is_orthogonal_and_invertible(d):
    signs = random_signs(d, seed=9)
    x = np.random.default_rng(10).standard_normal((20, d))
    y = randomized_hadamard(x, signs)
    # Norm preserved (orthogonal rotation).
    assert np.allclose(np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1))
    # Exact inverse.
    assert np.allclose(inverse_randomized_hadamard(y, signs), x)


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fast_hadamard_transform(np.zeros((3, 6)))


def test_rotation_uniformizes_coordinate_variance():
    # A rotated unit vector has coordinate variance ~= 1/d (the key marginal fact
    # the fixed codebook relies on).
    d = 128
    rng = np.random.default_rng(11)
    q = random_orthogonal(d, seed=12)
    x = rng.standard_normal((20000, d))
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = apply_rotation(u, q)
    assert abs(np.mean(y)) < 0.01
    assert np.isclose(np.var(y), 1.0 / d, rtol=0.05)
