"""Phase 3 offline tests - attention primitives + key-score estimators on synthetic data.

No model download, no GPU. The *real-model* MSE-vs-Prod finding lives in
``scripts/run_phase3.py``; here we check the mechanical guarantees: softmax/mask/attention
correctness, the fidelity metrics, and that MSE-reconstructed attention converges to the
reference as bits grow.
"""

from __future__ import annotations

import numpy as np

from turboquant import (
    softmax,
    causal_additive_mask,
    attention,
    attention_from_scores,
    mse_key_scores,
    prod_key_scores,
    cosine_similarity_rows,
    kl_divergence_rows,
    TurboQuantMSE,
    QJL,
)


def _qkv(b, t, d, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((b, t, d)),
            rng.standard_normal((b, t, d)),
            rng.standard_normal((b, t, d)))


def test_softmax_normalizes():
    x = np.random.default_rng(0).standard_normal((4, 5))
    w = softmax(x, axis=-1)
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert np.all(w >= 0.0)


def test_causal_mask_blocks_future():
    mask = causal_additive_mask(4, 4)
    # query i may attend to key j iff j <= i
    for i in range(4):
        for j in range(4):
            assert mask[i, j] == (0.0 if j <= i else -np.inf)


def test_attention_weights_are_causal():
    Q, K, V = _qkv(2, 6, 8, seed=1)
    _, w = attention(Q, K, V, causal=True)
    # strictly-future weights must be exactly zero
    for i in range(6):
        assert np.allclose(w[:, i, i + 1:], 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_attention_matches_manual_small():
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((1, 3, 4))
    K = rng.standard_normal((1, 3, 4))
    V = rng.standard_normal((1, 3, 4))
    out, w = attention(Q, K, V, causal=True)
    scale = 1.0 / np.sqrt(4)
    # explicit reference for query position 2 (attends to keys 0,1,2)
    logits = (Q[0, 2] @ K[0].T) * scale
    e = np.exp(logits - logits.max())
    ref_w = e / e.sum()
    assert np.allclose(w[0, 2], ref_w)
    assert np.allclose(out[0, 2], ref_w @ V[0])


def test_cosine_rows_perfect_and_orthogonal():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert abs(cosine_similarity_rows(a, a) - 1.0) < 1e-9
    b = np.array([[0.0, 1.0], [3.0, 0.0]])
    assert abs(cosine_similarity_rows(a, b) - 0.0) < 1e-9


def test_kl_zero_for_identical_and_positive_otherwise():
    p = softmax(np.random.default_rng(3).standard_normal((5, 7)))
    assert abs(kl_divergence_rows(p, p)) < 1e-9
    q = softmax(np.random.default_rng(4).standard_normal((5, 7)))
    assert kl_divergence_rows(p, q) > 0.0


def test_mse_attention_converges_to_reference_with_bits():
    # On real-valued K/V, MSE-reconstructed attention should approach the fp64 reference
    # as bits grow (cosine up, KL down).
    d = 64
    Q, K, V = _qkv(8, 24, d, seed=5)
    scale = 1.0 / np.sqrt(d)
    out_ref, w_ref = attention(Q, K, V, causal=True)
    cosines, kls = [], []
    for b in (1, 2, 3, 4):
        quant = TurboQuantMSE(d, b, seed=0)
        k_hat = quant.reconstruct(K)
        v_hat = quant.reconstruct(V)
        s = np.matmul(Q, np.swapaxes(k_hat, -1, -2))
        out, w = attention_from_scores(s, v_hat, scale=scale, causal=True)
        cosines.append(cosine_similarity_rows(out, out_ref))
        kls.append(kl_divergence_rows(w_ref, w))
    assert cosines[0] < cosines[1] < cosines[2] < cosines[3]
    assert kls[0] > kls[1] > kls[2] > kls[3]
    assert cosines[-1] > 0.95  # 4-bit MSE attention is close to the reference


def test_prod_key_scores_shape_and_finiteness():
    d = 64
    Q, K, _ = _qkv(3, 16, d, seed=6)
    quant = TurboQuantMSE(d, 2, seed=0)
    qjl = QJL(d, d, seed=1)
    s = prod_key_scores(Q, K, quant, qjl)
    assert s.shape == (3, 16, 16)
    assert np.all(np.isfinite(s))


def test_prod_estimates_inner_product_unbiasedly():
    # Averaged over many QJL projections, the Prod score is an unbiased estimate of <q,k>.
    d = 64
    rng = np.random.default_rng(7)
    Q = rng.standard_normal((1, 1, d))
    K = rng.standard_normal((1, 1, d))
    true_ip = float(Q[0, 0] @ K[0, 0])
    quant = TurboQuantMSE(d, 1, seed=0)
    ests = []
    for s in range(200):
        qjl = QJL(d, d, seed=1000 + s)
        ests.append(float(prod_key_scores(Q, K, quant, qjl)[0, 0, 0]))
    # mean of the unbiased estimator should land near the true inner product
    assert abs(np.mean(ests) - true_ip) < 0.15 * (abs(true_ip) + 1.0)
