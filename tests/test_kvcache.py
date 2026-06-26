"""Phase 4 offline tests - the TurboQuant KV cache, on synthetic tensors (no model/GPU).

These check the cache *mechanics* against the validated ``TurboQuantMSE`` quantizer:
full-precision window passthrough, exactly-once quantization of evicted tokens, that
token-by-token streaming (generation) reconstructs identically to a one-shot prefill, and
that asymmetric K/V bit-rates are honoured. The real-model quality numbers live in
``scripts/run_phase4.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from turboquant import TurboQuantMSE  # noqa: E402
from turboquant.kvcache import TurboQuantCache, TurboQuantLayer  # noqa: E402


def _kv(b, h, t, d, seed=0):
    g = np.random.default_rng(seed)
    k = torch.from_numpy(g.standard_normal((b, h, t, d)))  # float64
    v = torch.from_numpy(g.standard_normal((b, h, t, d)))
    return k, v


def _layer(d, kbits, vbits, window):
    kq = TurboQuantMSE(d, kbits, seed=0)
    vq = kq if vbits == kbits else TurboQuantMSE(d, vbits, seed=0)
    return TurboQuantLayer(kq, vq, window), kq, vq


def test_window_passthrough_is_exact():
    # window >= sequence length -> nothing is quantized -> output equals input bit-for-bit.
    k, v = _kv(1, 2, 10, 8, seed=1)
    layer, _, _ = _layer(8, kbits=2, vbits=2, window=64)
    ok, ov = layer.update(k, v)
    assert torch.equal(ok, k)
    assert torch.equal(ov, v)
    assert layer.get_seq_length() == 10
    assert layer._fp_tail == 10


def test_eviction_quantizes_old_keeps_window_exact():
    d, T, W = 8, 20, 5
    k, v = _kv(1, 3, T, d, seed=2)
    layer, kq, vq = _layer(d, kbits=2, vbits=2, window=W)
    ok, ov = layer.update(k, v)
    # the last W tokens are untouched...
    assert torch.equal(ok[..., -W:, :], k[..., -W:, :])
    assert torch.equal(ov[..., -W:, :], v[..., -W:, :])
    # ...and the older T-W tokens match a direct TurboQuant reconstruction
    exp_k = torch.from_numpy(kq.reconstruct(k[..., :T - W, :].numpy()))
    exp_v = torch.from_numpy(vq.reconstruct(v[..., :T - W, :].numpy()))
    assert torch.allclose(ok[..., :T - W, :], exp_k)
    assert torch.allclose(ov[..., :T - W, :], exp_v)
    # the evicted region is genuinely lossy (not accidentally a no-op)
    assert not torch.allclose(ok[..., :T - W, :], k[..., :T - W, :])


def test_streaming_equals_one_shot():
    # Decoding token-by-token must reconstruct identically to a single prefill: each token
    # is quantized exactly once, the moment it leaves the window, regardless of chunking.
    d, T, W = 8, 17, 4
    k, v = _kv(1, 2, T, d, seed=3)

    one, _, _ = _layer(d, 3, 3, W)
    ok1, ov1 = one.update(k, v)

    stream, _, _ = _layer(d, 3, 3, W)
    ok2 = ov2 = None
    for t in range(T):
        ok2, ov2 = stream.update(k[..., t:t + 1, :], v[..., t:t + 1, :])
    assert ok2.shape == ok1.shape
    assert torch.allclose(ok1, ok2)
    assert torch.allclose(ov1, ov2)
    assert stream.get_seq_length() == T


def test_asymmetric_bits_use_distinct_quantizers():
    d, T, W = 8, 16, 2
    k, v = _kv(1, 1, T, d, seed=4)
    layer, kq, vq = _layer(d, kbits=4, vbits=1, window=W)
    assert kq is not vq
    ok, ov = layer.update(k, v)
    exp_k = torch.from_numpy(kq.reconstruct(k[..., :T - W, :].numpy()))
    exp_v = torch.from_numpy(vq.reconstruct(v[..., :T - W, :].numpy()))
    assert torch.allclose(ok[..., :T - W, :], exp_k)
    assert torch.allclose(ov[..., :T - W, :], exp_v)
    # 1-bit values must be coarser than 4-bit keys (larger reconstruction error)
    err_k = (ok[..., :T - W, :] - k[..., :T - W, :]).abs().mean()
    err_v = (ov[..., :T - W, :] - v[..., :T - W, :]).abs().mean()
    assert err_v > err_k


def test_from_model_config_shapes_and_sharing():
    from transformers import GPT2Config

    cfg = GPT2Config(n_layer=3, n_head=4, n_embd=32)
    cache = TurboQuantCache.from_model_config(cfg, kbits=2, window=4)
    assert len(cache.layers) == 3
    assert cache.head_dim == 8 and cache.kbits == 2 and cache.vbits == 2
    # symmetric K/V share a single oblivious quantizer across every layer
    q0 = cache.layers[0].kquant
    assert all(l.kquant is q0 and l.vquant is q0 for l in cache.layers)

    asym = TurboQuantCache.from_model_config(cfg, kbits=4, vbits=2, window=4)
    assert asym.layers[0].kquant is not asym.layers[0].vquant


def test_seq_length_tracks_streaming_updates():
    d = 8
    k, v = _kv(1, 2, 6, d, seed=5)
    layer, _, _ = _layer(d, 2, 2, window=3)
    layer.update(k[..., :4, :], v[..., :4, :])
    assert layer.get_seq_length() == 4
    layer.update(k[..., 4:, :], v[..., 4:, :])
    assert layer.get_seq_length() == 6
    assert layer._fp_tail == 3  # window is full
