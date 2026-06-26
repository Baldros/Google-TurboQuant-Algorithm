"""Phase 4 - a TurboQuant KV cache for HuggingFace generation.

A drop-in HuggingFace ``Cache`` that fake-quantizes the key/value cache with TurboQuant
Stage-1 (MSE) while keeping a *residual window* of the most recent tokens in full
precision. Keys and values may use different bit-rates (asymmetric K/V).

This is the Phase-4 artifact: it lets us measure **end-to-end generation quality** under
TurboQuant compression by plugging straight into ``model.generate()`` or a manual
``model(..., past_key_values=cache)`` forward.

Importing this module requires ``torch`` + ``transformers``; it is deliberately *not* part
of the top-level ``turboquant`` package import (the numeric core stays dependency-light).
Use ``from turboquant.kvcache import TurboQuantCache``.

Design
------
* **One shared, oblivious quantizer pair** (keys at ``kbits``, values at ``vbits``) is
  built once from the head dimension and reused across every layer and head - the codebook
  and rotation depend only on ``(head_dim, bits, seed)``. That sharing *is* TurboQuant's
  zero-per-block-overhead property; it is not a shortcut.
* **Residual window.** Each layer keeps the most recent ``window`` tokens in full
  precision (standard in streaming KV quantization - the recent tokens carry the most
  weight in the next softmax and are cheapest to keep exact). Older tokens are
  *fake-quantized* (quantized then immediately dequantized) exactly once, the moment they
  fall out of the window, and stored as the reconstructed tensor.
* **Why fake-quantization.** Phase 4 measures *quality*, not memory (packing/throughput is
  Phase 5). The reconstructed tensor is bit-for-bit what attention would read back from a
  real packed store, so the quality number is faithful while the experiment stays fast and
  stays on-device. Bit-rate is accounted analytically from ``kbits``/``vbits``/``window``.

The layer subclasses :class:`DynamicLayer` and keeps ``self.keys`` / ``self.values`` as the
canonical *reconstructed* tensors that attention reads, so every inherited bit of
``generate()`` plumbing (masking, cropping, beam reorder) keeps working unchanged. An
internal counter ``_fp_tail`` tracks how many trailing tokens are still full-precision
(the live window) and therefore not yet quantized.

Shapes follow the HF convention ``(batch, n_kv_heads, seq, head_dim)``; TurboQuant acts on
the last axis (``head_dim``) and broadcasts over batch, heads, and sequence.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from .quantizers import TurboQuantMSE


def _fake_quantize(quant: TurboQuantMSE, t: torch.Tensor) -> torch.Tensor:
    """Quantize then dequantize a torch tensor ``(..., d)``; preserve dtype/device/shape."""
    arr = t.detach().to(torch.float32).cpu().numpy().astype(np.float64)
    recon = quant.reconstruct(arr)
    return torch.from_numpy(recon).to(dtype=t.dtype, device=t.device)


class TurboQuantLayer(DynamicLayer):
    """One transformer layer's KV cache: quantized history + a full-precision window."""

    is_sliding = False

    def __init__(self, kquant: TurboQuantMSE, vquant: TurboQuantMSE, window: int):
        super().__init__()
        self.kquant = kquant
        self.vquant = vquant
        self.window = int(window)
        self._fp_tail = 0  # trailing tokens still in full precision (the live window)

    def lazy_initialization(self, key_states, value_states) -> None:
        super().lazy_initialization(key_states, value_states)
        self._fp_tail = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        n_new = key_states.shape[-2]
        # New tokens arrive in full precision; append them to the canonical tensors.
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        self._fp_tail += n_new

        # Everything beyond the most recent `window` tokens must be quantized now. The
        # full-precision region is the last `_fp_tail` tokens; quantize all but the final
        # `window` of them - exactly once each.
        to_quant = self._fp_tail - self.window
        if to_quant > 0:
            total = self.keys.shape[-2]
            start = total - self._fp_tail          # first full-precision token
            stop = start + to_quant                # exclusive; keep [stop:] in fp
            self.keys = torch.cat(
                [self.keys[..., :start, :],
                 _fake_quantize(self.kquant, self.keys[..., start:stop, :]),
                 self.keys[..., stop:, :]], dim=-2)
            self.values = torch.cat(
                [self.values[..., :start, :],
                 _fake_quantize(self.vquant, self.values[..., start:stop, :]),
                 self.values[..., stop:, :]], dim=-2)
            self._fp_tail = self.window

        return self.keys, self.values

    def crop(self, max_length: int) -> None:
        super().crop(max_length)
        # Tokens are dropped from the tail (the full-precision window) first.
        self._fp_tail = min(self._fp_tail, self.get_seq_length())

    def reset(self) -> None:
        super().reset()
        self._fp_tail = 0


class TurboQuantCache(DynamicCache):
    """A :class:`DynamicCache` whose layers fake-quantize their KV history.

    Build it with :meth:`from_model_config`; the raw constructor takes a pre-built list of
    :class:`TurboQuantLayer` (one per decoder layer).
    """

    def __init__(self, layers):
        # Bypass DynamicCache.__init__ (which would build plain DynamicLayers); install our
        # pre-built TurboQuantLayers straight onto the Cache base.
        Cache.__init__(self, layers=list(layers))

    @classmethod
    def from_model_config(
        cls,
        config,
        *,
        kbits: int,
        vbits: int | None = None,
        window: int = 32,
        seed: int = 0,
    ) -> "TurboQuantCache":
        """Build a cache sized for ``config`` with shared, oblivious K/V quantizers.

        ``kbits``/``vbits`` are the asymmetric key/value bit-rates (``vbits`` defaults to
        ``kbits``); ``window`` is the number of most-recent tokens kept in full precision.
        One quantizer pair is shared across every layer (and reused for V when
        ``vbits == kbits``), which is exactly TurboQuant's zero-per-block-overhead claim.
        """
        decoder = config.get_text_config(decoder=True)
        n_layers = decoder.num_hidden_layers
        head_dim = getattr(decoder, "head_dim", None) or (
            decoder.hidden_size // decoder.num_attention_heads)
        vbits = kbits if vbits is None else vbits
        kquant = TurboQuantMSE(head_dim, kbits, seed=seed)
        vquant = kquant if vbits == kbits else TurboQuantMSE(head_dim, vbits, seed=seed)
        layers = [TurboQuantLayer(kquant, vquant, window) for _ in range(n_layers)]
        cache = cls(layers)
        cache.kbits, cache.vbits, cache.window = kbits, vbits, window
        cache.head_dim = head_dim
        return cache
