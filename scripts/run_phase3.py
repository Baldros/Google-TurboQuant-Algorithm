"""Phase 3 - KV-cache attention fidelity on a real model (GPT-2).

Captures the real Q/K/V of every head in every layer for a piece of text, compresses K
and V with TurboQuant, recomputes attention, and measures how close the compressed-cache
attention is to the fp32 reference. Two things are reported:

  1. **Compressed-cache fidelity** (MSE variant, both K and V quantized at ``b`` bits):
     mean attention-output cosine similarity and mean softmax KL vs the fp32 reference.
     This is the headline KV-cache number - how lossless is TurboQuant at 2-4 bits.

  2. **MSE-only vs MSE+QJL on the *keys*** (the central gotcha). Holding the
     MSE-reconstructed values fixed, we score ``<q,k>`` two ways at the *same* bit budget:
       * MSE   : ``<q, k_hat>``                       (biased, low variance)
       * Prod  : MSE@``b-1`` + 1-bit QJL residual     (unbiased, higher variance)
     The paper's KV lesson - and our Phase-2 search finding - predicts **MSE wins**: an
     unbiased-but-noisy inner product is the wrong tool for a softmax that must discriminate
     scores finely. This script confirms it directly on attention.

Extraction is self-validated: recomputed keys are checked against the model's own KV cache,
and our softmax against the model's returned attention weights, so the Q we recompute (the
cache does not store it) is trustworthy.

Run:  python scripts/run_phase3.py [--bits 2 3 4] [--text-file FILE]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from turboquant import (  # noqa: E402
    TurboQuantMSE,
    QJL,
    attention,
    attention_from_scores,
    mse_key_scores,
    prod_key_scores,
    cosine_similarity_rows,
    kl_divergence_rows,
)

_DEFAULT_TEXT = (
    "The history of science is the study of the development of human understanding of the "
    "natural world. Early civilisations made systematic observations of the stars and the "
    "seasons, recording them on clay tablets and in stone. The scientific method, with its "
    "emphasis on controlled experiment and reproducible measurement, emerged only gradually, "
    "and transformed medicine, physics, and engineering over the following centuries. Today "
    "the same method underwrites everything from vaccine design to the quantisation of the "
    "key-value caches that make large language models affordable to run at scale."
)


def extract_qkv(model, tok, text: str, device: str):
    """Return (Q, K, V, scale, val) with Q/K/V shaped (n_layer*n_head, T, head_dim).

    ``val`` is a dict of validation residuals (max abs diff of recomputed keys vs the model
    cache, and of our softmax vs the model's attention weights).
    """
    import torch

    cfg = model.config
    n_head, head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
    ids = tok(text, return_tensors="pt").input_ids.to(device)

    hidden = {}

    def mk(i):
        def hook(mod, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            hidden[i] = hs.detach()
        return hook

    handles = [blk.attn.register_forward_pre_hook(mk(i), with_kwargs=True)
               for i, blk in enumerate(model.transformer.h)]
    with torch.no_grad():
        out = model(ids, use_cache=True, output_attentions=True)
    for h in handles:
        h.remove()

    cache = out.past_key_values
    T = ids.shape[1]
    Qs, Ks, Vs = [], [], []
    k_diff = 0.0
    with torch.no_grad():
        for i, blk in enumerate(model.transformer.h):
            W = blk.attn.c_attn.weight        # (n_embd, 3*n_embd)
            b = blk.attn.c_attn.bias          # (3*n_embd,)
            qkv = hidden[i] @ W + b           # (1, T, 3*n_embd)
            q, k, _ = qkv.split(cfg.n_embd, dim=-1)

            def heads(x):  # (1,T,n_embd) -> (n_head, T, head_dim)
                return x.view(T, n_head, head_dim).permute(1, 0, 2).contiguous()

            q = heads(q); k = heads(k)
            k_cache = cache.layers[i].keys[0]   # (n_head, T, head_dim)
            v_cache = cache.layers[i].values[0]
            k_diff = max(k_diff, float((k - k_cache).abs().max()))
            Qs.append(q.cpu().numpy())
            Ks.append(k_cache.cpu().numpy())
            Vs.append(v_cache.cpu().numpy())

    Q = np.concatenate(Qs, 0).astype(np.float64)   # (L*H, T, d)
    K = np.concatenate(Ks, 0).astype(np.float64)
    V = np.concatenate(Vs, 0).astype(np.float64)
    scale = head_dim ** -0.5

    # validate attention math against the model's own weights (layer 0 heads)
    _, w_ours = attention(Q[:n_head], K[:n_head], V[:n_head], scale=scale, causal=True)
    w_model = out.attentions[0][0].cpu().numpy()
    w_diff = float(np.abs(w_ours - w_model).max())
    return Q, K, V, scale, {"key_vs_cache": k_diff, "softmax_vs_model": w_diff, "T": T,
                            "heads": Q.shape[0]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--cos-threshold", type=float, default=0.99,
                    help="DoD: MSE compressed-cache cosine at max bits must exceed this")
    args = ap.parse_args()

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    text = (pathlib.Path(args.text_file).read_text(encoding="utf-8")
            if args.text_file else _DEFAULT_TEXT)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, attn_implementation="eager").to(device)
    model.eval()
    d = model.config.n_embd // model.config.n_head

    Q, K, V, scale, val = extract_qkv(model, tok, text, device)
    print(f"\nTurboQuant Phase 3 - attention fidelity on {args.model}  (device={device})")
    print(f"  {val['heads']} heads x {val['T']} tokens x {d} head-dim")
    print(f"  extraction check: max|recomputed K - cache K| = {val['key_vs_cache']:.2e}"
          f"   max|our softmax - model attn| = {val['softmax_vs_model']:.2e}")
    if val["key_vs_cache"] > 1e-3 or val["softmax_vs_model"] > 1e-3:
        print("  WARNING: extraction validation residual is large; numbers below are suspect.")

    out_ref, w_ref = attention(Q, K, V, scale=scale, causal=True)

    print(f"\n{'bits':>4} | {'MSE cos':>8} {'MSE KL':>8} | {'Prod cos':>8} {'Prod KL':>8} | gotcha")
    print("-" * 62)
    mse_cos_at_max = 0.0
    gotcha_holds = True
    for b in args.bits:
        quant = TurboQuantMSE(d, b, seed=0)              # shared, oblivious
        v_hat = quant.reconstruct(V)                     # identical V for both variants
        # MSE keys
        s_mse = mse_key_scores(Q, K, quant)
        out_mse, w_mse = attention_from_scores(s_mse, v_hat, scale=scale, causal=True)
        cos_mse = cosine_similarity_rows(out_mse, out_ref)
        kl_mse = kl_divergence_rows(w_ref, w_mse)
        # Prod keys: MSE@(b-1) + 1-bit QJL residual  => same b*d budget
        quant_km1 = TurboQuantMSE(d, max(b - 1, 1), seed=0)
        qjl = QJL(d, d, seed=1)
        s_prod = prod_key_scores(Q, K, quant_km1, qjl)
        out_prod, w_prod = attention_from_scores(s_prod, v_hat, scale=scale, causal=True)
        cos_prod = cosine_similarity_rows(out_prod, out_ref)
        kl_prod = kl_divergence_rows(w_ref, w_prod)

        win = cos_mse > cos_prod and kl_mse < kl_prod
        gotcha_holds = gotcha_holds and win
        if b == max(args.bits):
            mse_cos_at_max = cos_mse
        print(f"{b:>4} | {cos_mse:>8.4f} {kl_mse:>8.4f} | {cos_prod:>8.4f} {kl_prod:>8.4f} | "
              f"{'MSE wins' if win else 'PROD wins'}")

    print("-" * 62)
    fidelity_ok = mse_cos_at_max >= args.cos_threshold
    print(f"\nDoD 1 - compressed-cache fidelity: MSE cosine @ {max(args.bits)} bits = "
          f"{mse_cos_at_max:.4f}  (>= {args.cos_threshold}? {'PASS' if fidelity_ok else 'FAIL'})")
    print(f"DoD 2 - MSE-only > MSE+QJL on attention at every bit-rate: "
          f"{'PASS' if gotcha_holds else 'FAIL'}")
    print(f"\nresult: {'PASS' if (fidelity_ok and gotcha_holds) else 'FAIL'}")
    return 0 if (fidelity_ok and gotcha_holds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
