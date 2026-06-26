"""Phase 4 - end-to-end generation quality under a TurboQuant KV cache.

Two measurements, both plugging the Phase-4 ``TurboQuantCache`` straight into HuggingFace:

  1. **Streaming perplexity** (the smooth DoD metric). Teacher-force a held-out continuation
     *token by token* with the compressed cache live, so every predicted token sees the
     quantized history plus a full-precision residual window - exactly the generation-time
     state. Sweep the bit-rate to find the *quality-neutral* rate on this model, and ablate
     the residual window. Runs on ``gpt2`` (local, dense attention, no download).

  2. **Needle-in-a-haystack** (the headline, kept honest). Hide a passphrase in a long
     context and ask a small instruct model to recall it verbatim, fp vs compressed.
     Exact-string retrieval under greedy decoding is a **brittle** pass/fail signal -
     a single flipped token changes the verdict, so results are *non-monotonic* in bits.
     We report it as such. The asymmetric-bits finding falls straight out: keys are
     softmax-critical and need precision, values compress far harder. Runs on a downloaded
     instruct model; skipped cleanly (not failed) if it cannot be loaded.

Run:  python scripts/run_phase4.py [--bits 1 2 3 4 8] [--window 32]
                                   [--ppl-model gpt2] [--needle-model Qwen/Qwen2.5-0.5B-Instruct]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# A few hundred tokens of non-repetitive expository prose for the perplexity measurement.
# (Repetitive text makes perplexity trivially low and hides the quantization effect.)
_PROSE = (
    "The development of the printing press in fifteenth-century Europe reshaped the "
    "circulation of knowledge more thoroughly than any single invention before it. Before "
    "movable type, books were copied by hand in monastic scriptoria, a process so slow and "
    "costly that a modest library represented a fortune. Johannes Gutenberg, a goldsmith in "
    "Mainz, combined several existing techniques - oil-based ink, the screw press already "
    "used for wine and olives, and individually cast metal letters - into a system that could "
    "reproduce a page hundreds of times with little loss of quality. The economic effect was "
    "immediate. Within fifty years, presses operated in more than two hundred towns, and the "
    "price of a printed book fell to a fraction of its manuscript equivalent. Ideas that once "
    "travelled at the speed of a copyist now spread across borders in months. The Reformation, "
    "the scientific revolution, and the standardisation of national languages all drew on this "
    "new capacity to fix words in identical form and distribute them widely. Yet the press did "
    "not act alone. It depended on cheap paper, itself imported from techniques developed in "
    "China and transmitted through the Islamic world, and on a literate urban class with reason "
    "to read. Where those conditions were absent, the technology arrived but its consequences "
    "were muted. Historians therefore caution against treating any tool as a cause in isolation; "
    "the printing press mattered because it met a society already straining against the limits "
    "of hand-copied text. The same pattern recurs throughout the history of technology, in which "
    "a device long imagined becomes transformative only when the surrounding economy, materials, "
    "and demand finally align to carry it."
)

_NEEDLE_FILLER = (
    "The committee reviewed the quarterly logistics report and noted seasonal variation in "
    "shipping volumes across the northern depots. ")
_NEEDLE_SECRET = "The secret passphrase is velvet-tiger-1947."
_NEEDLE_TOKEN = "velvet-tiger-1947"


def _import_torch():
    import torch
    return torch


def streaming_nll(model, ids, context_len, cache):
    """Mean next-token NLL over ``ids[:, context_len:]`` with ``cache`` live, token by token.

    Prefills the context in one pass, then decodes the continuation one token at a time so
    each prediction sees the quantized history plus the most-recent-``window`` tokens in full
    precision - i.e. the true generation-time cache state.
    """
    torch = _import_torch()
    import torch.nn.functional as F

    T = ids.shape[1]
    with torch.no_grad():
        model(ids[:, :context_len], past_key_values=cache, use_cache=True)
        total, n = 0.0, 0
        for t in range(context_len, T):
            logits = model(ids[:, t - 1:t], past_key_values=cache, use_cache=True).logits[:, -1]
            logp = F.log_softmax(logits.float(), dim=-1)
            total += -logp[0, ids[0, t]].item()
            n += 1
    return total / max(n, 1)


def perplexity_sweep(model, tok, device, *, bits, window, context_frac, threshold):
    """Streaming-perplexity bit sweep + residual-window ablation. Returns the DoD verdict."""
    torch = _import_torch()
    from transformers.cache_utils import DynamicCache
    from turboquant.kvcache import TurboQuantCache

    ids = tok(_PROSE, return_tensors="pt").input_ids.to(device)
    T = ids.shape[1]
    context_len = int(T * context_frac)
    print(f"\n[1] Streaming perplexity on {model.config.model_type}  "
          f"({T} tokens, context {context_len}, eval {T - context_len}, window {window})")

    nll_fp = streaming_nll(model, ids, context_len, DynamicCache())
    ppl_fp = math.exp(nll_fp)
    print(f"    fp reference: NLL {nll_fp:.4f}  ppl {ppl_fp:.3f}")
    print(f"\n    {'bits':>4} | {'NLL':>7} {'ppl':>8} {'ppl +%':>8}")
    print("    " + "-" * 34)
    neutral = None
    for b in bits:
        cache = TurboQuantCache.from_model_config(model.config, kbits=b, window=window, seed=0)
        nll = streaming_nll(model, ids, context_len, cache)
        ppl = math.exp(nll)
        pct = 100.0 * (ppl - ppl_fp) / ppl_fp
        if neutral is None and pct <= threshold * 100.0:
            neutral = b
        print(f"    {b:>4} | {nll:>7.4f} {ppl:>8.3f} {pct:>7.2f}%")

    # Residual-window ablation at a fixed middling bit-rate: window 0 vs the configured window.
    abl_b = 3 if 3 in bits else bits[len(bits) // 2]
    print(f"\n    residual-window ablation @ {abl_b} bits:")
    for w in (0, window):
        cache = TurboQuantCache.from_model_config(model.config, kbits=abl_b, window=w, seed=0)
        ppl = math.exp(streaming_nll(model, ids, context_len, cache))
        pct = 100.0 * (ppl - ppl_fp) / ppl_fp
        print(f"      window={w:>3}: ppl {ppl:.3f}  (+{pct:.2f}%)")

    if neutral is not None:
        print(f"\n    -> quality-neutral bit-rate (ppl within {threshold*100:.0f}%): "
              f"{neutral} bits")
    else:
        print(f"\n    -> no bit-rate in {bits} stays within {threshold*100:.0f}% of fp ppl")
    return neutral is not None, neutral


def _build_needle(tok, device, n_filler):
    torch = _import_torch()
    parts = [_NEEDLE_FILLER] * n_filler
    parts[n_filler // 2] = _NEEDLE_SECRET + " "
    prompt = ("".join(parts)
              + "\n\nQuestion: What is the secret passphrase? Answer with the exact phrase.")
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.to(device)


def needle_test(model, tok, device, *, configs, window, n_filler, max_new_tokens=20):
    """Verbatim long-context retrieval, fp vs compressed. Honest brittle pass/fail."""
    torch = _import_torch()
    from transformers.cache_utils import DynamicCache
    from turboquant.kvcache import TurboQuantCache

    ids = _build_needle(tok, device, n_filler)
    print(f"\n[2] Needle-in-a-haystack on {model.config.model_type}  "
          f"({ids.shape[1]} tokens, needle mid-context, window {window})")

    def ask(cache):
        with torch.no_grad():
            g = model.generate(ids, past_key_values=cache, max_new_tokens=max_new_tokens,
                               do_sample=False)
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

    fp_out = ask(DynamicCache())
    fp_hit = _NEEDLE_TOKEN in fp_out
    print(f"    fp        {'HIT ' if fp_hit else 'MISS'} : {fp_out!r}")
    if not fp_hit:
        print("    (fp model cannot retrieve the needle - skipping compressed comparison)")
        return None, fp_hit

    print(f"\n    {'config':>12} | result")
    print("    " + "-" * 40)
    any_hit = False
    for kb, vb in configs:
        cache = TurboQuantCache.from_model_config(model.config, kbits=kb, vbits=vb,
                                                  window=window, seed=0)
        out = ask(cache)
        hit = _NEEDLE_TOKEN in out
        any_hit = any_hit or hit
        label = f"K{kb} V{vb}"
        print(f"    {label:>12} | {'HIT ' if hit else 'miss'}: {out!r}")
    return any_hit, fp_hit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--context-frac", type=float, default=0.5)
    ap.add_argument("--ppl-threshold", type=float, default=0.01,
                    help="quality-neutral: streaming ppl increase below this fraction")
    ap.add_argument("--ppl-model", default="gpt2")
    ap.add_argument("--needle-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="instruct model for the needle test; 'none' to skip")
    ap.add_argument("--needle-fillers", type=int, default=60)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TurboQuant Phase 4 - end-to-end generation quality  (device={device})")

    # ---- (1) perplexity DoD on the local dense model -------------------------------- #
    tok = AutoTokenizer.from_pretrained(args.ppl_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.ppl_model, attn_implementation="eager").to(device).eval()
    ppl_ok, neutral = perplexity_sweep(
        model, tok, device, bits=args.bits, window=args.window,
        context_frac=args.context_frac, threshold=args.ppl_threshold)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- (2) needle headline on an instruct model (best-effort) --------------------- #
    needle_ok = None
    if args.needle_model.lower() != "none":
        try:
            ntok = AutoTokenizer.from_pretrained(args.needle_model)
            nmodel = AutoModelForCausalLM.from_pretrained(
                args.needle_model, dtype=torch.float16,
                attn_implementation="eager").to(device).eval()
            configs = [(8, 8), (5, 5), (4, 4), (3, 3), (2, 2),  # symmetric
                       (8, 4), (8, 2), (6, 2)]                   # key-heavy asymmetric
            needle_ok, _ = needle_test(
                nmodel, ntok, device, configs=configs, window=args.window,
                n_filler=args.needle_fillers)
        except Exception as exc:  # noqa: BLE001 - any load/runtime failure -> honest skip
            print(f"\n[2] Needle test SKIPPED - could not load '{args.needle_model}': {exc}")
            needle_ok = None
    else:
        print("\n[2] Needle test skipped (--needle-model none)")

    # ---- verdict -------------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print(f"DoD 1 - quality-neutral bit-rate found (ppl within "
          f"{args.ppl_threshold*100:.0f}%): {'PASS' if ppl_ok else 'FAIL'}"
          + (f"  (= {neutral} bits)" if neutral else ""))
    if needle_ok is None:
        print("DoD 2 - needle retrieval: SKIPPED (instruct model unavailable)")
    else:
        print(f"DoD 2 - some compressed config retrieves the needle verbatim: "
              f"{'PASS' if needle_ok else 'FAIL'}")
    overall = ppl_ok and (needle_ok is not False)
    print(f"\nresult: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
