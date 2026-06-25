# 03 — Recommended stack, architecture, and phased roadmap

This is the engineering plan. It is deliberately ordered so that the **cheapest, most
verifiable** work happens first and each phase produces a number we can check against the paper
(the "scoreboard" in `docs/01 §6`). Hardware feasibility for *your* machine is in `docs/05` —
the short version: an RTX 3060 (12 GB) + 16 GB RAM is enough for everything except the
heaviest production-kernel work, and is literally the card the best community repo was tested
on.

## Recommended stack

### Core / reference (Phases 0–4) — pure Python, no compiler needed

| Concern | Choice | Why |
|---------|--------|-----|
| Language | **Python 3.11** | Matches the papers and every reference repo; fastest path to a *correct* implementation |
| Tensor lib | **PyTorch** (CUDA 12.x wheel, `cu124`/`cu128`, builds for sm_86) | GPU-ready, what HF + the references use |
| Numeric oracle | **NumPy + SciPy** | Tiny dependency-light "ground truth" for Lloyd–Max, the Beta marginal, and unbiasedness checks |
| Fast rotation | **`fast-hadamard-transform`** (Dao lab) or a hand-rolled Walsh–Hadamard | `O(d log d)` randomized Hadamard rotation instead of dense `O(d²)` matmul |
| LLM integration | **HuggingFace Transformers** | `Cache` API lets us slot in a custom KV cache without touching model code |
| Vector-search baseline | **FAISS** (`faiss-cpu` is fine) | Provides Product Quantization (PQ) to benchmark recall@k against |
| Datasets | **GloVe / SIFT1M** (search), **LongBench / RULER / needle** (LLM) | The paper's own benchmarks |
| Testing | **pytest** + **hypothesis** | Property tests for *unbiasedness* and *distortion ≤ bound*, not just point checks |
| Packaging | **uv** (or pip) + `pyproject.toml`, `src/` layout | Reproducible env; uv is fastest on Windows |

### Production / kernels (Phase 5 only — needs a toolchain you don't yet have)

| Concern | Choice | Note |
|---------|--------|------|
| Local C++ inference | **llama.cpp (from source)** | The right target for a GGUF/ggml TurboQuant KV path. **ollama is not** — it's a wrapper; see `docs/05` |
| GPU kernels | **CUDA C++** (via llama.cpp/ggml) or **Triton** (via PyTorch) | Triton on native Windows is painful → prefer **WSL2** or CUDA C++ for kernel work |
| Build deps | **MSVC Build Tools + CUDA Toolkit 12.x** | Not currently installed (`docs/05`); only needed if/when you reach Phase 5 |

> **Why Python first and C++ much later.** The C++/ggml path is where 8+ community forks live,
> but it is also where you can spend weeks fighting build systems instead of understanding the
> algorithm. Get a *correct, validated* Python reference first; port to llama.cpp only once you
> know exactly what "correct" looks like and have numbers to match.

### Why not start directly in C or Rust?

A reasonable instinct ("the paper is about speed, so use a fast language"). The resolution is
**right tool per phase**, not one language for the whole project:

- **Validation phase → Python wins, for replication-specific reasons:** (1) *oracle parity* —
  the authors' reference (`amirzandieh/QJL`) and every benchmark harness (LongBench, RULER, HF
  models) are Python; doc 04's method is "diff your tensors against the authors' code," which
  you forfeit in Rust/C. (2) The numbers we must reproduce are one import each (QR rotation,
  Lloyd–Max, Beta dist, FAISS PQ, model loading); in C/Rust you'd rebuild all of it before
  testing a single claim. (3) Bugs in the core (rotation + Lloyd–Max) are found in 10 lines of
  NumPy, not 500 lines of Rust + a hand-rolled harness.
- **The speed myth:** the paper's 8× comes from *GPU kernels*, not the host language. PyTorch's
  matmuls/rotations already run as cuBLAS/CUDA C++; only naive bit-packing is Python-slow (and
  that moves to a kernel anyway). A fast host language does **not** buy the paper's speed during
  replication.
- **The clincher:** a from-scratch C/Rust version has no way to verify its own correctness. A
  validated Python reference becomes a **bit-for-bit test oracle** for the later port ("does
  Rust match Python on 10k random vectors?"). Python-first doesn't add work — it's what makes
  the C/Rust version *provably correct*.

**Production phase → C/Rust wins (your instinct, just sequenced second):** integrate into
**llama.cpp/ggml → C/C++** (ggml is C/C++; the stated LLM target lives here); standalone library
or vector-search with Python bindings → **Rust** (safety + portable-SIMD + PyO3, as `turbovec`
did). Freeze the Python spec, then port against it as the oracle.

## Proposed code architecture

```
turbo-quant/
  pyproject.toml
  src/turboquant/
    rotation.py        # Haar random orthogonal (QR) + randomized Hadamard (fast path)
    scalar_quant.py    # Lloyd–Max codebooks for the Beta/Gaussian coordinate marginal
    qjl.py             # QJL sketch (sign(S·r)) + asymmetric unbiased inner-product estimator
    quantizers.py      # TurboQuantMSE (Stage 1), TurboQuantProd (Stage 1+2), config dataclass
    packing.py         # bit-packing / unpacking -> REAL compression ratios
    kvcache.py         # HF-compatible Cache: asymmetric K/V bits, residual window
    search.py          # vector-search index (MIPS) using Stage 1 (+QJL) for ANN
    metrics.py         # distortion, recall@k, attention cosine sim, perplexity helpers
  tests/
    test_distortion.py # Phase 0: MSE vs paper bound (property + point)
    test_unbiased.py   # Phase 1: E[QJL estimate] == true inner product (hypothesis)
    test_packing.py    # packed bytes == advertised ratio
  benchmarks/
    bench_search.py    # recall@k vs FAISS PQ on GloVe
    bench_kv_fidelity.py  # attention cosine sim on a small HF model
    bench_generation.py   # needle-in-a-haystack end-to-end (the real test)
    bench_llamacpp_baseline.py  # drive existing E:\llama.cpp for a q4_0/q8_0 KV baseline
  docs/                # you are here
```

Design rules carried over from the community's lessons (`docs/02`):
- QJL is **off by default for attention**, **on for search**.
- KV cache uses **asymmetric bits** (keys > values) and a **residual window** (recent tokens in
  fp16).
- Compression ratios are always measured on **bit-packed** output.

## Phased roadmap (each phase has a "definition of done" tied to a number)

### Phase 0 — Synthetic distortion (no model, no GPU needed) ✅ start here
Implement `rotation.py` + `scalar_quant.py`. Quantize random unit vectors at b = 1,2,3,4 bits.
- **DoD:** measured normalized MSE is **below the paper's upper bound** and within the ≈2.7×
  optimality factor. Target table (d=128): 3-bit MSE ≈ 0.034 (bound 0.043). This single result
  validates the entire core idea and costs minutes.

### Phase 1 — QJL unbiased estimator (synthetic)
Implement `qjl.py`. Estimate `⟨q, x⟩` for random vectors.
- **DoD:** estimate is **unbiased** (mean error ≈ 0 over many seeds, a `hypothesis` property
  test) and variance shrinks as `m` grows like the theory predicts.

### Phase 2 — Vector search (QJL's home turf)
Build `search.py`; evaluate recall@k on GloVe (d=200) vs FAISS Product Quantization.
- **DoD:** at matched bit-rate, **recall@k ≥ PQ baseline** (paper claims superiority). This is
  where QJL is *supposed* to win, so it's a clean check that our QJL is correct.

### Phase 3 — KV-cache attention fidelity (small HF model)
Capture K/V tensors from a small model; compare attention scores fp16 vs compressed.
- **DoD:** high attention cosine similarity at 3–4 bits — **but treat this as necessary, not
  sufficient** (see Phase 4). Confirm the **MSE-only > MSE+QJL** finding on your own setup.

### Phase 4 — End-to-end generation (the real test)
Plug `kvcache.py` into HF `generate()`; run needle-in-a-haystack + a LongBench subset on a
small instruct model (something in the 0.5–3B class fits the 3060 easily).
- **DoD:** **exact needle retrieval** at the target bit-rate with a residual window; quantify
  the quality-neutral bit-rate (paper says ~3.5) on *your* model. This is the number that
  actually matters.

### Phase 5 — Performance / production (optional, heaviest lift)
Bit-packed storage benchmarks; then either Triton (WSL2) or a llama.cpp/ggml C++ path for fused
dequant+attention.
- **DoD:** measured memory reduction (target ≥5–6×) on packed data and a speed number; ideally
  a working GGUF inference path. **Requires installing MSVC + CUDA Toolkit + llama.cpp source**
  (`docs/05`). Don't start this until Phases 0–4 are green.

## A useful free baseline you already have

`E:\llama.cpp` ships **prebuilt CUDA binaries** (`llama-perplexity.exe`, `llama-bench.exe`,
`llama-quantize.exe`, ...). Before writing any kernel, use them to measure llama.cpp's existing
KV-cache quantization (`q8_0`, `q4_0`) for perplexity and speed on a model you have. That gives
you a **real, honest baseline** to beat — and it needs zero new code. See `docs/05` for exact
commands.

## Suggested first action

Set up the Python environment (you currently only have the Windows Store Python stub — see
`docs/05`), then implement Phase 0. Phase 0 is self-contained, runs in minutes on CPU, and
proves the core of the entire paper. I can scaffold `rotation.py`, `scalar_quant.py`, and
`tests/test_distortion.py` on request.
