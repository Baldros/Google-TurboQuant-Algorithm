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

### Phase 0 — Synthetic distortion (no model, no GPU needed) ✅ **DONE**
Implement `rotation.py` + `scalar_quant.py`. Quantize random unit vectors at b = 1,2,3,4 bits.
- **DoD:** measured normalized MSE is **below the paper's upper bound** and within the ≈2.7×
  optimality factor. Target table (d=128): 3-bit MSE ≈ 0.034 (bound 0.043). This single result
  validates the entire core idea and costs minutes.
- **Result (reproduced, d=128, 50k unit vectors — `scripts/run_phase0.py`):**

  | bits | measured MSE | optimal Lloyd–Max | paper bound `2.7·2⁻²ᵇ` | meas/bound |
  |------|--------------|-------------------|------------------------|------------|
  | 1 | 0.3608 | 0.3634 | 0.6750 | 0.54 |
  | 2 | 0.1160 | 0.1175 | 0.1688 | 0.69 |
  | 3 | **0.0340** | 0.0345 | 0.0422 | 0.81 |
  | 4 | 0.0093 | 0.0095 | 0.0105 | 0.88 |

  All bit-rates land under the bound and within ≈2.7× of the `2⁻²ᵇ` floor. The fast
  randomized-Hadamard rotation matches the dense Haar oracle to 5 decimals (0.03395 vs
  0.03393), confirming the `O(d log d)` production path is distortion-equivalent. Covered by
  `tests/test_distortion.py` + `tests/test_rotation.py` (22 tests green).

### Phase 1 — QJL unbiased estimator (synthetic) ✅ **DONE**
Implement `qjl.py`. Estimate `⟨q, x⟩` for random vectors.
- **DoD:** estimate is **unbiased** (mean error ≈ 0 over many seeds, a `hypothesis` property
  test) and variance shrinks as `m` grows like the theory predicts.
- **Result (reproduced, d=128, 4k projections — `scripts/run_phase1.py`):**

  | m bits | mean est | bias | 4·SE | emp var | closed var `(π/2‖q‖²‖x‖²−⟨q,x⟩²)/m` | var·m |
  |--------|----------|------|------|---------|--------------------------------------|-------|
  | 16  | −6.145 | −0.040 | 2.43 | 1475 | 1449 | 23603 |
  | 32  | −5.940 | +0.165 | 1.71 | 727  | 725  | 23276 |
  | 64  | −5.983 | +0.122 | 1.19 | 354  | 362  | 22642 |
  | 128 | −5.986 | +0.119 | 0.86 | 184  | 181  | 23509 |
  | 256 | −6.168 | −0.063 | 0.61 | 92   | 91   | 23497 |

  True `⟨q,x⟩ = −6.105`. Bias stays inside 4·SE at every `m` (unbiased), empirical variance
  matches the closed form, and `var·m` is flat to **1.5%** — i.e. variance ∝ `1/m` exactly as
  the theory predicts. The estimator is the **asymmetric** one: only the stored side is the
  1-bit sign code `sign(Sx)`, the query stays full-precision, de-biased by `√(π/2)/m`. Covered
  by `tests/test_unbiased.py` (10 tests, incl. a `hypothesis` property test over random `q,x`).
  This is the standalone QJL stage; composing it onto the Stage-1 residual (`TurboQuantProd`)
  comes when search/KV need it. **Keep QJL off by default for attention** (see `docs/02`).

### Phase 2 — Vector search (GloVe-200 vs FAISS PQ) ✅ **DONE (with an honest gap)**
Build `search.py`; evaluate recall@k on GloVe-200 (the paper's own dataset) vs FAISS Product
Quantization, every method an *exhaustive* ADC scan so we measure only quantizer quality.
- **DoD (as written):** at matched bit-rate, **recall@k ≥ PQ baseline** (paper claims
  superiority). **Outcome:** met only as a *tie at 4 bits with the MSE variant*; the paper's
  headline claim (TurboQuant-**Prod** beats PQ at 2–4 bits) **does not reproduce** here. This is
  a real, documented finding, not a bug — see below.
- **Result (full DB, 1.18M × 200, 500 queries — `scripts/run_phase2.py`; ground truth agrees
  1.000 with the dataset's own neighbour list):**

  | bits/dim | TurboQuant-MSE R@10 | TurboQuant-Prod R@10 | FAISS PQ R@10 |
  |----------|---------------------|----------------------|---------------|
  | 1 | 0.335 | *(QJL-only 0.197)* | **0.378** |
  | 2 | 0.561 | 0.341 | **0.602** |
  | 4 | **0.844** | 0.710 | **0.844** |

  **Finding 1 — TurboQuant-MSE is competitive.** Pure rotation + per-coordinate Lloyd–Max
  *ties* learned PQ at 4 bits and trails by only ~0.04 at 1–2 bits — the expected small
  "obliviousness tax" of scalar-vs-learned-vector quantization, paid with **zero** indexing /
  training cost (the paper's Table-2 advantage, which is real). At low bit-rates PQ's
  data-dependent codebooks keep a small edge; the gap closes as bits grow.

  **Finding 2 — TurboQuant-Prod (the paper's stated search method: MSE at `b−1` bits + a 1-bit
  QJL residual sketch) underperforms badly, and we explain why.** It is the *worst* method at
  every rate. `scripts/run_phase2_prod_diagnostic.py` proves the implementation is correct —
  Prod's recall climbs monotonically to the ceiling as QJL bits `m` grow (variance ∝ `1/m`):

  | method | bits/vec | R@10 | inner-product est. RMSE |
  |--------|----------|------|--------------------------|
  | Stage1 `b=2` | 400 | 0.578 | 0.026 |
  | Prod `b=1, m=200` | 400 | 0.327 | 0.053 |
  | Prod `b=1, m=800` | 1000 | 0.598 | 0.027 |
  | Prod `b=1, m=1600` | 1800 | 0.705 | 0.019 |

  The 1-bit QJL sign sketch is **unbiased but bit-inefficient**: matching the inner-product
  accuracy of *one extra scalar bit* (RMSE 0.026 at 400 bits) costs the QJL residual ~1000
  bits. At the paper's matched budget (1 QJL bit/coordinate) the residual's variance swamps the
  tiny score gaps between real GloVe neighbours and wrecks ranking. **This is the search-side
  echo of the KV-cache lesson `MSE-only > MSE+QJL` (docs/02):** an unbiased-but-high-variance
  inner product is the wrong tool when scores must be discriminated finely — ranking here,
  softmax there. Why the paper reports Prod > PQ we could not recover with a standard `nbits=8`
  PQ baseline and pure ADC recall (candidate explanations: re-ranking, a weaker PQ baseline, or
  different bit-accounting); the honest reproduction is the table above. Covered by
  `tests/test_search.py` (9 tests: recall metric, exact oracle, Stage-1 recall ↑ with bits, QJL
  ≫ chance, Prod ≥ Stage-1 on well-separated synthetic data).

  > The open questions above (ISS-01 … ISS-03, ISS-05) are tracked as numbered entries in the
  > issues register, **`docs/06`**.

### Phase 3 — KV-cache attention fidelity (small HF model)
Capture K/V tensors from a small model; compare attention scores fp16 vs compressed.
- **DoD:** high attention cosine similarity at 3–4 bits — **but treat this as necessary, not
  sufficient** (see Phase 4). Confirm the **MSE-only > MSE+QJL** finding on your own setup.
- **Environment ready (2026-06-26):** `torch 2.11.0+cu128` (CUDA verified on the RTX 3060,
  capability sm_86, bf16 supported) and `transformers 5.12.1` installed in the venv via
  `pip install torch --index-url https://download.pytorch.org/whl/cu128` followed by
  `pip install transformers`. numpy 2.5.0 / faiss-cpu 1.14.3 unaffected.

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
