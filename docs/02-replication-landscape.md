# 02 — The replication landscape: who has tried, and what they learned

One of your goals was to understand **how many people have attempted this** and **which
attempts to trust**. This document answers both, and — more importantly — distills the
hard-won lessons the community has already paid for, so we don't repeat their mistakes.

## How many people have tried?

A GitHub search for `turboquant` returns **~880 repositories**. Most of that number is noise
(forks, stubs, name-squats), but the count of *substantive, independent* implementations is
genuinely large for a paper this recent — on the order of **30–40 real projects**, spanning an
unusually wide set of languages and runtimes:

| Ecosystem | Example repos | What they target |
|-----------|---------------|------------------|
| **Official author code** | [`amirzandieh/QJL`](https://github.com/amirzandieh/QJL) (confirmed real, ships a CUDA kernel) | The QJL component — the gold-standard oracle |
| **From-scratch PyTorch** | `tonbistudio/turboquant-pytorch`, `OnlyTerp/turboquant`, `back2matching/turboquant`, `scos-lab/turboquant` | Reference correctness, HF integration |
| **vLLM integration** | `0xSero/turboquant`, `mitkox/vllm-turboquant`, `Alberto-Codes/turboquant-vllm`, `varjoranta/turboquant-vllm` | Production serving, Triton kernels |
| **llama.cpp / ggml** | `AmesianX/TurboQuant` (C++), `animehacker/llama-turboquant`, `atomicmilkshake/llama-cpp-turboquant`, `domvox/...-hip` (AMD), `unixsysdev/llama-turboquant`, `BoFan-tunning/llama.cpp-MTP-TurboQuant` | CPU/GPU local inference |
| **Apple MLX / Metal** | `arozanov/turboquant-mlx`, `sharpner/turboquant-mlx`, `manjunathshiva/turboquant-mlx` | Apple Silicon |
| **Other languages** | `botirk38/turboquant` (Zig), `teamchong/turboquant-wasm` (WASM/Zig), `AbdelStark/turboquant` (Rust) | Portability demos |
| **Vector search** | `RyanCodrai/turbovec` (Rust + Python ANN index), `bigmacfive/turbo-graph` | The *other* use case (no softmax) |

**Takeaway:** TurboQuant has been independently reimplemented dozens of times across Python,
C++, Rust, Zig, WASM, MLX, CUDA, HIP, and Triton — within months of release. That breadth is
itself a useful signal: the *core* algorithm is small and reproducible enough that one person
can implement it in a weekend. The hard part is not the rotation + quantizer; it's making it
actually help a real LLM.

## ⚠️ Do not trust star counts

In the raw search results, several repos show wildly inflated star counts (one vector-search
repo shows >12k stars, a "plus" fork shows ~7k) that are inconsistent with a niche, months-old
research topic. **Treat stars as close to meaningless here.** Judge trust by the criteria
below instead.

### Trust checklist (how we rank a replication)

A replication is trustworthy to the degree that it:

1. **Matches the paper's math** — random orthogonal rotation, per-coordinate Lloyd–Max on a
   Beta/Gaussian marginal, the `sqrt(π/2)/m` QJL de-bias factor. If the code can't point to
   these, it's cargo-culting.
2. **Is honest about limitations** — reports where it *fails*, not just where it wins.
3. **Reproduces a paper number** — distortion-vs-bound, recall@k, or a LongBench score, with
   the measurement code included.
4. **Tests the right thing** — end-to-end *generation*, not just attention-score cosine
   similarity (these disagree — see below).
5. **References the official papers/code** and engages with other implementations' findings.

By these criteria, the standouts are:

- **`amirzandieh/QJL`** — the authors' own QJL code (CUDA kernel, LongBench eval harness).
  This is the canonical oracle for the QJL stage. Use it to validate your inner-product
  estimator.
- **`tonbistudio/turboquant-pytorch`** — clean from-scratch PyTorch, validated against the
  paper's distortion bounds, and *unusually honest*: it publicly retracted an over-optimistic
  early claim ("18/18 perfect generation") after a community member found the benchmark was
  bugged (compression wasn't actually happening). That self-correction is exactly the behavior
  that makes a replication trustworthy.
- **`scos-lab/turboquant`** — multi-model benchmark that quantified the QJL-vs-MSE question.

## The single most important finding the community has already made

> **The paper's headline two-stage method (MSE + QJL) is *worse* than a one-stage MSE
> quantizer for LLM KV-cache attention.**

Why: QJL gives an *unbiased* inner-product estimate but with *higher variance*. Attention runs
scores through **softmax**, which is exponential and therefore **amplifies variance**. The
lower-variance (but biased) MSE estimate wins after softmax. Independent teams measured this
across Python, C, and Rust:

- `scos-lab` reported **+300% attention error with QJL vs +7.6% without** on GPT-2.
- `tonbistudio` V2 (with QJL): **0/27 generation tests passed**; V3 (MSE-only): **18/18 passed**.
- 6+ teams converged on the same conclusion.

QJL still **wins for vector search** (maximum-inner-product search has *no softmax*), which is
the paper's *other* advertised use case. So the lesson is not "QJL is wrong" — it's "QJL is
right for the regime it was proven in, and KV-cache attention is a different regime."

**Design consequence for us:** make the QJL stage *optional and off by default for attention*,
on by default for vector search. This is baked into our architecture (`docs/03`).

### Other lessons already paid for

- **Asymmetric K/V bit allocation is essential.** In real models, key vectors have far larger
  norms than value vectors (reported: Qwen key norms ~172–778 vs value norms ~2–4). Give keys
  more bits than values. `K=4 bits / V=2 bits` (3-bit average) beats uniform `3/3` decisively.
- **Attention-score similarity ≠ working generation.** 99.5%+ cosine similarity on attention
  scores can still produce garbled text. **Always test end-to-end generation** (e.g.
  needle-in-a-haystack), not just a similarity metric. This is the #1 way replications fool
  themselves.
- **Keep a "residual window."** Storing the most recent N tokens in full precision (fp16) while
  compressing the rest dramatically improves generation quality. This knob is *not in the
  theory* but matters a lot in practice.
- **Bit-packing is not optional.** Naive index storage can end up *larger* than fp16 (one
  early version stored tensors 38% bigger than uncompressed). You must pack indices into bits
  to realize the advertised compression ratio — and you must measure the *packed* size, not the
  theoretical one.
- **Where the method actually wins:** at *high* compression (≤3-bit / ≥5×), where block-based
  methods (e.g. llama.cpp's `Q4_0`) can't go. At ~4-bit, well-tuned block quantization is
  competitive. So benchmark in the 2.5–3.5-bit regime where TurboQuant is designed to shine.

## What this means for our build

We get to start at the community's current frontier instead of the paper's starting line:

1. Implement **Stage 1 (rotation + Lloyd–Max MSE)** first and treat it as the *primary*
   method for KV cache.
2. Implement **QJL** but gate it behind a flag, and validate it on **vector search**, where it
   is supposed to (and does) help.
3. Use **asymmetric K/V bits**, a **residual window**, and **real bit-packing** from the start.
4. **Always run a generation test**, not just attention similarity.

See `docs/03` for how these decisions turn into modules and a build order, and `docs/05` for
whether *your specific machine* can run all of this (short answer: yes, for the realistic
scope).
