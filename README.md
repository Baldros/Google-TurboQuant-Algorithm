# turbo-quant — replicating Google's TurboQuant

A from-scratch, well-documented replication of **TurboQuant**, Google + NYU's data-oblivious
vector quantization algorithm for extreme compression of LLM KV caches and vector-search
indexes.

This repository starts as a **research + planning effort** (read the papers, survey the
field, choose a stack) and grows into a working implementation. The docs below capture the
*whole* process so the reasoning is reproducible, not just the final code.

> Status: **Planning / research phase.** No implementation code yet — that's deliberate.
> We map the territory first (see `docs/`), then build in verifiable phases.

## The mission in one paragraph

TurboQuant compresses high-dimensional vectors (the keys/values an LLM stores while it reads
a long context, or the embeddings in a vector database) down to ~2.5–4 bits per number with
**near-optimal distortion** and **without storing any per-block scaling constants** — the
"memory overhead" that sinks most quantizers. It does this by *randomly rotating* each vector
so its coordinates become statistically predictable, then applying a fixed, optimal
per-coordinate quantizer. Google reports ≥6× KV-cache memory reduction and up to 8× faster
attention on H100 GPUs, with no training or fine-tuning.

## The three papers

TurboQuant is the capstone of a trilogy by overlapping author teams (Amir Zandieh is on all
three):

| # | Paper | arXiv | Role |
|---|-------|-------|------|
| 1 | **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead** | [2406.03482](https://arxiv.org/abs/2406.03482) | The 1-bit sign-sketch building block + unbiased inner-product estimator |
| 2 | **PolarQuant: Quantizing KV Caches with Polar Transformation** | [2502.02617](https://arxiv.org/abs/2502.02617) | Polar-coordinate trick: quantize angles whose distribution is analytically known |
| 3 | **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** | [2504.19874](https://arxiv.org/abs/2504.19874) | The capstone: rotation + optimal scalar quantizer, with QJL as an optional 2nd stage |

## Documentation index

Read in this order:

1. [`docs/01-turboquant-explained.md`](docs/01-turboquant-explained.md) — what the algorithm
   *is*, mathematically, with all three papers broken down precisely.
2. [`docs/02-replication-landscape.md`](docs/02-replication-landscape.md) — who else has tried
   this, how many, which repos to trust, and the **single most important community finding**.
3. [`docs/03-stack-and-roadmap.md`](docs/03-stack-and-roadmap.md) — the recommended tech stack,
   the proposed code architecture, and a phased build plan from "synthetic test" to "real LLM".
4. [`docs/04-how-to-replicate-papers.md`](docs/04-how-to-replicate-papers.md) — general,
   reusable methodology for turning a scientific paper into working code.
5. [`docs/05-feasibility.md`](docs/05-feasibility.md) — can *this* machine run it? Measured
   hardware/toolchain inventory, the verdict, and the **llama.cpp-vs-ollama** decision (with
   how ollama stores GGUF). Read this before installing anything.

## Quick orientation for a non-specialist

- You do **not** need to understand the proofs to replicate this. You need to reproduce a
  handful of *numbers* (distortion at each bit-width, recall@k, attention fidelity) and check
  a couple of *properties* (the estimator is unbiased; distortion stays under the paper's
  bound). The docs are organized around exactly those checkpoints.
- The biggest trap, already discovered by the community: the paper's headline two-stage
  method (MSE + QJL) is **worse than a one-stage MSE quantizer for LLM attention**, because
  softmax amplifies QJL's variance. QJL still wins for vector search. We design around this
  from day one. See `docs/02`.
