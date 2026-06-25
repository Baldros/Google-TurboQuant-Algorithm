# 01 — TurboQuant explained (precisely)

This document is the technical reference. It states the algorithm in enough detail to
implement it, separating **what the math guarantees** from **what the experiments report**.

Sources: the blog post (research.google), arXiv [2504.19874](https://arxiv.org/abs/2504.19874)
(TurboQuant), [2406.03482](https://arxiv.org/abs/2406.03482) (QJL),
[2502.02617](https://arxiv.org/abs/2502.02617) (PolarQuant), and the authors' reference code
at [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL).

---

## 0. The problem being solved

You have many high-dimensional vectors `x ∈ ℝ^d` (d ≈ 64–256 per attention head). You want to
store them in far fewer than 16 bits/number and still be able to compute, accurately:

- **inner products** `⟨q, x⟩` — this is what attention does (query · key), and what
  maximum-inner-product vector search does (query · database vector); and/or
- **reconstructions** `x̂ ≈ x` — needed for the *values* that get averaged in attention.

The classic way (per-block affine quantization: store `scale` and `zero_point` per block of
numbers) wastes bits on those constants. That overhead is exactly what this line of work
removes. The key word in the papers is **data-oblivious** / **zero-overhead**: the codebook is
fixed in advance and depends only on `d`, never on the specific data, so there is *nothing
per-block to store* except, at most, a single norm.

---

## 1. The one idea that makes it all work: random rotation

Take any vector and multiply it by a **random orthogonal matrix** `Π` (a rotation). Rotation
preserves all inner products and norms (`⟨Πq, Πx⟩ = ⟨q, x⟩`), so it changes nothing about the
quantities you care about. But it makes the *coordinates* of the rotated vector behave like a
known, fixed probability distribution:

- For a unit vector in `ℝ^d`, after a Haar-random rotation, each coordinate is the marginal of
  a uniform point on the sphere. That marginal is a (scaled) **Beta distribution**;
  for large `d` it is approximately `N(0, 1/d)` (Gaussian). The paper's phrasing: rotation
  *"induces a concentrated Beta distribution on coordinates."*
- Distinct coordinates become **near-independent** in high dimensions.

Consequence: instead of needing a data-dependent quantizer, you can precompute **one optimal
scalar quantizer for that known marginal distribution** and apply it to every coordinate of
every (rotated) vector. No per-block statistics needed → zero overhead.

### Which rotation, in practice
- **Reference / correctness:** a true Haar-random orthogonal `Π`, generated as the `Q` factor
  of a QR decomposition of a Gaussian matrix (sign-corrected so `det = +1`). Cost `O(d²)` per
  vector — fine for tests, too slow for production.
- **Production:** a **randomized Hadamard transform** (a.k.a. random sign flips + Walsh–
  Hadamard transform), which approximates a random rotation at `O(d log d)` cost and needs no
  stored matrix. This is the standard trick (also used by QuaRot/SpinQuant). Recommended for
  the fast path.

---

## 2. Stage 1 — the MSE quantizer (reconstruction-optimal)

Goal: best possible `x̂ ≈ x` for a fixed bit budget `b` bits/coordinate.

```
encode(x):
    s   = ||x||                      # one scalar norm per vector (the only "side info")
    u   = x / s                      # unit vector
    y   = Π u                        # random rotation
    idx = [ nearest_centroid(y_i) for each coordinate i ]   # per-coordinate Lloyd–Max
    return (idx, s)

decode(idx, s):
    ŷ   = [ centroid[idx_i] for each i ]
    return s * (Πᵀ ŷ)                # undo rotation, restore norm
```

- `nearest_centroid` uses a **Lloyd–Max optimal scalar quantizer** for the known coordinate
  marginal (Beta≈Gaussian). The centroids and decision boundaries are **precomputed once** for
  each `(d, b)` and baked into the code — they are not learned from your data.
- Storage per vector: `b · d` bits for indices `+ ~16` bits for the single norm `s`. As `d`
  grows the norm cost is negligible, so the effective rate → `b` bits/coordinate.

**The guarantee (the paper's headline theorem).** This achieves a **distortion rate within a
small constant factor (≈ 2.7×) of the information-theoretic lower bound**, simultaneously for
all bit-widths and all dimensions. "Distortion" here is normalized MSE,
`E‖x − x̂‖² / E‖x‖²`. This is the single most important number to reproduce first (it needs no
LLM — see `docs/03`, Phase 0).

---

## 3. Stage 2 — QJL residual (inner-product-optimal, *optional*)

Stage 1 minimizes reconstruction error, but its inner-product estimate `⟨q, x̂⟩` is **biased**.
For applications that only need inner products (attention scores, MIPS search), TurboQuant adds
a second stage built on **QJL** (paper #1) applied to the Stage-1 residual `r = u − û`.

QJL = **Quantized Johnson–Lindenstrauss**: a random projection followed by keeping only the
**sign bit** of each projected coordinate.

```
S ~ N(0,1)^{m×d}            # random projection, m ≈ d, shared & fixed (not stored per vector)
store:  signs = sign(S r),   and  ||r||           # m sign bits + one norm
```

The **asymmetric, unbiased inner-product estimator** (QJL's core contribution): quantize only
the stored side (keys / database) with the sign; keep the *query* in full precision. Then

```
⟨q, x⟩  ≈  ⟨q, x̂_stage1⟩  +  ||r|| · sqrt(π/2) / m · ⟨ S q , sign(S r) ⟩
```

The `sqrt(π/2)/m` factor de-biases the sign sketch (it's the expected value correction for
`sign(g)·g` with Gaussian `g`). Result: an **unbiased** estimator of the true inner product
with controllable variance, and — crucially — **no per-block scale/zero-point**, just `m` sign
bits and one norm. "Asymmetric" = data side quantized, query side not; this halves the noise
versus quantizing both.

Net rate: Stage 1 with `b−1` bits + 1 sign bit ≈ `b` bits/coordinate, but now inner-product
unbiased instead of reconstruction-optimal.

### ⚠️ The catch that the papers under-emphasize
QJL is unbiased but **higher-variance** than the biased Stage-1 estimate. In **attention**, the
scores pass through **softmax**, which is exponential and therefore *amplifies variance*. The
community has repeatedly found that **MSE-only (Stage 1) beats MSE+QJL for LLM KV-cache
attention**, even though QJL is the "more correct" estimator. QJL *does* help for **vector
search**, where there is no softmax. Plan for both modes. (Full evidence in `docs/02`.)

---

## 4. PolarQuant (paper #2) — an alternative, complementary trick

PolarQuant attacks the same overhead problem from a different angle, specifically for KV
caches:

1. **Random preconditioning** (a rotation, same spirit as above).
2. Convert each (sub)vector from Cartesian to **polar coordinates**: a **radius** `ρ = ‖x‖`
   and a set of **angles** `θ`. (The blog's analogy: "Go 3 east, 4 north" → "go 5 at 37°".)
3. After preconditioning, the **angles follow a tightly concentrated distribution with an
   analytically computable form** → you can quantize the angles with a fixed codebook and
   **avoid storing scale/zero-point** entirely. The radius is cheap (one number).

Reported: **> 4.2× KV-cache compression** while staying competitive on long-context quality.
Think of PolarQuant and TurboQuant's Stage 1 as two routes to the same destination
("data-oblivious scalar quantization after a rotation"); TurboQuant's framing (Beta marginals +
Lloyd–Max + optional QJL) is the more general one and is the primary target of this repo.

---

## 5. Putting it together: the TurboQuant pipeline

```
                         ┌─────────────────────────────────────────────┐
   x  ──normalize──►  u  │  Π (random rotation)  ──►  per-coord Lloyd–Max │ ──► idx, ‖x‖
                         └─────────────────────────────────────────────┘   (Stage 1: MSE)
                                          │ residual r = u − û
                                          ▼
                         ┌─────────────────────────────────────────────┐
                         │  S (random proj)  ──►  sign(·)  ──► signs    │ ──► m sign bits, ‖r‖
                         └─────────────────────────────────────────────┘   (Stage 2: QJL, optional)
```

- **Keys** (attention) and **search vectors**: need inner products → Stage 1 (+ Stage 2 for
  search; Stage 1 *only* for attention, per §3 catch).
- **Values** (attention): get averaged → need reconstruction → Stage 1 only.
- **Asymmetric bit allocation** is essential in practice: in real models the key vectors have
  much larger norms than value vectors, so give keys more bits than values (e.g. K=4 bits,
  V=2 bits → 3 bits average, but far better than uniform 3/3). More in `docs/02`.

---

## 6. Headline numbers to reproduce (the scoreboard)

| Claim | Value | Where it comes from | Cheapest way to test |
|------|-------|--------------------|----------------------|
| Distortion vs lower bound | within ≈2.7× | TurboQuant theorem | Synthetic random unit vectors, no model |
| MSE @ 3-bit (d=128) | ≈0.034 (≤0.043 bound) | community synthetic test | Synthetic, no model |
| KV memory reduction | ≥ 6× | TurboQuant | Bit-packed storage accounting |
| Attention speedup (H100) | up to 8× (4-bit) | TurboQuant | GPU kernel benchmark (late phase) |
| Quality-neutral bit-rate | 3.5 bits/channel | TurboQuant | LongBench / needle test |
| Marginal-degradation rate | 2.5 bits/channel | TurboQuant | LongBench / needle test |
| QJL KV reduction | > 5× | QJL paper | LongBench (Llama/Mistral) |
| PolarQuant compression | > 4.2× | PolarQuant | Long-context eval |

Benchmarks named across the papers: **LongBench, Needle-In-A-Haystack, ZeroSCROLLS, RULER,
L-Eval**; models **Llama, Mistral, Gemma**; vector-search dataset **GloVe (d=200)** with
baselines **Product Quantization (PQ)** and **RabitQ**.

---

## 7. Notation / assumptions log (fill this in as you implement)

Keep a running list of every choice the papers leave implicit. Starter set:

- Rotation: Haar (QR) for tests vs randomized Hadamard for speed — **must match seeds** to be
  reproducible.
- Marginal model for Lloyd–Max: exact Beta vs Gaussian approximation — check which one
  reproduces the 3-bit MSE number above.
- QJL projection dim `m` (default `m = d`) and whether `S` is Gaussian vs `±1` (Rademacher).
- Norm precision: is `‖x‖` stored in fp16? (Yes in reference code.)
- Per-head vs per-tensor quantization for KV cache.
- "Residual window": how many recent tokens are kept in fp16 (a practical knob, not in the
  theory, but it strongly affects generation quality — see `docs/02`).
