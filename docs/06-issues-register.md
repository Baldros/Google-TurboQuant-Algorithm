# 06 — Issues register (honest gaps, open questions, risks)

This is the single canonical list of every place where **what the paper claims** and **what we
could reproduce on this machine** diverge, plus open questions and known risks. It exists because
the user explicitly asked me to surface the paper-claim-vs-reproducible-reality gap rather than
paper over it. Findings are also discussed in context inside `docs/03` (the roadmap); this file is
the index you check first.

**Organised by phase.** Each phase has its own section; phases with nothing outstanding say so
explicitly, so a clean phase is distinguishable from an unreviewed one.

**Conventions.** Each issue has a stable ID (`ISS-NN`) so it can be referenced from code, commits,
and memory. *Severity* = impact on the mission (faithfully replicating TurboQuant). *Status* =
`OPEN` (unresolved), `DOCUMENTED` (understood, accepted as-is, no further action planned for now),
`MITIGATED` (worked around), `RESOLVED` (closed with evidence). Convert to `RESOLVED` only with a
number, never a hope.

---

## Summary table (by phase)

| ID | Phase | Issue | Severity | Status |
|----|-------|-------|----------|--------|
| — | 0 | *(no open issues — distortion reproduced under bound)* | — | — |
| — | 1 | *(no open issues — unbiasedness + variance law reproduced)* | — | — |
| [ISS-01](#iss-01) | 2 | Paper's "TurboQuant-Prod beats PQ at 2–4 bit" search claim does **not** reproduce | **High** | OPEN |
| [ISS-02](#iss-02) | 2 | DoD met only as a *tie at 4 bit* (MSE variant); a true win never observed | Medium | DOCUMENTED |
| [ISS-03](#iss-03) | 2 | Our PQ baseline strength vs the paper's is unverified | Medium | OPEN |
| [ISS-05](#iss-05) | 2 | Bit-accounting assumes normalized vectors (norm treated as ≈ free) | Low | DOCUMENTED |
| [ISS-04](#iss-04) | 3 | The core KV-cache claim was untested — now closed by Phase 3 | High | **RESOLVED** |

> IDs are assigned in discovery order, not phase order, so they stay stable as the table is
> re-sorted. **ISS-04** was closed on 2026-06-26 by Phase 3 (kept with evidence, not deleted).
> **ISS-06** (GPU stack not installed) was resolved on 2026-06-26 and removed — `torch
> 2.11.0+cu128` + `transformers 5.12.1` are installed and CUDA-verified on the RTX 3060; IDs are
> not reused.

---

## Phase 0 — Synthetic distortion

**No open issues.** Normalized MSE reproduced below the paper's `2.7·2⁻²ᵇ` bound at b = 1–4
(3-bit = 0.034), and the fast randomized-Hadamard rotation matched the dense Haar oracle to 5
decimals. See `docs/03` Phase 0. If a discrepancy surfaces later, file it here as `ISS-NN` under
this heading.

---

## Phase 1 — QJL unbiased estimator

**No open issues.** The asymmetric 1-bit estimator is unbiased (bias within 4·SE at every `m`) and
its variance follows the closed form ∝ `1/m` to 1.5%. See `docs/03` Phase 1. Note that the *use* of
QJL as a residual sketch is where the Phase 2 problem appears — see [ISS-01](#iss-01) — but the
estimator itself is sound.

---

## Phase 2 — Vector search (GloVe-200 vs FAISS PQ)

### ISS-01 — Paper's headline search claim does not reproduce {#iss-01}

**Severity:** High · **Status:** OPEN · **First seen:** `fefc938`

**Claim under test.** The TurboQuant paper's stated search method is **TurboQuant-Prod** (MSE
scalar quantization at `b−1` bits **plus** a 1-bit QJL sign sketch of the residual, giving an
*unbiased* inner product). The paper reports Prod *beating* FAISS Product Quantization at 2–4
bits/dim on GloVe-200.

**What we observed.** On the full GloVe-200 set (1.18 M × 200, 500 queries, exact ground truth
agreeing 1.000 with the dataset's own neighbour list), every method run as an *exhaustive* ADC
scan so only quantizer quality is measured:

| bits/dim | TurboQuant-MSE R@10 | TurboQuant-Prod R@10 | FAISS PQ (nbits=8) R@10 |
|----------|---------------------|----------------------|-------------------------|
| 1 | 0.335 | *(QJL-only 0.197)* | **0.378** |
| 2 | 0.561 | 0.341 | **0.602** |
| 4 | **0.844** | 0.710 | **0.844** |

Prod is the **worst** method at every rate — the opposite of the paper's ranking.

**Why it happens (this part is understood and is *not* a bug).**
`scripts/run_phase2_prod_diagnostic.py` shows Prod's recall climbing monotonically to the exact
ceiling as QJL bits `m` grow (variance ∝ `1/m`), and its inner-product-estimate RMSE falling in
lock-step:

| method | bits/vec | R@10 | IP-estimate RMSE |
|--------|----------|------|------------------|
| Stage1 `b=2`     |  400 | 0.578 | 0.026 |
| Prod `b=1,m=200` |  400 | 0.327 | 0.053 |
| Prod `b=1,m=800` | 1000 | 0.598 | 0.027 |
| Prod `b=1,m=1600`| 1800 | 0.705 | 0.019 |

The 1-bit QJL sketch is **unbiased but bit-inefficient**: matching the inner-product accuracy of
*one extra scalar bit* (RMSE 0.026 at 400 bits) costs the QJL residual ~1000 bits. At the paper's
matched budget (1 QJL bit/coordinate) the residual variance swamps the tiny score gaps between
real GloVe neighbours and wrecks ranking. **This is the search-side echo of the KV-cache lesson
`MSE-only > MSE+QJL`** (`docs/02`): an unbiased-but-high-variance inner product is the wrong tool
when scores must be discriminated finely — ranking here, softmax there.

**What we could *not* recover.** *Why* the paper reports Prod > PQ. Candidate explanations, none
verified: (a) a re-ranking / exact-rescoring stage on top of ADC; (b) a weaker PQ baseline than
our `nbits=8` (see [ISS-03](#iss-03)); (c) different bit-accounting (e.g. counting QJL bits
differently, or not normalizing). The honest reproduction is the table above.

**What would resolve it.** Any one of: locate the paper's exact PQ configuration and rerun;
implement re-ranking and re-measure; or get clarification from the authors' reference code. Until
then this stays OPEN — reproduced faithfully, contradicts the paper at the cited budget.

**Decision on record.** The user chose to *document the gap and finalize* Phase 2 rather than chase
the Prod claim. This issue tracks the open question without blocking later phases.

### ISS-02 — DoD met only as a 4-bit tie {#iss-02}

**Severity:** Medium · **Status:** DOCUMENTED · **First seen:** `fefc938`

The Phase 2 definition of done was *"at matched bit-rate, recall@k ≥ PQ baseline."* The best
TurboQuant variant (MSE) **ties** PQ at 4 bit (0.844 = 0.844) and trails by ~0.04 at 1–2 bit. So
the DoD is met only at the top of the range, and only as equality, never a strict win.

This is the expected **obliviousness tax**: a data-*oblivious* per-coordinate scalar quantizer
versus PQ's data-*dependent* learned vector codebooks. The compensating advantage — which is real
and is the paper's Table-2 point — is that TurboQuant pays **zero** training/indexing cost. We
consider this understood and accepted: MSE is *competitive*, not *winning*, against a tuned
baseline. Documented rather than "failed."

### ISS-03 — PQ baseline strength vs the paper is unverified {#iss-03}

**Severity:** Medium · **Status:** OPEN · **First seen:** `fefc938`

Our baseline is FAISS `IndexPQ` with `nbits=8` (256 centroids per subquantizer) trained on a
100 k subsample, scored by pure exhaustive L2 ADC on normalized vectors. That is a *strong*,
standard PQ. If the paper compared against a thinner or differently-trained PQ, their reported gap
would shrink or invert. We did **not** sweep PQ configurations. Closely tied to [ISS-01](#iss-01);
sweeping PQ (`nbits`, training size, OPQ rotation) is the most direct experiment to test whether
the paper's advantage is baseline-dependent.

### ISS-05 — Bit-accounting assumes normalized vectors {#iss-05}

**Severity:** Low · **Status:** DOCUMENTED · **First seen:** `fefc938`

In Phase 2 the GloVe vectors are L2-normalized, so the per-vector **norm costs ≈ 0 bits** and we
do not include it in the bit budget (Stage1 = `b·d`, QJL = `m`, Prod = `b·d + m`, PQ = `M·8`). In
the general (un-normalized) setting — and in the KV-cache — the norm must be stored, typically a
few bits per vector. This makes our Phase 2 accounting slightly favourable to the methods that
lean on the norm (QJL/Prod rescale by `‖x‖`). The effect is small at `d=200` but is a real
assumption to carry forward; KV-cache phases must budget the norm explicitly.

### Phase 2 methodology gotchas (lessons, not open issues)

Already tripped over and corrected, recorded so they are not repeated:

- **Don't cite recall from a `--db-limit` subset.** The dataset's precomputed neighbour indices
  point into the *full* 1.18 M database; running on a 100 k subset makes ground-truth agreement
  collapse (0.084) and recall meaningless. Citable Phase 2 numbers must use the full DB (agreement
  then 1.000). The 100 k subset is fine only for *internal relative* comparisons like the Prod
  diagnostic.
- **Estimate dataset sizes from the actual file, not memory.** `glove-200-angular.hdf5` is 919 MB,
  not "a few hundred MB" as first stated — corrected on the spot. Check the byte count before
  quoting a download size.

---

## Phase 3 — KV-cache attention fidelity (DONE)

### ISS-04 — The core KV-cache claim is now tested {#iss-04}

**Severity:** High · **Status:** RESOLVED (2026-06-26) · **First seen:** —

*Was:* everything reproduced through Phase 2 was synthetic distortion, synthetic unbiasedness, or
vector search — the thing TurboQuant actually exists for (compressing a live LLM **KV-cache**
without degrading attention) had not been touched. This was the highest-value gap in the project.

*Closed by Phase 3.* On real GPT-2 K/V (144 heads × 104 tokens × 64 head-dim, extraction
self-validated against the model's own cache to 2.9e-6), the MSE-compressed KV cache is
near-lossless — attention-output cosine vs fp32 is **0.991** at 4 bits (0.969 at 3, 0.912 at 2),
softmax KL **0.023** at 4 bits. **And the `MSE-only > MSE+QJL` lesson reproduces directly on
attention:** at every bit-rate the MSE key score beats the unbiased TurboQuant-Prod key score
(same `b·d` budget) on both cosine and KL — the search-side Phase-2 finding now confirmed where the
paper actually claims it. See `docs/03` Phase 3 for the full table; covered by
`tests/test_attention.py` and `scripts/run_phase3.py`. End-to-end *generation* quality (the
"necessary, not sufficient" caveat) is Phase 4's job.

---

## Phase 4 — End-to-end generation (not yet started)

No issues filed yet. The real quality-neutral bit-rate (paper says ~3.5) and exact needle
retrieval will be tested here; expect new `ISS-NN` entries once it runs.

---

## Phase 5 — Performance / production (not yet started)

No issues filed yet. Requires a toolchain not yet installed (MSVC + CUDA Toolkit + llama.cpp from
source — see `docs/05`).

---

## How to use this file

- When a new gap is found, add an `ISS-NN` row to the summary table and a section under the
  relevant phase; reference the ID in the commit message and in `docs/03` where the phase is
  discussed.
- When a gap closes, flip its status to `RESOLVED` and append the number/evidence that closed it —
  do not delete the entry (the audit trail is the point).
