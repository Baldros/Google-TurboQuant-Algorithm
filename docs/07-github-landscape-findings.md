# 07 — GitHub landscape: what other implementations tell us about our open issues

**Date:** 2026-06-26
**Author:** investigation via the GitHub MCP, at the user's request
**Purpose:** Survey public implementations of TurboQuant / QJL / PolarQuant and use what
they reveal to make progress on the open items in [`06-issues-register.md`](06-issues-register.md)
— principally **ISS-01** (paper's "TurboQuant-Prod beats FAISS PQ at 2–4 bit" did not
reproduce) and **ISS-03** (our PQ baseline strength is unverified).

This is a **reading-only** study. No code was run; nothing here changes the repo. It is a
basis for deciding what to do next.

---

## 1. How to rank "significance" — and why stars are the wrong metric here

The natural first instinct is to sort by GitHub stars. For *this* question that is actively
misleading, and the data proves it. What we need from another repo is not popularity but
**(a)** whether it is the *authors'* reference code, and **(b)** whether it actually contains
the experiment our issue is about (vector-search recall vs PQ). A 12,000-star vector-search
*product* that never benchmarks against PQ tells us nothing about ISS-01; a 1-star faithful
*reproduction* can crack it open.

So the ranking used here is, in order:

1. **Authorship** — is it from Zandieh / the paper authors? Reference code is the gold standard.
2. **Contains the experiment** — does it run the search-recall-vs-PQ benchmark (ISS-01/03)?
3. **Algorithmic fidelity** — real TurboQuant/Prod + Lloyd–Max, not just a sign sketch.
4. **Then** stars / recency / maintenance, as tie-breakers only.

### The evidence that stars mislead

The star-sorted top of the `TurboQuant` search is dominated by downstream products and
inference engines created in the weeks *after* the paper, riding the hype:

| Repo | Stars | What it actually is | Runs search-vs-PQ? |
| --- | ---: | --- | --- |
| `RyanCodrai/turbovec` | ~12,200 | Rust vector-index product | No |
| `TheTom/turboquant_plus` | ~7,000 | KV-cache engine | No |
| `0xSero/turboquant` | ~1,600 | KV-cache + Triton/vLLM | No |
| `scrya-com/rotorquant` | ~1,000 | A *competing* method ("beats TurboQuant") | No |
| `tonbistudio/turboquant-pytorch` | ~1,000 | From-scratch KV-cache port | No |

Meanwhile the two most authoritative repos for *correctness* sit at **100 and 45 stars**,
precisely because they are the authors' own research code, not consumer tools.

> Note: the star counts above are read from the GitHub API; I inferred each repo's nature
> from its description/structure, not a deep code audit. The relevance argument holds
> regardless of the exact numbers.

---

## 2. The landscape in one paragraph

There are **hundreds** of repos name-dropping "TurboQuant" (the search reports ~887), but the
overwhelming majority are KV-cache inference engines, framework ports (MLX/llama.cpp/vLLM/WASM),
or hype/AI-generated scaffolding created in late March 2026. **There is no official TurboQuant
repository** — the authors never released reference code for arXiv 2504.19874. The only
*author* code that exists is for the two predecessor papers (QJL, PolarQuant), and both are
KV-cache / LLM-benchmark focused, not vector-search. Serious, honest *reproductions* are rare;
the best one found is `NeuraLiying/TurboQuant`.

---

## 3. The five most significant repos (relevance-ranked)

### 3.1 `amirzandieh/QJL` — official QJL (100★) — AUTHORITATIVE
<https://github.com/amirzandieh/QJL>
Amir Zandieh's own QJL implementation (his GitHub user is `amir619`; he is an author on all
three papers). Contents: `eval_long_bench.py`, `run_longbench.py`, `qjl_kernel/` (CUDA),
`plot_distortion.py`. **Scope: KV-cache / LongBench only — no vector-search benchmark.**
Value to us: the gold-standard reference for the QJL stage math and bit-accounting, and a
distortion plot we can sanity-check Phase 1 against. Does **not** directly address ISS-01.

### 3.2 `ericshwu/PolarQuant` — official PolarQuant (45★) — AUTHORITATIVE
<https://github.com/ericshwu/PolarQuant>
"Official Implementation For PolarQuant." Contents: `test4gsm8k.py`, `test4long.py`,
`benchmark/`, `models/`, `utils/`. **Scope: KV-cache + GSM8K/long-context only — no
search-vs-PQ.** Value: reference for the polar/angle quantizer; confirms the author line of
work is KV-centric.

### 3.3 `NeuraLiying/TurboQuant` — the serious reproduction (1★, 2026-06-13) — MOST RELEVANT
<https://github.com/NeuraLiying/TurboQuant>
A genuine reproduction effort mirroring our own: a `turboquant/` package, 91 unit tests, a
full `reproduce/` audit trail (plans, manifests, comparison tables), plus
`experiments/{longbench,ann_search,needle}/`. This single repo carries most of the payoff
below. **It is the closest thing to a peer of our project.**

### 3.4 `RecursiveIntell/turbo-quant` — Rust sidecar codec (28★) — MECHANISM FOR ISS-01
<https://github.com/RecursiveIntell/turbo-quant>
A Rust crate implementing TurboQuant/PolarQuant/QJL as **search "sidecars."** Its entire
design philosophy is the key insight: the compressed sketch returns *approximate candidates*
plus a receipt that **mandates an exact rerank** (`exact_rerank_required`, "approximate scores
are not ground truth"). It explicitly is *not* meant to be the final ranker.

### 3.5 The official QJL/PolarQuant predecessors aside — honourable mentions
`mindtro/semafold` (19★, retrieval-focused Python), `RecursiveIntell` above, and
`claudiusthebot/turboquant-cpp` (0★, "validates paper theorems empirically") are the only
others that even gesture at the search/distortion side rather than KV inference.

---

## 4. What this tells us about our open issues

### 4.1 ISS-01 / ISS-03 — the Prod-vs-PQ gap — **major update**

Three independent facts converge:

**(a) Our reproduction gap is externally corroborated.** `NeuraLiying`'s own reproduction
table — on the *paper's own* Llama-3.1-8B LongBench KV benchmark — lands **below** the
published TurboQuant numbers:

| Method | Bits | Paper avg | Their local avg | Shortfall |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant | 2.5 | 49.44 | 45.42 | **−4.02** |
| TurboQuant | 3.5 | 50.06 | 49.38 | −0.68 |

A second, independent team could not match the paper's TurboQuant headline either. This is
strong outside evidence that our ISS-01/ISS-02 reading — *the paper's TurboQuant claims are
optimistic; there is a real "obliviousness tax"* — is sound and **not a bug in our code.**

**(b) Nobody reproduces the head-to-head Prod-vs-PQ at all.** Not even the serious
reproduction. `NeuraLiying`'s search script (`experiments/ann_search/run_turboquant_ann.py`):
- measures **`recall_1@k`** — "is the *true* nearest neighbour inside my approximate top-*k*?",
  swept k = 1…64. That is a **shortlist / candidate-generation** metric, *not* our stricter
  top-10 *set-overlap* recall@10;
- uses **`TurboQuantMSE` only — no Prod / QJL residual** for search;
- has **no PQ / FAISS baseline whatsoever** — it compares only against exact search.

**(c) The real-world pattern is shortlist-then-rerank.** `RecursiveIntell/turbo-quant` is
built entirely around "approximate candidate generation → **exact rerank / exact fallback**."
The sketch is for *recall*, never for final ranking.

**Synthesis.** Our Phase 2 measured the **harshest possible** test for Prod: pure-ADC
**recall@10 set-overlap, with no rerank.** An unbiased-but-noisy estimator like the 1-bit QJL
residual is designed to do the opposite job well — to keep the true neighbour *somewhere in a
shortlist* so an exact rescoring step can finish the ranking. The most likely explanation for
the paper's "Prod beats PQ" is therefore a **protocol difference (shortlist-recall and/or
exact rerank), not a baseline-strength difference.** This *reframes* ISS-03: sweeping PQ
configs is still fair, but the evidence now points more at *how recall is measured* than at
*how strong our PQ is*.

### 4.2 ISS-02 (4-bit tie / obliviousness tax) — reinforced
The external reproduction shortfall in 4.1(a) is the same tax showing up on a different
benchmark. Stays correctly DOCUMENTED; now with outside support.

### 4.3 ISS-07 / ISS-08 (needle retrieval) — a peer exists
`NeuraLiying/TurboQuant` has an `experiments/needle/` directory. If we revisit the
exact-retrieval caveat, that is a ready-made point of comparison on a larger (Llama-3.1-8B,
non-tiny) model than our Qwen2.5-0.5B.

### 4.4 Our central thesis — independently confirmed
`NeuraLiying`'s incremental methods (Unified Regular-Gain Gate, Rate-Hadamard Value MSE) keep
**K on TurboQuant-MSE and modify only the V path**, and their search uses **MSE, not Prod**.
That independently echoes our three-way finding: **keys carry the softmax and need precision;
values compress harder; MSE-only is the workhorse and QJL stays off the key/attention path.**

---

## 5. Recommended next experiments (evidence-backed)

Ordered by value-for-effort. All are self-contained, need no new toolchain, and run on data
we already have (`data/glove-200-angular.hdf5`, faiss-cpu).

1. **Re-frame the search metric (directly targets ISS-01).** Re-run Phase 2 adding a
   **shortlist recall metric** the way real implementers measure it — recall@k for larger k
   (e.g. the true top-1/top-10 captured within an approximate top-k candidate list). Hypothesis:
   Prod's unbiasedness makes it *recover* on this metric, even though it loses on strict
   recall@10 set-overlap. This is the single most likely resolution of ISS-01.

2. **Add an exact-rerank stage (the `RecursiveIntell` pattern, also ISS-01).** Shortlist with
   the Prod sketch, then rescore the top candidates with exact (or higher-bit) inner products,
   and re-measure. If quality recovers, we have found the paper's missing piece and can move
   ISS-01 toward RESOLVED with a number.

3. **Sweep the PQ baseline (ISS-03).** Vary FAISS `nbits` (4/6/8), training-set size, and add
   OPQ. Still worth doing for fairness, but per §4.1 I now expect the protocol (1, 2) to matter
   more than baseline strength.

4. **(Optional) Needle on a bigger model (ISS-07).** Use `NeuraLiying`'s needle setup as a
   reference to test whether the exact-retrieval gap narrows on Llama-3.1-8B vs our small GQA
   model.

My recommendation: **start with experiment 1**, because it is the cheapest and most directly
tests the most probable cause of the only High-severity open issue. Experiments 1 + 2 together
either resolve ISS-01 or harden it into a confident, externally-corroborated "the paper's
search claim depends on a shortlist+rerank protocol it does not foreground."

---

## 6. Honest caveats

- This is a **metadata + README + key-file** survey, not a deep audit of every repo. I read
  the structure and the decisive files (NeuraLiying's README + ANN runner, RecursiveIntell's
  README); I did **not** clone or execute anything.
- The high-star "product" repos were judged from descriptions, not inspected line-by-line.
- `NeuraLiying`'s numbers are *their* reproduction, with *their* environment caveats — useful
  as corroboration, not as ground truth.
- The "shortlist + rerank" explanation for ISS-01 is a **strong hypothesis grounded in how
  other implementers build this**, not a proven fact. It still has to be tested (experiments
  1–2) before ISS-01 can be marked RESOLVED.
