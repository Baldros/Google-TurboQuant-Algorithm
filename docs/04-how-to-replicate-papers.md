# 04 — How to replicate a scientific paper (a practical method)

You said you're not an expert at turning papers into code, and that papers are "structured one
way but replicating them requires adjustments." That instinct is exactly right. This is the
reusable method I'd apply to any systems/ML paper — illustrated with TurboQuant.

## The core mindset: papers optimize for *acceptance*, not *reproduction*

A paper is written to convince reviewers a contribution is novel and correct. It is **not**
written as an implementation manual. So the things you most need (exact constants, seeds,
preprocessing, the one trick that makes it actually work) are often compressed into a footnote,
an appendix, or omitted as "standard." Replication is the act of **recovering the implicit
spec**. Expect to make and *document* decisions the paper didn't state.

## Read in passes, not linearly

Don't read front-to-back. Read in widening loops:

1. **Abstract + figures + tables** — what does it claim, and what numbers prove it? Write those
   numbers down; they become your scoreboard (`docs/01 §6`).
2. **Algorithm boxes / method section** — the actual procedure. Translate each line into a
   function signature *before* writing any code.
3. **Experiments section** — what data, what models, what metric, what baseline. This tells you
   what "done" means.
4. **Proofs / theory** — read **last**, and only as far as you need to extract *constants and
   guarantees you can test* (e.g. "distortion within 2.7× of the bound"). You almost never need
   to re-derive the math to reproduce the result.

## Build a "claim ladder" and climb the cheap rungs first

List every claim, ordered by cost-to-verify. For TurboQuant:

| Rung | Claim | Cost | Needs |
|------|-------|------|-------|
| 1 | Distortion within 2.7× of bound | minutes | random vectors, CPU |
| 2 | QJL estimator is unbiased | minutes | random vectors, CPU |
| 3 | recall@k beats PQ | hour | GloVe + FAISS |
| 4 | attention fidelity at 3-bit | hours | small model + GPU |
| 5 | quality-neutral at 3.5 bits | day | LongBench + GPU |
| 6 | 8× speedup on H100 | weeks | custom kernel + datacenter GPU |

**Climb from the bottom.** If rung 1 fails, nothing above it matters and you've spent minutes,
not weeks. Most failed replications die because they jump straight to rung 5 (the headline
benchmark) and can't tell whether the bug is in the algorithm, the integration, or the harness.

## Build a slow "ground-truth oracle" before you optimize

Write the dumbest possible correct version first: dense NumPy, explicit loops, true Haar
rotation via QR — even if it's 1000× too slow. This oracle is your **reference for everything
after**. When the fast/GPU/packed version disagrees with it, the oracle is right and the fast
version has the bug. (For TurboQuant: dense rotation matrix before randomized Hadamard; Python
loop Lloyd–Max before vectorized; fp32 before bit-packing.)

## Reproduce exactly one number before scaling

Pick the single most central number (here: 3-bit distortion ≈ 0.034) and reproduce *it*, on
*your* machine, deterministically, before touching anything else. One reproduced number is
worth more than a whole pipeline that "runs" but matches nothing.

## Test invariants, not just end metrics

End metrics (perplexity, BLEU) hide bugs because many wrong implementations still produce a
plausible-looking number. Test **properties** the math guarantees:

- *Unbiasedness:* `E[estimate] == truth` over many seeds (a statistical test, not `==`).
- *Bound:* measured distortion `≤` the paper's stated bound, at every bit-width.
- *Invariance:* rotation preserves inner products (`⟨Πq, Πx⟩ == ⟨q, x⟩` to float tolerance).

These catch the bugs that a single accuracy number sails right past.

## Beware metric mismatch (the trap that just bit this community)

A high score on a *proxy* metric does not imply the real task works. For TurboQuant, **99.5%
attention-score cosine similarity still produced garbled text**, because softmax amplifies the
small residual error. Always close the loop with the *actual* downstream task (generation),
not a convenient surrogate. When two metrics disagree, trust the one closest to the user-visible
outcome.

## Keep an assumptions / decisions log

Every time the paper is silent and you choose, write it down (one line): what was ambiguous,
what you picked, and how you'd know if it was wrong. This log is the difference between "it
works and I don't know why" and a result you can defend. Seed everything; record seeds.

## Use the authors' code as an oracle — but verify it's really theirs

If official code exists (here: `amirzandieh/QJL`), diff your intermediate tensors against it on
the same input. It resolves ambiguities the prose can't. But confirm provenance (author's
account, linked from the paper) — the ecosystem is full of look-alike reimplementations of
varying fidelity (`docs/02`). Treat third-party repos as *hypotheses*, the authors' repo as
*evidence*.

## Know the difference between "the math" and "the system"

The paper proves things about an idealized object (unbiased estimator, distortion bound). The
*system* (an LLM doing softmax attention, a GPU with memory bandwidth limits) lives in a
different regime where those guarantees can be irrelevant or even counterproductive — exactly
why QJL helps search but hurts attention. A faithful replication of the math can still be the
wrong engineering choice. Replicate the math to *understand*; then measure the system to
*decide*.

## Know when to stop

You will not reproduce the "8× on H100" number on a desktop GPU, and that's fine. Decide up
front which rungs are **in scope** (for us: rungs 1–5 on a 12 GB GPU) and which are
**reference-only**. A replication that nails rungs 1–5 and honestly says "rung 6 needs hardware
I don't have" is a *success*, not a failure. Honesty about scope is a feature — it's also the
trait that separates the trustworthy repos from the rest (`docs/02`).

## Checklist (tape this to the wall)

- [ ] Wrote down the scoreboard numbers from abstract/tables.
- [ ] Turned each algorithm-box line into a function signature.
- [ ] Built a slow NumPy oracle first.
- [ ] Reproduced ONE central number deterministically.
- [ ] Tests assert properties (unbiased, ≤ bound, invariances), not just metrics.
- [ ] Closed the loop on the *real* task, not a proxy.
- [ ] Logged every assumption + seed.
- [ ] Diffed against official code where it exists.
- [ ] Declared which claims are in-scope vs reference-only.
