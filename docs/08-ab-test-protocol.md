# 08 — A/B test protocol: quantizing a heavy model's KV cache

**Date:** 2026-06-26
**Status:** protocol (to be executed)
**Track:** KV-cache / generation quality (NOT the vector-search / ISS-01 track — that is separate)
**Also serves:** the ISS-07 "does a bigger model narrow the exact-retrieval gap?" follow-up

---

## 1. The idea, in one line

Freeze our algorithm exactly as we built it, measure it on a heavy model, save the numbers,
then branch, improve, and re-measure against the *same* frozen yardstick. A clean A/B.

```
baseline (as-built, tagged)  ──run eval──▶  results/baseline/…
        │
        └─ branch: improve/kv-*  ──same eval──▶  results/improve-*/…  ──diff──▶ verdict
```

---

## 2. Three honest constraints that shape this test

These are not optional caveats; they determine what the test *can* and *cannot* show.

1. **Provider is Hugging Face `transformers` — not ollama / llama.cpp.** Our `TurboQuantCache`
   is a HF `DynamicCache` subclass; it plugs into `model.generate()` in Python. ollama and
   llama.cpp are compiled GGUF/CUDA runtimes with their own KV cache — running our algorithm
   there means re-writing the quantizer as a C++/CUDA kernel, which **is Phase 5** and is
   blocked on the MSVC + CUDA toolchain we have not installed. See [`05-feasibility.md`](05-feasibility.md).
   They remain a *future* row, not part of this A/B.

2. **Our cache is "fake-quant": it measures quality, not memory.** It quantizes then
   dequantizes back to fp before attention, so it faithfully measures *quality loss* per
   bit-rate but does **not** shrink VRAM. A heavy model here buys **realism**, not a memory
   win. The real memory saving is Phase 5 (bit-packed storage + fused kernel). Do not expect
   this test to make anything "fit."

3. **The 4-bit weights are a confound — which we neutralize by holding them constant.** We
   chose Qwen2.5-7B, which only fits on 12 GB with 4-bit *weights*. That adds a second
   quantization on top of our KV quantization. We keep the A/B valid by using the **identical
   4-bit weights in every condition** (baseline and all treatments). The weight quantization
   is then a fixed constant; the **only** variable is KV precision. Consequence: absolute
   quality numbers are degraded by the 4-bit weights, but the **delta** between fp16-KV and
   quantized-KV — the thing we actually study — stays isolated.

---

## 3. Model & provider (the definitions you asked for)

| Item | Choice | Notes |
| --- | --- | --- |
| **Model** | `Qwen/Qwen2.5-7B-Instruct` | 7.6 B params, 28 layers, GQA (28 query / 4 KV heads, head_dim 128), Apache-2.0, not gated. |
| **Weight precision** | 4-bit NF4 (bitsandbytes) | ~5 GB VRAM for weights; held **constant** across all conditions. Alt: a pre-quantized AWQ/GPTQ-Int4 checkpoint (smaller download, needs autoawq/auto-gptq). |
| **Provider / runtime** | Hugging Face `transformers` | The only runtime our cache plugs into (see §2.1). |
| **KV reference (control)** | fp16 KV via `DynamicCache` | "the original model" in your framing: 4-bit weights + full-precision KV. |
| **KV treatment** | `TurboQuantCache` at swept bit-rates | 4-bit weights + our quantized KV. |
| **Device** | `cuda:0` (RTX 3060, 12 GB) | weights ~5 GB + fp-KV (fake-quant) + activations; leaves room for a multi-thousand-token context. |
| **New dependency** | `bitsandbytes` | verify Windows + CUDA 12.x build loads before committing to this path. |

**Disk / download:** the fp16 checkpoint (~15 GB) downloads to the HF cache, quantized at load
time; or ~5–6 GB if using a pre-quantized AWQ/GPTQ checkpoint. Confirm free space on `E:` /
`HF_HOME`.

---

## 4. Experimental design

**One variable only: KV precision.** Everything else is pinned.

| Pinned | Value |
| --- | --- |
| Model + revision | `Qwen/Qwen2.5-7B-Instruct` (record the exact commit hash) |
| Weights | 4-bit NF4, identical bytes in every run |
| Decoding | greedy, `do_sample=False` |
| Seed | `0` (quantizer rotation seed; deterministic) |
| Prompts / inputs | identical fixed set across all conditions |
| Residual window | swept as part of the design (below), else fixed at 32 |

**Conditions (the KV bit sweep):**

- **Control:** fp16 KV (`DynamicCache`) — the reference every treatment is scored against.
- **Symmetric:** K = V ∈ {8, 5, 4, 3, 2} bits.
- **Asymmetric (key-heavy):** K8V4, K8V2, K5V2, K4V2 — tests our central thesis that keys need
  precision and values compress harder.
- **Residual-window ablation:** window ∈ {0, 32} at a fixed middling rate (3 bits).

---

## 5. Metrics & definition of done

For every condition we record numbers **and** raw outputs.

1. **Streaming perplexity** (smooth quality metric). Teacher-force a held-out continuation
   token-by-token with the cache live, so each prediction sees quantized history + fp residual
   window. Report NLL, ppl, and **ppl increase % vs the fp16-KV control**. Primary DoD:
   the **quality-neutral bit-rate** (lowest bits within a chosen ppl threshold, e.g. ≤1 %).
2. **Needle-in-a-haystack** (exact verbatim retrieval, the ISS-07/08 metric). Hide a passphrase
   mid-context, greedy-decode, exact-string match. Record HIT/MISS per config. **This is the
   bigger-model re-test of ISS-07** — does a 7B retrieve at fewer bits than the 0.5B did?
   (Honest: exact-match is brittle/non-monotonic — reported, never a hard pass/fail assertion.)
3. **(Optional) LongBench-lite task score.** A small subset (e.g. one MultiQA + one synthetic
   task, NeuraLiying-style) for a "real task" number rather than only perplexity. Decide
   whether to include based on runtime budget.

**Per-condition record:** bits (K, V), window, NLL, ppl, ppl Δ%, needle hit/miss, optional task
score, plus VRAM used, tokens/s, git SHA, model revision, dtypes. (VRAM is logged with the
explicit note that it is **not** reduced here — honesty about fake-quant.)

---

## 6. Artifacts & data capture

```
results/
  <branch-or-tag>/<YYYY-MM-DD>/
    summary.json        # all per-condition metrics (committed — small, diffable)
    raw_generations.jsonl   # every prompt + decoded output (so we can re-score later)
    env.json            # git SHA, package versions, GPU name, model revision, seed
```

`summary.json` and `env.json` are committed (small, the comparison record). `raw_generations.jsonl`
is committed if small, otherwise gitignored like the datasets. The point: a baseline run and an
improvement run are **diffable file-to-file**, and any number can be recomputed from raw outputs
without re-running the model.

---

## 7. Procedure (the workflow, made reproducible)

0. **Freeze the baseline.** Tag the current `dev` algorithm, e.g. `git tag baseline-kv-v0`,
   and record the SHA in `env.json`. This is the immovable reference point.
1. **Run the eval on the baseline** → `results/baseline-kv-v0/<date>/`.
2. **Branch for improvements:** `git checkout -b improve/kv-<idea>` from the tagged point.
3. **Implement the improvement.** The eval harness stays **byte-identical** — only the
   algorithm (the quantizer / cache) changes. If the harness must change, it changes on *both*
   sides and the baseline is re-run.
4. **Run the same eval on the branch** → `results/improve-<idea>/<date>/`.
5. **Compare** via a small diff script → verdict table (baseline vs improvement, per condition).
6. Same model revision, seed, prompts throughout, so any delta is the algorithm, not noise.

---

## 8. What counts as an improvement (success criteria)

Pick the bar *before* running so we don't rationalise after. Candidate primary criteria:

- **Lower perplexity at a fixed bit budget** (e.g. beats baseline ppl at 3-bit symmetric), or
- **Quality-neutral at a lower average bit-rate** (e.g. matches fp16-KV within 1 % ppl at
  2.5 effective bits instead of 3), or
- **Needle HIT at fewer bits** than the as-built version.

An improvement must win on the chosen primary metric **without** regressing the others beyond a
stated tolerance.

---

## 9. Candidate improvements to try on the branch

Drawn from the GitHub survey ([`07-github-landscape-findings.md`](07-github-landscape-findings.md)) —
these are *hypotheses to test*, not commitments:

- **Value-path-only refinement (NeuraLiying pattern).** Keep K on TurboQuant-MSE; apply a
  rate-aware value preconditioner (e.g. per-vector / late-layer Hadamard) on V only. Their
  reproduction improved MultiQA averages with exactly this; it matches our thesis that V is
  where the slack is.
- **Residual-window tuning.** We already see a large win from a small fp window; sweep it as a
  cheap quality lever.
- **Asymmetric K/V budgets.** Formalise key-heavy allocation (e.g. K-rich, V-poor) as the
  default, since keys carry the softmax.

---

## 10. Risks & honest caveats

- **Fake-quant ≠ memory win.** This test proves quality, not the headline compression. State
  it in the report.
- **4-bit-weight confound.** Absolute numbers are weight-degraded; only the KV-precision delta
  is clean (§2.3). Don't compare these ppl values to a full-fp16-weights run elsewhere.
- **Single model, single GPU.** One model is not a general claim; it's a representative,
  honest data point.
- **Brittle needle metric (ISS-08).** Exact-string retrieval is non-monotonic; reported as a
  signal, not a unit-test assertion.
- **Toolchain/dependency risk.** `bitsandbytes` on Windows + CUDA 12.x must be verified to load;
  if it fails, fall back to a pre-quantized AWQ/GPTQ checkpoint (held constant the same way).

---

## 11. Implementation notes (for when we execute)

- The eval harness already mostly exists: `scripts/run_phase4.py` has streaming perplexity +
  needle and accepts `--ppl-model` / `--needle-model`. Two small changes: (a) add a
  **4-bit load path** (`BitsAndBytesConfig` / quantization_config) so the heavy model loads in
  NF4; (b) point **both** perplexity and needle at the same 7B for one coherent report, and
  emit the `results/.../summary.json` + `raw_generations.jsonl` + `env.json` artifacts in §6.
- Keep the quantizer seed at 0 and decoding greedy so runs are deterministic and diffable.

---

## 12. Open parameters to confirm before running

Sensible defaults proposed; adjust after study:

- Context length for the perplexity / needle prompts (longer = harder, more realistic KV test).
- Number of needle fillers / needle position.
- Whether to include the optional LongBench-lite task (§5.3) given runtime budget.
- Perplexity threshold for "quality-neutral" (default 1 %).
