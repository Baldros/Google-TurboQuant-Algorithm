# 05 — Feasibility: can *this* machine run it, and is llama.cpp or ollama the target?

You asked two concrete questions: (1) does this machine have the compute and resources to
actually implement and run TurboQuant on LLMs, and (2) is **llama.cpp** or **ollama** the right
platform to implement it in — including whether ollama even exposes models the same way
llama.cpp does. I inspected the machine directly; here are the findings and the verdict.

## Machine inventory (measured, 2026-06-25)

| Component | Detected | Assessment for this project |
|-----------|----------|-----------------------------|
| CPU | Intel **i5-10400F**, 6 cores / 12 threads, 2.9 GHz | Fine for Phases 0–2 (synthetic + search) on CPU |
| RAM | **16 GB** | The binding constraint. Comfortable for ≤8B models; rules out 13B+ and large batches |
| GPU | **NVIDIA RTX 3060, 12 GB VRAM**, Ampere, compute cap **8.6**, CUDA 12, ~10.5 GB free | The hero component. CUDA-capable; enough for small-model dev and llama.cpp CUDA inference |
| Disk | **E: 190 GB free**, C: 79 GB free | Plenty; keep the project + models on E: |
| OS | Windows 11 | Works; some kernel tooling (Triton) prefers WSL2/Linux |

> **Gotcha worth noting:** the Windows "Display adapter" panel reports the 3060 as **4 GB** —
> that's a misread of a legacy WMI field. `nvidia-smi` is authoritative: **12288 MiB (12 GB)**.
> Always trust `nvidia-smi` for VRAM. 12 GB is exactly the card `tonbistudio/turboquant-pytorch`
> (the strongest community repo) was developed and tested on — so the realistic scope is known
> to fit.

## Toolchain inventory (measured)

| Tool | State | Implication |
|------|-------|-------------|
| **ollama** | Installed, **v0.30.10**; models: `gemma4:e4b` (9.6 GB), `qwen3.5:0.8b` (1.0 GB), `embeddinggemma` (621 MB) | Useful for quick local inference + as GGUF source; **not** an implementation target (see below) |
| **llama.cpp** | `E:\llama.cpp` — **prebuilt CUDA release binaries** (`llama-cli/server/bench/perplexity/quantize.exe`, `ggml-cuda.dll`, `cublas64_12.dll`). **Binaries only — no source, not a git checkout** | Great as a *baseline runner* today; to *modify* it you must clone the source and build |
| **Python** | Only the **Windows Store stub** on PATH (`WindowsApps\python.exe`); `python --version` fails | **Must install a real Python** before any Phase-0 work |
| **PyTorch** | Not installed | Install the CUDA 12.x wheel (sm_86) |
| **Build tools** | `cmake` ✅, `git` ✅; **missing**: MSVC `cl`, `nvcc` (CUDA Toolkit), `gcc/g++`, `uv` | Phases 0–4 need none of these; Phase 5 (C++ kernel) needs MSVC + CUDA Toolkit |

## Verdict: is it feasible?

**Yes — for the realistic scope (rungs 1–5 of the claim ladder in `docs/04`).** Nothing about
Phases 0–4 strains this hardware:

- Phases 0–2 (synthetic distortion, QJL unbiasedness, vector search) run in **minutes on the
  CPU**. No GPU required.
- Phases 3–4 (KV-cache fidelity + end-to-end generation) fit comfortably: a 0.5–3B model in
  fp16 uses ~1–6 GB VRAM, leaving headroom on the 12 GB card for the KV cache and activations.
  You can even mirror your existing `qwen3.5:0.8b`-class model.
- **Out of scope on this machine:** the paper's "8× speedup on H100" (rung 6) — that needs
  datacenter hardware and a custom kernel. That's reference-only, and saying so is fine
  (`docs/04`, "know when to stop").

**Memory budget rule of thumb (12 GB VRAM):** keep `model_weights + kv_cache + activations`
under ~10 GB. For ≤3B fp16 models this is trivial; for a 7–8B model use 4-bit weights
(~4–5 GB) and you still fit. 16 GB system RAM means do **one** model at a time and avoid huge
eval batches.

### Prerequisites to install (in order)

1. **A real Python 3.11** — python.org installer or `winget install astral-sh.uv` then
   `uv python install 3.11`. (The current Store-stub Python cannot run the project.)
2. **PyTorch (CUDA 12.x)** — `pip install torch --index-url https://download.pytorch.org/whl/cu124`
   (cu124/cu128 both ship sm_86 builds for the 3060).
3. **Project deps** — `numpy scipy transformers faiss-cpu pytest hypothesis`.
4. *(Phase 5 only, defer):* **Visual Studio Build Tools (MSVC)** + **CUDA Toolkit 12.x** to
   compile a modified llama.cpp; optionally WSL2 for Triton.

## Your specific question: llama.cpp vs ollama

### How they relate (and the GGUF question)
**ollama is a wrapper around llama.cpp.** Under the hood ollama bundles a fork of
llama.cpp/`ggml` as its inference engine. Both use the **GGUF** model format. The difference is
*storage layout*, which is exactly what you were unsure about:

- **llama.cpp** loads a plain `model.gguf` file you point it at.
- **ollama** stores the *same GGUF data* as **content-addressed blobs** under
  `~/.ollama/models/blobs/sha256-…`, described by an OCI-style **manifest**. I confirmed this on
  your machine: the manifest layer `application/vnd.ollama.image.model` is the GGUF weight blob
  (e.g. the 8.95 GB blob = `gemma4`), with separate layers for params/license/template.

So ollama models *are* GGUF — just renamed to their hash and wrapped in a manifest. You can
extract/convert between the two (`ollama` can import a GGUF via a `Modelfile`; the blob *is* a
GGUF you could copy out). They are **not different formats**, just different filing systems.

### Which one can you implement TurboQuant in?

**llama.cpp — yes. ollama — no (not directly).**

- **TurboQuant for a KV cache is an engine-level change**: it lives in `ggml`'s KV-cache code
  and quantization types. llama.cpp is where those live, and where ~8 community forks have
  already added TurboQuant paths (`docs/02`). It is the correct C++ target.
- **ollama does not expose KV-cache internals for custom quantization.** It only lets you
  *select* KV-cache quant types that its bundled llama.cpp *already* implements (via
  `OLLAMA_KV_CACHE_TYPE=q8_0`/`q4_0` + flash attention). To get TurboQuant into ollama you would
  have to fork ollama's *vendored* llama.cpp, implement it there, and rebuild ollama — strictly
  more work than just using llama.cpp, with no benefit.

**Conclusion:** for the *eventual* C++/production path, target **llama.cpp from source**. Use
**ollama** only as a convenient runner and as a GGUF source. And do neither until the Python
reference (Phases 0–4) is validated — that's where the understanding gets built.

### One caveat about your current llama.cpp
`E:\llama.cpp` is a **binaries-only release drop**, not a source checkout. You can *run* it
today, but you cannot *modify* it. When you reach Phase 5 you'll
`git clone https://github.com/ggml-org/llama.cpp`, install MSVC + CUDA Toolkit, and build —
keeping the existing binaries around as your unmodified baseline.

## What you can do *today*, with zero new installs

Use the existing `E:\llama.cpp` binaries to establish an honest baseline to beat (these are the
numbers TurboQuant must improve on at matched compression):

```powershell
# 1) Speed + a quick functional check with KV cache quantized to 8-bit (q8_0)
E:\llama.cpp\llama-bench.exe -m <path-to-some.gguf> -ctk q8_0 -ctv q8_0 -fa 1

# 2) Quality baseline: perplexity with fp16 KV vs quantized KV
E:\llama.cpp\llama-perplexity.exe -m <path-to-some.gguf> -f <text.txt>            # fp16 KV
E:\llama.cpp\llama-perplexity.exe -m <path-to-some.gguf> -f <text.txt> -ctk q4_0 -ctv q4_0 -fa 1
```

(`-ctk`/`-ctv` = cache type for keys/values; `-fa 1` = flash attention, required for quantized
KV.) You'll need a `.gguf` on disk — either download a small one, or extract the GGUF blob that
backs your `qwen3.5:0.8b` ollama model. Comparing fp16-KV vs `q4_0`/`q8_0`-KV perplexity gives
you the exact "quality vs compression" trade-off curve that your TurboQuant implementation will
later try to beat in the 2.5–3.5-bit regime where block methods can't reach.

## Feasibility summary

- **Compute:** ✅ sufficient for Phases 0–5 dev (rungs 1–5). RTX 3060 12 GB is the proven
  sweet spot for this exact project.
- **Resources:** ✅ disk plentiful; ⚠️ 16 GB RAM means small models + modest batches.
- **Blockers:** install a **real Python + CUDA PyTorch** (5-minute fix) before Phase 0; install
  **MSVC + CUDA Toolkit** only when/if you reach the Phase-5 C++ path.
- **Platform choice:** implement in **llama.cpp (from source)** for C++, **never ollama**; both
  speak GGUF and ollama just stores it as hashed blobs.
- **Out of scope:** datacenter-GPU speed claims (rung 6) — reference-only, and that's okay.
