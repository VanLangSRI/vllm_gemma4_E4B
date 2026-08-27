# Performance report — Gemma 4 E4B on one RTX 3090 Ti

Full re-run of every performance test on the current stack, with vLLM measured
across a **2×2 factorial of `SPEC` × `PREFIX_CACHE`** and Ollama as an external
reference point.

**Date:** 2026-08-26 · **Raw log:** `bench/results-2026-08-26.txt`

| | |
|---|---|
| GPU | RTX 3090 Ti, sm86, 24 GB, **450 W limit** (default 450 W, persistence on) |
| Model | `google/gemma-4-E4B-it-qat-w4a16-ct` — official QAT int4, compressed-tensors |
| Stack | vLLM 0.27.1, torch 2.13.0+cu130, FlashInfer, Marlin kernels |
| Baseline flags | `MAX_SEQS=64`, `GPU_UTIL=0.90`, fp8 KV, `FULL_AND_PIECEWISE` CUDA graphs |
| Reference | Ollama `gemma4:e4b` (GGUF Q4_K_M), `q8_0` KV, `NUM_PARALLEL=16` |

Every number excludes warmup. Each config was booted fresh, verified against the
engine's own log (prefix-caching state, KV pool size, cudagraph mode, speculative
token count) and then benchmarked.

---

## The eight configurations

| | cudagraph | `SPEC` | `PREFIX_CACHE` | KV pool | `GPU_UTIL` | notes |
|---|---|---|---|---|---|---|
| **A** | on | off | 1 | 1,059,707 tok | 0.90 | **shipped default** |
| **B** | on | off | 0 | 1,063,902 tok | 0.90 | |
| **C** | on | ngram | 1 | 904,588 tok | 0.85 | |
| **D** | on | ngram | 0 | 904,588 tok | 0.85 | |
| **E** | off | off | 1 | 1,065,580 tok | 0.90 | |
| **F** | off | off | 0 | 1,059,967 tok | 0.90 | |
| **G** | off | ngram | 1 | 909,385 tok | 0.85 | ⚠ crashes |
| **H** | off | ngram | 0 | 914,987 tok | 0.85 | ⚠ crashes |

`SPEC=ngram` is not a free switch: speculation allocates on top of a KV pool
already sized to fill `GPU_UTIL`, so the launcher drops `GPU_UTIL` to 0.85 to
avoid an OOM at startup. **The pool shrinks by 15% (1.06 M → 0.90 M tokens)**,
which is part of the cost of the feature and is why C/D are not a like-for-like
memory comparison against A/B.

---

## Headline: general-purpose serving

Real-prompt cohort (8 prompts from `bench/prompts_real.jsonl`, 1024 output
tokens). This is the workload most deployments actually look like.

| config | C1 tok/s | C4 tok/s | C8 tok/s | TPOT @C8 |
|---|---|---|---|---|
| **A — default** | 87.9 | **340.8** | **655.2** | 11.2 ms |
| B — no prefix cache | **93.7** | 340.1 | 653.1 | 11.2 ms |
| C — ngram | 65.6 | 188.2 | 340.0 | 19.1 ms |
| D — ngram, no prefix cache | 66.1 | 191.4 | 345.7 | 18.9 ms |

**Speculation costs ~48% of general throughput.** C and D land at roughly half
of A and B at every concurrency, and TPOT rises from 11 ms to 19 ms. On ordinary
chat traffic there is nothing in the prompt to draft from, so every proposed
token is rejected and the verification passes are wasted work.

**Prefix caching is invisible here** (A vs B within 1%), and that is expected:
the eight cohort prompts share no common prefix, so there is nothing to reuse.
The A-vs-B C1 gap (87.9 vs 93.7) is run-to-run variance on a single-stream
measurement, not a real effect — the C4/C8 rows, which average over more
requests, agree to within 0.4%.

### Peak aggregate throughput

Synthetic saturation, 128 in / 512 out at C64:

| config | tok/s | median TPOT |
|---|---|---|
| **A — default** | **2621.0** | 22.8 ms |
| B | 2625.7 | 22.8 ms |
| C — ngram | 773.9 | 52.0 ms |
| D — ngram, no prefix cache | 580.7 | 66.2 ms |

At C64 speculation is actively harmful — **3.4x slower than the default**. With
64 sequences in flight the GPU is already compute-saturated, so the extra draft
and verify passes compete for the same tensor cores that the real work needs.

---

## Where `SPEC=ngram` earns its keep

The cohort above is the wrong workload for speculation. `bench/spec_bench.py`
isolates the effect by running two prompts of the same length side by side:

- **copy** — "reproduce this log file exactly" → the answer is already in the prompt
- **chat** — an open-ended technical explanation → nothing to draft from

| workload | A (no spec) | C (ngram) | change |
|---|---|---|---|
| **copy, C1** | 95.1 | **326.0** | **+243%** |
| **copy, C4** | 366.5 | **919.2** | **+151%** |
| chat, C1 | 95.7 | 64.1 | −33% |
| chat, C4 | 378.4 | 232.7 | −39% |

This is the sharpest result in the report. On copy-style work `SPEC=ngram` is
worth **3.4x**, peaking at **919 tok/s** — the highest single-configuration
number measured anywhere in this project. On ordinary chat it costs a third of
throughput.

The knob is therefore **workload-selective, not a global optimisation**:

- Turn it **on** for quoting, editing, reformatting, translation-with-source,
  extractive RAG — anything that reproduces its input.
- Leave it **off** for open-ended generation, and for any mixed or unknown
  traffic mix, which is why the default is off.

### Why `ngram_gpu` is *slower* than no speculation at all

A fair objection: the config is `method="ngram_gpu"` — the GPU implementation —
so why is it slower than the default on ordinary traffic?

**Because "gpu" names where the n-gram *search* runs, not a promise of speed.**
`NgramProposerGPU` moves the suffix-matching scan onto the GPU (a compiled
Triton kernel) instead of doing it on the CPU. That makes *proposing* drafts
cheap. It does nothing to make the drafts *correct* — and a wrong draft still
has to be verified by a full forward pass through the target model, which is
the expensive part.

The engine's own counters, taken straight from the config C and D runs:

| config | drafted tokens | accepted | acceptance | wasted |
|---|---|---|---|---|
| C — ngram + prefix cache | 162,928 | 5,124 | **3.1%** | **96.9%** |
| D — ngram, no prefix cache | 162,136 | 4,103 | **2.5%** | **97.5%** |

With `num_speculative_tokens=8`, the engine drafts eight tokens per step and
verifies them in one batched pass. On this cohort **roughly 97% of that drafted
work is discarded**. The verification passes are pure overhead, and the measured
~48% throughput loss follows directly.

Acceptance is low here for a structural reason, not a tuning failure: n-gram
lookup drafts by finding the longest suffix of what has been generated *inside
the request's own prompt* and proposing whatever followed it. When the model is
composing new prose, that text simply is not in the prompt, so the lookup
returns noise. Nothing about the GPU kernel can fix that.

The same mechanism explains the copy result. When the answer *is* in the prompt,
acceptance jumps and the arithmetic inverts:

| acceptance rate | extra tokens gained per verify step | outcome |
|---|---|---|
| 3% (measured, chat) | ~0.03 | verification cost dominates → **net loss** |
| 30% | ~0.43 | still a loss |
| 80% (copy-style work) | ~3.3 | **net win**, ≈3.4x observed |

So the rule is about the *workload*, not the backend:

- **`ngram` vs `ngram_gpu` is not a speed choice.** `ngram_gpu` is simply the
  variant that vLLM permits alongside `--async-scheduling`; the CPU `ngram` is
  rejected outright with *"async scheduling is only supported with
  EAGLE/MTP/Draft Model/NGram GPU/DSpark"*. Both draft the same way and would
  show the same ~3% acceptance on this cohort.
- **Speculation is a bet.** It pays when drafts are usually right and loses when
  they are usually wrong. Acceptance rate is the number to watch, and vLLM logs
  it — grep the server log for `Per-position acceptance rate` before deciding
  whether the feature belongs in your deployment.

If you want to try improving the bet rather than abandoning it, lowering
`SPEC_TOKENS` (default 8) reduces the wasted verification width, at the cost of
a smaller win on the workloads where drafting does succeed. It was not tuned for
this report.

### Speculation and prefix caching interact

Comparing C against D shows the two knobs are not independent:

| workload | C (ngram + prefix cache) | D (ngram, no cache) | cost of losing the cache |
|---|---|---|---|
| copy, C1 | 326.0 | 246.1 | **−25%** |
| copy, C4 | 919.2 | 527.7 | **−43%** |
| chat, C1 | 64.1 | 60.4 | −6% |
| chat, C4 | 232.7 | 214.8 | −8% |

On the copy workload, disabling prefix caching costs **43% at C4** — even though
the same knob was worth nothing at all in the general cohort (A vs B). The
reason: `spec_bench` sends the *same* long document to every concurrent request,
so the prefix cache has something substantial to reuse, and the two features
compound. Speculation makes each request finish faster, which raises the rate at
which requests hit a warm cache.

**Practical consequence:** if you enable `SPEC=ngram` for a copy-heavy workload,
keep `PREFIX_CACHE=1`. The pairing is worth considerably more than either knob
alone.

---

## CUDA graphs on vs off

The same battery re-run with `CUDAGRAPH=NONE`, completing a 2×2×2 factorial.
This is how the repo shipped before the misdiagnosed-OOM fix, so it also
re-establishes the cost of that bug honestly — on the current checkpoint, at
450 W, with the corrected harness.

| config | cudagraph | `SPEC` | `PREFIX_CACHE` | C1 | C4 | C8 | sat C64 |
|---|---|---|---|---|---|---|---|
| **A** | on | off | 1 | **87.9** | **340.8** | **655.2** | **2621.0** |
| B | on | off | 0 | 93.7 | 340.1 | 653.1 | 2625.7 |
| C | on | ngram | 1 | 65.6 | 188.2 | 340.0 | 773.9 |
| D | on | ngram | 0 | 66.1 | 191.4 | 345.7 | 580.7 |
| E | **off** | off | 1 | 20.0 | 75.1 | 145.9 | 1122.3 |
| F | **off** | off | 0 | 20.6 | 75.7 | 146.2 | 1151.7 |
| G | **off** | ngram | 1 | 22.1 | 70.3 | 128.9 | 562.5 ⚠ |
| H | **off** | ngram | 0 | 23.4 | 88.4 | 157.3 | 612.1 ⚠ |

⚠ = the engine **crashed** during this run; see below.

### CUDA graphs are worth ~4.5x

Comparing like for like (A vs E, both `SPEC` off with prefix caching on):

| workload | graphs on | graphs off | speedup |
|---|---|---|---|
| cohort C1 | 87.9 | 20.0 | **4.40x** |
| cohort C4 | 340.8 | 75.1 | **4.54x** |
| cohort C8 | 655.2 | 145.9 | **4.49x** |
| saturation C64 | 2621.0 | 1122.3 | **2.34x** |

TPOT tells the same story: **11 ms with graphs, 113–118 ms without** — a 10x
latency penalty per token. Decode is a long chain of small kernels, and without
graph capture the CPU pays a launch cost for every one of them; the GPU spends
most of its time waiting.

The gap narrows at C64 (2.34x) because a 64-wide batch gives each kernel enough
work to partly hide the launch overhead. It never disappears.

**Disabling CUDA graphs does not buy back meaningful memory.** The KV pool moves
from 1,059,707 to 1,065,580 tokens — **+0.55%**, because the 0.56 GB of capture
allocation is a small fraction of an 11 GB pool. Boot time actually got *worse*
in these runs (340 s vs 150 s), since skipping capture does not skip
`torch.compile`. There is no scenario in this data where turning graphs off is
the right trade.

### `CUDAGRAPH=NONE` + `SPEC=ngram` crashes the engine

Configs G and H both **died mid-benchmark** with the same failure:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 156.00 MiB.
GPU 0 has ... 39.88 MiB is free.
  at model_executor/layers/utils.py:98  default_unquantized_gemm
-> EngineDeadError (x65 subsequent requests)
```

This is **reproducible, not a flake**. On a fresh boot of config G the C64
saturation run completed (394 tok/s) and the server was dead immediately after —
`spec_bench` and `prefix_bench` then got `Connection refused`, which is why those
rows are missing for G and H.

The mechanism: CUDA-graph capture pre-reserves a fixed activation arena at
startup. With `cudagraph_mode=NONE` nothing reserves it, so speculation's extra
draft and verify buffers are allocated ad hoc at peak batch size — and at C64
there is no headroom left. Configs E and F (graphs off, no speculation) never
crash; configs A–D (graphs on, with speculation) never crash. **Only the
combination fails.**

Note the G/H C64 numbers in the table above were recorded *before* the crash and
are real, but the configuration is not usable: treat them as measurements of a
server that is about to fall over.

**Do not combine `CUDAGRAPH=NONE` with `SPEC=ngram`.** Both are non-default, and
together they are unsafe at high concurrency.

---

## What `PREFIX_CACHE` is actually worth

`bench/prefix_bench.py` sends a 7,363-token shared system prompt at C8, three
rounds, first discarded so the cache is warm. The `unique` arm sends a
same-length but per-request-unique prefix — the control that proves the win is
prefix reuse rather than warm-up.

| config | shared prefix TTFT | unique prefix TTFT | shared round time |
|---|---|---|---|
| **A — cache on** | **249 ms** | 5163 ms | **1.02 s** |
| B — cache off | 5170 ms | 5175 ms | 9.57 s |
| **C — cache on + ngram** | **245 ms** | 5281 ms | 1.61 s |
| D — cache off + ngram | 5294 ms | 5299 ms | 10.37 s |

**20.8x lower TTFT and 9.4x faster rounds** when a prefix is shared. The control
column is flat across every config (5163–5299 ms), which is exactly what it
should be: with nothing to reuse, the cache changes nothing.

Two conclusions:

- **The multiplier is a property of your workload, not the server.** It scales
  with how much of each request is shared prefix versus fresh tokens. 7.4 k
  shared tokens against 64 output tokens gives 21x; a short system prompt with
  long answers would give almost nothing. Re-measure for your own traffic.
- **Leaving it on is free.** With no shared prefix the cost is block-table
  bookkeeping, invisible in every measurement here (9.57 s vs 9.59 s).

---

## vLLM vs Ollama

Same card, same day, same eight prompts, both engines tuned, **thinking disabled
on both sides**. vLLM is config A (the shipped default); Ollama runs `q8_0` KV
with `NUM_PARALLEL=16`.

<!-- generated by bench/make_report.py --table vs -->
| cohort | vLLM e2e tok/s | Ollama e2e tok/s | winner | vLLM TPOT | Ollama TPOT |
|---|---|---|---|---|---|
| C1 | 93.6 | **113.0** | Ollama 1.21x | 10.65 ms | 8.74 ms |
| C2 | **174.2** | 157.8 | **vLLM 1.10x** | 10.65 ms | 11.01 ms |
| C4 | **343.2** | 189.0 | **vLLM 1.82x** | 10.77 ms | 19.44 ms |
| C8 | **661.8** | 251.3 | **vLLM 2.63x** | 11.18 ms | 29.41 ms |

Both engines generated the same amount of work — ~7,600 output tokens across the
8-prompt cohort — so these rates compare like for like.

**The crossover sits between C1 and C2.** Ollama leads single-stream by 1.21x,
the two are within 10% at two streams, and from four streams up vLLM pulls away
to 2.63x.

There is no C16 row: `bench/prompts_real.jsonl` holds 8 prompts and the cohort
issues 8 requests, so a 16-wide pool cannot be filled. When it was run anyway it
simply re-measured C8 (vLLM 655.2 vs 652.9; Ollama 251.0 vs 251.3), so it has
been dropped from the harness rather than reported as an extra data point.
Around 660 tok/s is where this card saturates on this workload.

**The TPOT columns explain the shape**, and they matter more than the
aggregate. TPOT is the time between output tokens — what a user actually feels.
Ollama starts ahead (8.74 ms vs 10.65 ms at one stream) and then degrades
steeply: 11.0 → 19.4 → 29.4 ms as streams are added. vLLM barely moves:
**10.65 → 11.18 ms from one stream to eight.**

That flatness is the whole argument for continuous batching. vLLM reads the
weight set once per step and amortises it across everything in flight, so the
eighth concurrent user sees essentially the latency the first one did. Ollama's
scheduler does not, so each added stream slows down all the others.

Time-to-first-token follows the same pattern: vLLM 49 → 116 ms across C1–C8,
Ollama 84 → 232 ms.

Accuracy is **not** a differentiator: measured separately at GSM8K n=1000, the
two are a statistical tie (64.2% vs 62.5%, z = 0.79). The choice between engines
is purely about serving characteristics.

> **The per-stream columns in this table were wrong until now, and the tables
> are now generated rather than typed.** Two adjacent columns were computed by
> different formulas: vLLM's "per-stream" was me dividing aggregate throughput
> by the concurrency, while Ollama's was a genuine measured median. At C2 that
> printed "Ollama 91.3 vs vLLM 87.1" — appearing to favour Ollama — even though
> Ollama's aggregate was *lower* (160.9 vs 174.2). Two different quantities in
> neighbouring columns is not a comparison.
>
> Both columns are now **TPOT**, a per-request quantity both engines measure
> directly. And the table is emitted by `bench/make_report.py` from
> `bench/measurements.json`, so ratios are computed instead of asserted:
> `python bench/make_report.py --check` fails the build if a C1 aggregate
> contradicts its own TPOT, if throughput falls as concurrency rises, or if the
> two engines did not generate comparable token counts.

> **These Ollama rows are a correction.** The first pass through this matrix ran
> Ollama with **thinking left on**, because `bench/ollama_bench.py` had no
> control for it and posts to `/api/chat`, which applies the chat template.
> Probed on the first cohort prompt, `gemma4:e4b` returned **0 characters of
> answer and 485 characters of hidden reasoning** — it was being scored on a
> different task than vLLM.
>
> The vLLM side was never affected: `vllm bench serve` defaults to
> `--endpoint /v1/completions`, a raw completion path with no chat template, so
> thinking never fires there; `spec_bench.py` and `prefix_bench.py` already sent
> `enable_thinking:false` explicitly.
>
> With `think:false` on both sides the picture changes in both directions —
> Ollama gets **faster** at low concurrency (C2 126.0 → 157.9, C4 160.2 → 188.3)
> and **slower** at high concurrency (C8 307.8 → 251.0), while its C1 TTFT falls
> from 3.9 s to 79 ms. The corrected result widens vLLM's C8 lead from 2.13x to
> 2.61x. `bench/ollama_bench.py` now sends `think:false` on both the measured
> call and the warmup.

---

## Multimodal: text + image, text + audio, text + both

Gemma 4 E4B carries an image tower and an audio tower. These runs attach real
media to a text instruction and measure what that ingestion costs. Assets are
genuine, not synthetic filler: a 1024×683 photograph and 15 s of recorded speech
(`bash bench/fetch_mm_assets.sh`).

**Comprehension is verified before any timing** — `bench/mm_check.py` asks
questions only answerable from the attachment. All five cases pass:

| case | model replied |
|---|---|
| photo → what animal, on what? | "a cat, and it is standing on snow" |
| photo → OCR the large letters | "HOLLYWOOD" |
| photo → describe the structure | "stone castle on a small island surrounded by water" |
| **audio → transcribe** | **verbatim, all five Harvard sentences** |
| photo + audio in one request | cat *and* "The birch canoe slid…" |

This gate exists because throughput alone cannot tell a working pipeline from a
broken one: a server that silently drops the media still emits tokens at full
speed. That failure actually shipped here once, when `soundfile` was missing.

### Results

512 output tokens, warmup excluded, thinking off on both engines:

| modality | engine | C1 | C2 | C4 | C8 | C1 TTFT | per-stream @C8 |
|---|---|---|---|---|---|---|---|
| text + image | **vLLM** | 91.0 | 171.9 | **328.5** | **671.5** | **47 ms** | **88.2** |
| text + image | ollama | **111.3** | 166.7 | 215.6 | 263.2 | 97 ms | 35.5 |
| text + audio | **vLLM** | 92.7 | **182.4** | **337.9** | **630.1** | **48 ms** | **86.3** |
| text + audio | ollama | **104.0** | 169.6 | 212.7 | 225.5 | 133 ms | 33.9 |
| text + image + audio | **vLLM** | 91.5 | 174.9 | **338.1** | **670.3** | **63 ms** | **86.4** |
| text + image + audio | ollama | **101.8** | 160.9 | 181.7 | 205.6 | 166 ms | 31.2 |

(e2e tok/s unless noted.) The pattern repeats the text-only cohort exactly, which
is the sanity check that matters: **Ollama wins single-stream by 1.1–1.2x, vLLM
wins from C2 up, reaching 2.55x (image), 2.79x (audio) and 3.26x (both) at C8.**

Three things worth drawing out:

- **Attaching media is nearly free for throughput.** vLLM does 91.0 tok/s with an
  image against 87.9 text-only at C1 — the towers run once inside prefill and
  decode is untouched. The cost lands in TTFT, and even there it is 47–63 ms.
- **Carrying both modalities costs almost nothing extra.** vLLM's "both" column
  tracks its single-modality columns within a few percent (670.3 vs 671.5 at C8).
  Ollama degrades noticeably instead: 263.2 → 205.6 at C8, a 22% drop for adding
  the second attachment.
- **The gap is widest on the hardest case.** vLLM's lead grows from 2.55x with one
  image to 3.26x with image + audio together, because batched prefill absorbs the
  extra encoder work while Ollama pays it per request.

The audio tower runs bf16 in this checkpoint, which is why the server must use
`--dtype bfloat16`; with `float16` every audio request fails on a dtype mismatch
inside the tower.

---

### Long context: 64K and 128K

The model supports 131,072 tokens, but `MAX_LEN` defaults to 32768 because a
full-context request is expensive: at ~24.5 KB/token of fp8 KV, one 128K request
holds **3.1 GB** of cache. Long context therefore needs `MAX_SEQS` lowered to
match — and since `MAX_SEQS` also sets the CUDA-graph capture ceiling, dropping
it frees memory on both sides of the ledger.

Configurations that boot cleanly on this card:

| context | command | KV pool | concurrency headroom |
|---|---|---|---|
| 32K (default) | `bash batch/start_gemma.sh` | 1,071,949 tok | 32x |
| 64K | `MAX_LEN=65536 MAX_SEQS=8 bash batch/start_gemma.sh` | 1,229,591 tok | 18.8x |
| 128K | `MAX_LEN=131072 MAX_SEQS=4 bash batch/start_gemma.sh` | 1,329,564 tok | 10.1x |

Note the pool gets *larger* at longer context: fewer capture sizes means less
graph memory, which vLLM hands back to the KV cache.

Measured, 512 output tokens per request:

| context | conc | output tok/s | TTFT | median TPOT |
|---|---|---|---|---|
| 64K | C1 | 23.8 | 15.5 s | 11.7 ms |
| 64K | C2 | 46.5 | 11.8 s | 12.8 ms |
| 64K | C4 | 53.1 | 18.1 s | 27.3 ms |
| 128K | C1 | 9.9 | 45.6 s | 12.5 ms |
| 128K | C2 | 19.2 | 34.7 s | 14.7 ms |
| 128K + `INT8_ACT=int8` | C1 | **11.3** | **38.8 s** | 12.9 ms |
| 128K + `INT8_ACT=int8` | C2 | **21.9** | **29.6 s** | 15.1 ms |

Output tok/s looks low, but that is an artifact of the ratio: a 128K request
spends nearly all its time *reading* 131,072 tokens to emit 512. The number that
shows decode is still healthy is **TPOT, which stays at 12–15 ms even with a
128K-token KV cache** — essentially the same as at 32K. **TTFT is the real cost
of long context**, and because prefill is compute-bound, `INT8_ACT=int8` cuts it
by ~15%. For long-context work, turn it on.

#### vs Ollama — read the ingest counts first

Ollama does not ingest the context you configure. With
`OLLAMA_CONTEXT_LENGTH=65536` it reports `prompt_eval_count = 32,771`; at
`131072` it reports `65,539`. **It silently clamps to half the configured
window**, regardless of how long the prompt is (verified by direct probe with
20k- and 60k-word prompts). So "Ollama at 128K" is really Ollama at 64K.

To compare honestly, vLLM was re-measured at Ollama's *actual* ingest sizes:

| prompt tokens ingested | conc | vLLM out tok/s | Ollama out tok/s | vLLM TTFT | Ollama TTFT |
|---|---|---|---|---|---|
| 32,771 | C1 | **42.3** | 33.9 | 6.3 s | 6.4 s |
| 32,771 | C2 | **82.5** | 40.5 | **4.9 s** | 7.3 s |
| 65,539 | C1 | **31.9** | 18.1 | **10.1 s** | 16.1 s |
| 65,539 | C2 | **61.5** | 19.9 | **7.9 s** | 18.0 s |

vLLM is **1.25x to 3.09x faster**, and unlike the short-prompt cohorts — where
Ollama wins single-stream — it leads at every point here. The gap widens with
both context length and concurrency: at 65,539 tokens and C2 it is 3.1x on
throughput and 2.3x on TTFT, because Ollama's per-request cost stays flat while
vLLM's continuous batching shares the prefill and decode work.

> **Two measurement traps found while building this comparison**, both of which
> would have flattered Ollama, and both worth knowing if you benchmark it
> yourself:
> 1. **Prompt-prefix caching.** Re-sending the same long prompt made a 65k-token
>    prefill appear to take 12.5 s instead of 28.3 s. Every request must carry a
>    unique prompt, salted per run.
> 2. **Repetitive filler tokenizes densely.** A "65,548-token" prompt built by
>    repeating a stock sentence tokenized to only 49,625 real tokens. Use random
>    words from a large vocabulary and verify with `prompt_eval_count`.
>
> The harness used here is `bench/ollama_longctx.py`; it reports
> `prompt_eval_count` on every row so the ingest size is always visible:
> ```bash
> CTX_LEN=131072 NUM_PARALLEL=2 bash batch/start_ollama.sh
> venv/bin/python bench/ollama_longctx.py --ctx 131072 --conc 1 2
> ```

Ollama ran with `NUM_PARALLEL=2` for these rows: it allocates KV as
`CONTEXT_LENGTH × NUM_PARALLEL`, so the default 16 does not fit a long window in
24 GB. vLLM's paged KV cache has no such multiplication, which is why it can
offer 10x concurrency headroom at 128K out of the same card.

---

### Why the official QAT checkpoint

This deployment serves Google's own **quantization-aware-trained** int4 build
rather than a community post-training quantization (PTQ). The two were measured
head to head on this card before the switch, and the result was decisive:

- **Speed was a wash** — every throughput row landed within 2%, i.e. run-to-run
  noise. Both formats end up on the same Marlin kernels, so the QAT build's
  slightly larger weights (9.7 GB vs 9.4 GB, from finer group-32 scales) cost
  nothing measurable.
- **Quality was not.** GSM8K at n=1000: **64.2% for QAT versus 50.9% for the PTQ
  build** — +13.3 points, z = 6.02, p < 0.001. Perplexity improved ~4x on English
  and ~3x on code.

QAT trains the model *with* the quantization error in the loop instead of fitting
scales to a frozen model afterwards. On a small model, where every weight carries
more of the capability, that matters far more than it would at 27B.

The lesson generalises: **prefer a first-party QAT checkpoint over a community
PTQ one when both exist**, and re-check whenever the model vendor publishes new
artifacts. `verify.sh` accepts either on-disk format (`weight_packed` for
compressed-tensors, `qweight` for GPTQ), so swapping `MODEL=` still works if you
want to compare against another build yourself.

### Why the GPU only draws 200 W

A reasonable worry when watching `nvidia-smi` during inference: the card is
allowed 450 W but decode sits around 200 W, so surely something is being left on
the table. It is not. Measured on this box at a 450 W limit (medians over a run,
synthetic `ignore-eos` loads):

| workload | power | SM busy | mem controller | clock | throughput |
|---|---|---|---|---|---|
| idle | 15 W | 3% | 4% | 210 MHz | — |
| decode, C1 | 200 W | 72% | 45% | 1695 MHz | 94.9 tok/s |
| decode, C8 | 204 W | 73% | 45% | 1695 MHz | 702 tok/s |

---

## Recommendations

| your workload | configuration | expected |
|---|---|---|
| **General serving / mixed traffic** | **default** (`SPEC` off, `PREFIX_CACHE=1`) | 655 tok/s @C8, 11 ms TPOT |
| Shared system prompt or few-shot block | default — cache already on | **21x lower TTFT** |
| Quote / edit / reformat / extractive RAG | `SPEC=ngram` **plus** `PREFIX_CACHE=1` | up to **919 tok/s** |
| Single interactive user, short prompts | either engine; Ollama edges it 1.21x | 94–113 tok/s |
| Image / audio / both attached | default — no extra tuning needed | **630–671 tok/s @C8**, TTFT 47–63 ms |
| Maximum aggregate throughput | default at C64 | **2621 tok/s** |

**Never disable CUDA graphs.** They are worth **4.4–4.5x** on this model and
cost only 0.55% of the KV pool. Combining `CUDAGRAPH=NONE` with `SPEC=ngram`
additionally crashes the engine at high concurrency.

**Do not enable `SPEC=ngram` globally.** It is a 3.4x win on copy-style work and
a 33–48% loss on everything else, and at C64 it costs 3.4x. It belongs on a
dedicated endpoint for a known workload, not on a general-purpose server.

Before enabling it anywhere, measure the acceptance rate on *your* traffic —
vLLM logs it. On this cohort only **3% of drafted tokens were accepted**, so 97%
of the speculative compute was discarded:

```bash
grep -o "Per-position acceptance rate: [0-9.]*" gemma.log | tail -20
```

Below roughly 50% acceptance the verification passes cost more than the drafts
save, and speculation is a net loss regardless of which n-gram backend is used.

**Leave `PREFIX_CACHE=1` on unconditionally.** It is worth up to 21x when
prefixes are shared and costs nothing measurable when they are not.

---

## Notes on method

- **Warmup is never timed.** Every config fires an untimed request first; model
  load, JIT and `torch.compile` stay out of the numbers.
- **Thinking is disabled on both engines.** vLLM's cohort and saturation runs go
  through `vllm bench serve` against `/v1/completions` — a raw completion path
  with no chat template, so thinking never fires; `spec_bench.py` and
  `prefix_bench.py` send `enable_thinking:false` explicitly. Ollama needs
  `think:false`, which `bench/ollama_bench.py` now sends. Without it the model
  spends its whole token budget on hidden reasoning and returns no answer, which
  is a different task from the one vLLM is being timed on.
- **Token counts come from each engine's own API response**, never from a log
  or a client-side re-tokenization. vLLM: `vllm bench serve` sets
  `stream_options.include_usage` and reads `usage.completion_tokens`. Ollama:
  `bench/ollama_bench.py` reads `eval_count` from the final `/api/chat` chunk —
  its native endpoint and native counter. The two engines generated ~7,600
  tokens each on the cohort, confirming equal work.
- **Per-stream rate is wall-clock on both engines.** Ollama's `eval_duration`
  covers decode only and excludes prefill; dividing by it reports a rate no
  client observes. On this text cohort the difference is ~1% (108.8 vs 110.0
  tok/s at C1) because text prefill is short — but the same defect inflated
  Ollama 2.1x in the multimodal harness, so both now use wall-clock. The ROW
  output prints the decode-only figure alongside, for reference.
- **Knobs were verified, not assumed.** Each config's engine log was grepped for
  `enable_prefix_caching`, `num_speculative_tokens`, `cudagraph_mode`,
  `gpu_memory_utilization` and the KV pool size, and those values are recorded
  in the raw log next to the results.
- **One engine at a time.** A 24 GB card holds one of these servers; Ollama
  returns HTTP 500 if vLLM still has the memory. The driver stops one before
  starting the other.
- **Two data-quality corrections made during this run**, both recorded in the
  raw log rather than silently fixed:
  1. The driver labelled cohort columns `tok/s,TPOT,TTFT`, but `vllm bench serve`
     prints throughput → **TTFT** → **TPOT**. The values were always right; the
     header was wrong. Verified against a live run and corrected in place.
  2. Configs C and D produced no rows from `spec_bench`/`prefix_bench` inside the
     unattended driver, and the first explanation offered here — "a timing
     artifact" — was wrong. Once the driver was changed to keep stderr, configs
     G and H showed the real cause: `Connection refused`, because the engine had
     already crashed. C and D were re-run standalone and completed cleanly, so
     their numbers stand, but the diagnosis in the earlier revision did not.
     `bench/reproduce_report.sh` now tees every benchmark's full output.
- **Power limit.** The GPU is at its 450 W default. Decode never approaches it
  (~200–232 W); only int8 prefill was ever power-bound, and that case is not part
  of this matrix.

### Reproducing

Everything in this report is regenerated by one script:

```bash
bash bench/reproduce_report.sh --help     # phases, configs, runtimes
bash bench/reproduce_report.sh all        # everything (~5-6 hours)
```

It can also be run a piece at a time, which is how you should do it the first
time — nothing is overwritten, so an interrupted run resumes at phase
granularity:

```bash
bash bench/reproduce_report.sh matrix     # the 8-config factorial (~3.5 h)
bash bench/reproduce_report.sh arm1       # just configs A-D (cudagraphs on)
bash bench/reproduce_report.sh arm2       # just configs E-H (cudagraphs off)
bash bench/reproduce_report.sh config C   # a single config (~8 min)
bash bench/reproduce_report.sh ollama     # Ollama text cohort (~10 min)
bash bench/reproduce_report.sh mm         # multimodal, both engines (~45 min)
```

Results append to `bench/results-<date>.txt`; the full stdout+stderr of every
individual benchmark is kept under `bench/repro-logs/`.

The script encodes the methodology, not just the commands. In particular it:

- **records what the engine actually did** — every config's `cudagraph_mode`,
  `enable_prefix_caching`, `num_speculative_tokens`, `gpu_memory_utilization`
  and KV pool size are grepped out of the server log and written next to the
  results, because an environment variable being *set* is not evidence that it
  took effect;
- **keeps the full output of every benchmark** rather than piping straight into
  `grep '^ROW'`. That shortcut hid an engine crash for two entire rounds of
  measurement in this project;
- **detects the G/H crash and continues.** Those two configurations are expected
  to die at C64; the script records the OOM, notes it, and moves on instead of
  aborting the matrix;
- **polls `/health`** instead of sleeping a fixed time — boots range from 130 s
  to 380 s depending on speculation and compile-cache state;
- **runs the comprehension gate before any multimodal timing**, so a silently
  broken media path cannot be reported as fast;
- **warns if the power limit is below 400 W**, since these numbers were taken at
  the card's stock 450 W.

Expect small run-to-run variation. Re-running config A while writing this
section gave 86.4 / 316.8 / 628.4 tok/s at C1/C4/C8 against the 87.9 / 340.8 /
655.2 recorded above — within a few percent, and the ordering of every
comparison is unchanged.
