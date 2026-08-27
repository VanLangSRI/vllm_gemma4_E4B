# Gemma 4 E4B on one RTX 3090 Ti

vLLM serving setup for [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it)
on a single 24 GB card, with a full benchmark suite behind it.

This README covers **how to run the scripts** and **what the measurements mean**.
The measurements themselves — every table, every configuration, the raw numbers —
live in **[`report.md`](report.md)**.

```
Checkpoint    google/gemma-4-E4B-it-qat-w4a16-ct   (official QAT int4, 9.7 GB)
Throughput    662 tok/s @ 8 concurrent · 2630 tok/s @ C64 · 94 tok/s single
Latency       10.7-11.2 ms per token, flat from 1 to 8 streams
Context       32K default, 64K and 128K supported
Quality       GSM8K 64.2% (n=1000) — ties Ollama's Q4_K_M
```

---

## Contents

- [Install and run](#install-and-run) · [Configuration knobs](#configuration-knobs)
- [The benchmark scripts](#the-benchmark-scripts) · [Reproducing report.md](#reproducing-reportmd)
- [What the results say](#what-the-results-say) · [How to read a benchmark here](#how-to-read-a-benchmark-here)
- [Troubleshooting](#troubleshooting) · [Provenance](#provenance)

---

## Install and run

```bash
git clone <this repo> ~/vllm_3090 && cd ~/vllm_3090
bash setup.sh                                       # venv, vLLM 0.27.1, patches, model (~11 GB)
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env  # required if it leaves this machine
bash batch/start_gemma.sh                           # serves :18030, logs to gemma.log
```

First boot takes ~4 minutes (torch.compile + FlashInfer JIT); later boots ~140 s
from cache. Docker works too: `docker compose up -d`.

| script | what it does |
|---|---|
| `setup.sh` | venv, pinned vLLM, applies all six patches, downloads the model |
| `prepare/fetch_models.sh` | model download only (~11 GB) |
| `verify.sh --no-server` | checks GPU, patches, checkpoint, audio decode. **Run this first when anything misbehaves** |
| `batch/start_gemma.sh` | starts vLLM; every knob below is an env var |
| `batch/start_ollama.sh` | `start` / `stop` / `status` for the Ollama comparison |

For unattended operation there is a user unit; it waits for the GPU to be free
before starting, because profiling while another process still holds VRAM
permanently shrinks the KV pool for that run:

```bash
cp batch/gemma-serving.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now gemma-serving
loginctl enable-linger $USER
```

Test it:

```bash
curl http://localhost:18030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-e4b",
       "messages":[{"role":"user","content":"What is the capital of Denmark?"}],
       "max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

The server binds `0.0.0.0` and is **unauthenticated unless you give it a key**
(`api_key.txt` or `VLLM_API_KEY`). Thinking mode is off per request by default;
set `enable_thinking:true` to enable it. Tool calling works via `tools` +
`tool_choice:"auto"`. Images and audio are accepted as `image_url` / `audio_url`
content parts.

---

## Configuration knobs

All are environment variables read by `batch/start_gemma.sh`; `.env` works too.

| variable | default | effect |
|---|---|---|
| `MODEL` | `models/gemma-4-E4B-it-qat-w4a16-ct` | the QAT int4 checkpoint |
| `PORT` | `18030` | |
| `MAX_LEN` | `32768` | context window; see [long context](report.md#long-context-64k-and-128k) for 64K/128K settings |
| `MAX_SEQS` | `64` | admission ceiling **and** CUDA-graph capture ceiling |
| `GPU_UTIL` | `0.90` | fraction of VRAM vLLM may claim |
| `CUDAGRAPH` | `FULL_AND_PIECEWISE` | `PIECEWISE` or `NONE`. **Leave it alone** — see below |
| `PREFIX_CACHE` | `1` | reuse KV across requests sharing a prompt prefix |
| `SPEC` | (off) | `ngram` enables lookup speculative decoding |
| `SPEC_TOKENS` | `8` | tokens drafted per step when `SPEC=ngram` |
| `BATCHED_TOKENS` | `2048` | chunked-prefill budget |
| `VISION` / `AUDIO` / `TOOLS` | `1` | accept images / audio / tool calls |
| `INT8_ACT` | (off) | `int8` runs MLP GEMMs on int8 tensor cores |
| `EXTRA_ARGS` | | appended verbatim to `vllm serve` |

**`MAX_SEQS` and `GPU_UTIL` are coupled.** vLLM sizes the KV cache to fill
`GPU_UTIL`, *then* captures one CUDA graph per entry in a list derived from
`MAX_SEQS`. Push either too high and the engine OOMs at startup. The launcher
generates the capture list for you; if you raise `MAX_SEQS`, lower `GPU_UTIL`.

```bash
MAX_LEN=131072 MAX_SEQS=4 bash batch/start_gemma.sh   # 128K context
MAX_SEQS=256 GPU_UTIL=0.88 bash batch/start_gemma.sh  # tuned for C256
SPEC=ngram bash batch/start_gemma.sh                  # copy-heavy workloads only
```

---

## The benchmark scripts

Each isolates one thing. All exclude warmup, and all disable thinking on both
engines so the two are timed on the same task.

| script | measures | typical use |
|---|---|---|
| `bench/run_benchmarks.sh` | cohort C1–C8 + saturation rows | overall throughput |
| `bench/spec_bench.py` | copy-style vs chat-style prompts | is `SPEC=ngram` worth it here? |
| `bench/prefix_bench.py` | shared vs unique prompt prefix | is `PREFIX_CACHE` doing anything? |
| `bench/ollama_bench.py` | same cohort against Ollama | engine comparison |
| `bench/ollama_longctx.py` | 64K/128K prompts against Ollama | long-context comparison |
| `bench/mm_check.py` | **does the model actually see/hear the media?** | run before any multimodal timing |
| `bench/mm_bench.py` | image / audio / both, either engine | multimodal throughput |
| `bench/quality_battery.py` | GSM8K + perplexity | accuracy after any quant/kernel change |
| `bench/make_report.py` | turns `bench/measurements.json` into report tables; `--check` validates consistency | after updating any measurement |
| `bench/fetch_mm_assets.sh` | downloads real photos + recorded speech | once, before `mm_bench` |
| `bench/fetch_quality_data.sh` | downloads GSM8K + wikitext | once, before `quality_battery` |

```bash
# throughput
bash bench/run_benchmarks.sh

# is a knob earning its keep on YOUR workload?
venv/bin/python bench/spec_bench.py   --label mine --conc 1 4
venv/bin/python bench/prefix_bench.py --label mine --conc 8 --rounds 3

# multimodal: fetch assets, verify comprehension, then time it
bash bench/fetch_mm_assets.sh
venv/bin/python bench/mm_check.py
venv/bin/python bench/mm_bench.py --api vllm --modality both --conc 1 2 4 8

# quality (works against Ollama too, via the env overrides)
bash bench/fetch_quality_data.sh
venv/bin/python bench/quality_battery.py qat --gsm-only --gsm-n 1000
VLLM_API=http://127.0.0.1:11434/v1 VLLM_MODEL=gemma4:e4b \
  venv/bin/python bench/quality_battery.py ollama --gsm-only --gsm-n 1000
```

> **Only one engine can hold the card at a time.** 24 GB fits one of these
> servers; Ollama returns HTTP 500 if vLLM still has the memory. Stop one before
> starting the other.

---

## Reproducing report.md

One script regenerates every number in the report:

```bash
bash bench/reproduce_report.sh --help     # phases, configs, runtimes
bash bench/reproduce_report.sh all        # everything (~5-6 hours)
```

Run it in pieces the first time — nothing is overwritten, so an interrupted run
resumes at phase granularity:

```bash
bash bench/reproduce_report.sh arm1       # configs A-D (CUDA graphs on)
bash bench/reproduce_report.sh arm2       # configs E-H (CUDA graphs off)
bash bench/reproduce_report.sh config C   # one config (~8 min)
bash bench/reproduce_report.sh ollama     # Ollama cohort (~10 min)
bash bench/reproduce_report.sh mm         # multimodal, both engines (~45 min)
```

Results append to `bench/results-<date>.txt`; full stdout+stderr of every
individual run lands in `bench/repro-logs/`.

Expect a few percent of run-to-run variation. Re-running config A gave
86.4 / 316.8 / 628.4 tok/s at C1/C4/C8 against 87.9 / 340.8 / 655.2 in the
report — the ordering of every comparison was unchanged.

---

## What the results say

Full tables and methodology in **[`report.md`](report.md)**. This is the summary
and what to do with it.

### Pick a configuration

| your workload | configuration | expect |
|---|---|---|
| **General serving, mixed traffic** | **defaults** | 662 tok/s @C8, 11.2 ms/token |
| Shared system prompt or few-shot block | defaults (cache already on) | **21x lower TTFT** |
| Quote / edit / reformat / extractive RAG | `SPEC=ngram` **and** `PREFIX_CACHE=1` | up to **919 tok/s** |
| Batch, prefill-heavy, or long context | `INT8_ACT=int8` | +20% @C64, −15% TTFT @128K |
| Image / audio attached | defaults | 630–671 tok/s @C8, TTFT 47–63 ms |
| Maximum aggregate | defaults at C64 | **2630 tok/s** |
| One interactive user, short prompts | either engine | Ollama edges it 1.21x |

**The defaults are the right answer for most cases.** Every non-default knob
below is a trade, and two of them are traps.

### The three findings that matter

**1. CUDA graphs are worth 4.5x — never turn them off.**

| | graphs on | graphs off |
|---|---|---|
| cohort C1 | 87.9 | 20.0 tok/s |
| cohort C4 | 340.8 | 75.1 tok/s |
| cohort C8 | 655.2 | 145.9 tok/s |
| per-token latency | 11 ms | 113–118 ms |

(These four rows come from the 8-configuration matrix — configs A and E — which
was run as one batch. The vLLM-vs-Ollama table further down is a later
re-measurement of the same default configuration, hence 661.8 rather than 655.2
at C8. Both are in `bench/measurements.json`; the 1% gap is run-to-run noise.)

Disabling them frees just 0.55% of the KV pool, and `CUDAGRAPH=NONE` combined
with `SPEC=ngram` **crashes the engine** at high concurrency (reproducible OOM).
This repo once shipped with graphs disabled because a startup OOM was
misdiagnosed as a torch bug; that mistake cost 4.5x.

**2. Speculation is workload-selective, not a global win.**

| | default | `SPEC=ngram` |
|---|---|---|
| copy-style prompt, C4 | 366.5 | **919.2 tok/s** |
| ordinary chat, C1 | 95.7 | 64.1 tok/s |
| saturation C64 | 2621.0 | 773.9 tok/s |

The mechanism explains the split: n-gram lookup drafts by searching the
request's own prompt. When the answer is already there it wins 3.4x; when the
model is composing new text only **3% of drafted tokens are accepted** and 97%
of the speculative compute is discarded. Measure before enabling:

```bash
grep -o "Per-position acceptance rate: [0-9.]*" gemma.log | tail -20
```

Below roughly 50% acceptance, speculation is a net loss on any backend. Note
`ngram_gpu` means the *search* runs on the GPU — it is not a faster kind of
speculation.

**3. Prefix caching is free, and occasionally enormous.**

With a 7.4k-token shared system prompt at C8: TTFT **249 ms vs 5170 ms** (20.8x),
round time 9.4x faster. With no shared prefix it changes nothing measurable —
so leaving it on costs nothing. The multiplier depends entirely on how much of
each request is shared prefix; re-measure for your own traffic rather than
quoting 21x.

### vLLM vs Ollama

| concurrency | vLLM | Ollama | | vLLM TPOT | Ollama TPOT |
|---|---|---|---|---|---|
| C1 | 93.6 | **113.0** | Ollama 1.21x | 10.65 ms | 8.74 ms |
| C2 | **174.2** | 157.8 | vLLM 1.10x | 10.65 ms | 11.01 ms |
| C4 | **343.2** | 189.0 | vLLM 1.82x | 10.77 ms | 19.44 ms |
| C8 | **661.8** | 251.3 | **vLLM 2.63x** | 11.18 ms | 29.41 ms |

Ollama wins single-stream; vLLM wins everywhere else and the gap widens with
load. TPOT — the gap between output tokens, i.e. what a user feels — shows why:
vLLM stays at 10.7–11.2 ms from one stream to eight, while Ollama degrades from
8.7 to 29.4 ms. On long prompts vLLM leads at
*every* point, 1.25–3.09x — and Ollama silently ingests only **half** the context
it is configured for.

Accuracy is a tie (GSM8K n=1000: 64.2% vs 62.5%, z = 0.79), so the choice is
purely about serving characteristics: one interactive user → either; more than
one → vLLM.

### Where the hardware limit is

Single-stream decode reaches 91.4 tok/s against a hard ceiling of
1008 GB/s ÷ 9.7 GB = **104 tok/s**. That is 88% of what the memory bus can
deliver, so **there is no single-stream headroom left** without changing the
quantization. Decode draws only ~200 W of the card's 450 W not because it is
idle but because it is bandwidth-bound — prefill on the same card pulls 319 W.
Throughput past that point comes from batching, which is exactly what the C8 and
C64 numbers show.

---

## How to read a benchmark here

Several numbers in this project were wrong before they were right, always for
the same reason: the harness was measuring something other than what it claimed.
The habits below are baked into the scripts, and are worth keeping if you extend
them.

- **An absent flag is not a disabled feature.** `PREFIX_CACHE=0` did nothing for
  a while: the launcher simply omitted `--enable-prefix-caching`, and vLLM
  enables it by default. Always confirm from the engine log, not the env var.
- **Verify the knob took effect.** `bench/reproduce_report.sh` greps
  `cudagraph_mode`, `enable_prefix_caching`, `num_speculative_tokens` and the KV
  pool size out of the server log and records them next to every result.
- **Never `cmd | grep` a benchmark.** That shortcut discarded stderr and hid an
  engine crash across two full rounds of measurement. Tee first, extract after.
- **Compare like with like.** Ollama's `eval_duration` excludes prefill and the
  vision/audio encoder; dividing by it inflated its per-stream rate 2.1x and made
  it look faster than it was. Use wall-clock on both sides.
- **Disable thinking on both engines,** or one of them spends its whole token
  budget on hidden reasoning and returns an empty answer at full apparent speed.
- **Throughput cannot detect a broken pipeline.** A server that silently drops an
  attachment still emits tokens quickly. That is why `bench/mm_check.py` asks
  questions only answerable from the media and runs *before* any timing.
- **Check the sample size.** GSM8K at n=200 has a ±6.9 point confidence interval.
  A 7-point "difference" at that size is noise; the n=1000 run showed a tie.
- **When a result contradicts everything else you have measured, suspect the
  measurement.** Every incorrect table in this project's history was a harness
  bug, not a surprising property of the engine.

---

## Troubleshooting

Run `bash verify.sh --no-server` first — it checks the GPU, all six patches, the
checkpoint format and the audio decode path.

| symptom | cause and fix |
|---|---|
| OOM during "Capturing CUDA graphs" | `MAX_SEQS`/`GPU_UTIL` too high together. Use defaults, or lower `GPU_UTIL` when raising `MAX_SEQS` |
| Engine dies mid-run with `EngineDeadError` | `CUDAGRAPH=NONE` together with `SPEC=ngram`. Do not combine them |
| Audio silently ignored, no error | `soundfile` missing. `venv/bin/pip install soundfile librosa`. Grep the log for `Failed to load audio via soundfile` |
| `vllm bench serve` wants `vllm[bench]` | lazy pandas import: `pip install pandas`. Do **not** install the extra — it re-resolves vLLM and reverts the patches |
| Ollama returns HTTP 500 | vLLM still holds the VRAM; stop it first |
| Empty reply from Ollama | it emitted only `thinking`; raise `num_predict` or send `think:false` |
| Every audio request fails on dtype | the audio tower is bf16 — the server must run `--dtype bfloat16`, not `float16` |
| `INT8_ACT=int8` does nothing | the Marlin patches are missing; re-run `bash setup.sh` |
| Slow first boot | normal: torch.compile + FlashInfer JIT. Later boots use the cache |
| WSL2 | set `VLLM_WSL2_ENABLE_PIN_MEMORY=1` |

Six source patches ship in `patches/` and are applied by `setup.sh`: four make
vLLM 0.27.1 handle Gemma 4's heterogeneous attention head dimensions (256 on
sliding layers, 512 on full-attention ones) and fp8 KV on sm86; two more are
only needed for `INT8_ACT=int8`. `verify.sh` reports the state of each.

---

## Provenance

- Model: [`google/gemma-4-E4B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct),
  Google's quantization-aware-trained int4 build of
  [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) (Apache-2.0).
  QAT beat a community post-training int4 build by 13.3 GSM8K points at identical
  speed — [see report.md](report.md#why-the-official-qat-checkpoint).
- Marlin int8 patches, `bench/prompts_real.jsonl` and the verify/bench structure
  come from [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
  (Apache-2.0), against the same pinned vLLM 0.27.1.
- `examples_tool_chat_template_gemma4.jinja` is vendored from
  [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0).
- Multimodal assets: Wikimedia Commons photographs and Open Speech Repository
  Harvard sentences, fetched by `bench/fetch_mm_assets.sh`.
