#!/bin/bash
# Gemma 4 E4B (8B total, effective-4B) on one 24 GB card — BATCH / THROUGHPUT mode.
# Tuned for the RTX 3090 Ti 24 GB (sm86 / Ampere): Marlin int4 weights, fp8 KV,
# 128k-capable context, high admission ceiling for short/medium prompts.
#
# This mirrors the approach of the qwen38-27b-rtx3090 reference repo (vLLM 0.27.1,
# quantized weights, tuned vLLM flags, OpenAI API) adapted to a pure-transformer
# model, so several of Qwen's knobs do not apply here:
#   - no --mamba-ssm-cache-dtype: Gemma 4 is plain attention (35 sliding-window
#     512 + 7 full layers), there is no recurrent state to store.
#   - no speculative decoding in batch mode: Gemma 4's MTP drafter is nightly-only
#     in vLLM, and batch throughput comes from batching, not speculation.
#
# Hard-won settings, don't change casually:
#  - --kv-cache-dtype fp8: KV per token drops from ~49 KB to ~24.5 KB on this
#    architecture ((35 x 256 + 7 x 512) x 2 for K+V), roughly doubling the token
#    pool. FlashInfer fp8 KV runs fine on Ampere (same as the reference box).
#  - --async-scheduling: overlapping scheduler and engine iteration, a
#    pure-throughput win on v1 engines.
#  - --max-num-batched-tokens 2048: bigger chunks inflate the profiled
#    activation peak, which comes out of the KV pool and shrinks it. 2048 is
#    the reference box's measured sweet spot and stays a good default here.
#  - --gpu-memory-utilization 0.90: the box runs a display (X holds ~300-500 MB)
#    AND cudagraph capture needs ~0.56 GiB that vLLM only partly accounts for.
#    0.93 profiles an 11.8 GiB KV pool and then dies with a CUDA OOM in the
#    post-capture kernel warmup -- measured, not theoretical. 0.90 leaves an
#    11.7 GiB pool (1.13M tokens) with enough slack to start reliably.
#  - --max-num-seqs 64: this is ALSO the cudagraph capture ceiling. vLLM derives
#    its capture-size list from max-num-seqs, and every captured size costs
#    VRAM on top of an already-full KV pool. At 256 the 51-entry capture list
#    OOMs the engine during capture (that is the real cause of the "cudagraph
#    crash" this repo used to blame on torch 2.13 -- see README). 64 captures
#    in ~2 s for 0.56 GiB and still saturates this card: the cohort bench tops
#    out at C8 (658 tok/s aggregate), well inside the window.
#
# Usage:
#   bash batch/start_gemma.sh                     # defaults
#   MAX_SEQS=256 GPU_UTIL=0.88 bash batch/start_gemma.sh    # tuned for C256 workloads
#   CUDAGRAPH=NONE bash batch/start_gemma.sh      # disable CUDA graphs (see README)
#   INT8_ACT=int8 bash batch/start_gemma.sh       # int8 activations (needs the Marlin patches)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

# Google's official QAT w4a16 checkpoint (compressed-tensors): 9.7 GB of weights,
# leaving ~11 GB for the KV pool plus activations and cudagraph capture.
# Quantization-aware training makes this markedly more accurate than the
# post-training int4 build at the same speed: GSM8K 64.2% vs 50.9% (n=1000,
# p<0.001), perplexity ~4x lower. Set MODEL= to override.
MODEL=${MODEL:-$REPO/models/gemma-4-E4B-it-qat-w4a16-ct}
PORT=${PORT:-18030}
SERVED_NAME=${SERVED_NAME:-gemma-4-e4b}
MAX_SEQS=${MAX_SEQS:-64}
# Default context: 32k keeps the pool big for concurrent short/medium prompts
# (a full 128k request costs 3.1 GB of fp8 KV, i.e. only ~4 fit in the pool).
# Set MAX_LEN=131072 for long-document work; expect single-digit concurrency.
MAX_LEN=${MAX_LEN:-32768}
BATCHED_TOKENS=${BATCHED_TOKENS:-2048}
GPU_UTIL=${GPU_UTIL:-}

# INT8_ACT=int8: int8 activations (W4A8) on the Marlin GEMMs — int8 tensor
# cores on the MLP linears while the weights stay int4. Same mechanism the
# reference box measured at +35% aggregate on the 27B model. Requires the two
# patches in patches/ to be applied to the venv (bash setup.sh). Off by
# default so the stock 0.27.1 wheel works unchanged; enable and re-bench.
INT8_ACT=${INT8_ACT-}

# PREFIX_CACHE=1 (default on): reuse the KV of a shared prompt prefix across
# requests. For an API backend where every request carries the same system
# prompt / few-shot block, the prefix is prefilled once instead of per request.
# Measured here with a 7.4k-token shared system prompt at C8: TTFT 249 ms with
# the cache vs 5176 ms without it (~21x), and the round runs 9.2x faster. With
# no shared prefix it is inert -- costs block-table bookkeeping only.
# NOTE: leaving the flag off entirely does NOT disable prefix caching — vLLM
# 0.27.1 enables it by default, so PREFIX_CACHE=0 silently did nothing until the
# explicit --no-enable-prefix-caching branch below was added.
if [ "${PREFIX_CACHE:-1}" = "1" ]; then
  PREFIX_ARGS="--enable-prefix-caching"
else
  PREFIX_ARGS="--no-enable-prefix-caching"
fi

# SPEC=ngram: lookup speculative decoding. Drafts the next tokens by finding the
# longest suffix of what has been generated inside the request's own prompt, and
# proposing whatever followed it. This is the model-agnostic version of the
# reference repo's DFlash2 lookup-drafting (DFlash2 itself needs a Qwen drafter
# checkpoint that does not exist for Gemma 4 E4B).
#
# It is a big win ONLY when the output largely already appears in the prompt --
# quoting, editing, reformatting, extractive RAG. Measured here (512 out, C1):
#   verbatim-copy prompt:   95 -> 310 tok/s   (+226%)
#   ordinary chat prompt:   96 ->  62 tok/s   (-35%)
# On ordinary generation every rejected draft is wasted compute, hence OFF by
# default. Note it also forces GPU_UTIL down (spec state needs VRAM the KV pool
# would otherwise take) and downgrades cudagraphs to PIECEWISE on FlashInfer.
#
# SPEC_TOKENS is not worth tuning down to rescue chat: at 3 instead of 8 the
# acceptance rate stays ~0.06/position and chat still loses 45-57% (measured
# 92.6 -> 50.5 at C1, 711.5 -> 390.9 at C8), while copy keeps most of its win
# (699 -> 857 at C8). The KV pool also shrinks 1,059,778 -> 898,939 tokens from
# the GPU_UTIL drop alone. Below ~50% acceptance speculation cannot pay for its
# verify pass, and n-gram lookup only clears that bar when the answer is already
# in the prompt.
#
# No other speculator is available for this checkpoint: vLLM 0.27.1 does ship
# gemma4_mtp/gemma4_dspark and Gemma4ForCausalLM declares SupportsEagle3, but
# all of them need drafter weights, and the QAT checkpoint has none (0 of its
# 2763 tensors are mtp/nextn/eagle). `suffix` decoding exists but is rejected
# under --async-scheduling, which is worth more than the speculator would be.
SPEC=${SPEC:-}
SPEC_ARGS=""
if [ "$SPEC" = "ngram" ]; then
  # 'ngram_gpu', not 'ngram': the CPU variant is rejected alongside
  # --async-scheduling ("async scheduling is only supported with
  # EAGLE/MTP/Draft Model/NGram GPU/DSpark kind of speculative decoding").
  SPEC_ARGS="--speculative-config {\"method\":\"ngram_gpu\",\"num_speculative_tokens\":${SPEC_TOKENS:-8},\"prompt_lookup_min\":3,\"prompt_lookup_max\":8}"
  # Speculation allocates on top of a KV pool already sized to fill GPU_UTIL;
  # at 0.90 capture OOMs the engine, so back off unless the caller was explicit.
  GPU_UTIL=${GPU_UTIL:-0.85}
elif [ -n "$SPEC" ]; then
  echo "unknown SPEC=$SPEC (supported: ngram)" >&2; exit 1
fi

GPU_UTIL=${GPU_UTIL:-0.90}   # after SPEC, which may lower it

# Tool / function calling. gemma4 has its own call format with dedicated
# special tokens, so BOTH the reasoning parser and the tool parser must be
# named gemma4, and the chat template must be vLLM's gemma4 tool template
# (vendored at the repo root) — vLLM's built-in gemma4 chat template does not
# carry the tool protocol. TOOLS=0 turns it off entirely.
TOOL_ARGS=$([ "${TOOLS:-1}" = 1 ] && echo --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template "$REPO/examples_tool_chat_template_gemma4.jinja")

# Multimodal. The model has image + audio towers (0.9 GB combined), and both are
# ON by default here (VISION=1, AUDIO=1) — this box serves multimodal clients.
# Cost, if you don't need them: the encoders' profiling peak is charged against
# the KV pool, and the towers' weights stay resident either way (vLLM 0.27.1 has
# no --language-model-only for Gemma4ForConditionalGeneration).
#   VISION=1  accept 1 image per prompt (capped at 2048 image tokens)
#   AUDIO=1   accept 1 audio clip per prompt (up to ~30 s of speech, per the
#             model card; clients send it as an audio_url content part)
# The two are independent: set either to 0 for a leaner profile.
if [ "${VISION:-1}" = 1 ] || [ "${AUDIO:-1}" = 1 ]; then
  IMG=0; [ "${VISION:-1}" = 1 ] && IMG='{"count":1}'
  AUD=0; [ "${AUDIO:-1}" = 1 ] && AUD=1
  MM_ARGS="--limit-mm-per-prompt {\"image\":$IMG,\"audio\":$AUD}"
  [ "${VISION:-1}" = 1 ] && MM_ARGS="$MM_ARGS --mm-processor-kwargs {\"size\":{\"shortest_edge\":65536,\"longest_edge\":2097152}}"
else
  MM_ARGS='--limit-mm-per-prompt {"image":0,"audio":0}'
fi

export PATH="$REPO/venv/bin:$PATH"
# The reference box needs this on bare metal (transient prefill workspaces
# fragment the allocator); harmless elsewhere.
if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
# int8 activations: "off" for these env vars is UNSET, not empty — vLLM's
# env_with_choices rejects the empty string. Export only when non-empty.
[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INT8_INCLUDE_RE=${INT8_LAYERS:-mlp}

# CUDA graphs. THE single biggest decode win on this box, and it was switched
# off for the wrong reason: capture used to die with a CUDA OOM, which an
# earlier revision of this repo misread as a torch-2.13 stable-ABI bug and
# "fixed" by setting cudagraph_mode=NONE. The real story: vLLM sizes the KV
# pool to fill GPU_UTIL, THEN captures graphs, and the capture list it derives
# from --max-num-seqs 256 (51 sizes, up to 512) needs more VRAM than the few
# hundred MB left over -> OOM during capture -> engine dies. Bounding the list
# fixes it outright; nothing about torch was ever broken.
#
# Measured on the RTX 3090 Ti (cohort: 8 real prompts, 1024 out, GPTQ int4):
#   cudagraph NONE          C1  24.2  C4  89.2  C8 175.5 tok/s   (the old default)
#   PIECEWISE               C1  62.1  C4 235.9  C8 456.7 tok/s
#   FULL_AND_PIECEWISE      C1  92.5  C4 340.5  C8 657.7 tok/s   <- default now
# i.e. ~3.8x single-stream and ~3.7x aggregate purely from turning graphs back
# on. Decode TPOT drops from ~40 ms to ~11 ms.
#
# FULL_AND_PIECEWISE: full graphs for pure-decode steps (the hot path),
# piecewise for mixed prefill+decode steps. Capture cost here: ~0.56 GiB, 2 s.
#   CUDAGRAPH=PIECEWISE  leaner (0.11 GiB) but ~33% slower decode
#   CUDAGRAPH=NONE       last-resort escape hatch if capture ever misbehaves
CUDAGRAPH=${CUDAGRAPH:-FULL_AND_PIECEWISE}
if [ "$CUDAGRAPH" = "NONE" ]; then
  CG_CFG='"cudagraph_mode":"NONE"'
else
  # Capture sizes: powers/steps up to MAX_SEQS only. Keeping the list short is
  # what keeps capture inside the VRAM left after the KV pool is carved out.
  CG_SIZES=$(python3 -c "
m=$MAX_SEQS
c=[s for s in (1,2,4,8,16,24,32,48,64,96,128,192,256) if s<m]
print(','.join(str(s) for s in c+[m]))")
  CG_CFG="\"cudagraph_mode\":\"$CUDAGRAPH\",\"cudagraph_capture_sizes\":[$CG_SIZES]"
fi

# API key: api_key.txt in the repo root, or VLLM_API_KEY in the environment/.env.
if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

# Guard: this card is often shared with the Qwen serving container (23 GB).
# Two 24 GB servers do not fit; fail fast with a clear message instead of
# vLLM's free-memory error after a long model download/load.
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${FREE}" -lt 12000 ]; then
    echo "WARNING: only ${FREE} MiB free on the GPU — another process (the Qwen"
    echo "container on port 18020?) is holding it. Stop it before starting this"
    echo "server: the two models do not fit on one 24 GB card."
  fi
fi

LOG=$REPO/gemma.log
echo "serving $SERVED_NAME from $MODEL on :$PORT (log: $LOG)"
exec venv/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --async-scheduling \
  --max-num-batched-tokens $BATCHED_TOKENS \
  --kv-cache-dtype fp8 \
  --dtype bfloat16 \
  --compilation-config "{\"mode\":3,$CG_CFG,\"custom_ops\":[\"+rms_norm\"]}" \
  ${PREFIX_ARGS} \
  ${SPEC_ARGS} \
  ${TOOL_ARGS} \
  ${MM_ARGS} \
  ${EXTRA_ARGS} 2>&1 | tee "$LOG"
