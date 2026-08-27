#!/bin/bash
# Regenerate every number in report.md.
#
#   bash bench/reproduce_report.sh all        # everything (~5-6 hours)
#   bash bench/reproduce_report.sh matrix     # the 8-config factorial (~3.5 h)
#   bash bench/reproduce_report.sh ollama     # Ollama text cohort (~10 min)
#   bash bench/reproduce_report.sh mm         # multimodal, both engines (~45 min)
#   bash bench/reproduce_report.sh config C   # one config of the factorial
#   bash bench/reproduce_report.sh --list     # show the config table and exit
#
# Results are APPENDED to bench/results-<date>.txt; per-run stdout+stderr is
# kept under bench/repro-logs/. Nothing is overwritten, so an interrupted run
# can be resumed one phase at a time.
#
# Requirements: bash setup.sh has been run, both models present, GPU idle.
# The GPU holds ONE engine at a time -- the script stops one before starting
# the other. Expect the whole thing to take an afternoon.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"

PORT=${PORT:-18030}
MODEL_DIR=${MODEL_DIR:-models/gemma-4-E4B-it-qat-w4a16-ct}
SERVED=${SERVED:-gemma-4-e4b}
OLLAMA_MODEL=${OLLAMA_MODEL:-gemma4:e4b}
PY="$REPO/venv/bin/python"
VLLM="$REPO/venv/bin/vllm"

OUT="$REPO/bench/results-$(date +%F).txt"
LOGS="$REPO/bench/repro-logs"
mkdir -p "$LOGS"

BENCH="$VLLM bench serve --host 127.0.0.1 --port $PORT --model $MODEL_DIR --served-model-name $SERVED"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
emit() { echo "$*" >> "$OUT"; }

# ---------------------------------------------------------------- config table
# label : CUDAGRAPH SPEC PREFIX_CACHE
# G and H are EXPECTED TO CRASH at high concurrency -- see report.md. They are
# included because that crash is itself a documented result.
config_env() {
  case "$1" in
    A) echo "CUDAGRAPH=FULL_AND_PIECEWISE SPEC= PREFIX_CACHE=1" ;;
    B) echo "CUDAGRAPH=FULL_AND_PIECEWISE SPEC= PREFIX_CACHE=0" ;;
    C) echo "CUDAGRAPH=FULL_AND_PIECEWISE SPEC=ngram PREFIX_CACHE=1" ;;
    D) echo "CUDAGRAPH=FULL_AND_PIECEWISE SPEC=ngram PREFIX_CACHE=0" ;;
    E) echo "CUDAGRAPH=NONE SPEC= PREFIX_CACHE=1" ;;
    F) echo "CUDAGRAPH=NONE SPEC= PREFIX_CACHE=0" ;;
    G) echo "CUDAGRAPH=NONE SPEC=ngram PREFIX_CACHE=1" ;;
    H) echo "CUDAGRAPH=NONE SPEC=ngram PREFIX_CACHE=0" ;;
    *) return 1 ;;
  esac
}

list_configs() {
  cat <<'TXT'
  label  cudagraph            SPEC    PREFIX_CACHE   note
  A      FULL_AND_PIECEWISE   off     1              shipped default
  B      FULL_AND_PIECEWISE   off     0
  C      FULL_AND_PIECEWISE   ngram   1
  D      FULL_AND_PIECEWISE   ngram   0
  E      NONE                 off     1
  F      NONE                 off     0
  G      NONE                 ngram   1              expected to CRASH at C64
  H      NONE                 ngram   0              expected to CRASH at C64
TXT
}

# ------------------------------------------------------------------- utilities
stop_engines() {
  pkill -f "vllm serve"   >/dev/null 2>&1
  bash batch/start_ollama.sh stop >/dev/null 2>&1
  sleep 6
}

# Wait for the server, or give up. Never a blind sleep: boots range 130-380 s
# depending on torch.compile cache state and whether speculation is enabled.
wait_health() {
  local limit=${1:-600} i
  for ((i=1; i<=limit; i++)); do
    curl -sf -m2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "$i"; return 0; }
    grep -q "Engine core initialization failed" "$REPO/gemma.log" 2>/dev/null && return 1
    sleep 1
  done
  return 1
}

alive() { curl -sf -m3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

# Record what the engine ACTUALLY did with the knobs. Env vars can be silently
# ignored; this is the proof that a config was what it claimed to be.
emit_engine_facts() {
  grep -oE "enable_prefix_caching=[A-Za-z]+|GPU KV cache size: [0-9,]+ tokens|cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|num_speculative_tokens': [0-9]+|gpu_memory_utilization': [0-9.]+" \
    "$REPO/gemma.log" 2>/dev/null | sort -u | sed 's/^/# engine: /' >> "$OUT"
}

# Run a bench, keep the FULL output, then extract. Never `cmd | grep` directly:
# doing that once hid an engine crash for two whole benchmark rounds.
run_bench() {
  local name="$1" logfile="$LOGS/$2"; shift 2
  "$@" > "$logfile" 2>&1
  if grep -q '^ROW' "$logfile"; then
    grep '^ROW' "$logfile" >> "$OUT"
  else
    emit "$name produced no ROW -- see ${logfile#$REPO/}"
    if grep -qi "connection refused" "$logfile"; then
      emit "  cause: server was gone (engine crash) -- see gemma.log"
    fi
  fi
}

# ------------------------------------------------------- one factorial config
run_config() {
  local L="$1" envs boot
  envs="$(config_env "$L")" || { echo "unknown config '$L'"; return 1; }

  say "config $L : $envs"
  stop_engines
  emit ""
  emit "########## CONFIG $L : $envs ##########"
  emit "# started $(date -Is)"

  # shellcheck disable=SC2086
  env $envs bash batch/start_gemma.sh > "$LOGS/serve_$L.log" 2>&1 &
  if ! boot=$(wait_health 600); then
    emit "BOOT FAILED -- see ${LOGS#$REPO/}/serve_$L.log"
    stop_engines
    return 1
  fi
  emit "# booted in ${boot}s"
  emit_engine_facts

  # untimed warmup: keeps JIT and first-touch allocation out of every number
  $BENCH --dataset-name random --random-input-len 256 --random-output-len 128 \
         --num-prompts 8 --max-concurrency 4 >/dev/null 2>&1

  # 1. real-prompt cohort. `vllm bench serve` prints throughput, then Mean TTFT,
  #    then Mean TPOT -- in that order. Label accordingly.
  # Match the concurrency levels bench/ollama_bench.py uses so the
  # vLLM-vs-Ollama table has no holes. An earlier driver ran only 1/4/8 and the
  # report shipped with an empty C2 row.
  # C16 is deliberately absent: prompts_real.jsonl holds 8 prompts and the
  # cohort sends exactly 8 requests, so a 16-wide pool cannot be filled and
  # "C16" just re-measures C8 (655.2 vs 652.9 when both were run).
  local C R
  for C in ${COHORT_CONC:-1 2 4 8}; do
    R=$($BENCH --dataset-name custom --dataset-path bench/prompts_real.jsonl \
               --custom-output-len 1024 --num-prompts 8 --max-concurrency "$C" 2>&1 \
        | grep -E "Output token throughput|Mean TTFT|Mean TPOT" \
        | awk '{print $NF}' | paste -sd' ')
    emit "COHORT C$C tok/s,TTFTms,TPOTms = ${R:-FAILED}"
  done

  # 2. saturation. This is where G and H die.
  R=$($BENCH --dataset-name random --ignore-eos --random-input-len 128 \
             --random-output-len 512 --num-prompts 256 --max-concurrency 64 2>&1 \
      | grep -E "Output token throughput|Median TPOT" | awk '{print $NF}' | paste -sd' ')
  emit "SATURATION 128/512 C64 tok/s,medTPOTms = ${R:-FAILED}"

  if ! alive; then
    emit "*** ENGINE DIED during the saturation run ***"
    grep -oE "OutOfMemoryError: CUDA out of memory[^.]*\.|EngineDeadError" \
      "$REPO/gemma.log" 2>/dev/null | sort -u | head -2 | sed 's/^/#   /' >> "$OUT"
    emit "#   (expected for configs G and H -- see report.md)"
    stop_engines
    return 0
  fi

  # 3. isolate each knob's effect
  run_bench SPEC_BENCH   "spec_$L.txt"   "$PY" bench/spec_bench.py   --label "$L" --conc 1 4
  run_bench PREFIX_BENCH "prefix_$L.txt" "$PY" bench/prefix_bench.py --label "$L" --conc 8 --rounds 3

  # 4. speculation acceptance -- the number that explains why SPEC helps or hurts
  if [[ "$envs" == *"SPEC=ngram"* ]]; then
    "$PY" - "$REPO/gemma.log" >> "$OUT" <<'PYEOF'
import re, sys
t = open(sys.argv[1], errors="replace").read()
a = sum(int(x) for x in re.findall(r"Accepted: (\d+) tokens", t))
d = sum(int(x) for x in re.findall(r"Drafted: (\d+) tokens", t))
if d:
    print(f"SPEC_ACCEPTANCE accepted={a} drafted={d} rate={100*a/d:.1f}% wasted={100*(1-a/d):.1f}%")
PYEOF
  fi

  emit "# finished $(date -Is)"
  stop_engines
}

# ------------------------------------------------------------------ the phases
phase_matrix() {
  emit ""
  emit "############################################################"
  emit "### FACTORIAL  CUDAGRAPH x SPEC x PREFIX_CACHE   $(date -Is)"
  emit "############################################################"
  local L
  for L in "$@"; do run_config "$L"; done
}

phase_ollama() {
  say "Ollama text cohort"
  stop_engines
  emit ""
  emit "########## OLLAMA REFERENCE  $(date -Is) ##########"
  bash batch/start_ollama.sh start >/dev/null 2>&1
  sleep 6
  run_bench OLLAMA_COHORT "ollama_cohort.txt" \
    "$PY" bench/ollama_bench.py --model "$OLLAMA_MODEL" --conc 1 2 4 8
  bash batch/start_ollama.sh stop >/dev/null 2>&1
}

phase_mm() {
  say "multimodal (real assets, both engines)"
  bash bench/fetch_mm_assets.sh >/dev/null 2>&1 || \
    emit "# WARNING: asset fetch failed; mm_bench will fall back to synthetic filler"

  stop_engines
  emit ""
  emit "########## MULTIMODAL  $(date -Is) ##########"
  local boot m
  bash batch/start_gemma.sh > "$LOGS/serve_mm.log" 2>&1 &
  if ! boot=$(wait_health 600); then
    emit "# vLLM BOOT FAILED"; stop_engines; return 1
  fi
  emit "# vLLM booted in ${boot}s"

  # Comprehension gate FIRST. Throughput cannot distinguish a working media
  # pipeline from one that silently drops the attachment.
  emit "--- comprehension gate (bench/mm_check.py) ---"
  "$PY" bench/mm_check.py > "$LOGS/mm_check.txt" 2>&1
  grep -E "^  (PASS|FAIL)|^comprehension" "$LOGS/mm_check.txt" >> "$OUT"

  for m in image audio both; do
    run_bench "MM_VLLM_$m" "mm_vllm_$m.txt" \
      "$PY" bench/mm_bench.py --api vllm --modality "$m" --conc 1 2 4 8
  done
  stop_engines

  bash batch/start_ollama.sh start >/dev/null 2>&1
  sleep 6
  for m in image audio both; do
    run_bench "MM_OLLAMA_$m" "mm_ollama_$m.txt" \
      "$PY" bench/mm_bench.py --api ollama --modality "$m" --conc 1 2 4 8
  done
  bash batch/start_ollama.sh stop >/dev/null 2>&1
}

# ------------------------------------------------------------- preflight + main
preflight() {
  local fail=0
  [ -x "$PY" ]   || { echo "missing $PY -- run: bash setup.sh"; fail=1; }
  [ -d "$MODEL_DIR" ] || { echo "missing $MODEL_DIR -- run: bash prepare/fetch_models.sh"; fail=1; }
  [ -f bench/prompts_real.jsonl ] || { echo "missing bench/prompts_real.jsonl"; fail=1; }
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; fail=1; }
  [ "$fail" = 0 ] || exit 1

  emit "# =========================================================================="
  emit "# REPRODUCTION RUN  $(date -Is)"
  emit "# GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  emit "# power: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader) (default $(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader))"
  emit "# vllm $("$PY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null) | torch $("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
  emit "# model: $MODEL_DIR"
  emit "# =========================================================================="

  # The report's power discussion assumes the stock 450 W limit. Warn rather
  # than fail: the numbers are still valid, they just are not comparable.
  local pl
  pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
  if [ "${pl:-0}" -lt 400 ]; then
    echo "WARNING: power limit is ${pl} W; report.md was produced at 450 W."
    echo "         raise it with: sudo nvidia-smi -pl 450   (needs root)"
    emit "# WARNING: run made at ${pl} W, not the 450 W used for report.md"
  fi
}

usage() {
  # print the header comment block, stopping at the first non-comment line
  sed -n '2,${/^[^#]/q;p;}' "$0" | sed 's/^# \{0,1\}//'
  echo "configs:"
  list_configs
}

main() {
  case "${1:-}" in
    -h|--help|"") usage; exit 0 ;;
    --list)       list_configs; exit 0 ;;
    all|matrix|arm1|arm2|ollama|mm|config) ;;
    *) echo "unknown phase '$1'"; usage; exit 1 ;;
  esac

  # Validate config labels BEFORE preflight so a typo does not write a header
  # block (and a "finished" line) into the results file for a run that never was.
  if [ "$1" = config ]; then
    [ $# -ge 2 ] || { echo "usage: $0 config <A..H>"; exit 1; }
    local c
    for c in "${@:2}"; do
      config_env "$c" >/dev/null || { echo "unknown config '$c'"; list_configs; exit 1; }
    done
  fi

  preflight
  local t0=$SECONDS label="$1"

  case "$1" in
    all)     phase_matrix A B C D E F G H; phase_ollama; phase_mm ;;
    matrix)  phase_matrix A B C D E F G H ;;
    arm1)    phase_matrix A B C D ;;
    arm2)    phase_matrix E F G H ;;
    ollama)  phase_ollama ;;
    mm)      phase_mm ;;
    config)  label="config ${*:2}"; shift; phase_matrix "$@" ;;
  esac

  emit "# phase '$label' finished in $(( (SECONDS-t0)/60 )) min"

  # Remind the operator that report.md's tables are generated, not typed.
  if [ -f "$REPO/bench/measurements.json" ]; then
    echo
    echo "Next: copy the new numbers into bench/measurements.json, then"
    echo "  $PY bench/make_report.py --check    # validate consistency"
    echo "  $PY bench/make_report.py            # emit the report tables"
  fi
  say "done in $(( (SECONDS-t0)/60 )) min -- results appended to ${OUT#$REPO/}"
  echo "   per-run logs: ${LOGS#$REPO/}/"
}

main "$@"
