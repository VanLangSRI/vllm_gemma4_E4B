#!/bin/bash
# Reproduce the README numbers against the server you have running on this
# box (batch/start_gemma.sh, port 18030 by default). Prints one ROW line per
# measurement; ~15 min for the batch profile.
#
#   bash bench/run_benchmarks.sh              # batch profile: 128/512 @ C64 and
#                                             # @ C256, 256/256 @ C256, cohorts C1-C8
#   bash bench/run_benchmarks.sh --prefill    # + prefill matrix (1k..32k)
#   bash bench/run_benchmarks.sh --long       # + long-context rows (need a
#                                             # matching --max-model-len)
#
# Run it twice after a restart and keep the second numbers: the first run
# after start includes JIT warmup and reads low. Then run
# bench/quality_battery.py — a fast server that emits garbage is worth nothing.
#
# The cohorts use the 8 realistic prompts in bench/prompts_real.jsonl (same
# file as the qwen38-27b-rtx3090 reference repo, Apache-2.0) so results can be
# compared against that repo's tables where both models were measured the same
# way.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
DO_PREFILL=0; DO_LONG=0
for a in "$@"; do case $a in --prefill) DO_PREFILL=1;; --long) DO_LONG=1;; esac; done
export PATH="$REPO/venv/bin:$PATH"
export OPENAI_API_KEY=${VLLM_API_KEY:-$(cat "$REPO/api_key.txt" 2>/dev/null)}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-18030}
MODEL=${MODEL:-$REPO/models/gemma-4-E4B-it-qat-w4a16-ct}
B="venv/bin/vllm bench serve --host $HOST --port $PORT --model $MODEL --served-model-name gemma-4-e4b"
OUT=${OUT:-$HERE/results}; mkdir -p "$OUT"

curl -sf -o /dev/null http://$HOST:$PORT/health || { echo "no server on $HOST:$PORT"; exit 1; }
num() { awk "/$1/ {print \$$2}" "$3"; }
row() { # label logfile conc
  local L=$1 F=$2 C=$3
  local E2E=$(num "Output token throughput" 5 $F) MTPOT=$(num "Median TPOT" 4 $F) TTFT=$(num "Mean TTFT" 4 $F) DUR=$(num "Benchmark duration" 4 $F)
  local DEC=$(python3 -c "print(f'{$C*1000/$MTPOT:.0f}')" 2>/dev/null)
  echo "ROW $L | e2e=$E2E tok/s | decode(C/medTPOT)=$DEC | medTPOT=$MTPOT ms | meanTTFT=$TTFT ms | dur=${DUR}s"
}

echo "# $(date) server=$HOST:$PORT model=$(basename $MODEL)"
# Warmup prompt: loads weights, triggers torch.compile/CUDA-graph capture and
# FlashInfer JIT. Deliberately discarded — none of its time appears in a ROW.
$B --dataset-name random --random-input-len 256 --random-output-len 256 --num-prompts 16 --max-concurrency 8 > /dev/null 2>&1

# Random-workload saturation rows. C64 mirrors the reference repo's protocol
# (comparable across the two projects); C256 uses this card's larger KV pool.
for W in "128 512" "256 256"; do set -- $W
  $B --dataset-name random --ignore-eos --random-input-len $1 --random-output-len $2 --num-prompts 256 --max-concurrency 64 > $OUT/batch_${1}_${2}_c64.log 2>&1
  row "64conc $1in/$2out" $OUT/batch_${1}_${2}_c64.log 64
done
for W in "128 512" "256 256"; do set -- $W
  $B --dataset-name random --ignore-eos --random-input-len $1 --random-output-len $2 --num-prompts 512 --max-concurrency 256 > $OUT/batch_${1}_${2}_c256.log 2>&1
  row "256conc $1in/$2out" $OUT/batch_${1}_${2}_c256.log 256
done

# Real-prompt cohorts: 1,024-token answers, model-default sampling.
# C16 is intentionally absent: prompts_real.jsonl holds 8 prompts and this loop
# sends 8 requests, so a 16-wide pool cannot be filled -- it only re-measures C8.
for C in 1 2 4 8; do
  $B --dataset-name custom --dataset-path $HERE/prompts_real.jsonl --custom-output-len 1024 --num-prompts 8 --max-concurrency $C > $OUT/cohort_c$C.log 2>&1
  F=$OUT/cohort_c$C.log
  echo "ROW cohort C$C real prompts | e2e=$(num "Output token throughput" 5 $F) tok/s | decode(C/meanTPOT)=$(python3 -c "print(f'{$C*1000/$(num "Mean TPOT" 4 $F):.1f}')") | meanTTFT=$(num "Mean TTFT" 4 $F) ms"
done

if [ $DO_PREFILL = 1 ]; then
  pf() { LEN=$1; C=$2; N=$3
    $B --dataset-name random --random-output-len 1 --random-input-len $LEN --num-prompts $N --max-concurrency $C > $OUT/prefill_${LEN}_c$C.log 2>&1
    IN=$(num "Total input tokens" 4 $OUT/prefill_${LEN}_c$C.log); DUR=$(num "Benchmark duration" 4 $OUT/prefill_${LEN}_c$C.log)
    echo "ROW prefill len=$LEN conc=$C | $(python3 -c "print(f'{$IN/$DUR:.0f}')") tok/s | meanTTFT=$(num "Mean TTFT" 4 $OUT/prefill_${LEN}_c$C.log) ms"; }
  pf 1024 1 16;  pf 1024 4 32;   pf 1024 16 64
  pf 4096 1 8;   pf 4096 4 16;   pf 4096 16 32
  pf 16384 1 4;  pf 16384 4 8;   pf 16384 8 16
  pf 32768 1 2;  pf 32768 2 4
fi
if [ $DO_LONG = 1 ]; then
  # Only meaningful with --max-model-len >= 32768 (default) or 131072.
  $B --dataset-name random --ignore-eos --random-input-len 32768 --random-output-len 256 --num-prompts 1 --max-concurrency 1 > $OUT/long_32k.log 2>&1
  echo "ROW 1x32k/256 | meanTTFT=$(num "Mean TTFT" 4 $OUT/long_32k.log) ms | TPOT=$(num "Mean TPOT" 4 $OUT/long_32k.log) ms"
  $B --dataset-name random --ignore-eos --random-input-len 16384 --random-output-len 1024 --num-prompts 4 --max-concurrency 4 > $OUT/long_4x16k.log 2>&1
  echo "ROW 4x16k/1024 conc4 | e2e=$(num "Output token throughput" 5 $OUT/long_4x16k.log) tok/s | medITL=$(num "Median ITL" 4 $OUT/long_4x16k.log) ms | dur=$(num "Benchmark duration" 4 $OUT/long_4x16k.log)s"
fi
echo "# raw logs in $OUT"
