#!/usr/bin/env bash
# Manage the tuned Ollama instance that mirrors the vLLM server's knobs (see
# the comparison table in README.md). Values overridable via environment:
#   NUM_PARALLEL=8 KV_CACHE_TYPE=f16 bash batch/start_ollama.sh
#
# Requires the quantized model to exist already:
#   ollama pull gemma4:e4b   (or ollama create from a local GGUF)
set -eu

PID_FILE=".ollama.pid"
LOG="${LOG:-ollama_serve.log}"

# --- tuned environment (matches README's "best effort" side of the table) ---
export OLLAMA_NUM_PARALLEL="${NUM_PARALLEL:-16}"      # ~ vLLM --max-num-seqs
export OLLAMA_MAX_LOADED_MODELS=1                     # one model owns the VRAM
export OLLAMA_FLASH_ATTENTION=1                       # ~ vLLM's forced FlashInfer/FA2
export OLLAMA_KV_CACHE_TYPE="${KV_CACHE_TYPE:-q8_0}"  # ~ vLLM --kv-cache-dtype fp8
                                                      # KV_CACHE_TYPE=f16 if multimodal outputs look off (experimental per Ollama docs)
export OLLAMA_CONTEXT_LENGTH="${CTX_LEN:-32768}"      # unset -> tier-dependent; be explicit
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

# The default distro systemd unit may also be running — stop it too, only one
# Ollama can hold the model in VRAM.
stop_ollama() {
  if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "stopping systemd ollama unit..."
    systemctl stop ollama 2>/dev/null || true
  fi
  if pgrep -x ollama >/dev/null 2>&1; then
    echo "stopping existing ollama..."
    pkill -x ollama || true
    sleep 3
  fi
  rm -f "$PID_FILE"
}

wait_ready() {
  # /api/tags answers as soon as the HTTP API is up (model load happens lazily
  # on first request and is excluded from benchmarks by design).
  for _ in $(seq 1 30); do
    curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

case "${1:-start}" in
start)
  pgrep -x ollama >/dev/null 2>&1 && stop_ollama
  echo "starting ollama serve ($OLLAMA_HOST, num_parallel=$OLLAMA_NUM_PARALLEL, kv=$OLLAMA_KV_CACHE_TYPE)"
  nohup ollama serve > "$LOG" 2>&1 &
  echo $! > "$PID_FILE"
  if wait_ready; then
    echo "ollama API is up (pid $(cat "$PID_FILE")); model loads on first request"
    echo "bench with: python bench/ollama_bench.py --model gemma4:e4b"
  else
    echo "ERROR: ollama did not come up within 30s — check $LOG" >&2
    exit 1
  fi
;;
stop)
  stop_ollama
  echo "ollama stopped"
;;
restart)
  stop_ollama
  exec "$0" start
;;
status)
  if wait_ready; then
    echo "ollama is UP ($OLLAMA_HOST, pid $(cat "$PID_FILE" 2>/dev/null || echo '?'))"
  else
    echo "ollama is DOWN"
    exit 1
  fi
;;
*)
  echo "usage: $0 {start|stop|restart|status}" >&2
  exit 1
;;
esac
