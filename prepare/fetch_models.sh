#!/bin/bash
# Download the Gemma 4 E4B checkpoint this repo serves (hf_transfer for speed;
# ~11 GB). Idempotent — safe to re-run.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
export HF_HUB_ENABLE_HF_TRANSFER=1
HF=${HF:-$REPO/venv/bin/hf}

echo "Google QAT w4a16 compressed-tensors (~11 GB)"
"$HF" download google/gemma-4-E4B-it-qat-w4a16-ct \
  --local-dir "$REPO/models/gemma-4-E4B-it-qat-w4a16-ct"

echo "done. Model dirs:"
du -sh "$REPO"/models/* 2>/dev/null
