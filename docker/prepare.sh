#!/bin/bash
# Container-side prepare: download the Gemma 4 E4B checkpoint into the
# /app/models volume. CPU-only, safe to re-run.
set -eu
cd /app
export HF_HUB_ENABLE_HF_TRANSFER=1

venv/bin/hf download google/gemma-4-E4B-it-qat-w4a16-ct \
  --local-dir /app/models/gemma-4-E4B-it-qat-w4a16-ct
