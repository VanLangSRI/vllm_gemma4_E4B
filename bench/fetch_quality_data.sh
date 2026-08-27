#!/bin/bash
# Fetch the quality-battery datasets once (~100 MB total).
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
HF=${HF:-$REPO/venv/bin/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $HERE/quality-data

"$HF" download Salesforce/wikitext --repo-type dataset \
  --include "wikitext-2-raw-v1/test-*" --local-dir $HERE/quality-data/wikitext
"$HF" download openai/gsm8k --repo-type dataset \
  --include "main/test-*" --local-dir $HERE/quality-data/gsm8k
