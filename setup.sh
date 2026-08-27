#!/bin/bash
# One-time environment setup for this repo (mirrors the reference repo's Setup):
#   1. python3 -m venv venv (3.12)
#   2. pip install vllm==0.27.1  (+ huggingface_hub, hf_transfer, ninja)
#   3. download both model variants (prepare/fetch_models.sh)
#   4. apply the Marlin int8-activation patches to the venv's vLLM (optional,
#      only needed for INT8_ACT=int8)
#   5. bash verify.sh --no-server
#
# Everything except step 4 works on a stock PyPI vLLM 0.27.1 wheel; step 4 is
# the only divergence and it is written against exactly that version.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ ! -x venv/bin/python ]; then
  echo "creating venv (python 3.12 required; python3 here: $(python3 --version))"
  python3 -m venv venv
fi

echo "installing vllm 0.27.1 (large download: torch + CUDA libs)"
venv/bin/pip install --upgrade pip -q
venv/bin/pip install vllm==0.27.1 huggingface_hub hf_transfer ninja
venv/bin/pip install av soundfile librosa pyarrow  # audio decoding (vllm[audio]) + quality battery
# soundfile is what vllm/multimodal/audio.py actually imports; without it every
# audio request fails with "Please install vllm[audio] for audio support" and the
# server silently answers as if no audio were attached.

echo "downloading model (QAT w4a16, ~11 GB)"
bash prepare/fetch_models.sh

VSP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
if [ -d "$VSP" ]; then
  echo "applying Marlin int8-activation patches (optional, for INT8_ACT=int8)"
  applied=0
  for p in patches/marlin-int8-layer-select.patch patches/marlin-int8-negative-scales.patch; do
    if venv/bin/python patches/_check_applied.py "$p" "$VSP" 2>/dev/null; then
      echo "  $(basename "$p"): already applied"
    elif patch -p1 -N --dry-run -s -d "$VSP" < "$p" >/dev/null 2>&1; then
      patch -p1 -d "$VSP" < "$p" >/dev/null && echo "  $(basename "$p"): applied"
      applied=1
    else
      echo "  $(basename "$p"): NOT applicable (vLLM version mismatch?) — INT8_ACT will be unavailable"
    fi
  done
else
  echo "vllm not importable — skipping patch step"
fi

# Required patches: transformers-heterogeneity workarounds (see README) and
# the flashinfer sm86 large-head fp8-KV arch unlock for RTX 3090 Ti.
FSP=$(venv/bin/python -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))' 2>/dev/null)
for spec in   "patches/vllm-heterogeneous-config-global-access.patch|$VSP"   "patches/vllm-gemma4-per-layer-head-dim.patch|$VSP"   "patches/flashinfer-fa2-sm86-fp8-kv.patch|$FSP"; do
  p="${spec%%|*}"; d="${spec##*|}"
  [ -d "$d" ] || { echo "  $(basename "$p"): target missing, skipped"; continue; }
  if venv/bin/python patches/_check_applied.py "$p" "$d" 2>/dev/null; then
    echo "  $(basename "$p"): already applied"
  elif patch -p1 -N --dry-run -s -d "$d" < "$p" >/dev/null 2>&1; then
    patch -p1 -d "$d" < "$p" >/dev/null && echo "  $(basename "$p"): applied"
  else
    echo "  $(basename "$p"): NOT applicable — audio/long-prefill may fail; see README known issues"
  fi
done

bash verify.sh --no-server
