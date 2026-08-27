#!/bin/bash
# Check that this repo is set up to serve Gemma 4 E4B the way it is meant to be:
# venv + vLLM version, optional Marlin int8 patches, both model variants present
# and complete (shards, tokenizer, chat template), and — if a server is running —
# that it answers and which attention backend / KV pool it came up with.
#
#   bash verify.sh            # everything
#   bash verify.sh --no-server
#   bash verify.sh --install  # only the install (venv, vLLM, patches): no GPU,
#                             # model or server checks — what the Docker build runs
# Exit code: 0 all PASS (WARNs allowed), 1 if anything FAILs.
# PY=/path/to/python overrides the interpreter (default: this repo's venv).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
NOSRV=0; INSTALL=0
for a in "$@"; do case "$a" in --no-server) NOSRV=1;; --install) INSTALL=1; NOSRV=1;; esac; done
FAILS=0
ok()   { printf "  PASS  %s\n" "$1"; }
warn() { printf "  WARN  %s\n" "$1"; }
fail() { printf "  FAIL  %s\n" "$1"; FAILS=$((FAILS+1)); }
MODEL_DIR=${MODEL_DIR:-$HERE/models/gemma-4-E4B-it-qat-w4a16-ct}
PY=${PY:-$HERE/venv/bin/python}

echo "== environment"
[ -x "$PY" ] && ok "python: $PY" || { fail "no $PY (see README Setup)"; exit 1; }
VER=$($PY -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -n1)
[ "$VER" = "0.27.1" ] && ok "vllm $VER" || warn "vllm ${VER:-missing} (patches were written against 0.27.1)"
SP=$($PY -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] && ok "vllm package at $SP" || { fail "cannot import vllm with $PY"; exit 1; }
$PY -c "from vllm.model_executor.models import gemma4_mm, gemma4" 2>/dev/null \
  && ok "gemma4 model classes importable (Gemma4ForConditionalGeneration)" \
  || fail "vllm has no gemma4 model support (need vLLM >= 0.19.1, this is 0.27.1)"
$PY -c "from vllm.parser.gemma4 import Gemma4Parser" 2>/dev/null \
  && ok "Gemma4Parser available (reasoning-parser/tool-call-parser gemma4)" \
  || warn "Gemma4Parser missing — tool calling would need TOOLS=0"
if [ $INSTALL = 0 ]; then
$PY - <<'EOF' 2>/dev/null || fail "torch cannot see a CUDA GPU"
import torch; assert torch.cuda.is_available()
p=torch.cuda.get_device_properties(0)
print(f"  PASS  GPU: {p.name}, {p.total_memory/2**30:.1f} GiB, sm{p.major}{p.minor}, torch {torch.__version__}")
assert p.major == 8 and p.minor >= 6, f"  FAIL  expected Ampere sm8x, got sm{p.major}{p.minor}"
EOF
command -v nvidia-smi >/dev/null && { PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1); ok "power limit ${PL} W"; }
fi
for t in triton flashinfer compressed_tensors; do $PY -c "import $t" 2>/dev/null && ok "python module $t" || fail "python module $t missing"; done

echo "== required patches (heterogeneity workarounds + flashinfer sm86)"
FSP=$(venv/bin/python -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))' 2>/dev/null)
for spec in \
  "patches/vllm-heterogeneous-config-global-access.patch|$SP" \
  "patches/vllm-gemma4-per-layer-head-dim.patch|$SP" \
  "patches/flashinfer-fa2-sm86-fp8-kv.patch|$FSP"; do
  p="${spec%%|*}"; d="${spec##*|}"
  if [ -n "$d" ] && patch -p1 -R --dry-run -s -d "$d" < "$p" >/dev/null 2>&1; then
    ok "$(basename $p) applied"
  else
    warn "$(basename $p) NOT applied (audio requests / full-attention prefill will fail)"
  fi
done
(venv/bin/python -c "import av" 2>/dev/null) && ok "av (audio decode) installed" || warn "av missing — audio requests return 500"
# soundfile is the loader vLLM tries FIRST for audio. Missing it does not 500 the
# request: vLLM logs an error, falls through, and can end up answering with the
# audio silently dropped — a much nastier failure than a hard error.
(venv/bin/python -c "import soundfile" 2>/dev/null) && ok "soundfile (audio decode) installed" || fail "soundfile missing — audio is silently DROPPED from requests (pip install soundfile librosa)"
(venv/bin/python -c "import pyarrow.parquet" 2>/dev/null) && ok "pyarrow installed" || warn "pyarrow missing — quality battery unavailable"

echo "== vLLM patches (optional; only needed for INT8_ACT=int8)"
for p in patches/marlin-int8-layer-select.patch patches/marlin-int8-negative-scales.patch; do
  if [ -f "$SP/vllm" ] && patch -p1 -R --dry-run -s -d "$SP" < "$p" >/dev/null 2>&1; then ok "$(basename $p) applied"
  elif $PY patches/_check_applied.py "$p" "$SP" 2>/dev/null; then ok "$(basename $p) applied (content check)"
  elif patch -p1 -N --dry-run -s -d "$SP" < "$p" >/dev/null 2>&1; then warn "$(basename $p) not applied (bash setup.sh; INT8_ACT=int8 unavailable until then)"
  else warn "$(basename $p) neither applied nor applicable — vLLM version mismatch?"; fi
done

if [ $INSTALL = 0 ]; then
for M in "$MODEL_DIR"; do
  NAME=$(basename "$M")
  echo "== model at $M"
  if [ ! -f "$M/config.json" ]; then fail "$NAME: not found (bash prepare/fetch_models.sh)"; continue; fi
$PY - "$M" <<'EOF'
import json, os, sys
d = sys.argv[1].rstrip("/") + "/"
F = 0
def ok(m): print("  PASS  " + m)
def warn(m): print("  WARN  " + m)
def fail(m):
    global F; F += 1; print("  FAIL  " + m)
try:
    c = json.load(open(d + "config.json"))
    if c.get("architectures") != ["Gemma4ForConditionalGeneration"]:
        fail(f"unexpected architectures {c.get('architectures')}")
    tc = c.get("text_config", {})
    if tc.get("num_hidden_layers") == 42 and tc.get("num_key_value_heads") == 2:
        ok(f"config: 42 layers (35 sliding + 7 full), 2 KV heads, {tc.get('max_position_embeddings')} max pos, PLE {'yes' if tc.get('hidden_size_per_layer_input') else 'no'}")
    else:
        fail("text_config does not look like Gemma 4 E4B")
except Exception as e:
    fail(f"config.json unreadable: {e}"); sys.exit(1)
try:
    if os.path.exists(d + "model.safetensors.index.json"):
        idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
        keys = list(idx)
        missing = [f for f in set(idx.values()) if not os.path.exists(d + f)]
        if missing: fail(f"safetensors shards missing: {missing}")
        else: ok(f"{len(set(idx.values()))} safetensors shards present")
    elif os.path.exists(d + "model.safetensors"):
        # single-file checkpoint (compressed-tensors QAT ships one blob)
        from safetensors import safe_open
        keys = list(safe_open(d + "model.safetensors", "pt").keys())
        ok("single-file model.safetensors present")
    else:
        fail("no model.safetensors[.index.json] — incomplete download"); keys = []
    # Accept either quantization format: GPTQ uses qweight, compressed-tensors
    # uses weight_packed. Both end up on the same Marlin kernels.
    if any("weight_packed" in k for k in keys):
        qn = sum(1 for k in keys if "weight_packed" in k)
        ok(f"quantized weights: compressed-tensors format ({qn} packed tensors)")
    elif any("qweight" in k for k in keys):
        qn = sum(1 for k in keys if "qweight" in k)
        ok(f"quantized weights: GPTQ/AutoRound format ({qn} qweight tensors)")
    elif keys:
        fail("no quantized weights found — not a w4a16 checkpoint")
except Exception as e:
    fail(f"weight index unreadable: {e}")
try:
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(d).encode("hello", add_special_tokens=False)
    if ids: ok("tokenizer loads")
    else: fail("tokenizer encodes to [] — copy tokenizer.json + tokenizer_config.json from the base model")
except Exception as e:
    fail(f"tokenizer will not load ({type(e).__name__}: {str(e)[:70]})")
for f in ("chat_template.jinja", "processor_config.json"):
    if not os.path.exists(d + f): warn(f + " missing (tool calling / multimodal processor may be limited)")
sys.exit(1 if F else 0)
EOF
  [ $? -ne 0 ] && FAILS=$((FAILS+1))
done
fi  # INSTALL

echo "== keys / units"
# A key is optional: with neither api_key.txt nor VLLM_API_KEY the launcher
# exports nothing and vLLM serves unauthenticated, which is fine locally.
# Worth a WARN only because the launcher binds 0.0.0.0.
[ -s api_key.txt ] || [ -n "${VLLM_API_KEY:-}" ] && ok "API key configured (api_key.txt or VLLM_API_KEY)" \
  || warn "no API key — the server will accept any request, and it listens on 0.0.0.0. Fine behind a firewall; otherwise: openssl rand -hex 24 > api_key.txt"
[ -f examples_tool_chat_template_gemma4.jinja ] && ok "gemma4 tool chat template vendored" || fail "examples_tool_chat_template_gemma4.jinja missing (TOOL_ARGS in batch/start_gemma.sh needs it)"
if [ -f /.dockerenv ]; then :; elif systemctl --user is-active gemma-serving >/dev/null 2>&1; then ok "systemd user unit gemma-serving active"; else warn "gemma-serving unit not active (fine if you launch the script by hand)"; fi

if [ $NOSRV = 0 ]; then
  echo "== live server (127.0.0.1:${PORT:-18030})"
  PORT=${PORT:-18030}
  if curl -sf -o /dev/null http://127.0.0.1:$PORT/health; then
    ok "/health 200"
    KEY=${VLLM_API_KEY:-$(cat api_key.txt 2>/dev/null)}
    R=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"Hvad er hovedstaden i Danmark? Svar med ét ord."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
    echo "$R" | grep -qi "københavn\|copenhagen" && ok "chat completion answers ('$(echo "$R" | $PY -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null)')" || fail "chat completion wrong/failed: $(echo "$R" | head -c 200)"
    LOG=$HERE/gemma.log
    if [ -f "$LOG" ]; then
      grep -oE "Using [A-Z_0-9]+ attention backend" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -q "MarlinLinearKernel\|gptq" "$LOG" 2>/dev/null && ok "Marlin kernels in use" || true
    fi
  else warn "no server on :$PORT (start batch/start_gemma.sh, or pass --no-server)"; fi
fi
echo
[ $FAILS = 0 ] && echo "verify: OK ($FAILS failures)" || echo "verify: $FAILS FAILURE(S)"
exit $([ $FAILS = 0 ] && echo 0 || echo 1)
