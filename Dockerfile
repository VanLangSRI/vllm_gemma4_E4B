# Gemma 4 E4B on one RTX 3090-class 24 GB card, batch mode.
# Same stack as the README's venv install, frozen: Python 3.12 venv at /app/venv,
# vLLM 0.27.1 (torch 2.13 / cu130 / Triton 3.7.1), Marlin int8-activation
# patches applied (for INT8_ACT=int8), verify.sh --install run at build time.
#
# The base image is CUDA "base" + nvcc, not "devel": vLLM's wheels bring their
# own CUDA libraries, but FlashInfer JIT-compiles its fp8-KV attention kernel
# with nvcc on first use and Triton needs a C compiler for its launchers. The
# compiled kernels and the torch.compile cache live in the /cache volume, so
# that happens once.
#
#   docker compose up -d         (see README, "Docker")
FROM nvidia/cuda:13.0.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 \
      build-essential patch curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.12 -m venv venv && venv/bin/pip install --upgrade pip
COPY docker/requirements.txt docker/requirements.txt
RUN venv/bin/pip install -r docker/requirements.txt

COPY . .
RUN set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1); \
    FSP=$(venv/bin/python -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))' | tail -n1); \
    for spec in "patches/marlin-int8-layer-select.patch|$SP" "patches/marlin-int8-negative-scales.patch|$SP" \
                "patches/vllm-heterogeneous-config-global-access.patch|$SP" \
                "patches/vllm-gemma4-per-layer-head-dim.patch|$SP" \
                "patches/flashinfer-fa2-sm86-fp8-kv.patch|$FSP"; do \
      p="${spec%%|*}"; d="${spec##*|}"; echo "== $p"; \
      if venv/bin/python patches/_check_applied.py "$p" "$d" 2>/dev/null || patch -p1 -N --dry-run -s -d "$d" < "$p" >/dev/null 2>&1; then \
        patch -p1 -d "$d" < "$p" || true; \
      else echo "WARN $p not applicable (version drift)"; fi; \
    done; \
    venv/bin/pip install av pyarrow; \
    bash verify.sh --install

# HOME is a volume: torch.compile cache (~/.cache/vllm), Triton (~/.triton),
# FlashInfer JIT (~/.cache/flashinfer), HF hub cache.
RUN mkdir -p /cache /app/models && chmod 1777 /cache
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_ENABLE_HF_TRANSFER=1
VOLUME ["/cache", "/app/models"]
EXPOSE 18030
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
CMD ["batch"]
