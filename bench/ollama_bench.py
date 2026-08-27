#!/usr/bin/env python3
"""Throughput comparison: this repo's vLLM server vs Ollama's `gemma4:e4b`.

Runs the SAME protocol as the cohort rows of bench/run_benchmarks.sh — the 8
realistic prompts in bench/prompts_real.jsonl, model-default sampling, up to
1024 generated tokens, concurrency C in 1/2/4/8/16 — but against Ollama's
native API, and prints ROW lines in the same shape so the two tables can be
read side by side.

Ollama reports per-request counters (eval_count, eval_duration) which make the
aggregate honest even when requests finish at different times:
  aggregate tok/s = sum(eval_count) / wall time      (what the GPU delivered)
  per-stream tok/s = median(eval_count / wall time per request)
                     (eval_duration would exclude prefill -- see below)

WARMUP: both this script and bench/run_benchmarks.sh fire a warmup prompt
BEFORE any timed row and never count it — model load/JIT/compile time must not
pollute the throughput numbers. This script additionally retries the warmup
until Ollama answers successfully (first hit after `ollama serve` returns
"loading model" errors), so the load wait lands entirely outside the timings.

Setup for a FAIR batch comparison — Ollama defaults to serial request handling
(OLLAMA_NUM_PARALLEL=1-ish); restart it with a parallel window before benching,
e.g. (each var mirrors a knob of the vLLM server — see README for the table):
    pkill ollama
    OLLAMA_NUM_PARALLEL=16 OLLAMA_MAX_LOADED_MODELS=1 \\
    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=32768 \\
      ollama serve &
(q8_0 requires flash attention and is experimental with multimodal models;
verify the cache actually quantized via `grep "KV self size" ~/.ollama/logs/server.log`)
Note the quantizers differ (Ollama GGUF is Q4_K_M llama.cpp; ours is Google QAT
int4 g32 on Marlin with fp8 KV + prefix caching). Throughput differs a lot;
accuracy does not -- GSM8K n=1000 is a statistical tie (64.2% vs 62.5%, z=0.79).
Re-check with bench/quality_battery.py against each.

Usage:
  python bench/ollama_bench.py                     # C1..C8 vs localhost:11434
  python bench/ollama_bench.py --model gemma4:e4b --conc 1 2 4 8 16 32 --num-predict 1024
"""
import argparse, json, os, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.environ.get("PROMPTS", os.path.join(HERE, "prompts_real.jsonl"))

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=11434)
ap.add_argument("--model", default="gemma4:e4b")
ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--num-predict", type=int, default=1024)
ap.add_argument("--prompts", default=PROMPTS)
args = ap.parse_args()
API = f"http://{args.host}:{args.port}"

prompts = [json.loads(l)["prompt"] for l in open(args.prompts) if l.strip()]

def chat(prompt):
    """Streamed chat completion.

    Returns (ttft_ms, eval_count, eval_duration_ns, wall_s). Token counts come
    straight from Ollama's own final chunk -- eval_count is the number of tokens
    the engine says it generated, so nothing is estimated or re-tokenized here.
    """
    req = urllib.request.Request(API + "/api/chat", data=json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        # think:false is REQUIRED for a fair comparison. /api/chat applies the
        # chat template, and gemma4:e4b then spends its whole num_predict budget
        # on hidden reasoning: measured on the first cohort prompt, the reply was
        # 0 chars of content and 485 chars of `thinking`. The vLLM side of this
        # comparison runs `vllm bench serve` against /v1/completions, a raw
        # completion path with no chat template and therefore no thinking, so
        # leaving this on would benchmark two different tasks.
        "think": False,
        "options": {"num_predict": args.num_predict},
    }).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; last = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.strip():
                obj = json.loads(line)
                if ttft is None and obj.get("message", {}).get("content"):
                    ttft = time.perf_counter() - t0
                last = obj   # final chunk carries the counters
    return ((ttft or 0) * 1000, last.get("eval_count", 0),
            last.get("eval_duration", 0), time.perf_counter() - t0)

print(f"# {time.strftime('%F %T')} ollama {args.model} @ {API} num_predict={args.num_predict}")

def warmup():
    """Prompt mồi: đợi model load xong TRƯỚC khi đo, không tính vào kết quả."""
    req = lambda: urllib.request.Request(API + "/api/chat", data=json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "think": False, "options": {"num_predict": 4},
    }).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(req(), timeout=3600) as r:
                for line in r:
                    if line.strip(): json.loads(line)
            break
        except Exception as e:
            if time.time() - t0 > 1800: raise SystemExit(f"warmup failed for 30 min: {e}")
            time.sleep(3)   # still loading / JIT-compiling — keep waiting
    print(f"# warmup ok after {time.time()-t0:.0f}s (NOT counted in any ROW below)")

warmup()

for C in args.conc:
    jobs = [prompts[i % len(prompts)] for i in range(len(prompts))]  # 8 prompts, like the vLLM cohorts
    t0 = time.perf_counter()
    with ThreadPoolExecutor(C) as ex:
        res = list(ex.map(chat, jobs))
    wall = time.perf_counter() - t0
    total_tok = sum(r[1] for r in res)
    # Per-stream must be WALL-CLOCK, matching how the vLLM column is computed.
    # Ollama's eval_duration covers decode only: it excludes prefill and any
    # queueing, so dividing by it reports a rate the client never observes and
    # inflates Ollama against vLLM (measured 2.1x on one identical request in
    # the multimodal harness, which had the same defect).
    streams = sorted(r[1] / r[3] for r in res if r[3] > 0)
    med_stream = statistics.median(streams) if streams else 0
    # kept for reference: the decode-only rate Ollama itself would quote
    decode_only = sorted(r[1] * 1e9 / r[2] for r in res if r[2] > 0)
    med_decode = statistics.median(decode_only) if decode_only else 0
    mean_ttft = statistics.mean(r[0] for r in res)
    # TPOT = time per OUTPUT token after the first, i.e. the pure decode
    # interval a user sees. Both engines can report this, so it is the honest
    # per-stream comparison: unlike "e2e / concurrency" it is a measured
    # quantity, and unlike a rate it does not depend on how many streams ran.
    tpots = [ (r[3] - r[0]/1000.0) / (r[1]-1) * 1000.0
              for r in res if r[1] > 1 and r[3] > 0 ]
    med_tpot = statistics.median(tpots) if tpots else 0
    print(f"ROW ollama cohort C{C} real prompts | e2e={total_tok/wall:.1f} tok/s | "
          f"per-stream(med)={med_stream:.1f} tok/s | meanTTFT={mean_ttft:.0f} ms | "
          f"medTPOT={med_tpot:.2f} ms | dur={wall:.0f}s | tok={total_tok} | "
          f"decode-only(med)={med_decode:.1f} tok/s")
