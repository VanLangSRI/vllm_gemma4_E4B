#!/usr/bin/env python3
"""Long-context benchmark for Ollama, mirroring the vLLM long-context protocol.

Sends N concurrent requests each carrying a ~<ctx> token prompt and asking for
512 output tokens, then reports TTFT / output tok/s / duration the same way
`vllm bench serve` does. A warmup request is issued first and never timed.

usage: ollama_longctx.py --ctx 65536 --conc 1 2 [--model gemma4:e4b]
"""
import argparse, json, statistics as st, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma4:e4b")
ap.add_argument("--host", default="http://127.0.0.1:11434")
ap.add_argument("--ctx", type=int, default=65536)
ap.add_argument("--out", type=int, default=512)
ap.add_argument("--conc", type=int, nargs="+", default=[1, 2])
a = ap.parse_args()

# Build a UNIQUE prompt per request. Two traps this avoids:
#  1. Ollama (like vLLM) caches prompt prefixes; reusing one prompt makes TTFT
#     measure a cache hit, not real prefill. Observed: a 49k-token prompt
#     "ingested" in 2.1 s on the second call.
#  2. Repetitive filler tokenizes far denser than real text — "quick brown fox"
#     ad nauseam gave 49,625 tokens where 65,536 were intended. Random words
#     from a large vocabulary tokenize much closer to 1 token/word.
import random

VOCAB = [f"{w}{i}" for i, w in enumerate(
    ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
     "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
     "xray yankee zulu").split() * 400)]


RUN_SALT = int(time.time() * 1000) & 0xFFFFFF  # unique per invocation


def make_prompt(seed):
    # Salt with the run's start time: identical seeds across two invocations
    # would otherwise hit Ollama's prompt cache and report a fake TTFT
    # (observed: the same 65k prompt "prefilled" in 12.5 s instead of 28.6 s).
    r = random.Random(f"{seed}-{RUN_SALT}")
    # ~1.3 tokens per token-ish word; overshoot then trim by measured count.
    words = [r.choice(VOCAB) for _ in range(int(a.ctx * 0.95))]
    return ("Document " + str(seed) + ":\n" + " ".join(words)
            + "\n\nSummarize the document above.")


def one(results, idx, seed=None):
    body = json.dumps({
        "model": a.model, "prompt": make_prompt(idx if seed is None else seed),
        "stream": True,
        "options": {"num_predict": a.out, "num_ctx": a.ctx},
    }).encode()
    req = urllib.request.Request(a.host + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("response"):
                    n += 1
                # First token can arrive as an empty string chunk; count the
                # first chunk that carries the key at all, else TTFT reads 0.
                if ttft is None and "response" in d and not d.get("done"):
                    ttft = time.time() - t0
                if d.get("done"):
                    ev = d.get("eval_count") or n
                    # Ollama's stream does not reliably mark the first token, so
                    # prefer its own server-side prefill timer when present.
                    ped = d.get("prompt_eval_duration")
                    server_ttft = (ped / 1e9) if ped else None
                    results[idx] = (server_ttft or ttft or 0.0,
                                    time.time() - t0, ev,
                                    d.get("prompt_eval_count") or 0)
                    return
    except Exception as e:
        results[idx] = (None, time.time() - t0, 0, 0)
        print(f"  request {idx} failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


print(f"# ollama {a.model} ctx={a.ctx} out={a.out}, unique prompt per request")
warm = {}
t = time.time(); one(warm, 0, seed=999999)
print(f"# warmup done in {time.time()-t:.0f}s (not counted); "
      f"warmup ingested {warm.get(0,(0,0,0,0))[3]} prompt tokens")

for c in a.conc:
    res = {}
    base = c * 1000
    ths = [threading.Thread(target=one, args=(res, i, base + i)) for i in range(c)]
    t0 = time.time()
    for th in ths: th.start()
    for th in ths: th.join()
    dur = time.time() - t0
    ok = [v for v in res.values() if v[0] is not None and v[2]]
    if not ok:
        print(f"ROW ollama ctx{a.ctx//1024}K C{c} | ALL REQUESTS FAILED")
        continue
    toks = sum(v[2] for v in ok)
    ttfts = [v[0] * 1000 for v in ok]
    pin = st.mean([v[3] for v in ok])
    tot = (toks + pin * len(ok)) / dur
    print(f"ROW ollama ctx{a.ctx//1024}K C{c} | out={toks/dur:6.2f} tok/s | "
          f"total={tot:8.1f} tok/s | meanTTFT={st.mean(ttfts):8.0f} ms | "
          f"dur={dur:6.1f}s | ok={len(ok)}/{c} | prompt_tok={pin:.0f}")
