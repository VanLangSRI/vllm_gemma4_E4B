#!/usr/bin/env python3
"""Measure speculative decoding on the workload it is supposed to help:
reproducing text that is already in the prompt (quote / edit / re-emit).

DFlash2's lookup-drafting patch in the reference repo targets exactly this:
"draft from the context, not from the drafter, when the context already contains
the answer". vLLM ships the model-agnostic version as method='ngram'/'suffix'.

Two workloads:
  copy  — "repeat the document verbatim"  -> near-100% draftable
  chat  — ordinary Q&A, nothing to copy    -> speculation should NOT help,
                                              and must not hurt much

usage: spec_bench.py --label ngram [--conc 1 4]
"""
import argparse, json, statistics as st, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--host", default="http://127.0.0.1:18030")
ap.add_argument("--conc", type=int, nargs="+", default=[1, 4])
ap.add_argument("--out", type=int, default=512)
a = ap.parse_args()

DOC = "\n".join(
    f"{i:3d}. The subsystem {chr(97+i%26)}{i} reported status code {1000+i*7} "
    f"at timestamp 2026-08-{1+i%28:02d}T{i%24:02d}:00Z with latency {i*3.7:.1f} ms."
    for i in range(60)
)

WORK = {
    "copy": ("Here is a log file:\n\n" + DOC +
             "\n\nReproduce the log file above EXACTLY, line for line, "
             "starting from line 0. Output nothing else."),
    "chat": ("Explain, in a few paragraphs, why memory bandwidth rather than "
             "raw FLOPs usually limits single-stream LLM decoding on consumer "
             "GPUs. Be specific and technical."),
}


def run(kind, prompt, conc):
    res = {}

    def one(i):
        body = json.dumps({
            "model": "gemma-4-e4b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.out, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(a.host + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time(); ttft = None; n = 0
        with urllib.request.urlopen(req, timeout=1200) as r:
            for raw in r:
                if not raw.startswith(b"data: "):
                    continue
                chunk = raw[6:].strip()
                if chunk == b"[DONE]":
                    break
                d = json.loads(chunk)
                ch = d.get("choices") or []
                if ch and ch[0].get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
                if d.get("usage"):
                    n = d["usage"].get("completion_tokens", n)
        res[i] = (ttft or 0.0, time.time() - t0, n)

    ths = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    dur = time.time() - t0
    toks = sum(v[2] for v in res.values())
    per = st.mean([v[2] / v[1] for v in res.values()])
    print(f"ROW {a.label:<10} {kind:<5} C{conc:<3} | e2e={toks/dur:7.1f} tok/s | "
          f"per-stream={per:6.1f} tok/s | TTFT={st.mean([v[0]*1000 for v in res.values()]):7.0f} ms")


# warmup, never timed
run("copy", WORK["copy"], 1)
print(f"# {a.label}: warmup done")
for kind, prompt in WORK.items():
    for c in a.conc:
        run(kind, prompt, c)
