#!/usr/bin/env python3
"""Measure what --enable-prefix-caching is actually worth.

Prefix caching reuses the KV of a shared prompt prefix across requests. It is a
big win for API backends where every request carries the same system prompt or
few-shot block, and does exactly nothing when requests share no prefix. This
measures BOTH cases so the number is honest.

  shared — a long system prompt reused by every request (cache should hit)
  unique — an equally long but per-request-unique prefix (cache must miss)

The unique arm is the control: without it, "prefix caching is fast" could just
mean "the second request is fast because the GPU warmed up".

usage: prefix_bench.py --label on|off [--conc 8] [--rounds 3]
"""
import argparse, json, random, statistics as st, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--host", default="http://127.0.0.1:18030")
ap.add_argument("--conc", type=int, default=8)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--out", type=int, default=64)
a = ap.parse_args()

SALT = int(time.time() * 1000) & 0xFFFFFF

# A realistic long system prompt: policy block + few-shot examples, ~3k tokens.
SYSTEM = (
    "You are a meticulous technical support assistant for a cloud platform.\n"
    + "\n".join(
        f"Policy {i}: When a customer reports issue class {i}, first verify the "
        f"account tier, then check region {chr(65+i%26)}, then escalate to team "
        f"{100+i} if the error budget for the month exceeds {i*3}% of its limit. "
        f"Never promise a refund before ticket triage completes."
        for i in range(120)
    )
)

QUESTIONS = [
    "Customer reports issue class 7 in region C. What do I do first?",
    "A tier-2 account hit 40% error budget. Escalate or not?",
    "Summarize the escalation rule for issue class 12.",
    "Which team handles issue class 55?",
    "Can I promise a refund before triage? Cite the policy.",
    "What region does issue class 3 map to?",
    "Explain the error-budget threshold in one sentence.",
    "Who do I contact for issue class 88?",
]


def ask(system, question, res, i):
    body = json.dumps({
        "model": "gemma-4-e4b",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": question}],
        "max_tokens": a.out, "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(a.host + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            c = raw[6:].strip()
            if c == b"[DONE]":
                break
            d = json.loads(c)
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    res[i] = (ttft or 0.0, time.time() - t0, n)


def arm(kind):
    ttfts, durs = [], []
    for rnd in range(a.rounds):
        res = {}
        ths = []
        for i in range(a.conc):
            if kind == "shared":
                sysmsg = SYSTEM                       # identical for everyone
            else:
                sysmsg = f"Session {SALT}-{rnd}-{i}. " + SYSTEM  # unique prefix
            ths.append(threading.Thread(
                target=ask, args=(sysmsg, QUESTIONS[i % len(QUESTIONS)], res, i)))
        t0 = time.time()
        for t in ths: t.start()
        for t in ths: t.join()
        d = time.time() - t0
        if rnd == 0 and kind == "shared":
            continue          # round 0 populates the cache; not a hit yet
        durs.append(d)
        ttfts += [v[0] * 1000 for v in res.values()]
    print(f"ROW prefix={a.label:<3} {kind:<6} C{a.conc} | "
          f"meanTTFT={st.mean(ttfts):7.0f} ms | medTTFT={st.median(ttfts):7.0f} ms | "
          f"round={st.mean(durs):5.2f} s")


ask(SYSTEM, "warm up", {}, 0)     # never timed
print(f"# prefix={a.label}: warmup done, system prompt ~{len(SYSTEM)//4} tokens")
arm("shared")
arm("unique")
