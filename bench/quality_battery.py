#!/usr/bin/env python3
"""Quick quality battery against the running server. Catches "benchmarks great,
outputs garbage" in a minute — run it after every kernel/quant change.
  1. perplexity over ~300-token windows: wikitext-2 test (en), vLLM's own
     python source (code)
  2. GSM8K exact-match, 200 test questions, thinking off, greedy
Data (once): bash bench/fetch_quality_data.sh
Perplexity needs prompt_logprobs, which needs memory headroom: run the server
with GPU_UTIL=0.93 for this (the batch default).
Usage: python bench/quality_battery.py <tag> [--ppl-only] [--gsm-only] [--gsm-n 200]
"""
import json, os, sys, glob, re, math, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""
KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(HERE, "..", "api_key.txt"))
API = os.environ.get("VLLM_API", "http://127.0.0.1:18030/v1")
Q = os.environ.get("QUALITY_DATA", os.path.join(HERE, "quality-data"))
MODEL = os.environ.get("VLLM_MODEL", "gemma-4-e4b")

tag = sys.argv[1]
ppl_only = "--ppl-only" in sys.argv; gsm_only = "--gsm-only" in sys.argv
gsm_n = int(sys.argv[sys.argv.index("--gsm-n")+1]) if "--gsm-n" in sys.argv else 200

def post(path, payload, timeout=1200):
    req = urllib.request.Request(API+path, data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def docs():
    out = []
    t = pq.read_table(f"{Q}/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet").column("text").to_pylist()
    txt = "".join(t)
    for i in range(0, min(len(txt), 40*1200), 1200):
        if len(txt[i:i+1200]) > 800: out.append(("en", txt[i:i+1200]))
    # code: the installed vLLM's own source
    vsp = glob.glob(os.path.join(HERE, "..", "venv", "lib", "python*", "site-packages", "vllm", "v1", "core", "*.py"))
    for f in sorted(vsp):
        s = open(f).read()
        for i in range(0, min(len(s), 4*1200), 1200):
            if len(s[i:i+1200]) > 800: out.append(("code", s[i:i+1200]))
        if sum(1 for l,_ in out if l=="code") >= 40: break
    return out

def ppl_one(item):
    lang, text = item
    r = post("/completions", {"model":MODEL,"prompt":text,"max_tokens":1,"temperature":0,
                              "prompt_logprobs":0,"echo":False})
    pl = r["choices"][0]["prompt_logprobs"]  # list; first is None
    lps = []
    for e in pl[1:]:
        if e is None: continue
        # prompt_logprobs=0: the dict holds only the actual token
        v = list(e.values())[0]
        lps.append(v["logprob"] if isinstance(v, dict) else v)
    return lang, sum(lps), len(lps)

def run_ppl():
    items = docs()
    with ThreadPoolExecutor(2) as ex: res = list(ex.map(ppl_one, items))
    agg = {}
    for lang, s, n in res:
        a = agg.setdefault(lang, [0.0, 0]); a[0]+=s; a[1]+=n
    out = {lang: (math.exp(-s/n), n) for lang,(s,n) in agg.items()}
    tot_s = sum(s for s,n in agg.values()); tot_n = sum(n for _,n in agg.values())
    out["all"] = (math.exp(-tot_s/tot_n), tot_n)
    return out

def extract_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$",""))
    return m[-1].replace(",","") if m else None

def gsm_one(row):
    q, a = row
    gold = a.split("####")[-1].strip().replace(",","")
    r = post("/chat/completions", {"model":MODEL,
        "messages":[{"role":"user","content":q+"\n\nSolve step by step, then give the final answer as 'Final answer: <number>'."}],
        "max_tokens":768,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}})
    txt = r["choices"][0]["message"]["content"] or ""
    m = re.search(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", txt)
    pred = (m.group(1).replace(",","") if m else extract_num(txt))
    try: ok = abs(float(pred) - float(gold)) < 1e-6
    except Exception: ok = False
    return ok, r["usage"]["completion_tokens"]

def run_gsm():
    t = pq.read_table(f"{Q}/gsm8k/main/test-00000-of-00001.parquet")
    rows = list(zip(t.column("question").to_pylist(), t.column("answer").to_pylist()))[:gsm_n]
    with ThreadPoolExecutor(32) as ex: res = list(ex.map(gsm_one, rows))
    acc = sum(1 for ok,_ in res if ok)/len(res); toks = sum(n for _,n in res)/len(res)
    return acc, toks

t0=time.time(); result={"tag":tag}
if not gsm_only:
    result["ppl"] = run_ppl(); print(tag, "PPL", {k:(round(v[0],4),v[1]) for k,v in result["ppl"].items()}, f"{time.time()-t0:.0f}s", flush=True)
if not ppl_only:
    t1=time.time(); acc, toks = run_gsm(); result["gsm8k"]={"n":gsm_n,"acc":acc,"mean_tokens":toks}
    print(tag, f"GSM8K n={gsm_n} acc={acc:.3f} mean_tokens={toks:.0f} {time.time()-t1:.0f}s", flush=True)
with open(os.path.join(Q, f"result_{tag}.json"),"w") as f: json.dump(result,f)
