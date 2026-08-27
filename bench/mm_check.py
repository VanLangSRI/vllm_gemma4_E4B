#!/usr/bin/env python3
"""Comprehension check: does the server actually SEE / HEAR the attachment?

Throughput benchmarks cannot answer this — a model that silently drops the media
still emits tokens at full speed. Each case sends a real asset with a question
whose answer is only knowable from the media, then greps the reply for expected
keywords.
"""
import base64, json, os, sys, urllib.request

A = "bench/mm-assets"
HOST = "http://127.0.0.1:18030/v1/chat/completions"


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def ask(parts, maxtok=200):
    body = json.dumps({
        "model": "gemma-4-e4b",
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": maxtok, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    r = urllib.request.Request(HOST, data=body,
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))["choices"][0]["message"]["content"]


def img(path, mime="image/jpeg"):
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64(path)}"}}


def aud(path):
    return {"type": "audio_url",
            "audio_url": {"url": f"data:audio/wav;base64,{b64(path)}"}}


CASES = [
    ("IMAGE  cat photo", [
        {"type": "text", "text": "What animal is in this photo, and what is it standing on? Answer in one short sentence."},
        img(f"{A}/real_cat.jpg")],
     ["cat", "snow"]),
    ("IMAGE  Hollywood sign (OCR)", [
        {"type": "text", "text": "What word is spelled out by the large letters in this image? Reply with just that word."},
        img(f"{A}/real_text.jpg")],
     ["hollywood"]),
    ("IMAGE  castle scene", [
        {"type": "text", "text": "Describe the main structure in this photo in one sentence."},
        img(f"{A}/real_scene.jpg")],
     ["castle", "bridge", "stone", "water", "loch"]),
    ("AUDIO  Harvard sentences", [
        {"type": "text", "text": "Transcribe the speech in this audio as accurately as you can."},
        aud(f"{A}/real_speech.wav")],
     ["birch", "canoe", "planks", "glue", "sheet"]),
    ("BOTH   image + audio", [
        {"type": "text", "text": "You are given a photo and an audio clip. In one sentence each: what animal is in the photo, and what is the first sentence spoken?"},
        img(f"{A}/real_cat.jpg"), aud(f"{A}/real_speech.wav")],
     ["cat", "birch"]),
]

fails = 0
for name, parts, expect in CASES:
    try:
        out = ask(parts)
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e)[:150]}")
        fails += 1
        continue
    low = out.lower()
    hit = [k for k in expect if k in low]
    ok = len(hit) >= (2 if len(expect) > 3 else 1)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: matched {hit or 'NOTHING'} of {expect}")
    print(f"        reply: {out.strip()[:220].replace(chr(10), ' ')}")
    fails += 0 if ok else 1

print(f"\ncomprehension: {'ALL PASS' if not fails else f'{fails} FAILURE(S)'}")
sys.exit(1 if fails else 0)
