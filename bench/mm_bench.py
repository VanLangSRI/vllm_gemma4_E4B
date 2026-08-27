#!/usr/bin/env python3
"""Multimodal throughput comparison: this repo's vLLM server vs Ollama.

Measures what image/audio INPUT costs on top of text: the encoders run during
prefill, so the interesting numbers here are TTFT (dominated by the vision /
audio tower) and aggregate decode tok/s under concurrency — not just e2e.

Protocol (same shape as the text cohorts): N=8 identical-shape requests,
model-default sampling, up to NUM_PREDICT output tokens, concurrency C in
1/2/4/8. Assets come from bench/mm-assets/, preferring REAL media:
  real_cat.jpg     1024x683 photograph (Wikimedia Commons)
  real_speech.wav  15 s 16 kHz mono recorded speech (Harvard sentences)
Get them with `bash bench/fetch_mm_assets.sh`, then confirm the model actually
perceives them with `bench/mm_check.py`. If they are absent this falls back to
synthetic filler (gradient PNG, band-limited noise) which measures encoder cost
fine but CANNOT reveal a silently-broken media path -- a server that drops the
attachment still emits tokens at full speed.

Both backends take BOTH modalities:
  vllm    image + audio as data URIs over the OpenAI-compatible endpoint
          (`image_url` / `audio_url` content parts); requires the server
          started with VISION=1/AUDIO=1 — the defaults here
  ollama  there is no dedicated audio field: media goes into the message's
          "images" array as base64 and the engine sniffs magic bytes per blob
          (llm/media.go AudioFormat: RIFF..WAVE -> wav, ID3/MPEG -> mp3),
          routing WAV/MP3 to the audio encoder and everything else to vision.
          So a .wav passed like an image IS processed as audio — verified in
          llm/llama_server.go llamaServerChatMediaPart.

Fairness rules (an earlier version of this file broke all three, and every
break happened to favour Ollama — the corrected numbers reversed the result):
  1. per-stream rate is WALL-CLOCK on both engines. Ollama's `eval_duration`
     covers decode only and excludes prefill and the vision/audio encoder,
     i.e. precisely what this benchmark measures. Using it inflated Ollama by
     ~2.1x (113.0 vs 53.2 tok/s on one identical request).
  2. the same tokens are counted on both sides — every token the server says it
     generated (vLLM `usage.completion_tokens`, Ollama `eval_count`). Counting
     visible chunks on one engine and total tokens on the other is not a
     comparison. `visible=` in each ROW shows how many of those tokens were
     actually answer text rather than hidden reasoning.
  3. thinking is DISABLED on both (vLLM `enable_thinking:false`, Ollama
     `think:false`), matching the text cohorts. Left on, gemma4:e4b on Ollama
     spent its entire 512-token budget on hidden reasoning and returned an
     EMPTY answer, while still scoring a full-speed result.

Warmup: one tiny request fires first and its time is NEVER counted (same rule
as the other benches) — model load/JIT must stay out of the numbers.

Usage:
  # both sides, images:
  venv/bin/python bench/mm_bench.py --api vllm   --modality image
  venv/bin/python bench/mm_bench.py --api ollama --modality image
  # vLLM only:
  venv/bin/python bench/mm_bench.py --api vllm --modality audio
  venv/bin/python bench/mm_bench.py --api vllm --modality both
Run with the repo venv (needs pillow): venv/bin/python bench/mm_bench.py ...
"""
import argparse, base64, json, math, os, random, statistics, struct, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "mm-assets")
INSTRUCTION = ("Describe the attached media in one paragraph, then list five "
               "concrete details you can infer from it.")

ap = argparse.ArgumentParser()
ap.add_argument("--api", choices=["vllm", "ollama"], required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=None)      # default per api
ap.add_argument("--model", default=None)               # default per api
ap.add_argument("--modality", choices=["image", "audio", "both"], default="image")
ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--num-predict", type=int, default=512)
ap.add_argument("-n", "--num-requests", type=int, default=8)


# ---------------------------------------------------------------- assets ---
def ensure_assets():
    """Prefer the REAL assets from bench/fetch_mm_assets.sh; fall back to
    synthetic ones so the bench still runs offline.

    Synthetic media is fine for measuring encoder cost but useless for proving
    the pipeline works: a server that silently drops the audio (as this one did
    before soundfile was installed) produces identical throughput numbers. Run
    `bash bench/fetch_mm_assets.sh` and use --check to validate comprehension.
    """
    os.makedirs(ASSETS, exist_ok=True)
    real_png = os.path.join(ASSETS, "real_cat.jpg")
    real_wav = os.path.join(ASSETS, "real_speech.wav")
    if os.path.exists(real_png) and os.path.exists(real_wav):
        return real_png, real_wav
    png, wav = os.path.join(ASSETS, "photo.png"), os.path.join(ASSETS, "clip.wav")
    if not os.path.exists(png):
        from PIL import Image
        random.Random(0).seed(0)
        img = Image.new("RGB", (1024, 1024))
        px = img.load()
        for y in range(0, 1024, 4):          # coarse loop, smooth gradient
            for x in range(0, 1024, 4):
                c = ((x * 255) // 1024, (y * 255) // 1024, (x ^ y) % 256)
                for dy in range(4):
                    for dx in range(4):
                        px[x + dx, y + dy] = c
        img.save(png)
        print(f"# generated {png}")
    if not os.path.exists(wav):
        rate = 16000; secs = 15
        rng = random.Random(0)
        frames = bytearray()
        phase = 0.0
        for i in range(rate * secs):
            # speech-band signal: 200 Hz carrier with formant-ish jitter
            f = 200 + 60 * math.sin(i / rate * 2.1)
            phase += 2 * math.pi * f / rate
            s = 0.5 * math.sin(phase) + 0.25 * rng.uniform(-1, 1)
            frames += struct.pack("<h", int(s * 12000))
        import wave
        with wave.open(wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes(bytes(frames))
        print(f"# generated {wav}")
    return png, wav


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


# --------------------------------------------------------------- request ---
def build_payload(png64, wav64):
    """One chat completion for the chosen modality, on either API."""
    if args.api == "vllm":
        parts = [{"type": "text", "text": INSTRUCTION}]
        if args.modality in ("image", "both"):
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{png64}"}})
        if args.modality in ("audio", "both"):
            parts.append({"type": "audio_url",
                          "audio_url": {"url": f"data:audio/wav;base64,{wav64}"}})
        return {"model": MODEL, "messages": [{"role": "user", "content": parts}],
                "max_tokens": args.num_predict, "stream": True,
                # Match the text cohorts, and match Ollama's think=False below:
                # with thinking on, the engines spend different fractions of the
                # token budget on hidden reasoning and the rates stop comparing.
                "chat_template_kwargs": {"enable_thinking": False},
                "stream_options": {"include_usage": True}}
    # ollama: one "images" array carries every modality — the engine sniffs
    # each blob's magic bytes and routes WAV to the audio encoder (see module
    # docstring). Order matters only for the model's [img-N] references.
    media = []
    if args.modality in ("image", "both"):
        media.append(png64)
    if args.modality in ("audio", "both"):
        media.append(wav64)
    return {"model": MODEL,
            "messages": [{"role": "user", "content": INSTRUCTION,
                          "images": media}],
            "stream": True, "think": False,
            "options": {"num_predict": args.num_predict}}


def post(payload):
    """Returns (ttft_ms, completion_tokens, eval_duration_ns_or_0)."""
    path = "/v1/chat/completions" if args.api == "vllm" else "/api/chat"
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; n_tok = 0; n_vis = 0; eval_ns = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            # SSE framing: lines look like "data: {...}" with a final
            # "data: [DONE]" sentinel -- strip the prefix before parsing.
            line = raw.strip()
            if not line:
                continue
            if line.endswith(b"[DONE]"):
                continue
            if line.startswith(b"data:"):
                line = line[len(b"data:"):].strip()
                if not line or line == b"[DONE]":
                    continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Count the SAME thing on both engines: every token the server
            # generated, as the server itself reports it. Counting streamed
            # .content chunks on one side and eval_count (which includes hidden
            # thinking tokens) on the other is not a comparison.
            if args.api == "vllm":
                ch = obj.get("choices") or []
                if ch:
                    delta = ch[0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        if ttft is None: ttft = time.perf_counter() - t0
                        n_vis += 1 if delta.get("content") else 0
                if obj.get("usage"):
                    n_tok = obj["usage"].get("completion_tokens", n_tok)
            else:
                msg = obj.get("message", {})
                if msg.get("content") or msg.get("thinking"):
                    if ttft is None: ttft = time.perf_counter() - t0
                    n_vis += 1 if msg.get("content") else 0
                if obj.get("done"):
                    n_tok = obj.get("eval_count", n_tok)
                    eval_ns = obj.get("eval_duration", 0)
    return ((ttft or 0) * 1000, n_tok, eval_ns, n_vis,
            time.perf_counter() - t0)


def warmup(png64, wav64):
    t0 = time.time()
    while True:
        try:
            post(build_payload(png64, wav64)); break
        except Exception as e:
            if time.time() - t0 > 1800:
                raise SystemExit(f"warmup failed for 30 min: {e}")
            time.sleep(3)       # model still loading / JIT — keep waiting
    print(f"# warmup ok after {time.time()-t0:.0f}s (NOT counted in any ROW below)")


def main():
    png, wav = ensure_assets()
    png64, wav64 = b64(png), b64(wav)
    print(f"# {time.strftime('%F %T')} {args.api} {MODEL} @ {API} "
          f"modality={args.modality} num_predict={args.num_predict}")
    warmup(png64, wav64)

    for C in args.conc:
        jobs = [build_payload(png64, wav64) for _ in range(args.num_requests)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(C) as ex:
            res = list(ex.map(post, jobs))
        wall = time.perf_counter() - t0
        total = sum(r[1] for r in res)
        visible = sum(r[3] for r in res)
        # Per-stream must be WALL-CLOCK on both engines. Ollama reports
        # eval_duration, which covers decode only and excludes prefill and the
        # vision/audio encoder -- exactly the work this benchmark exists to
        # measure. Dividing by it inflated Ollama's per-stream rate ~2.1x
        # (measured: 113.0 tok/s by eval_duration vs 53.2 tok/s wall-clock on
        # the same request) and made a slower engine look faster.
        streams = sorted(r[1] / r[4] for r in res if r[4] > 0)
        med = statistics.median(streams) if streams else 0
        note = "" if visible else "  [!] server emitted 0 visible tokens (all thinking)"
        print(f"ROW mm-{args.modality} {args.api} C{C} | e2e={total/wall:.1f} tok/s | "
              f"per-stream(med)={med:.1f} tok/s | meanTTFT={statistics.mean(r[0] for r in res):.0f} ms "
              f"| dur={wall:.0f}s | visible={visible}/{total} tok{note}")


if __name__ == "__main__":
    args = ap.parse_args()
    if args.api == "vllm":
        PORT = args.port or int(os.environ.get("PORT", 18030))
        MODEL = args.model or "gemma-4-e4b"
    else:
        PORT = args.port or 11434
        MODEL = args.model or "gemma4:e4b"
    API = f"http://{args.host}:{PORT}"   # used by post()/build_payload via globals
    main()
