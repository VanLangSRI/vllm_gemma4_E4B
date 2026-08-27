#!/bin/bash
# Download REAL image and audio assets for bench/mm_bench.py.
#
# Why not the synthetic fallbacks the bench can generate (a gradient PNG,
# band-limited noise)? They exercise the encoders but say nothing about whether
# the model UNDERSTANDS the input — and a model that cannot see the picture
# still emits tokens at full speed, so a throughput-only benchmark looks
# identical whether the pipeline works or is silently broken. That is exactly
# how the audio path here stayed broken through an entire benchmark round
# (missing soundfile: vLLM logged an error, dropped the audio, and answered
# anyway). Real assets with known content let the bench assert on the REPLY,
# not just its token rate.
#
# All sources are permissive and need no API key:
#   images  Wikimedia Commons originals (public domain / CC BY-SA), downscaled
#   audio   "Open Speech Repository" Harvard sentences (free for research use),
#           a real recorded voice reading a known, fixed transcript
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A="$HERE/mm-assets"
mkdir -p "$A"
PY="$HERE/../venv/bin/python"

# Wikimedia requires a descriptive User-Agent and rejects hotlinking its
# /thumb/ renders (HTTP 400); fetch the original and downscale locally instead.
UA="vllm-3090-bench/1.0 (multimodal benchmark asset fetch)"

dl() { # url dest
  [ -s "$2" ] && { echo "  have     $(basename "$2")"; return 0; }
  printf '  fetching %s ... ' "$(basename "$2")"
  if curl -fsSL --retry 3 --max-time 180 -A "$UA" -o "$2.part" "$1"; then
    mv "$2.part" "$2"; echo "ok ($(du -h "$2" | cut -f1))"
  else
    rm -f "$2.part"; echo "FAILED"; return 1
  fi
}

# Downscale to 1024 px on the long edge: that matches the server's 2048
# image-token cap and keeps every request the same shape.
shrink() { "$PY" "$HERE/_shrink.py" "$1" "$2"; }

cat > "$HERE/_shrink.py" <<'PYEOF'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB")
im.thumbnail((1024, 1024), Image.LANCZOS)
im.save(sys.argv[2], quality=92)
print(f"  -> {sys.argv[2].rsplit('/', 1)[-1]} {im.size[0]}x{im.size[1]}")
PYEOF

echo "images (Wikimedia Commons originals, downscaled to 1024 px):"
C="https://upload.wikimedia.org/wikipedia/commons"
if dl "$C/b/b6/Felis_catus-cat_on_snow.jpg" "$A/_cat_orig.jpg"; then
  shrink "$A/_cat_orig.jpg" "$A/real_cat.jpg"; fi
if dl "$C/d/dc/Eilean_Donan_castle_-_95mm.jpg" "$A/_scene_orig.jpg"; then
  shrink "$A/_scene_orig.jpg" "$A/real_scene.jpg"; fi
if dl "$C/5/5a/Hollywood_Sign_%28Zuschnitt%29.jpg" "$A/_sign_orig.jpg"; then
  shrink "$A/_sign_orig.jpg" "$A/real_text.jpg"; fi
rm -f "$A"/_*_orig.jpg "$A"/_*_orig.png "$HERE/_shrink.py"

echo "audio (Open Speech Repository, Harvard sentences, real voice):"
if dl "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav" "$A/osr_raw.wav"; then
  # Gemma 4's audio tower wants 16 kHz mono; the source is 8 kHz. Resample and
  # trim to ~15 s so the clip sits inside the model's ~30 s window.
  "$PY" - "$A/osr_raw.wav" "$A/real_speech.wav" <<'PYEOF'
import sys, soundfile as sf, librosa
y, sr = sf.read(sys.argv[1], dtype="float32")
if y.ndim > 1:
    y = y.mean(axis=1)
y = librosa.resample(y, orig_sr=sr, target_sr=16000)
y = y[: 16000 * 15]
sf.write(sys.argv[2], y, 16000, subtype="PCM_16")
print(f"  -> real_speech.wav ({len(y)/16000:.1f}s @ 16 kHz mono)")
PYEOF
  # The Harvard sentences read at the start of this recording. mm_bench.py
  # --check greps the reply for these keywords to prove the audio was heard.
  cat > "$A/real_speech.txt" <<'TXT'
The birch canoe slid on the smooth planks
Glue the sheet to the dark blue background
It is easy to tell the depth of a well
These days a chicken leg is a rare dish
Rice is often served in round bowls
TXT
  rm -f "$A/osr_raw.wav"
  echo "  -> real_speech.txt (reference transcript)"
fi

echo
echo "assets in $A:"
ls -lh "$A" | tail -n +2 | awk '{printf "  %-20s %8s\n", $9, $5}'
