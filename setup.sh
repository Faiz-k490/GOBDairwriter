#!/usr/bin/env bash
# Air Writer — one-command setup.
#
# Gets a clone from nothing to runnable: virtualenv, pinned dependencies, and
# the two MediaPipe model files (both gitignored, so a fresh clone has neither
# and the app will not start without them).
#
#   ./setup.sh
#   .venv/bin/python main.py
#
# Safe to re-run; it skips whatever is already in place.

set -euo pipefail
cd "$(dirname "$0")"

# mediapipe 0.10.33 ships no 3.14 wheels, and 3.14 is the macOS/brew default.
PY=""
for c in python3.13 /opt/homebrew/opt/python@3.13/libexec/bin/python3 python3.12; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
    echo "error: need python 3.13 or 3.12 (mediapipe has no 3.14 wheels)" >&2
    echo "       try: brew install python@3.13" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "==> creating .venv with $PY"
    "$PY" -m venv .venv
fi

echo "==> installing dependencies"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# Model files are gitignored — they are large binaries, not source.
fetch () {
    if [ -f "$1" ]; then
        echo "==> $1 already present"
    else
        echo "==> downloading $1"
        curl -fsSL -o "$1" "$2"
    fi
}
fetch blaze_face_short_range.tflite \
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
fetch hand_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "==> created .env from .env.example — add your Gmail app password"
fi

echo
echo "Done.  Run it with:  .venv/bin/python main.py"
echo "Emailing needs credentials in .env — see .env.example."
