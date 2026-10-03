#!/bin/sh
# Starts the livedub web page on http://localhost:8000 (macOS / Linux).
set -e
cd "$(dirname "$0")"

PY=""
for candidate in "${PYTHON:-}" python3.13 python3.12 python3.11 python3; do
  if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11 or newer is needed: https://www.python.org/downloads/" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating the Python environment (first run only)..."
  "$PY" -m venv .venv
fi
echo "Installing/updating packages..."
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
command -v ffmpeg >/dev/null 2>&1 ||
  echo "WARNING: ffmpeg was not found (macOS: brew install ffmpeg, Linux: sudo apt install ffmpeg)"

exec .venv/bin/python -m livedub.web --open "$@"
