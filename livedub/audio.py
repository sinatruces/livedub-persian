"""Audio helpers: decode any media file to raw PCM, split it into segments, write WAV."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

# The Live API takes 16-bit mono PCM at 16 kHz and answers with 16-bit mono PCM at 24 kHz.
INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
SAMPLE_WIDTH = 2

MEDIA_EXTENSIONS = {
    ".aac", ".aiff", ".amr", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus", ".wav", ".wma",
    ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".webm", ".wmv",
}


def decode_to_pcm16(path: str | Path, rate: int = INPUT_RATE) -> bytes:
    """Decode the audio track of any file ffmpeg understands to 16-bit mono PCM at `rate`."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH; install it first (see README)")
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(rate),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path}: {detail}")
    if not proc.stdout:
        raise RuntimeError(f"{path} has no audio track")
    return proc.stdout


def split_points(
    samples: np.ndarray,
    rate: int,
    max_seconds: float,
    search_seconds: float = 20.0,
) -> list[tuple[int, int]]:
    """Split `samples` into [start, end) ranges no longer than `max_seconds`.

    Each cut is placed at the quietest ~300 ms within the last `search_seconds`
    before the limit, so segments end in a pause rather than mid-word.
    """
    n = len(samples)
    max_len = int(max_seconds * rate)
    search = min(int(search_seconds * rate), max_len // 2)
    frame = rate // 20  # 50 ms
    smooth = 6  # frames, i.e. 300 ms
    if search < frame * smooth:
        raise ValueError("max_seconds is too short to search for a pause")

    bounds = []
    start = 0
    while n - start > max_len:
        hi = start + max_len
        lo = hi - search
        window = samples[lo:hi].astype(np.float32)
        frames = window[: len(window) // frame * frame].reshape(-1, frame)
        energy = np.convolve((frames**2).mean(axis=1), np.ones(smooth) / smooth, mode="valid")
        cut = lo + (int(np.argmin(energy)) + smooth // 2) * frame
        bounds.append((start, cut))
        start = cut
    bounds.append((start, n))
    return bounds


def wav_bytes(pcm: bytes, rate: int) -> bytes:
    """Wrap 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[bytes, int]:
    """Return (16-bit mono PCM, sample rate) from WAV bytes."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError("expected 16-bit mono WAV")
        return w.readframes(w.getnframes()), w.getframerate()


def write_wav(path: str | Path, pcm: bytes, rate: int) -> None:
    """Write 16-bit mono PCM to `path`, via a temp file so a crash never leaves a partial WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes(pcm, rate))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def rate_from_mime(mime_type: str | None, default: int) -> int:
    """Sample rate from a MIME type such as "audio/pcm;rate=24000"."""
    match = re.search(r"rate=(\d+)", mime_type or "")
    return int(match.group(1)) if match else default


def silence(seconds: float, rate: int) -> bytes:
    return bytes(int(seconds * rate) * SAMPLE_WIDTH)
