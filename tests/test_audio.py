import shutil
import wave

import numpy as np
import pytest

from livedub.audio import INPUT_RATE, decode_to_pcm16, split_points, write_wav

RATE = INPUT_RATE


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-8000, 8000, int(seconds * RATE), dtype=np.int16)


def test_short_audio_is_one_segment():
    samples = noise(30)
    assert split_points(samples, RATE, max_seconds=60) == [(0, len(samples))]


def test_cut_lands_in_the_pause():
    samples = noise(100)
    samples[50 * RATE : 51 * RATE] = 0
    bounds = split_points(samples, RATE, max_seconds=60)
    assert len(bounds) == 2
    cut = bounds[0][1]
    assert 50 * RATE <= cut <= 51 * RATE


def test_segments_are_contiguous_and_bounded():
    samples = noise(1000, seed=1)
    bounds = split_points(samples, RATE, max_seconds=60)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(samples)
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        assert end == start
    assert all(0 < end - start <= 60 * RATE for start, end in bounds)


def test_too_short_limit_is_rejected():
    with pytest.raises(ValueError):
        split_points(noise(5), RATE, max_seconds=0.5)


def test_write_wav_roundtrip(tmp_path):
    pcm = noise(1).tobytes()
    out = tmp_path / "sub" / "x.wav"
    write_wav(out, pcm, 24000)
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert w.readframes(w.getnframes()) == pcm
    assert [p.name for p in out.parent.iterdir()] == ["x.wav"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_decode_resamples_to_16k_mono(tmp_path):
    src = tmp_path / "stereo.wav"
    stereo = np.random.default_rng(2).integers(-8000, 8000, (2 * 44100, 2), dtype=np.int16)
    with wave.open(str(src), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(stereo.tobytes())
    pcm = decode_to_pcm16(src)
    assert abs(len(pcm) // 2 - 2 * RATE) < 200


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_decode_reports_bad_files(tmp_path):
    bad = tmp_path / "notes.mp3"
    bad.write_text("not audio")
    with pytest.raises(RuntimeError, match="could not decode"):
        decode_to_pcm16(bad)
