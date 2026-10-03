import asyncio
import shutil
import wave

import numpy as np
import pytest
from google.genai import errors

from livedub import cli, pipeline
from livedub.pipeline import Job, collect_jobs, translate_file
from livedub.translator import Translator
from tests.fakes import FakeClient, FakeModels, daily_quota_error

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make_wav(path, seconds, rate=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.random.default_rng(0).integers(-8000, 8000, int(seconds * rate), dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def wav_frames(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 24000
        return w.getnframes()


def run_file(client, job, segment_seconds=10, jobs=1, **kwargs):
    translator = Translator(client, "fa", pace=100, idle_timeout=0.2, with_text=kwargs.get("save_text", False))
    return asyncio.run(
        translate_file(job, translator, segment_seconds=segment_seconds, sessions=asyncio.Semaphore(jobs), **kwargs)
    )


def test_collect_jobs_mirrors_folders(tmp_path):
    src = tmp_path / "in"
    (src / "a").mkdir(parents=True)
    (src / "a" / "talk.MP3").touch()
    (src / "b.mp4").touch()
    (src / "notes.txt").touch()
    single = tmp_path / "single.m4a"
    single.touch()

    jobs = collect_jobs([src, single], tmp_path / "out", "fa")
    assert [(j.source.name, j.output.relative_to(tmp_path / "out").as_posix()) for j in jobs] == [
        ("talk.MP3", "a/talk.fa.wav"),
        ("b.mp4", "b.fa.wav"),
        ("single.m4a", "single.fa.wav"),
    ]


def test_collect_jobs_rejects_clashes_and_missing_paths(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "talk.mp3").touch()
    (tmp_path / "talk.wav").touch()
    with pytest.raises(ValueError, match="both be written"):
        collect_jobs([tmp_path / "x" / "talk.mp3", tmp_path / "talk.wav"], tmp_path / "out", "fa")
    with pytest.raises(FileNotFoundError):
        collect_jobs([tmp_path / "nope.mp3"], tmp_path / "out", "fa")


@needs_ffmpeg
def test_long_file_is_split_translated_and_joined(tmp_path):
    job = Job(make_wav(tmp_path / "talk.wav", 25), tmp_path / "out" / "talk.fa.wav")
    client = FakeClient()
    run_file(client, job, jobs=3)

    assert len(client.sessions) >= 3
    replies = sum(len(s.sent) // 10 + 1 for s in client.sessions)
    gaps = (len(client.sessions) - 1) * int(pipeline.SEGMENT_GAP_SECONDS * 24000)
    assert wav_frames(job.output) == replies * 2400 + gaps


@needs_ffmpeg
def test_failed_segment_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RETRY_BASE_SECONDS", 0.01)
    job = Job(make_wav(tmp_path / "talk.wav", 5), tmp_path / "talk.fa.wav")
    client = FakeClient(fail_first=1)
    run_file(client, job)
    assert len(client.sessions) == 2
    assert job.output.exists()


@needs_ffmpeg
def test_gives_up_after_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RETRY_BASE_SECONDS", 0.01)
    job = Job(make_wav(tmp_path / "talk.wav", 5), tmp_path / "talk.fa.wav")
    client = FakeClient(fail_first=10)
    with pytest.raises(errors.APIError):
        run_file(client, job, retries=2)
    assert len(client.sessions) == 3
    assert not job.output.exists()


def test_run_all_keeps_the_original_order():
    async def slow(i):
        await asyncio.sleep((5 - i) * 0.01)  # later items finish first
        return i

    assert asyncio.run(pipeline.run_all(slow(i) for i in range(5))) == [0, 1, 2, 3, 4]


def test_run_all_cancels_the_rest_on_failure():
    finished = []

    async def item(i):
        if i == 0:
            raise ValueError("boom")
        await asyncio.sleep(1)
        finished.append(i)

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(pipeline.run_all(item(i) for i in range(3)))
    assert finished == []


def rate_limit_error(delay="0s"):
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}})


def flaky(*failures):
    calls = []

    async def call():
        calls.append(1)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return "ok"

    return call, calls


def test_daily_quota_fails_at_once(monkeypatch):
    call, calls = flaky(daily_quota_error())
    with pytest.raises(pipeline.QuotaExhausted) as info:
        asyncio.run(pipeline.retry(call, retries=5, label="x"))
    assert len(calls) == 1
    assert (info.value.model, info.value.per_day, info.value.wait_seconds) == ("gemini-3.8-flash", True, 34463)


def test_short_rate_limits_are_waited_out(monkeypatch):
    monkeypatch.setattr(pipeline, "RATE_LIMIT_PADDING", 0)
    call, calls = flaky(rate_limit_error(), rate_limit_error())
    assert asyncio.run(pipeline.retry(call, retries=0, label="x")) == "ok"
    assert len(calls) == 3


def test_long_rate_limit_counts_as_used_up(monkeypatch):
    call, calls = flaky(rate_limit_error("600s"))
    with pytest.raises(pipeline.QuotaExhausted) as info:
        asyncio.run(pipeline.retry(call, retries=5, label="x"))
    assert not info.value.per_day and len(calls) == 1


def test_friendly_errors():
    hint = pipeline.friendly_error(pipeline.QuotaExhausted("gemini-3.8-flash", 34463, True, ValueError()))
    assert "gemini-3.8-flash" in hint and "۹ ساعت و ۳۴ دقیقه" in hint and "روزانه" in hint
    assert "۱۱ دقیقه" in pipeline.friendly_error(pipeline.QuotaExhausted(None, 600, False, ValueError()))
    assert "مدل پیدا نشد" in pipeline.friendly_error(errors.ClientError(404, {"error": {"message": "nope"}}))
    assert pipeline.friendly_error(RuntimeError("boom")) is None


def test_auth_errors_are_not_retried():
    assert not pipeline._retryable(errors.ClientError(403, {"error": {"message": "bad key"}}))
    assert pipeline._retryable(errors.APIError(1011, "internal"))


@needs_ffmpeg
def test_transcripts_are_saved(tmp_path):
    job = Job(make_wav(tmp_path / "talk.wav", 3), tmp_path / "talk.fa.wav")
    run_file(FakeClient(text=True), job, save_text=True)
    assert (tmp_path / "talk.fa.txt").read_text(encoding="utf-8").startswith("سلام")
    assert (tmp_path / "talk.source.txt").read_text(encoding="utf-8").startswith("hello")


@needs_ffmpeg
def test_cli_skips_finished_files_and_reports_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RETRY_BASE_SECONDS", 0.01)
    make_wav(tmp_path / "in" / "done.wav", 3)
    make_wav(tmp_path / "in" / "new.wav", 3)
    (tmp_path / "in" / "broken.mp3").write_text("not audio")
    out = tmp_path / "out"
    out.mkdir()
    (out / "done.fa.wav").write_bytes(b"keep me")

    args = cli.parse_args([str(tmp_path / "in"), "-o", str(out), "--mode", "live", "--pace", "100"])
    client = FakeClient(on_end="close")
    assert asyncio.run(cli.run(args, client)) == 1  # broken.mp3 fails to decode

    assert (out / "done.fa.wav").read_bytes() == b"keep me"
    assert wav_frames(out / "new.fa.wav") > 0
    assert len(client.sessions) == 1


@needs_ffmpeg
def test_cli_quality_mode(tmp_path):
    make_wav(tmp_path / "talk.wav", 3)
    out = tmp_path / "out"
    args = cli.parse_args([str(tmp_path / "talk.wav"), "-o", str(out), "--voice", "Puck", "--style", "casual"])
    assert (args.mode, args.jobs) == ("quality", 4)
    client = FakeClient()
    assert asyncio.run(cli.run(args, client)) == 0
    assert wav_frames(out / "talk.fa.wav") > 0
    assert (out / "talk.fa.txt").read_text(encoding="utf-8").startswith("ترجمه")
    assert client.models.of("speak")[0].config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck"
    assert not client.sessions


@needs_ffmpeg
def test_cli_quality_mode_resumes_after_quota(tmp_path):
    make_wav(tmp_path / "talk.wav", 3)
    out = tmp_path / "out"
    args = cli.parse_args([str(tmp_path / "talk.wav"), "-o", str(out)])
    first = FakeClient(models=FakeModels(quota={"speak": 0}))
    assert asyncio.run(cli.run(args, first)) == 1
    assert (out / ".talk.fa.wav.parts").is_dir()

    second = FakeClient()
    assert asyncio.run(cli.run(args, second)) == 0
    assert not second.models.of("transcribe") and not second.models.of("translate")
    assert wav_frames(out / "talk.fa.wav") > 0
    assert not (out / ".talk.fa.wav.parts").exists()


def test_cli_needs_an_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert cli.main([str(tmp_path)]) == 2
