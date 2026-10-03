import asyncio
import shutil
import time
import wave
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from livedub.web import InvalidKey, WebApp, load_env_key, save_env_key
from tests.fakes import FakeClient, FakeModels
from tests.test_pipeline import make_wav

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make_app(tmp_path, api_key="test-key", fake=None, check=None, **options):
    fake = fake or FakeClient(on_end="close")

    async def accept(client, model):
        return None

    app = WebApp(
        tmp_path / "data",
        api_key=api_key,
        make_client=lambda key: fake,
        check_key=check or accept,
        env_file=tmp_path / ".env",
        translator_options={"pace": options.pop("pace", 100), "idle_timeout": 0.2},
        **options,
    )
    return app, fake


def run(app, scenario):
    async def go():
        async with TestClient(TestServer(app.build())) as client:
            return await scenario(client)

    return asyncio.run(go())


def upload_form(*paths, lang="fa", save_text=False, mode="live", **fields):
    form = aiohttp.FormData(quote_fields=False)  # send UTF-8 file names raw, like browsers do
    form.add_field("lang", lang)
    form.add_field("mode", mode)
    form.add_field("save_text", "1" if save_text else "0")
    for name, value in fields.items():
        form.add_field(name, value)
    for path in paths:
        form.add_field("file", Path(path).read_bytes(), filename=Path(path).name)
    return form


async def wait_for(client, job_id, statuses=("done", "error", "cancelled"), timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = await (await client.get("/api/jobs")).json()
        job = next(j for j in jobs if j["id"] == job_id)
        if job["status"] in statuses:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job stuck: {job}")


def test_page_and_status(tmp_path):
    app, _ = make_app(tmp_path, api_key=None)

    async def scenario(client):
        page = await client.get("/")
        assert page.status == 200 and "livedub" in await page.text()
        status = await (await client.get("/api/status")).json()
        assert status["has_key"] is False and status["key_hint"] is None
        assert status["models"]["tts"] == "gemini-3.8-flash-tts"
        assert {"name": "Kore", "gender": "female", "character": "firm"} in status["voices"]
        assert (await client.get("/api/voice-sample?voice=Kore")).status == 400
        refused = await client.post("/api/jobs", data=upload_form())
        assert refused.status == 400

    run(app, scenario)


def test_setting_a_key(tmp_path):
    async def check(client, model):
        if client.key == "bad":
            raise InvalidKey("rejected")
        return "model missing" if client.key == "warn" else None

    def make_client(key):
        fake = FakeClient()
        fake.key = key
        return fake

    app = WebApp(tmp_path / "data", api_key=None, make_client=make_client, check_key=check, env_file=tmp_path / ".env")

    async def scenario(client):
        bad = await client.post("/api/key", json={"key": "bad"})
        assert bad.status == 400 and (await bad.json())["error"] == "rejected"
        assert not (tmp_path / ".env").exists()
        assert (await client.post("/api/key", json={"key": "  "})).status == 400

        ok = await client.post("/api/key", json={"key": "AIzaGood1234"})
        assert await ok.json() == {"ok": True, "warning": None}
        status = await (await client.get("/api/status")).json()
        assert status["has_key"] and status["key_hint"] == "1234"
        assert load_env_key(tmp_path / ".env") == "AIzaGood1234"

        warned = await client.post("/api/key", json={"key": "warn"})
        assert (await warned.json())["warning"] == "model missing"

    run(app, scenario)


def test_env_file_keeps_other_settings(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nGEMINI_API_KEY=old\n")
    save_env_key(env, "new")
    assert env.read_text() == "OTHER=1\nGEMINI_API_KEY=new\n"
    assert load_env_key(env) == "new"
    assert load_env_key(tmp_path / "missing") is None


@needs_ffmpeg
def test_upload_translate_and_download(tmp_path):
    app, fake = make_app(tmp_path, fake=FakeClient(on_end="close", text=True))
    src = make_wav(tmp_path / "سخنرانی.wav", 3)

    async def scenario(client):
        created = await client.post("/api/jobs", data=upload_form(src, save_text=True))
        assert created.status == 201
        [job] = await created.json()
        assert job["name"] == "سخنرانی.wav" and job["status"] == "running"

        job = await wait_for(client, job["id"])
        assert job["status"] == "done", job["error"]
        assert job["duration"] == 3.0 and job["fraction"] == 1.0

        audio = await client.get(job["audio_url"])
        body = await audio.read()
        assert audio.status == 200 and body[:4] == b"RIFF"
        out = tmp_path / "out.wav"
        out.write_bytes(body)
        with wave.open(str(out)) as w:
            assert w.getframerate() == 24000 and w.getnframes() > 0

        download = await client.get(job["audio_url"] + "?download=1")
        assert "filename*=UTF-8''%D8%B3" in download.headers["Content-Disposition"]
        text = await client.get(job["text_url"])
        assert (await text.text()).startswith("سلام")
        source = await client.get(job["source_url"])
        assert await source.read() == src.read_bytes()

    run(app, scenario)
    assert fake.configs[0][1].translation_config.target_language_code == "fa"


@needs_ffmpeg
def test_failed_translation_is_reported(tmp_path):
    app, _ = make_app(tmp_path, fake=FakeClient(fail_after_chunks=1), retries=0)
    src = make_wav(tmp_path / "talk.wav", 2)

    async def scenario(client):
        [job] = await (await client.post("/api/jobs", data=upload_form(src))).json()
        job = await wait_for(client, job["id"])
        assert job["status"] == "error" and "1011" in job["error"]
        assert job["audio_url"] is None

    run(app, scenario)


def test_rejects_unsupported_files_and_languages(tmp_path):
    app, _ = make_app(tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"x")

    async def scenario(client):
        bad_type = await client.post("/api/jobs", data=upload_form(notes))
        assert bad_type.status == 400 and ".txt" in (await bad_type.json())["error"]
        bad_lang = await client.post("/api/jobs", data=upload_form(clip, lang="fa;rm -rf"))
        assert bad_lang.status == 400
        for field in ({"mode": "fast"}, {"style": "loud"}, {"voice": "../x"}):
            assert (await client.post("/api/jobs", data=upload_form(clip, **field))).status == 400
        assert (await client.post("/api/jobs", data=upload_form())).status == 400
        assert await (await client.get("/api/jobs")).json() == []

    run(app, scenario)
    assert not any((tmp_path / "data").glob("*/*"))


@needs_ffmpeg
def test_cancel_and_delete(tmp_path):
    app, _ = make_app(tmp_path, fake=FakeClient(), pace=1)
    src = make_wav(tmp_path / "long.wav", 60)

    async def scenario(client):
        [job] = await (await client.post("/api/jobs", data=upload_form(src))).json()
        await asyncio.sleep(0.5)
        cancelled = await (await client.post(f"/api/jobs/{job['id']}/cancel")).json()
        assert cancelled["status"] == "cancelled"
        folder = tmp_path / "data" / job["id"]
        assert folder.exists()
        assert (await client.delete(f"/api/jobs/{job['id']}")).status == 200
        assert not folder.exists()
        assert (await client.get(f"/api/jobs/{job['id']}/audio")).status == 404

    run(app, scenario)


@needs_ffmpeg
def test_quality_job(tmp_path, monkeypatch):
    monkeypatch.setattr("livedub.pipeline.RETRY_BASE_SECONDS", 0.01)
    app, fake = make_app(tmp_path)
    src = make_wav(tmp_path / "talk.wav", 3)

    async def scenario(client):
        form = upload_form(src, mode="quality", voice="Charon", style="casual")
        [job] = await (await client.post("/api/jobs", data=form)).json()
        assert (job["mode"], job["voice"], job["style"]) == ("quality", "Charon", "casual")
        job = await wait_for(client, job["id"])
        assert job["status"] == "done", job["error"]
        assert (job["stage"], job["done"]) == ("speak", job["total"])
        assert job["text_url"]  # quality mode always keeps the transcript
        text = await (await client.get(job["text_url"])).text()
        assert text.startswith("ترجمه «Part 1 sentence 1.»")
        audio = await client.get(job["audio_url"])
        assert (await audio.read())[:4] == b"RIFF"

    run(app, scenario)
    speak = fake.models.of("speak")
    assert speak[0].config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Charon"
    assert not fake.sessions


def test_voice_sample_is_cached(tmp_path):
    app, fake = make_app(tmp_path)

    async def scenario(client):
        for _ in range(2):
            res = await client.get("/api/voice-sample?voice=Sulafat&lang=fa&style=formal")
            assert res.status == 200 and res.content_type == "audio/wav"
            assert (await res.read())[:4] == b"RIFF"
        assert (await client.get("/api/voice-sample?voice=Sulafat&lang=fa&style=nope")).status == 400

    run(app, scenario)
    assert len(fake.models.of("speak")) == 1


def test_voice_sample_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("livedub.pipeline.RETRY_BASE_SECONDS", 0.01)
    app, _ = make_app(tmp_path, fake=FakeClient(models=FakeModels(failures={"speak": 5})), retries=1)

    async def scenario(client):
        res = await client.get("/api/voice-sample?voice=Kore&lang=fa&style=formal")
        assert res.status == 502 and "overloaded" in (await res.json())["error"]

    run(app, scenario)
