"""Local web interface: python -m livedub.web, then open http://localhost:8000"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import shutil
import sys
import time
import uuid
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from urllib.parse import quote

from aiohttp import web
from google.genai import errors

from .audio import MEDIA_EXTENSIONS
from .dubber import DEFAULT_VOICE, STYLES, TEXT_MODEL, TTS_MODEL, VOICES, Dubber, DubOptions
from .pipeline import Job, Progress, friendly_error, transcript_paths, translate_file
from .translator import DEFAULT_MODEL, Translator

log = logging.getLogger("livedub.web")

STATIC_DIR = Path(__file__).with_name("static")
SEGMENT_SECONDS = 300
MAX_UPLOAD_BYTES = 4 * 1024**3
LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
VOICE_RE = re.compile(r"^[A-Za-z]{2,30}$")
MODES = ("quality", "live")


class InvalidKey(Exception):
    pass


async def check_key(client, model: str) -> str | None:
    """Raise InvalidKey if Google rejects the key; return a warning if it cannot be fully verified."""
    try:
        await client.aio.models.get(model=model)
    except errors.ClientError as e:
        if e.code == 404:
            return f"کلید درسته، ولی مدل {model} پیدا نشد؛ شاید اسمش عوض شده."
        if e.code in (400, 401, 403):
            raise InvalidKey("گوگل این کلید رو قبول نکرد. کلید رو دوباره کپی کنید.") from e
        return f"کلید ذخیره شد ولی بررسی نشد: {e}"
    except Exception as e:
        return f"کلید ذخیره شد ولی بررسی نشد (اینترنت؟): {e}"
    return None


def load_env_key(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        name, sep, value = line.partition("=")
        if sep and name.strip() == "GEMINI_API_KEY":
            return value.strip().strip("\"'") or None
    return None


def save_env_key(path: Path, key: str) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    lines = [line for line in lines if line.partition("=")[0].strip() != "GEMINI_API_KEY"]
    lines.append(f"GEMINI_API_KEY={key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)


def _disposition(kind: str, filename: str) -> str:
    fallback = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return f"{kind}; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"


@dataclass
class WebJob:
    id: str
    name: str
    lang: str
    save_text: bool
    dir: Path
    source: Path
    mode: str = "quality"  # quality | live
    voice: str = DEFAULT_VOICE
    style: str = "formal"
    created: float = field(default_factory=time.time)
    started: float = field(default_factory=time.time)
    status: str = "running"  # running | done | error | cancelled
    error: str = ""
    hint: str | None = None  # a Persian explanation of the error, when there is one
    progress: Progress = field(default_factory=Progress)
    finished: float | None = None
    task: asyncio.Task | None = None

    @property
    def output(self) -> Path:
        return self.dir / "output.wav"

    @property
    def work_dir(self) -> Path:
        return self.dir / "work"

    @property
    def text_file(self) -> Path:
        return transcript_paths(self.output)[0]

    @property
    def download_stem(self) -> str:
        return f"{Path(self.name).stem}.{self.lang}"

    def to_json(self) -> dict:
        done = self.status == "done"
        return {
            "id": self.id,
            "name": self.name,
            "lang": self.lang,
            "mode": self.mode,
            "voice": self.voice if self.mode == "quality" else None,
            "style": self.style if self.mode == "quality" else None,
            "status": self.status,
            "error": self.error,
            "hint": self.hint,
            "duration": round(self.progress.duration, 1),
            "stage": self.progress.stage,
            "done": self.progress.done,
            "total": self.progress.total,
            "fraction": round(self.progress.fraction, 4),
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "audio_url": f"/api/jobs/{self.id}/audio" if done else None,
            "text_url": f"/api/jobs/{self.id}/text" if done and self.text_file.exists() else None,
            "source_url": f"/api/jobs/{self.id}/source",
        }


class WebApp:
    def __init__(
        self,
        data_dir: Path,
        *,
        api_key: str | None,
        make_client: Callable[[str], object],
        check_key: Callable[[object, str], Awaitable[str | None]] = check_key,
        env_file: Path | None = None,
        live_model: str = DEFAULT_MODEL,
        text_model: str = TEXT_MODEL,
        tts_model: str = TTS_MODEL,
        parallel: int = 3,
        retries: int = 2,
        translator_options: dict | None = None,
    ):
        self.data_dir = data_dir
        self.make_client = make_client
        self.check_key = check_key
        self.env_file = env_file
        self.live_model = live_model
        self.text_model = text_model
        self.tts_model = tts_model
        self.parallel = parallel
        self.retries = retries
        self.translator_options = translator_options or {}
        self.api_key = api_key
        self.client = make_client(api_key) if api_key else None
        self.sessions = asyncio.Semaphore(parallel)
        self.jobs: dict[str, WebJob] = {}
        self.samples: dict[tuple[str, str, str], bytes] = {}  # voice samples by (voice, lang, style)

    def build(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.get("/", self.index),
            web.get("/api/status", self.status),
            web.post("/api/key", self.set_key),
            web.get("/api/voice-sample", self.voice_sample),
            web.get("/api/jobs", self.list_jobs),
            web.post("/api/jobs", self.create_jobs),
            web.post("/api/jobs/{id}/cancel", self.cancel_job),
            web.post("/api/jobs/{id}/resume", self.resume_job),
            web.delete("/api/jobs/{id}", self.delete_job),
            web.get("/api/jobs/{id}/audio", self.job_audio),
            web.get("/api/jobs/{id}/text", self.job_text),
            web.get("/api/jobs/{id}/source", self.job_source),
        ])
        app.on_shutdown.append(self.shutdown)
        return app

    # --- pages and settings ---

    async def index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    async def status(self, request: web.Request) -> web.Response:
        return web.json_response({
            "has_key": self.client is not None,
            "key_hint": self.api_key[-4:] if self.api_key else None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "models": {"live": self.live_model, "text": self.text_model, "tts": self.tts_model},
            "voices": [{"name": n, "gender": g, "character": c} for n, (g, c) in VOICES.items()],
            "default_voice": DEFAULT_VOICE,
            "styles": list(STYLES),
        })

    async def set_key(self, request: web.Request) -> web.Response:
        try:
            key = str((await request.json()).get("key", "")).strip()
        except ValueError:
            key = ""
        if not key:
            return _error(400, "کلید خالیه.")
        client = self.make_client(key)
        try:
            warning = await self.check_key(client, self.text_model)
        except InvalidKey as e:
            return _error(400, str(e))
        self.api_key, self.client = key, client
        self.samples.clear()
        if self.env_file:
            save_env_key(self.env_file, key)
        log.info("API key updated")
        return web.json_response({"ok": True, "warning": warning})

    # --- jobs ---

    async def list_jobs(self, request: web.Request) -> web.Response:
        jobs = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)
        return web.json_response([j.to_json() for j in jobs])

    async def create_jobs(self, request: web.Request) -> web.Response:
        if self.client is None:
            return _error(400, "اول کلید Gemini API رو وارد کنید.")
        if not request.content_type.startswith("multipart/"):
            return _error(400, "هیچ فایلی انتخاب نشده.")
        fields: dict[str, str] = {}
        uploads: list[tuple[str, Path, Path]] = []
        try:
            reader = await request.multipart()
            async for part in reader:
                if part.name == "file" and part.filename:
                    uploads.append(await self._store_upload(part))
                elif part.name:
                    fields[part.name] = (await part.text()).strip()
            settings = self._job_settings(fields)
        except ValueError as e:
            for _, folder, _ in uploads:
                shutil.rmtree(folder, ignore_errors=True)
            return _error(400, str(e))
        if not uploads:
            return _error(400, "هیچ فایلی انتخاب نشده.")

        created = []
        for name, folder, source in uploads:
            job = WebJob(folder.name, name, source=source, dir=folder, **settings)
            self.jobs[job.id] = job
            self._start(job)
            created.append(job.to_json())
        return web.json_response(created, status=201)

    @staticmethod
    def _job_settings(fields: dict[str, str]) -> dict:
        settings = {
            "lang": fields.get("lang") or "fa",
            "mode": fields.get("mode") or "quality",
            "voice": fields.get("voice") or DEFAULT_VOICE,
            "style": fields.get("style") or "formal",
            "save_text": fields.get("save_text", "").lower() in ("1", "true", "on"),
        }
        if not LANG_RE.match(settings["lang"]):
            raise ValueError(f"کد زبان نامعتبره: {settings['lang']}")
        if settings["mode"] not in MODES:
            raise ValueError(f"حالت نامعتبره: {settings['mode']}")
        if not VOICE_RE.match(settings["voice"]):
            raise ValueError(f"اسم صدا نامعتبره: {settings['voice']}")
        if settings["style"] not in STYLES:
            raise ValueError(f"لحن نامعتبره: {settings['style']}")
        return settings

    async def _store_upload(self, part) -> tuple[str, Path, Path]:
        name = PureWindowsPath(part.filename).name  # strips folders from both / and \ paths
        suffix = Path(name).suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            raise ValueError(f"فرمت {suffix or '(بدون پسوند)'} پشتیبانی نمیشه: {name}")
        folder = self.data_dir / uuid.uuid4().hex[:12]
        folder.mkdir(parents=True)
        source = folder / f"input{suffix}"
        size = 0
        try:
            with open(source, "wb") as f:
                while chunk := await part.read_chunk(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise ValueError(f"{name} از ۴ گیگابایت بزرگ‌تره.")
                    f.write(chunk)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        log.info("%s: uploaded (%.1f MB)", name, size / 1e6)
        return name, folder, source

    def _dubber(self, lang: str, voice: str, style: str) -> Dubber:
        options = DubOptions(
            language=lang,
            voice=voice,
            style=style,
            text_model=self.text_model,
            tts_model=self.tts_model,
            parallel=self.parallel,
            retries=self.retries,
        )
        return Dubber(self.client, options)

    async def _translate(self, job: WebJob) -> None:
        if job.mode == "quality":
            dubber = self._dubber(job.lang, job.voice, job.style)
            await dubber.dub_file(Job(job.source, job.output), job.progress, work_dir=job.work_dir)
            shutil.rmtree(job.work_dir, ignore_errors=True)
            return
        translator = Translator(
            self.client, job.lang, self.live_model, with_text=job.save_text, **self.translator_options
        )
        await translate_file(
            Job(job.source, job.output),
            translator,
            segment_seconds=SEGMENT_SECONDS,
            sessions=self.sessions,
            retries=self.retries,
            save_text=job.save_text,
            progress=job.progress,
        )

    async def _run(self, job: WebJob) -> None:
        try:
            await self._translate(job)
            job.status = "done"
        except asyncio.CancelledError:
            job.status = "cancelled"
            log.info("%s: cancelled", job.name)
        except Exception as e:
            job.status, job.error, job.hint = "error", f"{type(e).__name__}: {e}", friendly_error(e)
            log.error("%s: failed: %s", job.name, job.error)
        finally:
            job.finished = time.time()

    def _start(self, job: WebJob) -> None:
        job.status, job.error, job.hint, job.finished = "running", "", None, None
        job.started = time.time()
        job.progress = Progress()
        job.task = asyncio.create_task(self._run(job))

    def _job(self, request: web.Request) -> WebJob:
        job = self.jobs.get(request.match_info["id"])
        if job is None:
            raise web.HTTPNotFound()
        return job

    async def voice_sample(self, request: web.Request) -> web.Response:
        if self.client is None:
            return _error(400, "اول کلید Gemini API رو وارد کنید.")
        q = request.query
        try:
            settings = self._job_settings({"lang": q.get("lang", ""), "voice": q.get("voice", ""), "style": q.get("style", "")})
        except ValueError as e:
            return _error(400, str(e))
        key = (settings["voice"], settings["lang"], settings["style"])
        if key not in self.samples:
            try:
                dubber = self._dubber(settings["lang"], settings["voice"], settings["style"])
                self.samples[key] = await dubber.preview()
            except Exception as e:
                log.error("voice sample failed: %s: %s", type(e).__name__, e)
                return _error(502, friendly_error(e) or f"ساخت نمونه صدا ناموفق بود: {e}")
        return web.Response(body=self.samples[key], content_type="audio/wav")

    async def resume_job(self, request: web.Request) -> web.Response:
        job = self._job(request)
        if job.status not in ("error", "cancelled"):
            return _error(409, "این کار در حال اجرا یا تمام‌شده‌ست.")
        if self.client is None:
            return _error(400, "اول کلید Gemini API رو وارد کنید.")
        self._start(job)
        return web.json_response(job.to_json())

    async def cancel_job(self, request: web.Request) -> web.Response:
        job = self._job(request)
        if job.task and not job.task.done():
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        return web.json_response(job.to_json())

    async def delete_job(self, request: web.Request) -> web.Response:
        job = self._job(request)
        if job.task and not job.task.done():
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        del self.jobs[job.id]
        shutil.rmtree(job.dir, ignore_errors=True)
        return web.json_response({"ok": True})

    async def job_audio(self, request: web.Request) -> web.StreamResponse:
        job = self._job(request)
        if job.status != "done":
            raise web.HTTPNotFound()
        headers = {}
        if "download" in request.query:
            headers["Content-Disposition"] = _disposition("attachment", f"{job.download_stem}.wav")
        return web.FileResponse(job.output, headers=headers)

    async def job_text(self, request: web.Request) -> web.Response:
        job = self._job(request)
        if job.status != "done" or not job.text_file.exists():
            raise web.HTTPNotFound()
        headers = {}
        if "download" in request.query:
            headers["Content-Disposition"] = _disposition("attachment", f"{job.download_stem}.txt")
        text = job.text_file.read_text(encoding="utf-8")
        return web.Response(text=text, content_type="text/plain", charset="utf-8", headers=headers)

    async def job_source(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(self._job(request).source)

    async def shutdown(self, app: web.Application) -> None:
        running = [j.task for j in self.jobs.values() if j.task and not j.task.done()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m livedub.web", description="Run the livedub web page on localhost.")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1, this computer only)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--data-dir", type=Path, default=Path("web_data"), help="where uploads and results are kept")
    parser.add_argument("-j", "--parallel", type=int, default=3, help="requests or live sessions run at once (default: 3)")
    parser.add_argument("--text-model", default=TEXT_MODEL, help=f"transcription/translation model (default: {TEXT_MODEL})")
    parser.add_argument("--tts-model", default=TTS_MODEL, help=f"voice model (default: {TTS_MODEL})")
    parser.add_argument("--live-model", default=DEFAULT_MODEL, help=f"live mode model (default: {DEFAULT_MODEL})")
    parser.add_argument("--open", action="store_true", help="open the page in the browser once the server is up")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 11):
        print("livedub needs Python 3.11 or newer.", file=sys.stderr)
        return 1
    from google import genai

    from .cli import setup_logging

    args = parse_args(argv)
    setup_logging(args.verbose)
    env_file = Path(".env")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or load_env_key(env_file)

    def make_client(key: str):
        return genai.Client(api_key=key, http_options={"api_version": "v1beta"})

    app = WebApp(args.data_dir, api_key=api_key, make_client=make_client, env_file=env_file,
                 live_model=args.live_model, text_model=args.text_model, tts_model=args.tts_model,
                 parallel=args.parallel).build()
    url = f"http://{'localhost' if args.host in ('127.0.0.1', 'localhost') else args.host}:{args.port}"

    async def announce(_app: web.Application) -> None:
        log.info("livedub is running at %s (press Ctrl+C to stop)", url)
        if args.open:
            asyncio.get_running_loop().call_later(0.5, webbrowser.open, url)

    app.on_startup.append(announce)
    try:
        web.run_app(app, host=args.host, port=args.port, print=None)
    except OSError as e:
        log.error("could not listen on port %d (%s); try another one with --port 8001", args.port, e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
