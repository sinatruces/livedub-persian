"""A stand-in for genai.Client (Live Translate sessions and generate_content), for offline tests."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from types import SimpleNamespace

from google.genai import errors, types

from livedub.audio import wav_bytes

# One "translated" chunk is 0.1 s of a constant non-zero sample at 24 kHz.
REPLY_CHUNK = (1000).to_bytes(2, "little", signed=True) * 2400
# Fake speech is 50 ms (1200 samples at 24 kHz) per character, about how fast people talk.
SAMPLES_PER_CHAR = 1200


def audio_message(data: bytes, rate: int = 24000) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(
                role="model",
                parts=[types.Part(inline_data=types.Blob(data=data, mime_type=f"audio/pcm;rate={rate}"))],
            )
        )
    )


def text_message(translated: str, source: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            output_transcription=types.Transcription(text=translated),
            input_transcription=types.Transcription(text=source),
        )
    )


class FakeSession:
    """Answers every second of received audio with one reply chunk, then a turn_complete.

    `on_end` decides what happens after audio_stream_end: "idle" keeps the socket open and
    silent, "close" closes it normally, and "error" fails it like an internal server error.
    """

    def __init__(self, on_end: str = "idle", fail_after_chunks: int | None = None, text: bool = False):
        self.on_end = on_end
        self.fail_after_chunks = fail_after_chunks
        self.text = text
        self.sent: list[bytes] = []
        self.stream_ended = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def send_realtime_input(self, *, audio=None, audio_stream_end=None):
        if audio_stream_end:
            self.stream_ended = True
            await self._queue.put(audio_message(REPLY_CHUNK))
            await self._queue.put(types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True)))
            await self._queue.put(self.on_end)
            return
        assert audio.mime_type == "audio/pcm;rate=16000"
        self.sent.append(audio.data)
        if self.fail_after_chunks is not None and len(self.sent) >= self.fail_after_chunks:
            await self._queue.put("error")
        elif len(self.sent) % 10 == 0:
            await self._queue.put(audio_message(REPLY_CHUNK))
            if self.text:
                await self._queue.put(text_message("سلام ", "hello "))

    async def receive(self):
        while True:
            item = await self._queue.get()
            if item == "idle":
                await asyncio.Event().wait()
            if item == "close":
                errors.APIError.raise_error(1000, "", None)
            if item == "error":
                errors.APIError.raise_error(1011, "Internal error encountered.", None)
            yield item
            if item.server_content and item.server_content.turn_complete:
                return


def daily_quota_error(model: str = "gemini-3.8-flash", retry_delay: str = "34463s") -> errors.ClientError:
    """The 429 the Gemini API sends when a free-tier daily quota is used up."""
    return errors.ClientError(429, {"error": {
        "code": 429,
        "message": f"You exceeded your current quota... limit: 20, model: {model}",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                "quotaDimensions": {"location": "global", "model": model},
                "quotaValue": "20",
            }]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ],
    }})


def _response(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


class FakeModels:
    """Stands in for client.aio.models: transcribes, translates and speaks predictably.

    Transcription returns `paragraphs` sentences per audio part, labelled by a hash of the audio so
    the same audio always gives the same text; translation wraps each paragraph in «ترجمه …»; and
    speech is SAMPLES_PER_CHAR samples per character. `failures` makes the first N calls
    of a kind ("transcribe", "translate", "speak") fail like an overloaded server; `quota` lets N
    calls of a kind succeed and answers the rest with a used-up daily quota.
    """

    def __init__(self, paragraphs: int = 2, silent: bool = False, wav_tts: bool = False, failures=None, quota=None):
        self.paragraphs = paragraphs
        self.silent = silent
        self.wav_tts = wav_tts
        self.failures = dict(failures or {})
        self.quota = dict(quota or {})
        self.calls: list[SimpleNamespace] = []

    def of(self, kind: str) -> list[SimpleNamespace]:
        return [c for c in self.calls if c.kind == kind]

    async def generate_content(self, *, model, contents, config=None):
        if config is not None and config.response_modalities == ["AUDIO"]:
            kind = "speak"
        elif isinstance(contents, list):
            kind = "transcribe"
        else:
            kind = "translate"
        self.calls.append(SimpleNamespace(kind=kind, model=model, contents=contents, config=config))
        await asyncio.sleep(0)
        if kind in self.quota:
            if self.quota[kind] <= 0:
                raise daily_quota_error(model)
            self.quota[kind] -= 1
        if self.failures.get(kind):
            self.failures[kind] -= 1
            raise errors.ServerError(503, {"error": {"message": "The model is overloaded."}})

        if kind == "transcribe":
            tag = hashlib.sha1(contents[0].inline_data.data).hexdigest()[:6]
            paragraphs = [] if self.silent else [f"Part {tag} sentence {i}." for i in range(1, self.paragraphs + 1)]
            return _response(types.Part(text=json.dumps({"paragraphs": paragraphs})))
        if kind == "translate":
            request = json.loads(contents)
            translated = [f"ترجمه «{p}»" for p in request["paragraphs"]]
            return _response(types.Part(text=json.dumps({"paragraphs": translated}, ensure_ascii=False)))

        text = contents.split("\n\n", 1)[1]  # drop the delivery direction line
        pcm = REPLY_CHUNK[:2] * (SAMPLES_PER_CHAR * len(text))
        if self.wav_tts:
            blob = types.Blob(data=wav_bytes(pcm, 24000), mime_type="audio/wav")
        else:
            blob = types.Blob(data=pcm, mime_type="audio/L16;codec=pcm;rate=24000")
        return _response(types.Part(inline_data=blob))


class FakeClient:
    """Hands out a new FakeSession per connect(); the first `fail_first` of them break mid-stream."""

    def __init__(self, fail_first: int = 0, models: FakeModels | None = None, **session_kwargs):
        self.fail_first = fail_first
        self.session_kwargs = session_kwargs
        self.sessions: list[FakeSession] = []
        self.configs = []
        self.models = models or FakeModels()
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect), models=self.models)

    @contextlib.asynccontextmanager
    async def _connect(self, *, model, config):
        self.configs.append((model, config))
        kwargs = dict(self.session_kwargs)
        if len(self.sessions) < self.fail_first:
            kwargs["fail_after_chunks"] = 3
        session = FakeSession(**kwargs)
        self.sessions.append(session)
        yield session
