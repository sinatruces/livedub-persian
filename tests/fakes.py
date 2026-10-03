"""A stand-in for genai.Client that behaves like a Live Translate session, for offline tests."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

from google.genai import errors, types

# One "translated" chunk is 0.1 s of a constant non-zero sample at 24 kHz.
REPLY_CHUNK = (1000).to_bytes(2, "little", signed=True) * 2400


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


class FakeClient:
    """Hands out a new FakeSession per connect(); the first `fail_first` of them break mid-stream."""

    def __init__(self, fail_first: int = 0, **session_kwargs):
        self.fail_first = fail_first
        self.session_kwargs = session_kwargs
        self.sessions: list[FakeSession] = []
        self.configs = []
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect))

    @contextlib.asynccontextmanager
    async def _connect(self, *, model, config):
        self.configs.append((model, config))
        kwargs = dict(self.session_kwargs)
        if len(self.sessions) < self.fail_first:
            kwargs["fail_after_chunks"] = 3
        session = FakeSession(**kwargs)
        self.sessions.append(session)
        yield session
