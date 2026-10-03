"""Streams one audio segment through Gemini Live Translate and collects the translated speech."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

from google.genai import errors, types

from .audio import INPUT_RATE, OUTPUT_RATE, SAMPLE_WIDTH, silence

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-live-translate-preview"

CHUNK_SECONDS = 0.1
# Sent after the real audio so the model hears the last sentence end before the stream closes.
TRAILING_SILENCE_SECONDS = 1.0
INPUT_MIME = f"audio/pcm;rate={INPUT_RATE}"
NORMAL_CLOSURE = 1000


@dataclass
class SegmentResult:
    audio: bytes  # 16-bit mono PCM
    rate: int
    translated_text: str = ""
    source_text: str = ""


def build_config(target_language: str, with_text: bool = False) -> types.LiveConnectConfig:
    extra = {}
    if with_text:
        extra["input_audio_transcription"] = types.AudioTranscriptionConfig()
        extra["output_audio_transcription"] = types.AudioTranscriptionConfig()
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        translation_config=types.TranslationConfig(target_language_code=target_language),
        **extra,
    )


def _rate_from_mime(mime_type: str | None, default: int) -> int:
    match = re.search(r"rate=(\d+)", mime_type or "")
    return int(match.group(1)) if match else default


@dataclass
class _Collector:
    chunks: list[bytes] = field(default_factory=list)
    rate: int = OUTPUT_RATE
    translated: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    last_activity: float = field(default_factory=time.monotonic)

    async def run(self, session) -> None:
        try:
            while True:
                # receive() stops at the end of each model turn, so keep re-entering it.
                async for message in session.receive():
                    self.handle(message)
        except errors.APIError as e:
            if e.code != NORMAL_CLOSURE:
                raise

    def handle(self, message: types.LiveServerMessage) -> None:
        if message.go_away is not None:
            log.debug("server will close the session soon (time left: %s)", message.go_away.time_left)
        content = message.server_content
        if content is None:
            return
        self.last_activity = time.monotonic()
        if content.model_turn and content.model_turn.parts:
            for part in content.model_turn.parts:
                blob = part.inline_data
                if blob and blob.data:
                    self.rate = _rate_from_mime(blob.mime_type, self.rate)
                    self.chunks.append(blob.data)
        if content.output_transcription and content.output_transcription.text:
            self.translated.append(content.output_transcription.text)
        if content.input_transcription and content.input_transcription.text:
            self.source.append(content.input_transcription.text)

    def result(self) -> SegmentResult:
        return SegmentResult(
            audio=b"".join(self.chunks),
            rate=self.rate,
            translated_text="".join(self.translated).strip(),
            source_text="".join(self.source).strip(),
        )


class Translator:
    """Translates PCM audio (16-bit mono, 16 kHz) with one Live API session per call."""

    def __init__(
        self,
        client,
        target_language: str = "fa",
        model: str = DEFAULT_MODEL,
        *,
        pace: float = 1.0,
        idle_timeout: float = 6.0,
        max_tail: float = 120.0,
        with_text: bool = False,
    ):
        if pace <= 0:
            raise ValueError("pace must be positive")
        self.client = client
        self.model = model
        self.config = build_config(target_language, with_text)
        self.pace = pace
        self.idle_timeout = idle_timeout
        self.max_tail = max_tail

    async def translate(self, pcm: bytes) -> SegmentResult:
        collector = _Collector()
        async with self.client.aio.live.connect(model=self.model, config=self.config) as session:
            receiver = asyncio.create_task(collector.run(session))
            try:
                await self._send(session, pcm, receiver)
                await self._wait_for_tail(collector, receiver)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        return collector.result()

    async def _send(self, session, pcm: bytes, receiver: asyncio.Task) -> None:
        pcm = pcm + silence(TRAILING_SILENCE_SECONDS, INPUT_RATE)
        chunk = int(INPUT_RATE * CHUNK_SECONDS) * SAMPLE_WIDTH
        bytes_per_second = INPUT_RATE * SAMPLE_WIDTH * self.pace
        start = time.monotonic()
        for offset in range(0, len(pcm), chunk):
            if receiver.done():
                receiver.result()  # re-raises the receiver's error, if it had one
                raise RuntimeError("the server closed the session before all audio was sent")
            await session.send_realtime_input(
                audio=types.Blob(data=pcm[offset : offset + chunk], mime_type=INPUT_MIME)
            )
            # Stream at (a multiple of) real time: the model is built for live speech.
            delay = start + (offset + chunk) / bytes_per_second - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        await session.send_realtime_input(audio_stream_end=True)

    async def _wait_for_tail(self, collector: _Collector, receiver: asyncio.Task) -> None:
        """Wait until the translation stops arriving after the input has ended."""
        sent_at = time.monotonic()
        while True:
            if receiver.done():
                receiver.result()  # a clean close means the server has nothing more to say
                return
            now = time.monotonic()
            if now - max(collector.last_activity, sent_at) >= self.idle_timeout:
                return
            if now - sent_at >= self.max_tail:
                log.warning("translation was still arriving after %.0fs; the end may be cut off", self.max_tail)
                return
            await asyncio.sleep(0.2)
