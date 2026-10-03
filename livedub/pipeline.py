"""Shared plumbing for turning media files into translated WAV files, plus the live mode."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np
from google.genai import errors

from .audio import (
    INPUT_RATE,
    MEDIA_EXTENSIONS,
    SAMPLE_WIDTH,
    decode_to_pcm16,
    silence,
    split_points,
    write_wav,
)
from .translator import SegmentResult, Translator

log = logging.getLogger(__name__)
T = TypeVar("T")

# Pause inserted where two segments are joined back together.
SEGMENT_GAP_SECONDS = 0.3
# Retries wait 2, 4, 8... seconds.
RETRY_BASE_SECONDS = 2


@dataclass
class Job:
    source: Path
    output: Path


@dataclass
class Progress:
    """Live view of one file's translation, updated in place while it runs."""

    duration: float = 0.0  # seconds of input audio; 0 until the file is decoded
    stage: str = "prepare"  # prepare | live | transcribe | translate | speak
    done: int = 0  # steps finished in the current stage
    total: int = 0  # steps in the current stage
    streamed: dict[int, float] = field(default_factory=dict)  # live mode: seconds sent, per segment

    def start(self, stage: str, total: int) -> None:
        self.stage, self.done, self.total = stage, 0, total

    @property
    def fraction(self) -> float:
        """How far the current stage is, from 0 to 1."""
        if self.stage == "live":
            return min(1.0, sum(self.streamed.values()) / self.duration) if self.duration else 0.0
        return self.done / self.total if self.total else 0.0


def collect_jobs(inputs: list[Path], out_dir: Path, language: str) -> list[Job]:
    """Expand files and folders into jobs; folders are searched recursively and mirrored in out_dir."""
    jobs: list[Job] = []
    for item in inputs:
        if item.is_dir():
            for path in sorted(p for p in item.rglob("*") if p.is_file()):
                if path.suffix.lower() in MEDIA_EXTENSIONS:
                    rel = path.relative_to(item)
                    jobs.append(Job(path, out_dir / rel.parent / f"{rel.stem}.{language}.wav"))
        elif item.is_file():
            jobs.append(Job(item, out_dir / f"{item.stem}.{language}.wav"))
        else:
            raise FileNotFoundError(f"{item} does not exist")

    seen: dict[Path, Path] = {}
    for job in jobs:
        if job.output in seen:
            raise ValueError(f"{seen[job.output]} and {job.source} would both be written to {job.output}")
        seen[job.output] = job.source
    return jobs


def _retryable(exc: Exception) -> bool:
    # Bad requests, keys or model names will not fix themselves; quota errors (429) might.
    return not (isinstance(exc, errors.ClientError) and exc.code != 429)


async def retry(call: Callable[[], Awaitable[T]], retries: int, label: str) -> T:
    """Await `call()`, retrying transient failures with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            return await call()
        except Exception as e:
            if attempt == retries or not _retryable(e):
                raise
            wait = RETRY_BASE_SECONDS * 2**attempt
            log.warning("%s failed (%s: %s); retrying in %gs", label, type(e).__name__, e, wait)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def run_all(coros: Iterable[Awaitable[T]]) -> list[T]:
    """Run coroutines concurrently; on the first failure cancel the rest and raise that error."""
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(c) for c in coros]
    except ExceptionGroup as group:
        raise group.exceptions[0] from None
    return [t.result() for t in tasks]


def transcript_paths(output: Path) -> tuple[Path, Path]:
    """Where the translated and the source transcripts of `output` are saved."""
    return output.with_suffix(".txt"), output.with_name(output.stem.rsplit(".", 1)[0] + ".source.txt")


def write_transcripts(job: Job, translated: str, source: str) -> None:
    if not translated:
        log.warning("%s: the model sent no transcript", job.source.name)
    translated_path, source_path = transcript_paths(job.output)
    translated_path.write_text(translated + "\n", encoding="utf-8")
    source_path.write_text(source + "\n", encoding="utf-8")


async def translate_with_retry(
    translator: Translator,
    pcm: bytes,
    retries: int,
    label: str,
    on_sent: Callable[[float], None] | None = None,
) -> SegmentResult:
    return await retry(lambda: translator.translate(pcm, on_sent), retries, label)


async def translate_file(
    job: Job,
    translator: Translator,
    *,
    segment_seconds: float,
    sessions: asyncio.Semaphore,
    retries: int = 2,
    save_text: bool = False,
    progress: Progress | None = None,
) -> None:
    progress = progress or Progress()
    pcm = await asyncio.to_thread(decode_to_pcm16, job.source)
    samples = np.frombuffer(pcm, dtype="<i2")
    bounds = split_points(samples, INPUT_RATE, segment_seconds)
    total = len(samples) / INPUT_RATE
    progress.duration = total
    progress.start("live", len(bounds))
    log.info("%s: %.1f min of audio, %d segment(s)", job.source.name, total / 60, len(bounds))

    async def run(index: int, start: int, end: int) -> SegmentResult:
        label = f"{job.source.name} [{index + 1}/{len(bounds)}]"
        async with sessions:
            log.info("%s: translating %.0fs-%.0fs", label, start / INPUT_RATE, end / INPUT_RATE)
            result = await translate_with_retry(
                translator,
                samples[start:end].tobytes(),
                retries,
                label,
                on_sent=lambda seconds: progress.streamed.__setitem__(index, seconds),
            )
        progress.done += 1
        if not result.audio:
            log.warning("%s: no translated speech came back (music or silence?)", label)
        return result

    results = await run_all(run(i, a, b) for i, (a, b) in enumerate(bounds))

    if not any(r.audio for r in results):
        raise RuntimeError("the model returned no audio at all; check the model name and target language")
    rates = {r.rate for r in results if r.audio}
    if len(rates) != 1:
        raise RuntimeError(f"segments came back at different sample rates: {sorted(rates)}")
    rate = rates.pop()

    gap = silence(SEGMENT_GAP_SECONDS, rate)
    audio = gap.join(r.audio for r in results if r.audio)
    await asyncio.to_thread(write_wav, job.output, audio, rate)

    if save_text:
        write_transcripts(
            job,
            "\n\n".join(r.translated_text for r in results if r.translated_text),
            "\n\n".join(r.source_text for r in results if r.source_text),
        )
    log.info("%s: wrote %s (%.1f min)", job.source.name, job.output, len(audio) / SAMPLE_WIDTH / rate / 60)
