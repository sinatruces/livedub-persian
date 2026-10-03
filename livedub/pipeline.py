"""Turns whole media files into translated WAV files, one Live API session per segment."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

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

# Pause inserted where two segments are joined back together.
SEGMENT_GAP_SECONDS = 0.3
# Retries wait 2, 4, 8... seconds.
RETRY_BASE_SECONDS = 2


@dataclass
class Job:
    source: Path
    output: Path


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


async def translate_with_retry(
    translator: Translator, pcm: bytes, retries: int, label: str
) -> SegmentResult:
    for attempt in range(retries + 1):
        try:
            return await translator.translate(pcm)
        except Exception as e:
            if attempt == retries or not _retryable(e):
                raise
            wait = RETRY_BASE_SECONDS * 2**attempt
            log.warning("%s failed (%s: %s); retrying in %gs", label, type(e).__name__, e, wait)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def translate_file(
    job: Job,
    translator: Translator,
    *,
    segment_seconds: float,
    sessions: asyncio.Semaphore,
    retries: int = 2,
    save_text: bool = False,
) -> None:
    pcm = await asyncio.to_thread(decode_to_pcm16, job.source)
    samples = np.frombuffer(pcm, dtype="<i2")
    bounds = split_points(samples, INPUT_RATE, segment_seconds)
    total = len(samples) / INPUT_RATE
    log.info("%s: %.1f min of audio, %d segment(s)", job.source.name, total / 60, len(bounds))

    async def run(index: int, start: int, end: int) -> SegmentResult:
        label = f"{job.source.name} [{index + 1}/{len(bounds)}]"
        async with sessions:
            log.info("%s: translating %.0fs-%.0fs", label, start / INPUT_RATE, end / INPUT_RATE)
            result = await translate_with_retry(translator, samples[start:end].tobytes(), retries, label)
        if not result.audio:
            log.warning("%s: no translated speech came back (music or silence?)", label)
        return result

    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run(i, a, b)) for i, (a, b) in enumerate(bounds)]
    except ExceptionGroup as group:
        raise group.exceptions[0] from None
    results = [t.result() for t in tasks]

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
        translated = "\n\n".join(r.translated_text for r in results if r.translated_text)
        source = "\n\n".join(r.source_text for r in results if r.source_text)
        if not translated:
            log.warning("%s: the model sent no transcript", job.source.name)
        job.output.with_suffix(".txt").write_text(translated + "\n", encoding="utf-8")
        job.output.with_name(job.output.stem.rsplit(".", 1)[0] + ".source.txt").write_text(
            source + "\n", encoding="utf-8"
        )
    log.info("%s: wrote %s (%.1f min)", job.source.name, job.output, len(audio) / SAMPLE_WIDTH / rate / 60)
