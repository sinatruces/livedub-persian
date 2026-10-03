"""Shared plumbing for turning media files into translated WAV files, plus the live mode."""

from __future__ import annotations

import asyncio
import logging
import re
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
# A rate limit asking us to wait longer than this is treated as "out of quota for now".
MAX_RATE_LIMIT_WAIT = 120
# How many times we wait out a short rate limit before giving up.
MAX_RATE_LIMIT_WAITS = 8
# Added to the server's suggested wait, so the retry lands safely after it.
RATE_LIMIT_PADDING = 1


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


class QuotaExhausted(Exception):
    """The API refused a request because a quota is used up and will not refill soon."""

    def __init__(self, model: str | None, wait_seconds: float | None, per_day: bool, cause: Exception):
        self.model, self.wait_seconds, self.per_day = model, wait_seconds, per_day
        when = f"retry in about {wait_seconds / 3600:.1f} h" if wait_seconds else "retry later"
        scope = "daily quota" if per_day else "quota"
        super().__init__(f"{scope} of {model or 'the model'} is used up; {when}. ({cause})")


def _error_body(exc: errors.APIError) -> dict:
    body = exc.details if isinstance(exc.details, dict) else {}
    return body.get("error", body) if isinstance(body.get("error", body), dict) else {}


def quota_wait(exc: errors.APIError) -> float | None:
    """Seconds the server asked us to wait before retrying, if it said."""
    for detail in _error_body(exc).get("details") or []:
        delay = str(detail.get("retryDelay", "")) if isinstance(detail, dict) else ""
        if match := re.fullmatch(r"([\d.]+)s", delay):
            return float(match.group(1))
    return None


def _quota_scope(exc: errors.APIError) -> tuple[str | None, bool]:
    """(model, whether the exhausted quota is a per-day one) from a 429 error."""
    for detail in _error_body(exc).get("details") or []:
        for violation in (detail.get("violations") or []) if isinstance(detail, dict) else []:
            model = (violation.get("quotaDimensions") or {}).get("model")
            return model, "PerDay" in str(violation.get("quotaId", ""))
    return None, False


def _retryable(exc: Exception) -> bool:
    # Bad requests, keys or model names will not fix themselves.
    return not isinstance(exc, (errors.ClientError, QuotaExhausted))


async def retry(call: Callable[[], Awaitable[T]], retries: int, label: str) -> T:
    """Await `call()`, retrying transient failures with exponential backoff.

    Rate limits (HTTP 429) are waited out as the server asks, without using up `retries`;
    a quota that will not refill within MAX_RATE_LIMIT_WAIT raises QuotaExhausted at once.
    """
    attempt = rate_waits = 0
    while True:
        try:
            return await call()
        except errors.ClientError as e:
            if e.code != 429:
                raise
            wait = quota_wait(e)
            model, per_day = _quota_scope(e)
            if per_day or (wait or 0) > MAX_RATE_LIMIT_WAIT or rate_waits >= MAX_RATE_LIMIT_WAITS:
                raise QuotaExhausted(model, wait, per_day, e) from e
            rate_waits += 1
            wait = wait if wait is not None else 30
            log.warning("%s: rate limited; waiting %.0fs as the server asked", label, wait)
            await asyncio.sleep(wait + RATE_LIMIT_PADDING)
        except Exception as e:
            if attempt >= retries or not _retryable(e):
                raise
            wait = RETRY_BASE_SECONDS * 2**attempt
            attempt += 1
            log.warning("%s failed (%s: %s); retrying in %gs", label, type(e).__name__, e, wait)
            await asyncio.sleep(wait)


_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def friendly_error(exc: BaseException) -> str | None:
    """A short Persian explanation of errors people can act on, or None."""
    if isinstance(exc, QuotaExhausted):
        model = exc.model or "این مدل"
        scope = "سهمیه‌ی رایگان روزانه‌ی" if exc.per_day else "سهمیه‌ی"
        when = ""
        if exc.wait_seconds:
            hours, minutes = divmod(int(exc.wait_seconds // 60), 60)
            h, m = str(hours).translate(_FA_DIGITS), str(minutes + (0 if hours else 1)).translate(_FA_DIGITS)
            when = f"حدود {h} ساعت و {m} دقیقه‌ی دیگه " if hours else f"حدود {m} دقیقه‌ی دیگه "
        return (
            f"{scope} {model} تموم شده. {when}دوباره امتحان کنید؛ برای کار نیمه‌تموم دکمه‌ی «ادامه» رو بزنید "
            "تا از همون‌جا ادامه پیدا کنه و قسمت‌های انجام‌شده تکرار نشن. برای محدودیت بیشتر، در Google AI "
            "Studio برای پروژه‌ی کلیدتون پرداخت (billing) رو فعال کنید."
        )
    if isinstance(exc, errors.ClientError) and exc.code == 404:
        return "مدل پیدا نشد یا برای کلید شما فعال نیست. اسم مدل رو بررسی کنید."
    if isinstance(exc, errors.ClientError) and exc.code in (401, 403):
        return "گوگل اجازه‌ی این درخواست رو نداد؛ کلید API رو بررسی کنید."
    return None


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
