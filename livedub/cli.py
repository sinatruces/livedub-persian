"""Command line entry point: python -m livedub INPUT [INPUT ...]"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .dubber import DEFAULT_VOICE, STYLES, TEXT_MODEL, TTS_MODEL, VOICES, Dubber, DubOptions
from .pipeline import collect_jobs, translate_file
from .translator import DEFAULT_MODEL, Translator

log = logging.getLogger("livedub")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m livedub",
        description="Translate the speech in audio/video files with Gemini and save it as WAV.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="media files and/or folders (searched recursively)")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("output"), help="where to write results (default: output)")
    parser.add_argument("-l", "--lang", default="fa", help="target language as a BCP-47 code (default: fa = Persian)")
    parser.add_argument("--mode", choices=("quality", "live"), default="quality",
                        help="quality: transcribe, translate the whole text, then voice it (default); "
                             "live: Gemini Live Translate, a real-time interpreter")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="requests (quality) or live sessions run at once (default: 4 for quality, 1 for live)")
    parser.add_argument("--retries", type=int, default=2, help="retries per request on errors (default: 2)")

    quality = parser.add_argument_group("quality mode")
    quality.add_argument("--voice", default=DEFAULT_VOICE,
                         help=f"Gemini voice (default: {DEFAULT_VOICE}; e.g. {', '.join(VOICES)})")
    quality.add_argument("--style", choices=STYLES, default="formal",
                         help="formal: fluent narration; casual: everyday spoken language (default: formal)")
    quality.add_argument("--text-model", default=TEXT_MODEL, help=f"transcription/translation model (default: {TEXT_MODEL})")
    quality.add_argument("--tts-model", default=TTS_MODEL, help=f"voice model (default: {TTS_MODEL})")

    live = parser.add_argument_group("live mode")
    live.add_argument("--model", default=DEFAULT_MODEL, help=f"Live API model (default: {DEFAULT_MODEL})")
    live.add_argument("--segment-minutes", type=float, default=5.0,
                      help="longest piece sent in one session; the API caps sessions at ~15 min (default: 5)")
    live.add_argument("--pace", type=float, default=1.0,
                      help="streaming speed as a multiple of real time (default: 1.0; higher is experimental)")
    live.add_argument("--save-text", action="store_true",
                      help="also save the transcripts (quality mode always saves them)")
    parser.add_argument("--overwrite", action="store_true", help="redo files whose output already exists")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    args = parser.parse_args(argv)

    if args.segment_minutes < 0.5 or args.segment_minutes > 14:
        parser.error("--segment-minutes must be between 0.5 and 14")
    if args.jobs is None:
        args.jobs = 4 if args.mode == "quality" else 1
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.pace <= 0:
        parser.error("--pace must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    return args


async def run(args: argparse.Namespace, client) -> int:
    jobs = collect_jobs(args.inputs, args.out_dir, args.lang)
    if not args.overwrite:
        skipped = [j for j in jobs if j.output.exists()]
        for job in skipped:
            log.info("%s: already done, skipping (use --overwrite to redo)", job.source.name)
        jobs = [j for j in jobs if not j.output.exists()]
    if not jobs:
        log.info("nothing to do")
        return 0

    if args.mode == "quality":
        dubber = Dubber(client, DubOptions(
            language=args.lang,
            voice=args.voice,
            style=args.style,
            text_model=args.text_model,
            tts_model=args.tts_model,
            parallel=args.jobs,
            retries=args.retries,
        ))
    else:
        translator = Translator(client, args.lang, args.model, pace=args.pace, with_text=args.save_text)
        sessions = asyncio.Semaphore(args.jobs)

    failed = []
    for job in jobs:
        try:
            if args.mode == "quality":
                await dubber.dub_file(job)
            else:
                await translate_file(
                    job,
                    translator,
                    segment_seconds=args.segment_minutes * 60,
                    sessions=sessions,
                    retries=args.retries,
                    save_text=args.save_text,
                )
        except Exception as e:
            log.error("%s: failed: %s: %s", job.source.name, type(e).__name__, e)
            log.debug("details", exc_info=True)
            failed.append(job)

    done = len(jobs) - len(failed)
    log.info("finished: %d translated, %d failed", done, len(failed))
    return 1 if failed else 0


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        # The SDK's and web server's own chatter is not useful here.
        for name in ("google_genai", "websockets", "aiohttp.access"):
            logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 11):
        print("livedub needs Python 3.11 or newer.", file=sys.stderr)
        return 1
    args = parse_args(argv)
    setup_logging(args.verbose)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("set GEMINI_API_KEY first (create one at https://aistudio.google.com/apikey)")
        return 2

    from google import genai

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    try:
        return asyncio.run(run(args, client))
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
