"""High-quality mode: transcribe the speech, translate the whole text, then voice it with Gemini TTS.

The live mode is a simultaneous interpreter: it translates phrase by phrase without seeing what
comes next and speaks with its own voice. Here the translator sees the full text first and is told
how the result will be read aloud, and the voice is a studio TTS voice you pick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

import numpy as np
from google.genai import types
from pydantic import BaseModel

from .audio import (
    INPUT_RATE,
    OUTPUT_RATE,
    SAMPLE_WIDTH,
    decode_to_pcm16,
    rate_from_mime,
    read_wav,
    silence,
    split_points,
    wav_bytes,
    write_wav,
)
from .pipeline import Job, Progress, retry, run_all, write_transcripts

log = logging.getLogger(__name__)

TEXT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_VOICE = "Kore"

# A selection of Gemini's prebuilt voices: name -> (gender, character).
VOICES = {
    "Kore": ("female", "firm"),
    "Sulafat": ("female", "warm"),
    "Aoede": ("female", "breezy"),
    "Despina": ("female", "smooth"),
    "Charon": ("male", "informative"),
    "Achird": ("male", "friendly"),
    "Algieba": ("male", "smooth"),
    "Puck": ("male", "upbeat"),
}
STYLES = ("formal", "casual")
LANGUAGES = {
    "fa": "Persian (Farsi) as spoken in Iran",
    "en": "English",
    "ar": "Arabic",
    "tr": "Turkish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

TRANSCRIBE_SECONDS = 300  # audio per transcription request
TRANSLATE_WORDS = 2500  # source words per translation request
SPEAK_CHARS = 600  # characters per TTS request
CONTEXT_PARAGRAPHS = 3  # earlier paragraphs the translator sees for continuity
PIECE_GAP_SECONDS = 0.35  # pause between separately voiced pieces
NO_TOOLS = types.AutomaticFunctionCallingConfig(disable=True)

TRANSCRIBE_PROMPT = """Transcribe the speech in this audio, in the language it is spoken.
- Write what is said, but leave out filler words (um, uh), false starts and stutters.
- Ignore music, noise and other non-speech sounds.
- Split the text into paragraphs of one to three sentences, in order.
- If there is no speech, return an empty list."""

SAMPLE_TEXT = {
    ("fa", "formal"): "سلام! این نمونه‌ای از صدای من است. فایل‌های شما با همین صدا و همین لحن به فارسی خوانده می‌شوند.",
    ("fa", "casual"): "سلام! این یه نمونه از صدای منه. فایل‌هاتون با همین صدا و همین لحن به فارسی خونده میشن.",
    ("en", None): "Hello! This is a sample of my voice. Your files will be read in this voice and tone.",
    ("ar", None): "مرحباً! هذه عينة من صوتي. ستُقرأ ملفاتك بهذا الصوت وهذه النبرة.",
    ("tr", None): "Merhaba! Bu benim sesimden bir örnek. Dosyalarınız bu ses ve bu tonla okunacak.",
    ("de", None): "Hallo! Das ist eine Hörprobe meiner Stimme. Ihre Dateien werden mit dieser Stimme gelesen.",
    ("fr", None): "Bonjour ! Voici un extrait de ma voix. Vos fichiers seront lus avec cette voix et ce ton.",
    ("es", None): "¡Hola! Esta es una muestra de mi voz. Tus archivos se leerán con esta voz y este tono.",
}


class _Paragraphs(BaseModel):
    paragraphs: list[str]


def translation_instructions(language: str, style: str) -> str:
    register = {
        "formal": "fluent, natural standard language that suits a professional voice-over narration; never stiff or word-for-word",
        "casual": "natural everyday spoken language, the way native speakers really talk, while staying clear and polite",
    }[style]
    lines = [
        f"You are a professional dubbing translator. Translate the `paragraphs` into {LANGUAGES.get(language, language)}.",
        f"Write {register}.",
        "Translate the meaning, not the words: rebuild sentences so they sound as if they were first said in the "
        "target language, and keep the speaker's tone and intent.",
        "`previous_source` and `previous_translation` are the paragraphs just before these; use them only for "
        "context and consistent terminology, and do not translate them again.",
        "A text-to-speech voice will read your translation aloud, so write numbers, dates, units and symbols the "
        "way they should be spoken, and spell out abbreviations that are said as words.",
        "Keep the names of people, places and products, written in the target script the way they are pronounced.",
        "Return exactly one translated paragraph for each input paragraph, in the same order.",
    ]
    if language == "fa":
        lines += [
            "For Persian:",
            "- Use the zero-width non-joiner correctly (e.g. «می‌شود»، «کتاب‌ها»، «خانه‌ی ما»).",
            "- Use Persian punctuation (، ؛ ؟ « »).",
            "- Add the ezafe kasra (ـِ) and other short-vowel marks only where a word could otherwise be "
            "mispronounced, so the voice reads it correctly.",
        ]
        if style == "casual":
            lines.append("- Use colloquial Tehrani forms where they sound natural (e.g. «می‌خوام»، «میشه»، «اینو»).")
    return "\n".join(lines)


def speech_direction(language: str, style: str) -> str:
    name = "Persian with a native Iranian (Tehrani) accent" if language == "fa" else LANGUAGES.get(language, language)
    if style == "casual":
        return f"Read this aloud in {name}, in a natural, friendly, conversational tone:"
    return f"Read this aloud in {name} like a professional voice-over narrator, natural, warm and clear, at a relaxed pace:"


def batch_by_words(paragraphs: list[str], max_words: int) -> list[list[str]]:
    batches: list[list[str]] = []
    words = 0
    for para in paragraphs:
        n = len(para.split())
        if batches and words + n <= max_words:
            batches[-1].append(para)
            words += n
        else:
            batches.append([para])
            words = n
    return batches


def _split_sentences(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?؟…])\s+", text):
        if out and len(out[-1]) + 1 + len(sentence) <= max_chars:
            out[-1] += " " + sentence
        else:
            out.append(sentence)
    return out


def group_for_speech(paragraphs: list[str], max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of up to `max_chars` for TTS, splitting long ones at sentence ends."""
    pieces: list[str] = []
    for para in paragraphs:
        for block in _split_sentences(para, max_chars):
            if pieces and len(pieces[-1]) + 2 + len(block) <= max_chars:
                pieces[-1] += "\n\n" + block
            else:
                pieces.append(block)
    return pieces


def _audio_from(response: types.GenerateContentResponse) -> tuple[bytes, int]:
    chunks, rate = [], OUTPUT_RATE
    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else None) or []:
            blob = part.inline_data
            if not (blob and blob.data):
                continue
            if blob.data[:4] == b"RIFF":
                pcm, rate = read_wav(blob.data)
                chunks.append(pcm)
            else:
                rate = rate_from_mime(blob.mime_type, rate)
                chunks.append(blob.data)
    if not chunks:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        raise RuntimeError(f"the voice model returned no audio ({reason})")
    return b"".join(chunks), rate


@dataclass
class DubOptions:
    language: str = "fa"
    voice: str = DEFAULT_VOICE
    style: str = "formal"
    text_model: str = TEXT_MODEL
    tts_model: str = TTS_MODEL
    parallel: int = 4
    retries: int = 2


class Dubber:
    def __init__(self, client, options: DubOptions | None = None):
        self.client = client
        self.options = options or DubOptions()
        if self.options.style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}")

    async def _paragraphs(self, contents, system: str | None = None) -> list[str]:
        response = await self.client.aio.models.generate_content(
            model=self.options.text_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=_Paragraphs,
                automatic_function_calling=NO_TOOLS,
            ),
        )
        parsed = response.parsed
        if not isinstance(parsed, _Paragraphs):
            parsed = _Paragraphs.model_validate_json(response.text or "")
        return [p.strip() for p in parsed.paragraphs if p.strip()]

    async def transcribe(self, pcm: bytes) -> list[str]:
        audio = types.Part.from_bytes(data=wav_bytes(pcm, INPUT_RATE), mime_type="audio/wav")
        return await self._paragraphs([audio, TRANSCRIBE_PROMPT])

    async def translate(self, paragraphs: list[str], previous_source: list[str], previous_translation: list[str]) -> list[str]:
        request = {
            "previous_source": previous_source,
            "previous_translation": previous_translation,
            "paragraphs": paragraphs,
        }
        result = await self._paragraphs(
            json.dumps(request, ensure_ascii=False, indent=1),
            translation_instructions(self.options.language, self.options.style),
        )
        if len(result) != len(paragraphs):
            log.debug("asked to translate %d paragraphs, got %d back", len(paragraphs), len(result))
        if not result:
            raise RuntimeError("the translation came back empty")
        return result

    async def speak(self, text: str) -> tuple[bytes, int]:
        response = await self.client.aio.models.generate_content(
            model=self.options.tts_model,
            contents=f"{speech_direction(self.options.language, self.options.style)}\n\n{text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.options.voice)
                    )
                ),
                automatic_function_calling=NO_TOOLS,
            ),
        )
        return _audio_from(response)

    async def preview(self) -> bytes:
        """A short sample sentence in the chosen voice, language and style, as WAV."""
        o = self.options
        text = SAMPLE_TEXT.get((o.language, o.style)) or SAMPLE_TEXT.get((o.language, None)) or SAMPLE_TEXT[("en", None)]
        pcm, rate = await retry(lambda: self.speak(text), o.retries, f"voice sample {o.voice}")
        return wav_bytes(pcm, rate)

    async def dub_file(self, job: Job, progress: Progress | None = None) -> None:
        """Translate and voice `job.source`, writing the WAV and both transcripts next to `job.output`."""
        progress = progress or Progress()
        o = self.options
        name = job.source.name
        limit = asyncio.Semaphore(o.parallel)

        pcm = await asyncio.to_thread(decode_to_pcm16, job.source)
        samples = np.frombuffer(pcm, dtype="<i2")
        progress.duration = len(samples) / INPUT_RATE
        bounds = split_points(samples, INPUT_RATE, TRANSCRIBE_SECONDS)
        log.info("%s: %.1f min of audio; transcribing in %d part(s)", name, progress.duration / 60, len(bounds))

        progress.start("transcribe", len(bounds))

        async def transcribe(i: int, start: int, end: int) -> list[str]:
            async with limit:
                chunk = samples[start:end].tobytes()
                result = await retry(lambda: self.transcribe(chunk), o.retries, f"{name}: transcribing part {i + 1}")
            progress.done += 1
            return result

        parts = await run_all(transcribe(i, a, b) for i, (a, b) in enumerate(bounds))
        source = [p for part in parts for p in part]
        if not source:
            raise RuntimeError("no speech was found in this file")

        # Batches run in order so each one sees how the previous one was translated.
        batches = batch_by_words(source, TRANSLATE_WORDS)
        progress.start("translate", len(batches))
        log.info("%s: translating %d paragraph(s)", name, len(source))
        translated: list[str] = []
        done_source: list[str] = []
        for i, batch in enumerate(batches):
            previous = (done_source[-CONTEXT_PARAGRAPHS:], translated[-CONTEXT_PARAGRAPHS:])
            result = await retry(lambda: self.translate(batch, *previous), o.retries, f"{name}: translating part {i + 1}")
            translated += result
            done_source += batch
            progress.done += 1

        pieces = group_for_speech(translated, SPEAK_CHARS)
        progress.start("speak", len(pieces))
        log.info("%s: voicing %d piece(s) with %s", name, len(pieces), o.voice)

        async def speak(i: int, text: str) -> tuple[bytes, int]:
            async with limit:
                result = await retry(lambda: self.speak(text), o.retries, f"{name}: voicing piece {i + 1}")
            progress.done += 1
            return result

        voiced = await run_all(speak(i, t) for i, t in enumerate(pieces))
        rates = {rate for _, rate in voiced}
        if len(rates) != 1:
            raise RuntimeError(f"voice pieces came back at different sample rates: {sorted(rates)}")
        rate = rates.pop()
        audio = silence(PIECE_GAP_SECONDS, rate).join(pcm for pcm, _ in voiced)
        await asyncio.to_thread(write_wav, job.output, audio, rate)
        write_transcripts(job, "\n\n".join(translated), "\n\n".join(source))
        log.info("%s: wrote %s (%.1f min)", name, job.output, len(audio) / SAMPLE_WIDTH / rate / 60)
