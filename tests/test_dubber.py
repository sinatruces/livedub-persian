import asyncio
import json
import shutil
import wave

import pytest

from livedub import dubber, pipeline
from livedub.dubber import (
    Dubber,
    DubOptions,
    batch_by_words,
    group_for_speech,
    speech_direction,
    translation_instructions,
)
from livedub.pipeline import Job, Progress
from tests.fakes import FakeClient, FakeModels
from tests.test_pipeline import make_wav

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_persian_instructions_cover_writing_and_register():
    formal = translation_instructions("fa", "formal")
    assert "Persian (Farsi)" in formal
    assert "zero-width non-joiner" in formal and "ezafe" in formal
    assert "می‌خوام" not in formal
    assert "می‌خوام" in translation_instructions("fa", "casual")
    assert "For Persian" not in translation_instructions("de", "formal")
    assert "Tehrani" in speech_direction("fa", "formal")
    assert "conversational" in speech_direction("fa", "casual")


def test_batch_by_words():
    para = " ".join(["word"] * 10)
    assert batch_by_words([para] * 5, 25) == [[para, para], [para, para], [para]]
    long = " ".join(["word"] * 40)
    assert batch_by_words([para, long, para], 25) == [[para], [long], [para]]


def test_group_for_speech_packs_and_splits():
    assert group_for_speech(["a" * 10, "b" * 10, "c" * 10], 25) == ["a" * 10 + "\n\n" + "b" * 10, "c" * 10]
    sentences = ["جمله‌ی اول است.", "جمله‌ی دوم است؟", "جمله‌ی سوم است!"]
    pieces = group_for_speech([" ".join(sentences)], 35)
    assert pieces == [f"{sentences[0]} {sentences[1]}", sentences[2]]
    assert all(len(p) <= 35 for p in pieces)


def dub(client, source, output, monkeypatch, **options):
    monkeypatch.setattr(pipeline, "RETRY_BASE_SECONDS", 0.01)
    progress = Progress()
    job = Job(source, output)
    asyncio.run(Dubber(client, DubOptions(**options)).dub_file(job, progress))
    return progress


@needs_ffmpeg
def test_dub_file_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(dubber, "TRANSCRIBE_SECONDS", 5)
    monkeypatch.setattr(dubber, "TRANSLATE_WORDS", 8)
    src = make_wav(tmp_path / "talk.wav", 12)
    out = tmp_path / "talk.fa.wav"
    client = FakeClient()
    progress = dub(client, src, out, monkeypatch, voice="Charon", style="casual")
    models = client.models

    transcribe = models.of("transcribe")
    assert len(transcribe) >= 3
    audio_part = transcribe[0].contents[0]
    assert audio_part.inline_data.mime_type == "audio/wav"
    assert audio_part.inline_data.data[:4] == b"RIFF"

    # Each translation request carries the paragraphs translated just before it.
    translate = models.of("translate")
    assert len(translate) == len(transcribe)  # 8 words per batch = the two 4-word paragraphs of one part
    first, second = (json.loads(c.contents) for c in translate[:2])
    assert first["previous_source"] == [] and first["previous_translation"] == []
    assert second["previous_source"] == first["paragraphs"]
    assert second["previous_translation"] == [f"ترجمه «{p}»" for p in first["paragraphs"]]
    assert "می‌خوام" in translate[0].config.system_instruction

    speak = models.of("speak")
    assert speak[0].model == "gemini-3.8-flash-tts"
    assert speak[0].config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Charon"

    translated = out.with_suffix(".txt").read_text(encoding="utf-8").strip().split("\n\n")
    source = (tmp_path / "talk.source.txt").read_text(encoding="utf-8").strip().split("\n\n")
    assert len(source) == 2 * len(transcribe)
    assert translated == [f"ترجمه «{p}»" for p in source]

    spoken = sum(len(c.contents.split("\n\n", 1)[1]) for c in speak)
    gaps = (len(speak) - 1) * int(dubber.PIECE_GAP_SECONDS * 24000)
    with wave.open(str(out)) as w:
        assert w.getframerate() == 24000
        assert w.getnframes() == spoken * 240 + gaps

    assert progress.duration == pytest.approx(12, abs=0.05)
    assert (progress.stage, progress.done, progress.total) == ("speak", len(speak), len(speak))


@needs_ffmpeg
def test_wav_wrapped_speech_and_retries(tmp_path, monkeypatch):
    src = make_wav(tmp_path / "talk.wav", 3)
    out = tmp_path / "talk.fa.wav"
    client = FakeClient(models=FakeModels(wav_tts=True, failures={"translate": 1, "speak": 1}))
    dub(client, src, out, monkeypatch)
    assert len(client.models.of("translate")) == 2
    assert len(client.models.of("speak")) == 2
    with wave.open(str(out)) as w:
        assert w.getframerate() == 24000 and w.getnframes() > 0


@needs_ffmpeg
def test_no_speech_is_an_error(tmp_path, monkeypatch):
    src = make_wav(tmp_path / "music.wav", 3)
    client = FakeClient(models=FakeModels(silent=True))
    with pytest.raises(RuntimeError, match="no speech"):
        dub(client, src, tmp_path / "music.fa.wav", monkeypatch)
    assert not client.models.of("speak")


def test_voice_preview_reads_a_sample_in_the_chosen_style():
    client = FakeClient()
    data = asyncio.run(Dubber(client, DubOptions(voice="Sulafat", style="casual")).preview())
    assert data[:4] == b"RIFF"
    [call] = client.models.of("speak")
    assert "یه نمونه از صدای منه" in call.contents
    assert call.config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Sulafat"


def test_rejects_unknown_style():
    with pytest.raises(ValueError):
        Dubber(FakeClient(), DubOptions(style="shouty"))
