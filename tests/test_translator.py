import asyncio
import time

import pytest
from google.genai import errors

from livedub.translator import Translator, build_config
from tests.fakes import REPLY_CHUNK, FakeClient

ONE_SECOND = bytes(16000 * 2)


def translate(client, pcm, **kwargs):
    kwargs.setdefault("pace", 100)
    kwargs.setdefault("idle_timeout", 0.3)
    return asyncio.run(Translator(client, "fa", **kwargs).translate(pcm))


def test_config_targets_the_language():
    config = build_config("fa")
    assert config.translation_config.target_language_code == "fa"
    assert config.output_audio_transcription is None
    assert build_config("fa", with_text=True).output_audio_transcription is not None


def test_streams_everything_and_collects_the_reply():
    client = FakeClient()
    result = translate(client, ONE_SECOND * 3)
    session = client.sessions[0]
    # 3 s of audio plus 1 s of trailing silence, in 100 ms chunks, then end-of-stream.
    assert len(session.sent) == 40
    assert all(len(chunk) == 3200 for chunk in session.sent)
    assert session.stream_ended
    assert result.rate == 24000
    assert result.audio == REPLY_CHUNK * 5


def test_clean_close_ends_without_waiting_for_idle_timeout():
    client = FakeClient(on_end="close")
    start = time.monotonic()
    result = translate(client, ONE_SECOND, idle_timeout=5)
    assert time.monotonic() - start < 2
    assert result.audio == REPLY_CHUNK * 3


def test_server_error_while_sending_is_raised():
    client = FakeClient(fail_after_chunks=5)
    with pytest.raises(errors.APIError) as info:
        translate(client, ONE_SECOND * 3)
    assert info.value.code == 1011
    assert len(client.sessions[0].sent) < 40


def test_server_error_after_the_end_is_raised():
    with pytest.raises(errors.APIError):
        translate(FakeClient(on_end="error"), ONE_SECOND)


def test_transcripts_are_collected():
    result = translate(FakeClient(text=True), ONE_SECOND * 2, with_text=True)
    assert result.translated_text == "سلام سلام سلام"
    assert result.source_text == "hello hello hello"


def test_streams_in_real_time_by_default():
    start = time.monotonic()
    translate(FakeClient(on_end="close"), ONE_SECOND[: len(ONE_SECOND) // 2], pace=1)
    # Half a second of audio plus one second of trailing silence.
    assert time.monotonic() - start >= 1.4
