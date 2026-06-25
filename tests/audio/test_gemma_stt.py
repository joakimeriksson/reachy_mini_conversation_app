"""Tests for the Gemma native-audio STT wrapper (response parsing + transcribe)."""

import numpy as np
import pytest

from reachy_local_assistant.audio.gemma_stt import GemmaSTT


def test_parse_strict_json():
    text, lang = GemmaSTT._parse('{"language": "en", "text": "hello there"}')
    assert text == "hello there"
    assert lang == "en"


def test_parse_json_embedded_in_prose():
    raw = 'Sure! {"language": "sv", "text": "hej"} hope that helps'
    text, lang = GemmaSTT._parse(raw)
    assert text == "hej"
    assert lang == "sv"


def test_parse_language_full_name_aliased_to_iso():
    text, lang = GemmaSTT._parse('{"language": "Swedish", "text": "hej"}')
    assert lang == "sv"


def test_parse_unknown_long_language_dropped():
    text, lang = GemmaSTT._parse('{"language": "klingon", "text": "Qapla"}')
    assert text == "Qapla"
    assert lang == ""  # not an ISO code we recognise


def test_parse_non_json_falls_back_to_raw_text():
    text, lang = GemmaSTT._parse("just plain words")
    assert text == "just plain words"
    assert lang == ""


@pytest.mark.asyncio
async def test_transcribe_sends_audio_and_returns_text(monkeypatch):
    captured = {}

    class FakeAsyncClient:
        def __init__(self, host):
            captured["host"] = host

        async def chat(self, model, messages, think, stream, **kwargs):
            captured["model"] = model
            captured["messages"] = messages
            return {"message": {"content": '{"language": "en", "text": "good morning"}'}}

    import ollama

    monkeypatch.setattr(ollama, "AsyncClient", FakeAsyncClient)

    stt = GemmaSTT("gemma-test", "http://localhost:11434")
    audio = (np.sin(np.arange(16000) * 0.1) * 10000).astype(np.int16)
    text, lang = await stt.transcribe(audio)

    assert text == "good morning"
    assert lang == "en"
    assert captured["model"] == "gemma-test"
    # Audio must be delivered through the multimodal `images` field as WAV bytes.
    msg = captured["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["images"], list) and isinstance(msg["images"][0], (bytes, bytearray))
    assert msg["images"][0][:4] == b"RIFF"  # WAV header


@pytest.mark.asyncio
async def test_transcribe_swallows_backend_errors(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, host):
            pass

        async def chat(self, **kwargs):
            raise RuntimeError("ollama down")

    import ollama

    monkeypatch.setattr(ollama, "AsyncClient", FakeAsyncClient)
    stt = GemmaSTT("gemma-test", "http://localhost:11434")
    text, lang = await stt.transcribe(np.zeros(1600, dtype=np.int16))
    assert (text, lang) == ("", "")
