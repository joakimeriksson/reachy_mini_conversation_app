"""Tests for sentence-streamed synthesis."""

import numpy as np
import pytest

from reachy_local_assistant.conversation.speech import stream_sentences


class FakeTts:
    def __init__(self, rate=24000):
        self.rate = rate
        self.seen = []

    def synthesize(self, text, voice=None, language=None, language_hint=None):
        self.seen.append(text)
        yield self.rate, np.zeros(max(1, len(text)), dtype=np.int16)


@pytest.mark.asyncio
async def test_streams_each_sentence():
    tts = FakeTts()
    out = []
    async for sr, pcm in stream_sentences(
        tts, "Hello there. How are you?", voice="v", language="en", should_stop=lambda: False
    ):
        out.append((sr, len(pcm)))
    assert len(tts.seen) == 2  # split into two sentences
    assert all(sr == 24000 for sr, _ in out)


@pytest.mark.asyncio
async def test_passes_voice_and_language_through():
    seen = {}

    class Recorder(FakeTts):
        def synthesize(self, text, voice=None, language=None, language_hint=None):
            seen["voice"] = voice
            seen["language"] = language
            seen["language_hint"] = language_hint
            return super().synthesize(text)

    async for _ in stream_sentences(Recorder(), "Hej.", voice="sv_female", language="sv", should_stop=lambda: False):
        pass
    assert seen == {"voice": "sv_female", "language": "sv", "language_hint": None}


@pytest.mark.asyncio
async def test_the_hint_reaches_every_sentence():
    """Splitting shortens each sentence, which is when the hint matters most."""
    seen = []

    class Recorder(FakeTts):
        def synthesize(self, text, voice=None, language=None, language_hint=None):
            seen.append((text, language_hint))
            return super().synthesize(text)

    async for _ in stream_sentences(
        Recorder(), "Absolut. Visst.", voice=None, language=None,
        language_hint="sv", should_stop=lambda: False,
    ):
        pass
    assert seen == [("Absolut.", "sv"), ("Visst.", "sv")]


@pytest.mark.asyncio
async def test_should_stop_halts_before_any_synthesis():
    tts = FakeTts()
    out = [chunk async for chunk in stream_sentences(
        tts, "One. Two. Three.", voice=None, language=None, should_stop=lambda: True
    )]
    assert out == []
    assert tts.seen == []  # stopped before touching the TTS
