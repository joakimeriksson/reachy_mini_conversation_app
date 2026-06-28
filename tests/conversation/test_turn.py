"""Tests for the shared turn step (understand utterance -> reply)."""

import numpy as np
import pytest

from reachy_local_assistant.conversation.turn import TurnResult, generate_reply


class FakeChat:
    def __init__(self, reply="hi"):
        self.reply = reply
        self.calls = []

    async def respond(self, user_text, image=None, audio=None):
        self.calls.append({"text": user_text, "image": image, "audio": audio})
        return self.reply


class FakeStt:
    def __init__(self, text="hello", lang="en"):
        self._text = text
        self._lang = lang

    async def transcribe(self, audio):
        return self._text, self._lang


@pytest.mark.asyncio
async def test_direct_audio_skips_stt_and_passes_wav():
    chat = FakeChat("svar")
    utt = np.zeros(1600, dtype=np.int16)
    res = await generate_reply(FakeStt(), chat, utt, direct_audio=True, sample_rate=16000)
    assert res == TurnResult("", "svar", None)
    assert chat.calls[0]["audio"] is not None  # the WAV was attached
    assert chat.calls[0]["text"] == ""


@pytest.mark.asyncio
async def test_cascade_transcribes_then_chats():
    chat = FakeChat("answer")
    res = await generate_reply(
        FakeStt("what time", "en"), chat, np.zeros(10, np.int16), direct_audio=False, sample_rate=16000
    )
    assert res.user_text == "what time"
    assert res.reply == "answer"
    assert res.language == "en"
    assert chat.calls[0]["text"] == "what time"


@pytest.mark.asyncio
async def test_empty_transcription_returns_no_reply_and_skips_chat():
    chat = FakeChat("nope")
    res = await generate_reply(
        FakeStt("   ", "en"), chat, np.zeros(10, np.int16), direct_audio=False, sample_rate=16000
    )
    assert res.reply == ""
    assert chat.calls == []  # never bothered the chat model


@pytest.mark.asyncio
async def test_capture_image_injected_in_cascade():
    chat = FakeChat()
    captured = {}

    def cap(text):
        captured["text"] = text
        return b"JPEG"

    await generate_reply(
        FakeStt("look", "en"), chat, np.zeros(10, np.int16),
        direct_audio=False, sample_rate=16000, capture_image=cap,
    )
    assert captured["text"] == "look"
    assert chat.calls[0]["image"] == b"JPEG"


@pytest.mark.asyncio
async def test_on_user_text_called_before_reply():
    chat = FakeChat()
    seen = []

    async def show(t):
        seen.append(t)

    await generate_reply(
        FakeStt("hi", "en"), chat, np.zeros(10, np.int16),
        direct_audio=False, sample_rate=16000, on_user_text=show,
    )
    assert seen == ["hi"]
