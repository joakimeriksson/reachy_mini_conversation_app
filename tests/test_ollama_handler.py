"""Tests for the conversation handler's output-flush path and transcript."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_local_assistant.transcript import PENDING_USER_TEXT
from reachy_local_assistant.ollama_handler import OllamaConversationHandler


def _handler() -> OllamaConversationHandler:
    """Build a handler with stub deps (no robot, no backends)."""
    deps = MagicMock()
    deps.head_wobbler = MagicMock()
    return OllamaConversationHandler(deps)


@pytest.mark.asyncio
async def test_flush_output_drains_the_queue_and_resets_the_wobbler() -> None:
    """Queued reply audio is dropped so playback stops fast."""
    handler = _handler()
    await handler.output_queue.put((24000, np.zeros(10, dtype=np.int16)))
    await handler.output_queue.put((24000, np.zeros(10, dtype=np.int16)))

    handler._flush_output()

    assert handler.output_queue.empty()
    handler.deps.head_wobbler.reset.assert_called_once()


@pytest.mark.asyncio
async def test_flush_output_also_flushes_the_player() -> None:
    """The player hook must fire, or the already-buffered tail keeps playing.

    Emptying output_queue only stops frames not yet handed to the robot; audio
    already pushed into the player's appsrc plays on until it is flushed too.
    """
    handler = _handler()
    clear_player = MagicMock()
    handler._clear_queue = clear_player

    handler._flush_output()

    clear_player.assert_called_once()


@pytest.mark.asyncio
async def test_flush_output_survives_a_failing_player_flush() -> None:
    """A media-layer error during flush must not escape into the audio loop."""
    handler = _handler()
    handler._clear_queue = MagicMock(side_effect=RuntimeError("appsrc gone"))

    handler._flush_output()  # must not raise

    handler.deps.head_wobbler.reset.assert_called_once()


@pytest.mark.asyncio
async def test_flush_output_without_a_hook_is_a_no_op() -> None:
    """The hook is optional: local_chat.py drives the handler with no LocalStream."""
    handler = _handler()
    assert handler._clear_queue is None

    handler._flush_output()  # must not raise


# --- transcript ------------------------------------------------------------


def _ready_handler(reply: str = "hej själv") -> OllamaConversationHandler:
    """Handler with stubbed backends, ready to run one turn."""
    handler = _handler()
    handler._stt = MagicMock()
    handler._chat = MagicMock()
    handler._chat.respond = AsyncMock(return_value=reply)
    handler._tts = MagicMock()
    handler._speak = AsyncMock()
    handler._stt_remote = MagicMock(enabled=False)
    return handler


@pytest.mark.asyncio
async def test_a_turn_lands_in_the_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both sides of the exchange must reach the settings page."""
    monkeypatch.setattr("reachy_local_assistant.ollama_handler.config.OLLAMA_DIRECT_AUDIO", True)
    handler = _ready_handler("hej själv")

    await handler._handle_turn(np.zeros(1600, dtype=np.int16))

    assert [(m.role, m.content) for m in handler.transcript.messages()] == [
        ("user", PENDING_USER_TEXT),
        ("assistant", "hej själv"),
    ]


@pytest.mark.asyncio
async def test_whisper_upgrades_the_pending_user_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The late transcript replaces the mic glyph instead of adding a line."""
    monkeypatch.setattr("reachy_local_assistant.ollama_handler.config.OLLAMA_DIRECT_AUDIO", True)
    handler = _ready_handler("svar")
    handler._stt_remote = MagicMock(enabled=True)
    handler._stt_remote.transcribe = AsyncMock(return_value=("hej Reachy", "sv"))

    await handler._handle_turn(np.zeros(1600, dtype=np.int16))

    assert [(m.role, m.content) for m in handler.transcript.messages()] == [
        ("user", "hej Reachy"),
        ("assistant", "svar"),
    ]


@pytest.mark.asyncio
async def test_a_failed_turn_does_not_record_a_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend error must not leave a phantom assistant turn on the page."""
    monkeypatch.setattr("reachy_local_assistant.ollama_handler.config.OLLAMA_DIRECT_AUDIO", True)
    handler = _ready_handler()
    handler._chat.respond = AsyncMock(side_effect=RuntimeError("ollama down"))

    await handler._handle_turn(np.zeros(1600, dtype=np.int16))  # must not raise

    assert [m.role for m in handler.transcript.messages()] == ["user"]


# --- conversation-language hint --------------------------------------------


@pytest.mark.asyncio
async def test_whisper_language_is_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-turn language Whisper reports is the hint's whole source."""
    monkeypatch.setattr("reachy_local_assistant.ollama_handler.config.OLLAMA_DIRECT_AUDIO", True)
    handler = _ready_handler("svar")
    handler._stt_remote = MagicMock(enabled=True)
    handler._stt_remote.transcribe = AsyncMock(return_value=("hej Reachy", "sv"))

    await handler._handle_turn(np.zeros(1600, dtype=np.int16))

    assert handler._lang_history.hint() == "sv"


@pytest.mark.asyncio
async def test_speak_passes_the_hint_to_the_voice_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short reply must carry the conversation's language to the server."""
    handler = _handler()
    handler._tts = MagicMock()
    handler._lang_history.record("sv")
    captured: dict[str, object] = {}

    async def _fake_stream(*args: object, **kwargs: object):
        captured.update(kwargs)
        return
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr("reachy_local_assistant.ollama_handler.stream_sentences", _fake_stream)

    await handler._speak("Absolut.")

    assert captured["language_hint"] == "sv"
    assert captured["language"] is None  # direct-audio: nothing authoritative to send


@pytest.mark.asyncio
async def test_an_explicit_language_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cascade STT knows the language outright; the hint must not displace it."""
    handler = _handler()
    handler._tts = MagicMock()
    handler._lang_history.record("sv")
    captured: dict[str, object] = {}

    async def _fake_stream(*args: object, **kwargs: object):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    monkeypatch.setattr("reachy_local_assistant.ollama_handler.stream_sentences", _fake_stream)

    await handler._speak("Bien sûr.", "fr")

    assert captured["language"] == "fr"


@pytest.mark.asyncio
async def test_personality_switch_clears_the_hint() -> None:
    """A new personality may speak another language; the old hint must go."""
    handler = _handler()
    handler._chat = MagicMock()
    handler._lang_history.record("sv")

    await handler.apply_personality("noir_detective")

    assert handler._lang_history.hint() == ""
