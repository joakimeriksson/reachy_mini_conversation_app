"""Tests for the headless console stream."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from reachy_mini.media.media_manager import MediaBackend
from reachy_local_assistant.console import LocalStream


def test_clear_audio_queue_prefers_clear_player_when_available() -> None:
    """Local GStreamer audio should use the lower-level player flush when available."""
    handler = MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.LOCAL))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_player.assert_called_once()
    audio.clear_output_buffer.assert_not_called()


def test_clear_audio_queue_uses_output_buffer_for_webrtc() -> None:
    """WebRTC audio should flush queued playback via the output buffer API."""
    handler = MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.WEBRTC))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    audio.clear_player.assert_not_called()


def test_clear_audio_queue_falls_back_when_backend_is_unknown() -> None:
    """Unknown backends should still best-effort flush pending playback."""
    handler = MagicMock()
    audio = SimpleNamespace(clear_output_buffer=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=None))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()


def test_local_stream_installs_the_player_flush_hook() -> None:
    """The handler must be able to flush the player, or barge-in cuts nothing off.

    Draining the handler's own output_queue leaves the frames already pushed into
    the player's appsrc playing, so LocalStream hands it this hook on construction.
    """
    handler = MagicMock()
    audio = SimpleNamespace(clear_output_buffer=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.WEBRTC))
    stream = LocalStream(handler, robot)

    assert handler._clear_queue == stream.clear_audio_queue

    handler._clear_queue()
    audio.clear_output_buffer.assert_called_once()
