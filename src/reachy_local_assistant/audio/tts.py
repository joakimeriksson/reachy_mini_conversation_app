"""TTS backend: an external OpenAI-compatible voice generator (remote).

The backend exposes ``synthesize(text, voice=None, language=None) ->
Iterator[(sample_rate, int16 pcm)]`` so the conversation handler stays
backend-agnostic. Point it at the voice server via ``TTS_URL`` (see
``scripts/voice_server.py`` for the Kokoro/Swedish server).
"""

from __future__ import annotations
import logging

from reachy_local_assistant.audio.protocols import TtsBackend


logger = logging.getLogger(__name__)


def make_tts() -> TtsBackend:
    """Construct the remote TTS backend (an OpenAI-compatible voice server)."""
    from reachy_local_assistant.config import config
    from reachy_local_assistant.audio.remote_tts import RemoteTTS

    logger.info(
        "TTS backend: remote (%s, model=%s, voice=%s)",
        config.TTS_URL, config.TTS_MODEL, config.TTS_VOICE,
    )
    return RemoteTTS(
        url=config.TTS_URL,
        model=config.TTS_MODEL,
        default_voice=config.TTS_VOICE,
        response_format=config.TTS_FORMAT,
        api_key=config.TTS_API_KEY,
    )
