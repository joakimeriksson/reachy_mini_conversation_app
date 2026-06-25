"""TTS backend selection: local Piper or an external voice generator.

Both backends expose the same interface —
``synthesize(text, voice=None) -> Iterator[(sample_rate, int16 pcm)]`` — so the
conversation handler is backend-agnostic. Selected via ``TTS_BACKEND``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def make_tts() -> Any:
    """Construct the configured TTS backend (``piper`` or ``remote``)."""
    from reachy_local_assistant.config import config

    if config.TTS_BACKEND == "remote":
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

    from reachy_local_assistant.audio.piper_tts import PiperTTS

    logger.info("TTS backend: piper (voice=%s)", config.PIPER_VOICE)
    return PiperTTS(config.PIPER_VOICE, config.PIPER_DATA_DIR)
