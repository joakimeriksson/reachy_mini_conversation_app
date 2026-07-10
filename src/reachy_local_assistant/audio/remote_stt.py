"""Speech-to-text via the voice server's OpenAI-compatible /v1/audio/transcriptions.

Used in direct-audio mode to get a TEXT transcript of the user's turn *after* the
LLM has already replied — so the conversation history can store lightweight text
(KV-cache-friendly, no audio bloat) instead of the raw audio blob. Runs off the
reply's critical path; a failure just leaves the audio in history (graceful).
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_local_assistant.audio.dsp import pcm16_to_wav_bytes


logger = logging.getLogger(__name__)


class RemoteSTT:
    """Transcribe int16 PCM via a remote Whisper endpoint (returns text + language)."""

    def __init__(self, url: str, timeout: float = 15.0) -> None:
        """Configure the /v1/audio/transcriptions endpoint (empty url disables it)."""
        self._url = url
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        """True when a transcription URL is configured."""
        return bool(self._url)

    async def transcribe(
        self, audio: NDArray[np.int16], sample_rate: int, language: Optional[str] = None
    ) -> Tuple[str, str]:
        """Return ``(text, detected_language)`` for *audio*; ``("", "")`` on failure/disabled."""
        if not self._url or audio is None or not len(audio):
            return "", ""
        import httpx

        wav = pcm16_to_wav_bytes(audio, sample_rate)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data = {"model": "whisper-1"}
        if language:
            data["language"] = language
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, files=files, data=data)
                resp.raise_for_status()
                body = resp.json()
                return (body.get("text") or "").strip(), body.get("language") or ""
        except Exception as exc:
            logger.warning("Remote STT failed (%s): %s — keeping audio in history", self._url, exc)
            return "", ""
