"""Text-to-speech via an external, OpenAI-compatible voice generator.

Calls a remote ``/v1/audio/speech`` endpoint (Kokoro-FastAPI, openedai-speech,
or any OpenAI-compatible TTS server) so the voice generator can run on a
separate on-prem host — keeping the client thin (no local Piper/onnxruntime).

Same interface as :class:`audio.piper_tts.PiperTTS`:
``synthesize(text, voice) -> Iterator[(sample_rate, int16 pcm)]``.
"""

from __future__ import annotations
import io
import logging
from typing import Tuple, Iterator, Optional

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class RemoteTTS:
    """Streams audio from an OpenAI-compatible ``/v1/audio/speech`` endpoint."""

    def __init__(
        self,
        url: str,
        model: str,
        default_voice: str,
        response_format: str = "wav",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """Configure the remote /v1/audio/speech endpoint."""
        # Tolerate an unset URL so the app can start UNCONFIGURED. The on-robot app
        # ships without a .env, so TTS_URL is empty until the user sets it on the
        # settings page — synthesize() then no-ops (logs a warning) instead of
        # crashing start_up(), and applies live once the URL is saved.
        self._url = url
        self._model = model
        self._default_voice = default_voice
        self._format = response_format
        self._api_key = api_key
        self._timeout = timeout

    def synthesize(
        self, text: str, voice: Optional[str] = None, language: Optional[str] = None
    ) -> Iterator[Tuple[int, NDArray[np.int16]]]:
        """Yield ``(sample_rate, int16_pcm)`` for *text* from the remote service.

        *language* (e.g. the STT-detected language) is sent as an extra field so a
        multilingual server (our Kokoro voice server) speaks it correctly instead
        of guessing; servers that don't understand it ignore it.
        """
        text = (text or "").strip()
        if not text:
            return
        if not self._url:
            logger.warning("TTS not configured — set the voice-server URL (TTS_URL) on the settings page.")
            return

        import httpx

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "input": text,
            "voice": voice or self._default_voice,
            "response_format": self._format,
        }
        if language:
            payload["language"] = language
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url, json=payload, headers=headers)
                resp.raise_for_status()
                audio_bytes = resp.content
        except Exception as exc:
            logger.error("Remote TTS request failed (%s): %s", self._url, exc)
            return

        sr, pcm = self._decode(audio_bytes)
        if pcm.size:
            yield sr, pcm

    def _decode(self, audio_bytes: bytes) -> Tuple[int, NDArray[np.int16]]:
        """Decode the response (wav/flac/ogg) to (sample_rate, int16 mono)."""
        import soundfile as sf

        try:
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        except Exception as exc:
            logger.error(
                "Remote TTS decode failed (format=%s — use wav/flac/ogg): %s", self._format, exc
            )
            return 0, np.empty(0, dtype=np.int16)
        data = np.asarray(data)
        if data.ndim > 1:  # stereo -> mono
            data = data[:, 0]
        return int(sr), data.astype(np.int16)
