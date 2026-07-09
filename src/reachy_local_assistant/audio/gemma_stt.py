"""Speech-to-text via a native-audio Gemma model on Ollama.

Gemma audio models understand raw speech directly, so there is no separate ASR
step: a 16 kHz mono utterance from :class:`audio.vad.VadSegmenter` is wrapped as
a WAV and handed to Ollama through the multimodal ``images`` field. The model
returns a JSON object with the transcription and a best-effort language guess.

Async by design — the conversation handler awaits :meth:`GemmaSTT.transcribe`
off the audio thread.
"""

from __future__ import annotations
import re
import json
import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_local_assistant.audio.dsp import pcm16_to_wav_bytes


logger = logging.getLogger(__name__)

# Ask for strict JSON so we can recover both text and language. Gemma is an LLM,
# so this is reliable; we still parse defensively.
_PROMPT = (
    "Listen to the audio and transcribe the speech exactly, word for word. "
    "Detect the language being spoken. Respond with ONLY a single JSON object "
    'of the form {"language": "<ISO 639-1 code>", "text": "<exact transcription>"} '
    "and nothing else. If there is no intelligible speech, use an empty text."
)

_LANG_ALIASES = {
    "swedish": "sv", "svenska": "sv", "english": "en", "norwegian": "no",
    "danish": "da", "finnish": "fi", "german": "de", "french": "fr", "spanish": "es",
}


class GemmaSTT:
    """Native-audio speech-to-text via a multimodal Gemma model on Ollama."""

    def __init__(self, model: str, host: str, sample_rate: int = 16000) -> None:
        """Build the async Ollama client for the STT model."""
        import ollama  # imported lazily so the dep is only needed for this backend

        self._model = model
        self._sample_rate = sample_rate
        self._client = ollama.AsyncClient(host=host)

    def set_host(self, host: str) -> None:
        """Re-point the STT client at a new Ollama host."""
        import ollama

        self._client = ollama.AsyncClient(host=host)

    async def transcribe(self, audio: NDArray[np.int16]) -> Tuple[str, str]:
        """Transcribe a 16 kHz mono int16 utterance.

        Returns ``(text, language)``; ``text`` is empty when no speech is found.
        """
        wav_bytes = self._to_wav_bytes(audio)
        try:
            from reachy_local_assistant.config import config

            response = await self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": _PROMPT, "images": [wav_bytes]}],
                think=False,  # transcription must not "think"
                stream=False,
                keep_alive=config.OLLAMA_KEEP_ALIVE,
            )
        except Exception as exc:  # network / model errors are not fatal to the loop
            logger.error("Gemma STT request failed: %s", exc)
            return "", ""

        raw = (response.get("message", {}).get("content") or "").strip()
        return self._parse(raw)

    def _to_wav_bytes(self, audio: NDArray[np.int16]) -> bytes:
        return pcm16_to_wav_bytes(audio, self._sample_rate)

    @staticmethod
    def _parse(raw: str) -> Tuple[str, str]:
        """Extract ``(text, language)`` from Gemma's reply, tolerating extra prose."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                text = str(obj.get("text", "")).strip()
                language = str(obj.get("language", "")).strip().lower()
                language = _LANG_ALIASES.get(language, language)
                if len(language) > 3:  # not an ISO code we recognise
                    language = ""
                return text, language
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Gemma STT: JSON parse failed, using raw text")
        return raw.strip(), ""
