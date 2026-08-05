"""Structural interfaces (Protocols) for the pluggable speech backends.

Make the implicit contracts explicit so the conversation engine, the handler and
the fake-robot runner share one definition instead of ``Any`` — and mypy can check
that Piper/Remote/Gemma actually satisfy them.
"""

from __future__ import annotations
from typing import Tuple, Iterator, Optional, Protocol

import numpy as np
from numpy.typing import NDArray


class TtsBackend(Protocol):
    """Text-to-speech: yield ``(sample_rate, int16 PCM)`` chunks for a piece of text."""

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> Iterator[Tuple[int, NDArray[np.int16]]]:
        """Yield ``(sample_rate, int16 PCM)`` chunks for *text*.

        *language* is authoritative; *language_hint* is the conversation's language,
        advisory, used only when the backend cannot classify *text* confidently.
        """
        ...


class SttBackend(Protocol):
    """Speech-to-text: transcribe a 16 kHz int16 utterance to ``(text, language)``."""

    async def transcribe(self, audio: NDArray[np.int16]) -> Tuple[str, str]:
        """Return ``(text, language)`` for a 16 kHz int16 *audio* utterance."""
        ...


class ChatBackend(Protocol):
    """LLM chat: answer user text (optionally with an image or audio blob) as a string."""

    async def respond(
        self, user_text: str, image: Optional[bytes] = None, audio: Optional[bytes] = None
    ) -> str:
        """Return the assistant's reply to *user_text* (+ optional image / audio)."""
        ...
