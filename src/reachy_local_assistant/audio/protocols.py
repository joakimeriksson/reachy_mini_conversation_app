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
        self, text: str, voice: Optional[str] = None, language: Optional[str] = None
    ) -> Iterator[Tuple[int, NDArray[np.int16]]]: ...


class SttBackend(Protocol):
    """Speech-to-text: transcribe a 16 kHz int16 utterance to ``(text, language)``."""

    async def transcribe(self, audio: NDArray[np.int16]) -> Tuple[str, str]: ...
