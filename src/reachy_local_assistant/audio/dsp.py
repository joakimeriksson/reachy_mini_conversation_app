"""Shared audio DSP helpers (channel/dtype coercion, resampling, WAV encoding).

Centralized so the handler, the fake-robot runner, and the STT wrapper all use one
implementation instead of copy-pasting resampling/mono logic.
"""

from __future__ import annotations
import io
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample


def to_mono(audio: NDArray[Any]) -> NDArray[Any]:
    """Flatten a possibly-2D frame to a mono 1-D array (take the first channel)."""
    if audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:  # scipy channels-last convention
            audio = audio.T
        if audio.shape[1] > 1:
            audio = audio[:, 0]
    return audio.reshape(-1)


def resample_int16(audio: NDArray[Any], src_rate: int, dst_rate: int) -> NDArray[np.int16]:
    """Coerce *audio* to int16 and resample it from *src_rate* to *dst_rate*."""
    audio = np.asarray(audio)
    if audio.dtype != np.int16:
        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
    if src_rate == dst_rate:
        return audio
    n = int(len(audio) * dst_rate / src_rate)
    if n <= 0:
        return np.empty(0, dtype=np.int16)
    resampled = resample(audio.astype(np.float32), n)
    return np.clip(resampled, -32768, 32767).astype(np.int16)  # type: ignore[no-any-return]


def pcm16_to_wav_bytes(audio: NDArray[np.int16], sample_rate: int = 16000) -> bytes:
    """Wrap a mono int16 PCM utterance as WAV bytes (for Ollama's ``images`` field)."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio).reshape(-1), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
