"""Voice-activity-detection utterance segmenter.

Feeds 16 kHz mono int16 frames through WebRTC VAD and emits complete utterance
buffers once trailing silence is detected.  Used by the Ollama conversation
handler to turn a continuous mic stream into discrete turns for Gemma STT, and
(in full-duplex mode) to detect barge-in while Reachy is speaking.

Kept deliberately small and synchronous — the handler owns all async/threading
concerns and resampling to 16 kHz before calling :meth:`VadSegmenter.feed`.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# WebRTC VAD only accepts 8/16/32/48 kHz, 10/20/30 ms frames of 16-bit mono PCM.
VAD_SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = VAD_SAMPLE_RATE * FRAME_MS // 1000  # 320 samples / 640 bytes


class VadSegmenter:
    """Turn a continuous 16 kHz int16 stream into discrete utterances.

    Call :meth:`feed` with arbitrary-length int16 chunks; it returns a list of
    finished utterances (each an int16 ndarray at 16 kHz), normally empty or of
    length one.  A short pre-roll is prepended so the leading phoneme is not
    clipped.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        silence_ms: int = 800,
        min_speech_ms: int = 200,
        max_utterance_ms: int = 15000,
        preroll_ms: int = 300,
    ) -> None:
        try:
            import webrtcvad  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without dep
            raise ImportError(
                "webrtcvad is required for VAD. Install with: uv sync (it is a core dependency)."
            ) from exc

        self._vad = webrtcvad.Vad(aggressiveness)
        self._silence_frames = max(1, silence_ms // FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._max_frames = max(1, max_utterance_ms // FRAME_MS)
        self._preroll_frames = max(0, preroll_ms // FRAME_MS)

        # Pending samples not yet aligned to a full 20 ms frame.
        self._tail = np.empty(0, dtype=np.int16)
        # Ring buffer of recent frames used as pre-roll before speech starts.
        self._preroll: Deque[NDArray[np.int16]] = deque(maxlen=self._preroll_frames or 1)

        self._in_speech = False
        self._speech: List[NDArray[np.int16]] = []
        self._trailing_silence = 0

    def reset(self) -> None:
        """Drop any in-progress utterance (e.g. after a barge-in interrupt)."""
        self._tail = np.empty(0, dtype=np.int16)
        self._preroll.clear()
        self._in_speech = False
        self._speech = []
        self._trailing_silence = 0

    @property
    def in_speech(self) -> bool:
        """True while an utterance is actively being captured."""
        return self._in_speech

    def feed(self, pcm: NDArray[np.int16]) -> List[NDArray[np.int16]]:
        """Process a 16 kHz int16 chunk; return any completed utterances."""
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        if pcm.ndim > 1:
            pcm = pcm.reshape(-1)

        self._tail = np.concatenate([self._tail, pcm]) if self._tail.size else pcm
        finished: List[NDArray[np.int16]] = []

        n_frames = self._tail.size // FRAME_SAMPLES
        for i in range(n_frames):
            frame = self._tail[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            utterance = self._process_frame(frame)
            if utterance is not None:
                finished.append(utterance)

        # Keep the unaligned remainder for next call.
        self._tail = self._tail[n_frames * FRAME_SAMPLES :].copy()
        return finished

    def _process_frame(self, frame: NDArray[np.int16]) -> NDArray[np.int16] | None:
        is_speech = self._vad.is_speech(frame.tobytes(), VAD_SAMPLE_RATE)

        if not self._in_speech:
            self._preroll.append(frame)
            if is_speech:
                # Start a new utterance, seeded with buffered pre-roll.
                self._in_speech = True
                self._speech = list(self._preroll)
                self._preroll.clear()
                self._trailing_silence = 0
            return None

        # Already capturing speech.
        self._speech.append(frame)
        self._trailing_silence = 0 if is_speech else self._trailing_silence + 1

        too_long = len(self._speech) >= self._max_frames
        ended = self._trailing_silence >= self._silence_frames
        if ended or too_long:
            return self._finish_utterance(too_long)
        return None

    def _finish_utterance(self, too_long: bool) -> NDArray[np.int16] | None:
        frames = self._speech
        self._in_speech = False
        self._speech = []
        self._trailing_silence = 0

        if len(frames) < self._min_speech_frames + self._preroll_frames:
            logger.debug("Discarding sub-threshold utterance (%d frames)", len(frames))
            return None

        if too_long:
            logger.info("Max utterance length reached; flushing %d frames", len(frames))
        return np.concatenate(frames).astype(np.int16)
