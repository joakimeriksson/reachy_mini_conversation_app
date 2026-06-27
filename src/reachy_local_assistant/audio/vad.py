"""Voice-activity-detection utterance segmenter.

Feeds 16 kHz mono int16 frames through a VAD and emits complete utterance buffers
once trailing silence is detected. Used by the Ollama conversation handler to turn
a continuous mic stream into discrete turns, and (in full-duplex mode) to detect
barge-in while Reachy is speaking.

Two interchangeable per-frame detectors (the segmentation state machine is shared):
- ``webrtc`` — classic WebRTC VAD (``webrtcvad``, a core dep). Light, no model.
- ``silero`` — Silero VAD via onnxruntime (torch-free; model auto-downloads). More
  accurate at telling speech from noise / residual echo.

Kept deliberately small and synchronous — the handler owns async/threading and
resampling to 16 kHz before calling :meth:`VadSegmenter.feed`.
"""

from __future__ import annotations
import os
import logging
from typing import List, Deque, Protocol
from pathlib import Path
from collections import deque

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = VAD_SAMPLE_RATE * FRAME_MS // 1000  # 320 samples (webrtc default frame)

_SILERO_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"


class _Detector(Protocol):
    """Per-frame speech/non-speech classifier."""

    frame_samples: int

    def is_speech(self, frame: NDArray[np.int16]) -> bool: ...

    def reset(self) -> None: ...


class _WebrtcDetector:
    """Classic WebRTC VAD (8/16/32/48 kHz, 10/20/30 ms frames)."""

    frame_samples = FRAME_SAMPLES  # 320 = 20 ms @ 16 kHz

    def __init__(self, aggressiveness: int) -> None:
        try:
            import webrtcvad
        except ImportError as exc:  # pragma: no cover
            raise ImportError("webrtcvad is required for the webrtc VAD backend (it is a core dep).") from exc
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: NDArray[np.int16]) -> bool:
        return bool(self._vad.is_speech(frame.tobytes(), VAD_SAMPLE_RATE))

    def reset(self) -> None:
        pass


class _SileroDetector:
    """Silero VAD (neural) via onnxruntime — torch-free; the model auto-downloads."""

    frame_samples = 512  # silero v5 processes 512-sample steps @ 16 kHz
    _CONTEXT = 64  # ...but the model input is the prev 64 samples + this 512 = 576

    def __init__(self, threshold: float = 0.5) -> None:
        import onnxruntime as ort

        self._sess = ort.InferenceSession(_silero_model_path(), providers=["CPUExecutionProvider"])
        self._threshold = threshold
        self._sr = np.array(VAD_SAMPLE_RATE, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context: NDArray[np.float32] = np.zeros(self._CONTEXT, dtype=np.float32)

    def is_speech(self, frame: NDArray[np.int16]) -> bool:
        chunk = frame.astype(np.float32) / 32768.0
        x = np.concatenate([self._context, chunk]).reshape(1, -1)  # 64 + 512 = 576
        out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = chunk[-self._CONTEXT :]
        return float(np.asarray(out).reshape(-1)[0]) >= self._threshold

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._CONTEXT, dtype=np.float32)


def _silero_model_path() -> str:
    """Return the cached Silero VAD onnx model, downloading it on first use."""
    import urllib.request

    cache = Path(os.environ.get("SILERO_VAD_DIR", str(Path.home() / ".cache" / "silero-vad")))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "silero_vad.onnx"
    if not path.is_file():
        logger.info("Downloading Silero VAD model to %s", path)
        urllib.request.urlretrieve(_SILERO_URL, str(path))
    return str(path)


class VadSegmenter:
    """Turn a continuous 16 kHz int16 stream into discrete utterances.

    Call :meth:`feed` with arbitrary-length int16 chunks; it returns a list of
    finished utterances (each an int16 ndarray at 16 kHz), normally empty or of
    length one. A short pre-roll is prepended so the leading phoneme is not clipped.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        silence_ms: int = 800,
        min_speech_ms: int = 200,
        max_utterance_ms: int = 15000,
        preroll_ms: int = 300,
        backend: str = "webrtc",
        threshold: float = 0.5,
    ) -> None:
        """Build the chosen VAD detector and derive frame-count thresholds."""
        self._detector: _Detector = (
            _SileroDetector(threshold) if backend == "silero" else _WebrtcDetector(aggressiveness)
        )
        self._frame_samples = self._detector.frame_samples
        frame_ms = max(1, self._frame_samples * 1000 // VAD_SAMPLE_RATE)
        self._silence_frames = max(1, silence_ms // frame_ms)
        self._min_speech_frames = max(1, min_speech_ms // frame_ms)
        self._max_frames = max(1, max_utterance_ms // frame_ms)
        self._preroll_frames = max(0, preroll_ms // frame_ms)

        # Pending samples not yet aligned to a full detector frame.
        self._tail: NDArray[np.int16] = np.empty(0, dtype=np.int16)
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
        self._detector.reset()

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

        n_frames = self._tail.size // self._frame_samples
        for i in range(n_frames):
            frame = self._tail[i * self._frame_samples : (i + 1) * self._frame_samples]
            utterance = self._process_frame(frame)
            if utterance is not None:
                finished.append(utterance)

        # Keep the unaligned remainder for next call.
        self._tail = self._tail[n_frames * self._frame_samples :].copy()
        return finished

    def _process_frame(self, frame: NDArray[np.int16]) -> NDArray[np.int16] | None:
        is_speech = self._detector.is_speech(frame)

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
