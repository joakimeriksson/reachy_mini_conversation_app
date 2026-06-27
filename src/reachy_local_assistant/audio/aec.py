"""Acoustic echo cancellation via livekit's WebRTC AudioProcessingModule.

Cleans Reachy's own voice out of the mic so barge-in detection doesn't trigger on
playback. Feed the played audio as the far-end reference (:meth:`play_reference`)
and the mic as near-end (:meth:`clean`); both are reframed to 10 ms @ 16 kHz
internally (what the APM expects).

Optional dependency: ``livekit`` (the ``aec`` extra).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


AEC_SAMPLE_RATE = 16000
_FRAME = AEC_SAMPLE_RATE // 100  # 160 samples = 10 ms


class EchoCanceller:
    """WebRTC echo canceller (near = mic, far = the audio we play)."""

    def __init__(self) -> None:
        """Build the APM with echo cancellation + noise suppression enabled."""
        from livekit import rtc  # optional dep (aec extra)

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=False,
        )
        self._near: NDArray[np.int16] = np.zeros(0, dtype=np.int16)
        self._far: NDArray[np.int16] = np.zeros(0, dtype=np.int16)

    def play_reference(self, pcm16: NDArray[np.int16]) -> None:
        """Feed far-end (played) 16 kHz audio so the canceller knows what to subtract."""
        self._far = np.concatenate([self._far, np.asarray(pcm16, dtype=np.int16).reshape(-1)])
        while len(self._far) >= _FRAME:
            frame, self._far = self._far[:_FRAME].copy(), self._far[_FRAME:]
            self._apm.process_reverse_stream(self._rtc.AudioFrame(frame.tobytes(), AEC_SAMPLE_RATE, 1, _FRAME))

    def clean(self, pcm16: NDArray[np.int16]) -> NDArray[np.int16]:
        """Return the mic audio with Reachy's echo removed (10 ms-aligned; may buffer)."""
        self._near = np.concatenate([self._near, np.asarray(pcm16, dtype=np.int16).reshape(-1)])
        out = []
        while len(self._near) >= _FRAME:
            frame, self._near = self._near[:_FRAME].copy(), self._near[_FRAME:]
            af = self._rtc.AudioFrame(frame.tobytes(), AEC_SAMPLE_RATE, 1, _FRAME)
            self._apm.process_stream(af)  # cleans af.data in place
            out.append(np.frombuffer(af.data, dtype=np.int16).copy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)
