"""Auto-calibrate the mic↔speaker echo delay with a short startup chirp.

A known probe gives a clean cross-correlation peak, so the echo-path delay can be
*measured* automatically and fed to the AEC — no hand-tuned AEC_STREAM_DELAY_MS, and
it adapts to whatever path is in use (sounddevice on the Mac, robot.media on the robot).

``local_chat`` measures via sounddevice before opening its streams; the robot handler
plays the same probe through its normal output and times the echo in ``receive``.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import chirp, correlate


logger = logging.getLogger(__name__)

CALIB_SAMPLE_RATE = 16000


def make_probe(seconds: float = 1.0, sample_rate: int = CALIB_SAMPLE_RATE) -> NDArray[np.float32]:
    """Short repeated log-chirps — broadband with clean cross-correlation peaks."""
    sig = np.zeros(int(seconds * sample_rate), dtype=np.float32)
    t = np.linspace(0, 0.15, int(0.15 * sample_rate), endpoint=False)
    one = chirp(t, f0=400, f1=3500, t1=0.15, method="logarithmic").astype(np.float32)
    one *= np.hanning(len(one))
    for start in range(0, len(sig) - len(one), int(0.3 * sample_rate)):
        sig[start : start + len(one)] += one
    return (sig * 0.3).astype(np.float32)


def estimate_delay_ms(near: NDArray[Any], far: NDArray[Any], sample_rate: int = CALIB_SAMPLE_RATE) -> float:
    """Cross-correlate the recorded mic (*near*) against the played probe (*far*) -> delay (ms)."""
    near_f = np.asarray(near, dtype=np.float32).reshape(-1)
    far_f = np.asarray(far, dtype=np.float32).reshape(-1)
    if len(far_f) == 0 or len(near_f) < len(far_f):
        return 0.0
    window = near_f[: len(far_f) + sample_rate]  # search up to ~1 s of lag
    xc = correlate(window, far_f, mode="full", method="fft")
    lag = int(np.argmax(np.abs(xc))) - (len(far_f) - 1)
    return max(0.0, lag) / sample_rate * 1000.0


def measure_delay_via_sounddevice(seconds: float = 1.0) -> Optional[float]:
    """Play a probe on the default speaker while recording the default mic; return delay (ms).

    Returns None if sounddevice is unavailable or nothing was captured (so the caller can
    fall back to the APM's own estimator).
    """
    try:
        import threading

        import sounddevice as sd
    except Exception:
        return None

    sr = CALIB_SAMPLE_RATE
    probe = make_probe(seconds, sr)
    probe_i16 = np.clip(probe * 32768, -32768, 32767).astype(np.int16)
    n = len(probe_i16)
    pos = [0]
    rec: list[NDArray[np.int16]] = []
    done = threading.Event()

    def out_cb(outdata: Any, frames: int, t: Any, status: Any) -> None:  # speaker
        p = pos[0]
        end = min(p + frames, n)
        outdata[: end - p, 0] = probe_i16[p:end]
        if end - p < frames:
            outdata[end - p :, 0] = 0
            done.set()
        pos[0] = end

    def in_cb(indata: Any, frames: int, t: Any, status: Any) -> None:  # mic
        rec.append(indata[:, 0].copy())

    try:
        with sd.OutputStream(
            samplerate=sr, channels=1, dtype="int16", blocksize=320, callback=out_cb
        ), sd.InputStream(samplerate=sr, channels=1, dtype="int16", blocksize=320, callback=in_cb):
            while not done.is_set():
                sd.sleep(50)
            sd.sleep(300)
    except Exception as exc:
        logger.warning("Echo calibration failed to open audio streams: %s", exc)
        return None

    if not rec:
        return None
    return estimate_delay_ms(np.concatenate(rec), probe, sr)
