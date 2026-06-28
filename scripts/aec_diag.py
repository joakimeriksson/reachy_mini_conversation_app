"""Diagnose the mic↔speaker echo path and how well the AEC removes it.

Plays a short probe through the speaker while recording the mic (the SAME two-stream
setup ``local_chat`` uses on Mac, where mic and speakers are separate devices), then
measures, on *your* hardware:

  - echo delay (ms)   — time from playing a sample to its echo reaching the mic; this
                        is what AEC_STREAM_DELAY_MS should be set to.
  - echo level / ERLE — how loud Reachy's voice is in the mic, and how many dB the AEC
                        removes (higher ERLE = better; barge-in needs the residual near
                        the noise floor).
  - signal health     — input level, noise floor, clipping, DC offset.

Run it on the robot's host (needs mic + speaker, the ``localdev`` + ``aec`` extras):

    .venv/bin/python scripts/aec_diag.py

It is a diagnostic only — it does not change any config; it prints a recommended
AEC_STREAM_DELAY_MS and a verdict.
"""
import sys
import threading
from pathlib import Path


# Make the in-repo package importable when run from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import sounddevice as sd
from scipy.signal import chirp, correlate


SR = 16000
BLOCK = 320  # 20 ms, matches local_chat


def _make_probe(seconds: float = 4.0) -> np.ndarray:
    """Repeated short log-chirps (clean for delay xcorr + broadband for ERLE) + silence tail."""
    sig = np.zeros(int(seconds * SR), dtype=np.float32)
    t = np.linspace(0, 0.2, int(0.2 * SR), endpoint=False)
    one = chirp(t, f0=300, f1=3000, t1=0.2, method="logarithmic").astype(np.float32)
    one *= np.hanning(len(one))
    for start in range(0, len(sig) - len(one), int(0.5 * SR)):
        sig[start : start + len(one)] += one
    sig *= 0.3  # moderate, comfortable level
    tail = np.zeros(SR, dtype=np.float32)  # 1 s silence -> noise floor
    return np.concatenate([sig, tail])


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def _dbfs(x: np.ndarray) -> float:
    return 20.0 * np.log10(_rms(x) / 32768.0 + 1e-12)


def _capture(probe_i16: np.ndarray) -> np.ndarray:
    """Play *probe_i16* while recording the mic; return the recording (int16)."""
    n = len(probe_i16)
    rec_buf: list = []
    pos = [0]
    done = threading.Event()

    def out_cb(outdata, frames, t, status):  # speaker
        p = pos[0]
        end = min(p + frames, n)
        chunk = probe_i16[p:end]
        outdata[: len(chunk), 0] = chunk
        if len(chunk) < frames:
            outdata[len(chunk) :, 0] = 0
            done.set()
        pos[0] = end

    def in_cb(indata, frames, t, status):  # mic
        rec_buf.append(indata[:, 0].copy())

    with sd.OutputStream(samplerate=SR, channels=1, dtype="int16", blocksize=BLOCK, callback=out_cb), sd.InputStream(
        samplerate=SR, channels=1, dtype="int16", blocksize=BLOCK, callback=in_cb
    ):
        while not done.is_set():
            sd.sleep(50)
        sd.sleep(400)  # capture the tail
    return np.concatenate(rec_buf) if rec_buf else np.zeros(0, dtype=np.int16)


def _measure_delay_ms(recorded: np.ndarray, probe: np.ndarray) -> float:
    """Cross-correlate the first chirps against the recording to find the echo delay."""
    ref = probe[: int(1.5 * SR)].astype(np.float32)
    seg = recorded[: int(2.5 * SR)].astype(np.float32)
    if len(seg) < len(ref):
        return 0.0
    xc = correlate(seg, ref, mode="full", method="fft")
    lag = int(np.argmax(np.abs(xc))) - (len(ref) - 1)
    return max(0.0, lag) / SR * 1000.0


def _erle_db(recorded: np.ndarray, probe: np.ndarray, delay_ms: int) -> float:
    """Run the AEC offline (echo=recorded, far=probe) and return the echo reduction in dB."""
    try:
        from reachy_local_assistant.audio.aec import EchoCanceller
    except Exception as exc:  # pragma: no cover
        print(f"   (AEC unavailable: {exc}; install the 'aec' extra)")
        return float("nan")

    aec = EchoCanceller(stream_delay_ms=delay_ms)
    n = min(len(recorded), len(probe))
    near_in = recorded[:n].astype(np.int16)
    far_in = np.clip(probe[:n] * 32768, -32768, 32767).astype(np.int16)
    cleaned = []
    for i in range(0, n - BLOCK, BLOCK):
        aec.play_reference(far_in[i : i + BLOCK])
        cleaned.append(aec.clean(near_in[i : i + BLOCK]))
    out = np.concatenate(cleaned) if cleaned else np.zeros(0, dtype=np.int16)
    # Compare over the playback region, skipping the first ~0.8 s (filter convergence).
    skip, end = int(0.8 * SR), min(int(3.5 * SR), len(out))
    if end <= skip:
        return float("nan")
    return _dbfs(near_in[skip:end]) - _dbfs(out[skip:end])


def main() -> int:
    print("AEC / echo diagnostic — playing a 5 s probe, recording the mic…\n")
    probe = _make_probe()
    probe_i16 = np.clip(probe * 32768, -32768, 32767).astype(np.int16)
    recorded = _capture(probe_i16)
    if len(recorded) < SR:
        print("FAIL: almost nothing recorded — is the mic enabled / the right input device?")
        return 1

    play_region = recorded[: int(3.5 * SR)]
    tail = recorded[-int(0.8 * SR) :]
    delay_ms = _measure_delay_ms(recorded, probe)
    echo_dbfs = _dbfs(play_region)
    noise_dbfs = _dbfs(tail)
    clip = float(np.mean(np.abs(recorded) > 32000) * 100)
    dc = float(np.mean(recorded))
    erle = _erle_db(recorded, probe, int(round(delay_ms)))

    print("── incoming audio ─────────────────────────────")
    print(f"  echo in mic (playback):  {echo_dbfs:6.1f} dBFS")
    print(f"  noise floor (silence):   {noise_dbfs:6.1f} dBFS")
    print(f"  echo-to-noise:           {echo_dbfs - noise_dbfs:6.1f} dB")
    print(f"  clipping:                {clip:6.2f} %  (want ~0)")
    print(f"  DC offset:               {dc:6.1f}     (want ~0)")
    print("── echo path ──────────────────────────────────")
    print(f"  measured echo delay:     {delay_ms:6.0f} ms")
    print(f"  AEC reduction (ERLE):    {erle:6.1f} dB  (higher is better)")
    print("───────────────────────────────────────────────")

    print("\nverdict:")
    if clip > 1:
        print("  ⚠ mic is clipping — lower input gain / move the mic away from the speaker.")
    if echo_dbfs - noise_dbfs < 6:
        print("  ⚠ very little echo captured — was the speaker actually playing out loud?")
    if not np.isnan(erle):
        if erle >= 18:
            print(f"  ✅ AEC removes ~{erle:.0f} dB — strong. If it still self-interrupts, raise")
            print("     BARGE_IN_SPEECH_MS (e.g. 600–800) or VAD_THRESHOLD.")
        elif erle >= 8:
            print(f"  ◐ AEC removes ~{erle:.0f} dB — partial. Set the delay below AND raise")
            print("     BARGE_IN_SPEECH_MS to ride out the residual.")
        else:
            print(f"  ✗ AEC barely helps (~{erle:.0f} dB) — the residual echo is what self-triggers.")
    print(f"\n  → try:  AEC_STREAM_DELAY_MS={int(round(delay_ms))}  (export it before running local_chat)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
