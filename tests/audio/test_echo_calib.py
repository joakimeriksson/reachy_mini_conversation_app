"""Tests for echo-delay auto-calibration (the cross-correlation math, no hardware)."""

import numpy as np

from reachy_local_assistant.audio.echo_calib import CALIB_SAMPLE_RATE, make_probe, estimate_delay_ms


def test_make_probe_length_and_range():
    sr = CALIB_SAMPLE_RATE
    p = make_probe(1.0, sr)
    assert len(p) == sr
    assert p.dtype == np.float32
    assert np.max(np.abs(p)) <= 1.0


def test_estimate_delay_recovers_known_shift():
    sr = CALIB_SAMPLE_RATE
    probe = make_probe(1.0, sr)
    shift = int(0.05 * sr)  # 50 ms echo delay
    near = np.concatenate([np.zeros(shift, dtype=np.float32), probe])
    near += np.random.RandomState(0).randn(len(near)).astype(np.float32) * 0.01  # mild noise
    assert abs(estimate_delay_ms(near, probe, sr) - 50.0) < 5.0


def test_estimate_delay_zero_when_aligned():
    sr = CALIB_SAMPLE_RATE
    probe = make_probe(1.0, sr)
    assert estimate_delay_ms(probe, probe, sr) < 5.0


def test_estimate_delay_handles_too_short_near():
    sr = CALIB_SAMPLE_RATE
    probe = make_probe(1.0, sr)
    assert estimate_delay_ms(probe[: sr // 2], probe, sr) == 0.0
