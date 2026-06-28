"""Tests for the shared audio DSP helpers."""

import io

import numpy as np
import soundfile as sf

from reachy_local_assistant.audio.dsp import to_mono, resample_int16, pcm16_to_wav_bytes


def test_to_mono_flattens_stereo():
    stereo = np.zeros((100, 2), dtype=np.int16)
    stereo[:, 0] = 1
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert mono.shape == (100,)
    assert np.all(mono == 1)  # takes the first channel


def test_to_mono_passthrough_mono():
    mono = np.arange(50, dtype=np.int16)
    assert np.array_equal(to_mono(mono), mono)


def test_resample_identity_when_rates_match():
    audio = np.arange(320, dtype=np.int16)
    out = resample_int16(audio, 16000, 16000)
    assert out.dtype == np.int16
    assert np.array_equal(out, audio)


def test_resample_changes_length_by_ratio():
    audio = np.zeros(2400, dtype=np.int16)  # 0.1 s @ 24 kHz
    out = resample_int16(audio, 24000, 16000)
    assert out.dtype == np.int16
    assert abs(len(out) - 1600) <= 1  # 0.1 s @ 16 kHz


def test_resample_coerces_float_to_int16():
    audio = np.ones(100, dtype=np.float32) * 0.5  # mid-scale float
    out = resample_int16(audio, 16000, 16000)
    assert out.dtype == np.int16
    assert out[0] == 16384  # 0.5 * 32768


def test_resample_empty_when_target_is_zero_length():
    out = resample_int16(np.ones(1, dtype=np.int16), 48000, 1)
    assert out.dtype == np.int16
    assert len(out) == 0


def test_pcm16_to_wav_roundtrip():
    audio = (np.sin(np.linspace(0, 6.28, 1600)) * 1000).astype(np.int16)
    wav = pcm16_to_wav_bytes(audio, 16000)
    data, rate = sf.read(io.BytesIO(wav), dtype="int16")
    assert rate == 16000
    assert len(data) == 1600
