"""Tests for the Piper TTS wrapper (chunk extraction, voice resolution)."""

import numpy as np
import pytest

from reachy_local_assistant.audio.piper_tts import PiperTTS


class _IntChunk:
    def __init__(self, arr):
        self.audio_int16_array = arr


class _FloatChunk:
    def __init__(self, arr):
        self.audio_float_array = arr


def test_chunk_int16_array_passthrough():
    out = PiperTTS._chunk_to_int16(_IntChunk(np.array([1, -2, 3], dtype=np.int16)))
    assert out.dtype == np.int16
    assert out.tolist() == [1, -2, 3]


def test_chunk_float_array_scaled_to_int16():
    out = PiperTTS._chunk_to_int16(_FloatChunk(np.array([0.0, 1.0, -1.0], dtype=np.float32)))
    assert out.dtype == np.int16
    assert out[0] == 0
    assert out[1] == 32767
    assert out[2] == -32767


def test_chunk_unrecognised_raises():
    with pytest.raises(TypeError):
        PiperTTS._chunk_to_int16(object())


def test_resolve_explicit_onnx_path(tmp_path):
    model = tmp_path / "myvoice.onnx"
    model.write_bytes(b"fake-onnx")
    tts = PiperTTS(default_voice=str(model))
    assert tts._resolve_model_path(str(model)) == model


def test_resolve_name_in_data_dir(tmp_path):
    (tmp_path / "en_US-test.onnx").write_bytes(b"fake-onnx")
    tts = PiperTTS(default_voice="en_US-test", data_dir=str(tmp_path))
    assert tts._resolve_model_path("en_US-test").name == "en_US-test.onnx"


def test_resolve_missing_voice_raises_actionable_error(tmp_path):
    tts = PiperTTS(default_voice="nope", data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not found"):
        tts._resolve_model_path("nope")


def test_synthesize_empty_text_yields_nothing():
    tts = PiperTTS(default_voice="unused")
    assert list(tts.synthesize("   ")) == []


def test_synthesize_streams_chunks_from_voice(monkeypatch):
    """synthesize() should yield (sample_rate, int16) without touching disk/ONNX."""

    class _Cfg:
        sample_rate = 22050

    class _FakeVoice:
        config = _Cfg()

        def synthesize(self, text, syn_config=None):
            yield _IntChunk(np.array([5, 6], dtype=np.int16))
            yield _FloatChunk(np.array([1.0], dtype=np.float32))

    tts = PiperTTS(default_voice="fake")
    monkeypatch.setattr(tts, "_get_voice", lambda name: _FakeVoice())

    chunks = list(tts.synthesize("hello"))
    assert [sr for sr, _ in chunks] == [22050, 22050]
    assert chunks[0][1].tolist() == [5, 6]
    assert chunks[1][1][0] == 32767
