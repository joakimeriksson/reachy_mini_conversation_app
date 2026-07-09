"""Tests for the remote (OpenAI-compatible) TTS backend."""

import io

import numpy as np
import pytest
import soundfile as sf

from reachy_local_assistant.audio.remote_tts import RemoteTTS


def _wav_bytes(samples: np.ndarray, sr: int = 22050) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class _FakeResp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class _FakeClient:
    """Captures the request and returns scripted audio bytes."""

    last_call = {}

    def __init__(self, audio_bytes):
        self._audio = audio_bytes

    def __call__(self, *a, **k):  # httpx.Client(timeout=...)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        _FakeClient.last_call = {"url": url, "json": json, "headers": headers}
        return _FakeResp(self._audio)


def _patch_httpx(monkeypatch, audio_bytes):
    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient(audio_bytes))


def test_empty_url_no_ops_instead_of_crashing():
    # The app must start UNCONFIGURED (on-robot ships without a .env); an empty
    # TTS_URL must not crash start_up() — synthesize() just yields nothing until
    # the URL is set on the settings page.
    tts = RemoteTTS(url="", model="tts-1", default_voice="Stina")
    assert list(tts.synthesize("hej")) == []


def test_empty_text_yields_nothing():
    tts = RemoteTTS(url="http://x/v1/audio/speech", model="tts-1", default_voice="alloy")
    assert list(tts.synthesize("   ")) == []


def test_synthesize_decodes_wav_response(monkeypatch):
    pcm = (np.sin(np.arange(8000) * 0.1) * 10000).astype(np.int16)
    _patch_httpx(monkeypatch, _wav_bytes(pcm, sr=22050))

    tts = RemoteTTS(url="http://host/v1/audio/speech", model="kokoro", default_voice="af_sky")
    chunks = list(tts.synthesize("hello", voice="custom"))

    assert len(chunks) == 1
    sr, out = chunks[0]
    assert sr == 22050
    assert out.dtype == np.int16
    assert len(out) == 8000
    # Request shape is OpenAI-compatible and carries the override voice.
    call = _FakeClient.last_call
    assert call["url"] == "http://host/v1/audio/speech"
    assert call["json"] == {
        "model": "kokoro", "input": "hello", "voice": "custom", "response_format": "wav",
    }


def test_default_voice_used_when_none(monkeypatch):
    _patch_httpx(monkeypatch, _wav_bytes(np.zeros(100, dtype=np.int16)))
    tts = RemoteTTS(url="http://h/v1/audio/speech", model="m", default_voice="alloy")
    list(tts.synthesize("hi"))
    assert _FakeClient.last_call["json"]["voice"] == "alloy"


def test_api_key_sets_bearer_header(monkeypatch):
    _patch_httpx(monkeypatch, _wav_bytes(np.zeros(100, dtype=np.int16)))
    tts = RemoteTTS(url="http://h/v1/audio/speech", model="m", default_voice="v", api_key="secret")
    list(tts.synthesize("hi"))
    assert _FakeClient.last_call["headers"]["Authorization"] == "Bearer secret"


def test_request_failure_yields_nothing(monkeypatch):
    import httpx

    class Boom(_FakeClient):
        def post(self, *a, **k):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(httpx, "Client", Boom(b""))
    tts = RemoteTTS(url="http://h/v1/audio/speech", model="m", default_voice="v")
    assert list(tts.synthesize("hi")) == []


def test_stereo_is_downmixed_to_mono(monkeypatch):
    stereo = np.stack([np.arange(50), np.arange(50)], axis=1).astype(np.int16)
    _patch_httpx(monkeypatch, _wav_bytes(stereo, sr=16000))
    tts = RemoteTTS(url="http://h/v1/audio/speech", model="m", default_voice="v")
    sr, out = list(tts.synthesize("hi"))[0]
    assert sr == 16000 and out.ndim == 1 and len(out) == 50
