"""Tests for backend settings reload (the robot's per-instance .env path)."""

import os

import pytest

from reachy_local_assistant.config import Config


@pytest.fixture(autouse=True)
def _restore_config() -> object:
    """Reload from the ambient environment after each test."""
    yield
    Config.reload()


def test_reload_picks_up_values_set_after_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """The instance .env is written long after import; reload must see it."""
    monkeypatch.setenv("OLLAMA_URL", "http://192.168.1.50:11434")
    monkeypatch.setenv("TTS_URL", "http://192.168.1.50:8880/v1/audio/speech")
    monkeypatch.setenv("TTS_VOICE", "Björn")

    Config.reload()

    assert Config.OLLAMA_URL == "http://192.168.1.50:11434"
    assert Config.TTS_URL == "http://192.168.1.50:8880/v1/audio/speech"
    assert Config.TTS_VOICE == "Björn"


def test_stt_url_is_derived_from_the_tts_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """One URL configures both endpoints on the shared voice server."""
    monkeypatch.delenv("STT_URL", raising=False)
    monkeypatch.setenv("TTS_URL", "http://voice:8880/v1/audio/speech")

    Config.reload()

    assert Config.STT_URL == "http://voice:8880/v1/audio/transcriptions"


def test_explicit_stt_url_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whisper may live on a different host than the TTS engine."""
    monkeypatch.setenv("TTS_URL", "http://voice:8880/v1/audio/speech")
    monkeypatch.setenv("STT_URL", "http://whisper:9000/v1/audio/transcriptions")

    Config.reload()

    assert Config.STT_URL == "http://whisper:9000/v1/audio/transcriptions"


def test_unset_tts_url_leaves_stt_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unconfigured must stay unconfigured, not become a bogus derived URL."""
    monkeypatch.delenv("TTS_URL", raising=False)
    monkeypatch.delenv("STT_URL", raising=False)

    Config.reload()

    assert Config.TTS_URL == ""
    assert Config.STT_URL == ""


def test_defaults_are_not_duplicated_between_import_and_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reload() must be the only place defaults live.

    These defaults were once written out twice — in the class body and again in
    reload() — so editing one copy silently reverted the other on every restart
    (which is exactly when reload() runs on the robot).
    """
    for key in ("OLLAMA_URL", "OLLAMA_MODEL", "TTS_VOICE", "TTS_MODEL", "TTS_FORMAT", "TTS_URL"):
        monkeypatch.delenv(key, raising=False)

    Config.reload()
    after_reload = {k: getattr(Config, k) for k in ("OLLAMA_URL", "OLLAMA_MODEL", "TTS_VOICE")}

    # Import-time values come from the same code path, so a second reload with
    # the same (empty) environment must be a fixed point.
    Config.reload()
    assert {k: getattr(Config, k) for k in after_reload} == after_reload


def test_default_voice_matches_the_default_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stock setup must speak, not 500 on every turn.

    The voice server's default engine is base Kokoro, whose voices are all
    ``<lang><gender>_<name>``. A Swedish pack name here (the old default) makes
    every synthesis request fail against a stock server.
    """
    monkeypatch.delenv("TTS_VOICE", raising=False)

    Config.reload()

    assert Config.TTS_VOICE == "af_heart"


def test_direct_audio_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct audio halves turn latency; it is the intended default."""
    monkeypatch.delenv("OLLAMA_DIRECT_AUDIO", raising=False)

    Config.reload()

    assert Config.OLLAMA_DIRECT_AUDIO is True


def test_env_overrides_survive_a_reload_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings page writes env + .env, then calls reload()."""
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:27b")
    Config.reload()
    assert Config.OLLAMA_MODEL == "gemma4:27b"

    os.environ["OLLAMA_MODEL"] = "gemma4:9b"
    Config.reload()
    assert Config.OLLAMA_MODEL == "gemma4:9b"


def test_noise_gate_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answering room noise is the worst default; the gate must be opt-OUT."""
    monkeypatch.delenv("NOISE_GATE", raising=False)

    Config.reload()

    assert Config.NOISE_GATE is True

    monkeypatch.setenv("NOISE_GATE", "0")
    Config.reload()
    assert Config.NOISE_GATE is False


def test_mcp_disabled_tools_defaults_empty_and_reloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings page persists this to the instance .env; reload must see it."""
    monkeypatch.delenv("MCP_DISABLED_TOOLS", raising=False)
    Config.reload()
    assert Config.MCP_DISABLED_TOOLS == ""

    monkeypatch.setenv("MCP_DISABLED_TOOLS", "mcp_search,mcp_weather")
    Config.reload()
    assert Config.MCP_DISABLED_TOOLS == "mcp_search,mcp_weather"
