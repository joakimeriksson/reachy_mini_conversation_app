"""Tests for the backend reachability probes."""

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from reachy_local_assistant.health import ProbeResult, BackendHealth, check_backends


class FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(self, status_code: int = 200, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> Any:
        return self._body


class FakeClient:
    """Async client stub that maps URLs to responses (or raises)."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self._routes = routes

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, url: str) -> FakeResponse:
        result = self._routes.get(url)
        if result is None:
            raise httpx.ConnectError(f"nothing listening at {url}")
        if isinstance(result, Exception):
            raise result
        return result


def _patched(routes: dict[str, Any]) -> Any:
    return patch("httpx.AsyncClient", lambda **kw: FakeClient(routes))


OLLAMA_TAGS = "http://ollama:11434/api/tags"
VOICE_HEALTH = "http://voice:8880/health"


@pytest.mark.asyncio
async def test_all_backends_up() -> None:
    """Everything reachable and the model pulled -> ok."""
    routes = {
        OLLAMA_TAGS: FakeResponse(body={"models": [{"name": "gemma4:latest"}]}),
        VOICE_HEALTH: FakeResponse(body={"status": "ok", "tts": True, "stt": True, "engine": "KokoroSVMLEngine"}),
    }
    with _patched(routes):
        health = await check_backends(
            "http://ollama:11434", "gemma4:latest",
            "http://voice:8880/v1/audio/speech", "http://voice:8880/v1/audio/transcriptions",
        )
    assert health.ok
    assert len(health.probes) == 3


@pytest.mark.asyncio
async def test_voice_server_down_is_reported() -> None:
    """A missing voice server is the silent-robot failure — it must not pass."""
    routes = {OLLAMA_TAGS: FakeResponse(body={"models": [{"name": "gemma4:latest"}]})}
    with _patched(routes):
        health = await check_backends("http://ollama:11434", "gemma4:latest", "http://voice:8880/v1/audio/speech")

    assert not health.ok
    tts = next(p for p in health.probes if "TTS" in p.name)
    assert not tts.ok
    assert "Unreachable" in tts.detail
    assert tts.hint


@pytest.mark.asyncio
async def test_unset_tts_url_is_a_failure_with_a_hint() -> None:
    """An unconfigured voice server must be flagged, not treated as 'nothing to check'."""
    routes = {OLLAMA_TAGS: FakeResponse(body={"models": [{"name": "gemma4:latest"}]})}
    with _patched(routes):
        health = await check_backends("http://ollama:11434", "gemma4:latest", "")

    assert not health.ok
    tts = next(p for p in health.probes if "TTS" in p.name)
    assert "Not configured" in tts.detail
    assert "TTS_URL" in tts.hint


@pytest.mark.asyncio
async def test_server_up_but_transcription_disabled() -> None:
    """Up-but-no-Whisper is a real state: the app keeps audio in history silently."""
    routes = {
        OLLAMA_TAGS: FakeResponse(body={"models": [{"name": "gemma4:latest"}]}),
        VOICE_HEALTH: FakeResponse(body={"status": "ok", "tts": True, "stt": False}),
    }
    with _patched(routes):
        health = await check_backends(
            "http://ollama:11434", "gemma4:latest",
            "http://voice:8880/v1/audio/speech", "http://voice:8880/v1/audio/transcriptions",
        )

    assert not health.ok
    stt = next(p for p in health.probes if "STT" in p.name)
    assert "disabled" in stt.detail
    assert "--whisper" in stt.hint


@pytest.mark.asyncio
async def test_missing_ollama_model_is_reported() -> None:
    """Ollama up but the conversation model never pulled."""
    routes = {
        OLLAMA_TAGS: FakeResponse(body={"models": [{"name": "llama3:latest"}]}),
        VOICE_HEALTH: FakeResponse(body={"status": "ok", "tts": True, "stt": True}),
    }
    with _patched(routes):
        health = await check_backends("http://ollama:11434", "gemma4:latest", "http://voice:8880/v1/audio/speech")

    ollama = next(p for p in health.probes if p.name == "Ollama")
    assert not ollama.ok
    assert "ollama pull gemma4:latest" in ollama.hint


@pytest.mark.asyncio
async def test_bare_model_name_matches_latest_tag() -> None:
    """`gemma4` and `gemma4:latest` are the same model; don't cry wolf."""
    routes = {
        OLLAMA_TAGS: FakeResponse(body={"models": [{"name": "gemma4:latest"}]}),
        VOICE_HEALTH: FakeResponse(body={"status": "ok", "tts": True, "stt": True}),
    }
    with _patched(routes):
        health = await check_backends("http://ollama:11434", "gemma4", "http://voice:8880/v1/audio/speech")

    assert next(p for p in health.probes if p.name == "Ollama").ok


@pytest.mark.asyncio
async def test_third_party_server_without_health_route_passes() -> None:
    """A 404 on /health means reachable-but-different-server, not down."""
    routes = {
        OLLAMA_TAGS: FakeResponse(body={"models": [{"name": "gemma4:latest"}]}),
        VOICE_HEALTH: FakeResponse(status_code=404),
    }
    with _patched(routes):
        health = await check_backends("http://ollama:11434", "gemma4:latest", "http://voice:8880/v1/audio/speech")

    assert health.ok


def test_empty_health_is_ok_by_default() -> None:
    """A handler that has not probed yet must not report a false failure."""
    assert BackendHealth().ok
    assert BackendHealth().as_dict() == {"ok": True, "probes": []}


def test_probe_result_serializes_for_the_settings_page() -> None:
    """The settings page consumes these as plain JSON."""
    probe = ProbeResult("Ollama", "http://x", False, "down", "start it")
    assert probe.as_dict() == {
        "name": "Ollama", "url": "http://x", "ok": False, "detail": "down", "hint": "start it",
    }
