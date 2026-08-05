"""End-to-end tests for the settings-page HTTP routes the UI polls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.media.media_manager import MediaBackend
from reachy_local_assistant.health import ProbeResult, BackendHealth
from reachy_local_assistant.console import LocalStream
from reachy_local_assistant.transcript import PENDING_USER_TEXT, Transcript


@pytest.fixture()
def client() -> TestClient:
    """A settings app with the routes mounted over a stub handler."""
    handler = MagicMock()
    handler.transcript = Transcript()
    handler.health = BackendHealth()
    robot = SimpleNamespace(
        media=SimpleNamespace(audio=SimpleNamespace(clear_output_buffer=MagicMock()), backend=MediaBackend.WEBRTC)
    )
    settings_app = FastAPI()
    stream = LocalStream(handler, robot, settings_app=settings_app)
    stream._init_settings_ui_if_needed()
    test_client = TestClient(settings_app)
    test_client.handler = handler  # type: ignore[attr-defined]
    return test_client


def test_history_starts_empty(client: TestClient) -> None:
    """A fresh app reports no conversation, not an error."""
    body = client.get("/history").json()

    assert body == {"messages": [], "latest_seq": 0, "generation": 1}


def test_history_returns_recorded_turns(client: TestClient) -> None:
    """Turns recorded by the handler are visible to the page."""
    client.handler.transcript.add("user", "hej")  # type: ignore[attr-defined]
    client.handler.transcript.add("assistant", "hej själv")  # type: ignore[attr-defined]

    body = client.get("/history").json()

    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "hej"),
        ("assistant", "hej själv"),
    ]
    assert body["latest_seq"] == 2


def test_history_since_returns_only_new_turns(client: TestClient) -> None:
    """The page polls with since=N; resending everything would flicker the view."""
    transcript: Transcript = client.handler.transcript  # type: ignore[attr-defined]
    transcript.add("user", "one")
    transcript.add("assistant", "two")

    body = client.get("/history", params={"since": 1}).json()

    assert [m["content"] for m in body["messages"]] == ["two"]


def test_history_clear_bumps_generation(client: TestClient) -> None:
    """The page detects a reset by generation, since seq stays monotonic."""
    transcript: Transcript = client.handler.transcript  # type: ignore[attr-defined]
    transcript.add("user", PENDING_USER_TEXT)
    before = client.get("/history").json()["generation"]

    cleared = client.post("/history/clear").json()
    after = client.get("/history").json()

    assert cleared["ok"] is True
    assert after["generation"] == before + 1
    assert after["messages"] == []
    # seq must not rewind, or a page polling with since=N would miss what follows.
    assert after["latest_seq"] == cleared["latest_seq"] == 1


def test_backends_status_includes_health(client: TestClient) -> None:
    """The page renders per-probe status from this payload."""
    client.handler.health = BackendHealth(  # type: ignore[attr-defined]
        [ProbeResult("Voice server (TTS)", "http://voice:8880/x", False, "Unreachable", "Start it")]
    )

    body = client.get("/backends/status").json()

    assert body["health"]["ok"] is False
    assert body["health"]["probes"][0]["name"] == "Voice server (TTS)"
    assert body["health"]["probes"][0]["hint"] == "Start it"


def test_backends_check_without_a_loop_reports_not_ready(client: TestClient) -> None:
    """Before the audio loop starts there is nowhere to run the probe."""
    resp = client.post("/backends/check")

    assert resp.status_code == 503
