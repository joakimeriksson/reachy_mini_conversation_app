"""Tests for the go_to_sleep tool."""

from unittest.mock import MagicMock

import pytest

from reachy_local_assistant.tools.go_to_sleep import GoToSleep


@pytest.mark.asyncio
async def test_calls_the_hook_and_returns_its_result() -> None:
    """The tool is a thin bridge to the lifecycle hook main.run() installs."""
    deps = MagicMock()
    deps.go_to_sleep = MagicMock(return_value={"status": "sleeping", "app_stopped": "stop_event"})

    result = await GoToSleep()(deps)

    deps.go_to_sleep.assert_called_once_with()
    assert result["status"] == "sleeping"


@pytest.mark.asyncio
async def test_without_a_hook_reports_unavailable() -> None:
    """Runners with no app lifecycle (local_chat) must get an error, not a crash."""
    deps = MagicMock()
    deps.go_to_sleep = None

    result = await GoToSleep()(deps)

    assert "unavailable" in result["error"]


@pytest.mark.asyncio
async def test_a_failing_hook_is_reported_not_raised() -> None:
    """A motor/daemon error mid-sleep must come back as a tool result."""
    deps = MagicMock()
    deps.go_to_sleep = MagicMock(side_effect=RuntimeError("motors offline"))

    result = await GoToSleep()(deps)

    assert "motors offline" in result["error"]
