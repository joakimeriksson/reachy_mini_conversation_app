"""Tests for the MCP client integration module."""

from __future__ import annotations
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.mcp_client as mcp_mod
from reachy_mini_conversation_app.mcp_client import (
    McpToolWrapper,
    shutdown_mcp,
    _parse_mcp_urls,
    register_mcp_tools,
)


# ---------------------------------------------------------------------------
# _parse_mcp_urls
# ---------------------------------------------------------------------------


def test_parse_mcp_urls_empty() -> None:
    """Empty or None input returns an empty list."""
    assert _parse_mcp_urls(None) == []
    assert _parse_mcp_urls("") == []
    assert _parse_mcp_urls("   ") == []


def test_parse_mcp_urls_single() -> None:
    """Single URL is parsed correctly."""
    assert _parse_mcp_urls("http://host:8000/sse") == ["http://host:8000/sse"]


def test_parse_mcp_urls_multiple() -> None:
    """Comma-separated URLs are split and trimmed."""
    result = _parse_mcp_urls("http://a:8000/sse , http://b:9000/sse")
    assert result == ["http://a:8000/sse", "http://b:9000/sse"]


def test_parse_mcp_urls_trailing_comma() -> None:
    """Trailing comma does not produce an empty entry."""
    result = _parse_mcp_urls("http://a:8000/sse,")
    assert result == ["http://a:8000/sse"]


# ---------------------------------------------------------------------------
# McpToolWrapper
# ---------------------------------------------------------------------------


def _make_fake_mcp_tool(name: str = "get_lights", description: str = "Get all lights") -> MagicMock:
    """Create a mock mcp.types.Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {
        "type": "object",
        "properties": {"room": {"type": "string"}},
    }
    return tool


def test_wrapper_spec_has_mcp_prefix() -> None:
    """Wrapper spec uses mcp_ prefix and correct structure."""
    mcp_tool = _make_fake_mcp_tool()
    client = MagicMock()
    wrapper = McpToolWrapper(mcp_tool, client)

    spec = wrapper.spec()
    assert spec["type"] == "function"
    assert spec["name"] == "mcp_get_lights"
    assert spec["description"] == "Get all lights"
    assert "properties" in spec["parameters"]


@pytest.mark.asyncio
async def test_wrapper_call_delegates_to_client() -> None:
    """Wrapper __call__ delegates to client.call_tool with original name."""
    mcp_tool = _make_fake_mcp_tool()
    client = AsyncMock()

    # Simulate a CallToolResult with a .content list
    content_item = MagicMock()
    content_item.text = '{"lights": [{"name": "Kitchen"}]}'
    call_result = MagicMock()
    call_result.content = [content_item]
    client.call_tool.return_value = call_result

    wrapper = McpToolWrapper(mcp_tool, client)
    result = await wrapper(deps=None, room="kitchen")

    client.call_tool.assert_awaited_once_with("get_lights", {"room": "kitchen"})
    assert result == {"lights": [{"name": "Kitchen"}]}


@pytest.mark.asyncio
async def test_wrapper_call_plain_text() -> None:
    """Plain text results are wrapped in a result dict."""
    mcp_tool = _make_fake_mcp_tool()
    client = AsyncMock()

    content_item = MagicMock()
    content_item.text = "Light turned on"
    call_result = MagicMock()
    call_result.content = [content_item]
    client.call_tool.return_value = call_result

    wrapper = McpToolWrapper(mcp_tool, client)
    result = await wrapper(deps=None)

    assert result == {"result": "Light turned on"}


@pytest.mark.asyncio
async def test_wrapper_call_error_handling() -> None:
    """Wrapper returns error dict when remote call fails."""
    mcp_tool = _make_fake_mcp_tool()
    client = AsyncMock()
    client.call_tool.side_effect = ConnectionError("server down")

    wrapper = McpToolWrapper(mcp_tool, client)
    result = await wrapper(deps=None)

    assert "error" in result
    assert "server down" in result["error"]


# ---------------------------------------------------------------------------
# register_mcp_tools / shutdown_mcp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_returns_zero_when_not_configured(monkeypatch: Any) -> None:
    """register_mcp_tools returns 0 when no URLs are configured."""
    # Reset module state
    monkeypatch.setattr(mcp_mod, "_manager", None)
    monkeypatch.setattr(mcp_mod.config, "MCP_SERVER_URLS", None)

    count = await register_mcp_tools()
    assert count == 0


@pytest.mark.asyncio
async def test_register_is_idempotent(monkeypatch: Any) -> None:
    """Second call to register_mcp_tools is a no-op."""
    monkeypatch.setattr(mcp_mod, "_manager", MagicMock())  # pretend already registered

    count = await register_mcp_tools()
    assert count == 0


@pytest.mark.asyncio
async def test_shutdown_safe_when_not_initialised(monkeypatch: Any) -> None:
    """shutdown_mcp does not raise when never initialised."""
    monkeypatch.setattr(mcp_mod, "_manager", None)
    await shutdown_mcp()  # should not raise
