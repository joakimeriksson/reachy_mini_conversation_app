"""MCP (Model Context Protocol) client integration for external tool servers.

Connects to remote MCP servers via SSE or Streamable HTTP transport and
exposes their tools alongside the built-in local tools.  Fully optional
— the app works unchanged when no ``MCP_SERVER_URLS`` is configured and
``fastmcp`` is not installed.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

# Module-level manager singleton (set by register_mcp_tools, cleared by shutdown_mcp)
_manager: McpClientManager | None = None


def _parse_mcp_urls(raw: str | None) -> List[str]:
    """Parse a comma-separated string of MCP server URLs.

    Returns an empty list when *raw* is ``None`` or blank.
    """
    if not raw or not raw.strip():
        return []
    return [url.strip() for url in raw.split(",") if url.strip()]


class McpToolWrapper:
    """Wraps a remote MCP tool as a local tool compatible with the dispatch system.

    The tool name is prefixed with ``mcp_`` to avoid collisions with built-in
    tools.  The ``__call__`` signature matches the ``Tool`` base class so that
    ``dispatch_tool_call`` can invoke it transparently.
    """

    def __init__(self, mcp_tool: Any, client: Any) -> None:
        """Initialise from an ``mcp.types.Tool`` and its owning ``fastmcp.Client``."""
        self._original_name: str = mcp_tool.name
        self.name: str = f"mcp_{mcp_tool.name}"
        self.description: str = mcp_tool.description or ""
        self.parameters_schema: Dict[str, Any] = mcp_tool.inputSchema or {"type": "object", "properties": {}}
        self._client = client

    def spec(self) -> Dict[str, Any]:
        """Return the function spec for LLM consumption."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def __call__(self, deps: Any, **kwargs: Any) -> Dict[str, Any]:
        """Call the remote MCP tool and return a dict result.

        *deps* is accepted for signature compatibility with local ``Tool``
        instances but is not used.
        """
        try:
            result = await self._client.call_tool(self._original_name, kwargs)
        except Exception as exc:
            logger.error("MCP tool '%s' call failed: %s", self.name, exc)
            return {"error": f"MCP tool call failed: {exc}"}

        # Convert list[Content] → dict
        parts: List[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)

        combined = "\n".join(parts) if parts else ""

        # If the combined text is valid JSON dict, return it directly
        try:
            parsed = json.loads(combined)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        return {"result": combined}


class McpClientManager:
    """Manages connections to one or more MCP servers."""

    def __init__(self) -> None:
        """Initialise with empty client list."""
        self._clients: List[Any] = []
        self._tool_names: List[str] = []

    async def connect_and_register(self, server_urls: List[str]) -> int:
        """Connect to each MCP server and register discovered tools.

        Returns the total number of MCP tools registered.
        """
        try:
            from fastmcp import Client
        except ImportError:
            logger.warning(
                "fastmcp is not installed — MCP tools will not be available. Install with: uv sync --extra mcp"
            )
            return 0

        from reachy_mini_conversation_app.tools.core_tools import ALL_TOOLS, ALL_TOOL_SPECS

        total = 0
        for url in server_urls:
            try:
                # Pass URL directly — fastmcp auto-detects transport
                # (tries Streamable HTTP first, falls back to SSE)
                client = Client(url)
                await client.__aenter__()
                self._clients.append(client)

                tools = await client.list_tools()
                for mcp_tool in tools:
                    wrapper = McpToolWrapper(mcp_tool, client)
                    ALL_TOOLS[wrapper.name] = wrapper  # type: ignore[assignment]
                    ALL_TOOL_SPECS.append(wrapper.spec())
                    self._tool_names.append(wrapper.name)
                    logger.info("Registered MCP tool: %s (from %s)", wrapper.name, url)
                    total += 1

                logger.info("Connected to MCP server %s — %d tools", url, len(tools))
            except Exception as exc:
                logger.warning("Failed to connect to MCP server %s: %s", url, exc)

        return total

    async def shutdown(self) -> None:
        """Disconnect all MCP clients and remove their tools from the registry."""
        from reachy_mini_conversation_app.tools.core_tools import ALL_TOOLS, ALL_TOOL_SPECS

        for name in self._tool_names:
            ALL_TOOLS.pop(name, None)

        # Remove specs whose name matches an MCP tool
        mcp_names = set(self._tool_names)
        ALL_TOOL_SPECS[:] = [s for s in ALL_TOOL_SPECS if s.get("name") not in mcp_names]

        for client in self._clients:
            try:
                await client.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("Error closing MCP client: %s", exc)

        self._clients.clear()
        self._tool_names.clear()
        logger.info("MCP clients shut down")


async def register_mcp_tools() -> int:
    """Register tools from configured MCP servers.

    Reads ``MCP_SERVER_URLS`` from the environment (via ``config``).
    Returns the number of tools registered, or ``0`` when not configured.
    Idempotent — subsequent calls are no-ops while the manager is alive.
    """
    global _manager

    if _manager is not None:
        logger.debug("MCP tools already registered; skipping.")
        return 0

    urls = _parse_mcp_urls(getattr(config, "MCP_SERVER_URLS", None))
    if not urls:
        logger.debug("No MCP_SERVER_URLS configured; skipping MCP registration.")
        return 0

    _manager = McpClientManager()
    count = await _manager.connect_and_register(urls)
    logger.info("MCP registration complete: %d tools from %d server(s)", count, len(urls))
    return count


async def shutdown_mcp() -> None:
    """Shut down the MCP client manager.  Safe to call when not initialised."""
    global _manager

    if _manager is None:
        return

    await _manager.shutdown()
    _manager = None
