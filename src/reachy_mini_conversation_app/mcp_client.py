"""MCP (Model Context Protocol) client integration for external tool servers.

Connects to remote MCP servers via SSE or Streamable HTTP transport and
exposes their tools alongside the built-in local tools.  Fully optional
— the app works unchanged when no ``MCP_SERVER_URLS`` is configured and
``fastmcp`` is not installed.
"""

from __future__ import annotations
import os
import json
import logging
import dataclasses
from typing import Any, Dict, List

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

# Module-level manager singleton (set by register_mcp_tools, cleared by shutdown_mcp)
_manager: McpClientManager | None = None


@dataclasses.dataclass
class McpServerConfig:
    """Configuration for a single MCP server endpoint."""

    url: str
    token: str | None = None
    api_key: str | None = None


def _parse_mcp_servers(raw: str | None) -> list[McpServerConfig]:
    """Parse MCP server config from a comma-separated string.

    Each entry is a URL optionally followed by ``token=<bearer_token>``.
    Plain URLs (no token) are supported for backward compatibility.

    Examples::

        "http://host:8000/sse"
        "http://host:8000/sse,https://other/mcp token=eyJ..."
        "https://dirigera.botbox.se/mcp api_key=a574..."
    """
    if not raw or not raw.strip():
        return []
    servers: list[McpServerConfig] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(None, 1)  # split on first whitespace
        url = parts[0]
        token = None
        api_key = None
        if len(parts) > 1:
            suffix = parts[1]
            if suffix.startswith("token="):
                token = suffix[len("token=") :]
            elif suffix.startswith("api_key="):
                api_key = suffix[len("api_key=") :]
        servers.append(McpServerConfig(url=url, token=token, api_key=api_key))
    return servers


async def _fetch_token(server: McpServerConfig) -> str:
    """Exchange an API key for a JWT token via the server's ``/auth/token`` endpoint.

    Derives the auth URL from the MCP URL by replacing the path with ``/auth/token``.
    """
    from urllib.parse import urljoin

    auth_url = urljoin(server.url, "/auth/token")
    logger.info("Fetching MCP token from %s", auth_url)

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(auth_url, json={"api_key": server.api_key})
            resp.raise_for_status()
            token = resp.json()["token"]
            logger.info("Successfully obtained MCP token from %s", auth_url)
            return token
    except Exception as exc:
        logger.error("Failed to fetch MCP token from %s: %s", auth_url, exc)
        raise


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
        logger.info("MCP tool '%s' calling with args: %s", self.name, kwargs)
        try:
            result = await self._client.call_tool(self._original_name, kwargs)
            logger.info("MCP tool '%s' raw result type: %s", self.name, type(result).__name__)
        except Exception as exc:
            logger.error("MCP tool '%s' call failed: %s", self.name, exc, exc_info=True)
            return {"error": f"MCP tool call failed: {exc}"}

        # Convert result → dict (handles both list[Content] and CallToolResult)
        try:
            content = result.content if hasattr(result, "content") else result
            if isinstance(content, list):
                items = content
            else:
                items = list(content)
            logger.info("MCP tool '%s' got %d content items", self.name, len(items))
        except Exception as exc:
            logger.error("MCP tool '%s' failed to iterate result: %s (result=%r)", self.name, exc, result)
            return {"error": f"Failed to parse MCP result: {exc}"}

        parts: List[str] = []
        for item in items:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)

        combined = "\n".join(parts) if parts else ""
        logger.info("MCP tool '%s' combined text length: %d chars", self.name, len(combined))

        # If the combined text is valid JSON dict, return it directly
        try:
            parsed = json.loads(combined)
            if isinstance(parsed, dict):
                logger.info("MCP tool '%s' returning parsed dict with keys: %s", self.name, list(parsed.keys()))
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        logger.info("MCP tool '%s' returning as plain text", self.name)
        return {"result": combined}


class McpClientManager:
    """Manages connections to one or more MCP servers."""

    def __init__(self) -> None:
        """Initialise with empty client list."""
        self._clients: List[Any] = []
        self._tool_names: List[str] = []

    async def connect_and_register(self, servers: list[McpServerConfig]) -> int:
        """Connect to each MCP server and register discovered tools.

        Returns the total number of MCP tools registered.
        """
        try:
            import fastmcp
            from fastmcp import Client

            logger.info("fastmcp version: %s", getattr(fastmcp, "__version__", "unknown"))
        except ImportError:
            logger.warning(
                "fastmcp is not installed — MCP tools will not be available. Install with: uv sync --extra mcp"
            )
            return 0

        from reachy_mini_conversation_app.tools.core_tools import ALL_TOOLS, ALL_TOOL_SPECS

        total = 0
        for server in servers:
            try:
                # Resolve auth: api_key → fetch JWT, token → use directly
                auth = server.token
                if server.api_key and not auth:
                    auth = await _fetch_token(server)

                # Pass URL directly — fastmcp auto-detects transport
                # (tries Streamable HTTP first, falls back to SSE)
                client = Client(server.url, auth=auth)
                await client.__aenter__()
                self._clients.append(client)

                tools = await client.list_tools()
                for mcp_tool in tools:
                    wrapper = McpToolWrapper(mcp_tool, client)
                    ALL_TOOLS[wrapper.name] = wrapper  # type: ignore[assignment]
                    ALL_TOOL_SPECS.append(wrapper.spec())
                    self._tool_names.append(wrapper.name)
                    logger.info("Registered MCP tool: %s (from %s)", wrapper.name, server.url)
                    total += 1

                logger.info("Connected to MCP server %s — %d tools", server.url, len(tools))
            except Exception as exc:
                logger.warning("Failed to connect to MCP server %s: %s", server.url, exc)

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

    raw = getattr(config, "MCP_SERVER_URLS", None) or os.environ.get("MCP_SERVER_URLS")
    servers = _parse_mcp_servers(raw)
    if not servers:
        logger.debug("No MCP_SERVER_URLS configured; skipping MCP registration.")
        return 0

    _manager = McpClientManager()
    count = await _manager.connect_and_register(servers)
    logger.info("MCP registration complete: %d tools from %d server(s)", count, len(servers))
    return count


async def shutdown_mcp() -> None:
    """Shut down the MCP client manager.  Safe to call when not initialised."""
    global _manager

    if _manager is None:
        return

    await _manager.shutdown()
    _manager = None
