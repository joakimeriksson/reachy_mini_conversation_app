"""Tests for MCP client configuration parsing."""

from reachy_local_assistant.mcp_client import McpServerConfig, _parse_mcp_servers


class TestParseMcpServers:
    """Tests for _parse_mcp_servers."""

    def test_none_returns_empty(self) -> None:
        """Return empty list for None input."""
        assert _parse_mcp_servers(None) == []

    def test_empty_string_returns_empty(self) -> None:
        """Return empty list for empty string."""
        assert _parse_mcp_servers("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        """Return empty list for whitespace-only string."""
        assert _parse_mcp_servers("   ") == []

    def test_single_url(self) -> None:
        """Parse a single URL without token."""
        result = _parse_mcp_servers("http://host:8000/sse")
        assert result == [McpServerConfig(url="http://host:8000/sse", token=None)]

    def test_multiple_urls(self) -> None:
        """Parse comma-separated URLs without tokens."""
        result = _parse_mcp_servers("http://a:8000/sse,http://b:9000/mcp")
        assert result == [
            McpServerConfig(url="http://a:8000/sse", token=None),
            McpServerConfig(url="http://b:9000/mcp", token=None),
        ]

    def test_single_url_with_token(self) -> None:
        """Parse a URL with a bearer token."""
        result = _parse_mcp_servers("https://secure.host/mcp token=eyJhbGci.payload.sig")
        assert result == [McpServerConfig(url="https://secure.host/mcp", token="eyJhbGci.payload.sig")]

    def test_mixed_with_and_without_token(self) -> None:
        """Parse mix of URLs with and without tokens."""
        result = _parse_mcp_servers("http://a:8000/sse,https://b/mcp token=mytoken123")
        assert result == [
            McpServerConfig(url="http://a:8000/sse", token=None),
            McpServerConfig(url="https://b/mcp", token="mytoken123"),
        ]

    def test_whitespace_tolerance(self) -> None:
        """Handle extra whitespace around entries."""
        result = _parse_mcp_servers("  http://a:8000/sse , https://b/mcp token=tok  ")
        assert result == [
            McpServerConfig(url="http://a:8000/sse", token=None),
            McpServerConfig(url="https://b/mcp", token="tok"),
        ]

    def test_trailing_comma_ignored(self) -> None:
        """Ignore trailing comma in the input."""
        result = _parse_mcp_servers("http://a:8000/sse,")
        assert result == [McpServerConfig(url="http://a:8000/sse", token=None)]

    def test_single_url_with_api_key(self) -> None:
        """Parse a URL with an API key for token exchange."""
        result = _parse_mcp_servers("https://dirigera.botbox.se/mcp api_key=abc123")
        assert result == [McpServerConfig(url="https://dirigera.botbox.se/mcp", token=None, api_key="abc123")]

    def test_mixed_token_and_api_key(self) -> None:
        """Parse mix of token and api_key entries."""
        result = _parse_mcp_servers("https://a/mcp token=tok1,https://b/mcp api_key=key2")
        assert result == [
            McpServerConfig(url="https://a/mcp", token="tok1", api_key=None),
            McpServerConfig(url="https://b/mcp", token=None, api_key="key2"),
        ]

    def test_unknown_suffix_ignored(self) -> None:
        """Non-token= suffix after whitespace is ignored (only URL is kept)."""
        result = _parse_mcp_servers("http://a:8000/sse something_else")
        assert result == [McpServerConfig(url="http://a:8000/sse", token=None)]
