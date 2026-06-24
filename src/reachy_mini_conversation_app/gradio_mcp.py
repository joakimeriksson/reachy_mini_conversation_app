"""Gradio MCP server configuration UI components and wiring.

This module encapsulates the UI elements for configuring remote MCP
(Model Context Protocol) servers so that ``main.py`` stays lean.
"""

from __future__ import annotations
import os
import logging
from typing import Any

import gradio as gr

from .config import config


logger = logging.getLogger(__name__)


class McpUI:
    """Container for MCP server configuration Gradio components."""

    def __init__(self) -> None:
        """Initialize the McpUI instance."""
        self.servers_textarea: gr.TextArea
        self.connect_btn: gr.Button
        self.status_md: gr.Markdown

    def create_components(self) -> None:
        """Instantiate Gradio components for MCP server configuration."""
        # Pre-populate from config, converting commas to newlines for display
        initial_value = getattr(config, "MCP_SERVER_URLS", None) or ""
        if initial_value:
            initial_value = "\n".join(entry.strip() for entry in initial_value.split(",") if entry.strip())

        self.servers_textarea = gr.TextArea(
            label="MCP Servers (one per line, optional: url token=jwt)",
            placeholder="http://host:8000/sse\nhttps://other/mcp token=eyJ...",
            lines=3,
            value=initial_value,
        )
        self.connect_btn = gr.Button("Connect MCP Servers")
        self.status_md = gr.Markdown(value="")

    def additional_inputs_ordered(self) -> list[Any]:
        """Return the additional inputs in the expected order for Stream."""
        return [
            self.servers_textarea,
            self.connect_btn,
            self.status_md,
        ]

    def wire_events(self, handler: Any, blocks: gr.Blocks) -> None:
        """Attach event handlers to components within a Blocks context."""

        async def _connect_mcp(servers_text: str) -> str:
            from .mcp_client import shutdown_mcp, register_mcp_tools

            try:
                # 1. Disconnect existing MCP servers
                await shutdown_mcp()

                # 2. Convert newlines to commas for config format
                raw = servers_text.strip()
                csv_value = ",".join(line.strip() for line in raw.splitlines() if line.strip())

                # 3. Update config and environment
                config.MCP_SERVER_URLS = csv_value if csv_value else None
                if csv_value:
                    os.environ["MCP_SERVER_URLS"] = csv_value
                else:
                    os.environ.pop("MCP_SERVER_URLS", None)

                if not csv_value:
                    return "MCP servers disconnected. No servers configured."

                # 4. Register tools from new config
                # The Ollama chat loop reads the live tool registry on every turn,
                # so newly registered MCP tools take effect without a session restart.
                count = await register_mcp_tools()
                return f"Connected! {count} MCP tool(s) registered."

            except Exception as e:
                logger.error("MCP connect failed: %s", e)
                return f"Error: {e}"

        with blocks:
            self.connect_btn.click(
                fn=_connect_mcp,
                inputs=[self.servers_textarea],
                outputs=[self.status_md],
            )
