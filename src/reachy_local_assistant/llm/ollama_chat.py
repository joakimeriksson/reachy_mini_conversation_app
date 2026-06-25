"""Ollama chat loop with tool calling.

Drives a conversation against a local Ollama model, reusing the app's existing
tool registry (`tools.core_tools`) and MCP client unchanged — those sit above
the LLM, so only the tool-spec *shape* needs adapting (the registry stores flat
Realtime-style specs; Ollama's chat API wants the nested
``{"type":"function","function":{...}}`` form).

The conversation history lives in this object so personality/voice can change
between turns without losing context.
"""

from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List

from reachy_local_assistant.config import config


if TYPE_CHECKING:
    from reachy_local_assistant.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

# Cap tool-call iterations per user turn to avoid runaway loops.
MAX_TOOL_ROUNDS = 6


def to_ollama_tools(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert flat Realtime tool specs to Ollama/OpenAI nested function specs."""
    tools: List[Dict[str, Any]] = []
    for spec in specs:
        if spec.get("type") != "function":
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return tools


class OllamaChat:
    """Stateful chat session against an Ollama model with tool dispatch."""

    def __init__(
        self,
        model: str,
        host: str,
        deps: "ToolDependencies | None",
        system_prompt: str,
        enable_tools: bool = True,
    ) -> None:
        """Start a chat session against the given Ollama model."""
        import ollama  # lazy import: only needed for the local backend

        self._client = ollama.AsyncClient(host=host)
        self._model = model
        self._deps = deps
        self._enable_tools = enable_tools
        self._system_prompt = system_prompt
        self._messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def set_system_prompt(self, system_prompt: str) -> None:
        """Swap the system prompt (e.g. on personality change), keeping history."""
        self._system_prompt = system_prompt
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = system_prompt
        else:
            self._messages.insert(0, {"role": "system", "content": system_prompt})

    def reset(self) -> None:
        """Clear conversation history, keeping the current system prompt."""
        self._messages = [{"role": "system", "content": self._system_prompt}]

    async def respond(self, user_text: str, image: bytes | None = None) -> str:
        """Run one user turn (with tool calls) and return the assistant's text.

        *image* optionally attaches an encoded image (e.g. JPEG bytes) to the
        user turn, so a multimodal model like Gemma can "see" while it answers.
        """
        user_msg: Dict[str, Any] = {"role": "user", "content": user_text}
        if image is not None:
            user_msg["images"] = [image]
        self._messages.append(user_msg)
        tools = to_ollama_tools(self._get_tool_specs()) if self._enable_tools else []

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = await self._client.chat(
                    model=self._model,
                    messages=self._messages,
                    tools=tools or None,
                    think=config.OLLAMA_THINK,
                    stream=False,
                    options=self._options(),
                    keep_alive=config.OLLAMA_KEEP_ALIVE,
                )
            except Exception as exc:
                logger.error("Ollama chat request failed: %s", exc)
                return ""

            message = response.get("message", {}) or {}
            tool_calls = message.get("tool_calls") or []
            # Record the assistant turn (text and/or tool calls) for context.
            self._messages.append(self._normalize_assistant(message))

            if not tool_calls:
                return (message.get("content") or "").strip()

            for call in tool_calls:
                await self._run_tool_call(call)

        logger.warning("Tool-call loop hit MAX_TOOL_ROUNDS; returning best-effort text")
        return (self._messages[-1].get("content") or "").strip()

    @staticmethod
    def _options() -> Dict[str, Any] | None:
        """Ollama generation options from config (temperature, context length)."""
        opts: Dict[str, Any] = {}
        if config.OLLAMA_TEMPERATURE is not None:
            opts["temperature"] = config.OLLAMA_TEMPERATURE
        if config.OLLAMA_NUM_CTX > 0:
            opts["num_ctx"] = config.OLLAMA_NUM_CTX
        return opts or None

    @staticmethod
    def _get_tool_specs() -> List[Dict[str, Any]]:
        """Fetch tool specs lazily; degrade to none if the registry is unavailable."""
        try:
            from reachy_local_assistant.tools.core_tools import get_tool_specs

            return get_tool_specs()
        except Exception as exc:  # robot SDK / registry not importable (e.g. standalone runner)
            logger.debug("Tool registry unavailable; running tool-less: %s", exc)
            return []

    async def _run_tool_call(self, call: Dict[str, Any]) -> None:
        from reachy_local_assistant.tools.core_tools import dispatch_tool_call

        fn = call.get("function", {}) or {}
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        # Ollama may hand back a dict or a JSON string; dispatch wants a string.
        args_json = args if isinstance(args, str) else json.dumps(args or {})

        logger.info("LLM tool call: %s(%s)", name, args_json)
        result = await dispatch_tool_call(name, args_json, self._deps)  # type: ignore[arg-type]

        # If a tool returned an image (e.g. the camera), don't dump the base64 as
        # text — attach it so the multimodal model can actually see it next turn.
        image_b64 = result.pop("b64_im", None) if isinstance(result, dict) else None

        self._messages.append(
            {
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, default=str),
            }
        )
        if image_b64:
            self._messages.append(
                {"role": "user", "content": "(camera image)", "images": [image_b64]}
            )

    @staticmethod
    def _normalize_assistant(message: Dict[str, Any]) -> Dict[str, Any]:
        """Strip provider-internal fields, keep what's needed for context."""
        out: Dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            out["tool_calls"] = message["tool_calls"]
        return out
