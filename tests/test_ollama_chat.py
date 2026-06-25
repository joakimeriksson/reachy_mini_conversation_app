"""Tests for the Ollama chat loop: spec adaptation + tool-call dispatch.

``llm.ollama_chat`` imports ``tools.core_tools``, which pulls in the robot SDK.
To keep this a fast unit test (and runnable without the SDK), we inject a fake
``core_tools`` module via ``monkeypatch.setitem`` — automatically restored after
each test, so the real module is untouched for the rest of the suite.
"""

import sys
import types
import importlib

import pytest


@pytest.fixture
def chat_mod(monkeypatch):
    calls = []

    async def fake_dispatch(name, args_json, deps):
        calls.append((name, args_json, deps))
        return {"ok": True, "tool": name}

    fake = types.ModuleType("reachy_local_assistant.tools.core_tools")
    fake.ToolDependencies = object
    fake.dispatch_tool_call = fake_dispatch
    fake.get_tool_specs = lambda exclusion_list=[]: [
        {
            "type": "function",
            "name": "get_time",
            "description": "Get the time in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    fake._calls = calls

    monkeypatch.setitem(sys.modules, "reachy_local_assistant.tools.core_tools", fake)
    monkeypatch.delitem(sys.modules, "reachy_local_assistant.llm.ollama_chat", raising=False)
    mod = importlib.import_module("reachy_local_assistant.llm.ollama_chat")
    yield mod, fake
    monkeypatch.delitem(sys.modules, "reachy_local_assistant.llm.ollama_chat", raising=False)


class FakeOllamaClient:
    """Returns a scripted sequence of chat() responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat(self, model, messages, tools, think, stream, **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, **kwargs})
        return self._responses.pop(0)


def _patch_client(monkeypatch, client):
    import ollama

    monkeypatch.setattr(ollama, "AsyncClient", lambda host: client)


def test_to_ollama_tools_flat_to_nested(chat_mod):
    mod, fake = chat_mod
    nested = mod.to_ollama_tools(fake.get_tool_specs())
    assert nested == [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the time in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_to_ollama_tools_skips_non_function_specs(chat_mod):
    mod, _ = chat_mod
    assert mod.to_ollama_tools([{"type": "other", "name": "x"}]) == []


@pytest.mark.asyncio
async def test_respond_plain_text_no_tools(chat_mod, monkeypatch):
    mod, _ = chat_mod
    client = FakeOllamaClient([{"message": {"content": "Hi there!"}}])
    _patch_client(monkeypatch, client)

    chat = mod.OllamaChat("m", "http://x", deps=object(), system_prompt="sys")
    reply = await chat.respond("hello")

    assert reply == "Hi there!"
    assert len(client.calls) == 1
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "sys"}


@pytest.mark.asyncio
async def test_respond_runs_tool_then_returns_text(chat_mod, monkeypatch):
    mod, fake = chat_mod
    client = FakeOllamaClient(
        [
            {"message": {"content": "", "tool_calls": [
                {"function": {"name": "get_time", "arguments": {"city": "Paris"}}}
            ]}},
            {"message": {"content": "It is noon in Paris."}},
        ]
    )
    _patch_client(monkeypatch, client)

    chat = mod.OllamaChat("m", "http://x", deps="DEPS", system_prompt="sys")
    reply = await chat.respond("time in Paris?")

    assert reply == "It is noon in Paris."
    # Tool was dispatched with JSON-encoded args and the real deps object.
    assert fake._calls == [("get_time", '{"city": "Paris"}', "DEPS")]
    # Second round must include a tool-result message for context.
    second_round_msgs = client.calls[1]["messages"]
    assert any(m.get("role") == "tool" and m.get("tool_name") == "get_time" for m in second_round_msgs)


@pytest.mark.asyncio
async def test_respond_handles_string_arguments(chat_mod, monkeypatch):
    mod, fake = chat_mod
    client = FakeOllamaClient(
        [
            {"message": {"tool_calls": [
                {"function": {"name": "get_time", "arguments": '{"city": "Oslo"}'}}
            ]}},
            {"message": {"content": "done"}},
        ]
    )
    _patch_client(monkeypatch, client)
    chat = mod.OllamaChat("m", "http://x", deps=None, system_prompt="sys")
    await chat.respond("q")
    assert fake._calls[0][1] == '{"city": "Oslo"}'


@pytest.mark.asyncio
async def test_set_system_prompt_updates_in_place(chat_mod, monkeypatch):
    mod, _ = chat_mod
    client = FakeOllamaClient([{"message": {"content": "ok"}}])
    _patch_client(monkeypatch, client)
    chat = mod.OllamaChat("m", "http://x", deps=None, system_prompt="old")
    chat.set_system_prompt("new")
    await chat.respond("hi")
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "new"}


@pytest.mark.asyncio
async def test_respond_returns_empty_on_backend_error(chat_mod, monkeypatch):
    mod, _ = chat_mod

    class Boom:
        async def chat(self, **kwargs):
            raise RuntimeError("down")

    _patch_client(monkeypatch, Boom())
    chat = mod.OllamaChat("m", "http://x", deps=None, system_prompt="sys")
    assert await chat.respond("hi") == ""
