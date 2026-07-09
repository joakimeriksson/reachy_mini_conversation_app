"""Bidirectional local audio stream with optional settings UI.

The conversation backend is fully local (Ollama + Piper) and needs no API key.
In headless mode there is no Gradio UI, but when a Reachy Mini Apps settings
server is available we still expose a minimal settings page (served from this
package's ``static/`` folder) with personality selection and MCP server config.
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Optional
from pathlib import Path

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from reachy_local_assistant.config import LOCKED_PROFILE, config
from reachy_local_assistant.audio.dsp import to_mono
from reachy_local_assistant.ollama_handler import OllamaConversationHandler
from reachy_local_assistant.headless_personality_ui import mount_personality_routes


try:
    # FastAPI is provided by the Reachy Mini Apps runtime
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore
    FileResponse = object  # type: ignore
    JSONResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    BaseModel = object  # type: ignore


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: OllamaConversationHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
    ):
        """Initialize the stream with a conversation handler and media pipelines.

        - ``settings_app``: the Reachy Mini Apps FastAPI to attach settings endpoints.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        """
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._settings_initialized = False
        self._asyncio_loop = None

    # ---- Settings UI (only when API key is missing) ----
    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        inst = env_path.parent
        try:
            if env_path.exists():
                try:
                    return env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return []
            template_text = None
            ex = inst / ".env.example"
            if ex.exists():
                try:
                    template_text = ex.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                try:
                    cwd_example = Path.cwd() / ".env.example"
                    if cwd_example.exists():
                        template_text = cwd_example.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                packaged = Path(__file__).parent / ".env.example"
                if packaged.exists():
                    try:
                        template_text = packaged.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None
            return template_text.splitlines() if template_text else []
        except Exception:
            return []


    def _persist_personality(self, profile: Optional[str]) -> None:
        """Persist the startup personality to the instance .env and config."""
        if LOCKED_PROFILE is not None:
            return
        selection = (profile or "").strip() or None
        try:
            from reachy_local_assistant.config import set_custom_profile

            set_custom_profile(selection)
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                    if selection:
                        lines[i] = f"REACHY_MINI_CUSTOM_PROFILE={selection}"
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if selection and not replaced:
                lines.append(f"REACHY_MINI_CUSTOM_PROFILE={selection}")
            if selection is None and not env_path.exists():
                return
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted startup personality to %s", env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist REACHY_MINI_CUSTOM_PROFILE: %s", e)

    def _persist_mcp_servers(self, csv_value: Optional[str]) -> None:
        """Persist MCP_SERVER_URLS to the instance .env file."""
        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith("MCP_SERVER_URLS="):
                    if csv_value:
                        lines[i] = f'MCP_SERVER_URLS="{csv_value}"'
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if csv_value and not replaced:
                lines.append(f'MCP_SERVER_URLS="{csv_value}"')
            if csv_value is None and not env_path.exists():
                return
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted MCP_SERVER_URLS to %s", env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist MCP_SERVER_URLS: %s", e)

    def _persist_env_var(self, key: str, value: Optional[str], quote: bool = True) -> None:
        """Persist a single ``KEY=value`` to the instance .env (replace/append/remove)."""
        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            rendered = (f'{key}="{value}"' if quote else f"{key}={value}") if value else None
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith(f"{key}="):
                    if value and rendered is not None:
                        lines[i] = rendered
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if value and rendered is not None and not replaced:
                lines.append(rendered)
            if not value and not env_path.exists():
                return
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Persisted %s to %s", key, env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist %s: %s", key, e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read persisted startup personality from instance .env (if any)."""
        if not self._instance_path:
            return None
        env_path = Path(self._instance_path) / ".env"
        try:
            if env_path.exists():
                for ln in env_path.read_text(encoding="utf-8").splitlines():
                    if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                        _, _, val = ln.partition("=")
                        v = val.strip()
                        return v or None
        except Exception:
            pass
        return None

    def _init_settings_ui_if_needed(self) -> None:
        """Attach minimal settings UI to the settings app.

        Always mounts the UI when a settings_app is provided so that users
        see a confirmation message even if the API key is already configured.
        """
        if self._settings_initialized:
            return
        if self._settings_app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(self._settings_app, "mount"):
            try:
                # Serve /static/* assets
                self._settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                pass

        # GET / -> index.html (local-backend status/landing page)
        @self._settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        # GET /favicon.ico -> optional, avoid noisy 404s on some browsers
        @self._settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        # GET /ready -> whether backend finished loading tools
        @self._settings_app.get("/ready")
        def _ready() -> JSONResponse:
            try:
                mod = sys.modules.get("reachy_local_assistant.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        # ---------- MCP routes ----------

        @self._settings_app.get("/mcp/status")
        def _mcp_status() -> JSONResponse:
            servers = getattr(config, "MCP_SERVER_URLS", None) or ""
            from reachy_local_assistant.mcp_client import _manager

            tool_count = len(_manager._tool_names) if _manager is not None else 0
            return JSONResponse({"servers": servers, "tool_count": tool_count})

        class McpConnectPayload(BaseModel):
            servers: str

        @self._settings_app.post("/mcp/connect")
        def _mcp_connect(payload: McpConnectPayload) -> JSONResponse:
            from reachy_local_assistant.mcp_client import shutdown_mcp, register_mcp_tools

            raw = payload.servers.strip()
            csv_value = ",".join(line.strip() for line in raw.splitlines() if line.strip())

            # Update config and env (sync — safe from any thread)
            config.MCP_SERVER_URLS = csv_value if csv_value else None
            if csv_value:
                os.environ["MCP_SERVER_URLS"] = csv_value
            else:
                os.environ.pop("MCP_SERVER_URLS", None)
            self._persist_mcp_servers(csv_value if csv_value else None)

            if not csv_value:
                return JSONResponse({"status": "Disconnected. No servers configured.", "tool_count": 0})

            # Schedule async MCP work on the main event loop (avoid cross-loop errors)
            loop = self._asyncio_loop
            if loop is None:
                return JSONResponse({"error": "Event loop not ready"}, status_code=503)

            async def _do_connect() -> dict[str, object]:
                await shutdown_mcp()
                count = await register_mcp_tools()
                try:
                    await self.handler._restart_session()
                    return {
                        "status": f"Connected! {count} MCP tool(s) registered. Session restarted.",
                        "tool_count": count,
                    }
                except Exception as e:
                    return {
                        "status": f"Connected {count} tool(s), but session restart failed: {e}",
                        "tool_count": count,
                    }

            try:
                import asyncio as _aio

                fut = _aio.run_coroutine_threadsafe(_do_connect(), loop)
                result = fut.result(timeout=30)
                return JSONResponse(result)
            except Exception as e:
                logger.error("MCP connect failed: %s", e)
                return JSONResponse({"error": str(e)}, status_code=500)

        # ---------- Local backend URLs (Ollama + remote TTS voice server) ----------

        @self._settings_app.get("/backends/status")
        def _backends_status() -> JSONResponse:
            return JSONResponse(
                {
                    "ollama_url": getattr(config, "OLLAMA_URL", "") or "",
                    "ollama_model": getattr(config, "OLLAMA_MODEL", "") or "",
                    "tts_backend": getattr(config, "TTS_BACKEND", "piper") or "piper",
                    "tts_url": getattr(config, "TTS_URL", "") or "",
                    "tts_voice": getattr(config, "TTS_VOICE", "") or "",
                }
            )

        class BackendsPayload(BaseModel):
            ollama_url: Optional[str] = None
            tts_url: Optional[str] = None
            tts_voice: Optional[str] = None

        @self._settings_app.post("/backends/save")
        def _backends_save(payload: BackendsPayload) -> JSONResponse:
            ollama = (payload.ollama_url or "").strip()
            tts_url = (payload.tts_url or "").strip()
            tts_voice = (payload.tts_voice or "").strip()
            changed = []
            if ollama:
                config.OLLAMA_URL = ollama
                os.environ["OLLAMA_URL"] = ollama
                self._persist_env_var("OLLAMA_URL", ollama)
                changed.append("OLLAMA_URL")
            # A TTS URL implies the remote voice-server backend.
            if tts_url:
                config.TTS_URL = tts_url
                config.TTS_BACKEND = "remote"
                os.environ["TTS_URL"] = tts_url
                os.environ["TTS_BACKEND"] = "remote"
                self._persist_env_var("TTS_URL", tts_url)
                self._persist_env_var("TTS_BACKEND", "remote")
                changed.append("TTS_URL")
            if tts_voice:
                config.TTS_VOICE = tts_voice
                os.environ["TTS_VOICE"] = tts_voice
                self._persist_env_var("TTS_VOICE", tts_voice)
                changed.append("TTS_VOICE")
            if not changed:
                return JSONResponse({"ok": False, "status": "Nothing to save."}, status_code=400)
            # Persisted to the instance .env and applied to the live config/env; the
            # backends (Ollama client + TTS) are built at start_up(), so a restart of
            # the app picks them up. Use the dashboard's "Restart" for the app.
            status = f"Saved {', '.join(changed)} to the robot. Restart the app to apply."
            return JSONResponse(
                {
                    "ok": True,
                    "status": status,
                    "ollama_url": getattr(config, "OLLAMA_URL", ""),
                    "tts_backend": getattr(config, "TTS_BACKEND", ""),
                    "tts_url": getattr(config, "TTS_URL", ""),
                    "tts_voice": getattr(config, "TTS_VOICE", ""),
                }
            )

        self._settings_initialized = True

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        Loads the instance ``.env`` (profile / MCP / Ollama+Piper config) and,
        when available, mounts the settings UI before starting the media streams.
        """
        self._stop_event.clear()

        # Try to load an existing instance .env first (covers subsequent runs)
        if self._instance_path:
            try:
                from dotenv import load_dotenv

                from reachy_local_assistant.config import set_custom_profile

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    # Update config with newly loaded values (profile, MCP, Ollama/Piper)
                    if LOCKED_PROFILE is None:
                        new_profile = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
                        if new_profile is not None:
                            try:
                                set_custom_profile(new_profile.strip() or None)
                            except Exception:
                                pass  # Best-effort profile update
            except Exception:
                pass  # Instance .env loading is optional; continue with defaults

        # The local backend (Ollama + Piper) needs no API key. Still expose the
        # settings UI (personality/MCP routes) when a settings app is available.
        self._init_settings_ui_if_needed()

        # Start media
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]
            # Mount personality routes now that loop and handler are available
            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        self.handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                    )
            except Exception:
                pass
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="ollama-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (ollama-handler, record-loop, play-loop)
        """
        logger.info("Stopping LocalStream...")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback (may already be stopped): {e}")

        # Signal stop + cancel tasks ON the loop thread. _stop_event and the
        # tasks are owned by the asyncio loop; mutating them from this watcher
        # thread is unsafe, so marshal via call_soon_threadsafe (#414).
        loop = self._asyncio_loop
        if loop is None or not loop.is_running():
            self._stop_event.set()
            return
        loop.call_soon_threadsafe(self._stop_event.set)
        for task in self._tasks:
            if not task.done():
                loop.call_soon_threadsafe(task.cancel)

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        backend = getattr(self._robot.media, "backend", None)
        audio = getattr(self._robot.media, "audio", None)

        def _try_clear(name: str) -> bool:
            fn = getattr(audio, name, None)
            if callable(fn):
                fn()
                return True
            return False

        if audio is not None:
            # Prefer the backend's native flush; fall back to whichever clear method exists.
            preferred = "clear_player" if backend == MediaBackend.LOCAL else "clear_output_buffer"
            fallback = "clear_output_buffer" if preferred == "clear_player" else "clear_player"
            _try_clear(preferred) or _try_clear(fallback)
        self.handler.output_queue = asyncio.Queue()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug(f"Audio recording started at {input_sample_rate} Hz")

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                # Mono + float32 for the robot media sink
                audio_frame = audio_to_float32(to_mono(audio_data))

                # Resample if needed
                if input_sample_rate != output_sample_rate:
                    audio_frame = resample(
                        audio_frame,
                        int(len(audio_frame) * output_sample_rate / input_sample_rate),
                    )

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
