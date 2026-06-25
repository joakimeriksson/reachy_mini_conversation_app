"""Local conversation handler: Ollama (Gemma audio) + Piper TTS.

Drop-in replacement for ``OpenaiRealtimeHandler`` that runs the whole pipeline
locally instead of streaming to OpenAI. It implements the same interface the
``LocalStream`` loops and the fastrtc ``Stream`` drive — ``start_up``,
``receive``, ``emit``, ``shutdown``, ``copy`` — and feeds the head wobbler with
24 kHz PCM exactly as the OpenAI path did, so robot motion is unchanged.

Pipeline (turn-based):

    mic ──receive──▶ resample 16 kHz ──▶ VAD ──utterance──▶ Gemma STT ──text──▶
        Ollama chat (+tools) ──reply──▶ Piper TTS ──▶ wobbler + output_queue ──▶ speaker

Phase 1 is half-duplex: mic frames are ignored while Reachy is speaking. Phase 2
adds AEC + barge-in (see ``audio/aec.py``).
"""

from __future__ import annotations
import base64
import asyncio
import logging
from typing import Any, Tuple, Optional

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_local_assistant.config import config, set_custom_profile
from reachy_local_assistant.prompts import get_session_voice, get_session_instructions
from reachy_local_assistant.audio.tts import make_tts
from reachy_local_assistant.audio.vad import VAD_SAMPLE_RATE, VadSegmenter
from reachy_local_assistant.mcp_client import shutdown_mcp, register_mcp_tools
from reachy_local_assistant.audio.gemma_stt import GemmaSTT
from reachy_local_assistant.llm.ollama_chat import OllamaChat
from reachy_local_assistant.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)

# The head wobbler hardcodes 24 kHz; mirror the OpenAI output rate so motion timing
# and the downstream player resampling behave identically.
WOBBLER_SAMPLE_RATE = 24000

# Extra time to keep the mic gated after playback is queued, covering device /
# network / jitter-buffer latency so the tail of Reachy's speech isn't heard.
PLAYBACK_TAIL_S = 0.6


class OllamaConversationHandler(AsyncStreamHandler):
    """Turn-based local conversation handler (Ollama + Piper)."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        """Initialise the handler with its tool dependencies."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=WOBBLER_SAMPLE_RATE,
            input_sample_rate=WOBBLER_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        self._utterances: "asyncio.Queue[NDArray[np.int16]]" = asyncio.Queue()

        self._vad_aggr = config.VAD_AGGRESSIVENESS
        self._vad_silence = config.VAD_SILENCE_MS
        self._vad = VadSegmenter(
            aggressiveness=self._vad_aggr,
            silence_ms=self._vad_silence,
            min_speech_ms=config.VAD_MIN_SPEECH_MS,
            max_utterance_ms=config.VAD_MAX_UTTERANCE_MS,
        )
        # Backends are built in start_up() so construction stays cheap/import-safe.
        self._stt: Optional[GemmaSTT] = None
        self._chat: Optional[OllamaChat] = None
        self._tts: Any = None  # PiperTTS or RemoteTTS (audio.tts.make_tts)
        self._voice = config.PIPER_VOICE

        self._speaking = False  # True while synthesizing/playing a reply (half-duplex gate)
        self._stop = asyncio.Event()
        self.last_activity_time = asyncio.get_event_loop().time()

    def copy(self) -> "OllamaConversationHandler":
        """Return a fresh handler (required by the fastrtc Stream)."""
        return OllamaConversationHandler(self.deps, self.gradio_mode, self.instance_path)

    # --- lifecycle -------------------------------------------------------

    async def start_up(self) -> None:
        """Build backends, register MCP tools, and run the conversation loop."""
        instructions = get_session_instructions(self.instance_path)
        self._stt = GemmaSTT(config.OLLAMA_STT_MODEL, config.OLLAMA_URL)
        self._chat = OllamaChat(config.OLLAMA_MODEL, config.OLLAMA_URL, self.deps, instructions)
        self._tts = make_tts()
        self._voice = get_session_voice()

        try:
            count = await register_mcp_tools()
            if count:
                logger.info("Registered %d MCP tool(s)", count)
        except Exception as exc:
            logger.warning("MCP registration failed: %s", exc)

        self._set_listening(True)
        logger.info("Ollama conversation handler ready (model=%s)", config.OLLAMA_MODEL)

        while not self._stop.is_set():
            try:
                utterance = await self._utterances.get()
            except asyncio.CancelledError:
                break
            await self._handle_turn(utterance)

    async def _handle_turn(self, utterance: NDArray[np.int16]) -> None:
        assert self._stt and self._chat
        self._speaking = True  # gate the mic for the whole turn (half-duplex)
        self._set_listening(False)
        try:
            text, _lang = await self._stt.transcribe(utterance)
            if not text:
                logger.debug("Empty transcription; ignoring utterance")
                return
            self.last_activity_time = asyncio.get_event_loop().time()
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))

            reply = await self._chat.respond(text)
            if reply:
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": reply}))
                await self._speak(reply)
        except Exception as exc:
            logger.exception("Turn failed: %s", exc)
        finally:
            self._speaking = False
            self._vad.reset()  # drop any in-progress utterance captured at the edges
            # Drop any utterances that slipped into the queue during the turn
            # (e.g. echo captured in the brief gate transitions).
            while not self._utterances.empty():
                try:
                    self._utterances.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._set_listening(True)
            self.last_activity_time = asyncio.get_event_loop().time()

    async def _speak(self, text: str) -> None:
        """Synthesize *text* with Piper and stream it to wobbler + player."""
        assert self._tts
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

        loop = asyncio.get_running_loop()
        # Piper synthesis is blocking/CPU-bound — run it off the event loop.
        chunks = await loop.run_in_executor(None, self._synthesize, text)
        total_samples = 0
        for src_rate, pcm in chunks:
            if self._stop.is_set():
                break
            pcm24 = self._resample_int16(pcm, src_rate, WOBBLER_SAMPLE_RATE)
            total_samples += len(pcm24)
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(base64.b64encode(pcm24.tobytes()).decode("utf-8"))
            await self.output_queue.put((WOBBLER_SAMPLE_RATE, pcm24))

        # Hold the half-duplex mic gate for the ACTUAL playback time. Synthesis is
        # fast, but the audio plays out over `duration` (plus device/network
        # latency); releasing the gate at synthesis time lets the mic capture
        # Reachy's own voice and loop. Keep `_speaking` True until playback ends.
        duration = total_samples / WOBBLER_SAMPLE_RATE
        await asyncio.sleep(duration + PLAYBACK_TAIL_S)

    def _synthesize(self, text: str) -> list[Tuple[int, NDArray[np.int16]]]:
        assert self._tts
        return list(self._tts.synthesize(text, voice=self._voice))

    # --- audio I/O (driven by LocalStream / fastrtc) ---------------------

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Feed a mic frame into the VAD; enqueue completed utterances."""
        if self._speaking:
            return  # half-duplex: ignore the mic while Reachy talks (Phase 1)

        # Live VAD tuning: rebuild the segmenter if the config values changed.
        if (config.VAD_AGGRESSIVENESS, config.VAD_SILENCE_MS) != (self._vad_aggr, self._vad_silence):
            self._vad_aggr = config.VAD_AGGRESSIVENESS
            self._vad_silence = config.VAD_SILENCE_MS
            self._vad = VadSegmenter(
            aggressiveness=self._vad_aggr,
            silence_ms=self._vad_silence,
            min_speech_ms=config.VAD_MIN_SPEECH_MS,
            max_utterance_ms=config.VAD_MAX_UTTERANCE_MS,
        )
            logger.info("VAD updated: aggressiveness=%d silence_ms=%d", self._vad_aggr, self._vad_silence)

        sample_rate, audio = frame
        audio = self._to_mono(audio)
        pcm16 = self._resample_int16(audio, sample_rate, VAD_SAMPLE_RATE)
        for utterance in self._vad.feed(pcm16):
            await self._utterances.put(utterance)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Return the next output item (audio chunk or transcript)."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Stop the loop and disconnect MCP clients."""
        self._stop.set()
        try:
            await shutdown_mcp()
        except Exception as exc:
            logger.debug("MCP shutdown error: %s", exc)
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # --- personality -----------------------------------------------------

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new profile at runtime by swapping the chat system prompt."""
        try:
            set_custom_profile(profile)
            instructions = get_session_instructions(self.instance_path)
            self._voice = get_session_voice()
            if self._chat is not None:
                self._chat.set_system_prompt(instructions)
            return "Applied personality."
        except Exception as exc:
            logger.error("Failed to apply personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    # --- helpers ---------------------------------------------------------

    def _set_listening(self, listening: bool) -> None:
        mm = getattr(self.deps, "movement_manager", None)
        if mm is not None and hasattr(mm, "set_listening"):
            try:
                mm.set_listening(listening)
            except Exception as exc:
                logger.debug("set_listening failed: %s", exc)

    @staticmethod
    def _to_mono(audio: NDArray[np.int16]) -> NDArray[np.int16]:
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:  # scipy channels-last convention
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]
        return audio.reshape(-1)

    @staticmethod
    def _resample_int16(audio: NDArray[Any], src_rate: int, dst_rate: int) -> NDArray[np.int16]:
        audio = np.asarray(audio)
        if audio.dtype != np.int16:
            if np.issubdtype(audio.dtype, np.floating):
                audio = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        if src_rate == dst_rate:
            return audio
        n = int(len(audio) * dst_rate / src_rate)
        if n <= 0:
            return np.empty(0, dtype=np.int16)
        resampled = resample(audio.astype(np.float32), n)
        return np.clip(resampled, -32768, 32767).astype(np.int16)  # type: ignore[no-any-return]
