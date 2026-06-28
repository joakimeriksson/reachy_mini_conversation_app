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

from reachy_local_assistant.config import config, set_custom_profile
from reachy_local_assistant.prompts import get_session_voice, get_session_instructions
from reachy_local_assistant.audio.dsp import to_mono, resample_int16
from reachy_local_assistant.audio.tts import make_tts
from reachy_local_assistant.audio.vad import VAD_SAMPLE_RATE, VadSegmenter
from reachy_local_assistant.mcp_client import shutdown_mcp, register_mcp_tools
from reachy_local_assistant.audio.gemma_stt import GemmaSTT
from reachy_local_assistant.llm.ollama_chat import OllamaChat
from reachy_local_assistant.tools.core_tools import ToolDependencies
from reachy_local_assistant.conversation.turn import generate_reply
from reachy_local_assistant.conversation.speech import stream_sentences


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
            backend=config.VAD_BACKEND,
            threshold=config.VAD_THRESHOLD,
        )
        # Backends are built in start_up() so construction stays cheap/import-safe.
        self._stt: Optional[GemmaSTT] = None
        self._chat: Optional[OllamaChat] = None
        self._tts: Any = None  # PiperTTS or RemoteTTS (audio.tts.make_tts)
        self._voice = config.PIPER_VOICE

        self._speaking = False  # True while synthesizing/playing a reply (half-duplex gate)
        self._stop = asyncio.Event()
        self._interrupt = asyncio.Event()  # set when the user barges in mid-reply
        self._barge_vad: Optional[VadSegmenter] = None  # strict VAD for barge-in (built if BARGE_IN)
        self._aec: Any = None  # EchoCanceller when AEC enabled (cleans mic of Reachy's voice)
        self.last_activity_time = asyncio.get_event_loop().time()

    def copy(self) -> "OllamaConversationHandler":
        """Return a fresh handler (required by the fastrtc Stream)."""
        return OllamaConversationHandler(self.deps, self.gradio_mode, self.instance_path)

    # --- lifecycle -------------------------------------------------------

    async def start_up(self) -> None:
        """Build backends, register MCP tools, and run the conversation loop."""
        instructions = get_session_instructions(self.instance_path)
        if config.OLLAMA_DIRECT_AUDIO:
            instructions += (
                "\n\nThe user speaks to you through attached audio. Listen and respond "
                "directly and naturally to what they say — never transcribe or repeat it back."
            )
        self._stt = GemmaSTT(config.OLLAMA_STT_MODEL, config.OLLAMA_URL)
        self._chat = OllamaChat(config.OLLAMA_MODEL, config.OLLAMA_URL, self.deps, instructions)
        self._tts = make_tts()
        self._voice = get_session_voice()
        if config.BARGE_IN:
            self._barge_vad = VadSegmenter(
                aggressiveness=3,  # strict — resist playback echo
                silence_ms=200,
                min_speech_ms=config.BARGE_IN_SPEECH_MS,
                max_utterance_ms=config.VAD_MAX_UTTERANCE_MS,
                backend=config.VAD_BACKEND,
                threshold=config.VAD_THRESHOLD,
            )
            logger.info("Barge-in enabled (sustained speech >= %d ms)", config.BARGE_IN_SPEECH_MS)
        if config.AEC:
            try:
                from reachy_local_assistant.audio.aec import EchoCanceller

                self._aec = EchoCanceller(stream_delay_ms=config.AEC_STREAM_DELAY_MS)
                logger.info("AEC enabled (echo cancellation; stream delay %d ms)", config.AEC_STREAM_DELAY_MS)
            except Exception as exc:  # missing livekit / init failure shouldn't kill the app
                logger.warning("AEC requested but unavailable (%s); install the 'aec' extra", exc)

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
        self._interrupt.clear()
        if self._barge_vad is not None:
            self._barge_vad.reset()
        self._speaking = True  # gate the mic for the whole turn (half-duplex)
        self._set_listening(False)
        try:

            async def _show_user(text: str) -> None:
                # Show the user's turn (transcript, or a mic glyph in direct-audio mode).
                await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text or "🎤 …"}))

            turn = await generate_reply(
                self._stt,
                self._chat,
                utterance,
                direct_audio=config.OLLAMA_DIRECT_AUDIO,
                sample_rate=VAD_SAMPLE_RATE,
                on_user_text=_show_user,
            )
            self.last_activity_time = asyncio.get_event_loop().time()
            if turn.reply:
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": turn.reply}))
                await self._speak(turn.reply, turn.language)
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

    async def _speak(self, text: str, language: str | None = None) -> None:
        """Synthesize *text* sentence-by-sentence and stream each to wobbler + player.

        Splitting lets the first sentence start playing while later ones are still
        synthesizing (low latency to first word), and keeps each utterance within
        Kokoro's per-utterance token limit. *language* (from STT) tells a
        multilingual voice server which language to speak.
        """
        assert self._tts
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

        loop = asyncio.get_running_loop()
        total_samples = 0
        play_start: float | None = None

        def should_stop() -> bool:
            return self._stop.is_set() or self._interrupt.is_set()

        async for src_rate, pcm in stream_sentences(
            self._tts, text, voice=self._voice, language=language, should_stop=should_stop, loop=loop
        ):
            pcm24 = resample_int16(pcm, src_rate, WOBBLER_SAMPLE_RATE)
            if play_start is None:
                play_start = loop.time()  # the first audio is about to play
            total_samples += len(pcm24)
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(base64.b64encode(pcm24.tobytes()).decode("utf-8"))
            await self.output_queue.put((WOBBLER_SAMPLE_RATE, pcm24))

        if play_start is None:
            return
        # Hold the half-duplex mic gate until playback actually finishes. Audio began
        # playing at `play_start`; subtract the synthesis time already elapsed so we
        # don't over-hold (releasing early would let the mic capture Reachy's voice).
        duration = total_samples / WOBBLER_SAMPLE_RATE
        remaining = max(0.0, (play_start + duration) - loop.time())
        try:
            # Wait out the real playback time, but cut it short if the user barges in.
            await asyncio.wait_for(self._interrupt.wait(), timeout=remaining + PLAYBACK_TAIL_S)
            self._flush_output()  # barged in during playback — drop the rest
        except asyncio.TimeoutError:
            pass  # played to completion

    # --- audio I/O (driven by LocalStream / fastrtc) ---------------------

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Feed a mic frame into the VAD; enqueue completed utterances.

        While Reachy speaks the mic is normally gated (half-duplex). With BARGE_IN
        on, a strict VAD watches for the user talking over playback and interrupts
        (flush queued audio, stop the reply, re-listen).
        """
        sample_rate, audio = frame
        audio = to_mono(audio)
        pcm16 = resample_int16(audio, sample_rate, VAD_SAMPLE_RATE)
        if self._aec is not None:
            pcm16 = self._aec.clean(pcm16)  # remove Reachy's echo (10 ms-aligned; may buffer)
            if len(pcm16) == 0:
                return

        if self._speaking:
            if self._barge_vad is None:
                return  # half-duplex: ignore the mic while Reachy talks
            self._barge_vad.feed(pcm16)
            if self._barge_vad.in_speech and not self._interrupt.is_set():
                logger.info("Barge-in: user spoke over playback — interrupting reply")
                self._interrupt.set()
                self._flush_output()
            return

        # Live VAD tuning: rebuild the segmenter if the config values changed.
        if (config.VAD_AGGRESSIVENESS, config.VAD_SILENCE_MS) != (self._vad_aggr, self._vad_silence):
            self._vad_aggr = config.VAD_AGGRESSIVENESS
            self._vad_silence = config.VAD_SILENCE_MS
            self._vad = VadSegmenter(
                aggressiveness=self._vad_aggr,
                silence_ms=self._vad_silence,
                min_speech_ms=config.VAD_MIN_SPEECH_MS,
                max_utterance_ms=config.VAD_MAX_UTTERANCE_MS,
                backend=config.VAD_BACKEND,
                threshold=config.VAD_THRESHOLD,
            )
            logger.info("VAD updated: aggressiveness=%d silence_ms=%d", self._vad_aggr, self._vad_silence)

        for utterance in self._vad.feed(pcm16):
            await self._utterances.put(utterance)

    def _flush_output(self) -> None:
        """Drop queued output audio (and reset the wobbler) so playback stops fast."""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Return the next output item (audio chunk or transcript).

        When AEC is on, the audio handed to the player is also fed to the canceller
        as the far-end reference (resampled to 16 kHz) so it can subtract the echo.
        """
        item = await wait_for_item(self.output_queue)
        if self._aec is not None and isinstance(item, tuple):
            rate, pcm = item
            self._aec.play_reference(resample_int16(pcm, rate, VAD_SAMPLE_RATE))
        return item  # type: ignore[no-any-return]

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

