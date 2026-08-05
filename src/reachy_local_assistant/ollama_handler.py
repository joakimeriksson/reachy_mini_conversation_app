"""Local conversation handler: Ollama (Gemma) + the remote voice server.

Runs the whole pipeline on hardware you own instead of streaming to a cloud API.
``LocalStream`` drives it through ``start_up`` / ``receive`` / ``emit`` /
``shutdown`` / ``copy``, and it feeds the head wobbler 24 kHz PCM so robot motion
is timed the same way regardless of the speech backend.

Pipeline (turn-based):

    mic ──receive──▶ resample 16 kHz ──▶ VAD ──utterance──▶ Ollama chat
        (audio in directly, or via Gemma STT first) ──reply──▶ voice server TTS
        ──▶ wobbler + output_queue ──▶ speaker

Half-duplex by default: mic frames are ignored while Reachy speaks. With
``BARGE_IN`` a stricter VAD listens through playback so the user can interrupt,
and ``AEC`` removes Reachy's own voice from the mic so that works on open
speakers (see ``audio/aec.py``).
"""

from __future__ import annotations
import math
import base64
import asyncio
import logging
from typing import Any, Tuple, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from reachy_local_assistant.config import config, set_custom_profile
from reachy_local_assistant.health import BackendHealth, log_health, check_backends
from reachy_local_assistant.prompts import get_session_voice, get_session_instructions
from reachy_local_assistant.language import LanguageHistory
from reachy_local_assistant.audio.dsp import to_mono, resample_int16
from reachy_local_assistant.audio.tts import make_tts
from reachy_local_assistant.audio.vad import VAD_SAMPLE_RATE, VadSegmenter
from reachy_local_assistant.mcp_client import shutdown_mcp, register_mcp_tools
from reachy_local_assistant.transcript import PENDING_USER_TEXT, Transcript
from reachy_local_assistant.stream_shim import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from reachy_local_assistant.audio.gemma_stt import GemmaSTT
from reachy_local_assistant.llm.ollama_chat import OllamaChat
from reachy_local_assistant.audio.remote_stt import RemoteSTT
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
    """Turn-based local conversation handler (Ollama + remote voice server)."""

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
        # Flush hook installed by LocalStream: drops audio already handed to the
        # robot's player. Draining output_queue alone is not enough to stop a reply —
        # frames pushed into the player's appsrc keep playing until it is flushed too.
        self._clear_queue: Optional[Callable[[], None]] = None

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
        self._tts: Any = None  # RemoteTTS (audio.tts.make_tts)
        self._stt_remote: Optional[RemoteSTT] = None  # Whisper transcript for text history (direct-audio)
        self._voice = config.TTS_VOICE

        self._speaking = False  # True while synthesizing/playing a reply (half-duplex gate)
        self._stop = asyncio.Event()
        self._interrupt = asyncio.Event()  # set when the user barges in mid-reply
        self._barge_vad: Optional[VadSegmenter] = None  # strict VAD for barge-in (built if BARGE_IN)
        self._aec: Any = None  # EchoCanceller when AEC enabled (cleans mic of Reachy's voice)
        # AEC debug meter (AEC_DEBUG): per-turn echo/residual energy through the live path.
        self._aec_debug = config.AEC_DEBUG
        self._erle_pre_sq = self._erle_post_sq = 0.0
        self._erle_pre_n = self._erle_post_n = 0
        # Startup echo-delay auto-calibration (captures the mic while a probe plays).
        self._calibrating = False
        self._calib_near: list[NDArray[np.int16]] = []
        # Last backend reachability result, surfaced on the settings page.
        self.health = BackendHealth()
        # Human-readable conversation log for the settings page (the LLM's own
        # history lives in OllamaChat and carries prompts/tool calls/audio).
        self.transcript = Transcript()
        self._pending_user_seq: Optional[int] = None
        # Which language the conversation is in, from Whisper's per-turn detection.
        # Passed to the voice server as a tie-breaker for short replies.
        self._lang_history = LanguageHistory()
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
        self._stt_remote = RemoteSTT(config.STT_URL)
        self._voice = get_session_voice()
        if config.BARGE_IN:
            self._barge_vad = VadSegmenter(
                aggressiveness=3,  # strict — resist playback echo
                silence_ms=200,
                min_speech_ms=config.BARGE_IN_SPEECH_MS,
                max_utterance_ms=config.VAD_MAX_UTTERANCE_MS,
                backend=config.VAD_BACKEND,
                threshold=config.BARGE_IN_THRESHOLD,  # stricter than listening, resists noise/echo
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

        await self.refresh_health()

        self._set_listening(True)
        logger.info("Ollama conversation handler ready (model=%s)", config.OLLAMA_MODEL)

        if self._aec is not None and config.AEC_STREAM_DELAY_MS == 0:
            await asyncio.sleep(0.3)  # let the record/play loops spin up first
            await self._calibrate_echo()

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

            # Remembered so the Whisper transcript can upgrade this turn in place
            # rather than appending a second line for the same utterance.
            self._pending_user_seq = None

            async def _show_user(text: str) -> None:
                # Show the user's turn (transcript, or a mic glyph in direct-audio mode).
                content = text or PENDING_USER_TEXT
                self._pending_user_seq = self.transcript.add("user", content)
                await self.output_queue.put(AdditionalOutputs({"role": "user", "content": content}))

            turn = await generate_reply(
                self._stt,
                self._chat,
                utterance,
                direct_audio=config.OLLAMA_DIRECT_AUDIO,
                sample_rate=VAD_SAMPLE_RATE,
                on_user_text=_show_user,
            )
            self.last_activity_time = asyncio.get_event_loop().time()
            # Cascade mode detects the language before replying, so it can inform
            # this very turn (direct-audio mode records it after, via Whisper).
            self._lang_history.record(turn.language, turn.user_text)
            if turn.reply:
                self.transcript.add("assistant", turn.reply)
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": turn.reply}))
                # Speak the reply and (direct-audio) transcribe the turn concurrently:
                # Whisper hides under the reply's playback, then swaps the audio blob in
                # history for text — lighter + KV-cache-friendly. Best-effort.
                await asyncio.gather(
                    self._speak(turn.reply, turn.language),
                    self._transcribe_and_swap(utterance),
                )
        except Exception as exc:
            logger.exception("Turn failed: %s", exc)
        finally:
            self._speaking = False
            self._log_aec_debug()  # report this turn's echo cancellation (AEC_DEBUG)
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

    async def _transcribe_and_swap(self, utterance: NDArray[np.int16]) -> None:
        """Direct-audio: transcribe the turn via Whisper and store TEXT in history.

        Runs concurrently with the reply's playback (off the critical path). Replaces
        the audio blob in the chat history with the transcript so history stays light
        and KV-cache-friendly. Best-effort: if Whisper is unreachable the audio stays.
        """
        if not (config.OLLAMA_DIRECT_AUDIO and self._stt_remote and self._stt_remote.enabled):
            return
        text, lang = await self._stt_remote.transcribe(utterance, VAD_SAMPLE_RATE)
        # Whisper heard the user directly, which is a far better language signal
        # than guessing from Reachy's own reply — remember it for later turns,
        # weighted by how much was actually said.
        self._lang_history.record(lang, text)
        if text and self._chat is not None:
            self._chat.replace_last_user_audio_with_text(text)
            # Upgrade the placeholder line in place so the transcript shows one
            # user turn, not a mic glyph followed by its own transcription.
            if self._pending_user_seq is not None:
                self.transcript.update(self._pending_user_seq, text)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": f"📝 {text}"}))

    async def _speak(self, text: str, language: str | None = None) -> None:
        """Synthesize *text* sentence-by-sentence and stream each to wobbler + player.

        Splitting lets the first sentence start playing while later ones are still
        synthesizing (low latency to first word), and keeps each utterance within
        Kokoro's per-utterance token limit. *language* (from STT) tells a
        multilingual voice server which language to speak; without one it also
        gets the conversation's language as a hint, so a short reply the server
        can't confidently classify on its own isn't spoken in the wrong accent.
        """
        assert self._tts
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

        loop = asyncio.get_running_loop()
        total_samples = 0
        play_start: float | None = None
        # Read once: the concurrent Whisper pass for THIS turn may land mid-reply,
        # and a hint that changes between sentences would shift accent mid-sentence.
        hint = self._lang_history.hint()

        def should_stop() -> bool:
            return self._stop.is_set() or self._interrupt.is_set()

        async for src_rate, pcm in stream_sentences(
            self._tts, text, voice=self._voice, language=language,
            language_hint=hint, should_stop=should_stop, loop=loop,
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
        if self._calibrating:  # startup probe: collect raw mic to time the echo
            self._calib_near.append(pcm16.copy())
            return
        if self._aec is not None:
            if self._aec_debug and self._speaking:  # echo energy before cancellation
                self._erle_pre_sq += float(np.sum(pcm16.astype(np.float64) ** 2))
                self._erle_pre_n += len(pcm16)
            pcm16 = self._aec.clean(pcm16)  # remove Reachy's echo (10 ms-aligned; may buffer)
            if self._aec_debug and self._speaking and len(pcm16):  # residual after cancellation
                self._erle_post_sq += float(np.sum(pcm16.astype(np.float64) ** 2))
                self._erle_post_n += len(pcm16)
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

    async def _calibrate_echo(self) -> None:
        """Play a short probe and time its echo in the mic to set the AEC stream delay.

        Works through the robot's normal audio path (output_queue → player → robot.media →
        speaker → mic → receive), so it measures the *actual* echo delay on the device.
        """
        from reachy_local_assistant.audio.echo_calib import CALIB_SAMPLE_RATE, make_probe, estimate_delay_ms

        if self._aec is None:
            return
        probe = make_probe(1.0, CALIB_SAMPLE_RATE)
        probe_i16 = np.clip(probe * 32768, -32768, 32767).astype(np.int16)
        self._calib_near = []
        self._calibrating = True
        try:
            await self.output_queue.put((CALIB_SAMPLE_RATE, probe_i16))  # plays via the normal path
            await asyncio.sleep(2.0)  # let it play, echo, and be captured in receive()
        finally:
            self._calibrating = False
        near = np.concatenate(self._calib_near) if self._calib_near else np.zeros(0, dtype=np.int16)
        self._calib_near = []
        if len(near) > len(probe):
            delay = estimate_delay_ms(near, probe, CALIB_SAMPLE_RATE)
            self._aec.set_stream_delay_ms(int(round(delay)))
            logger.info("AEC auto-calibrated: echo delay = %.0f ms", delay)
        else:
            logger.warning(
                "AEC calibration: too little mic captured (%d samples); keeping delay=%d ms",
                len(near), self._aec.stream_delay_ms,
            )

    def _log_aec_debug(self) -> None:
        """Log this turn's echo / residual / ERLE (AEC_DEBUG) and reset the meter."""
        if self._aec_debug and self._erle_pre_n and self._erle_post_n:
            pre = self._erle_pre_sq / self._erle_pre_n
            post = self._erle_post_sq / self._erle_post_n
            echo_db = 10 * math.log10(pre / 32768**2 + 1e-12)
            res_db = 10 * math.log10(post / 32768**2 + 1e-12)
            logger.info(
                "AEC turn: echo=%.0f dBFS  residual=%.0f dBFS  ERLE=%.1f dB  (delay=%d ms)",
                echo_db, res_db, echo_db - res_db, config.AEC_STREAM_DELAY_MS,
            )
        self._erle_pre_sq = self._erle_post_sq = 0.0
        self._erle_pre_n = self._erle_post_n = 0

    def _flush_output(self) -> None:
        """Drop queued output audio (and reset the wobbler) so playback stops fast."""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Also flush what already reached the player, or a barge-in only stops the
        # *next* chunk while the buffered tail of Reachy's reply plays on.
        if self._clear_queue is not None:
            try:
                self._clear_queue()
            except Exception as exc:
                logger.debug("Player flush failed: %s", exc)
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
                # Clear the conversation so the new personality/language takes full
                # effect immediately — otherwise the old history's momentum (e.g.
                # English) keeps dominating and the switch "doesn't seem to work".
                self._chat.reset()
                # Keep the visible transcript in step with the model's history;
                # leaving the old turns on screen implies context that is gone.
                self.transcript.clear()
                # The new personality may speak a different language; carrying the
                # old hint over would fight the switch on every short reply.
                self._lang_history.clear()
            return "Applied personality."
        except Exception as exc:
            logger.error("Failed to apply personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def refresh_health(self) -> BackendHealth:
        """Re-probe the backends, log the result, and cache it for the settings page."""
        try:
            self.health = await check_backends(
                ollama_url=config.OLLAMA_URL,
                ollama_model=config.OLLAMA_MODEL,
                tts_url=config.TTS_URL,
                stt_url=config.STT_URL,
            )
            log_health(self.health)
        except Exception as exc:  # a probe failure must never block the conversation
            logger.warning("Backend health check failed: %s", exc)
        return self.health

    async def apply_backends(self) -> str:
        """Apply Ollama/TTS config changes live (no restart).

        Re-points the STT + chat clients at the current ``OLLAMA_URL`` (keeping
        history/tools) and rebuilds the TTS backend from the current TTS config —
        so a URL/voice change from the settings page takes effect immediately,
        like a personality Apply.
        """
        try:
            if self._stt is not None and hasattr(self._stt, "set_host"):
                self._stt.set_host(config.OLLAMA_URL)
            if self._chat is not None and hasattr(self._chat, "set_host"):
                self._chat.set_host(config.OLLAMA_URL)
            self._tts = make_tts()  # reads TTS_BACKEND / TTS_URL / TTS_VOICE from config
            self._voice = get_session_voice()
            logger.info(
                "Applied backends live: ollama=%s tts=%s voice=%s",
                config.OLLAMA_URL, config.TTS_BACKEND, getattr(config, "TTS_VOICE", ""),
            )
            # Re-probe so the settings page reports the URL the user just saved,
            # rather than the state of the previous one.
            health = await self.refresh_health()
            if not health.ok:
                down = ", ".join(p.name for p in health.probes if not p.ok)
                return f"Applied backend settings (no restart). Still unreachable: {down}."
            return "Applied backend settings (no restart). All backends reachable."
        except Exception as exc:
            logger.error("Failed to apply backends: %s", exc)
            return f"Failed to apply backends: {exc}"

    # --- helpers ---------------------------------------------------------

    def _set_listening(self, listening: bool) -> None:
        mm = getattr(self.deps, "movement_manager", None)
        if mm is not None and hasattr(mm, "set_listening"):
            try:
                mm.set_listening(listening)
            except Exception as exc:
                logger.debug("set_listening failed: %s", exc)

