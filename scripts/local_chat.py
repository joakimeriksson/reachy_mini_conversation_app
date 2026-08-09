"""Standalone local voice chat — talk to the pipeline on a laptop, no robot.

Uses your computer's microphone and speakers (sounddevice) instead of the Reachy
robot's media, so you can exercise the conversation pipeline
(VAD → Ollama/Gemma → voice-server TTS) without hardware. Half-duplex: the mic is
muted while it speaks. Needs Ollama and the voice server running (see SETUP.md).

    # talk live (Ctrl+C to quit)
    python scripts/local_chat.py

    # with the webcam, so it can "see" when you ask to look
    python scripts/local_chat.py --vision

    # headless end-to-end check (no mic): synth a phrase → STT → LLM → TTS → wav
    python scripts/local_chat.py --self-test

    python scripts/local_chat.py --list-devices

This is a development/test harness only — it is NOT part of the installed package
(it lives outside ``src/`` so it never ships in the robot wheel). It imports the
real app modules, so run it from the repo root. Requires the ``localdev`` extra
(``sounddevice``); ``opencv-python`` for ``--vision`` is already a core dep.
"""

from __future__ import annotations
import sys
import asyncio
import logging
import argparse
import threading
from pathlib import Path
from collections import deque


# Make the in-repo package importable when run as a plain script from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from numpy.typing import NDArray

from reachy_local_assistant.prompts import get_session_voice
from reachy_local_assistant.language import LanguageHistory
from reachy_local_assistant.audio.dsp import resample_int16, pcm16_to_wav_bytes
from reachy_local_assistant.audio.tts import make_tts
from reachy_local_assistant.audio.vad import FRAME_SAMPLES, VAD_SAMPLE_RATE, VadSegmenter
from reachy_local_assistant.audio.gemma_stt import GemmaSTT
from reachy_local_assistant.llm.ollama_chat import OllamaChat
from reachy_local_assistant.audio.remote_stt import RemoteSTT
from reachy_local_assistant.conversation.turn import generate_reply
from reachy_local_assistant.conversation.speech import stream_sentences


logger = logging.getLogger("local_chat")

DEFAULT_SYSTEM_PROMPT = (
    "You are Reachy Mini, a small friendly desk robot. Keep replies short, warm and "
    "conversational — one or two sentences, suitable for being spoken aloud. Do not use "
    "markdown, emoji, or lists. When an image is provided, it is what you see through "
    "your camera right now — describe it naturally as your own view."
)

# Utterances containing any of these (when --vision is on) attach a webcam frame
# so the model can actually look, mimicking Reachy's camera.
VISION_INTENT_WORDS = (
    "see", "look", "looking", "watch", "show", "camera", "front of you", "in front",
    "holding", "wearing", "what is this", "what's this", "what am i", "who am i",
    "recognize", "recognise", "color", "colour", "read this",
)


def _wants_vision(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in VISION_INTENT_WORDS)


class CameraGrabber:
    """Background webcam reader exposing the latest frame as JPEG bytes."""

    def __init__(self, device: int = 0) -> None:
        import threading

        self._device = device
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import threading

        import cv2

        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self._device}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Camera %d started", self._device)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def get_jpeg(self, max_width: int = 768) -> bytes | None:
        import cv2

        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > max_width:  # downscale to keep the request small/fast
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._cap is not None:
            self._cap.release()


class LocalVoiceChat:
    """Wires the local backend modules to a simple turn-based loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        from reachy_local_assistant.config import config

        self._direct_audio = config.OLLAMA_DIRECT_AUDIO
        self._noise_gate = config.NOISE_GATE
        prompt = _load_system_prompt(args.profile)
        if self._direct_audio:
            prompt += (
                "\n\nThe user speaks to you through attached audio. Listen and respond "
                "directly and naturally to what they say — never transcribe or repeat it."
            )
        self._stt = GemmaSTT(args.stt_model or args.model, args.ollama_url)
        self._chat = OllamaChat(
            args.model, args.ollama_url, deps=None,
            system_prompt=prompt, enable_tools=False,
        )
        # make_tts() picks Piper (local) or RemoteTTS (the voice server) from
        # TTS_BACKEND/TTS_URL — so this no-robot runner speaks through the exact
        # same path as the real app/robot (incl. the Swedish Kokoro voice server).
        self._tts = make_tts()
        self._voice = get_session_voice()
        # Conversation language, used as a tie-breaker when the voice server can't
        # confidently classify a short reply on its own.
        self._lang_history = LanguageHistory()
        # Whisper on the voice server: noise gate + per-turn language evidence.
        self._stt_remote = RemoteSTT(config.STT_URL)
        self._vad = VadSegmenter(
            aggressiveness=args.aggressiveness, silence_ms=args.silence_ms,
            backend=config.VAD_BACKEND, threshold=config.VAD_THRESHOLD,
        )
        self._speaking = False
        # Full-duplex playback: the output stream pulls from this buffer (16 kHz),
        # so the input stream can keep listening with echo cancelled (barge-in).
        self._play_buf: "deque[NDArray[np.int16]]" = deque()
        self._play_lock = threading.Lock()
        self._tts_active = False  # True only while audio is queued/playing
        # Barge-in: a strict VAD watches for the user talking over playback.
        self._interrupt = threading.Event()
        self._barge_vad: VadSegmenter | None = None
        if config.BARGE_IN:
            self._barge_vad = VadSegmenter(
                aggressiveness=3, silence_ms=200, min_speech_ms=config.BARGE_IN_SPEECH_MS,
                backend=config.VAD_BACKEND, threshold=config.BARGE_IN_THRESHOLD,
            )
        # Acoustic echo cancellation — required for barge-in on open speakers (else the
        # mic hears Reachy and self-interrupts). Shared by both audio callbacks via a lock.
        self._aec = None
        self._aec_lock = threading.Lock()
        if config.AEC:
            try:
                from reachy_local_assistant.audio.aec import EchoCanceller

                self._aec = EchoCanceller(stream_delay_ms=config.AEC_STREAM_DELAY_MS)
                _mode = "auto-calibrating" if config.AEC_STREAM_DELAY_MS == 0 else f"{config.AEC_STREAM_DELAY_MS} ms"
                print(f"🎚️  Echo cancellation on ({_mode})")
            except Exception as exc:
                print(f"   (AEC unavailable: {exc} — on speakers barge-in will hear itself)")

        self._always_vision = args.always_vision
        self._camera: CameraGrabber | None = None
        if args.vision or args.always_vision:
            self._camera = CameraGrabber(args.camera)

    def _maybe_capture(self, text: str) -> bytes | None:
        """Grab a webcam frame when vision is on and the user is asking to look."""
        if self._camera is None:
            return None
        if not (self._always_vision or _wants_vision(text)):
            return None
        jpeg = self._camera.get_jpeg()
        if jpeg:
            print("   📷 (looking through the camera…)")
        return jpeg

    async def handle_utterance(self, utterance: NDArray[np.int16]) -> None:
        """One full turn: STT → LLM → TTS → speaker."""
        self._speaking = True
        try:

            async def _show_user(text: str) -> None:
                print(f"\n🧑  You:    {text or '🎤 …'}")

            # Transcribe concurrently with the LLM call (direct-audio): powers the
            # noise gate below and this turn's language hint at zero added latency.
            transcribe_task = None
            if self._direct_audio and self._stt_remote.enabled:
                transcribe_task = asyncio.create_task(self._stt_remote.transcribe(utterance, VAD_SAMPLE_RATE))

            turn = await generate_reply(
                self._stt,
                self._chat,
                utterance,
                direct_audio=self._direct_audio,
                sample_rate=VAD_SAMPLE_RATE,
                on_user_text=_show_user,
                capture_image=self._maybe_capture,
            )
            self._lang_history.record(turn.language, turn.user_text)

            if transcribe_task is not None:
                try:
                    text, lang = await transcribe_task
                except Exception as exc:
                    text, lang = "…", ""  # Whisper down: keep the turn
                    logger.debug("turn transcription failed: %s", exc)
                self._lang_history.record(lang, text)
                if text.strip():
                    if text != "…":
                        print(f"    📝 heard: {text}")
                        self._chat.replace_last_user_audio_with_text(text)
                elif self._noise_gate:
                    # The VAD fired on room noise, not words — undo and stay quiet.
                    print("    🔇 (noise — ignored)")
                    self._chat.drop_last_exchange()
                    return

            if turn.reply:
                print(f"🤖  Reachy: {turn.reply}\n")
                await self._play(turn.reply, turn.language)
        finally:
            self._speaking = False
            self._vad.reset()

    async def _play(self, text: str, language: str | None = None) -> None:
        loop = asyncio.get_running_loop()
        self._interrupt.clear()
        if self._barge_vad is not None:
            self._barge_vad.reset()
        self._tts_active = True
        try:
            # Stream sentence-by-sentence into the playback buffer (resampled to the
            # 16 kHz output stream). First sentence plays while the next synthesizes;
            # the output callback drains the buffer and feeds it to the echo canceller.
            async for src_rate, pcm in stream_sentences(
                self._tts, text, voice=self._voice, language=language,
                language_hint=self._lang_history.hint(),
                should_stop=self._interrupt.is_set, loop=loop,
            ):
                pcm16 = resample_int16(pcm, src_rate, VAD_SAMPLE_RATE)
                with self._play_lock:
                    self._play_buf.append(pcm16)
            # Wait for playback to drain (or a barge-in to cut it short).
            while not self._interrupt.is_set():
                with self._play_lock:
                    pending = sum(len(c) for c in self._play_buf)
                if pending == 0:
                    break
                await asyncio.sleep(0.03)
        finally:
            if self._interrupt.is_set():
                with self._play_lock:
                    self._play_buf.clear()
            self._tts_active = False

    async def run_live(self, input_device: int | None) -> None:
        """Full-duplex loop: paired in/out streams share the AEC; barge-in is live.

        The realtime callbacks only do fast work (buffer copy + WebRTC AEC). The
        (heavier) VAD runs in an async task off the audio threads. On Mac the mic and
        speakers are separate devices, so two streams (not one duplex) are used.
        """
        import sounddevice as sd

        from reachy_local_assistant.config import config
        from reachy_local_assistant.audio.echo_calib import measure_delay_via_sounddevice

        # Auto-calibrate the echo delay with a startup chirp (unless overridden by
        # AEC_STREAM_DELAY_MS). Runs before the live streams open so it has the device.
        if self._aec is not None and config.AEC_STREAM_DELAY_MS == 0:
            print("🎚️  Calibrating echo delay (chirp)…")
            delay = measure_delay_via_sounddevice(1.0)
            if delay is not None:
                self._aec.set_stream_delay_ms(int(round(delay)))
                print(f"🎚️  Echo delay: {delay:.0f} ms (auto-calibrated)")

        loop = asyncio.get_running_loop()
        utt_q: asyncio.Queue[NDArray[np.int16]] = asyncio.Queue()
        mic_q: asyncio.Queue[NDArray[np.int16]] = asyncio.Queue()

        def out_callback(outdata, frames, time_info, status):  # speaker thread
            out = np.zeros(frames, dtype=np.int16)
            with self._play_lock:
                filled = 0
                while filled < frames and self._play_buf:
                    chunk = self._play_buf[0]
                    take = min(len(chunk), frames - filled)
                    out[filled : filled + take] = chunk[:take]
                    if take == len(chunk):
                        self._play_buf.popleft()
                    else:
                        self._play_buf[0] = chunk[take:]
                    filled += take
            outdata[:, 0] = out
            if self._aec is not None:
                with self._aec_lock:
                    self._aec.play_reference(out)  # far-end reference for the canceller

        def in_callback(indata, frames, time_info, status):  # mic thread
            if status:
                logger.debug("audio status: %s", status)
            mic = indata[:, 0].copy()
            if self._aec is not None:
                with self._aec_lock:
                    mic = self._aec.clean(mic)  # subtract Reachy's echo
            if len(mic):
                loop.call_soon_threadsafe(mic_q.put_nowait, mic)

        async def process_mic() -> None:
            while True:
                mic = await mic_q.get()
                if self._tts_active:  # listening for barge-in over the (echo-cancelled) playback
                    if self._barge_vad is not None:
                        self._barge_vad.feed(mic)
                        if self._barge_vad.in_speech and not self._interrupt.is_set():
                            self._interrupt.set()
                            print("   ✋ (barge-in)")
                elif not self._speaking:
                    for utt in self._vad.feed(mic):
                        utt_q.put_nowait(utt)

        if self._camera is not None:
            self._camera.start()
            print("👁️  Vision on" + (" (always)" if self._always_vision else " (when you ask to look)"))

        print("🎙️  Listening… speak to Reachy Mini. Press Ctrl+C to stop.\n")
        mic_task = asyncio.create_task(process_mic())
        try:
            with sd.OutputStream(
                samplerate=VAD_SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=FRAME_SAMPLES, callback=out_callback,
            ), sd.InputStream(
                samplerate=VAD_SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=FRAME_SAMPLES, callback=in_callback, device=input_device,
            ):
                while True:
                    utterance = await utt_q.get()
                    await self.handle_utterance(utterance)
        finally:
            mic_task.cancel()
            if self._camera is not None:
                self._camera.stop()

    async def self_test(self) -> int:
        """Feed a synthesized phrase through the pipeline; save the reply to a wav."""
        import soundfile as sf

        phrase = "Hello, who are you? Please answer in one short sentence."
        print(f"[self-test] synthesizing prompt: {phrase!r}")
        synth = list(self._tts.synthesize(phrase, voice=self._voice))
        if not synth:
            print("[self-test] FAIL: TTS produced no audio (is the voice server running / TTS_URL right?)")
            return 1
        sr = synth[0][0]
        spoken = np.concatenate([c[1] for c in synth])
        pcm16 = resample_int16(spoken, sr, VAD_SAMPLE_RATE)

        lang: str | None
        if self._direct_audio:
            print("[self-test] asking the model directly from audio (one call)…")
            reply = await self._chat.respond("", audio=pcm16_to_wav_bytes(pcm16, VAD_SAMPLE_RATE))
            lang = None
        else:
            print("[self-test] transcribing via Gemma…")
            heard, lang = await self._stt.transcribe(pcm16)
            print(f"[self-test] heard ({lang or '?'}): {heard!r}")
            if not heard.strip():
                print("[self-test] FAIL: empty transcription")
                return 1
            print("[self-test] asking the model…")
            reply = await self._chat.respond(heard)
        print(f"[self-test] reply: {reply!r}")
        if not reply.strip():
            print("[self-test] FAIL: empty reply")
            return 1

        out_parts = []
        out_sr = 24000
        async for sr, pcm in stream_sentences(
            self._tts, reply, voice=self._voice, language=lang, should_stop=lambda: False
        ):
            out_sr = sr
            out_parts.append(pcm)
        out = np.concatenate(out_parts) if out_parts else np.zeros(0, dtype=np.int16)
        out_path = Path("/tmp/reachy_reply.wav")
        sf.write(out_path, out, out_sr, subtype="PCM_16")
        print(f"[self-test] PASS — reply audio: {out_path} ({len(out)/out_sr:.1f}s @ {out_sr} Hz)")
        return 0


def _load_system_prompt(profile: str | None) -> str:
    """Use the app's profile instructions if available, else a built-in prompt."""
    try:
        from reachy_local_assistant.config import set_custom_profile
        from reachy_local_assistant.prompts import get_session_instructions

        if profile:
            set_custom_profile(profile)
        return get_session_instructions()
    except Exception as exc:
        logger.debug("Falling back to default system prompt: %s", exc)
        return DEFAULT_SYSTEM_PROMPT


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from reachy_local_assistant.config import config

    p = argparse.ArgumentParser(description="Talk to the local Ollama+Piper pipeline (no robot).")
    p.add_argument("--model", default=config.OLLAMA_MODEL, help="Ollama conversation model")
    p.add_argument("--stt-model", default=config.OLLAMA_STT_MODEL, help="Ollama audio STT model")
    p.add_argument("--ollama-url", default=config.OLLAMA_URL)
    p.add_argument("--voice", default=config.TTS_VOICE, help="voice-server voice name (e.g. Stina, af_heart)")
    p.add_argument("--voice-dir", default=None)
    p.add_argument("--profile", default=None, help="Personality profile to load")
    p.add_argument("--silence-ms", type=int, default=config.VAD_SILENCE_MS,
                   help="Trailing silence (ms) that ends an utterance")
    p.add_argument("--aggressiveness", type=int, default=config.VAD_AGGRESSIVENESS, choices=[0, 1, 2, 3],
                   help="WebRTC VAD aggressiveness: 0=lenient, 3=aggressive (better at ignoring noise/silence)")
    p.add_argument("--input-device", type=int, default=None, help="sounddevice input id")
    p.add_argument("--vision", action="store_true", help="Enable webcam; look when asked")
    p.add_argument("--always-vision", action="store_true", help="Attach a webcam frame every turn")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera device index")
    p.add_argument("--self-test", action="store_true", help="Headless pipeline check (no mic)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    print(f"Model: {args.model} | STT: {args.stt_model} | Voice: {args.voice} | Ollama: {args.ollama_url}")
    chat = LocalVoiceChat(args)

    if args.self_test:
        return asyncio.run(chat.self_test())

    try:
        asyncio.run(chat.run_live(args.input_device))
    except KeyboardInterrupt:
        print("\n👋  Bye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
