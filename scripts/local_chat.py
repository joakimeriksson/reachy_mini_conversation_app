"""Standalone local voice chat — talk to the Ollama + Piper pipeline, no robot.

Uses your computer's microphone and speakers (sounddevice) instead of the Reachy
robot's media, so you can exercise the conversation pipeline
(VAD → Gemma STT → Ollama → Piper) on a laptop. Half-duplex: the mic is muted
while it speaks.

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
from pathlib import Path


# Make the in-repo package importable when run as a plain script from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_local_assistant.audio.vad import FRAME_SAMPLES, VAD_SAMPLE_RATE, VadSegmenter
from reachy_local_assistant.audio.gemma_stt import GemmaSTT
from reachy_local_assistant.audio.piper_tts import PiperTTS
from reachy_local_assistant.llm.ollama_chat import OllamaChat


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


def _ensure_voice(voice: str, data_dir: Path) -> str:
    """Return a usable Piper voice, downloading it into *data_dir* if missing."""
    if Path(voice).suffix == ".onnx" and Path(voice).is_file():
        return voice
    onnx = data_dir / f"{voice}.onnx"
    if onnx.is_file():
        return voice
    logger.info("Downloading Piper voice %s into %s ...", voice, data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", voice, "--data-dir", str(data_dir)],
        check=True,
    )
    return voice


class LocalVoiceChat:
    """Wires the local backend modules to a simple turn-based loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        data_dir = Path(args.voice_dir).resolve()
        voice = _ensure_voice(args.voice, data_dir)

        self._stt = GemmaSTT(args.stt_model or args.model, args.ollama_url)
        self._chat = OllamaChat(
            args.model, args.ollama_url, deps=None,
            system_prompt=_load_system_prompt(args.profile), enable_tools=False,
        )
        self._tts = PiperTTS(default_voice=voice, data_dir=str(data_dir))
        self._vad = VadSegmenter(aggressiveness=args.aggressiveness, silence_ms=args.silence_ms)
        self._speaking = False

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
            text, _lang = await self._stt.transcribe(utterance)
            if not text.strip():
                return
            print(f"\n🧑  You:    {text}")
            image = self._maybe_capture(text)
            reply = await self._chat.respond(text, image=image)
            print(f"🤖  Reachy: {reply}\n")
            if reply.strip():
                await self._play(reply)
        finally:
            self._speaking = False
            self._vad.reset()

    async def _play(self, text: str) -> None:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(None, lambda: list(self._tts.synthesize(text)))
        if not chunks:
            return
        sr = chunks[0][0]
        pcm = np.concatenate([c[1] for c in chunks])
        await loop.run_in_executor(None, lambda: (sd.play(pcm, sr), sd.wait()))

    async def run_live(self, input_device: int | None) -> None:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[NDArray[np.int16]] = asyncio.Queue()

        def callback(indata, frames, time_info, status):  # runs on the audio thread
            if status:
                logger.debug("audio status: %s", status)
            if self._speaking:
                return  # half-duplex: ignore mic while speaking
            pcm = indata[:, 0].copy()
            for utt in self._vad.feed(pcm):
                loop.call_soon_threadsafe(queue.put_nowait, utt)

        if self._camera is not None:
            self._camera.start()
            print("👁️  Vision on" + (" (always)" if self._always_vision else " (when you ask to look)"))

        print("🎙️  Listening… speak to Reachy Mini. Press Ctrl+C to stop.\n")
        try:
            with sd.InputStream(
                samplerate=VAD_SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=FRAME_SAMPLES, callback=callback, device=input_device,
            ):
                while True:
                    utterance = await queue.get()
                    await self.handle_utterance(utterance)
        finally:
            if self._camera is not None:
                self._camera.stop()

    async def self_test(self) -> int:
        """Feed a synthesized phrase through the pipeline; save the reply to a wav."""
        import soundfile as sf

        phrase = "Hello, who are you? Please answer in one short sentence."
        print(f"[self-test] synthesizing prompt: {phrase!r}")
        synth = list(self._tts.synthesize(phrase))
        sr = synth[0][0]
        spoken = np.concatenate([c[1] for c in synth])
        pcm16 = np.clip(
            resample(spoken.astype(np.float32), int(len(spoken) * VAD_SAMPLE_RATE / sr)),
            -32768, 32767,
        ).astype(np.int16)

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

        out_chunks = list(self._tts.synthesize(reply))
        out_sr = out_chunks[0][0]
        out = np.concatenate([c[1] for c in out_chunks])
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
    p.add_argument("--voice", default=config.PIPER_VOICE, help="Piper voice name or .onnx path")
    p.add_argument("--voice-dir", default=config.PIPER_DATA_DIR or "piper_voices")
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
