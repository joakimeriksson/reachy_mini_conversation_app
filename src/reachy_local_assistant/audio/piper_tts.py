"""Text-to-speech via Piper (offline neural TTS, ONNX).

Synthesizes assistant text into 16-bit PCM chunks. The chunks are streamed back
one at a time so the conversation handler can start playback early and stop
mid-utterance on a barge-in.

Synthesis is CPU-bound and blocking, so the handler runs :meth:`synthesize` in a
thread executor; this module stays free of asyncio.
"""

from __future__ import annotations
import os
import logging
from typing import Any, Dict, Tuple, Iterator, Optional
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class PiperTTS:
    """Lazy-loading Piper voice wrapper producing int16 PCM chunks."""

    def __init__(self, default_voice: str, data_dir: Optional[str] = None) -> None:
        """Initialise with a default voice and optional voice directory."""
        self._default_voice = default_voice
        # Where downloaded voices live / are searched. Defaults to ./piper_voices.
        self._data_dir = Path(data_dir) if data_dir else Path("piper_voices")
        self._voices: Dict[str, "object"] = {}  # name/path -> PiperVoice

    def synthesize(
        self, text: str, voice: Optional[str] = None
    ) -> Iterator[Tuple[int, NDArray[np.int16]]]:
        """Yield ``(sample_rate, int16_pcm)`` chunks for *text*.

        *voice* overrides the configured default (a voice name or .onnx path).
        """
        text = (text or "").strip()
        if not text:
            return

        piper_voice = self._get_voice(voice or self._default_voice)
        sample_rate = piper_voice.config.sample_rate
        syn_config = self._synthesis_config()
        for chunk in piper_voice.synthesize(text, syn_config=syn_config):
            pcm = self._chunk_to_int16(chunk)
            if pcm.size:
                yield sample_rate, pcm

    @staticmethod
    def _synthesis_config() -> Any:
        """Build a Piper SynthesisConfig from config (None fields use voice defaults)."""
        from piper import SynthesisConfig

        from reachy_local_assistant.config import config

        kwargs: Dict[str, Any] = {}
        if config.PIPER_LENGTH_SCALE is not None:
            kwargs["length_scale"] = config.PIPER_LENGTH_SCALE
        if config.PIPER_NOISE_SCALE is not None:
            kwargs["noise_scale"] = config.PIPER_NOISE_SCALE
        if config.PIPER_NOISE_W_SCALE is not None:
            kwargs["noise_w_scale"] = config.PIPER_NOISE_W_SCALE
        if config.PIPER_VOLUME is not None:
            kwargs["volume"] = config.PIPER_VOLUME
        if config.PIPER_SPEAKER_ID >= 0:
            kwargs["speaker_id"] = config.PIPER_SPEAKER_ID
        return SynthesisConfig(**kwargs) if kwargs else None

    def _get_voice(self, name: str) -> Any:
        if name in self._voices:
            return self._voices[name]

        from piper import PiperVoice

        try:
            model_path = self._resolve_model_path(name)
        except FileNotFoundError:
            # Not on disk — try a one-time download (plain voice names only).
            model_path = self._download_voice(name)

        logger.info("Loading Piper voice: %s", model_path)
        voice = PiperVoice.load(str(model_path))
        self._voices[name] = voice
        return voice

    def _resolve_model_path(self, name: str) -> Path:
        """Resolve a voice name or path to an existing .onnx model file."""
        # Explicit path (with or without the .onnx suffix).
        candidate = Path(name)
        if candidate.suffix == ".onnx" and candidate.is_file():
            return candidate

        for directory in (self._data_dir, Path(os.getcwd())):
            onnx = directory / f"{name}.onnx"
            if onnx.is_file():
                return onnx

        raise FileNotFoundError(
            f"Piper voice {name!r} not found. Provide an absolute path in PIPER_VOICE, "
            f"or place {name}.onnx (and its .onnx.json) in PIPER_DATA_DIR. "
            f"Download voices from https://huggingface.co/rhasspy/piper-voices"
        )

    def _download_voice(self, name: str) -> Path:
        """Download a Piper voice by name into the data dir; return its .onnx path."""
        if "/" in name or name.endswith(".onnx"):
            # A path-like value that simply doesn't exist — don't try to download.
            raise FileNotFoundError(f"Piper voice file not found: {name!r}")

        import sys
        import subprocess

        self._data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Piper voice %r not found locally; downloading into %s ...", name, self._data_dir)
        subprocess.run(
            [sys.executable, "-m", "piper.download_voices", name, "--data-dir", str(self._data_dir)],
            check=True,
        )
        onnx = self._data_dir / f"{name}.onnx"
        if not onnx.is_file():
            raise FileNotFoundError(f"Piper voice {name!r} download did not produce {onnx}")
        return onnx

    @staticmethod
    def _chunk_to_int16(chunk: "object") -> NDArray[np.int16]:
        """Extract int16 PCM from a Piper AudioChunk across API versions."""
        arr = getattr(chunk, "audio_int16_array", None)
        if arr is not None:
            return np.asarray(arr, dtype=np.int16)
        raw = getattr(chunk, "audio_int16_bytes", None)
        if raw is not None:
            return np.frombuffer(raw, dtype=np.int16)
        floats = getattr(chunk, "audio_float_array", None)
        if floats is not None:
            return (np.asarray(floats, dtype=np.float32) * 32767.0).astype(np.int16)
        raise TypeError("Unrecognised Piper AudioChunk: no int16/float audio attribute")
