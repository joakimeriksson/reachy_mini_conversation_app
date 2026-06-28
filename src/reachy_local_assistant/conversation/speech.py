"""Sentence-streamed TTS — synthesize a reply piece-by-piece, interruptibly.

Splitting on sentences lets the first one start playing while later ones are still
synthesizing (low latency to first word) and keeps each within Kokoro's per-utterance
token cap. The caller owns the audio sink (resampling, queueing, the wobbler); this just
yields ``(sample_rate, int16 PCM)`` chunks and stops the moment *should_stop* turns true.
"""

from __future__ import annotations
import asyncio
from typing import Tuple, Callable, Optional, AsyncIterator
from concurrent.futures import Executor

import numpy as np
from numpy.typing import NDArray

from reachy_local_assistant.audio.protocols import TtsBackend
from reachy_local_assistant.audio.text_chunk import split_sentences


async def stream_sentences(
    tts: TtsBackend,
    text: str,
    *,
    voice: Optional[str],
    language: Optional[str],
    should_stop: Callable[[], bool],
    loop: Optional[asyncio.AbstractEventLoop] = None,
    executor: Optional[Executor] = None,
) -> AsyncIterator[Tuple[int, NDArray[np.int16]]]:
    """Yield ``(sample_rate, int16 PCM)`` chunks for *text*, one sentence at a time.

    Synthesis (CPU/network-bound) runs in *executor* off the event loop. *should_stop* is
    polled before each sentence and each chunk so a barge-in cuts playback promptly. The
    *should_stop* callable bridges both an ``asyncio.Event`` and a ``threading.Event``
    (pass ``event.is_set``).
    """
    loop = loop or asyncio.get_running_loop()

    def _synth(sentence: str) -> list[Tuple[int, NDArray[np.int16]]]:
        return list(tts.synthesize(sentence, voice=voice, language=language))

    for sentence in split_sentences(text):
        if should_stop():
            return
        chunks = await loop.run_in_executor(executor, _synth, sentence)
        for sample_rate, pcm in chunks:
            if should_stop():
                return
            yield sample_rate, pcm
