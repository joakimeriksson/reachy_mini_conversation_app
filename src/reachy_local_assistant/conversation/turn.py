"""Understand one utterance and produce a reply — the shared turn step.

Two modes (selected by *direct_audio*):
- direct-audio: feed the speech straight to the chat model (one call, no STT).
- cascade: Gemma STT → text → chat (with optional inline image capture).

Returns a :class:`TurnResult`; the caller owns display/transcript rendering (via the
optional *on_user_text* callback) and playback.
"""

from __future__ import annotations
from typing import Callable, Optional, Awaitable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_local_assistant.audio.dsp import pcm16_to_wav_bytes
from reachy_local_assistant.audio.protocols import SttBackend, ChatBackend


@dataclass
class TurnResult:
    """Outcome of understanding one utterance."""

    user_text: str  # transcript (cascade) or "" (direct-audio)
    reply: str  # assistant reply ("" when there is nothing to say)
    language: Optional[str]  # STT-detected language, or None (direct-audio)


async def generate_reply(
    stt: SttBackend,
    chat: ChatBackend,
    utterance: NDArray[np.int16],
    *,
    direct_audio: bool,
    sample_rate: int,
    on_user_text: Optional[Callable[[str], Awaitable[None]]] = None,
    capture_image: Optional[Callable[[str], Optional[bytes]]] = None,
) -> TurnResult:
    """Understand *utterance* and return the reply.

    *on_user_text* (if given) is awaited as soon as the user's input is known — with the
    transcript in cascade mode, or ``""`` in direct-audio mode — so the front-end can show
    the user's turn before the model finishes thinking. *capture_image* (cascade only) is
    called with the transcript to optionally attach a webcam frame.
    """
    if direct_audio:
        if on_user_text is not None:
            await on_user_text("")
        reply = await chat.respond("", audio=pcm16_to_wav_bytes(utterance, sample_rate))
        return TurnResult("", reply or "", None)

    text, language = await stt.transcribe(utterance)
    if not text.strip():
        return TurnResult("", "", None)
    if on_user_text is not None:
        await on_user_text(text)
    image = capture_image(text) if capture_image is not None else None
    reply = await chat.respond(text, image=image)
    return TurnResult(text, reply or "", language)
