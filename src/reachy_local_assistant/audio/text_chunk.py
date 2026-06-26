"""Split reply text into speakable sentence chunks for streaming TTS.

Synthesizing one sentence at a time lets playback of the first sentence start
while later ones are still synthesizing — low latency to the first spoken word —
and keeps each chunk within Kokoro's ~510-token per-utterance limit (longer text
raises an IndexError on its voicepack).
"""

import re


# Split after sentence-final punctuation followed by whitespace.
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Return *text* as a list of sentence chunks, hard-wrapping any over *max_chars*.

    A pathologically long sentence (no terminal punctuation) is split on the last
    space before *max_chars* so no single chunk exceeds the model's token cap.
    """
    text = (text or "").strip()
    if not text:
        return []
    out: list[str] = []
    for piece in _BOUNDARY.split(text):
        piece = piece.strip()
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            head = piece[:cut].strip()
            if head:
                out.append(head)
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out
