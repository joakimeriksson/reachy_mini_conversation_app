"""Split reply text into speakable sentence chunks for streaming TTS.

Synthesizing one sentence at a time lets playback of the first sentence start
while later ones are still synthesizing — low latency to the first spoken word —
and keeps each chunk within Kokoro's ~510-token per-utterance limit (longer text
raises an IndexError on its voicepack).
"""

import re


# Split after sentence-final punctuation followed by whitespace.
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")

# Emoji / pictographic symbols the LLM sometimes adds — TTS would try to *speak*
# them (or choke), so strip them before synthesis regardless of the prompt.
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoji, supplemental
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicator (flags)
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"             # zero-width joiner
    "]+",
    flags=re.UNICODE,
)


def strip_symbols(text: str) -> str:
    """Remove emoji/pictographic symbols and collapse the whitespace they leave."""
    return re.sub(r"[ \t]{2,}", " ", _EMOJI.sub("", text)).strip()


def split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Return *text* as a list of sentence chunks, hard-wrapping any over *max_chars*.

    Emoji/symbols are stripped first (they are spoken text otherwise). A
    pathologically long sentence (no terminal punctuation) is split on the last
    space before *max_chars* so no single chunk exceeds the model's token cap.
    """
    text = strip_symbols((text or "").strip())
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
