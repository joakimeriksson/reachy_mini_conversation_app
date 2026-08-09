"""In-memory conversation transcript for the settings page.

Dropping Gradio took the chat pane with it, leaving the conversation visible only
in the app's log. This is the replacement: a small ring buffer the handler writes
to as turns happen, which the settings page polls.

Kept separate from the LLM's own message history (``OllamaChat``), which carries
system prompts, tool calls and audio blobs. This holds only what a human would
want to read, and is capped so a long-running robot cannot grow it without bound.

Entries carry a monotonic ``seq`` so the page can poll for "everything after N"
instead of re-fetching the whole conversation.
"""

from __future__ import annotations
import threading
from typing import List
from dataclasses import asdict, dataclass


# Roughly an hour of conversation; old turns fall off the front.
DEFAULT_MAX_MESSAGES = 200

# Placeholder shown for a user turn whose audio has not been transcribed yet
# (direct-audio mode transcribes after replying, or not at all without Whisper).
PENDING_USER_TEXT = "🎤 …"


@dataclass
class Message:
    """One line of the visible conversation."""

    seq: int
    role: str
    content: str

    def as_dict(self) -> dict[str, object]:
        """Render for the settings-page JSON payload."""
        return asdict(self)


class Transcript:
    """Thread-safe, bounded conversation log.

    Written from the asyncio conversation loop and read from FastAPI's thread
    pool, so every operation takes a lock.
    """

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES) -> None:
        """Create an empty transcript holding at most *max_messages* entries."""
        self._max = max_messages
        self._messages: List[Message] = []
        self._next_seq = 1
        # Bumped on every clear. Since seq stays monotonic across clears, a
        # polling page cannot detect one by sequence alone — it would keep
        # showing turns the robot has forgotten. This is how it notices.
        self._generation = 1
        self._lock = threading.Lock()

    def add(self, role: str, content: str) -> int:
        """Append a message and return its sequence number."""
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            self._messages.append(Message(seq=seq, role=role, content=content))
            if len(self._messages) > self._max:
                del self._messages[: len(self._messages) - self._max]
            return seq

    def update(self, seq: int, content: str) -> bool:
        """Replace the content of message *seq*; False if it is gone or unknown.

        Used when the Whisper transcript arrives after the reply and upgrades a
        pending user turn in place, instead of appending a duplicate line.
        """
        with self._lock:
            for message in self._messages:
                if message.seq == seq:
                    message.content = content
                    return True
            return False

    def remove(self, seq: int) -> bool:
        """Delete message *seq*; False if it is gone or unknown.

        Used by the noise gate: a pending "🎤 …" line whose audio turned out to be
        room noise must disappear, not sit in the conversation as a ghost turn.
        """
        with self._lock:
            for i, message in enumerate(self._messages):
                if message.seq == seq:
                    del self._messages[i]
                    return True
            return False

    def messages(self, since: int = 0) -> List[Message]:
        """Return messages with ``seq > since`` (all of them when *since* is 0)."""
        with self._lock:
            return [m for m in self._messages if m.seq > since]

    def latest_seq(self) -> int:
        """Return the highest sequence number issued so far."""
        with self._lock:
            return self._next_seq - 1

    def generation(self) -> int:
        """Return the current generation; it increments on every clear."""
        with self._lock:
            return self._generation

    def clear(self) -> None:
        """Drop every message, keeping the sequence counter monotonic.

        The counter must not reset: a page polling with ``since=N`` would
        otherwise never see the new conversation, having already passed N.
        Readers detect the clear via the bumped generation instead.
        """
        with self._lock:
            self._messages.clear()
            self._generation += 1

    def as_dict(self, since: int = 0) -> dict[str, object]:
        """Render the poll response for the settings page."""
        with self._lock:
            return {
                "messages": [m.as_dict() for m in self._messages if m.seq > since],
                "latest_seq": self._next_seq - 1,
                "generation": self._generation,
            }
