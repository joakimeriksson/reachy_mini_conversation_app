"""Tests for the settings-page conversation transcript."""

from reachy_local_assistant.transcript import PENDING_USER_TEXT, Transcript


def test_add_and_read_back() -> None:
    """Messages come back in order with monotonic sequence numbers."""
    t = Transcript()
    first = t.add("user", "hej")
    second = t.add("assistant", "hej själv")

    assert (first, second) == (1, 2)
    assert [(m.role, m.content) for m in t.messages()] == [
        ("user", "hej"),
        ("assistant", "hej själv"),
    ]


def test_since_returns_only_newer_messages() -> None:
    """Polling with since=N must not resend what the page already has."""
    t = Transcript()
    t.add("user", "one")
    seq = t.add("assistant", "two")
    t.add("user", "three")

    assert [m.content for m in t.messages(since=seq)] == ["three"]
    assert t.messages(since=t.latest_seq()) == []


def test_pending_user_turn_is_upgraded_in_place() -> None:
    """Whisper's transcript replaces the mic glyph rather than adding a line."""
    t = Transcript()
    seq = t.add("user", PENDING_USER_TEXT)
    t.add("assistant", "reply")

    assert t.update(seq, "what I actually said")

    contents = [m.content for m in t.messages()]
    assert contents == ["what I actually said", "reply"]


def test_update_of_an_unknown_seq_is_reported() -> None:
    """A turn evicted by the ring buffer must not silently resurrect."""
    t = Transcript()
    t.add("user", "hi")

    assert not t.update(999, "nope")


def test_ring_buffer_evicts_oldest() -> None:
    """A long-running robot must not grow the transcript without bound."""
    t = Transcript(max_messages=3)
    for i in range(5):
        t.add("user", f"msg{i}")

    assert [m.content for m in t.messages()] == ["msg2", "msg3", "msg4"]
    assert t.latest_seq() == 5


def test_clear_keeps_seq_monotonic_and_bumps_generation() -> None:
    """A page polling with since=N must still see turns recorded after a clear."""
    t = Transcript()
    t.add("user", "old")
    gen_before = t.generation()
    last = t.latest_seq()

    t.clear()

    assert t.messages() == []
    assert t.generation() == gen_before + 1
    new_seq = t.add("user", "new")
    assert new_seq > last, "reusing sequence numbers would hide the new turn from pollers"
    assert [m.content for m in t.messages(since=last)] == ["new"]


def test_as_dict_shape_for_the_settings_page() -> None:
    """The page consumes messages + latest_seq + generation."""
    t = Transcript()
    t.add("user", "hi")

    payload = t.as_dict()

    assert payload["latest_seq"] == 1
    assert payload["generation"] == 1
    assert payload["messages"] == [{"seq": 1, "role": "user", "content": "hi"}]


def test_as_dict_since_filters() -> None:
    """The poll payload honours `since` the same way messages() does."""
    t = Transcript()
    t.add("user", "one")
    t.add("assistant", "two")

    payload = t.as_dict(since=1)

    assert [m["content"] for m in payload["messages"]] == ["two"]  # type: ignore[index,union-attr]
    assert payload["latest_seq"] == 2
