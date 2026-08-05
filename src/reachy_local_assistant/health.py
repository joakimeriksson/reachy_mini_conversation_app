"""Reachability probes for the local backends (Ollama + voice server).

Every piece of this app is a service someone has to start: Ollama, and the voice
server that provides both TTS (``/v1/audio/speech``) and Whisper STT
(``/v1/audio/transcriptions``). When one is down the app still runs — it listens,
thinks, and then produces silence — which looks like a bug in the robot rather
than a service that was never started.

These probes turn that into an explicit, actionable message at startup and a
status block on the settings page.
"""

from __future__ import annotations
import asyncio
import logging
from typing import List, Optional
from dataclasses import field, asdict, dataclass
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

# Probes must not delay startup: a down service should fail fast, not hang the
# conversation loop behind a TCP timeout.
PROBE_TIMEOUT_S = 3.0


@dataclass
class ProbeResult:
    """Outcome of probing one backend service."""

    name: str
    url: str
    ok: bool
    detail: str
    hint: str = ""

    def as_dict(self) -> dict[str, object]:
        """Render for the settings-page JSON payload."""
        return asdict(self)


@dataclass
class BackendHealth:
    """Aggregated probe results for all backends."""

    probes: List[ProbeResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every probed service answered."""
        return all(p.ok for p in self.probes)

    def as_dict(self) -> dict[str, object]:
        """Render for the settings-page JSON payload."""
        return {"ok": self.ok, "probes": [p.as_dict() for p in self.probes]}


def _origin(url: str) -> str:
    """Return the ``scheme://host:port`` origin of *url* (empty when unparseable)."""
    try:
        parts = urlparse(url)
        if not parts.scheme or not parts.netloc:
            return ""
        return f"{parts.scheme}://{parts.netloc}"
    except Exception:
        return ""


async def _probe_ollama(url: str, model: str) -> ProbeResult:
    """Check Ollama is up and *model* is pulled."""
    name = "Ollama"
    if not url:
        return ProbeResult(name, url, False, "No OLLAMA_URL configured.", "Set OLLAMA_URL.")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as client:
            resp = await client.get(f"{url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            tags = [m.get("name", "") for m in (resp.json().get("models") or [])]
    except Exception as exc:
        return ProbeResult(
            name, url, False, f"Unreachable: {exc}",
            f"Start Ollama, or point OLLAMA_URL at the host running it (currently {url}).",
        )

    if model and model not in tags:
        # Ollama resolves a bare name to ":latest", so accept either spelling.
        stem = model.split(":")[0]
        if not any(t == model or t.split(":")[0] == stem for t in tags):
            return ProbeResult(
                name, url, False, f"Model {model!r} is not pulled ({len(tags)} other model(s) available).",
                f"Run: ollama pull {model}",
            )
    return ProbeResult(name, url, True, f"Up, {len(tags)} model(s), {model} available.")


async def _probe_http_service(
    name: str, url: str, unset_hint: str, requires: str = "", missing_hint: str = ""
) -> ProbeResult:
    """Check *something* is listening at *url*'s origin, and can do *requires*.

    Deliberately probes the origin, not the endpoint: a POST-only endpoint like
    ``/v1/audio/speech`` answers GET with 405, and synthesizing a probe utterance
    on every start would be wasteful. The failure mode worth catching is "nothing
    is listening", and any HTTP response at all rules that out.

    Our own voice server's ``/health`` also reports which capabilities are
    enabled, so *requires* catches the subtler failure where the server is up but
    was started without the feature the app needs.
    """
    if not url:
        return ProbeResult(name, url, False, "Not configured.", unset_hint)

    origin = _origin(url)
    if not origin:
        return ProbeResult(name, url, False, f"Malformed URL: {url!r}", unset_hint)

    import httpx

    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as client:
            resp = await client.get(f"{origin}/health")
            if resp.status_code == 404:  # reachable, just no /health route
                return ProbeResult(name, url, True, f"Reachable at {origin} (no /health route).")
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:
        return ProbeResult(
            name, url, False, f"Unreachable: {exc}",
            f"Start the voice server on {origin} (see voice-server/README.md).",
        )

    if not isinstance(body, dict):  # a third-party /health with an unfamiliar shape
        return ProbeResult(name, url, True, f"Up at {origin}.")
    if requires and body.get(requires) is False:
        return ProbeResult(
            name, url, False, f"Up at {origin}, but {requires.upper()} is disabled on the server.",
            missing_hint,
        )
    engine = body.get("engine") or ""
    return ProbeResult(name, url, True, f"Up at {origin}." + (f" Engine: {engine}." if engine else ""))


async def check_backends(
    ollama_url: str,
    ollama_model: str,
    tts_url: str,
    stt_url: str = "",
) -> BackendHealth:
    """Probe every backend concurrently and return the aggregated result."""
    tasks = [
        _probe_ollama(ollama_url, ollama_model),
        _probe_http_service(
            "Voice server (TTS)", tts_url,
            "Set TTS_URL on the settings page — without it Reachy hears you but cannot speak.",
            requires="tts",
            missing_hint="Restart the voice server with a TTS engine (see voice-server/README.md).",
        ),
    ]
    # Probe STT even when it shares the TTS origin: the server can be up and
    # serving speech while transcription is off (started without --whisper), and
    # one extra GET at startup is cheaper than that going unnoticed.
    if stt_url:
        tasks.append(
            _probe_http_service(
                "Voice server (STT)", stt_url, "Set STT_URL.",
                requires="stt",
                missing_hint="Restart the voice server with --whisper to enable transcription.",
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    probes: List[ProbeResult] = []
    for result in results:
        if isinstance(result, ProbeResult):
            probes.append(result)
        elif isinstance(result, BaseException):
            probes.append(ProbeResult("unknown", "", False, f"Probe crashed: {result}"))
    return BackendHealth(probes)


def log_health(health: BackendHealth, log: Optional[logging.Logger] = None) -> None:
    """Log each probe, loudly enough that a missing service is obvious."""
    log = log or logger
    for probe in health.probes:
        if probe.ok:
            log.info("Backend OK — %s: %s", probe.name, probe.detail)
        else:
            log.error("Backend DOWN — %s (%s): %s", probe.name, probe.url or "unset", probe.detail)
            if probe.hint:
                log.error("  → %s", probe.hint)
    if not health.ok:
        log.error(
            "The conversation loop will run but cannot complete a turn until the "
            "backends above are reachable."
        )
