"""Patched Reachy Mini daemon launcher (macOS GStreamer crash workaround).

The stock ``reachy-mini-daemon`` segfaults on macOS during media startup: the
GStreamer ``Gst.DeviceMonitor`` teardown (``gst_device_monitor_stop``) races in
the AVFoundation device provider and crashes the process (intermittently). The
crash is in the reachy-mini SDK, not this app, so we patch it at launch instead
of forking the SDK.

The fix replaces ``reachy_mini.media.device_detection.gst_monitor_devices`` with
a version that (1) caches results per device class so the monitor runs at most
once, and (2) never calls ``monitor.stop()`` — the monitor is left running for
the process lifetime, which avoids the crashing teardown entirely. On a dev Mac
there is no "Reachy Mini" USB audio/camera anyway, so the device list is just
used to fall back to default sources.

Usage (same flags as the stock daemon):

    uv run python scripts/reachy_daemon.py --mockup-sim --no-preload-datasets --headless
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("reachy_daemon_patched")


def _install_device_monitor_patch() -> None:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    from reachy_mini.media import device_detection as dd

    if not Gst.is_initialized():
        Gst.init(None)

    _cache: dict[str, list] = {}

    def safe_gst_monitor_devices(filter_class: str) -> list:
        if filter_class in _cache:
            return _cache[filter_class]
        monitor = Gst.DeviceMonitor()
        monitor.add_filter(filter_class)
        monitor.start()
        try:
            devices = dd.gst_devices_to_device_infos(monitor.get_devices())
        except Exception:
            logger.exception("device enumeration failed for %s", filter_class)
            devices = []
        # Intentionally DO NOT call monitor.stop(): the AVFoundation provider
        # teardown races and segfaults on macOS. Leak it for the process.
        _cache[filter_class] = devices
        return devices

    dd.gst_monitor_devices = safe_gst_monitor_devices
    logger.info("Patched gst_monitor_devices (macOS device-monitor crash workaround)")

    # --- Expose the Mac webcam to the daemon's video pipeline ---------------
    # The stock find_video_device only accepts a "Reachy" camera, so on a dev
    # Mac it returns no camera. Fall back to the first enumerated Video/Source
    # (avfvideosrc uses the device index on macOS).
    import platform as _platform

    _orig_get_video_device = dd.get_video_device

    def patched_get_video_device():
        path, specs = _orig_get_video_device()
        if path or _platform.system() != "Darwin":
            return path, specs
        try:
            for d in dd.gst_monitor_devices("Video/Source"):
                idx = getattr(d, "index", None)
                if idx is None:
                    continue
                name = getattr(d, "display_name", "") or "camera"
                logger.info("Using macOS camera %r at avf index %s", name, idx)
                return str(idx), dd._make_camera_specs("mac")
        except Exception:
            logger.exception("macOS camera fallback failed")
        return path, specs

    dd.get_video_device = patched_get_video_device
    # media_server did `from ... import get_video_device`, so rebind there too
    # if it is already imported.
    _ms = sys.modules.get("reachy_mini.media.media_server")
    if _ms is not None and hasattr(_ms, "get_video_device"):
        _ms.get_video_device = patched_get_video_device
    logger.info("Patched get_video_device (expose macOS webcam to daemon)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    _install_device_monitor_patch()
    from reachy_mini.daemon.app.main import main as daemon_main

    daemon_main()


if __name__ == "__main__":
    sys.exit(main())
