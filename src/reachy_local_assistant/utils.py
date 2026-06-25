from __future__ import annotations
import logging
import argparse
import warnings
from typing import Optional

from reachy_mini import ReachyMini
from reachy_local_assistant.camera_worker import CameraWorker


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe"],
        default=None,
        help="Head-tracking backend: yolo uses a local face detector, mediapipe uses reachy_mini_toolbox. Disabled by default.",
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--local-webcam",
        default=False,
        action="store_true",
        help="Dev only: use the local machine's webcam (OpenCV) when robot.media has no camera",
    )
    parser.add_argument("--webcam-index", type=int, default=0, help="OpenCV webcam device index")
    parser.add_argument("--gradio", default=False, action="store_true", help="Open gradio interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name to target. Must match the daemon's --robot-name when connecting to a specific robot, mainly useful for development with multiple robots.",
    )
    parser.add_argument(
        "--mcp-servers",
        type=str,
        default=None,
        help="Comma-separated MCP server URLs (overrides MCP_SERVER_URLS env var). "
        "Append ' token=<jwt>' or ' api_key=<key>' after a URL for auth.",
    )
    return parser.parse_known_args()


def initialize_camera_and_vision(
    args: argparse.Namespace,
    current_robot: ReachyMini,
) -> CameraWorker | None:
    """Initialize camera capture and optional head tracking.

    Camera *vision* (describing what's seen) is handled by the multimodal LLM
    (Gemma) — frames are injected into the chat by the camera tool — so there is
    no separate vision model here.
    """
    camera_worker: Optional[CameraWorker] = None
    head_tracker = None

    local_webcam = getattr(args, "local_webcam", False)
    if not args.no_camera or local_webcam:
        if args.head_tracker is not None:
            if args.head_tracker == "yolo":
                from reachy_local_assistant.vision.yolo_head_tracker import HeadTracker

                head_tracker = HeadTracker()
            elif args.head_tracker == "mediapipe":
                from reachy_mini_toolbox.vision import HeadTracker  # type: ignore[no-redef]

                head_tracker = HeadTracker()

        camera_worker = CameraWorker(
            current_robot,
            head_tracker,
            local_webcam=local_webcam,
            webcam_index=getattr(args, "webcam_index", 0),
        )

    return camera_worker


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Quiet the per-frame "Camera is not initialized" spam when robot.media has
    # no camera (e.g. desktop sim with the local webcam fallback).
    logging.getLogger("reachy_mini.media.media_manager").setLevel(logging.ERROR)

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
