import logging
from typing import Any, Dict

from reachy_local_assistant.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class HeadTracking(Tool):
    """Toggle head tracking state."""

    name = "head_tracking"
    description = "Toggle head tracking state."
    parameters_schema = {
        "type": "object",
        "properties": {"start": {"type": "boolean"}},
        "required": ["start"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Enable or disable head tracking."""
        enable = bool(kwargs.get("start"))

        # Prefer daemon-side tracking (SDK >= 1.9): the Pi runs the face detector
        # itself, so this works with no client-side mediapipe installed at all.
        daemon_ok = False
        robot = deps.reachy_mini
        if robot is not None and hasattr(robot, "start_head_tracking"):
            try:
                if enable:
                    robot.start_head_tracking(weight=1.0)
                else:
                    robot.stop_head_tracking()
                daemon_ok = True
            except Exception as e:
                logger.warning("Daemon-side head tracking toggle failed: %s", e)

        # Client-side fallback (CameraWorker + mediapipe, if configured).
        if deps.camera_worker is not None:
            deps.camera_worker.set_head_tracking_enabled(enable)

        status = "started" if enable else "stopped"
        logger.info("Tool call: head_tracking %s (daemon=%s)", status, daemon_ok)
        return {"status": f"head tracking {status}"}
