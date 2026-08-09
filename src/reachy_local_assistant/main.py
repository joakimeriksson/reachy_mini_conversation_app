"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Optional

from fastapi import FastAPI

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_local_assistant.utils import (
    parse_args,
    setup_logger,
    initialize_camera_and_vision,
    log_connection_troubleshooting,
)


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_local_assistant.moves import MovementManager
    from reachy_local_assistant.console import LocalStream
    from reachy_local_assistant.ollama_handler import OllamaConversationHandler
    from reachy_local_assistant.tools.core_tools import ToolDependencies
    from reachy_local_assistant.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if hasattr(args, "mcp_servers") and args.mcp_servers:
        from reachy_local_assistant.config import config

        os.environ["MCP_SERVER_URLS"] = args.mcp_servers
        config.MCP_SERVER_URLS = args.mcp_servers
        logger.info("MCP server URLs set from CLI: %s", args.mcp_servers)

    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking disabled: --no-camera flag is set. Remove --no-camera to enable head tracking.")

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(f"Connection timeout: Failed to connect to Reachy Mini daemon. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(f"Connection failed: Unable to establish connection to Reachy Mini. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during robot initialization: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    camera_worker = initialize_camera_and_vision(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        head_wobbler=head_wobbler,
        instance_path=instance_path,
    )
    handler = OllamaConversationHandler(deps, gradio_mode=False, instance_path=instance_path)

    # Headless only: audio flows through the robot's media pipeline and the settings
    # UI is served by the FastAPI settings_app. gradio/fastrtc were removed — they
    # capped pydantic below what reachy-mini requires (see stream_shim.py).
    stream_manager = LocalStream(
        handler,
        robot,
        settings_app=settings_app,
        instance_path=instance_path,
    )

    def go_to_sleep() -> dict[str, str]:
        """Sleep pose + clean app stop; used by the tool and the inactivity timeout.

        Runs off the event loop (asyncio.to_thread) — it moves motors and talks to
        the daemon. Under the apps runtime we also POST stop-current-app so the
        runtime records a clean stop rather than a crash.
        """
        result: dict[str, str] = {"status": "sleeping"}
        try:
            robot.goto_sleep()
        except Exception as e:
            logger.warning("goto_sleep motion failed: %s", e)
            result["motion_error"] = str(e)
        client = getattr(robot, "client", None)
        host, port = getattr(client, "host", None), getattr(client, "port", None)
        if host and port:
            try:
                import urllib.request

                req = urllib.request.Request(f"http://{host}:{port}/api/apps/stop-current-app", method="POST")
                with urllib.request.urlopen(req, timeout=2.0):
                    pass
                result["app_stopped"] = "daemon"
                return result
            except Exception as e:  # standalone run, or a daemon without the route
                logger.info("Daemon stop-current-app unavailable (%s); stopping locally", e)
        if app_stop_event is not None:
            app_stop_event.set()
            result["app_stopped"] = "stop_event"
        else:
            stream_manager.close()
            result["app_stopped"] = "stream_closed"
        return result

    deps.go_to_sleep = go_to_sleep

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyLocalAssistant(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        # is_wireless = reachy_mini.client.get_status()["wireless_version"]
        # args.head_tracker = None if is_wireless else "mediapipe"

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyLocalAssistant()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
