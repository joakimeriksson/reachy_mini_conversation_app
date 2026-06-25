"""Tests for utility helpers."""

import argparse
from unittest.mock import MagicMock, patch

from reachy_mini_conversation_app.utils import initialize_camera_and_vision


def _args(**kw) -> argparse.Namespace:
    base = dict(no_camera=False, head_tracker=None, local_webcam=False, webcam_index=0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_no_camera_returns_none() -> None:
    assert initialize_camera_and_vision(_args(no_camera=True), MagicMock()) is None


def test_camera_enabled_creates_worker() -> None:
    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_cw:
        result = initialize_camera_and_vision(_args(no_camera=False), MagicMock())
    mock_cw.assert_called_once()
    assert result is mock_cw.return_value


def test_local_webcam_creates_worker_even_without_robot_camera() -> None:
    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_cw:
        result = initialize_camera_and_vision(_args(no_camera=True, local_webcam=True), MagicMock())
    mock_cw.assert_called_once()
    assert result is mock_cw.return_value
