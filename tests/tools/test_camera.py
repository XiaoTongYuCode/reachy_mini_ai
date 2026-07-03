"""Tests for the camera tool."""

import base64
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_camera_tool_preserves_frame_color_for_uploaded_jpeg() -> None:
    """The JPEG uploaded to the model should preserve the intended frame color."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "b64_im" in result

    jpeg_bytes = base64.b64decode(result["b64_im"])
    decoded = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    pixel = decoded.getpixel((0, 0))
    assert isinstance(pixel, tuple)
    red, green, blue = pixel

    assert red > 200
    assert green < 40
    assert blue < 40


@pytest.mark.asyncio
async def test_camera_tool_uses_local_vision_processor_when_available() -> None:
    """The camera tool should use on-demand local vision when configured."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((32, 32, 3), dtype=np.uint8)

    vision_processor = MagicMock()
    vision_processor.process_image.return_value = "A red cup on a table."

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_processor=vision_processor,
    )

    result = await Camera()(deps, question="What do you see?")

    assert result == {"image_description": "A red cup on a table."}
    vision_processor.process_image.assert_called_once_with(
        camera_worker.get_latest_frame.return_value,
        "What do you see?",
    )


@pytest.mark.asyncio
async def test_camera_tool_starts_one_fps_sequence_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The camera tool should request long sequences without sleeping through them."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "aliyun")
    monkeypatch.setattr(config, "MODEL_NAME", "qwen3.5-omni-flash-realtime")
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((16, 16, 3), dtype=np.uint8)

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    result = await Camera()(deps, question="Watch for changes.", duration_seconds=3)

    assert result["camera_sequence_requested"] is True
    assert result["duration_seconds"] == 3
    assert result["fps"] == 1
    assert base64.b64decode(result["b64_im"])
    camera_worker.get_latest_frame.assert_called_once()


@pytest.mark.asyncio
async def test_camera_tool_caps_sequence_duration_at_120_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The camera sequence duration should be bounded for runaway tool calls."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "aliyun")
    monkeypatch.setattr(config, "MODEL_NAME", "qwen3.5-omni-flash-realtime")
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    result = await Camera()(deps, question="Watch for a while.", duration_seconds=999)

    assert result["camera_sequence_requested"] is True
    assert result["duration_seconds"] == 120


@pytest.mark.asyncio
async def test_camera_tool_ignores_sequence_duration_for_non_aliyun_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only Aliyun should turn duration_seconds into a realtime image sequence."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")
    monkeypatch.setattr(config, "MODEL_NAME", "gpt-realtime")
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    result = await Camera()(deps, question="Watch for a while.", duration_seconds=10)

    assert "b64_im" in result
    assert "camera_sequence_requested" not in result
