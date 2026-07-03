from __future__ import annotations
import json
import base64
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.aliyun_realtime as aliyun_mod
from reachy_mini_conversation_app.config import ALIYUN_BACKEND, config, get_default_voice_for_backend
from reachy_mini_conversation_app.aliyun_realtime import AliyunRealtimeHandler, build_aliyun_realtime_ws_url
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.tool_constants import ToolState
from reachy_mini_conversation_app.tools.background_tool_manager import ToolNotification


def _make_handler() -> AliyunRealtimeHandler:
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return AliyunRealtimeHandler(deps)


def test_aliyun_backend_uses_qwen_omni_defaults() -> None:
    """Aliyun backend should default to Qwen3.5 Omni realtime settings."""
    assert get_default_voice_for_backend(ALIYUN_BACKEND) == "Ethan"
    assert config_mod._resolve_model_name(ALIYUN_BACKEND, None) == "qwen3.5-omni-flash-realtime"


def test_build_aliyun_realtime_ws_url_adds_model_query() -> None:
    """Aliyun native websocket URL should include the model query."""
    url = build_aliyun_realtime_ws_url(
        "wss://dashscope.example.test/api-ws/v1/realtime?workspace=default",
        "qwen3.5-omni-flash-realtime",
    )

    assert url == (
        "wss://dashscope.example.test/api-ws/v1/realtime?"
        "workspace=default&model=qwen3.5-omni-flash-realtime"
    )


def test_aliyun_session_config_sends_tools_for_provider_side_function_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aliyun session config should include active tools directly."""
    monkeypatch.setattr(aliyun_mod, "get_session_instructions", lambda: "你是 Reachy。")
    monkeypatch.setattr(aliyun_mod, "get_session_voice", lambda default=None: "Ethan")
    monkeypatch.setattr(config, "MEMORY_CONTEXT_ENABLED", False)

    handler = _make_handler()
    session = handler._get_session_config(
        [
            {
                "type": "function",
                "name": "dance",
                "description": "Dance now",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    )

    assert session["instructions"] == "你是 Reachy。"
    assert session["voice"] == "Ethan"
    assert session["modalities"] == ["text", "audio"]
    assert session["input_audio_format"] == "pcm"
    assert session["output_audio_format"] == "pcm"
    assert session["turn_detection"]["type"] == "semantic_vad"
    assert session["tool_choice"] == "auto"
    assert session["tools"][0]["name"] == "dance"


@pytest.mark.asyncio
async def test_aliyun_audio_event_aliases_emit_24khz_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    """DashScope-style audio event names should reuse the common realtime output path."""
    monkeypatch.setattr(aliyun_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(aliyun_mod, "get_session_voice", lambda default=None: "Ethan")
    monkeypatch.setattr(aliyun_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "aliyun")

    audio_delta = base64.b64encode(b"\x00\x00\x01\x00").decode("utf-8")

    class FakeEvent:
        def __init__(self, event_type: str, **kwargs: Any) -> None:
            self.type = event_type
            for key, value in kwargs.items():
                setattr(self, key, value)

    handler = _make_handler()
    await handler._handle_native_event(FakeEvent("response.created").__dict__)
    await handler._handle_native_event(FakeEvent("response.audio.delta", delta=audio_delta).__dict__)
    await handler._handle_native_event(FakeEvent("response.audio_transcript.done", transcript="你好").__dict__)
    await handler._handle_native_event(FakeEvent("response.audio.done").__dict__)
    await handler._handle_native_event(FakeEvent("response.done").__dict__)

    sample_rate, pcm = await handler.output_queue.get()
    transcript = await handler.output_queue.get()
    assert sample_rate == 24000
    assert pcm.tolist() == [[0, 1]]
    assert transcript.args[0] == {"role": "assistant", "content": "你好"}


@pytest.mark.asyncio
async def test_aliyun_handler_resamples_input_at_16khz(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aliyun input and output sample rates should not be forced to the same value."""
    monkeypatch.setattr(config, "ALIYUN_REALTIME_INPUT_SAMPLE_RATE", 16000)
    monkeypatch.setattr(config, "ALIYUN_REALTIME_OUTPUT_SAMPLE_RATE", 24000)

    handler = _make_handler()

    assert handler.input_sample_rate == 16000
    assert handler.output_sample_rate == 24000


@pytest.mark.asyncio
async def test_aliyun_camera_tool_result_uses_native_image_buffer() -> None:
    """Aliyun should send camera images through native input_image_buffer events."""

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    handler = _make_handler()
    fake_websocket = FakeWebSocket()
    handler._websocket = fake_websocket
    handler._has_sent_audio = True

    await handler._handle_tool_result(
        ToolNotification(
            id="call_camera_1",
            tool_name="camera",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"b64_im": base64.b64encode(b"jpeg-bytes").decode("ascii")},
        )
    )

    assert [event["type"] for event in fake_websocket.sent] == [
        "input_image_buffer.append",
        "conversation.item.create",
        "response.create",
    ]
    created_item = fake_websocket.sent[1]["item"]
    model_output = json.loads(created_item["output"])

    assert created_item["type"] == "function_call_output"
    assert model_output["image_attached"] is True


@pytest.mark.asyncio
async def test_aliyun_camera_tool_result_appends_sequence_images() -> None:
    """Aliyun should append each requested 1 FPS camera sequence frame natively."""

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    handler = _make_handler()
    fake_websocket = FakeWebSocket()
    handler._websocket = fake_websocket
    b64_images = [
        base64.b64encode(b"jpeg-frame-1").decode("ascii"),
        base64.b64encode(b"jpeg-frame-2").decode("ascii"),
    ]

    await handler._handle_tool_result(
        ToolNotification(
            id="call_camera_sequence_1",
            tool_name="camera",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"b64_images": b64_images, "frame_count": 2, "fps": 1},
        )
    )

    assert [event["type"] for event in fake_websocket.sent] == [
        "input_image_buffer.append",
        "input_image_buffer.append",
        "conversation.item.create",
        "response.create",
    ]
    model_output = json.loads(fake_websocket.sent[2]["item"]["output"])
    assert model_output == {"image_attached": True, "image_count": 2}


@pytest.mark.asyncio
async def test_aliyun_camera_sequence_tool_result_starts_async_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duration_seconds camera call should return immediately and start async frame sending."""

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    started: list[tuple[int, str, bool]] = []
    handler = _make_handler()
    handler._websocket = FakeWebSocket()

    async def fake_start_sequence(duration_seconds: int, call_id: str, is_idle_tool_call: bool) -> str:
        started.append((duration_seconds, call_id, is_idle_tool_call))
        return "aliyun_camera_sequence-call_camera_sequence_2_sequence-1"

    monkeypatch.setattr(handler, "_start_camera_sequence_sender", fake_start_sequence)

    await handler._handle_tool_result(
        ToolNotification(
            id="call_camera_sequence_2",
            tool_name="camera",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={
                "camera_sequence_requested": True,
                "duration_seconds": 120,
                "fps": 1,
                "b64_im": base64.b64encode(b"first-frame").decode("ascii"),
            },
        )
    )

    assert started == [(120, "call_camera_sequence_2", False)]
    assert [event["type"] for event in handler._websocket.sent] == ["conversation.item.create"]
    model_output = json.loads(handler._websocket.sent[0]["item"]["output"])
    assert model_output == {
        "camera_sequence_started": True,
        "duration_seconds": 120,
        "fps": 1,
        "tool_id": "aliyun_camera_sequence-call_camera_sequence_2_sequence-1",
    }


@pytest.mark.asyncio
async def test_aliyun_camera_sequence_sender_pushes_frames_then_requests_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async sequence sender should stream frames over time, then trigger the model answer."""

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    async def no_sleep(_seconds: float) -> None:
        return None

    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=camera_worker)
    handler = AliyunRealtimeHandler(deps)
    fake_websocket = FakeWebSocket()
    handler._websocket = fake_websocket
    monkeypatch.setattr(aliyun_mod.asyncio, "sleep", no_sleep)

    result = await handler._camera_sequence_sender_loop(2, "call_camera_sequence_3", False)

    assert [event["type"] for event in fake_websocket.sent] == [
        "input_image_buffer.append",
        "input_image_buffer.append",
        "response.create",
    ]
    assert camera_worker.get_latest_frame.call_count == 2
    assert result == {
        "status": "finished",
        "duration_seconds": 2,
        "sent_frames": 2,
        "response_requested": True,
    }


@pytest.mark.asyncio
async def test_aliyun_automatic_video_only_runs_during_speech_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle audio should not keep the automatic 1 FPS image stream open forever."""
    monkeypatch.setattr(config, "ALIYUN_REALTIME_VIDEO_ACTIVE_SECONDS", 10.0)

    handler = _make_handler()

    handler._has_sent_audio = False
    assert handler._should_send_video_frame() is False

    handler._has_sent_audio = True
    assert handler._should_send_video_frame() is False

    await handler._handle_native_event({"type": "input_audio_buffer.speech_started"})
    assert handler._should_send_video_frame() is True

    handler._video_active_until = 0.0
    assert handler._should_send_video_frame() is False


@pytest.mark.asyncio
async def test_aliyun_capture_and_send_image_uses_camera_frame() -> None:
    """Automatic speech-window video frames should use native image buffer events."""

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=camera_worker)
    handler = AliyunRealtimeHandler(deps)
    fake_websocket = FakeWebSocket()
    handler._websocket = fake_websocket

    assert await handler._capture_and_send_image() is True
    assert fake_websocket.sent[0]["type"] == "input_image_buffer.append"
    assert isinstance(fake_websocket.sent[0]["image"], str)
