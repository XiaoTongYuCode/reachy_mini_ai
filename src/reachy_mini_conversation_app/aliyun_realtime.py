"""Aliyun DashScope native realtime handler."""

from __future__ import annotations
import json
import uuid
import base64
import asyncio
import logging
from typing import Any
from urllib.parse import urlsplit, parse_qsl, urlencode, urlunsplit

import numpy as np
import gradio as gr
from fastrtc import AdditionalOutputs, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import (
    ALIYUN_BACKEND,
    ALIYUN_REALTIME_DEFAULT_VOICE,
    config,
)
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import BaseRealtimeHandler, to_realtime_tools_config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_active_tool_specs
from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg
from reachy_mini_conversation_app.tools.background_tool_manager import ToolCallRoutine, ToolNotification


logger = logging.getLogger(__name__)

__all__ = ["AliyunRealtimeHandler", "build_aliyun_realtime_ws_url"]

_ALIYUN_IMAGE_MAX_BASE64_BYTES = 256 * 1024
_ALIYUN_IMAGE_TARGET_MAX_SIDE = 640
_ALIYUN_CAMERA_SEQUENCE_TOOL_NAME = "aliyun_camera_sequence"


def build_aliyun_realtime_ws_url(realtime_url: str, model: str) -> str:
    """Return an Aliyun realtime websocket URL with the required model query."""
    parsed = urlsplit(realtime_url)
    scheme = parsed.scheme.lower()
    if scheme not in {"ws", "wss"}:
        raise ValueError(f"Expected Aliyun realtime URL to start with ws:// or wss://, got: {realtime_url}")

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key == "model" for key, _value in query_pairs):
        query_pairs.append(("model", model))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query_pairs), parsed.fragment))


def _encode_frame_for_aliyun(frame: NDArray[np.uint8]) -> bytes | None:
    """Encode a camera frame as a bounded JPEG for Aliyun image-buffer input."""
    try:
        jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
        if len(base64.b64encode(jpeg_bytes)) <= _ALIYUN_IMAGE_MAX_BASE64_BYTES:
            return jpeg_bytes

        height, width = frame.shape[:2]
        max_side = max(height, width)
        if max_side <= _ALIYUN_IMAGE_TARGET_MAX_SIDE:
            logger.warning(
                "Skipping Aliyun image frame: encoded base64 size %d exceeds 256KB",
                len(base64.b64encode(jpeg_bytes)),
            )
            return None

        step = max(1, int(np.ceil(max_side / _ALIYUN_IMAGE_TARGET_MAX_SIDE)))
        downsampled = np.ascontiguousarray(frame[::step, ::step])
        jpeg_bytes = encode_bgr_frame_as_jpeg(downsampled)
        if len(base64.b64encode(jpeg_bytes)) <= _ALIYUN_IMAGE_MAX_BASE64_BYTES:
            return jpeg_bytes

        logger.warning(
            "Skipping Aliyun image frame after downsample: encoded base64 size %d exceeds 256KB",
            len(base64.b64encode(jpeg_bytes)),
        )
        return None
    except Exception:
        logger.debug("Failed to encode Aliyun image frame.", exc_info=True)
        return None


class AliyunRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler for Aliyun DashScope Qwen Omni native websocket events."""

    BACKEND_PROVIDER = ALIYUN_BACKEND
    SAMPLE_RATE = 24000
    INPUT_SAMPLE_RATE = 16000
    OUTPUT_SAMPLE_RATE = 24000
    REFRESH_CLIENT_ON_RECONNECT = False
    AUDIO_INPUT_COST_PER_1M = 27.0
    AUDIO_OUTPUT_COST_PER_1M = 107.0
    TEXT_INPUT_COST_PER_1M = 3.3
    TEXT_OUTPUT_COST_PER_1M = 20.0
    IMAGE_INPUT_COST_PER_1M = 3.3

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: str | None = None,
        startup_voice: str | None = None,
    ) -> None:
        """Initialize Aliyun-specific native websocket state."""
        super().__init__(deps, gradio_mode, instance_path, startup_voice)
        self._websocket: Any | None = None
        self._stop_event = asyncio.Event()
        self._has_sent_audio = False
        self._video_active_until = 0.0

    def _get_session_instructions(self) -> str:
        """Return Aliyun session instructions."""
        return self._with_memory_context(get_session_instructions())

    def _get_input_sample_rate(self) -> int:
        """Return the configured Aliyun input sample rate."""
        return int(getattr(config, "ALIYUN_REALTIME_INPUT_SAMPLE_RATE", self.INPUT_SAMPLE_RATE) or self.INPUT_SAMPLE_RATE)

    def _get_output_sample_rate(self) -> int:
        """Return the configured Aliyun output sample rate."""
        return int(
            getattr(config, "ALIYUN_REALTIME_OUTPUT_SAMPLE_RATE", self.OUTPUT_SAMPLE_RATE) or self.OUTPUT_SAMPLE_RATE
        )

    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured Aliyun voice."""
        return get_session_voice(default or ALIYUN_REALTIME_DEFAULT_VOICE)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for provider-side function calling."""
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
        """Return Aliyun's native session config."""
        return {
            "modalities": ["text", "audio"],
            "voice": self.get_current_voice(),
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "instructions": self._get_session_instructions(),
            "input_audio_transcription": {
                "model": "qwen3-asr-flash-realtime",
            },
            "turn_detection": {
                "type": "semantic_vad",
                "threshold": 0.1,
                "prefix_padding_ms": 500,
                "silence_duration_ms": 900,
            },
            "tools": to_realtime_tools_config(tool_specs),
            "tool_choice": "auto",
        }

    async def _build_realtime_client(self) -> Any:
        """Unused: Aliyun native websocket does not use the OpenAI SDK client."""
        raise NotImplementedError("AliyunRealtimeHandler uses native websocket events")

    def _video_fps(self) -> float:
        """Return configured camera frame rate for Aliyun image-buffer input."""
        return max(0.0, float(getattr(config, "ALIYUN_REALTIME_VIDEO_FPS", 1.0) or 0.0))

    def _video_active_seconds(self) -> float:
        """Return how long speech should keep automatic camera frames active."""
        return max(0.0, float(getattr(config, "ALIYUN_REALTIME_VIDEO_ACTIVE_SECONDS", 10.0) or 0.0))

    def _activate_video_window(self) -> None:
        """Allow automatic camera frames briefly after actual speech is detected."""
        active_seconds = self._video_active_seconds()
        if active_seconds <= 0:
            return
        loop_time = asyncio.get_event_loop().time()
        self._video_active_until = max(self._video_active_until, loop_time + active_seconds)

    def _should_send_video_frame(self) -> bool:
        """Return whether the automatic Aliyun camera stream should send a frame now."""
        if not self._has_sent_audio:
            return False
        return asyncio.get_event_loop().time() < self._video_active_until

    def _api_key(self) -> str:
        api_key = (getattr(config, "DASHSCOPE_API_KEY", None) or "").strip()
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for Aliyun DashScope realtime")
        return api_key

    def _connect_url(self) -> str:
        realtime_url = (getattr(config, "ALIYUN_REALTIME_WS_URL", None) or "").strip()
        if not realtime_url:
            raise RuntimeError("ALIYUN_REALTIME_WS_URL is required for Aliyun DashScope realtime")
        model = (getattr(config, "MODEL_NAME", None) or "").strip()
        if not model:
            raise RuntimeError("MODEL_NAME is required for Aliyun DashScope realtime")
        return build_aliyun_realtime_ws_url(realtime_url, model)

    async def _send_event(self, event: dict[str, Any]) -> None:
        websocket = self._websocket
        if websocket is None:
            return
        payload = dict(event)
        payload.setdefault("event_id", f"event_{uuid.uuid4().hex}")
        await websocket.send(json.dumps(payload, ensure_ascii=False))

    async def _send_session_update(self) -> None:
        tool_specs = self._get_active_tool_specs()
        logger.info("Aliyun realtime tools: %s", [tool["name"] for tool in tool_specs])
        await self._send_event(
            {
                "type": "session.update",
                "session": self._get_session_config(tool_specs),
            }
        )
        self._agent_message_log.reset(self._get_session_instructions())

    async def start_up(self) -> None:
        """Open Aliyun's native websocket and run the receive loop."""
        import websockets

        self._stop_event.clear()
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                async with websockets.connect(
                    self._connect_url(),
                    additional_headers={"Authorization": f"Bearer {self._api_key()}"},
                    open_timeout=10,
                    close_timeout=2,
                ) as websocket:
                    self._websocket = websocket
                    self.connection = websocket  # type: ignore[assignment]
                    self._has_sent_audio = False
                    self._video_active_until = 0.0
                    await self._send_session_update()
                    await self._start_memory_session()
                    self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])
                    self._connected_event.set()
                    video_task = self._start_video_sender()
                    try:
                        async for message in websocket:
                            if self._stop_event.is_set():
                                break
                            await self._handle_native_message(message)
                    finally:
                        if video_task is not None:
                            video_task.cancel()
                            try:
                                await video_task
                            except asyncio.CancelledError:
                                pass
                        await self.tool_manager.shutdown()
                        await self._drain_memory_tasks()
                        await self._end_memory_session()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Aliyun realtime session ended unexpectedly (attempt %d/%d): %s", attempt, max_attempts, exc)
                if attempt >= max_attempts or self._stop_event.is_set():
                    raise
                await asyncio.sleep(min(attempt, 3))
            finally:
                self._websocket = None
                self.connection = None
                self._connected_event.clear()

    def _start_video_sender(self) -> asyncio.Task[None] | None:
        fps = self._video_fps()
        if fps <= 0 or self.deps.camera_worker is None:
            return None
        return asyncio.create_task(self._video_sender_loop(fps), name="aliyun-video-sender")

    async def _video_sender_loop(self, fps: float) -> None:
        """Send camera frames to Aliyun via input_image_buffer.append."""
        interval = 1.0 / fps
        logger.info("Aliyun video sender loop started (%.2f FPS)", fps)
        while not self._stop_event.is_set():
            try:
                if self._should_send_video_frame():
                    await self._capture_and_send_image()
            except Exception:
                if self._stop_event.is_set():
                    break
                logger.debug("Aliyun video sender error; will retry.", exc_info=True)
            await asyncio.sleep(interval)
        logger.info("Aliyun video sender loop stopped")

    async def _capture_and_send_image(self) -> bool:
        camera_worker = self.deps.camera_worker
        if camera_worker is None:
            return False
        frame = camera_worker.get_latest_frame()
        if frame is None:
            return False
        jpeg_bytes = _encode_frame_for_aliyun(frame)
        if jpeg_bytes is None:
            return False
        await self._append_image_bytes(jpeg_bytes)
        return True

    async def _start_camera_sequence_sender(
        self,
        duration_seconds: int,
        call_id: str,
        is_idle_tool_call: bool,
    ) -> str:
        """Start an async 1 FPS camera sequence for an Aliyun camera tool call."""
        duration_seconds = max(1, min(120, int(duration_seconds)))
        bg_tool = await self.tool_manager.start_coroutine_tool(
            call_id=f"{call_id}_sequence",
            tool_name=_ALIYUN_CAMERA_SEQUENCE_TOOL_NAME,
            coroutine_factory=lambda _tool_id: self._camera_sequence_sender_loop(
                duration_seconds,
                call_id,
                is_idle_tool_call,
            ),
            is_idle_tool_call=is_idle_tool_call,
        )
        return bg_tool.tool_id

    async def _camera_sequence_sender_loop(
        self,
        duration_seconds: int,
        call_id: str,
        is_idle_tool_call: bool,
    ) -> dict[str, Any]:
        """Push camera frames at 1 FPS, then ask Aliyun to answer from the sequence."""
        sent_frames = 0
        logger.info("Aliyun camera sequence started call_id=%s duration_seconds=%s", call_id, duration_seconds)
        try:
            for index in range(duration_seconds):
                if self._stop_event.is_set() or self._websocket is None:
                    break
                if await self._capture_and_send_image():
                    sent_frames += 1
                if index < duration_seconds - 1:
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Aliyun camera sequence cancelled call_id=%s sent_frames=%s", call_id, sent_frames)
            raise
        except Exception:
            logger.debug("Aliyun camera sequence failed call_id=%s", call_id, exc_info=True)
        finally:
            logger.info("Aliyun camera sequence stopped call_id=%s sent_frames=%s", call_id, sent_frames)

        if is_idle_tool_call or self._stop_event.is_set() or self._websocket is None:
            return {
                "status": "finished",
                "duration_seconds": duration_seconds,
                "sent_frames": sent_frames,
                "response_requested": False,
            }

        await self._send_event(
            {
                "type": "response.create",
                "response": {
                    "instructions": (
                        "Use the camera sequence just received and answer concisely in speech. "
                        f"The sequence request was {duration_seconds}s at 1 FPS; {sent_frames} frame(s) were sent."
                    ),
                },
            }
        )
        return {
            "status": "finished",
            "duration_seconds": duration_seconds,
            "sent_frames": sent_frames,
            "response_requested": True,
        }

    async def _append_image_bytes(self, image_bytes: bytes) -> None:
        await self._send_event(
            {
                "type": "input_image_buffer.append",
                "image": base64.b64encode(image_bytes).decode("ascii"),
            }
        )

    async def _handle_native_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON Aliyun realtime message: %s", message)
            return
        if isinstance(event, dict):
            await self._handle_native_event(event)

    async def _handle_native_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        logger.debug("Aliyun realtime event: %s", event_type)

        if event_type == "input_audio_buffer.speech_started":
            self._activate_video_window()
            self._mark_activity("user_speech_started")
            if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                self._clear_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            return

        if event_type == "input_audio_buffer.speech_stopped":
            self._activate_video_window()
            self._mark_activity("user_speech_stopped")
            self.deps.movement_manager.set_listening(False)
            return

        if event_type == "conversation.item.input_audio_transcription.delta":
            self._mark_activity("user_transcription_delta")
            preview = str(event.get("text") or "") + str(event.get("stash") or event.get("delta") or "")
            if preview:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": preview}))
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            await self._handle_completed_user_transcript(str(event.get("transcript") or ""))
            return

        if event_type == "response.created":
            self._mark_activity("response_created")
            self._response_done_event.clear()
            self._response_started_or_rejected_event.set()
            return

        if event_type == "response.done":
            self._response_done_event.set()
            self._response_started_or_rejected_event.set()
            self.is_idle_tool_call = False
            self._agent_message_log.reset_turn_log()
            return

        if event_type == "response.audio.delta":
            await self._handle_audio_delta(str(event.get("delta") or ""))
            return

        if event_type == "response.audio.done":
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.request_reset_after_current_audio()
            return

        if event_type in {"response.audio_transcript.done", "response.text.done"}:
            await self._handle_assistant_transcript(str(event.get("transcript") or event.get("text") or ""))
            return

        if event_type == "response.function_call_arguments.done":
            await self._handle_function_call_event(event)
            return

        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                await self._handle_function_call_event(item)
            return

        if event_type == "error":
            await self._handle_error_event(event)

    async def _handle_completed_user_transcript(self, transcript: str) -> None:
        transcript = transcript.strip()
        self._mark_activity("user_transcription_completed")
        self.deps.movement_manager.set_listening(False)
        if not transcript:
            return
        self._activate_video_window()
        item_id = None
        if self._should_ignore_completed_input_transcript(item_id, transcript):
            return
        self._schedule_memory_message("user", transcript)
        self._schedule_memory_context_refresh(transcript)
        self._agent_message_log.append("user", transcript)
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

    async def _handle_assistant_transcript(self, transcript: str) -> None:
        transcript = transcript.strip()
        if not transcript:
            return
        self._mark_activity("assistant_transcript_done")
        self._schedule_memory_message("assistant", transcript)
        self._agent_message_log.append("assistant", transcript)
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": transcript}))

    async def _handle_audio_delta(self, delta: str) -> None:
        if not delta:
            return
        decoded_pcm_bytes = base64.b64decode(delta)
        decoded_pcm = np.frombuffer(decoded_pcm_bytes, dtype=np.int16).reshape(1, -1)
        if self.gradio_mode and self.deps.head_wobbler is not None:
            self.deps.head_wobbler.feed_pcm(decoded_pcm, self.output_sample_rate)
        self._mark_activity("assistant_audio_delta")
        await self.output_queue.put((self.output_sample_rate, decoded_pcm))

    async def _handle_function_call_event(self, event: dict[str, Any]) -> None:
        tool_name = event.get("name")
        args_json_str = event.get("arguments")
        call_id = str(event.get("call_id") or event.get("id") or uuid.uuid4())
        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
            logger.error("Invalid Aliyun function call event: %s", event)
            return
        bg_tool = await self.tool_manager.start_tool(
            call_id=call_id,
            tool_call_routine=ToolCallRoutine(
                tool_name=tool_name,
                args_json_str=args_json_str,
                deps=self.deps,
            ),
            is_idle_tool_call=self.is_idle_tool_call,
        )
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": f"Used tool {tool_name} with args {args_json_str}. Tool ID: {bg_tool.tool_id}",
                }
            )
        )

    async def _handle_error_event(self, event: dict[str, Any]) -> None:
        error = event.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error)
            code = str(error.get("code") or error.get("type") or "")
        else:
            message = str(error or event)
            code = ""
        logger.error("Aliyun realtime error [%s]: %s", code, message)
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {message}"}))

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Send completed local tool results back to Aliyun native realtime."""
        if bg_tool.tool_name == _ALIYUN_CAMERA_SEQUENCE_TOOL_NAME:
            status_payload: dict[str, Any] = {
                "role": "assistant",
                "content": json.dumps(
                    bg_tool.result if bg_tool.error is None else {"error": bg_tool.error},
                    ensure_ascii=False,
                ),
                "metadata": {
                    "title": "Aliyun camera sequence",
                    "status": "done",
                },
            }
            await self.output_queue.put(AdditionalOutputs(status_payload))
            return

        if bg_tool.error is not None:
            tool_result: dict[str, Any] = {"error": bg_tool.error}
        elif bg_tool.result is not None:
            tool_result = bg_tool.result
        else:
            tool_result = {"error": "No result returned from tool execution"}

        defer_response_create = False
        if bg_tool.tool_name == "camera" and tool_result.get("camera_sequence_requested") is True:
            duration_seconds = max(1, min(120, int(tool_result.get("duration_seconds") or 1)))
            camera_sequence_tool_id = await self._start_camera_sequence_sender(
                duration_seconds,
                str(bg_tool.id),
                bg_tool.is_idle_tool_call,
            )
            tool_result_for_model: dict[str, Any] = {
                "camera_sequence_started": True,
                "duration_seconds": duration_seconds,
                "fps": 1,
                "tool_id": camera_sequence_tool_id,
            }
            defer_response_create = True
        elif bg_tool.tool_name == "camera" and ("b64_im" in tool_result or "b64_images" in tool_result):
            image_values = tool_result.get("b64_images")
            if not isinstance(image_values, list):
                image_values = [tool_result.get("b64_im")]
            image_count = 0
            errors = 0
            for b64_image in image_values:
                if not b64_image:
                    continue
                try:
                    await self._append_image_bytes(base64.b64decode(str(b64_image)))
                    image_count += 1
                except Exception as exc:
                    errors += 1
                    logger.warning("Failed to append camera image to Aliyun image buffer: %s", exc)
            if image_count:
                tool_result_for_model = {"image_attached": True, "image_count": image_count}
            else:
                tool_result_for_model = {"image_attached": False, "error": "Failed to attach camera image"}
            if errors:
                tool_result_for_model["failed_image_count"] = errors
        else:
            tool_result_for_model = self._sanitize_tool_result_for_model(bg_tool.tool_name, tool_result)

        await self._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": bg_tool.id,
                    "output": json.dumps(tool_result_for_model, ensure_ascii=False),
                },
            }
        )
        self._agent_message_log.append("tool", tool_result_for_model, call_id=bg_tool.id, name=bg_tool.tool_name)

        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": json.dumps(tool_result_for_model, ensure_ascii=False),
                    "metadata": {
                        "title": f"🛠️ Used tool {bg_tool.tool_name}",
                        "status": "done",
                    },
                }
            )
        )

        if bg_tool.tool_name == "camera" and self.deps.camera_worker is not None:
            np_img = self.deps.camera_worker.get_latest_frame()
            rgb_frame = np_img[:, :, ::-1].copy() if np_img is not None and np_img.ndim == 3 else np_img
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": gr.Image(value=rgb_frame)}))

        if not bg_tool.is_idle_tool_call and not defer_response_create:
            await self._send_event(
                {
                    "type": "response.create",
                    "response": {
                        "instructions": "Use the tool result just returned and answer concisely in speech.",
                    },
                }
            )

    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        """Receive microphone audio and send it to Aliyun's input audio buffer."""
        if self._websocket is None:
            return
        input_sample_rate, audio_frame = frame
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))
        audio_frame = audio_to_int16(audio_frame)
        try:
            await self._send_event(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio_frame.tobytes()).decode("ascii"),
                }
            )
            self._has_sent_audio = True
        except Exception as exc:
            logger.debug("Dropping Aliyun audio frame: connection not ready (%s)", exc)

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit the next audio or transcript item."""
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 180.0 and self._response_done_event.is_set() and self.deps.movement_manager.is_idle():
            await self.send_idle_signal(idle_duration)
            self.last_activity_time = asyncio.get_event_loop().time()
        return await self._wait_for_output_item()

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Ask Aliyun for an idle-only tool choice."""
        self.is_idle_tool_call = True
        await self._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"[Idle time update: no activity for {idle_duration:.1f}s]",
                        }
                    ],
                },
            }
        )
        await self._send_event(
            {
                "type": "response.create",
                "response": {
                    "instructions": (
                        "You MUST respond with function calls only. Use idle_do_nothing if no action is needed."
                    ),
                    "tool_choice": "required",
                },
            }
        )

    async def change_voice(self, voice: str) -> str:
        """Change the Aliyun voice and update the live session when connected."""
        resolved = self._resolve_backend_voice(
            voice,
            source="requested voice",
            fallback=ALIYUN_REALTIME_DEFAULT_VOICE,
        )
        self._voice_override = resolved
        if self._websocket is not None:
            await self._send_session_update()
            return f"Voice changed to {resolved}."
        return "Voice changed. Will take effect on next connection."

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile and update the live Aliyun session."""
        from reachy_mini_conversation_app.config import set_custom_profile

        try:
            set_custom_profile(profile)
            if self._websocket is not None:
                await self._send_session_update()
                return "Applied personality to Aliyun realtime session."
            return "Applied personality. Will take effect on next connection."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def shutdown(self) -> None:
        """Close the native websocket and background tasks."""
        self._stop_event.set()
        self._response_done_event.set()
        await self.tool_manager.shutdown()
        await self._drain_memory_tasks()
        await self._end_memory_session()
        websocket = self._websocket
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                pass
        self._websocket = None
        self.connection = None
