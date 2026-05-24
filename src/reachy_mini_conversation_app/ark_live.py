"""Volcengine Realtime Dialogue handler.

The public Realtime Dialogue API uses Volcengine's binary websocket protocol,
not the OpenAI Realtime JSON event protocol. This module keeps that transport
isolated while matching the app's existing ConversationHandler contract.
"""

from __future__ import annotations
import re
import json
import uuid
import struct
import asyncio
import logging
from typing import Any, Final, Optional

import numpy as np
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import (
    ARK_BACKEND,
    ARK_AVAILABLE_VOICES,
    ARK_REALTIME_DEFAULT_VOICE,
    config,
)
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.storage.memory import MemoryContextProvider
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_active_tool_specs
from reachy_mini_conversation_app.agent_observability import AgentMessageListLog
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

ARK_INPUT_SAMPLE_RATE: Final[int] = 16000
ARK_OUTPUT_SAMPLE_RATE: Final[int] = 24000
_ARK_TOOL_ROUTER_TIMEOUT_SECONDS: Final[float] = 20.0
_ARK_SIDECAR_HISTORY_MESSAGES: Final[int] = 8
_ARK_SUPPRESSED_TURN_DRAIN_DELAY_SECONDS: Final[float] = 1.0

_MESSAGE_TYPE_FULL_CLIENT_REQUEST = 0x1
_MESSAGE_TYPE_AUDIO_ONLY_REQUEST = 0x2
_MESSAGE_TYPE_FULL_SERVER_RESPONSE = 0x9
_MESSAGE_TYPE_AUDIO_ONLY_RESPONSE = 0xB
_MESSAGE_TYPE_ERROR_INFORMATION = 0xF
_MESSAGE_FLAG_CARRY_EVENT_ID = 0x4
_SERIALIZATION_RAW = 0x0
_SERIALIZATION_JSON = 0x1
_COMPRESSION_NONE = 0x0

_EVENT_START_CONNECTION = 1
_EVENT_FINISH_CONNECTION = 2
_EVENT_START_SESSION = 100
_EVENT_FINISH_SESSION = 102
_EVENT_TASK_REQUEST = 200
_EVENT_CHAT_RAG_TEXT = 502
_EVENT_CONNECTION_STARTED = 50
_EVENT_CONNECTION_FAILED = 51
_EVENT_CONNECTION_FINISHED = 52
_EVENT_SESSION_STARTED = 150
_EVENT_SESSION_FAILED = 153
_EVENT_USAGE = 154
_EVENT_TTS_RESPONSE = 352
_EVENT_TTS_ENDED = 359
_EVENT_ASR_RESPONSE = 451
_EVENT_ASR_ENDED = 459
_EVENT_CHAT_RESPONSE = 550
_EVENT_CHAT_ENDED = 559
_EVENT_DIALOG_COMMON_ERROR = 599


class ArkRealtimeAuthenticationError(RuntimeError):
    """Volcengine websocket authentication failed before protocol startup."""


class ArkRealtimeFrame(dict[str, Any]):
    """Parsed Volcengine realtime websocket frame."""


def _json_dumps_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _build_full_client_payload(
    event: int, *, session_id: str | None = None, payload: dict[str, Any] | None = None
) -> bytes:
    """Build a Volcengine full-client binary frame."""
    frame = bytearray(
        [
            (0x1 << 4) | 0x1,
            (_MESSAGE_TYPE_FULL_CLIENT_REQUEST << 4) | _MESSAGE_FLAG_CARRY_EVENT_ID,
            (_SERIALIZATION_JSON << 4) | _COMPRESSION_NONE,
            0,
        ]
    )
    frame.extend(struct.pack(">I", event))
    if session_id:
        session_bytes = session_id.encode("utf-8")
        frame.extend(struct.pack(">I", len(session_bytes)))
        frame.extend(session_bytes)
    payload_bytes = _json_dumps_bytes(payload or {})
    frame.extend(struct.pack(">I", len(payload_bytes)))
    frame.extend(payload_bytes)
    return bytes(frame)


def _build_audio_payload(session_id: str, audio_data: bytes) -> bytes:
    """Build a Volcengine audio-only TaskRequest frame."""
    session_bytes = session_id.encode("utf-8")
    frame = bytearray(
        [
            (0x1 << 4) | 0x1,
            (_MESSAGE_TYPE_AUDIO_ONLY_REQUEST << 4) | _MESSAGE_FLAG_CARRY_EVENT_ID,
            (_SERIALIZATION_RAW << 4) | _COMPRESSION_NONE,
            0,
        ]
    )
    frame.extend(struct.pack(">I", _EVENT_TASK_REQUEST))
    frame.extend(struct.pack(">I", len(session_bytes)))
    frame.extend(session_bytes)
    frame.extend(struct.pack(">I", len(audio_data)))
    frame.extend(audio_data)
    return bytes(frame)


def _openai_tool_specs_to_chat_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert app tool specs to Chat Completions tool definitions."""
    tools: list[dict[str, Any]] = []
    for spec in specs:
        name = spec.get("name")
        if spec.get("type") != "function" or not isinstance(name, str):
            continue
        function: dict[str, Any] = {
            "name": name,
            "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
        }
        description = spec.get("description")
        if isinstance(description, str):
            function["description"] = description
        tools.append({"type": "function", "function": function})
    return tools


def _sanitize_tool_result_for_model(tool_name: str, tool_result: dict[str, Any]) -> dict[str, Any]:
    """Remove large or binary fields before sending tool output through text model context."""
    del tool_name
    dropped_image = False
    large_field_names = {
        "b64_im",
        "image_base64",
        "base64_image",
        "image_b64",
        "audio_base64",
        "video_base64",
    }

    def _sanitize(value: Any) -> Any:
        nonlocal dropped_image
        if isinstance(value, dict):
            sanitized_dict: dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                key_lower = key_str.lower()
                if key_lower in large_field_names or ("base64" in key_lower and isinstance(item, str)):
                    dropped_image = dropped_image or "image" in key_lower or key_lower == "b64_im"
                    continue
                sanitized_value = _sanitize(item)
                if sanitized_value is not None:
                    sanitized_dict[key_str] = sanitized_value
            return sanitized_dict
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        if isinstance(value, (bytes, bytearray)):
            return None
        if isinstance(value, str) and len(value) > 4000:
            return f"{value[:4000]}... [truncated]"
        return value

    sanitized = _sanitize(tool_result)
    if not isinstance(sanitized, dict):
        sanitized = {"result": sanitized}
    if dropped_image:
        sanitized["image_attached"] = True
    return sanitized


def _is_email_intent(text: str) -> bool:
    normalized = text.lower()
    has_email_word = any(token in normalized for token in ("邮件", "email", "e-mail", "mail"))
    has_send_word = any(token in normalized for token in ("发", "发送", "send"))
    return has_email_word and has_send_word


def _is_cancel_intent(text: str) -> bool:
    normalized = text.lower()
    return any(token in normalized for token in ("取消", "算了", "不用", "别发", "不要发", "cancel", "never mind"))


def _build_email_args_from_followup(text: str) -> dict[str, str]:
    email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
    target_email = email_match.group(0) if email_match else ""
    body = text.replace(target_email, "").strip(" ，,。.") if target_email else text.strip()
    subject = "测试邮件" if "测试" in body else "Reachy Mini 邮件"
    args = {"subject": subject, "body": body or text.strip()}
    if target_email:
        args["target_email"] = target_email
    return args


def _parse_realtime_frame(data: bytes) -> ArkRealtimeFrame:
    """Parse a Volcengine Realtime Dialogue server frame."""
    if not isinstance(data, bytes):
        raise TypeError("Volcengine realtime frames must be bytes")
    if len(data) < 8:
        raise ValueError("Volcengine realtime frame is too short")

    message_type = (data[1] >> 4) & 0xF
    serialization = (data[2] >> 4) & 0xF
    offset = 4

    if message_type not in {
        _MESSAGE_TYPE_FULL_SERVER_RESPONSE,
        _MESSAGE_TYPE_AUDIO_ONLY_RESPONSE,
        _MESSAGE_TYPE_ERROR_INFORMATION,
    }:
        raise ValueError(f"Unsupported Volcengine realtime message type: {message_type}")

    event = struct.unpack(">I", data[offset : offset + 4])[0]
    offset += 4
    session_id = ""
    if len(data) >= offset + 4:
        session_length = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        if session_length:
            session_id = data[offset : offset + session_length].decode("utf-8", errors="replace")
            offset += session_length

    payload = b""
    if len(data) >= offset + 4:
        payload_length = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        payload = data[offset : offset + payload_length] if payload_length else b""

    payload_value: Any = payload
    if payload and serialization == _SERIALIZATION_JSON:
        payload_value = json.loads(payload.decode("utf-8"))

    return ArkRealtimeFrame(
        message_type=message_type,
        serialization=serialization,
        event=event,
        session_id=session_id,
        payload=payload_value,
        raw_payload=payload,
    )


def _resolve_ark_voice(voice: str | None) -> str:
    voice_value = (voice or "").strip()
    voices_by_lower = {candidate.lower(): candidate for candidate in ARK_AVAILABLE_VOICES}
    return voices_by_lower.get(voice_value.lower(), ARK_REALTIME_DEFAULT_VOICE)


def _configured_input_sample_rate() -> int:
    return int(getattr(config, "ARK_REALTIME_INPUT_SAMPLE_RATE", ARK_INPUT_SAMPLE_RATE) or ARK_INPUT_SAMPLE_RATE)


def _configured_output_sample_rate() -> int:
    return int(getattr(config, "ARK_REALTIME_OUTPUT_SAMPLE_RATE", ARK_OUTPUT_SAMPLE_RATE) or ARK_OUTPUT_SAMPLE_RATE)


class ArkLiveHandler(ConversationHandler):
    """Conversation handler for Volcengine's Realtime Dialogue websocket API."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        """Initialize Volcengine websocket state and app queues."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_configured_output_sample_rate(),
            input_sample_rate=_configured_input_sample_rate(),
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self.input_sample_rate = _configured_input_sample_rate()
        self.output_sample_rate = _configured_output_sample_rate()
        self.output_queue: "asyncio.Queue[tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        self._voice_override: str | None = _resolve_ark_voice(startup_voice) if startup_voice else None
        self._session_id = str(uuid.uuid4())
        self._connect_id = str(uuid.uuid4())
        self._connection: Any = None
        self._run_task: asyncio.Task[None] | None = None
        self._connected_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._pending_assistant_chunks: list[str] = []
        self._last_partial_transcript = ""
        self._last_user_transcript = ""
        self._suppress_realtime_response = False
        self._sidecar_history: list[dict[str, str]] = []
        self._pending_tool_request: dict[str, Any] | None = None
        self._queued_tool_rag_results: list[tuple[str, dict[str, Any]]] = []
        self._queued_tool_drain_task: asyncio.Task[None] | None = None
        self.tool_manager = BackgroundToolManager()
        self.memory_context_provider = MemoryContextProvider(getattr(deps, "memory_store", None))
        self._memory_session_id: str | None = None
        self._memory_tasks: set[asyncio.Task[None]] = set()
        self._agent_message_log = AgentMessageListLog(logger)

    def copy(self) -> "ArkLiveHandler":
        """Create a copy of the handler."""
        return ArkLiveHandler(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    def get_current_voice(self) -> str:
        """Return the current Volcengine voice."""
        return _resolve_ark_voice(self._voice_override or get_session_voice(ARK_REALTIME_DEFAULT_VOICE))

    async def get_available_voices(self) -> list[str]:
        """Return curated Volcengine realtime voices."""
        return list(ARK_AVAILABLE_VOICES)

    async def change_voice(self, voice: str) -> str:
        """Change the voice and restart the session when connected."""
        self._voice_override = _resolve_ark_voice(voice)
        if self._connection is not None:
            await self._restart_session()
            return f"Voice changed to {self._voice_override} and restarted Volcengine session."
        return "Voice changed. Will take effect on next connection."

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile and restart when needed."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            _ = get_session_instructions()
            if self._connection is not None:
                await self._restart_session()
                return "Applied personality and restarted Volcengine session."
            return "Applied personality. Will take effect on next connection."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def start_up(self) -> None:
        """Open the Volcengine websocket and run the receive loop."""
        self._stop_event.clear()
        await self._start_memory_session()
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_session()
                return
            except ArkRealtimeAuthenticationError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Volcengine realtime session ended unexpectedly (attempt %d/%d): %s", attempt, max_attempts, exc
                )
                if attempt >= max_attempts or self._stop_event.is_set():
                    raise
                await asyncio.sleep(min(attempt, 3))
            finally:
                self._connection = None
                self._connected_event.clear()

    async def shutdown(self) -> None:
        """Close the Volcengine session and drain background persistence tasks."""
        self._stop_event.set()
        connection = self._connection
        if connection is not None:
            try:
                await connection.send(_build_full_client_payload(_EVENT_FINISH_SESSION, session_id=self._session_id))
                await connection.send(_build_full_client_payload(_EVENT_FINISH_CONNECTION))
            except Exception:
                logger.debug("Failed to send Volcengine finish frames.", exc_info=True)
            try:
                await connection.close()
            except Exception:
                pass
        await self._drain_memory_tasks()
        if self._queued_tool_drain_task is not None:
            self._queued_tool_drain_task.cancel()
            try:
                await self._queued_tool_drain_task
            except asyncio.CancelledError:
                pass
            self._queued_tool_drain_task = None
        await self.tool_manager.shutdown()
        await self._end_memory_session()

    def _headers(self) -> dict[str, str]:
        app_id = (getattr(config, "ARK_REALTIME_APP_ID", None) or "").strip()
        access_key = (getattr(config, "ARK_REALTIME_ACCESS_KEY", None) or "").strip()
        app_key = (getattr(config, "ARK_REALTIME_APP_KEY", None) or "").strip()
        resource_id = (getattr(config, "ARK_REALTIME_RESOURCE_ID", None) or "").strip()
        missing = [
            name
            for name, value in (
                ("ARK_REALTIME_APP_ID", app_id),
                ("ARK_REALTIME_ACCESS_KEY", access_key),
                ("ARK_REALTIME_APP_KEY", app_key),
                ("ARK_REALTIME_RESOURCE_ID", resource_id),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Volcengine realtime credential(s): {', '.join(missing)}")
        return {
            "X-Api-App-ID": app_id,
            "X-Api-Access-Key": access_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-App-Key": app_key,
            "X-Api-Connect-Id": self._connect_id,
        }

    def _session_config_payload(self) -> dict[str, Any]:
        instructions = self._with_memory_context(get_session_instructions())
        self._agent_message_log.reset(instructions)
        return {
            "dialog": {
                "bot_name": getattr(config, "ARK_REALTIME_BOT_NAME", "Reachy Mini"),
                "system_role": instructions,
                "speaking_style": "",
            },
            "tts": {
                "speaker": self.get_current_voice(),
                "audio_config": {
                    "channel": 1,
                    "format": "pcm_s16le",
                    "sample_rate": self.output_sample_rate,
                },
            },
            "asr": {
                "audio_info": {
                    "format": "pcm",
                    "sample_rate": self.input_sample_rate,
                    "channel": 1,
                },
                "extra": {
                    "enable_custom_vad": False,
                    "enable_asr_twopass": False,
                },
            },
        }

    async def _run_session(self) -> None:
        import websockets

        url = (getattr(config, "ARK_REALTIME_WS_URL", None) or "").strip()
        if not url:
            raise RuntimeError("ARK_REALTIME_WS_URL is required for Volcengine realtime")

        try:
            async with websockets.connect(
                url,
                additional_headers=self._headers(),
                open_timeout=10,
                close_timeout=2,
            ) as websocket:
                self._connection = websocket
                await websocket.send(_build_full_client_payload(_EVENT_START_CONNECTION))
                await websocket.send(
                    _build_full_client_payload(
                        _EVENT_START_SESSION,
                        session_id=self._session_id,
                        payload=self._session_config_payload(),
                    )
                )
                self._connected_event.set()
                logger.info("Volcengine realtime session started with voice=%r", self.get_current_voice())
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

                try:
                    async for message in websocket:
                        if self._stop_event.is_set():
                            break
                        if isinstance(message, str):
                            logger.debug("Ignoring unexpected Volcengine text websocket message: %s", message)
                            continue
                        await self._handle_frame(_parse_realtime_frame(message))
                finally:
                    await self.tool_manager.shutdown()
        except Exception as exc:
            if "HTTP 401" in str(exc) or "status_code=401" in str(exc):
                raise ArkRealtimeAuthenticationError(
                    "Volcengine Realtime rejected the websocket handshake with HTTP 401. "
                    "Check that ARK_REALTIME_APP_ID, ARK_REALTIME_ACCESS_KEY, ARK_REALTIME_APP_KEY, "
                    "and ARK_REALTIME_RESOURCE_ID belong to the same enabled Volcengine Speech application."
                ) from exc
            raise

    async def _restart_session(self) -> None:
        await self.shutdown()
        self._session_id = str(uuid.uuid4())
        self._connect_id = str(uuid.uuid4())
        self._stop_event.clear()
        self._run_task = asyncio.create_task(self.start_up(), name="ark-live-restart")
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Volcengine session restart timed out; continuing in background.")

    async def _handle_frame(self, frame: ArkRealtimeFrame) -> None:
        event = int(frame["event"])
        payload = frame.get("payload")
        if event in {_EVENT_CONNECTION_STARTED, _EVENT_SESSION_STARTED, _EVENT_USAGE}:
            logger.debug("Volcengine realtime event=%s payload=%s", event, payload)
            return
        if event in {_EVENT_CONNECTION_FAILED, _EVENT_SESSION_FAILED, _EVENT_DIALOG_COMMON_ERROR}:
            message = self._payload_error_message(payload)
            logger.error("Volcengine realtime error event=%s: %s", event, message)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {message}"}))
            return
        if event == _EVENT_ASR_RESPONSE:
            await self._handle_asr_response(payload)
            return
        if event == _EVENT_CHAT_RESPONSE:
            if self._suppress_realtime_response:
                return
            await self._handle_chat_response(payload)
            return
        if event == _EVENT_CHAT_ENDED:
            if self._suppress_realtime_response:
                self._suppress_realtime_response = False
                self._pending_assistant_chunks.clear()
                await self._drain_queued_tool_rag_results()
                return
            await self._flush_assistant_chunks()
            return
        if event == _EVENT_TTS_ENDED:
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.request_reset_after_current_audio()
            return
        if event == _EVENT_TTS_RESPONSE and frame.get("serialization") != _SERIALIZATION_RAW:
            logger.debug("Ignoring non-audio Volcengine TTSResponse payload: %s", payload)
            return
        if event == _EVENT_TTS_RESPONSE or frame["message_type"] == _MESSAGE_TYPE_AUDIO_ONLY_RESPONSE:
            if self._suppress_realtime_response:
                return
            await self._handle_audio_payload(frame.get("raw_payload", b""))
            return
        if event == _EVENT_ASR_ENDED:
            self.deps.movement_manager.set_listening(False)
            return
        if event in {_EVENT_CONNECTION_FINISHED}:
            return
        logger.debug("Unhandled Volcengine realtime event=%s payload=%s", event, payload)

    async def _handle_asr_response(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        results = payload.get("results")
        if not isinstance(results, list):
            return
        text = "".join(str(item.get("text") or "") for item in results if isinstance(item, dict)).strip()
        if not text:
            return
        is_interim = any(bool(item.get("is_interim")) for item in results if isinstance(item, dict))
        if is_interim:
            if text == self._last_partial_transcript:
                return
            self._last_partial_transcript = text
            await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": text}))
            return
        if text == self._last_user_transcript:
            return
        self._last_partial_transcript = ""
        self._last_user_transcript = text
        self.deps.movement_manager.set_listening(False)
        self._schedule_memory_message("user", text)
        await self._refresh_memory_context_for_user_transcript(text)
        self._agent_message_log.append("user", text)
        self._agent_message_log.log_once_for_turn("Volcengine model response")
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))
        await self._maybe_start_tool_calls(text)
        self._append_sidecar_history("user", text)

    async def _handle_chat_response(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        content = str(payload.get("content") or "")
        if content:
            self._pending_assistant_chunks.append(content)

    async def _flush_assistant_chunks(self) -> None:
        if not self._pending_assistant_chunks:
            return
        content = "".join(self._pending_assistant_chunks).strip()
        self._pending_assistant_chunks.clear()
        if not content:
            return
        self._schedule_memory_message("assistant", content)
        self._agent_message_log.append("assistant", content)
        self._agent_message_log.reset_turn_log()
        self._append_sidecar_history("assistant", content)
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": content}))

    async def _handle_audio_payload(self, payload: Any) -> None:
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            return
        payload_bytes = bytes(payload)
        if len(payload_bytes) % np.dtype(np.int16).itemsize != 0:
            logger.debug("Skipping malformed Volcengine audio payload with %d byte(s).", len(payload_bytes))
            return
        decoded_pcm = np.frombuffer(payload_bytes, dtype=np.int16).reshape(1, -1)
        if self.gradio_mode and self.deps.head_wobbler is not None:
            self.deps.head_wobbler.feed_pcm(decoded_pcm, self.output_sample_rate)
        await self.output_queue.put((self.output_sample_rate, decoded_pcm))

    async def _maybe_start_tool_calls(self, transcript: str) -> bool:
        """Use a sidecar OpenAI-compatible LLM to route Ark turns to local tools."""
        tool_specs = get_active_tool_specs(self.deps)
        chat_tools = _openai_tool_specs_to_chat_tools(tool_specs)
        if not chat_tools:
            return False
        if await self._maybe_start_pending_tool(transcript):
            return True

        base_url = (getattr(config, "PIPELINE_LLM_BASE_URL", None) or "").strip()
        model = (getattr(config, "PIPELINE_LLM_MODEL", None) or "").strip()
        api_key = (getattr(config, "PIPELINE_LLM_API_KEY", None) or "").strip()
        if not base_url or not model:
            logger.debug("Skipping Volcengine sidecar tools: PIPELINE_LLM_BASE_URL or PIPELINE_LLM_MODEL is missing.")
            self._maybe_set_pending_tool_request(transcript, chat_tools)
            return False

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a function-calling router for a small robot. "
                    "Call tools only when the user explicitly asks for robot actions, camera use, "
                    "web/current information, memory updates, email, or task control. "
                    "Use the recent conversation to resolve short follow-up utterances. "
                    "For email, call send_email only when subject and body can be inferred; "
                    "otherwise do not call a tool."
                ),
            },
            *self._sidecar_history[-_ARK_SIDECAR_HISTORY_MESSAGES:],
            {"role": "user", "content": transcript},
        ]
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key or "DUMMY", base_url=base_url)
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=chat_tools,
                    tool_choice="auto",
                    temperature=0,
                ),
                timeout=_ARK_TOOL_ROUTER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.info("Volcengine sidecar tool routing timed out; falling back to realtime model.")
            self._maybe_set_pending_tool_request(transcript, chat_tools)
            return False
        except Exception:
            logger.debug("Volcengine sidecar tool routing failed; falling back to realtime model.", exc_info=True)
            self._maybe_set_pending_tool_request(transcript, chat_tools)
            return False

        choices = getattr(response, "choices", None) or []
        if not choices:
            self._maybe_set_pending_tool_request(transcript, chat_tools)
            return False
        message = getattr(choices[0], "message", None)
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if not tool_calls:
            logger.info("Volcengine sidecar selected no tool for transcript=%r", transcript)
            self._maybe_set_pending_tool_request(transcript, chat_tools)
            return False

        started_any = False
        logger.info(
            "Volcengine sidecar selected tool(s): %s",
            [getattr(getattr(call, "function", None), "name", None) for call in tool_calls],
        )
        for call in tool_calls:
            function = getattr(call, "function", None)
            tool_name = getattr(function, "name", None)
            args_json_str = getattr(function, "arguments", None) or "{}"
            call_id = str(getattr(call, "id", None) or uuid.uuid4())
            if not isinstance(tool_name, str) or not tool_name:
                continue
            bg_tool = await self._start_local_tool(call_id=call_id, tool_name=tool_name, args_json_str=args_json_str)
            started_any = True
            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": f"Used tool {tool_name} with args {args_json_str}. Tool ID: {bg_tool.tool_id}",
                    }
                )
            )

        self._suppress_realtime_response = started_any
        if started_any:
            self._pending_tool_request = None
        return started_any

    async def _maybe_start_pending_tool(self, transcript: str) -> bool:
        if not self._pending_tool_request:
            return False
        if self._pending_tool_request.get("tool_name") != "send_email":
            return False
        if _is_cancel_intent(transcript):
            logger.info("Cleared pending send_email request after user cancellation.")
            self._pending_tool_request = None
            return False

        args = _build_email_args_from_followup(transcript)
        bg_tool = await self._start_local_tool(
            call_id=str(uuid.uuid4()),
            tool_name="send_email",
            args_json_str=json.dumps(args, ensure_ascii=False),
        )
        self._pending_tool_request = None
        self._suppress_realtime_response = True
        logger.info("Started pending send_email tool from follow-up transcript.")
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": f"Used tool send_email with args {json.dumps(args, ensure_ascii=False)}. Tool ID: {bg_tool.tool_id}",
                }
            )
        )
        return True

    def _maybe_set_pending_tool_request(self, transcript: str, chat_tools: list[dict[str, Any]]) -> None:
        available_tool_names = {
            tool.get("function", {}).get("name")
            for tool in chat_tools
            if isinstance(tool.get("function"), dict)
        }
        if "send_email" not in available_tool_names:
            return
        if _is_email_intent(transcript):
            self._pending_tool_request = {"tool_name": "send_email"}
            logger.info("Set pending send_email request; waiting for email content.")

    async def _start_local_tool(self, *, call_id: str, tool_name: str, args_json_str: str) -> Any:
        return await self.tool_manager.start_tool(
            call_id=call_id,
            tool_call_routine=ToolCallRoutine(
                tool_name=tool_name,
                args_json_str=args_json_str,
                deps=self.deps,
            ),
            is_idle_tool_call=False,
        )

    def _append_sidecar_history(self, role: str, content: str) -> None:
        self._sidecar_history.append({"role": role, "content": content})
        if len(self._sidecar_history) > _ARK_SIDECAR_HISTORY_MESSAGES:
            self._sidecar_history = self._sidecar_history[-_ARK_SIDECAR_HISTORY_MESSAGES:]

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Send completed local tool output back through Volcengine ChatRAGText."""
        if bg_tool.error is not None:
            tool_result: dict[str, Any] = {"error": bg_tool.error}
        else:
            tool_result = bg_tool.result or {"status": "ok"}
        sanitized_result = _sanitize_tool_result_for_model(bg_tool.tool_name, tool_result)
        self._agent_message_log.append("tool", sanitized_result, call_id=bg_tool.id, name=bg_tool.tool_name)
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": json.dumps(sanitized_result, ensure_ascii=False),
                    "metadata": {
                        "title": f"Used tool {bg_tool.tool_name}",
                        "status": "done",
                    },
                }
            )
        )
        await self._send_or_queue_tool_result_to_realtime(bg_tool.tool_name, sanitized_result)

    async def _send_or_queue_tool_result_to_realtime(self, tool_name: str, tool_result: dict[str, Any]) -> None:
        if self._suppress_realtime_response:
            self._queued_tool_rag_results.append((tool_name, tool_result))
            logger.info("Queued Volcengine ChatRAGText tool result until suppressed realtime turn ends.")
            self._schedule_queued_tool_drain()
            return
        await self._send_tool_result_to_realtime(tool_name, tool_result)

    async def _drain_queued_tool_rag_results(self) -> None:
        self._suppress_realtime_response = False
        current_task = asyncio.current_task()
        if self._queued_tool_drain_task is not None and self._queued_tool_drain_task is not current_task:
            self._queued_tool_drain_task.cancel()
            self._queued_tool_drain_task = None
        elif self._queued_tool_drain_task is current_task:
            self._queued_tool_drain_task = None
        while self._queued_tool_rag_results:
            tool_name, tool_result = self._queued_tool_rag_results.pop(0)
            await self._send_tool_result_to_realtime(tool_name, tool_result)

    def _schedule_queued_tool_drain(self) -> None:
        if self._queued_tool_drain_task is not None and not self._queued_tool_drain_task.done():
            return

        async def _delayed_drain() -> None:
            try:
                await asyncio.sleep(_ARK_SUPPRESSED_TURN_DRAIN_DELAY_SECONDS)
                if self._queued_tool_rag_results:
                    logger.info("Draining queued Volcengine ChatRAGText tool result after suppressed turn timeout.")
                    await self._drain_queued_tool_rag_results()
            except asyncio.CancelledError:
                raise

        self._queued_tool_drain_task = asyncio.create_task(_delayed_drain(), name="ark-tool-rag-drain")

    async def _send_tool_result_to_realtime(self, tool_name: str, tool_result: dict[str, Any]) -> None:
        """Ask Volcengine Realtime to summarize a local tool result in speech."""
        if self._connection is None:
            return
        external_rag = (
            "Local robot tool execution result. "
            "Answer the user in one short Chinese sentence, and do not claim the tool is unavailable.\n"
            f"tool_name={tool_name}\n"
            f"tool_result={json.dumps(tool_result, ensure_ascii=False)}"
        )
        try:
            await self._connection.send(
                _build_full_client_payload(
                    _EVENT_CHAT_RAG_TEXT,
                    session_id=self._session_id,
                    payload={"external_rag": external_rag},
                )
            )
        except Exception:
            logger.debug("Failed to send Volcengine ChatRAGText tool result.", exc_info=True)

    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        """Receive microphone audio and send it as a Volcengine TaskRequest."""
        if self._connection is None:
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
            self.deps.movement_manager.set_listening(True)
            await self._connection.send(_build_audio_payload(self._session_id, audio_frame.tobytes()))
        except Exception as exc:
            logger.debug("Dropping Volcengine audio frame: connection not ready (%s)", exc)

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit the next audio or transcript item."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    def _with_memory_context(self, instructions: str) -> str:
        if not config.MEMORY_CONTEXT_ENABLED or not self.memory_context_provider.enabled:
            return instructions
        try:
            return self.memory_context_provider.format_session_instructions(
                instructions,
                relevant_context="",
            )
        except Exception:
            logger.warning("Failed to build Volcengine memory context; continuing without it.", exc_info=True)
            return instructions

    async def _start_memory_session(self) -> None:
        if not self.memory_context_provider.enabled:
            return
        try:
            store = self.memory_context_provider.store
            if store is not None:
                self._memory_session_id = await asyncio.to_thread(
                    store.create_session,
                    backend=ARK_BACKEND,
                    profile=getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                )
        except Exception:
            logger.warning(
                "Failed to create Volcengine memory session; transcript persistence disabled.", exc_info=True
            )
            self._memory_session_id = None

    async def _end_memory_session(self) -> None:
        if not self._memory_session_id or not self.memory_context_provider.enabled:
            return
        try:
            store = self.memory_context_provider.store
            if store is not None:
                await asyncio.to_thread(store.end_session, self._memory_session_id)
        except Exception:
            logger.debug("Failed to end Volcengine memory session.", exc_info=True)
        finally:
            self._memory_session_id = None

    def _schedule_memory_message(self, role: str, content: str) -> None:
        if not self._memory_session_id or not self.memory_context_provider.enabled:
            return
        store = self.memory_context_provider.store
        if store is None:
            return
        session_id = self._memory_session_id

        async def _record() -> None:
            try:
                await asyncio.to_thread(store.add_message, session_id=session_id, role=role, content=content)
            except Exception:
                logger.debug("Failed to persist Volcengine %s transcript.", role, exc_info=True)

        task = asyncio.create_task(_record(), name=f"ark-memory-record-{role}")
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_tasks.discard)

    async def _refresh_memory_context_for_user_transcript(self, transcript: str) -> None:
        if not config.MEMORY_CONTEXT_ENABLED or not self.memory_context_provider.enabled:
            return
        try:
            latest_context = await asyncio.to_thread(
                self.memory_context_provider.search_relevant_context,
                transcript,
                exclude_session_id=self._memory_session_id,
            )
            self._agent_message_log.set_scoped_message("system", latest_context, scope="memory_context")
        except Exception:
            logger.debug("Failed to refresh Volcengine relevant memory context.", exc_info=True)

    async def _drain_memory_tasks(self) -> None:
        if not self._memory_tasks:
            return
        pending = list(self._memory_tasks)
        try:
            await asyncio.wait(pending, timeout=2.0)
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()

    @staticmethod
    def _payload_error_message(payload: Any) -> str:
        if isinstance(payload, dict):
            return str(payload.get("message") or payload.get("error") or payload)
        return str(payload)
