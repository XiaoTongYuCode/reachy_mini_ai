"""Volcengine Realtime Dialogue handler.

The public Realtime Dialogue API uses Volcengine's binary websocket protocol,
not the OpenAI Realtime JSON event protocol. This module keeps that transport
isolated while matching the app's existing ConversationHandler contract.
"""

from __future__ import annotations
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
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_observability import AgentMessageListLog
from reachy_mini_conversation_app.conversation_handler import ConversationHandler


logger = logging.getLogger(__name__)

ARK_INPUT_SAMPLE_RATE: Final[int] = 16000
ARK_OUTPUT_SAMPLE_RATE: Final[int] = 24000

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

                async for message in websocket:
                    if self._stop_event.is_set():
                        break
                    if isinstance(message, str):
                        logger.debug("Ignoring unexpected Volcengine text websocket message: %s", message)
                        continue
                    await self._handle_frame(_parse_realtime_frame(message))
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
            await self._handle_chat_response(payload)
            return
        if event == _EVENT_CHAT_ENDED:
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
