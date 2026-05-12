import time
import logging
from typing import Any

import httpx
from openai import AsyncOpenAI
from typing_extensions import Literal, TypedDict
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad

from reachy_mini_conversation_app.config import (
    HF_BACKEND,
    HF_LOCAL_CONNECTION_MODE,
    config,
    get_hf_direct_ws_url,
    parse_hf_realtime_url,
    get_hf_realtime_language,
    get_hf_connection_selection,
)
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import (
    BaseRealtimeHandler,
    InputTranscriptChunksByItem,
    to_realtime_tools_config,
)
from reachy_mini_conversation_app.tools.core_tools import get_active_tool_specs
from reachy_mini_conversation_app.hf_realtime_gateway_process import HFRealtimeGatewayProcess


logger = logging.getLogger(__name__)

_DUPLICATE_TRANSCRIPT_WINDOW_SECONDS = 5.0


def _build_openai_compatible_client_from_realtime_url(
    realtime_url: str,
    bearer_token: str | None,
) -> tuple[AsyncOpenAI, dict[str, str]]:
    """Build an OpenAI-compatible realtime client from a direct websocket/base URL."""
    parsed = parse_hf_realtime_url(realtime_url)
    client = AsyncOpenAI(
        api_key=bearer_token or "DUMMY",
        base_url=parsed.base_url,
        websocket_base_url=parsed.websocket_base_url,
    )
    return client, parsed.connect_query


class HFNativeRateAudioPCM(TypedDict):
    """Hugging Face extension for native-rate PCM audio."""

    type: Literal["audio/pcm"]
    rate: None


def _native_rate_audio_pcm() -> HFNativeRateAudioPCM:
    """Return the Hugging Face native-rate PCM config."""
    return {"type": "audio/pcm", "rate": None}


class HuggingFaceRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler for Hugging Face endpoints."""

    BACKEND_PROVIDER = HF_BACKEND
    SAMPLE_RATE = 16000
    REFRESH_CLIENT_ON_RECONNECT = True
    AUDIO_INPUT_COST_PER_1M = 0.0
    AUDIO_OUTPUT_COST_PER_1M = 0.0
    TEXT_INPUT_COST_PER_1M = 0.0
    TEXT_OUTPUT_COST_PER_1M = 0.0
    IMAGE_INPUT_COST_PER_1M = 0.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Hugging Face realtime handler."""
        super().__init__(*args, **kwargs)
        self._managed_gateway: HFRealtimeGatewayProcess | None = None
        self._last_completed_transcript: tuple[str, float] | None = None

    def _get_session_instructions(self) -> str:
        """Return Hugging Face session instructions."""
        return self._with_memory_context(get_session_instructions())

    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured Hugging Face session voice."""
        return get_session_voice(default)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for the current session dependencies."""
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> RealtimeSessionCreateRequestParam:
        """Return the Hugging Face OpenAI-compatible session config."""
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=self._get_session_instructions(),
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    # The OpenAI SDK type only includes 24 kHz PCM, but the HF
                    # compatible server uses rate=None for native 16 kHz mode.
                    format=_native_rate_audio_pcm(),  # type: ignore[typeddict-item]
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-transcribe",
                        language=get_hf_realtime_language(),
                    ),
                    turn_detection=ServerVad(type="server_vad", interrupt_response=True),
                ),
                output=RealtimeAudioConfigOutputParam(
                    format=_native_rate_audio_pcm(),  # type: ignore[typeddict-item]
                    voice=self.get_current_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    def _record_partial_transcript_delta(
        self,
        input_transcript: InputTranscriptChunksByItem,
        item_id: str,
        delta: str,
    ) -> None:
        """Record a Hugging Face partial transcript snapshot."""
        input_transcript.item_id = item_id
        input_transcript.deltas = [delta]

    def _should_ignore_completed_input_transcript(self, item_id: str | None, transcript: str) -> bool:
        """Suppress duplicate completed transcripts emitted by local speech-to-speech servers."""
        if super()._should_ignore_completed_input_transcript(item_id, transcript):
            return True

        now = time.perf_counter()
        last = self._last_completed_transcript
        if last is not None:
            last_transcript, last_seen_at = last
            if transcript == last_transcript and now - last_seen_at <= _DUPLICATE_TRANSCRIPT_WINDOW_SECONDS:
                logger.debug("Ignoring repeated Hugging Face input transcript within duplicate window: %s", transcript)
                return True

        self._last_completed_transcript = (transcript, now)
        return False

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the Hugging Face OpenAI-compatible realtime client."""
        bearer_token = (config.HF_TOKEN or "").strip()
        connection_selection = get_hf_connection_selection()
        direct_realtime_url = get_hf_direct_ws_url()
        if connection_selection.mode == HF_LOCAL_CONNECTION_MODE:
            if not direct_realtime_url:
                raise RuntimeError("HF_REALTIME_WS_URL must be set when HF_REALTIME_CONNECTION_MODE=local")
            await self.prepare_auto_start_gateway()
            client, connect_query = _build_openai_compatible_client_from_realtime_url(
                direct_realtime_url,
                bearer_token,
            )
            self._realtime_connect_query = connect_query
            logger.info("Using direct Hugging Face realtime endpoint %s", direct_realtime_url)
            return client

        session_url = connection_selection.session_url
        if not session_url:
            raise RuntimeError("Built-in Hugging Face session proxy URL is unavailable")
        if direct_realtime_url:
            logger.info("HF_REALTIME_CONNECTION_MODE=deployed; ignoring HF_REALTIME_WS_URL.")

        language = get_hf_realtime_language()
        if language not in {"", "auto", "en"}:
            logger.warning(
                "HF_REALTIME_LANGUAGE=%r was requested in deployed Hugging Face mode. "
                "The deployed server may ignore this language hint because STT is configured server-side; "
                "use HF_REALTIME_CONNECTION_MODE=local with GATEWAY_STT=faster-whisper and GATEWAY_LANGUAGE=%s "
                "for reliable fixed-language recognition.",
                language,
                language,
            )
        allocator_headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else None
        allocator_payload = {"language": language}
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(session_url, headers=allocator_headers, json=allocator_payload)
            response.raise_for_status()
            payload = response.json()

        connect_url = payload.get("connect_url")
        if not isinstance(connect_url, str) or not connect_url:
            raise RuntimeError(f"Session allocator response did not contain a valid connect_url: {payload!r}")

        parsed_connect_url = parse_hf_realtime_url(connect_url)
        if not parsed_connect_url.has_realtime_path:
            raise ValueError(f"Expected realtime connect URL ending with /realtime, got: {connect_url}")

        logger.info("Allocated realtime session %s", payload.get("session_id") or "<unknown>")
        client, connect_query = _build_openai_compatible_client_from_realtime_url(
            connect_url,
            bearer_token,
        )
        self._realtime_connect_query = connect_query
        return client

    async def prepare_auto_start_gateway(self) -> None:
        """Start the local gateway early when configured to manage it."""
        if not config.HF_REALTIME_AUTO_START:
            return

        connection_selection = get_hf_connection_selection()
        if connection_selection.mode != HF_LOCAL_CONNECTION_MODE:
            return

        direct_realtime_url = get_hf_direct_ws_url()
        if not direct_realtime_url:
            raise RuntimeError("HF_REALTIME_WS_URL must be set when HF_REALTIME_AUTO_START=true")

        await self._ensure_local_gateway_started(direct_realtime_url)

    async def _ensure_local_gateway_started(self, direct_realtime_url: str) -> None:
        """Start the managed local gateway when HF_REALTIME_AUTO_START is enabled."""
        if self._managed_gateway is None:
            self._managed_gateway = HFRealtimeGatewayProcess()
        await self._managed_gateway.ensure_started(direct_realtime_url)

    async def shutdown(self) -> None:
        """Shutdown the realtime connection and any managed local gateway."""
        try:
            await super().shutdown()
        finally:
            if self._managed_gateway is not None:
                await self._managed_gateway.stop()
