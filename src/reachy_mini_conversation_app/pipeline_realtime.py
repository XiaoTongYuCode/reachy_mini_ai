"""Local STT/TTS pipeline backend.

``BACKEND_PROVIDER=pipeline`` is a convenience backend for the cost-saving
architecture used by Reachy Mini:

- local STT/TTS run inside ``services/hf_realtime_gateway`` via
  ``speech-to-speech``;
- the LLM is an OpenAI-compatible endpoint configured with ``GATEWAY_LLM_*``;
- this app remains a realtime audio client and keeps using the shared
  OpenAI-compatible event handling from ``BaseRealtimeHandler``.

This intentionally does not call a remote TTS API from the app process.
"""

from __future__ import annotations
import os
import logging

from openai import AsyncOpenAI

from reachy_mini_conversation_app.config import (
    PIPELINE_BACKEND,
    config,
    build_hf_direct_ws_url,
    parse_hf_direct_target,
)
from reachy_mini_conversation_app.huggingface_realtime import (
    HuggingFaceRealtimeHandler,
    _build_openai_compatible_client_from_realtime_url,
)
from reachy_mini_conversation_app.hf_realtime_gateway_process import (
    HFRealtimeGatewayProcess,
    _probe_realtime_websocket,
)


logger = logging.getLogger(__name__)

PIPELINE_REALTIME_WS_URL_ENV = "PIPELINE_REALTIME_WS_URL"
PIPELINE_DEFAULT_GATEWAY_HOST = "127.0.0.1"
PIPELINE_DEFAULT_GATEWAY_PORT = 8765


def get_pipeline_realtime_ws_url() -> str:
    """Return the local realtime websocket URL for the pipeline backend."""
    configured_url = (
        os.getenv(PIPELINE_REALTIME_WS_URL_ENV)
        or getattr(config, "HF_REALTIME_WS_URL", None)
        or ""
    ).strip()
    if configured_url:
        return configured_url
    return build_hf_direct_ws_url(PIPELINE_DEFAULT_GATEWAY_HOST, PIPELINE_DEFAULT_GATEWAY_PORT)


class PipelineRealtimeHandler(HuggingFaceRealtimeHandler):
    """Realtime client for the local speech-to-speech STT/LLM/TTS gateway."""

    BACKEND_PROVIDER = PIPELINE_BACKEND
    SAMPLE_RATE = 16000
    REFRESH_CLIENT_ON_RECONNECT = True
    AUDIO_INPUT_COST_PER_1M = 0.0
    AUDIO_OUTPUT_COST_PER_1M = 0.0
    TEXT_INPUT_COST_PER_1M = 0.0
    TEXT_OUTPUT_COST_PER_1M = 0.0
    IMAGE_INPUT_COST_PER_1M = 0.0

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build an OpenAI-compatible realtime client pointed at the local pipeline gateway."""
        realtime_url = get_pipeline_realtime_ws_url()
        await self._ensure_pipeline_gateway_ready(realtime_url)
        bearer_token = (config.HF_TOKEN or "").strip()
        client, connect_query = _build_openai_compatible_client_from_realtime_url(
            realtime_url,
            bearer_token,
        )
        self._realtime_connect_query = connect_query
        logger.info("Using local STT/TTS pipeline gateway at %s", realtime_url)
        return client

    async def _ensure_pipeline_gateway_ready(self, realtime_url: str) -> None:
        """Use an existing local gateway when reachable, otherwise start the managed one."""
        try:
            await _probe_realtime_websocket(realtime_url)
            logger.info("Local STT/TTS pipeline gateway already reachable at %s", realtime_url)
            return
        except Exception:
            logger.info("Local STT/TTS pipeline gateway is not reachable yet; starting managed gateway.")

        await self._ensure_local_gateway_started(realtime_url)

    async def _ensure_local_gateway_started(self, direct_realtime_url: str) -> None:
        """Start the local speech-to-speech gateway with the requested host/port if needed."""
        if self._managed_gateway is None:
            host, port = parse_hf_direct_target(direct_realtime_url)
            env = dict(os.environ)
            if host:
                env.setdefault("GATEWAY_HOST", host)
            if port:
                env.setdefault("GATEWAY_PORT", str(port))
            self._managed_gateway = HFRealtimeGatewayProcess(env=env)
        await self._managed_gateway.ensure_started(direct_realtime_url)
