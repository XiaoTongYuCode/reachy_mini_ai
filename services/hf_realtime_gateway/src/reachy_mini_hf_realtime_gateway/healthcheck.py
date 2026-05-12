from __future__ import annotations

import json
import asyncio
from typing import Any
from dataclasses import dataclass

from reachy_mini_hf_realtime_gateway.config import GatewayConfig


@dataclass(frozen=True)
class HealthcheckResult:
    """Result of probing the realtime WebSocket endpoint."""

    ok: bool
    url: str
    message: str


SESSION_SUCCESS_EVENTS = frozenset({"session.updated", "session.created"})


def _event_type(payload: Any) -> str:
    if isinstance(payload, dict):
        value = payload.get("type")
        return value if isinstance(value, str) else ""
    return ""


async def check_realtime_endpoint(config: GatewayConfig, *, timeout_seconds: float = 5.0) -> HealthcheckResult:
    """Connect to `/v1/realtime` and send a minimal `session.update` event."""
    try:
        import websockets
    except ImportError as exc:
        return HealthcheckResult(False, config.realtime_url, f"websockets is not installed: {exc}")

    async def _probe() -> HealthcheckResult:
        async with websockets.connect(config.realtime_url) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": "Healthcheck session.",
                        },
                    }
                )
            )
            while True:
                raw_message = await websocket.recv()
                try:
                    payload = json.loads(raw_message) if isinstance(raw_message, str) else {}
                except json.JSONDecodeError:
                    continue

                event_type = _event_type(payload)
                if event_type in SESSION_SUCCESS_EVENTS:
                    return HealthcheckResult(True, config.realtime_url, f"{event_type} received")
                if event_type == "error":
                    error = payload.get("error") if isinstance(payload, dict) else None
                    if isinstance(error, dict):
                        message = error.get("message") or error.get("code") or error.get("type") or payload
                    else:
                        message = payload
                    return HealthcheckResult(False, config.realtime_url, f"server error: {message}")

    try:
        return await asyncio.wait_for(_probe(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return HealthcheckResult(False, config.realtime_url, "timed out waiting for session.updated")
    except Exception as exc:
        return HealthcheckResult(False, config.realtime_url, f"{type(exc).__name__}: {exc}")


def run_healthcheck(config: GatewayConfig, *, timeout_seconds: float = 5.0) -> HealthcheckResult:
    """Synchronous wrapper for CLI use."""
    return asyncio.run(check_realtime_endpoint(config, timeout_seconds=timeout_seconds))
