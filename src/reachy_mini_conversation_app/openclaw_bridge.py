"""WebSocket bridge to an OpenClaw gateway."""

from __future__ import annotations
import json
import uuid
import asyncio
import logging
from typing import Any, AsyncIterator
from dataclasses import dataclass

import websockets

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 3


@dataclass(frozen=True)
class OpenClawResponse:
    """Response returned by the OpenClaw gateway."""

    content: str
    error: str | None = None


class OpenClawBridge:
    """Minimal OpenClaw gateway client using the native WebSocket protocol."""

    def __init__(
        self,
        *,
        gateway_url: str | None = None,
        gateway_token: str | None = None,
        agent_id: str | None = None,
        session_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the bridge from explicit values or runtime config."""
        self.gateway_url = self._normalize_ws_url(gateway_url or config.OPENCLAW_GATEWAY_URL or "")
        self.gateway_token = gateway_token if gateway_token is not None else config.OPENCLAW_TOKEN
        self.agent_id = agent_id or config.OPENCLAW_AGENT_ID
        self.session_key = session_key or config.OPENCLAW_SESSION_KEY
        self.timeout = timeout if timeout is not None else config.OPENCLAW_TIMEOUT_SECONDS

        self._ws: Any | None = None
        self._connected = False
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._run_events: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._listener_task: asyncio.Task[None] | None = None

    @staticmethod
    def _normalize_ws_url(url: str) -> str:
        candidate = url.strip()
        if candidate.startswith("http://"):
            return "ws://" + candidate[7:]
        if candidate.startswith("https://"):
            return "wss://" + candidate[8:]
        if candidate and not candidate.startswith(("ws://", "wss://")):
            return "ws://" + candidate
        return candidate

    @property
    def is_connected(self) -> bool:
        """Return whether the gateway handshake completed."""
        return self._connected

    async def connect(self) -> bool:
        """Connect and authenticate to the OpenClaw gateway."""
        if not self.gateway_url:
            logger.warning("OpenClaw gateway URL is not configured")
            return False
        if self._connected and self._ws is not None:
            return True

        try:
            origin = self.gateway_url.replace("ws://", "http://").replace("wss://", "https://")
            self._ws = await websockets.connect(
                self.gateway_url,
                origin=origin,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=5,
            )
            raw = await asyncio.wait_for(self._ws.recv(), timeout=min(self.timeout, 10.0))
            challenge = json.loads(raw)
            if challenge.get("event") != "connect.challenge":
                logger.debug("Unexpected OpenClaw first frame: %s", challenge.get("event"))

            req_id = str(uuid.uuid4())
            await self._ws.send(
                json.dumps(
                    {
                        "type": "req",
                        "id": req_id,
                        "method": "connect",
                        "params": {
                            "minProtocol": PROTOCOL_VERSION,
                            "maxProtocol": PROTOCOL_VERSION,
                            "auth": {"token": self.gateway_token} if self.gateway_token else {},
                            "client": {
                                "id": "reachy-mini-conversation-app",
                                "version": "1.0.0",
                                "platform": "python",
                                "mode": "voice-tool",
                            },
                            "role": "operator",
                            "scopes": ["chat", "operator.read", "operator.write"],
                        },
                    },
                    ensure_ascii=False,
                )
            )

            raw = await asyncio.wait_for(self._ws.recv(), timeout=min(self.timeout, 10.0))
            hello = json.loads(raw)
            if not hello.get("ok"):
                error = hello.get("error", {})
                logger.warning("OpenClaw connect failed: %s", error.get("message") or error)
                await self.disconnect()
                return False

            self._connected = True
            self._listener_task = asyncio.create_task(self._listen_loop(), name="openclaw-listener")
            logger.info("Connected to OpenClaw gateway at %s", self.gateway_url)
            return True
        except Exception as e:
            logger.warning("Failed to connect to OpenClaw gateway: %s", e)
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Close the gateway connection and clear pending requests."""
        self._connected = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
        self._listener_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result({"ok": False, "error": {"code": "DISCONNECTED", "message": "Disconnected"}})
        self._pending.clear()
        self._run_events.clear()

    async def _listen_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("OpenClaw listener stopped: %s", e)
        finally:
            self._connected = False

    async def _dispatch(self, message: dict[str, Any]) -> None:
        if message.get("type") == "res":
            req_id = message.get("id")
            future = self._pending.pop(req_id, None)
            if future and not future.done():
                future.set_result(message)
            return

        if message.get("type") == "event":
            payload = message.get("payload") or {}
            run_id = payload.get("runId")
            if run_id:
                event_queue = self._run_events.setdefault(run_id, asyncio.Queue())
                await event_queue.put(message)

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self._ws or not self._connected:
            return {"ok": False, "error": {"code": "NOT_CONNECTED", "message": "Not connected to OpenClaw"}}

        req_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        try:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "req",
                        "id": req_id,
                        "method": method,
                        "params": params,
                    },
                    ensure_ascii=False,
                )
            )
            return await asyncio.wait_for(future, timeout=timeout or self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": {"code": "TIMEOUT", "message": "Request timed out"}}
        except Exception as e:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": {"code": "ERROR", "message": str(e)}}

    def _full_session_key(self) -> str:
        return f"agent:{self.agent_id}:{self.session_key}"

    async def chat(self, message: str, *, system_context: str | None = None) -> OpenClawResponse:
        """Send a chat message to OpenClaw and collect the final text response."""
        if not self._connected:
            return OpenClawResponse(content="", error="Not connected to OpenClaw")

        final_message = message
        if system_context:
            final_message = f"[System: {system_context}]\n\n{message}"

        response = await self._send_request(
            "chat.send",
            {
                "idempotencyKey": str(uuid.uuid4()),
                "sessionKey": self._full_session_key(),
                "message": final_message,
            },
            timeout=min(self.timeout, 30.0),
        )
        if not response.get("ok"):
            error = response.get("error", {})
            return OpenClawResponse(
                content="",
                error=f"{error.get('code', 'UNKNOWN')}: {error.get('message', 'Unknown error')}",
            )

        run_id = (response.get("payload") or {}).get("runId")
        if not run_id:
            return OpenClawResponse(content="", error="No runId in OpenClaw response")

        event_queue = self._run_events.setdefault(run_id, asyncio.Queue())
        try:
            full_text = ""
            async for event in self._iter_run_events(event_queue):
                event_name = event.get("event", "")
                payload = event.get("payload") or {}
                if event_name == "agent":
                    data = payload.get("data") or {}
                    stream = payload.get("stream")
                    if stream == "assistant":
                        full_text = str(data.get("text") or full_text)
                    elif stream == "lifecycle" and data.get("phase") == "end":
                        break
                elif event_name == "chat" and payload.get("state") == "final":
                    final_text = self._extract_final_chat_text(payload)
                    if final_text:
                        full_text = final_text
                    break
            if not full_text:
                return OpenClawResponse(content="", error="OpenClaw returned an empty response")
            return OpenClawResponse(content=full_text)
        finally:
            self._run_events.pop(run_id, None)

    async def _iter_run_events(self, queue: asyncio.Queue[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        while True:
            try:
                yield await asyncio.wait_for(queue.get(), timeout=self.timeout)
            except asyncio.TimeoutError:
                raise TimeoutError("Timed out waiting for OpenClaw response") from None

    @staticmethod
    def _extract_final_chat_text(payload: dict[str, Any]) -> str:
        message = payload.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts).strip()
        return ""
