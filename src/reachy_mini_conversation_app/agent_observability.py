"""Agent-runtime observability helpers."""

from __future__ import annotations
import json
import logging
from typing import Any


class AgentMessageListLog:
    """Track and print the model-visible message list for realtime agent turns."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the message-list log with the caller's logger."""
        self._logger = logger
        self._message_list: list[dict[str, Any]] = []
        self._logged_for_turn = False

    def reset(self, instructions: str) -> None:
        """Start a fresh model-visible message list for a new realtime session."""
        self._message_list = [{"role": "system", "content": instructions}]
        self._logged_for_turn = False

    def update_system_message(self, instructions: str) -> None:
        """Update the leading system message after live instruction refreshes."""
        if self._message_list and self._message_list[0].get("role") == "system":
            self._message_list[0]["content"] = instructions
            return
        self._message_list.insert(0, {"role": "system", "content": instructions})

    def append(self, role: str, content: Any, **extra: Any) -> None:
        """Record one model-visible message."""
        message: dict[str, Any] = {"role": role, "content": self._format_content(content)}
        message.update({key: value for key, value in extra.items() if value is not None})
        self._message_list.append(message)

    def set_scoped_message(self, role: str, content: Any, *, scope: str) -> None:
        """Replace one scoped message, or remove it when content is empty."""
        for index, message in enumerate(self._message_list):
            if message.get("scope") != scope:
                continue
            if content:
                self._message_list[index] = {
                    "role": role,
                    "content": self._format_content(content),
                    "scope": scope,
                }
            else:
                del self._message_list[index]
            return
        if content:
            self.append(role, content, scope=scope)

    def reset_turn_log(self) -> None:
        """Allow the next model-response boundary to print this turn again."""
        self._logged_for_turn = False

    def log_once_for_turn(self, trigger: str, response_kwargs: dict[str, Any] | None = None) -> None:
        """Print once until reset_turn_log is called."""
        if self._logged_for_turn:
            return
        self.log(trigger, response_kwargs=response_kwargs)
        self._logged_for_turn = True

    def log(self, trigger: str, response_kwargs: dict[str, Any] | None = None) -> None:
        """Print the reconstructed message_list for a realtime model turn."""
        message_list = [dict(message) for message in self._message_list]
        message_list.extend(self._response_messages(response_kwargs))
        self._logger.info(
            "LLM message_list (%s):\n%s",
            trigger,
            json.dumps(message_list, ensure_ascii=False, indent=2, default=str),
        )

    def _response_messages(self, response_kwargs: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not response_kwargs:
            return []

        response = response_kwargs.get("response")
        if isinstance(response, dict):
            instructions = response.get("instructions")
            tool_choice = response.get("tool_choice")
        else:
            instructions = getattr(response, "instructions", None)
            tool_choice = getattr(response, "tool_choice", None)

        messages: list[dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions, "scope": "response"})
        if tool_choice:
            messages.append({"role": "system", "content": f"tool_choice={tool_choice}", "scope": "response"})
        return messages

    @classmethod
    def _format_content(cls, content: Any) -> Any:
        if isinstance(content, list):
            return [cls._format_part(part) for part in content]
        return content

    @staticmethod
    def _format_part(part: Any) -> Any:
        if not isinstance(part, dict):
            return part

        if part.get("type") != "input_image":
            return part

        image_url = part.get("image_url")
        if isinstance(image_url, str) and image_url.startswith("data:image/"):
            return {
                "type": "input_image",
                "image_url": f"<inline image data url, {len(image_url)} chars>",
            }
        if image_url == "<inline image bytes>":
            return {
                "type": "input_image",
                "image_url": "<inline image bytes>",
            }
        return part
