from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ManageMemory(Tool):
    """Create, update, forget, or search explicit long-term memories."""

    name = "manage_memory"
    description = (
        "Manage Reachy's explicit long-term memory. Use remember/update/forget only when the user clearly asks "
        "you to remember, change, or forget something. Use search when you need to check existing long-term memory."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "update", "forget", "search"],
                "description": "Memory operation to perform.",
            },
            "content": {
                "type": "string",
                "description": "Memory content for remember/update.",
            },
            "query": {
                "type": "string",
                "description": "Search or forget query.",
            },
            "memory_id": {
                "type": "string",
                "description": "Existing memory ID for update or forget.",
            },
            "kind": {
                "type": "string",
                "description": "Memory category such as fact, preference, identity, task, or note.",
            },
            "importance": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Importance from 1 to 5. Use 3 unless the user says it is important.",
            },
            "pinned": {
                "type": "boolean",
                "description": "Whether this memory should always be loaded into session instructions.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of search or forget results.",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute a memory operation against the local memory store."""
        memory_store = deps.memory_store
        if memory_store is None or not getattr(memory_store, "enabled", False):
            return {"error": "Persistent memory is disabled for this app instance."}

        action = str(kwargs.get("action") or "").strip().lower()
        if action == "remember":
            return await self._remember(memory_store, kwargs)
        if action == "update":
            return await self._update(memory_store, kwargs)
        if action == "forget":
            return await self._forget(memory_store, kwargs)
        if action == "search":
            return await self._search(memory_store, kwargs)
        return {"error": f"Unsupported memory action: {action or '<missing>'}"}

    async def _remember(self, memory_store: Any, kwargs: dict[str, Any]) -> Dict[str, Any]:
        content = _required_text(kwargs.get("content"), "content")
        memory = await asyncio.to_thread(
            memory_store.remember,
            content=content,
            kind=str(kwargs.get("kind") or "fact"),
            importance=int(kwargs.get("importance") or 3),
            pinned=bool(kwargs.get("pinned") or False),
            source="explicit",
        )
        logger.info("Stored explicit memory %s", memory.id)
        return {"status": "remembered", "memory": _memory_to_dict(memory)}

    async def _update(self, memory_store: Any, kwargs: dict[str, Any]) -> Dict[str, Any]:
        memory_id = _required_text(kwargs.get("memory_id"), "memory_id")
        content = _required_text(kwargs.get("content"), "content")
        memory = await asyncio.to_thread(
            memory_store.remember,
            memory_id=memory_id,
            content=content,
            kind=str(kwargs.get("kind") or "fact"),
            importance=int(kwargs.get("importance") or 3),
            pinned=bool(kwargs.get("pinned") or False),
            source="explicit",
        )
        logger.info("Updated explicit memory %s", memory.id)
        return {"status": "updated", "memory": _memory_to_dict(memory)}

    async def _forget(self, memory_store: Any, kwargs: dict[str, Any]) -> Dict[str, Any]:
        memory_id = str(kwargs.get("memory_id") or "").strip()
        if memory_id:
            forgotten = await asyncio.to_thread(memory_store.archive_memory, memory_id)
            return {
                "status": "forgotten" if forgotten else "not_found",
                "memory_id": memory_id,
            }

        query = _required_text(kwargs.get("query"), "query")
        limit = _limit(kwargs.get("limit"), default=5)
        memories = await asyncio.to_thread(memory_store.forget_by_query, query, limit=limit)
        return {
            "status": "forgotten",
            "count": len(memories),
            "memories": [_memory_to_dict(memory) for memory in memories],
        }

    async def _search(self, memory_store: Any, kwargs: dict[str, Any]) -> Dict[str, Any]:
        query = _required_text(kwargs.get("query"), "query")
        limit = _limit(kwargs.get("limit"), default=5)
        memories = await asyncio.to_thread(memory_store.search_long_term_memories, query, limit=limit)
        return {
            "status": "ok",
            "count": len(memories),
            "memories": [_memory_to_dict(memory) for memory in memories],
        }


def _required_text(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required.")
    return text


def _limit(value: object, *, default: int) -> int:
    try:
        if isinstance(value, (str, bytes, bytearray)):
            parsed = int(value)
        elif isinstance(value, int):
            parsed = value
        elif value is None:
            return default
        else:
            parsed = int(str(value))
        return min(20, max(1, parsed))
    except Exception:
        return default


def _memory_to_dict(memory: Any) -> dict[str, Any]:
    return {
        "id": memory.id,
        "content": memory.content,
        "kind": memory.kind,
        "importance": memory.importance,
        "pinned": memory.pinned,
        "source": memory.source,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "archived_at": memory.archived_at,
    }
