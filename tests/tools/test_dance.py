from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import dance as dance_mod
from reachy_mini_conversation_app.tools.dance import Dance
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_dance_unknown_move_falls_back_to_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid LLM-generated dance names should still queue a valid random move."""
    monkeypatch.setattr(dance_mod, "DANCE_AVAILABLE", True)
    monkeypatch.setattr(
        dance_mod,
        "AVAILABLE_MOVES",
        {
            "simple_nod": (None, None, {"description": "Simple nod"}),
            "side_to_side_sway": (None, None, {"description": "Sway"}),
        },
    )
    monkeypatch.setattr(dance_mod.random, "choice", lambda moves: "simple_nod")
    monkeypatch.setattr(dance_mod, "DanceQueueMove", lambda move_name: f"queued:{move_name}")
    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    result = await Dance()(deps, move="dance1", repeat=1)

    assert result == {"status": "queued", "move": "simple_nod", "repeat": 1}
    movement_manager.queue_move.assert_called_once_with("queued:simple_nod")
