"""Tests for the current location and weather tool."""

from __future__ import annotations
from typing import Any
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.current_location_weather as tool_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.current_location_weather import CurrentLocationWeather


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_current_location_weather_refreshes_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should fetch fresh location/weather data by default."""
    calls: list[bool] = []

    def fake_get_current_base_info(*, refresh: bool = False) -> dict[str, Any]:
        calls.append(refresh)
        return {
            "current_time": "2026-05-17T12:00:00+08:00",
            "address": "Shanghai, China",
            "weather": "晴，25°C，湿度 50%，风速 8 km/h",
            "location": {
                "city": "Shanghai",
                "region": "Shanghai",
                "country": "China",
                "latitude": 31.23,
                "longitude": 121.47,
                "ip": "203.0.113.1",
            },
            "weather_api_configured": True,
        }

    monkeypatch.setattr(tool_mod, "get_current_base_info", fake_get_current_base_info)

    result = await CurrentLocationWeather()(_deps())

    assert calls == [True]
    assert result["address"] == "Shanghai, China"
    assert result["weather"] == "晴，25°C，湿度 50%，风速 8 km/h"
    assert result["location"]["latitude"] == 31.23
    assert result["weather_api_configured"] is True


@pytest.mark.asyncio
async def test_current_location_weather_can_use_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should allow cached reads when explicitly requested."""
    calls: list[bool] = []

    def fake_get_current_base_info(*, refresh: bool = False) -> dict[str, Any]:
        calls.append(refresh)
        return {
            "current_time": "2026-05-17T12:00:00+08:00",
            "address": "未配置",
            "weather": "未配置",
            "location": {},
            "weather_api_configured": False,
        }

    monkeypatch.setattr(tool_mod, "get_current_base_info", fake_get_current_base_info)

    result = await CurrentLocationWeather()(_deps(), refresh=False)

    assert calls == [False]
    assert result["address"] == "未配置"
    assert result["weather"] == "未配置"
    assert result["weather_api_configured"] is False
