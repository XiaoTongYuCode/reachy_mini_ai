import logging
from typing import Any, Dict

from reachy_mini_conversation_app.prompts import get_current_base_info
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CurrentLocationWeather(Tool):
    """Fetch the current approximate address and weather."""

    name = "current_location_weather"
    description = (
        "Get the latest approximate current address and weather. "
        "Use this when the user asks about current location, local weather, temperature, humidity, or wind."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "refresh": {
                "type": "boolean",
                "description": "Whether to bypass cached location and weather data. Defaults to true.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return current location and weather data."""
        refresh = kwargs.get("refresh")
        should_refresh = True if refresh is None else bool(refresh)

        logger.info("Tool call: current_location_weather refresh=%s", should_refresh)
        base_info = get_current_base_info(refresh=should_refresh)
        return {
            "current_time": base_info["current_time"],
            "address": base_info["address"],
            "weather": base_info["weather"],
            "location": base_info["location"],
            "weather_api_configured": base_info["weather_api_configured"],
        }
