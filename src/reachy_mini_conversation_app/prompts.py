import os
import re
import sys
import json
import logging
from typing import Any
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config, get_default_voice_for_backend


logger = logging.getLogger(__name__)


PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"
BASE_INFO_PROMPT_FILENAME = "base_info.txt"
IPWHOIS_URL = "https://ipwho.is/"
IPWHOIS_TIMEOUT_SECONDS = 2.0
WEATHERAPI_CURRENT_URL = "https://api.weatherapi.com/v1/current.json"
WEATHERAPI_API_KEY_ENV = "WEATHERAPI_API_KEY"
WEATHERAPI_TIMEOUT_SECONDS = 2.0
_cached_location_payload: dict[str, Any] | None = None
_cached_current_address: str | None = None
_cached_current_weather: str | None = None


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r"^\[([a-zA-Z0-9/_-]+)\]$")

    lines = content.split("\n")
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return "\n".join(expanded_lines)


def _render_base_info_prompt() -> str:
    """Render the always-appended base information prompt block."""
    prompt_file = PROMPTS_LIBRARY_DIRECTORY / BASE_INFO_PROMPT_FILENAME
    try:
        template = prompt_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("Failed to read base info prompt '%s': %s", prompt_file, e)
        return ""

    base_info = get_current_base_info()
    return template.replace("{{current_time}}", base_info["current_time"]).replace(
        "{{address}}", base_info["address"]
    ).replace(
        "{{weather}}", base_info["weather"],
    )


def clear_base_info_cache() -> None:
    """Clear cached location and weather data."""
    global _cached_location_payload, _cached_current_address, _cached_current_weather
    _cached_location_payload = None
    _cached_current_address = None
    _cached_current_weather = None


def get_current_base_info(*, refresh: bool = False) -> dict[str, Any]:
    """Return current time, approximate address, and current weather."""
    if refresh:
        clear_base_info_cache()

    location_payload = _detect_current_location_payload()
    return {
        "current_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "address": _detect_current_address(),
        "weather": _detect_current_weather(),
        "location": {
            "city": location_payload.get("city"),
            "region": location_payload.get("region"),
            "country": location_payload.get("country"),
            "latitude": location_payload.get("latitude"),
            "longitude": location_payload.get("longitude"),
            "ip": location_payload.get("ip"),
        },
        "weather_api_configured": bool(os.getenv(WEATHERAPI_API_KEY_ENV)),
    }


def _detect_current_location_payload() -> dict[str, Any]:
    """Return an approximate location payload from the current public IP."""
    global _cached_location_payload
    if _cached_location_payload is not None:
        return _cached_location_payload

    try:
        with urlopen(IPWHOIS_URL, timeout=IPWHOIS_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to detect current address from %s: %s", IPWHOIS_URL, e)
        _cached_location_payload = {}
        return _cached_location_payload

    _cached_location_payload = payload if isinstance(payload, dict) else {}
    return _cached_location_payload


def _detect_current_address() -> str:
    """Return an approximate address from the current public IP."""
    global _cached_current_address
    if _cached_current_address is not None:
        return _cached_current_address

    _cached_current_address = _format_ipwhois_address(_detect_current_location_payload())
    return _cached_current_address


def _format_ipwhois_address(payload: dict[str, Any]) -> str:
    """Format an ipwho.is response into a concise location string."""
    if payload.get("success") is False:
        return "未配置"

    parts = []
    for key in ("city", "region", "country"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return ", ".join(dict.fromkeys(parts)) or "未配置"


def _detect_current_weather() -> str:
    """Return current weather from WeatherAPI.com for the detected location."""
    global _cached_current_weather
    if _cached_current_weather is not None:
        return _cached_current_weather

    api_key = os.getenv(WEATHERAPI_API_KEY_ENV)
    if not api_key:
        _cached_current_weather = "未配置"
        return _cached_current_weather

    query = _format_weatherapi_location_query(_detect_current_location_payload())
    if not query:
        _cached_current_weather = "未配置"
        return _cached_current_weather

    params = urlencode({"key": api_key, "q": query, "aqi": "no", "lang": "zh"})
    try:
        with urlopen(f"{WEATHERAPI_CURRENT_URL}?{params}", timeout=WEATHERAPI_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to detect current weather from WeatherAPI.com: %s", e)
        _cached_current_weather = "未配置"
        return _cached_current_weather

    _cached_current_weather = _format_weatherapi_current(payload)
    return _cached_current_weather


def _format_weatherapi_location_query(location_payload: dict[str, Any]) -> str:
    """Format an ipwho.is location payload into a WeatherAPI query."""
    latitude = location_payload.get("latitude")
    longitude = location_payload.get("longitude")
    if isinstance(latitude, int | float) and isinstance(longitude, int | float):
        return f"{latitude},{longitude}"
    address = _format_ipwhois_address(location_payload) if location_payload else ""
    return "" if address == "未配置" else address


def _format_weatherapi_current(payload: dict[str, Any]) -> str:
    """Format a WeatherAPI.com current weather response."""
    current = payload.get("current")
    if not isinstance(current, dict):
        return "未配置"

    parts = []
    condition = current.get("condition")
    if isinstance(condition, dict):
        text = condition.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    temp_c = current.get("temp_c")
    if isinstance(temp_c, int | float):
        parts.append(f"{temp_c:g}°C")

    humidity = current.get("humidity")
    if isinstance(humidity, int | float):
        parts.append(f"湿度 {humidity:g}%")

    wind_kph = current.get("wind_kph")
    if isinstance(wind_kph, int | float):
        parts.append(f"风速 {wind_kph:g} km/h")

    return "，".join(parts) or "未配置"


def _append_base_info_prompt(instructions: str) -> str:
    """Append the base information block to any resolved session instructions."""
    base_info = _render_base_info_prompt()
    if not base_info:
        return instructions.strip()
    return f"{instructions.strip()}\n\n{base_info}"


def get_session_instructions() -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.info(f"Loading default prompt from {PROMPTS_LIBRARY_DIRECTORY / 'default_prompt.txt'}")
        instructions_file = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    else:
        if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.info(
                "Loading prompt from external profile '%s' (root=%s)",
                profile,
                config.PROFILES_DIRECTORY,
            )
        else:
            logger.info(f"Loading prompt from profile '{profile}'")
        instructions_file = config.PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)
                return _append_base_info_prompt(expanded_instructions)
            logger.error(f"Profile '{profile}' has empty {INSTRUCTIONS_FILENAME}")
            sys.exit(1)
        logger.error(f"Profile {profile} has no {INSTRUCTIONS_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load instructions from profile '{profile}': {e}")
        sys.exit(1)


def get_session_voice(default: str | None = None) -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default or the active
    backend default voice.
    """
    fallback = get_default_voice_for_backend() if default is None else default
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return fallback
    try:
        voice_file = config.PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or fallback
    except Exception:
        pass
    return fallback
