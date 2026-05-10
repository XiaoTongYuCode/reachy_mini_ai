from __future__ import annotations
import os
import logging
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

DEFAULT_WEB_SEARCH_API_URL = "https://open.feedcoopapi.com/search_api/web_search"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_COUNT = 10
MAX_WEB_RESULTS = 10
MAX_IMAGE_RESULTS = 5
MAX_TEXT_CHARS = 1200


def _env_value(*names: str) -> str | None:
    """Return the first non-empty environment value for the given names."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_float(name: str, default: float) -> float:
    """Parse a floating-point environment value."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default=%s", name, raw, default)
        return default
    return max(1.0, value)


def _bool_or_default(value: Any, default: bool) -> bool:
    """Parse a bool-like tool argument."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _int_arg(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    """Parse and clamp an integer tool argument."""
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _non_empty_string(value: Any) -> str | None:
    """Return a stripped non-empty string, if available."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate(value: Any, limit: int = MAX_TEXT_CHARS) -> str | None:
    """Trim long search fields before returning them to the model."""
    text = _non_empty_string(value)
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_request_payload(query: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build the Volcengine web search product API request body."""
    requested_type = _non_empty_string(kwargs.get("search_type")) or "web"
    search_type = requested_type if requested_type in {"web", "web_summary", "image"} else "web"
    max_count = MAX_IMAGE_RESULTS if search_type == "image" else 50
    payload: dict[str, Any] = {
        "Query": query,
        "SearchType": search_type,
        "Count": _int_arg(kwargs.get("count"), default=DEFAULT_COUNT, minimum=1, maximum=max_count),
    }

    filter_payload: dict[str, Any] = {}
    if search_type == "image":
        for arg_name, field_name in (
            ("image_width_min", "ImageWidthMin"),
            ("image_height_min", "ImageHeightMin"),
            ("image_width_max", "ImageWidthMax"),
            ("image_height_max", "ImageHeightMax"),
        ):
            value = kwargs.get(arg_name)
            if value is not None:
                filter_payload[field_name] = _int_arg(value, default=0, minimum=0, maximum=100000)
        image_shapes = kwargs.get("image_shapes")
        if isinstance(image_shapes, list) and image_shapes:
            filter_payload["ImageShapes"] = [str(shape) for shape in image_shapes if str(shape).strip()]
    else:
        filter_payload["NeedContent"] = _bool_or_default(kwargs.get("need_content"), False)
        filter_payload["NeedUrl"] = _bool_or_default(kwargs.get("need_url"), True)
        sites = _non_empty_string(kwargs.get("sites"))
        block_hosts = _non_empty_string(kwargs.get("block_hosts"))
        if sites:
            filter_payload["Sites"] = sites
        if block_hosts:
            filter_payload["BlockHosts"] = block_hosts
        auth_info_level = kwargs.get("auth_info_level")
        if auth_info_level is not None:
            filter_payload["AuthInfoLevel"] = _int_arg(auth_info_level, default=0, minimum=0, maximum=1)

        payload["NeedSummary"] = True if search_type == "web_summary" else _bool_or_default(
            kwargs.get("need_summary"),
            True,
        )
        time_range = _non_empty_string(kwargs.get("time_range"))
        if time_range:
            payload["TimeRange"] = time_range
        content_format = _non_empty_string(kwargs.get("content_format"))
        if content_format in {"text", "markdown"}:
            payload["ContentFormats"] = content_format
        industry = _non_empty_string(kwargs.get("industry"))
        if industry in {"finance", "game"}:
            payload["Industry"] = industry

    if filter_payload:
        payload["Filter"] = filter_payload

    query_rewrite = kwargs.get("query_rewrite")
    if query_rewrite is not None:
        payload["QueryControl"] = {"QueryRewrite": _bool_or_default(query_rewrite, False)}

    return payload


def _metadata_error(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract product API error metadata, if present."""
    metadata = payload.get("ResponseMetadata")
    if not isinstance(metadata, dict):
        return None
    error = metadata.get("Error")
    if not isinstance(error, dict):
        return None
    message = error.get("Message") or "Volcengine web search returned an error"
    result: dict[str, Any] = {"error": str(message)}
    code = error.get("Code") or error.get("CodeN")
    if code is not None:
        result["code"] = str(code)
    request_id = metadata.get("RequestId")
    if isinstance(request_id, str) and request_id:
        result["request_id"] = request_id
    return result


def _result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the nested Result object when the API wraps responses."""
    result = payload.get("Result")
    return result if isinstance(result, dict) else payload


def _choice_text(choice: dict[str, Any]) -> str | None:
    """Extract text from a web_summary choice."""
    for key in ("Message", "Delta"):
        value = choice.get(key)
        if isinstance(value, dict):
            content = value.get("Content") or value.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _extract_summary(result: dict[str, Any]) -> str | None:
    """Extract web_summary model output text from Choices."""
    fragments: list[str] = []
    for choice in result.get("Choices") or []:
        if isinstance(choice, dict):
            text = _choice_text(choice)
            if text:
                fragments.append(text)
    return "\n".join(fragments).strip() or None


def _normalize_web_result(item: dict[str, Any]) -> dict[str, Any]:
    """Return a compact web result item."""
    normalized: dict[str, Any] = {}
    field_map = {
        "Id": "id",
        "SortId": "sort_id",
        "Title": "title",
        "SiteName": "site_name",
        "Url": "url",
        "PublishTime": "publish_time",
        "RankScore": "rank_score",
        "AuthInfoDes": "auth_info_des",
        "AuthInfoLevel": "auth_info_level",
        "ContentFormats": "content_format",
    }
    for source, target in field_map.items():
        value = item.get(source)
        if value is not None and value != "":
            normalized[target] = value
    for source, target, limit in (
        ("Snippet", "snippet", 500),
        ("Summary", "summary", 1000),
        ("Content", "content", MAX_TEXT_CHARS),
    ):
        text = _truncate(item.get(source), limit)
        if text:
            normalized[target] = text
    return normalized


def _normalize_image_result(item: dict[str, Any]) -> dict[str, Any]:
    """Return a compact image result item."""
    normalized: dict[str, Any] = {}
    for source, target in (
        ("Id", "id"),
        ("SortId", "sort_id"),
        ("Title", "title"),
        ("SiteName", "site_name"),
        ("Url", "url"),
        ("PublishTime", "publish_time"),
        ("RankScore", "rank_score"),
    ):
        value = item.get(source)
        if value is not None and value != "":
            normalized[target] = value
    image = item.get("Image")
    if isinstance(image, dict):
        normalized["image"] = {
            key: value
            for source, key in (
                ("Url", "url"),
                ("Width", "width"),
                ("Height", "height"),
                ("Shape", "shape"),
                ("BlurDes", "blur_des"),
                ("Category", "category"),
                ("Watermark", "watermark"),
            )
            if (value := image.get(source)) is not None and value != ""
        }
    return normalized


def _extract_context(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize SearchContext metadata."""
    context = result.get("SearchContext")
    if not isinstance(context, dict):
        return {}
    normalized: dict[str, Any] = {}
    for source, target in (("OriginQuery", "origin_query"), ("SearchType", "search_type")):
        value = context.get(source)
        if isinstance(value, str) and value:
            normalized[target] = value
    return normalized


def _build_answer(summary: str | None, web_results: list[dict[str, Any]], image_results: list[dict[str, Any]]) -> str:
    """Build a readable answer for the caller from structured search results."""
    if summary:
        return summary
    lines: list[str] = []
    for item in web_results[:5]:
        title = item.get("title") or "Untitled"
        body = item.get("summary") or item.get("snippet") or item.get("content") or ""
        url = item.get("url") or ""
        lines.append(f"{title}\n{body}\n{url}".strip())
    for item in image_results[:5]:
        title = item.get("title") or "Image result"
        image = item.get("image") if isinstance(item.get("image"), dict) else {}
        url = image.get("url") if isinstance(image, dict) else ""
        lines.append(f"{title}\n{url}".strip())
    return "\n\n".join(line for line in lines if line).strip()


class WebSearch(Tool):
    """Search through the Volcengine Web Search product API."""

    name = "web_search"
    description = (
        "Search current public web information through Volcengine Web Search product API. "
        "Use this for recent news, weather, prices, products, people, organizations, images, or other current facts."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Keep it concise; the API supports 1 to 100 characters.",
            },
            "search_type": {
                "type": "string",
                "enum": ["web", "web_summary", "image"],
                "description": "Search type. Defaults to web. Use web_summary when a one-paragraph summary is useful.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Number of results. web/web_summary max 50; image max 5. Defaults to 10.",
            },
            "need_summary": {
                "type": "boolean",
                "description": "Whether to request precise result summaries for web search. Defaults to true.",
            },
            "need_content": {
                "type": "boolean",
                "description": "Whether to return only results with page content. Defaults to false.",
            },
            "need_url": {
                "type": "boolean",
                "description": "Whether to return only results with original URLs. Defaults to true.",
            },
            "time_range": {
                "type": "string",
                "description": "Optional publish time filter: OneDay, OneWeek, OneMonth, OneYear, or YYYY-MM-DD..YYYY-MM-DD.",
            },
            "sites": {
                "type": "string",
                "description": "Optional site scope, separated by |, for example aliyun.com|mp.qq.com.",
            },
            "block_hosts": {
                "type": "string",
                "description": "Optional blocked hosts, separated by |.",
            },
            "auth_info_level": {
                "type": "integer",
                "enum": [0, 1],
                "description": "0 means no authority filter; 1 restricts to highly authoritative content.",
            },
            "query_rewrite": {
                "type": "boolean",
                "description": "Whether to enable query rewriting. It can improve recall but may increase latency.",
            },
            "content_format": {
                "type": "string",
                "enum": ["text", "markdown"],
                "description": "Returned content format for web results. Defaults to text.",
            },
            "industry": {
                "type": "string",
                "enum": ["finance", "game"],
                "description": "Optional industry-specific search type.",
            },
            "image_shapes": {
                "type": "array",
                "items": {"type": "string", "enum": ["横长方形", "竖长方形", "方形"]},
                "description": "Optional image shape filters for image search.",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run Volcengine Web Search product API and return structured results."""
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return {"error": "query must be a non-empty string"}

        api_key = _env_value("VOLCENGINE_WEB_SEARCH_API_KEY", "SEARCH_INFINITY_API_KEY")
        if not api_key:
            return {"error": "VOLCENGINE_WEB_SEARCH_API_KEY is required for the web_search tool"}

        url = _env_value("VOLCENGINE_WEB_SEARCH_API_URL") or DEFAULT_WEB_SEARCH_API_URL
        timeout = _env_float("VOLCENGINE_WEB_SEARCH_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        request_payload = _build_request_payload(query, kwargs)

        logger.info(
            "Tool call: web_search query=%r search_type=%s count=%s url=%s",
            query[:200],
            request_payload.get("SearchType"),
            request_payload.get("Count"),
            url,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as e:
            logger.exception("Volcengine Web Search HTTP error")
            body = e.response.text[:500] if e.response is not None else ""
            return {"error": f"web_search HTTP {e.response.status_code}: {body}"}
        except Exception as e:
            logger.exception("Volcengine Web Search failed")
            return {"error": f"web_search failed: {type(e).__name__}: {e}"}

        if not isinstance(payload, dict):
            return {"error": "web_search returned a non-object response"}

        metadata_error = _metadata_error(payload)
        if metadata_error is not None:
            return metadata_error

        result = _result_payload(payload)
        web_results = [
            _normalize_web_result(item)
            for item in result.get("WebResults") or []
            if isinstance(item, dict)
        ][:MAX_WEB_RESULTS]
        image_results = [
            _normalize_image_result(item)
            for item in result.get("ImageResults") or []
            if isinstance(item, dict)
        ][:MAX_IMAGE_RESULTS]
        summary = _extract_summary(result)
        answer = _build_answer(summary, web_results, image_results)

        return {
            "status": "ok",
            "query": query,
            "search_type": request_payload["SearchType"],
            "answer": answer,
            "summary": summary,
            "web_results": web_results,
            "image_results": image_results,
            "result_count": result.get("ResultCount"),
            "search_context": _extract_context(result),
            "time_cost_ms": result.get("TimeCost"),
            "log_id": result.get("LogId"),
            "usage": result.get("Usage"),
        }
