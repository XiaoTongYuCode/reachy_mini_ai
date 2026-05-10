"""Tests for the Volcengine Web Search product API tool."""

from __future__ import annotations
from typing import Any
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.web_search as web_search_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.web_search import WebSearch


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


class FakeHTTPResponse:
    """Minimal httpx-like response object."""

    text = "{}"
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store response payload."""
        self._payload = payload

    def raise_for_status(self) -> None:
        """Simulate a successful response."""
        return None

    def json(self) -> dict[str, Any]:
        """Return the JSON payload."""
        return self._payload


class FakeAsyncClient:
    """Capture httpx.AsyncClient requests."""

    instances: list["FakeAsyncClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        """Capture client constructor kwargs."""
        self.kwargs = kwargs
        self.post_kwargs: dict[str, Any] | None = None
        self.post_url: str | None = None
        self.instances.append(self)

    async def __aenter__(self) -> "FakeAsyncClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        """Exit async context manager."""
        return False

    async def post(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
        """Capture request and return a web search response."""
        self.post_url = url
        self.post_kwargs = kwargs
        return FakeHTTPResponse(
            {
                "ResponseMetadata": {
                    "RequestId": "req-1",
                    "Action": "WebSearch",
                    "Version": "2025-01-01",
                },
                "Result": {
                    "ResultCount": 1,
                    "WebResults": [
                        {
                            "Id": "result-1",
                            "SortId": 1,
                            "Title": "新闻来源",
                            "SiteName": "示例站点",
                            "Url": "https://example.com/news",
                            "Snippet": "普通摘要",
                            "Summary": "精准摘要",
                            "PublishTime": "2026-05-10T12:00:00+08:00",
                            "RankScore": 0.9,
                            "AuthInfoDes": "正常权威",
                            "AuthInfoLevel": 2,
                        }
                    ],
                    "SearchContext": {"OriginQuery": "今天有什么热点新闻", "SearchType": "web"},
                    "TimeCost": 123,
                    "LogId": "log-1",
                },
            }
        )


@pytest.fixture(autouse=True)
def clear_fake_clients() -> None:
    """Reset fake clients before each test."""
    FakeAsyncClient.instances.clear()


@pytest.mark.asyncio
async def test_web_search_calls_volcengine_product_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should call the Web Search product API endpoint with APIKey auth."""
    monkeypatch.setenv("VOLCENGINE_WEB_SEARCH_API_KEY", "web-search-key")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", FakeAsyncClient)

    result = await WebSearch()(
        _deps(),
        query="今天有什么热点新闻",
        count=3,
        time_range="OneDay",
        query_rewrite=True,
    )

    assert result["status"] == "ok"
    assert result["answer"] == "新闻来源\n精准摘要\nhttps://example.com/news"
    assert result["web_results"][0]["url"] == "https://example.com/news"
    assert result["search_context"] == {"origin_query": "今天有什么热点新闻", "search_type": "web"}
    assert result["log_id"] == "log-1"

    client = FakeAsyncClient.instances[-1]
    assert client.kwargs["timeout"] == 30.0
    assert client.post_url == "https://open.feedcoopapi.com/search_api/web_search"
    assert client.post_kwargs == {
        "headers": {
            "Authorization": "Bearer web-search-key",
            "Content-Type": "application/json",
        },
        "json": {
            "Query": "今天有什么热点新闻",
            "SearchType": "web",
            "Count": 3,
            "Filter": {
                "NeedContent": False,
                "NeedUrl": True,
            },
            "NeedSummary": True,
            "TimeRange": "OneDay",
            "QueryControl": {"QueryRewrite": True},
        },
    }


@pytest.mark.asyncio
async def test_web_summary_forces_need_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """web_summary requests must set NeedSummary=true."""
    monkeypatch.setenv("VOLCENGINE_WEB_SEARCH_API_KEY", "web-search-key")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", FakeAsyncClient)

    result = await WebSearch()(_deps(), query="北京游玩攻略", search_type="web_summary", need_summary=False)

    assert result["status"] == "ok"
    client = FakeAsyncClient.instances[-1]
    assert client.post_kwargs is not None
    assert client.post_kwargs["json"]["SearchType"] == "web_summary"
    assert client.post_kwargs["json"]["NeedSummary"] is True


@pytest.mark.asyncio
async def test_image_search_clamps_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image search should call SearchType=image and clamp count to the product max."""
    monkeypatch.setenv("VOLCENGINE_WEB_SEARCH_API_KEY", "web-search-key")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", FakeAsyncClient)

    result = await WebSearch()(_deps(), query="郭德纲", search_type="image", count=50)

    assert result["status"] == "ok"
    client = FakeAsyncClient.instances[-1]
    assert client.post_kwargs is not None
    assert client.post_kwargs["json"] == {
        "Query": "郭德纲",
        "SearchType": "image",
        "Count": 5,
    }


@pytest.mark.asyncio
async def test_web_search_reports_product_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should return product API errors from ResponseMetadata."""
    class ErrorClient(FakeAsyncClient):
        async def post(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
            """Return a product error payload."""
            self.post_url = url
            self.post_kwargs = kwargs
            return FakeHTTPResponse(
                {
                    "ResponseMetadata": {
                        "RequestId": "req-error",
                        "Error": {
                            "Code": "10406",
                            "Message": "free quota exhausted",
                        },
                    },
                    "Result": None,
                }
            )

    monkeypatch.setenv("VOLCENGINE_WEB_SEARCH_API_KEY", "web-search-key")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", ErrorClient)

    result = await WebSearch()(_deps(), query="北京天气")

    assert result == {
        "error": "free quota exhausted",
        "code": "10406",
        "request_id": "req-error",
    }


@pytest.mark.asyncio
async def test_web_search_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should report a clear configuration error when the key is missing."""
    monkeypatch.delenv("VOLCENGINE_WEB_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_INFINITY_API_KEY", raising=False)

    result = await WebSearch()(_deps(), query="北京天气")

    assert result == {"error": "VOLCENGINE_WEB_SEARCH_API_KEY is required for the web_search tool"}


@pytest.mark.asyncio
async def test_web_search_requires_query() -> None:
    """The tool should reject empty queries before calling the network."""
    result = await WebSearch()(_deps(), query=" ")

    assert result == {"error": "query must be a non-empty string"}
