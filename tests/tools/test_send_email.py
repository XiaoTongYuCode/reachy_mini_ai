"""Tests for the send_email tool."""

from __future__ import annotations
import smtplib
from typing import Any
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.send_email import SendEmail


class FakeSMTP:
    """Minimal SMTP fake used to capture sent messages."""

    instances: list["FakeSMTP"] = []

    def __init__(self, host: str, port: int, **kwargs: Any) -> None:
        """Capture SMTP connection arguments."""
        self.host = host
        self.port = port
        self.kwargs = kwargs
        self.logged_in_as: str | None = None
        self.started_tls = False
        self.sent_message = None
        self.to_addrs: list[str] | None = None
        FakeSMTP.instances.append(self)

    def __enter__(self) -> "FakeSMTP":
        """Return the fake SMTP context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the fake SMTP context manager."""
        return None

    def starttls(self, **kwargs: Any) -> None:
        """Record that STARTTLS was requested."""
        self.started_tls = True

    def login(self, username: str, password: str) -> None:
        """Record login credentials."""
        self.logged_in_as = username
        self.password = password

    def send_message(self, message: Any, to_addrs: list[str]) -> None:
        """Capture the sent message and recipient list."""
        self.sent_message = message
        self.to_addrs = to_addrs


@pytest.fixture(autouse=True)
def clear_fake_smtp() -> None:
    """Reset SMTP fake state before each test."""
    FakeSMTP.instances.clear()


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_send_email_uses_default_target_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should use default_target_email when target_email is omitted."""
    monkeypatch.setenv("SMTP_USERNAME", "sender@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("default_target_email", "default@example.com")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    result = await SendEmail()(_deps(), subject="Hello", body="Plain body")

    assert result == {
        "status": "sent",
        "recipients": ["default@example.com"],
        "subject": "Hello",
    }
    smtp = FakeSMTP.instances[-1]
    assert smtp.host == "smtp.gmail.com"
    assert smtp.port == 587
    assert smtp.started_tls is True
    assert smtp.logged_in_as == "sender@gmail.com"
    assert smtp.to_addrs == ["default@example.com"]
    assert smtp.sent_message["To"] == "default@example.com"
    assert smtp.sent_message.get_content().strip() == "Plain body"


@pytest.mark.asyncio
async def test_send_email_can_send_html_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should send HTML as a multipart alternative."""
    monkeypatch.setenv("GMAIL_EMAIL", "sender@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    result = await SendEmail()(
        _deps(),
        target_email="target@example.com",
        subject="HTML",
        body="<p>Hello <strong>there</strong></p>",
        is_html=True,
    )

    assert result["status"] == "sent"
    message = FakeSMTP.instances[-1].sent_message
    assert message.is_multipart()
    html_part = message.get_body(preferencelist=("html",))
    assert html_part is not None
    assert html_part.get_content().strip() == "<p>Hello <strong>there</strong></p>"
    assert "Hello there" in message.get_body(preferencelist=("plain",)).get_content()


@pytest.mark.asyncio
async def test_send_email_requires_recipient_when_default_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should require a recipient when no default is configured."""
    monkeypatch.setenv("SMTP_USERNAME", "sender@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.delenv("default_target_email", raising=False)

    result = await SendEmail()(_deps(), subject="Hello", body="Plain body")

    assert result == {"error": "target_email is required when default_target_email is not configured"}


@pytest.mark.asyncio
async def test_send_email_reports_authentication_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should return a user-facing error when SMTP auth fails."""
    class AuthFailSMTP(FakeSMTP):
        def login(self, username: str, password: str) -> None:
            """Simulate SMTP auth failure."""
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setenv("SMTP_USERNAME", "sender@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "bad-password")
    monkeypatch.setattr(smtplib, "SMTP", AuthFailSMTP)

    result = await SendEmail()(_deps(), target_email="target@example.com", subject="Hello", body="Plain body")

    assert result == {
        "error": "SMTP authentication failed. For Gmail, use an app password instead of your login password."
    }
