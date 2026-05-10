from __future__ import annotations
import os
import re
import ssl
import asyncio
import logging
import smtplib
from typing import Any, Dict
from email.utils import parseaddr, formataddr
from email.message import EmailMessage

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _get_env_value(*names: str) -> str | None:
    """Return the first non-empty environment value for the given names."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean value for %s=%r; using default=%s", name, value, default)
    return default


def _parse_recipients(raw: Any) -> list[str]:
    """Parse and validate one or more email recipients."""
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, list):
        candidates = [str(part).strip() for part in raw]
    else:
        candidates = [str(raw).strip()]

    recipients: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        _, address = parseaddr(candidate)
        if not address or "@" not in address:
            raise ValueError(f"Invalid email address: {candidate}")
        recipients.append(address)
    return recipients


def _plain_text_from_html(html: str) -> str:
    """Create a conservative plain-text fallback for HTML email."""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)
    text = re.sub(r"(?s)<br\s*/?>", "\n", text)
    text = re.sub(r"(?s)</p\s*>", "\n\n", text)
    text = re.sub(r"(?s)<.*?>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() or "This email contains HTML content."


def _send_message(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    use_ssl: bool,
    use_tls: bool,
    message: EmailMessage,
    recipients: list[str],
) -> None:
    """Send the email through SMTP."""
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as smtp:
            smtp.login(username, password)
            smtp.send_message(message, to_addrs=recipients)
        return

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls(context=ssl.create_default_context())
        smtp.login(username, password)
        smtp.send_message(message, to_addrs=recipients)


class SendEmail(Tool):
    """Send an email through a configured SMTP account."""

    name = "send_email"
    description = (
        "Send an email through the configured SMTP sender account. "
        "Use this only when the user explicitly asks to send an email."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Email subject line.",
            },
            "body": {
                "type": "string",
                "description": "Plain-text body, or HTML body when is_html is true.",
            },
            "target_email": {
                "type": "string",
                "description": (
                    "Recipient email address. Optional; defaults to the default_target_email environment variable."
                ),
            },
            "is_html": {
                "type": "boolean",
                "description": "Whether body should be sent as HTML. Defaults to false.",
            },
        },
        "required": ["subject", "body"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Send an email."""
        subject = str(kwargs.get("subject") or "").strip()
        body = str(kwargs.get("body") or "").strip()
        if not subject:
            return {"error": "subject must be a non-empty string"}
        if not body:
            return {"error": "body must be a non-empty string"}

        try:
            recipients = _parse_recipients(kwargs.get("target_email") or _get_env_value("default_target_email"))
        except ValueError as e:
            return {"error": str(e)}
        if not recipients:
            return {"error": "target_email is required when default_target_email is not configured"}

        smtp_host = _get_env_value("SMTP_HOST") or "smtp.gmail.com"
        smtp_port_raw = _get_env_value("SMTP_PORT") or "587"
        try:
            smtp_port = int(smtp_port_raw)
        except ValueError:
            return {"error": f"SMTP_PORT must be an integer, got: {smtp_port_raw!r}"}

        smtp_username = _get_env_value("SMTP_USERNAME", "GMAIL_EMAIL", "GMAIL_ADDRESS")
        smtp_password = _get_env_value("SMTP_PASSWORD", "GMAIL_APP_PASSWORD", "EMAIL_APP_PASSWORD")
        if not smtp_username or not smtp_password:
            return {
                "error": (
                    "SMTP sender credentials are not configured. Set SMTP_USERNAME and SMTP_PASSWORD "
                    "or GMAIL_EMAIL and GMAIL_APP_PASSWORD."
                )
            }

        from_email = _get_env_value("SMTP_FROM_EMAIL") or smtp_username
        from_name = _get_env_value("SMTP_FROM_NAME")
        is_html = bool(kwargs.get("is_html", False))

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = formataddr((from_name or "", from_email)) if from_name else from_email
        message["To"] = ", ".join(recipients)
        if is_html:
            message.set_content(_plain_text_from_html(body))
            message.add_alternative(body, subtype="html")
        else:
            message.set_content(body)

        use_ssl = _env_flag("SMTP_USE_SSL", smtp_port == 465)
        use_tls = _env_flag("SMTP_USE_TLS", not use_ssl)

        logger.info("Tool call: send_email recipients=%s subject=%r html=%s", recipients, subject[:120], is_html)
        try:
            await asyncio.to_thread(
                _send_message,
                host=smtp_host,
                port=smtp_port,
                username=smtp_username,
                password=smtp_password,
                use_ssl=use_ssl,
                use_tls=use_tls,
                message=message,
                recipients=recipients,
            )
        except smtplib.SMTPAuthenticationError:
            logger.exception("SMTP authentication failed")
            return {"error": "SMTP authentication failed. For Gmail, use an app password instead of your login password."}
        except Exception as e:
            logger.exception("Failed to send email")
            return {"error": f"Failed to send email: {type(e).__name__}: {e}"}

        return {
            "status": "sent",
            "recipients": recipients,
            "subject": subject,
        }
