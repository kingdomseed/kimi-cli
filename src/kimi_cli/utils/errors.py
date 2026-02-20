from __future__ import annotations

from urllib.parse import urlparse

from kosong.chat_provider import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ChatProviderError,
)

from kimi_cli.share import get_share_dir


def format_chat_provider_error(error: ChatProviderError) -> str:
    error_type = error.__class__.__name__
    message = str(error).strip()

    if not message:
        if isinstance(error, APITimeoutError):
            message = "Request timed out."
        elif isinstance(error, APIConnectionError):
            message = "Connection error."
        elif isinstance(error, APIStatusError):
            message = f"HTTP {error.status_code}."
        else:
            message = "Unknown provider error."

    if isinstance(error, APIStatusError) and str(error.status_code) not in message:
        message = f"HTTP {error.status_code}: {message}"

    return f"{error_type}: {message}"


def llm_log_path() -> str:
    return str(get_share_dir() / "logs" / "kimi.log")


def safe_base_url_host(base_url: str | None) -> str | None:
    if not base_url:
        return None
    try:
        host = urlparse(base_url).hostname
    except ValueError:
        host = None
    return host or None
