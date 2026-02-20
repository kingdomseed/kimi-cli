from __future__ import annotations

from inline_snapshot import snapshot
from kosong.chat_provider import APITimeoutError

from kimi_cli.utils.errors import format_chat_provider_error


def test_format_chat_provider_error_timeout_with_empty_message():
    err = APITimeoutError("")
    assert format_chat_provider_error(err) == snapshot("APITimeoutError: Request timed out.")
