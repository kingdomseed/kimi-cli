from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Self

from loguru import logger

from kosong.chat_provider import (
    APIConnectionError,
    APIEmptyResponseError,
    APIStatusError,
    APITimeoutError,
    ChatProvider,
    StreamedMessage,
    StreamedMessagePart,
    ThinkingEffort,
    TokenUsage,
)
from kosong.message import Message
from kosong.tooling import Tool

_DEFAULT_FAILOVER_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _provider_base_url(provider: ChatProvider) -> str | None:
    client = getattr(provider, "client", None)
    base_url = getattr(client, "base_url", None)
    if base_url is not None:
        try:
            return str(base_url)
        except Exception:
            return None
    base_url = getattr(provider, "base_url", None)
    if base_url is not None:
        try:
            return str(base_url)
        except Exception:
            return None
    return None


def _should_failover(err: BaseException) -> bool:
    # Cancellation should propagate immediately.
    if isinstance(err, (asyncio.CancelledError, KeyboardInterrupt)):
        return False

    if isinstance(err, TimeoutError):
        return True
    if isinstance(err, (APITimeoutError, APIConnectionError, APIEmptyResponseError)):
        return True
    if isinstance(err, APIStatusError):
        return err.status_code in _DEFAULT_FAILOVER_STATUS_CODES
    return False


@dataclass(slots=True)
class _PrefetchedStream(StreamedMessage):
    _stream: StreamedMessage
    _first: StreamedMessagePart
    _rest: AsyncIterator[StreamedMessagePart]

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        async def _iter() -> AsyncIterator[StreamedMessagePart]:
            yield self._first
            async for part in self._rest:
                yield part

        return _iter()

    @property
    def id(self) -> str | None:  # pragma: no cover - delegated property
        return self._stream.id

    @property
    def usage(self) -> TokenUsage | None:  # pragma: no cover - delegated property
        return self._stream.usage


class FailoverChatProvider:
    """
    A wrapper chat provider that fails over to the next provider on retriable errors.

    Important behavior:
    - Failover only happens if the upstream stream fails *before* yielding any parts.
      Once output has started, a mid-stream failure is surfaced to the caller.
    - The last successful provider becomes the preferred provider for subsequent calls.
    """

    name = "failover"

    def __init__(
        self,
        providers: Sequence[ChatProvider],
        *,
        first_token_warn_seconds: float | None = None,
        first_token_timeout_seconds: float | None = None,
    ):
        providers = list(providers)
        if not providers:
            raise ValueError("providers must not be empty")
        self._providers: list[ChatProvider] = providers
        self._active_index: int = 0
        self._first_token_warn_seconds: float | None = first_token_warn_seconds
        self._first_token_timeout_seconds: float | None = first_token_timeout_seconds

    @property
    def model_name(self) -> str:
        return self._providers[self._active_index].model_name

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        return self._providers[self._active_index].thinking_effort

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> StreamedMessage:
        last_err: BaseException | None = None

        candidates = self._candidate_indexes()
        total = len(candidates)

        for attempt, idx in enumerate(candidates, start=1):
            provider = self._providers[idx]
            provider_url = _provider_base_url(provider)
            provider_label = f"{provider.name}/{provider.model_name}"
            if provider_url:
                provider_label = f"{provider_label} ({provider_url})"

            try:
                logger.info(
                    "Failover: attempt {attempt}/{total} using {provider}",
                    attempt=attempt,
                    total=total,
                    provider=provider_label,
                )
                result = await self._attempt_provider(
                    provider, provider_label, system_prompt, tools, history
                )
                self._active_index = idx
                logger.info(
                    "Failover: selected {provider} (active_index={idx})",
                    provider=provider_label,
                    idx=idx,
                )
                return result
            except BaseException as err:
                last_err = err
                if not _should_failover(err):
                    logger.warning(
                        "Failover: non-retriable error from {provider}: {err_type}",
                        provider=provider_label,
                        err_type=type(err).__name__,
                    )
                    raise
                status_code = err.status_code if isinstance(err, APIStatusError) else None
                logger.warning(
                    "Failover: retriable error from {provider}: {err_type}{status}. "
                    "Trying next provider.",
                    provider=provider_label,
                    err_type=type(err).__name__,
                    status=f" (status={status_code})" if status_code is not None else "",
                )
                continue

        assert last_err is not None
        logger.error(
            "Failover: all providers failed; last error: {err_type}",
            err_type=type(last_err).__name__,
        )
        raise last_err

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        new_self = copy.copy(self)
        new_self._providers = [p.with_thinking(effort) for p in self._providers]
        return new_self

    def _candidate_indexes(self) -> list[int]:
        # Prefer current active provider first, then try the others.
        idxs = list(range(len(self._providers)))
        if self._active_index in idxs:
            idxs.remove(self._active_index)
        return [self._active_index, *idxs]

    async def _attempt_provider(
        self,
        provider: ChatProvider,
        provider_label: str,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> _PrefetchedStream:
        """Try a single provider: generate stream + wait for first token.

        The timeout (if configured) covers the entire attempt — HTTP connection
        establishment, response headers, AND the first content token.  This ensures
        failover fires when the endpoint is truly unresponsive, not just when the
        model is slow to think.

        The warn timer fires periodically to give visibility into slow responses
        without cancelling the request.
        """
        warn_s = self._first_token_warn_seconds
        timeout_s = self._first_token_timeout_seconds

        async def _connect_and_get_first() -> _PrefetchedStream:
            stream = await provider.generate(
                system_prompt=system_prompt,
                tools=tools,
                history=history,
            )
            it = stream.__aiter__()
            try:
                first = await it.__anext__()
            except StopAsyncIteration as e:
                raise APIEmptyResponseError(
                    f"Stream ended before first token from {provider_label}"
                ) from e
            return _PrefetchedStream(stream, first, it)

        # Fast path: no warning/timeout configured.
        if warn_s is None and timeout_s is None:
            return await _connect_and_get_first()

        loop = asyncio.get_running_loop()
        start = loop.time()
        warned = 0

        # Single task covers the entire attempt: HTTP connect → first content token.
        attempt_task = asyncio.create_task(_connect_and_get_first())

        while True:
            elapsed = loop.time() - start

            if timeout_s is not None and elapsed >= timeout_s:
                if not attempt_task.done():
                    attempt_task.cancel()
                    with suppress(BaseException):
                        await attempt_task
                raise TimeoutError(
                    f"No response from {provider_label} after {elapsed:.0f}s"
                )

            # Calculate how long to wait before the next check.
            if warn_s is None:
                assert timeout_s is not None
                step_timeout = max(0.0, timeout_s - elapsed)
            else:
                step_timeout = warn_s
                if timeout_s is not None:
                    step_timeout = min(step_timeout, max(0.0, timeout_s - elapsed))

            done, _ = await asyncio.wait({attempt_task}, timeout=step_timeout)
            if done:
                # Task completed (success or exception).  Re-raise any exception.
                return attempt_task.result()

            # Not done yet — log a warning and keep waiting.
            warned += 1
            logger.warning(
                "Failover: waiting {elapsed:.0f}s for first token from {provider} "
                "(warn #{warned})",
                elapsed=loop.time() - start,
                provider=provider_label,
                warned=warned,
            )
