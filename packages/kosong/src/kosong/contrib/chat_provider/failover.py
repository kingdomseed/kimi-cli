from __future__ import annotations

import copy
import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Self

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
from kosong.tooling import Tool
from kosong.message import Message


_DEFAULT_FAILOVER_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _should_failover(err: BaseException) -> bool:
    # Cancellation should propagate immediately.
    if isinstance(err, (asyncio.CancelledError, KeyboardInterrupt)):
        return False

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

    def __init__(self, providers: Sequence[ChatProvider]):
        providers = list(providers)
        if not providers:
            raise ValueError("providers must not be empty")
        self._providers: list[ChatProvider] = providers
        self._active_index: int = 0

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

        for idx in self._candidate_indexes():
            provider = self._providers[idx]
            try:
                stream = await provider.generate(system_prompt=system_prompt, tools=tools, history=history)
                it = stream.__aiter__()
                first = await it.__anext__()
                self._active_index = idx
                return _PrefetchedStream(stream, first, it)
            except BaseException as err:
                last_err = err
                if not _should_failover(err):
                    raise
                continue

        assert last_err is not None
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
