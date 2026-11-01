"""Small bounded async TTL cache with request coalescing."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, Hashable, Tuple, TypeVar

T = TypeVar("T")


class AsyncTTLCache:
    """Cache immutable results and share concurrent loads for the same key."""

    def __init__(self, maxsize: int = 2048) -> None:
        self.maxsize = max(1, maxsize)
        self._values: "OrderedDict[Hashable, Tuple[float, Any]]" = OrderedDict()
        self._inflight: Dict[Hashable, "asyncio.Task[Any]"] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self, key: Hashable, ttl_seconds: float, factory: Callable[[], Awaitable[T]]
    ) -> T:
        async with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                expires_at, value = cached
                if expires_at > time.monotonic():
                    self._values.move_to_end(key)
                    return value
                del self._values[key]

            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(self._load(key, ttl_seconds, factory))
                task.add_done_callback(self._consume_exception)
                self._inflight[key] = task

        # A caller timing out must not cancel a fetch that other callers share.
        return await asyncio.shield(task)

    async def _load(
        self, key: Hashable, ttl_seconds: float, factory: Callable[[], Awaitable[T]]
    ) -> T:
        try:
            value = await factory()
        except BaseException:
            async with self._lock:
                self._inflight.pop(key, None)
            raise

        async with self._lock:
            self._values[key] = (time.monotonic() + max(0.0, ttl_seconds), value)
            self._values.move_to_end(key)
            while len(self._values) > self.maxsize:
                self._values.popitem(last=False)
            self._inflight.pop(key, None)
        return value

    async def clear(self) -> None:
        async with self._lock:
            self._values.clear()

    @staticmethod
    def _consume_exception(task: "asyncio.Task[Any]") -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            pass


DEFAULT_CACHE = AsyncTTLCache(maxsize=512)
