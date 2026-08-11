"""Async↔sync bridging for Phase 15 parallel retrieval.

The public retrieval/hybrid services stay synchronous (sync FastAPI endpoints,
Celery tasks, existing tests) while the pipeline overlaps I/O with asyncio.
`run_coroutine` runs a coroutine on a fresh event loop: inline via
`asyncio.run` when the caller has no running loop, otherwise on a dedicated
thread with its own loop so it never nests inside the caller's loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_coroutine(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine from synchronous code, propagating its result or error."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    result: dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # re-raised in the caller thread below
            result["error"] = exc

    thread = threading.Thread(target=_target, name="sync-async-bridge", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]
