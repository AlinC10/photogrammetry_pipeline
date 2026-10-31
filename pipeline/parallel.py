"""
parallel.py — Process pool for the CPU work that runs beside the main steps (metrics, low-poly decimation, print
exports, export chains). Workers are spawned (the telemetry thread is running: fork() of a multi-threaded process can
deadlock) and their log records are forwarded to the main process, so they still reach run_log.txt.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("TurntableStudio.Parallel")


def _init_worker(queue, level: int) -> None:
    root = logging.getLogger()
    root.handlers[:] = [logging.handlers.QueueHandler(queue)]
    root.setLevel(level)


class _Forward(logging.Handler):
    """Hands a record from a worker to the logger of the same name in this process (its handlers write run_log.txt)."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


class LoggingProcessPool:
    """
    ProcessPoolExecutor (spawn) whose workers log into this process. Named jobs: submit(name, fn, ...) and later
    result(name) / results(); a failed job gives None (with a warning) instead of stopping the pipeline, unless
    result(name, raise_errors=True).
    """

    def __init__(self, workers: int, level: int = logging.INFO):
        ctx = multiprocessing.get_context("spawn")
        self._queue = ctx.Queue()
        self._listener = logging.handlers.QueueListener(self._queue, _Forward())
        self._listener.start()
        self._ex = ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=ctx,
                                       initializer=_init_worker, initargs=(self._queue, level))
        self._jobs: Dict[str, Future] = {}

    def submit(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        fut = self._ex.submit(fn, *args, **kwargs)
        self._jobs[name] = fut
        return fut

    def has(self, name: str) -> bool:
        return name in self._jobs

    def result(self, name: str, raise_errors: bool = False) -> Optional[Any]:
        fut = self._jobs.pop(name, None)
        if fut is None:
            return None
        try:
            return fut.result()
        except Exception as e:
            if raise_errors:
                raise
            logger.warning(f"Background job '{name}' failed: {e}")
            return None

    def results(self) -> Dict[str, Any]:
        """Waits for every job not collected yet."""
        return {name: self.result(name) for name in list(self._jobs)}

    def shutdown(self, cancel: bool = False) -> None:
        self._ex.shutdown(wait=True, cancel_futures=cancel)
        self._listener.stop()

    def __enter__(self) -> "LoggingProcessPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(cancel=exc_type is not None)
