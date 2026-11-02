"""
taskgraph.py — A small task executor for the steps of one scan that do not depend on each other (plans/
PLAN_STREAMING_DAG.md §3): tasks with dependencies and a resource class, run on threads (the heavy work is in
subprocesses, OpenCV / numpy / zlib, which release the GIL, or in process pools of their own).

- tasks can be added while the graph runs (streaming: one task per photo or level as it arrives);
- a task starts when all its dependencies finished without error; a failed task skips only its dependents;
- among the ready tasks, the one with the longest known chain of work after it (its `cost` plus that of its
  heaviest dependent chain) starts first;
- per-resource limits (e.g. {"cpu_heavy": 1, "gpu": 1, "io": 4}) on top of the thread count;
- every task leaves a record (start, end, state, overlapping tasks) for stats.json.

Standard library only.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("TurntableStudio.TaskGraph")

PENDING, RUNNING, DONE, FAILED, SKIPPED = "pending", "running", "done", "failed", "skipped"
FINISHED = (DONE, FAILED, SKIPPED)


class _Task:
    __slots__ = ("name", "fn", "args", "kwargs", "deps", "resource", "cost", "state", "result", "error",
                 "start", "end", "dependents")

    def __init__(self, name: str, fn: Callable[..., Any], args: tuple, kwargs: dict, deps: Sequence[str],
                 resource: str, cost: float):
        self.name, self.fn, self.args, self.kwargs = name, fn, args, kwargs
        self.deps = list(deps)
        self.resource, self.cost = resource, float(cost)
        self.state = PENDING
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.dependents: List[str] = []


class TaskError(RuntimeError):
    """result() of a task that failed (the original exception is the __cause__) or was skipped."""


class TaskGraph:
    """
    graph = TaskGraph(max_workers=4, limits={"cpu_heavy": 1})
    graph.add("a", fn_a, cost=5)
    graph.add("b", fn_b, x, deps=["a"], resource="cpu_heavy")
    graph.wait()                       # every task added so far (and those they add) finished
    graph.result("b")                  # value, or TaskError
    """

    def __init__(self, max_workers: int = 4, limits: Optional[Dict[str, int]] = None, name: str = "graph"):
        self.name = name
        self.limits = dict(limits or {})
        self._ex = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix=f"tg-{name}")
        self._tasks: Dict[str, _Task] = {}
        self._cv = threading.Condition()
        self._t0 = time.time()
        self._closed = False

    # ------------------------------------------------------------------ building

    def add(self, name: str, fn: Callable[..., Any], *args: Any, deps: Iterable[str] = (), resource: str = "cpu",
            cost: float = 1.0, **kwargs: Any) -> str:
        """Adds a task (any time, also from inside a running task). Unknown dependencies must be added later."""
        with self._cv:
            if name in self._tasks:
                raise ValueError(f"task '{name}' already exists")
            if self._closed:
                raise RuntimeError("task graph is shut down")
            t = _Task(name, fn, args, kwargs, list(deps), resource, cost)
            self._tasks[name] = t
            for d in t.deps:
                if d in self._tasks:
                    self._tasks[d].dependents.append(name)
            for other in self._tasks.values():          # tasks added before that wait for this one
                if name in other.deps and other.name not in t.dependents:
                    t.dependents.append(other.name)
            self._dispatch()
        return name

    def has(self, name: str) -> bool:
        with self._cv:
            return name in self._tasks

    # ------------------------------------------------------------------ scheduling

    def _priority(self, name: str, memo: Dict[str, float]) -> float:
        if name in memo:
            return memo[name]
        memo[name] = 0.0                                 # cycle guard
        t = self._tasks[name]
        memo[name] = t.cost + max((self._priority(d, memo) for d in t.dependents if d in self._tasks), default=0.0)
        return memo[name]

    def _dispatch(self) -> None:
        """Starts the ready tasks the limits allow (caller holds the lock)."""
        changed = True
        while changed:                                   # skipping a task can make others skippable
            changed = False
            for t in self._tasks.values():
                if t.state == PENDING and any(d in self._tasks and self._tasks[d].state in (FAILED, SKIPPED)
                                              for d in t.deps):
                    t.state, t.end = SKIPPED, time.time()
                    bad = [d for d in t.deps if self._tasks.get(d) and self._tasks[d].state in (FAILED, SKIPPED)]
                    t.error = TaskError(f"task '{t.name}' skipped: dependency {bad[0]} {self._tasks[bad[0]].state}")
                    changed = True
        ready = [t for t in self._tasks.values() if t.state == PENDING
                 and all(d in self._tasks and self._tasks[d].state == DONE for d in t.deps)]
        if not ready:
            self._cv.notify_all()
            return
        memo: Dict[str, float] = {}
        ready.sort(key=lambda t: -self._priority(t.name, memo))
        running = {}
        for t in self._tasks.values():
            if t.state == RUNNING:
                running[t.resource] = running.get(t.resource, 0) + 1
        for t in ready:
            lim = self.limits.get(t.resource)
            if lim is not None and running.get(t.resource, 0) >= lim:
                continue
            running[t.resource] = running.get(t.resource, 0) + 1
            t.state, t.start = RUNNING, time.time()
            self._ex.submit(self._run, t)
        self._cv.notify_all()

    def _run(self, t: _Task) -> None:
        try:
            value, error = t.fn(*t.args, **t.kwargs), None
        except BaseException as e:                       # noqa: BLE001 - reported through result()
            value, error = None, e
            logger.warning(f"Task '{t.name}' failed: {type(e).__name__}: {e}")
        with self._cv:
            t.end = time.time()
            t.result, t.error = value, error
            t.state = DONE if error is None else FAILED
            self._dispatch()

    # ------------------------------------------------------------------ results

    def wait(self, names: Optional[Iterable[str]] = None, timeout: Optional[float] = None) -> bool:
        """Blocks until the given tasks (default: all, also those added meanwhile) finished; False on timeout."""
        deadline = None if timeout is None else time.time() + timeout
        wanted = None if names is None else list(names)
        with self._cv:
            while True:
                pool = self._tasks.values() if wanted is None else [self._tasks[n] for n in wanted if n in self._tasks]
                missing = [] if wanted is None else [n for n in wanted if n not in self._tasks]
                if not missing and all(t.state in FINISHED for t in pool):
                    return True
                if not any(t.state == RUNNING for t in self._tasks.values()) and missing == [] and all(
                        any(d not in self._tasks for d in t.deps) for t in pool if t.state == PENDING):
                    return False                         # only tasks waiting for dependencies never added
                left = None if deadline is None else deadline - time.time()
                if left is not None and left <= 0:
                    return False
                left = None if deadline is None else deadline - time.time()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(timeout=left if left is not None else 1.0)

    def state(self, name: str) -> Optional[str]:
        with self._cv:
            t = self._tasks.get(name)
            return t.state if t else None

    def result(self, name: str, timeout: Optional[float] = None) -> Any:
        """Value of the task (waits for it); TaskError when it failed or was skipped."""
        if not self.wait([name], timeout):
            raise TimeoutError(f"task '{name}' not finished after {timeout} s")
        t = self._tasks[name]
        if t.state == DONE:
            return t.result
        if t.state == FAILED:
            raise TaskError(f"task '{name}' failed: {t.error}") from t.error
        raise t.error if isinstance(t.error, TaskError) else TaskError(f"task '{name}' {t.state}")

    def records(self) -> List[Dict[str, Any]]:
        """Per task: state, start / end (s from the graph start), duration, the tasks it overlapped (stats.json)."""
        with self._cv:
            tasks = list(self._tasks.values())
        out = []
        for t in tasks:
            rec: Dict[str, Any] = {"name": t.name, "state": t.state, "resource": t.resource}
            if t.start is not None:
                end = t.end if t.end is not None else time.time()
                rec.update(start_s=round(t.start - self._t0, 2), end_s=round(end - self._t0, 2),
                           duration_s=round(end - t.start, 2))
                rec["overlaps_with"] = sorted(
                    o.name for o in tasks if o is not t and o.start is not None
                    and o.start < end and (o.end if o.end is not None else time.time()) > t.start)
            if t.error is not None:
                rec["error"] = f"{type(t.error).__name__}: {t.error}"[:200]
            out.append(rec)
        return out

    def shutdown(self, wait: bool = True) -> None:
        with self._cv:
            self._closed = True
        self._ex.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> "TaskGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)
