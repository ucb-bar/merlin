"""Bounded device jobs under explicit caller-owned execution capabilities."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, field
from pathlib import Path


def build_workers() -> int:
    """Use declared shared capacity, retaining one job for legacy callers by default."""
    from merlin.common.cpu_pool import grading_cpus, shared_compute_enabled

    shared = shared_compute_enabled()
    capacity = len(grading_cpus()) if shared else len(os.sched_getaffinity(0))
    raw = os.environ.get("MERLIN_DEVICE_BUILD_JOBS")
    if raw is None:
        return capacity if shared else 1
    if not raw.isdecimal() or int(raw) <= 0:
        raise ValueError("MERLIN_DEVICE_BUILD_JOBS must be a positive integer")
    return min(int(raw), capacity)


class EmissionJobs:
    """Overlap only an explicitly concurrency-capable package execution owner.

    Each job receives its own copied context and private output directory from
    the caller. The owner remains alive until every admitted job has drained.
    Legacy and unscoped calls retain caller-thread execution and order.
    """

    def __init__(self, workers):
        from merlin.targetgen.package_runtime import active_package_executor

        owner = active_package_executor()
        capability = getattr(owner, "concurrent_entrypoints_v1", None)
        self._scope = getattr(owner, "concurrent_entrypoint_scope_v1", None)
        self.workers = workers if callable(capability) and capability() is True else 1
        if self.workers > 1 and not callable(self._scope):
            raise ValueError("concurrent package execution requires its original worker scope")
        self._executor = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def map(self, items, function):
        if self._executor is None:
            for item in items:
                future = Future()
                try:
                    future.set_result(function(item))
                except Exception as error:  # Returned through the ordinary caller's refusal boundary.
                    future.set_exception(error)
                yield item, future
            return
        pending, iterator, exhausted = {}, iter(items), False
        while pending or not exhausted:
            while not exhausted and len(pending) < self.workers:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = self._executor.submit(copy_context().run, self._run, function, item)
                pending[future] = item
            if pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Stable admission order when several jobs complete together.
                for future in tuple(pending):
                    if future in completed:
                        item = pending.pop(future)
                        yield item, future

    def _run(self, function, item):
        with self._scope():
            return function(item)


@dataclass(frozen=True)
class ArtifactBuild:
    raw: Path | None = None
    custody: dict = field(default_factory=dict)
    refusal: str = ""


class ArtifactCompiler:
    """Share only byte-identical raw compilations, including jobs already in flight.

    The caller bounds submissions and drains admitted jobs on failure. No compiler
    package service or context is moved into a worker; callbacks run external tools.
    """

    def __init__(self, workers: int):
        self.workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self._artifacts: dict[str, list[tuple[Path, Future]]] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def submit(self, artifact: Path, digest: str, compile_artifact) -> Future:
        for source, future in self._artifacts.get(digest, ()):
            if future.done() and future.result().refusal:
                continue  # A diagnostic build may retry failed bytes for another symbol.
            if source.read_bytes() == artifact.read_bytes():
                return future
        if self._executor is None:
            future = Future()
            future.set_result(compile_artifact())
        else:
            future = self._executor.submit(compile_artifact)
        self._artifacts.setdefault(digest, []).append((artifact, future))
        return future

    def has_failed(self) -> bool:
        return any(future.done() and future.result().refusal for rows in self._artifacts.values() for _, future in rows)

    def completed_count(self) -> int:
        return sum(
            future.done() and not future.result().refusal for rows in self._artifacts.values() for _, future in rows
        )
