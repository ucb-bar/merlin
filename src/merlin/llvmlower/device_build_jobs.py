"""Bounded external device tool jobs; package emission remains on its owner thread."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
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
