"""Bounded independent capture batches; dependent model writes retain declaration order."""

from __future__ import annotations

import contextvars
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import groupby
from pathlib import Path


def _available_memory(path: Path = Path("/proc/meminfo")) -> int:
    """Prefer reclaimable host memory; retain conservative free-page fallback."""
    try:
        rows = [
            line.partition(":")[2].split()
            for line in path.read_text().splitlines()
            if line.partition(":")[0] == "MemAvailable"
        ]
        if len(rows) == 1 and len(rows[0]) == 2 and rows[0][0].isdecimal() and rows[0][1] == "kB":
            return int(rows[0][0]) * 1024
    except OSError:
        pass
    try:
        pages, size = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return pages * size if pages >= 0 and size > 0 else 0
    except (AttributeError, OSError, ValueError):
        return 0


def _unescape_mount(value: str) -> str:
    for escaped, plain in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, plain)
    return value


def _cgroup_limits(proc: Path = Path("/proc")) -> tuple[int | None, int | None]:
    """Read visible unified ancestors, without configuring any resource owner.

    Missing Linux controls retain the affinity/free-memory fallback. Unsupported
    or incomplete visible controls conservatively select only one worker.
    """
    try:
        membership = (proc / "self/cgroup").read_text().splitlines()
    except FileNotFoundError:
        return None, None
    except OSError:
        return 1, 0
    try:
        selected = [line[3:] for line in membership if line.startswith("0::")]
        if not selected:
            return None, None
        if len(selected) != 1:
            raise ValueError("ambiguous cgroup membership")
        member = Path(selected[0])
        if not member.is_absolute() or ".." in member.parts:
            raise ValueError("unsupported cgroup membership")
        mounts = []
        for line in (proc / "self/mountinfo").read_text().splitlines():
            before, separator, after = line.partition(" - ")
            if separator and after.split()[0] == "cgroup2":
                fields = before.split()
                mounts.append((Path(_unescape_mount(fields[3])), Path(_unescape_mount(fields[4]))))
        if len(mounts) != 1:
            raise ValueError("ambiguous cgroup mount")
        root, mount = mounts[0]
        if not root.is_absolute() or not mount.is_absolute() or ".." in root.parts or ".." in mount.parts:
            raise ValueError("unsupported cgroup mount")
        current = mount / member.relative_to(root)
        cpus, memories = [], []
        for directory in (current, *current.parents):
            if not directory.is_relative_to(mount):
                break
            try:
                value = (directory / "memory.max").read_text().strip()
            except FileNotFoundError:
                value = "max"  # A controller may be absent, including at the mount root.
            if value != "max":
                used = (directory / "memory.current").read_text().strip()
                if not value.isdecimal() or not used.isdecimal():
                    raise ValueError("unsupported memory ceiling")
                memories.append(max(0, int(value) - int(used)))
            try:
                bandwidth = (directory / "cpu.max").read_text().split()
            except FileNotFoundError:
                bandwidth = ["max", "1"]
            if len(bandwidth) != 2 or not bandwidth[1].isdecimal() or int(bandwidth[1]) <= 0:
                raise ValueError("unsupported CPU ceiling")
            if bandwidth[0] != "max":
                if not bandwidth[0].isdecimal() or int(bandwidth[0]) <= 0:
                    raise ValueError("unsupported CPU quota")
                cpus.append(max(1, int(bandwidth[0]) // int(bandwidth[1])))
        return min(cpus, default=None), min(memories, default=None)
    except (OSError, ValueError, IndexError):
        return 1, 0


def worker_limit() -> int:
    """At most eight captures, bounded by affinity, host memory and cgroups."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    memory = _available_memory()
    cpu_ceiling, memory_ceiling = _cgroup_limits()
    if cpu_ceiling is not None:
        cpus = min(cpus, cpu_ceiling)
    if memory_ceiling is not None:
        memory = min(memory, memory_ceiling)
    return max(1, min(8, cpus, memory // (2 * 1024**3)))


def independent(entry: dict) -> bool:
    return bool(
        (entry.get("source") == "pytorch" or entry.get("pytorch_ref"))
        and not entry.get("materialized_capture")
        and not entry.get("micro_model")
        and entry.get("kind") != "model"
        and entry.get("op") != "model"
    )


def _fresh_model(entry):
    return bool(
        (entry.get("kind") == "model" or entry.get("op") == "model")
        and not entry.get("materialized_capture")
        and not entry.get("micro_model")
    )


def _model_jobs(pool, entries, prepare, workers):
    """Bound unpublished captures and drain/discard them on every consumer exit."""
    from .staging import PreparedWrite

    pending, active = deque(), []
    source = iter(entries)

    def submit():
        try:
            entry = next(source)
        except StopIteration:
            return False
        job = pool.submit(contextvars.copy_context().run, prepare, entry)
        pending.append((entry, job))
        active.append(job)
        return True

    def discard(job):
        try:
            _, written = job.result()
        except BaseException:
            return  # The ordinary caller owns failures; a failed preparation cleans its own stage.
        if isinstance(written, PreparedWrite):
            written.close()

    try:
        for _ in range(workers):
            if not submit():
                break
        while pending:
            entry, job = pending.popleft()
            yield entry, job
            discard(job)
            active.remove(job)
            submit()
    finally:
        for job in active:
            discard(job)  # Wait for owned captures before dropping their unpublished stages.


def ordered_jobs(entries, write, *, workers: int, prepare_model=None):
    """Yield futures in declared order, finishing each independent batch before dependent writes.

    Exceptions remain attached to their original entry for the ordinary failure ledger.
    Every thread receives its own copy of the caller's observed-facts/import context.
    """
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phase0-capture") as pool:

        def batch_kind(entry):
            if prepare_model is not None and _fresh_model(entry):
                return "model"
            return "independent" if independent(entry) else "serial"

        for kind, batch in groupby(entries, key=batch_kind):
            if kind == "model" and workers > 1:
                yield from _model_jobs(pool, batch, prepare_model, workers)
            elif kind == "independent" and workers > 1:
                jobs = [(entry, pool.submit(contextvars.copy_context().run, write, entry)) for entry in batch]
                yield from jobs
            else:
                for entry in batch:
                    job = Future()
                    try:
                        job.set_result(write(entry))
                    except Exception as exc:  # noqa: BLE001 -- the ordinary ledger owns failures
                        job.set_exception(exc)
                    yield entry, job
