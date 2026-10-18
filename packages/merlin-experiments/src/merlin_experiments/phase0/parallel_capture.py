"""Bounded independent capture batches; dependent model writes retain declaration order."""

from __future__ import annotations

import contextvars
import os
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import groupby


def worker_limit() -> int:
    """Use selected CPU affinity and leave at least two GiB per concurrent capture."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return max(1, min(8, cpus, memory // (2 * 1024**3)))


def independent(entry: dict) -> bool:
    return bool(
        (entry.get("source") == "pytorch" or entry.get("pytorch_ref"))
        and not entry.get("materialized_capture")
        and not entry.get("micro_model")
        and entry.get("kind") != "model"
        and entry.get("op") != "model"
    )


def ordered_jobs(entries, write, *, workers: int):
    """Yield futures in declared order, finishing each independent batch before dependent writes.

    Exceptions remain attached to their original entry for the ordinary failure ledger.
    Every thread receives its own copy of the caller's observed-facts/import context.
    """
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phase0-capture") as pool:
        for parallel, batch in groupby(entries, key=independent):
            if parallel and workers > 1:
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
