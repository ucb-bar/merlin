"""Attach a self-check's timing observations, each test case on its own budget, a few at a time.

Observations ran one after another against one shared budget, so a check of many cases returned cycles
for the first few only. Each case now gets :func:`joint_performance.observation_budget` of its own, and
when the calling thread holds no CPU slot the observations run concurrently: each leases its own slot
from the shared pool (as it always did) and the calling thread waits holding none. A thread that already
holds a slot keeps the sequential order, so no slot is ever held while waiting for another.
"""

from __future__ import annotations

import contextvars
import time
from concurrent.futures import ThreadPoolExecutor

#: Concurrent observations per self-check; each still waits for a slot of the shared pool.
MAX_CONCURRENT = 4


def attach_all(jobs, *, context, request_deadline):
    """``joint_performance.attach`` for every job dict (row, root, result, compiler, capsule)."""
    from merlin.common import cpu_pool

    from .joint_performance import attach, observation_budget

    def one(job):
        started = time.monotonic()
        attach(job["row"], job["root"], result=job["result"], context=context, compiler=job["compiler"],
               capsule=job["capsule"], remaining_s=observation_budget(request_deadline),
               request_deadline=request_deadline)
        performance = job["row"].get("performance")
        if isinstance(performance, dict):
            performance["observation_seconds"] = round(time.monotonic() - started, 1)

    jobs = list(jobs)
    width = min(MAX_CONCURRENT, cpu_pool.compute_capacity(), len(jobs))
    if width <= 1 or cpu_pool.current_thread_owns_cpu_lease():
        for job in jobs:
            one(job)
        return
    with ThreadPoolExecutor(width, thread_name_prefix="joint-observation") as pool:
        # Each worker carries the caller's lease priority (a context variable threads do not inherit).
        futures = [pool.submit(contextvars.copy_context().run, one, job) for job in jobs]
        for future in futures:
            future.result()
