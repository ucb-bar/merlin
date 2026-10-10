"""Pipelined, dependency-aware capsule grading: no wave barriers, a short calibration head.

The suite used to grade in dependency WAVES, each fully finished before the next began, and to open
every wave with a serial calibration head. Both serialize on the slowest member: one 3600 s RTL run
held twelve small capsules of the next wave for ~50 min while four of five workers idled, and a
corpus whose first member was a full-shape layer ran that one capsule alone for hours.

Here a capsule starts the moment a worker is free and the ``extends`` sibling it cites (if selected)
has FINISHED -- a dependent still never starts before its evidence exists, which is the one ordering
the waves were for. Ready members start longest first (declared operand volume). Calibration runs once
for the whole suite, on the CHEAPEST ready members, and only while tiers remain unpriced. Results come
back in input order.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from math import prod
from typing import Any

#: Wall seconds a serial calibration member may run before the pool fans out around it. Calibration
#: pays off when the head is cheap (a 0.29 s tier priced before 7 workers pay a 24.5 s one); a
#: full-shape layer at the head of a perf corpus took hours, all of it with every other worker idle.
CALIBRATION_BUDGET_S = 120.0


def dependencies(capsules: Sequence[Mapping]) -> dict[str, set[str]]:
    """``name -> {selected sibling}`` from ``extends``; validates names and rejects cycles."""
    names = [str(cap.get("name") or "") for cap in capsules]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("capsule dependency scheduling requires unique nonempty names")
    selected = set(names)
    deps = {
        name: ({str(cap["extends"])} if cap.get("extends") and str(cap["extends"]) in selected else set())
        for name, cap in zip(names, capsules, strict=True)
    }
    done: set[str] = set()
    pending = set(names)
    while pending:
        ready = {name for name in pending if deps[name] <= done}
        if not ready:
            raise ValueError("cyclic capsule extends dependencies: " + ", ".join(sorted(pending)))
        done |= ready
        pending -= ready
    return deps


def operand_volume(capsule: Mapping) -> int:
    """Declared input elements: a cheap, deterministic proxy for a member's simulation cost."""
    total = 0
    for leaf in capsule.get("inputs") or ():
        shape = leaf.get("shape") if isinstance(leaf, Mapping) else None
        if isinstance(shape, list) and all(isinstance(x, int) and x > 0 for x in shape):
            total += prod(shape)
    return total


def run_pipelined(
    capsules: Sequence[dict],
    run_one: Callable[[dict, int], dict],
    *,
    max_workers: int,
    calibrate: Callable[[], set[str]] | None = None,
    calibration_cap: int = 0,
    calibration_budget_s: float | None = CALIBRATION_BUDGET_S,
) -> list[dict]:
    """Grade ``capsules`` with at most ``max_workers`` in flight, each as soon as its sibling finished.

    ``run_one(capsule, n_parallel)`` grades one member. ``calibrate()`` returns the tiers priced so far;
    while it keeps growing, up to ``calibration_cap`` of the cheapest ready members run first, alone,
    so the ladder learns its tier order before the fan-out (a member that passes ends calibration).
    A head member still running after ``calibration_budget_s`` ends calibration and the pool fans out
    around it (``None`` waits for it).
    """
    deps = dependencies(capsules)
    names = [str(cap["name"]) for cap in capsules]
    index = {name: i for i, name in enumerate(names)}
    children: dict[str, list[str]] = {name: [] for name in names}
    for name, parents in deps.items():
        for parent in parents:
            children[parent].append(name)
    waiting = {name: len(parents) for name, parents in deps.items()}
    results: dict[int, dict] = {}
    # LONGEST FIRST (by declared operand volume, then input order): the classical makespan rule. The
    # member that bounds the suite starts as soon as a worker is free instead of after the small ones.
    volume = {name: operand_volume(cap) for name, cap in zip(names, capsules, strict=True)}
    ready: list[tuple[int, int, str]] = [(-volume[n], index[n], n) for n in names if waiting[n] == 0]
    heapq.heapify(ready)

    def finished(name: str, result: dict) -> None:
        results[index[name]] = result
        for child in children[name]:
            waiting[child] -= 1
            if waiting[child] == 0:
                heapq.heappush(ready, (-volume[child], index[child], child))

    if max_workers <= 1:
        while ready:
            *_, name = heapq.heappop(ready)
            finished(name, run_one(capsules[index[name]], 1))
        return [results[i] for i in range(len(capsules))]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        running: dict[Any, str] = {}
        if calibrate is not None and calibration_cap > 0:
            seen = calibrate()
            for _ in range(calibration_cap):
                if not ready:
                    break
                item = min(ready, key=lambda entry: (-entry[0], entry[1]))  # the cheapest ready member
                ready.remove(item)
                heapq.heapify(ready)
                future = pool.submit(run_one, capsules[item[1]], 1)
                done, _ = wait([future], timeout=calibration_budget_s)
                if not done:
                    # A slow head teaches its tier order too late to be worth the idle workers: keep it
                    # running and fan out now, instead of grading one capsule alone for hours.
                    running[future] = item[2]
                    break
                result = future.result()
                finished(item[2], result)
                now = calibrate()
                if result.get("status") == "pass" or now == seen:
                    break
                seen = now
        while ready or running:
            while ready and len(running) < max_workers:
                *_, name = heapq.heappop(ready)
                running[pool.submit(run_one, capsules[index[name]], max_workers)] = name
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                finished(running.pop(future), future.result())
    return [results[i] for i in range(len(capsules))]
