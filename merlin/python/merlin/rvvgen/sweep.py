"""Parallel isolated certification of many (package, workload) cells, + ranking aggregation.

Each cell gets a distinct run_id + workdir (full filesystem isolation — no git worktree needed,
packages are data not code edits). spike/K1 are subprocess/SSH-bound so a thread pool overlaps
them; FireSim still serializes through its own queue downstream. Used by the beam-search to
certify a whole generation of forks concurrently.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .runner import certify_rvv


def _max_workers(n: int) -> int:
    return max(1, min(n, min(8, (os.cpu_count() or 4) - 2)))


def run_sweep(jobs: list[dict[str, Any]], *, certify_fn: Callable = certify_rvv,
              max_workers: int | None = None) -> list[dict[str, Any]]:
    """Certify each job concurrently. Each job is a kwargs dict for ``certify_fn`` (must include
    package_dir, model_dir, runs_root, run_id). Returns results in submission order; a cell that
    raises is captured as ``{"status": "error", "error": ...}`` (the sweep never aborts)."""
    if not jobs:
        return []
    results: list[Any] = [None] * len(jobs)

    def _run(i: int, job: dict) -> None:
        try:
            results[i] = certify_fn(**job)
        except Exception as e:  # a harness bug in one cell must not kill the sweep
            results[i] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                          "run_id": job.get("run_id")}

    with ThreadPoolExecutor(max_workers=max_workers or _max_workers(len(jobs))) as ex:
        list(ex.map(lambda p: _run(*p), list(enumerate(jobs))))
    return results


def rank_results(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank scored nodes best-first. Key: correctness first, then REAL K1 speedup (measured silicon,
    the true driver when the beam ran the k1 target), then structural_match toward the expert (the
    proxy when there is no real measurement), then fewer spike cycles (weak functional tiebreak).

    A fork that broke numerics (gate_ok False) sorts last regardless of speed — the INLINED-VS-ROUTED
    / real-vs-fake discipline: no speed credit without correctness."""
    def key(n: dict) -> tuple:
        correct = 1 if n.get("gate_ok") else 0
        spd = n.get("speedup")            # real K1 speedup vs baseline (>1 faster); None if no k1 run
        sm = n.get("structural_match") or 0.0
        cyc = n.get("cycles")
        return (correct, spd if spd is not None else -1.0, sm,
                -(cyc if cyc is not None else float("inf")))
    return sorted(scored, key=key, reverse=True)
