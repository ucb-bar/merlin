"""Independent sealed-capture replays for ``derive``, run in separate interpreters.

Deriving the requirement replays every preselected capture: the selected runtime is
inventoried and compared with its receipt, the capture is re-executed in the sandbox
and its output compared byte for byte, and the resulting attestation re-reads the
bound evidence. The replays of different labels share no state, so they may overlap.

Each replay is exactly :func:`replay_one` (the same ``verify``, ``attest_sealed_m2m``
and ``load`` calls, with the same arguments), run once per label in a spawned worker
process. Nothing is cached or skipped. The caller still consumes results one label at
a time in declared order, so a failure surfaces at the same position as before, and
``MERLIN_DERIVE_REPLAY_WORKERS`` of 0 or 1 replays lazily in-process as it always did.
A worker error that cannot cross the process boundary, or a broken pool, makes the
caller repeat that label's replay in-process.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Sequence

WORKERS_ENV = "MERLIN_DERIVE_REPLAY_WORKERS"
DEFAULT_WORKERS = 8

Job = tuple[Path, str, Path]  # selection path, selection sha256, model path


class _Rerun:
    """A worker failure that could not cross the process boundary intact."""


def replay_one(selection_path: Path, selected_sha256: str, model_path: Path) -> tuple[dict, dict, dict]:
    """One label's complete replay: (replay record, sealed attestation, loaded selection)."""
    from .capture_execution_attestation import attest_sealed_m2m
    from .capture_selection import load, verify

    evidence = verify(selection_path, expected_sha256=selected_sha256, model_path=model_path)
    attestation = attest_sealed_m2m(evidence, selection_path=selection_path, model_path=model_path)
    selected = load(selection_path, expected_sha256=selected_sha256)
    return evidence, attestation, selected


def _task(replay, selection_path: Path, selected_sha256: str, model_path: Path):
    try:
        return replay(selection_path, selected_sha256, model_path)
    except Exception as exc:  # noqa: BLE001 -- re-raised in the caller, never handled here
        try:
            pickle.loads(pickle.dumps(exc))
        except Exception:  # noqa: BLE001 -- an unportable error is reproduced by the caller instead
            return _Rerun()
        raise


def worker_count(jobs: int) -> int:
    raw = os.environ.get(WORKERS_ENV)
    if raw is not None:
        if not raw.isdecimal():
            raise ValueError(f"{WORKERS_ENV} must be a non-negative integer, got {raw!r}")
        requested = int(raw)
    else:
        requested = DEFAULT_WORKERS
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(0, min(requested, cpus, jobs))


class Replays:
    """Replays of ``jobs``, indexable in declared order; index ``i`` returns or raises label ``i``'s outcome."""

    def __init__(self, jobs: Sequence[Job], *, replay=replay_one):
        self._jobs = list(jobs)
        self._replay = replay
        self._futures: list[Any] = []
        self._pool: ProcessPoolExecutor | None = None
        size = worker_count(len(self._jobs))
        if size > 1:
            try:
                self._pool = ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context("spawn"))
                self._futures = [self._pool.submit(_task, replay, *job) for job in self._jobs]
            except (OSError, ImportError, NotImplementedError, ValueError, RuntimeError):
                self.close()  # e.g. no usable /dev/shm: replays stay in-process, unchanged
                self._futures = []

    def __getitem__(self, index: int) -> tuple[dict, dict, dict]:
        if not self._futures:
            return self._replay(*self._jobs[index])
        try:
            result = self._futures[index].result()
        except (BrokenProcessPool, pickle.PicklingError):
            return self._replay(*self._jobs[index])
        return self._replay(*self._jobs[index]) if isinstance(result, _Rerun) else result

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)  # wait for owned sandboxes before returning or failing

    def __enter__(self) -> Replays:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
