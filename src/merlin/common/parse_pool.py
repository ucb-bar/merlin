"""Pure "bytes in, plain data out" functions, evaluated in spawned interpreters or remembered by content.

Some Phase 0 steps parse the same captured MLIR into a plain summary several times in one process
(every selection, every operation accounting), one file after another on one core. A job here is a
function of the exact bytes it is handed and nothing else, returning plain picklable data. The reads
stay where the serial code made them:

* :meth:`Prefetch.get` reads the file when the caller asks for it (the point where the serial code
  read it), hashes those bytes, and returns a value computed from exactly those bytes. A value a
  worker computed earlier is used only when the worker's bytes hash to the same digest; otherwise,
  or when the worker failed in any way, the function is run again in-process on the bytes just read,
  so a failure raises the exception the serial code raised, from the same place, in the same order.
* A completed value is remembered by ``(job, sha256 of the bytes)`` as pickled bytes and every
  caller gets its own freshly unpickled copy, so no caller can see another's mutation. The key is
  the content; a changed file is a different key.

``MERLIN_PARSE_WORKERS``: ``0`` is the serial behaviour (no pool, no memo); a positive number is the
worker count; unset picks ``min(jobs, usable CPUs, 12, available memory / 2 GiB)``.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import pickle
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Any

WORKERS_ENV = "MERLIN_PARSE_WORKERS"
_DEFAULT_CAP = 12
_BYTES_PER_WORKER = 2 << 30
_MEMO_MAX = 64

_MEMO: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_MEMO_LOCK = threading.Lock()


def _resolve(ref: str):
    module, _, name = ref.partition(":")
    function = import_module(module)
    for part in name.split("."):
        function = getattr(function, part)
    return function


def _mem_available() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def configured_workers() -> int | None:
    """The explicit worker count, or ``None`` when unset or unusable (use the default)."""
    value = os.environ.get(WORKERS_ENV)
    if value is None or not value.strip():
        return None
    try:
        count = int(value)
    except ValueError:
        return None
    return count if count >= 0 else None


def enabled() -> bool:
    return configured_workers() != 0


def worker_count(jobs: int) -> int:
    """How many worker interpreters to start for ``jobs`` independent jobs (0 or 1 means none)."""
    explicit = configured_workers()
    if explicit is not None:
        count = min(explicit, jobs)
        return count if count > 1 else 0
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    count = min(jobs, cpus, _DEFAULT_CAP)
    available = _mem_available()
    if available is not None:
        count = min(count, max(1, available // _BYTES_PER_WORKER))
    return count if count > 1 else 0


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _remember(key: tuple[str, str], value: Any) -> None:
    try:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 -- an unpicklable result is simply not remembered
        return
    with _MEMO_LOCK:
        _MEMO[key] = blob
        _MEMO.move_to_end(key)
        while len(_MEMO) > _MEMO_MAX:
            _MEMO.popitem(last=False)


def _recall(key: tuple[str, str]):
    with _MEMO_LOCK:
        blob = _MEMO.get(key)
        if blob is not None:
            _MEMO.move_to_end(key)
    return None if blob is None else (pickle.loads(blob),)


def clear_memo() -> None:
    with _MEMO_LOCK:
        _MEMO.clear()


def _worker_job(ref: str, source: str | bytes) -> tuple[str, Any]:
    """In a worker: take the bytes (read the file now when given a path), hash them, run the job on them."""
    raw = Path(source).read_bytes() if isinstance(source, str) else source
    return _digest(raw), _resolve(ref)(raw)


def _stop(pool: ProcessPoolExecutor) -> None:
    pool.shutdown(wait=False, cancel_futures=True)


class Prefetch:
    """Start ``ref`` (``"module:function"``, taking bytes) for ``sources`` in worker interpreters.

    ``sources`` maps a key to a file path (the bytes are read where :meth:`get` is called) or to the
    bytes themselves; an iterable of paths keys each path by its string. Use as a context manager
    around the loop that consumes the values with :meth:`get`, or call :meth:`start` and
    :meth:`close`; workers still running or queued are stopped when it closes or is dropped. With
    the serial setting, or fewer than two sources, nothing is started and :meth:`get` computes
    in-process.
    """

    def __init__(self, ref: str, sources) -> None:
        self.ref = ref
        if not isinstance(sources, dict):
            sources = {str(p): Path(p) for p in sources}
        self._sources = sources
        self._futures: dict[Any, Future] = {}
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Prefetch:
        return self.start()

    def start(self) -> Prefetch:
        workers = worker_count(len(self._sources)) if self._pool is None else 0
        if workers > 1:
            try:
                self._pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
                weakref.finalize(self, _stop, self._pool)
                for key, source in self._sources.items():
                    self._futures[key] = self._pool.submit(
                        _worker_job, self.ref, str(source) if isinstance(source, Path) else source
                    )
            except Exception:  # noqa: BLE001 -- no pool: every get() computes in-process
                self.close()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        pool, self._pool = self._pool, None
        self._futures = {}
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def get(self, key: str | Path, raw: bytes | None = None) -> Any:
        """The job's value for the bytes now at ``key`` (read here, not earlier) or for ``raw`` when given."""
        key = str(key)
        if raw is None:
            raw = Path(key).read_bytes()
        if not enabled():
            return _resolve(self.ref)(raw)
        digest = _digest(raw)
        memo_key = (self.ref, digest)
        hit = _recall(memo_key)
        if hit is not None:
            self._futures.pop(key, None)
            return hit[0]
        future = self._futures.pop(key, None)
        if future is not None:
            try:
                worker_digest, value = future.result()
            except BaseException:  # noqa: BLE001 -- recompute below; the serial code's own error surfaces there
                pass
            else:
                if worker_digest == digest:
                    _remember(memo_key, value)
                    return value
        value = _resolve(self.ref)(raw)
        _remember(memo_key, value)
        return value


def cached(ref: str, raw: bytes) -> Any:
    """The job's value for exactly ``raw``, computed in-process at most once per distinct content."""
    if not enabled():
        return _resolve(ref)(raw)
    key = (ref, _digest(raw))
    hit = _recall(key)
    if hit is not None:
        return hit[0]
    value = _resolve(ref)(raw)
    _remember(key, value)
    return value
