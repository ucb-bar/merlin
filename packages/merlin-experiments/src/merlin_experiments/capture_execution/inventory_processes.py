"""Full tree inventories in separate interpreters while threaded captures share one GIL.

Phase 0 issues sealed captures from threads of one process, and every capture
re-inventories the selected runtime (about 10^5 members) several times: the plan
selection, the cache admission, before and after hard-linking, and the staged guest
root. Hashing large files releases the interpreter lock, but the per-member stat,
open, xattr and row bookkeeping does not, so dozens of capture threads serialize on
one core while the host idles.

While active, each such inventory runs synchronously in a worker interpreter: the
same function, loaded from the same source file, reading every member afresh.
Nothing is cached, skipped, reordered or shared between callers. The calling thread
blocks until the fresh read completes, inside whatever locks it holds, and receives
the same rows or the same exception (with its cause chain, which failure reasons
report). A worker that cannot run or report an inventory makes the caller repeat it
in-process.

The issuer sources whose digests are bound into receipts are not edited. Routing is
a scoped rebinding of their module attributes, made only while each attribute is
still its source file's own definition, and undone when the last scope exits.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterator

PROCESSES_ENV = "MERLIN_SEALED_INVENTORY_PROCESSES"

#: (bound module, attribute, defining module, function): every full inventory entry point.
_ROUTES = (
    ("sealed_static", "_tree", "sealed_static", "_tree"),
    ("readonly_runtime", "_tree", "sealed_static", "_tree"),
    ("sealed_m2m", "_strict_tree", "sealed_static", "_tree"),
    ("sealed_m2m", "_source_tree", "sealed_m2m", "_source_tree"),
)
_CAUSE_DEPTH = 8

_lock = threading.Lock()
_depth = 0
_pool: ProcessPoolExecutor | None = None
_bound: list[tuple[Any, str, Callable[..., Any], Callable[..., Any]]] = []


class _Rerun:
    """A worker result that could not cross the process boundary intact."""


def _module(name: str):
    return import_module(f"{__package__}.{name}")


def _own(function: Any, defining: str, name: str) -> bool:
    module = _module(defining)
    code = getattr(function, "__code__", None)
    return (
        code is not None
        and getattr(function, "__qualname__", None) == name
        and getattr(function, "__module__", None) == module.__name__
        and Path(code.co_filename).resolve() == Path(module.__file__).resolve()
    )


def _source_digest(defining: str) -> str:
    from .sealed_static import _file_digest

    return _file_digest(Path(_module(defining).__file__))


def _inventory(defining: str, name: str, source_sha256: str, root: Path, kwargs: dict[str, Any]) -> Any:
    """Worker side: the unrebound definition, read afresh; failures keep their cause chain."""
    function = getattr(_module(defining), name)
    if not _own(function, defining, name) or _source_digest(defining) != source_sha256:
        return _Rerun()  # never run a definition other than the caller's own source bytes
    try:
        return ("rows", function(root, **kwargs))
    except Exception as exc:  # noqa: BLE001 -- re-raised in the caller, never handled here
        chain, current = [], exc
        while isinstance(current, BaseException) and len(chain) < _CAUSE_DEPTH and current not in chain:
            chain.append(current)
            current = current.__cause__
        flags = [error.__suppress_context__ for error in chain]
        try:
            pickle.loads(pickle.dumps(chain))
        except Exception:  # noqa: BLE001 -- an unportable error is reproduced by the caller instead
            return _Rerun()
        return ("raise", chain, flags)


def _routed(original: Callable[..., Any], defining: str, name: str, source_sha256: str) -> Callable[..., Any]:
    def inventory(root, **kwargs):
        pool = _pool
        if pool is None or type(root) is not type(Path()) or not root.is_absolute():
            return original(root, **kwargs)
        try:
            result = pool.submit(_inventory, defining, name, source_sha256, root, kwargs).result()
        except (BrokenProcessPool, RuntimeError, pickle.PicklingError):
            return original(root, **kwargs)
        if isinstance(result, _Rerun):
            return original(root, **kwargs)
        if result[0] == "rows":
            return result[1]
        _, chain, flags = result
        for error, cause, suppress in zip(chain, [*chain[1:], None], flags):
            error.__cause__ = cause
            error.__suppress_context__ = suppress
        raise chain[0]

    inventory.__wrapped__ = original
    inventory.__qualname__ = getattr(original, "__qualname__", name)
    inventory.__doc__ = getattr(original, "__doc__", None)
    return inventory


def _size(requested: int) -> int:
    raw = os.environ.get(PROCESSES_ENV)
    if raw is not None:
        if not raw.isdecimal():
            raise ValueError(f"{PROCESSES_ENV} must be a non-negative integer, got {raw!r}")
        requested = int(raw)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(0, min(requested, cpus))


@contextmanager
def offloaded_inventories(workers: int) -> Iterator[int]:
    """Route full inventories to up to ``workers`` interpreters (0 or 1: unchanged, in-process)."""
    global _depth, _pool
    with _lock:
        if _depth == 0:
            size = _size(workers)
            if size > 1:
                try:
                    _pool = ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context("spawn"))
                except (OSError, ImportError, NotImplementedError, ValueError):
                    _pool = None  # e.g. no usable /dev/shm: inventories stay in-process, unchanged
            if _pool is not None:
                for bound, attribute, defining, name in _ROUTES:
                    module = _module(bound)
                    current = getattr(module, attribute)
                    if _own(current, defining, name):
                        routed = _routed(current, defining, name, _source_digest(defining))
                        setattr(module, attribute, routed)
                        _bound.append((module, attribute, current, routed))
        _depth += 1
        active = _pool._max_workers if _pool is not None else 0
    try:
        yield active
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                pool, _pool = _pool, None
                for module, attribute, original, routed in reversed(_bound):
                    if getattr(module, attribute) is routed:
                        setattr(module, attribute, original)
                _bound.clear()
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
