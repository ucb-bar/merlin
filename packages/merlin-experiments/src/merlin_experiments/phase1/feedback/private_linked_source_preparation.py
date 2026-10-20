"""Owner-local parsed captures; every source witness still runs independently.

No source, tree, admission, roster or linked-build verdict is retained. Reuse is
explicit, belongs to one live owner/thread/process, and checks original bytes
and rendered IR at every use and before leaving each validation scope.
"""

from __future__ import annotations

import os
import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from merlin.common import mlir_query as mq
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.targetgen.application_inventory import PreparedApplicationSource

_ACTIVE = ContextVar("linked_source_preparation", default=None)
_ISSUED = weakref.WeakKeyDictionary()
_MAX_SOURCES = 16


class PreparationChanged(ValueError):
    """A prepared source or its live owner no longer matches."""


class Preparation:
    def __init__(self, owner):
        _ISSUED[self] = (weakref.ref(owner), os.getpid(), threading.get_ident(), {}, {"token": None})

    def _state(self, owner=None):
        state = _ISSUED.get(self) if type(self) is Preparation else None
        if (
            state is None
            or state[0]() is None
            or (owner is not None and state[0]() is not owner)
            or state[1:3] != (os.getpid(), threading.get_ident())
        ):
            raise PreparationChanged("linked source preparation has a foreign or expired owner")
        return state

    def _check(self):
        for path, prepared in self._state()[3].items():
            try:
                prepared.reopen(path)
            except (OSError, ValueError) as exc:
                raise PreparationChanged("linked prepared source or parsed IR changed") from exc

    @contextmanager
    def activate(self, owner):
        lifetime = self._state(owner)[4]
        if lifetime["token"] is not None:
            raise PreparationChanged("linked source preparation is already active")
        self._check()
        scope = lifetime["token"] = object()
        token = _ACTIVE.set((self, scope))
        try:
            yield
            self._check()
        finally:
            lifetime["token"] = None
            _ACTIVE.reset(token)

    def _read(self, path, normalized=None):
        cache = self._state()[3]
        # Reject aliases rather than sharing one preparation across spellings.
        if path != path.resolve(strict=True) or not path.is_file():
            raise PreparationChanged("linked source preparation requires a canonical file")
        if path not in cache:
            if len(cache) == _MAX_SOURCES:
                oldest = next(iter(cache))
                cache[oldest].reopen(oldest)
                del cache[oldest]
            if normalized is None:
                normalized, _ = normalize_capture_mlir(path.read_bytes().decode("utf-8"))
            cache[path] = PreparedApplicationSource(path, normalized)
        try:
            return cache[path].reopen(path)
        except (OSError, ValueError) as exc:
            raise PreparationChanged("linked prepared source or parsed IR changed") from exc


class _TemporaryOwner:
    pass


def _active():
    selected = _ACTIVE.get()
    if selected is None:
        return None
    current, scope = selected
    if current._state()[4]["token"] is not scope:
        raise PreparationChanged("linked source validation scope expired")
    return current


@contextmanager
def validation_scope():
    """Reuse a native partition's active preparation, or own one fresh join."""
    current = _active()
    if current is not None:
        current._state()
        yield
    else:
        owner = _TemporaryOwner()
        with Preparation(owner).activate(owner):
            yield


def source(path, *, raw=None):
    """Return original raw/normalized/receipt/module; never return a verdict."""
    path = Path(path)
    current = _active()
    if current is None:
        original = path.read_bytes() if raw is None else raw
        normalized, receipt = normalize_capture_mlir(original.decode("utf-8"))
        return original, normalized, receipt, mq.parse(normalized)
    result = current._read(path)
    if raw is not None and result[0] != raw:
        raise PreparationChanged("linked source differs from the caller's byte snapshot")
    return result


def parse(path, normalized, *, raw=None):
    """Keep the original unscoped parse and every preceding refusal in place."""
    current = _active()
    if current is None:
        return mq.parse(normalized)
    result = current._read(Path(path), normalized)
    if result[1] != normalized or (raw is not None and result[0] != raw):
        raise PreparationChanged("linked source differs from its original normalization or byte snapshot")
    return result[3]
