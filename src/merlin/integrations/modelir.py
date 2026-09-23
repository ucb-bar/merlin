"""Scoped access to the external ModeLIR Python package with explicit import-cache and working-directory lifecycles."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def importable(mlc_dir):
    """Temporarily expose a checkout when ``mlc`` is not importable; retain loaded modules.

    Only the path insertion is scoped. Module identities deliberately survive oracle calls;
    this does not isolate competing checkouts or roll back upstream import side effects.
    """
    added = None
    if mlc_dir is not None and importlib.util.find_spec("mlc") is None:
        added = str(mlc_dir)
        sys.path.insert(0, added)
    try:
        yield
    finally:
        if added is not None:
            try:
                sys.path.remove(added)
            except ValueError:
                pass


@contextmanager
def discovery_imports(root: str | Path | None):
    """Expose discovery imports, removing newly loaded ``mlc.*`` only if we inserted the root.

    Existing importable installations and borrowed modules are never evicted. This preserves
    discovery's availability behavior, not arbitrary import side effects or thread isolation.
    """
    d = root
    added = None
    before: frozenset[str] = frozenset()
    if d is not None and importlib.util.find_spec("mlc") is None:
        added = str(d)
        sys.path.insert(0, added)
        before = frozenset(sys.modules)
    try:
        yield
    finally:
        if added is not None:
            try:
                sys.path.remove(added)
            except ValueError:
                pass
            # Drop mlc modules loaded during the block so ``import mlc`` does not keep resolving from the
            # module cache after the path entry is gone (which would flip mlc_available() True process-wide).
            for name in [n for n in sys.modules if n == "mlc" or n.startswith("mlc.")]:
                if name not in before:
                    del sys.modules[name]


#: Serializes :func:`artifact_context`. The working directory and ``sys.path`` are PROCESS-global, but capsules
#: are graded on a ThreadPoolExecutor — so two threads entering this block interleave their chdirs, and
#: the first to leave restores the repo cwd out from under the one still inside. mlc then cannot resolve
#: its own ``runs/...`` artifacts and reports the arc model as ABSENT: a correct submission grades
#: ``incomplete`` on "mlc arc model unavailable", intermittently and only under parallel grading (the
#: worse tail is the loser restoring LAST, leaving every later thread with the wrong cwd). Reentrant so a
#: nested entry on the same thread does not deadlock.
_ARTIFACT_LOCK = threading.RLock()


@contextmanager
def artifact_context(root: str | Path | None = None, *, resolve_root: Callable[[], str | Path | None] | None = None):
    """mlc resolves its ``runs/...`` arc artifacts by paths RELATIVE to its own root, so its cosim +
    discovery entry points run with CWD = the mlc dir. Also CONTEXT-INSERT the mlc dir on ``sys.path`` so
    ``import mlc`` resolves even when the process's ``sys.path`` no longer carries the cwd (``''``) entry —
    mlc is not pip-installed, so without this the import relied on cwd-relative resolution and broke inside
    a process that rewrote ``sys.path`` (e.g. the capsule-bench driver's xdsl setup), silently failing the
    arc oracle preflight. The insert is context-managed (removed on exit), NOT global: a permanent insert
    flips ``mlc_available()`` process-wide and un-skips heavy tests.

    Held under :data:`_ARTIFACT_LOCK`: cwd/``sys.path`` are process state, and the grader runs capsules on
    threads, so concurrent entries must not interleave (see the lock's note). ``resolve_root`` is
    called inside that lock, preserving configured-root lookup ordering for bridge callers.
    Imported modules remain cached. Other code changing cwd without this context is not isolated.
    """
    if root is not None and resolve_root is not None:
        raise ValueError("provide an explicit ModeLIR root or resolve_root, not both")
    with _ARTIFACT_LOCK:
        d = resolve_root() if resolve_root is not None else root
        prev = os.getcwd()
        ds = str(d) if d is not None else None
        inserted = False
        if d is not None:
            os.chdir(d)
            if ds not in sys.path:
                sys.path.insert(0, ds)
                inserted = True
        try:
            yield
        finally:
            os.chdir(prev)
            if inserted:
                try:
                    sys.path.remove(ds)
                except ValueError:
                    pass
