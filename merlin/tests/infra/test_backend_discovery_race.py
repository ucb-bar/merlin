"""Out-of-tree backend discovery must be atomic: parallel grading resolves backends from threads.

The guard was a check-then-set on a module-level sentinel, published BEFORE the loading it guards. A
worker that arrived while another was still importing saw the sentinel set, returned immediately, and
then read an empty registry — ``KeyError: 'gemmini'``, surfaced by the runner as
``spike invocation failed: 'gemmini'`` / tool_crash. Because grading fans capsules across threads, this
is timing-dependent: on the first gemmini arm-4 run 15 of 20 capsules crashed and 5 passed, so an agent
that had solved all 20 was graded 5. A flaky isolation/oracle failure that looks like an agent failure
is the worst kind, so this pins it.
"""
from __future__ import annotations

import concurrent.futures

import pytest

from merlin.runtime.backends import base


@pytest.mark.parametrize("attempt", range(3))
def test_concurrent_get_backend_never_sees_a_half_built_registry(attempt, monkeypatch):
    # Force a fresh discovery pass for every attempt: clear the sentinel so the workers race on it.
    monkeypatch.setattr(base, "_oot_env_seen", object(), raising=False)

    def resolve():
        return base.get_backend("gemmini").__name__

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(resolve) for _ in range(16)]
        names = []
        for f in futures:
            names.append(f.result())     # a KeyError here IS the bug
    assert len(set(names)) == 1, f"threads resolved different modules: {set(names)}"


def test_the_sentinel_is_published_only_after_discovery(monkeypatch):
    """Directly: while the loader runs, the sentinel must not yet advertise completion."""
    seen_during_load = []
    real = base._load_oot_backend

    def slow(name, path, **kw):
        seen_during_load.append(base._oot_env_seen)
        return real(name, path, **kw)

    monkeypatch.setattr(base, "_load_oot_backend", slow)
    monkeypatch.setattr(base, "_oot_env_seen", object(), raising=False)
    key = base.os.environ.get("MERLIN_TARGET_PATH", "")
    base._ensure_oot_discovered()
    assert all(s != key for s in seen_during_load), (
        "the sentinel already equalled the current key while modules were still loading — another "
        "thread would have returned early onto an incomplete registry")
