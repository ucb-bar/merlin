"""Parallel sealed captures share one runtime store entry without ever linking a half-built one."""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import runtime_store

PLAN = {
    "selected_trees": {"venv": {"sha256": "v"}, "base": {"sha256": "b"}},
    "base": "/opt/base-python",
    "system_libs": [],
}
FILES = [f"opt/capture-venv/lib/pkg-{index}.dist-info/METADATA" for index in range(24)] + [
    f"opt/base-python/lib/module-{index}.py" for index in range(8)
]


def _slow_build(plan, destination: Path) -> None:
    # Widen the window in which another process could observe or replace a partial store.
    for name in FILES:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        time.sleep(0.002)


def _selected_plan(tmp_path: Path) -> dict:
    """A plan whose selected tree digests are those of the runtime the builder produces."""
    from merlin_experiments.capture_execution.sealed_m2m import _snapshot_tree

    built = tmp_path / "reference-build"
    _slow_build(PLAN, built)
    trees = {"venv": _snapshot_tree(built / "opt/capture-venv"), "base": _snapshot_tree(built / "opt/base-python")}
    return {**PLAN, "selected_trees": trees}


def _populate_and_link(store: str, runtime: str, rounds: int, queue, plan: dict) -> None:
    import os

    os.environ[runtime_store.STORE_ENV] = store
    runtime_store._build = _slow_build
    try:
        for index in range(rounds):
            target = Path(runtime) / str(index)
            target.mkdir(parents=True)
            # Alternate the two reader paths: a fresh populate-and-link, and a full verification
            # of the shared entry followed by a discounted, re-verified link.
            cache = runtime_store.verified_cached_entry(plan, target) if index % 2 else None
            if index % 2 and cache is None:
                raise AssertionError("a published store entry failed verification")
            runtime_store.link_runtime(plan, target, verified_cache=cache)
            linked = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
            if linked != sorted(FILES):
                raise AssertionError(f"incomplete runtime linked: {len(linked)} of {len(FILES)} members")
        queue.put(None)
    except BaseException as exc:  # noqa: BLE001 -- reported to the parent
        queue.put(f"{type(exc).__name__}: {exc}")


def _remove_published(store: Path) -> None:
    for marker in store.glob("*.complete.json"):
        marker.unlink()


@pytest.mark.parametrize("workers", [8])
def test_parallel_populate_and_link_never_sees_a_partial_store(tmp_path: Path, workers: int):
    context = multiprocessing.get_context("fork")
    plan = _selected_plan(tmp_path)
    store = tmp_path / "store"
    for attempt in range(3):
        # Each attempt starts from a store whose entry exists but is unpublished (no marker): the
        # crash-recovery path that once deleted an entry other captures were linking from.
        if attempt:
            _remove_published(store)
        queue = context.Queue()
        processes = [
            context.Process(
                target=_populate_and_link, args=(str(store), str(tmp_path / f"run-{attempt}-{n}"), 4, queue, plan)
            )
            for n in range(workers)
        ]
        for process in processes:
            process.start()
        errors = [queue.get(timeout=120) for _ in processes]
        for process in processes:
            process.join(timeout=60)
        assert errors == [None] * workers
        entries = [path for path in store.iterdir() if path.is_dir()]
        assert [path.name for path in entries] == [runtime_store._identity(plan)]
        assert not list(store.glob("*.building-*"))
