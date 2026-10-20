"""Whole-program cache/worker controls; stub predicates issue no proof authority."""

from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.llvmlower import toolchain
from merlin.perf import whole_model_build as W
from merlin.perf import whole_model_object_cache as O
from merlin.targetgen.contract import compile as C
from merlin.targetgen.contract import harness_abi
from merlin.verify import lean_emission_context as X


@pytest.fixture
def caller(tmp_path, monkeypatch):
    original = ContextVar("test_original_worker_selection", default=False)
    monkeypatch.setattr(X, "requires_fresh_object", lambda source: original.get())
    monkeypatch.setattr(W, "_object_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(W, "_object_cache_fingerprint", lambda target: "original-toolchain")
    monkeypatch.setattr(W, "prune_object_cache", lambda path: None)
    monkeypatch.setattr(harness_abi, "for_target", lambda target: SimpleNamespace(entry_symbol="fixture_entry"))
    monkeypatch.setattr(toolchain, "objcopy", lambda: Path("fixture-objcopy"))
    monkeypatch.setattr(toolchain, "nm", lambda: Path("fixture-nm"))

    def process(argv, **options):
        if argv[0] == "fixture-objcopy":
            destination = Path(argv[-1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(argv[-2], destination)
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert argv[0] == "fixture-nm"
        return subprocess.CompletedProcess(argv, 0, f"00 T fixture_entry_{Path(argv[-1]).stem}\n", "")

    monkeypatch.setattr(W.subprocess, "run", process)
    artifact = tmp_path / "actual-emission.mlir"
    artifact.write_text("same complete observed artifact bytes")
    rows = [{"group": group, "on": W.ON_PACKAGE, "artifact": str(artifact)} for group in (1, 2)]
    return original, rows, tmp_path


def test_pending_products_keep_original_worker_context_and_bypass_disk_and_share(caller, monkeypatch):
    original, rows, root = caller
    calls = []

    def compile_object(source, work, **options):
        assert original.get() is True
        calls.append(source)
        work.mkdir(parents=True)
        obj = work / "kernel.o"
        obj.write_bytes(b"native object boundary stub; not executable")
        return obj

    monkeypatch.setattr(C, "llvm_mlir_to_object", compile_object)
    monkeypatch.setattr(O, "load", lambda path: pytest.fail("a pending original trace cannot load an old object"))
    token = original.set(True)
    try:
        W._kernel_objects(rows, target="transport_fixture", out=root / "objects", jobs=2)
    finally:
        original.reset(token)
    assert len(calls) == 2  # Same bytes, but each pending source needs actual custody.
    assert all(row["on"] == W.ON_PACKAGE and row["object_cache"] == "miss" for row in rows)


@pytest.mark.parametrize("fresh", [True, False])
def test_incomplete_traced_consumer_refuses_build_untraced_fallback_unchanged(caller, monkeypatch, fresh):
    original, rows, root = caller
    monkeypatch.setattr(W, "_object_cache_root", lambda: None)

    def incomplete(*args, **kwargs):
        raise ValueError("incomplete original object boundary stub")

    monkeypatch.setattr(C, "llvm_mlir_to_object", incomplete)
    token = original.set(fresh)
    try:
        if fresh:
            with pytest.raises(W.WholeModelBuildError, match="traced object production did not complete"):
                W._kernel_objects(rows, target="transport_fixture", out=root / "objects", jobs=2)
            assert all(row["on"] == W.ON_PACKAGE for row in rows)
        else:
            W._kernel_objects(rows, target="transport_fixture", out=root / "objects", jobs=2)
            assert all(row["on"] == W.ON_VENDOR and row["cause"] == W.OBJECT_FAILED for row in rows)
    finally:
        original.reset(token)


def test_pending_trace_cannot_be_consumed_by_disposable_prewarm(caller, monkeypatch):
    original, _, root = caller
    monkeypatch.setattr(O, "object_cache_root", lambda: pytest.fail("pending prewarm must not select cached products"))
    token = original.set(True)
    try:
        assert O.prewarm("actual bytes", target="transport_fixture", work=root / "prewarm") == "off"
        assert O.prewarm_async("actual bytes", target="transport_fixture", jobs=2) is None
    finally:
        original.reset(token)


def test_untraced_background_prewarm_carries_original_context(tmp_path, monkeypatch):
    original = ContextVar("test_untraced_original_worker_selection", default=None)
    monkeypatch.setattr(X, "requires_fresh_object", lambda source: False)
    monkeypatch.setattr(O, "object_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(O, "load", lambda path: None)
    monkeypatch.setattr(O, "_FINGERPRINTS", {"transport_fixture": "original-toolchain"})
    monkeypatch.setattr(O, "_INFLIGHT", {})
    monkeypatch.setattr(O, "prewarm", lambda *a, **k: original.get())
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(O, "_pool", lambda jobs: pool)
        token = original.set("same original selection")
        try:
            future = O.prewarm_async("actual bytes", target="transport_fixture", jobs=1)
            assert future.result(timeout=5) == "same original selection"
        finally:
            original.reset(token)
