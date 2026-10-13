"""Independent tool jobs overlap without moving original package services to threads."""

from __future__ import annotations

import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.llvmlower import device_build as DB
from merlin.llvmlower import device_build_jobs as jobs


@pytest.fixture
def synthetic_build(tmp_path, monkeypatch):
    from merlin.compile import mesh
    from merlin.llvmlower import device_shim, toolchain
    from merlin.targetgen import corpus_spec, package_runtime

    monkeypatch.setattr(DB, "objects_buildable", lambda _device: None)
    monkeypatch.setattr(DB, "_objcopy", lambda: "objcopy")
    monkeypatch.setattr(mesh, "_mesh_tile_binding", lambda *a, **kw: None)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _device: SimpleNamespace(symbol="kernel"))
    monkeypatch.setattr(
        device_shim,
        "emit_translation_unit",
        lambda _d, signatures, *a, **kw: SimpleNamespace(
            symbols=tuple(signatures), text="void entry(void) {}", skipped=()
        ),
    )
    monkeypatch.setattr(corpus_spec, "build", lambda entry, _binding: (None, entry["name"]))
    monkeypatch.setattr(package_runtime, "load_package", lambda _path: SimpleNamespace(target="synthetic"))
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: "translate")
    monkeypatch.setattr(toolchain, "clang", lambda: "clang")
    monkeypatch.setattr(jobs, "build_workers", lambda: 2)
    owner = threading.get_ident()
    emitted, calls = [], []
    artifacts = {}

    def emit(_pkg, _name, path, **_kwargs):
        assert threading.get_ident() == owner
        name = path.read_text()
        emitted.append(name)
        return SimpleNamespace(returncode=0, stdout=artifacts.get(name, name), stderr="")

    monkeypatch.setattr(package_runtime, "run_entrypoint", emit)

    def tool(argv, *, timeout):
        calls.append((argv[0], Path(argv[-1]).name, threading.get_ident()))
        output = Path(argv[-1])
        output.write_bytes(str(argv).encode())
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(DB, "_run_build_tool", tool)

    def build(names=("a", "b", "c", "d"), **kwargs):
        return DB.build_device_objects(
            "synthetic",
            {s: (2, 2, 2) for s in names},
            {s: ("i8", "i8", "i32") for s in names},
            package_dir=tmp_path,
            workdir=tmp_path / "build",
            operand_dtype="i8",
            accum_dtype="i32",
            **kwargs,
        )

    return build, tool, emitted, calls, artifacts, owner


def test_independent_tools_overlap_and_emission_stays_on_original_owner(synthetic_build, monkeypatch):
    build, tool, emitted, calls, _, owner = synthetic_build
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = maximum = 0

    def concurrent(argv, *, timeout):
        nonlocal active, maximum
        if argv[0] == "clang" and Path(argv[-1]).name != "device_shim.o":
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait(timeout=3)
            result = tool(argv, timeout=timeout)
            with lock:
                active -= 1
            return result
        return tool(argv, timeout=timeout)

    monkeypatch.setattr(DB, "_run_build_tool", concurrent)
    result = build()
    assert result.ok and maximum == 2
    assert emitted == ["a", "b", "c", "d"]
    assert list(result.kernels) == emitted
    assert all(tid != owner for name, path, tid in calls if name == "translate" or path.endswith(".raw.o"))
    assert all(set(record) == {"translation", "compilation", "rename"} for record in result.custody.values())


def test_required_failure_stops_admission_but_drains_real_inflight_outputs(synthetic_build, monkeypatch):
    build, tool, emitted, calls, _, _ = synthetic_build
    barrier = threading.Barrier(2)
    failure_recorded = threading.Event()
    submit = jobs.ArtifactCompiler.submit

    def submitted(compiler, artifact, digest, callback):
        future = submit(compiler, artifact, digest, callback)
        if artifact.name == "a.device.mlir":
            future.add_done_callback(lambda _future: failure_recorded.set())
        return future

    def fail(argv, *, timeout):
        if argv[0] == "clang" and Path(argv[-1]).name.endswith(".raw.o"):
            barrier.wait(timeout=3)
            if Path(argv[-1]).name == "a.raw.o":
                return SimpleNamespace(returncode=124, stderr="timed out; incomplete output")
            assert failure_recorded.wait(3)
        return tool(argv, timeout=timeout)

    monkeypatch.setattr(jobs.ArtifactCompiler, "submit", submitted)
    monkeypatch.setattr(DB, "_run_build_tool", fail)
    result = build(stop_on_first_failure=True)
    assert emitted == ["a", "b"]
    assert result.skipped == (("a", "clang: timed out; incomplete output"),)
    assert set(result.kernels) == {"b"} and result.objects[0].is_file()
    assert result.shim_object is None
    assert not any(path == "device_shim.o" for _, path, _ in calls)


def test_slow_group_does_not_block_refill_of_other_compute_slot(synthetic_build, monkeypatch):
    build, tool, emitted, _, _, _ = synthetic_build
    replacement_started = threading.Event()

    def uneven(argv, *, timeout):
        name = Path(argv[-1]).name
        if argv[0] == "clang" and name == "a.raw.o":
            assert replacement_started.wait(3), "slow first group serialized unrelated later groups"
        if argv[0] == "clang" and name == "c.raw.o":
            replacement_started.set()
        return tool(argv, timeout=timeout)

    monkeypatch.setattr(DB, "_run_build_tool", uneven)
    result = build(names=("a", "b", "c"))
    assert result.ok and emitted == ["a", "b", "c"]
    assert list(result.kernels) == ["a", "b", "c"]
    assert [path.name for path in result.objects] == ["a.o", "b.o", "c.o", "device_shim.o"]


def test_pending_identical_artifacts_share_compile_but_keep_each_rename(synthetic_build):
    build, _, _, calls, artifacts, _ = synthetic_build
    artifacts.update(a="shared", b="shared", c="distinct")
    result = build(names=("a", "b", "c"))
    assert result.ok
    assert result.object_dedup == {"unique_artifacts": 2, "emitted_symbols": 3}
    assert Counter(name for name, _, _ in calls) == {"translate": 2, "clang": 3, "objcopy": 3}
    assert len(set(result.kernels.values())) == 3
    assert (result.objects[0].parent / "a.ll").read_bytes() == (result.objects[0].parent / "b.ll").read_bytes()


def test_every_tool_step_including_shim_owns_shared_compute_lease(synthetic_build, monkeypatch):
    from merlin.common import cpu_pool

    build, tool, _, calls, _, _ = synthetic_build
    local = threading.local()
    leases = []

    @contextmanager
    def lease(*, timeout_s):
        assert not getattr(local, "held", False)
        local.held = True
        leases.append(threading.get_ident())
        try:
            yield
        finally:
            local.held = False

    def guarded(argv, *, timeout):
        assert local.held
        return tool(argv, timeout=timeout)

    monkeypatch.setattr(cpu_pool, "shared_compute_pool", lease)
    monkeypatch.setattr(DB, "_run_build_tool", guarded)
    assert build(names=("a", "b")).ok
    assert len(leases) == len(calls) == 7


def test_worker_selection_uses_declared_pool_not_coordinator_affinity(monkeypatch):
    from merlin.common import cpu_pool

    monkeypatch.delenv("MERLIN_DEVICE_BUILD_JOBS", raising=False)
    monkeypatch.setattr(cpu_pool, "shared_compute_enabled", lambda: True)
    monkeypatch.setattr(cpu_pool, "grading_cpus", lambda: tuple(range(16, 31)))
    assert jobs.build_workers() == 15
    monkeypatch.setenv("MERLIN_DEVICE_BUILD_JOBS", "99")
    assert jobs.build_workers() == 15
    monkeypatch.setenv("MERLIN_DEVICE_BUILD_JOBS", "3")
    assert jobs.build_workers() == 3
    monkeypatch.setattr(cpu_pool, "shared_compute_enabled", lambda: False)
    monkeypatch.delenv("MERLIN_DEVICE_BUILD_JOBS")
    assert jobs.build_workers() == 1


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "", "two"])
def test_invalid_worker_selection_refuses(monkeypatch, value):
    monkeypatch.setenv("MERLIN_DEVICE_BUILD_JOBS", value)
    with pytest.raises(ValueError, match="positive integer"):
        jobs.build_workers()
