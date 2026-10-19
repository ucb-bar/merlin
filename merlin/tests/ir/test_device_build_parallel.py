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
    from merlin.targetgen import capsule_common, corpus_spec, package_runtime

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
    original_lower = capsule_common.lower_interface

    def lower(package, interface, generated, *, timeout, **kwargs):
        result = emit(package, "emit_target_artifact", interface, timeout=timeout)
        if result.returncode:
            raise ValueError(result.stderr)
        generated.mkdir(parents=True, exist_ok=True)
        (generated / "lowered.llvm.mlir").write_text(result.stdout)
        return {}, result.stdout

    monkeypatch.setattr(capsule_common, "lower_interface", lower)

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

    build.original_lower_interface = original_lower
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


def test_checked_emission_and_object_input_share_storage(synthetic_build):
    build, _, _, _, _, _ = synthetic_build
    result = build(names=("a",))
    work = result.objects[0].parent
    assert (work / "a.device.mlir").samefile(work / "a.lowering" / "lowered.llvm.mlir")


def test_full_model_replay_stops_after_first_required_emission_failure(synthetic_build, monkeypatch, tmp_path):
    from merlin.model_check import commands as C
    from merlin.model_check import diagnostics as DG
    from merlin.model_check import export as EX
    from merlin.model_check import replay as RP
    from merlin.targetgen import capsule_common

    _, _, emitted, calls, _, _ = synthetic_build
    original = capsule_common.lower_interface

    def reject_first(package, interface, generated, **kwargs):
        if interface.read_text() == "a":
            emitted.append("a")
            raise ValueError("first required interface could not be lowered")
        return original(package, interface, generated, **kwargs)

    monkeypatch.setattr(capsule_common, "lower_interface", reject_first)
    monkeypatch.setattr(RP, "toolchain_for", lambda _: SimpleNamespace(device_cflags=()))
    monkeypatch.setattr(C, "_kernel_identity", lambda *_: {})
    monkeypatch.setattr(EX, "describe", lambda *_: {"form": "synthetic"})
    monkeypatch.setattr(DG, "refusal", lambda *_: {})
    monkeypatch.setattr(DG, "localize", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        RP, "_write_image_sources", lambda *_args, **_kwargs: pytest.fail("failed build started replay")
    )
    sidecar = {
        "signatures": {s: (2, 2, 2) for s in ("a", "b")},
        "dtypes": {s: ("i8", "i8", "i32") for s in ("a", "b")},
        "entries": {s: {"name": s, "op": "matmul"} for s in ("a", "b")},
        "call_buffers": {s: [] for s in ("a", "b")},
    }
    catalog = SimpleNamespace(
        target="synthetic",
        sidecar=sidecar,
        interfaces={s: s for s in ("a", "b")},
        routing={"operand_dtype": "i8", "accum_dtype": "i32", "numeric_policy": None},
    )
    args = SimpleNamespace(mode="full", no_cache=True, host_package=tmp_path, submission=tmp_path, engine="spike")
    result = C._replay_cases(args, catalog, [SimpleNamespace(symbol=s) for s in ("a", "b")], tmp_path / "full")
    assert emitted == ["a"] and not calls
    assert result["build"]["built"] == 0 and result["build"]["unattempted"] == 1
    assert [(row["symbol"], row["status"]) for row in result["kernels"]] == [("a", "refused"), ("b", "unfinished")]
    assert list(result["refused"]) == ["a"]


def test_declined_command_buffer_cannot_compile_an_emitted_stub(synthetic_build, monkeypatch):
    import json

    from merlin.targetgen import capsule_common, package_runtime

    build, _, _, calls, _, _ = synthetic_build
    monkeypatch.setattr(capsule_common, "lower_interface", build.original_lower_interface)
    package = SimpleNamespace(manifest={"commands": {"emit_analysis_bundle": {"argv": []}}})
    monkeypatch.setattr(package_runtime, "load_package", lambda _path: package)

    def invoke(_package, command, source, output=None, **kwargs):
        if command == "emit_analysis_bundle":
            output.write_text(json.dumps({
                "abi_version": "0.1", "target": "synthetic", "tensors": {}, "commands": [],
                "params": {}, "declined": {"reason": "unsupported connected source"},
            }))
        return SimpleNamespace(returncode=0, stdout="module { llvm.func @kernel() { llvm.return } }", stderr="")

    monkeypatch.setattr(package_runtime, "run_entrypoint", invoke)
    result = build(names=("a",), stop_on_first_failure=True)
    assert not result.ok and "BackendDeclined" in result.skipped[0][1]
    assert not result.objects and not calls and result.shim_object is None


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


def test_static_budget_refuses_before_clang_and_retains_raw_products(synthetic_build, monkeypatch):
    import json

    from merlin.llvmlower.device_ir_scalability import DeviceIRBudget

    build, tool, _, calls, _, _ = synthetic_build
    def translate(argv, *, timeout):
        result = tool(argv, timeout=timeout)
        if argv[0] == "translate":
            Path(argv[-1]).write_text('define void @k() {\n'
                                     ' call void asm "x", ""()\n call void asm "y", ""()\n ret void\n}\n')
        return result
    monkeypatch.setattr(DB, "_run_build_tool", translate)
    result = build(names=("a",), stop_on_first_failure=True,
                   device_ir_budget=DeviceIRBudget(mode="enforce", max_inline_asm_sites=1))
    assert not result.ok and "device_ir_static_expansion_budget" in result.skipped[0][1]
    assert [name for name, _output, _thread in calls] == ["translate"]
    report = result.build_metrics["a"]
    assert not report["compiler_started"] and report["analysis"]["inline_asm_sites"] == 2
    path = Path(report["report_path"])
    assert json.loads(path.read_text())["decision"]["status"] == "refused"
    assert path.with_name("a.ll").is_file() and path.with_name("a.device.mlir").is_file()


def test_shared_artifact_analysis_runs_once_and_real_compilation_timings_are_retained(synthetic_build, monkeypatch):
    from merlin.llvmlower import device_ir_scalability as S

    build, _, _, _, artifacts, _ = synthetic_build
    artifacts.update(a="shared", b="shared")
    actual, calls = S.analyze_device_ir, []
    def measured(path):
        calls.append(path)
        return actual(path)
    monkeypatch.setattr(S, "analyze_device_ir", measured)
    result = build(names=("a", "b"))
    assert result.ok and len(calls) == 1
    for row in result.build_metrics.values():
        assert row["compiler_started"]
        assert all(value >= 0 for value in row["stages"].values())


def test_changed_llvm_after_analysis_refuses_before_process(synthetic_build, monkeypatch):
    from merlin.llvmlower import device_ir_scalability as S

    build, _, _, calls, _, _ = synthetic_build
    actual = S.inspect_device_ir
    def changed(path, budget):
        result = actual(path, budget)
        path.write_bytes(path.read_bytes() + b"changed bytes")
        return result
    monkeypatch.setattr(S, "inspect_device_ir", changed)
    result = build(names=("a",), stop_on_first_failure=True)
    assert not result.ok and "changed after" in result.skipped[0][1]
    assert [name for name, _output, _thread in calls] == ["translate"]
    assert result.build_metrics["a"]["decision"]["reason"] == "device_ir_changed_after_analysis"
