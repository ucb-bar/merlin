"""Ordinary source/object custody controls, with native/proof edges stubbed.

The original archive/snapshot/import admission is real. The host class checker
returns unsupported or a deliberate refusal; these tests establish no math,
numerical, machine-code or hardware qualification.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

import pytest
import yaml
from test_lean_emission_chain import FINAL, _Executor
from test_lean_emission_chain import emission as emission  # noqa: F401

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import package_runtime as P
from merlin.targetgen.common_transforms import CommonTransformService
from merlin.targetgen.contract import compile as C
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V1, COMMON_SUBSTRATE_V3, attach_common_substrate
from merlin.verify import lean_emission_chain as E
from merlin.verify import lean_emission_context as X

_ORIGINAL_VERIFY = CommonTransformService._verify


def _owner(emission, monkeypatch, version=COMMON_SUBSTRATE_V3):
    monkeypatch.setattr(CommonTransformService, "_verify", _ORIGINAL_VERIFY)
    repo = repo_root()
    workspace = emission["tmp"] / "workspace"
    workspace.mkdir()
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(bundle, version=version, python_source_root=repo / "src", host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(repo / "src" / row["path"].removeprefix("merlin/python/"))
    archive = emission["tmp"] / "frozen.yaml"
    archive.write_text(yaml.safe_dump(bundle))
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    return CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo)


@pytest.fixture
def ordinary(emission, monkeypatch):
    from merlin.llvmlower import codegen, pipeline
    from merlin.targetgen import host_optimization_admission

    owner = _owner(emission, monkeypatch)

    # The actual original ABI selector is independently covered by its own
    # frozen-contract controls; this fixture supplies only the native stub ABI.
    def original_entry(admission, target, build_service):
        assert admission.workspace == owner.workspace and build_service is None and target is None
        admission.verify()
        return "fixture_entry", None

    monkeypatch.setattr(host_optimization_admission, "_original_entry", original_entry)
    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lambda *_a, **_k: "define void @fixture_entry() { ret void }\n")

    def object_stub(source, output, *_args, **_kwargs):
        output.write_bytes(b"native boundary stub; not executable")
        return output

    monkeypatch.setattr(codegen, "compile_ll", object_stub)
    return emission, owner


def _scope(owner):
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import original_compiler_imports

    return original_compiler_imports(owner)


def _lower(fixture, name="generated", **options):
    return CC.lower_interface(
        fixture["package"],
        fixture["original_source"],
        fixture["tmp"] / name,
        contract=fixture["contract"],
        timeout=30,
        **options,
    )


def _reports(root):
    return [json.loads(path.read_text()) for path in sorted(root.rglob("emission_proof_chain.json"))]


def test_ordinary_capsule_and_whole_program_route_bind_actual_object(ordinary):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        cb, artifact = _lower(fixture)
        assert cb["kernel_abi"]["kind"] == "whole_program"
        assert artifact == FINAL
        assert len(X.active_emission_checks().records) == 1
        obj = C.llvm_mlir_to_object(artifact, fixture["tmp"] / "build")
        assert obj.is_file()
        assert X.active_emission_checks().records == {}
    reports = _reports(fixture["tmp"] / "build")
    assert len(reports) == 1
    report = reports[0]
    assert report["status"] == "partial"
    assert report["emitted_artifact"]["sha256"] == report["actual_consumer_source"]["sha256"]
    assert report["numerics"] == report["machine_semantics"] == "not_established"
    assert report["object_product"]["path"] == str(obj)
    assert len(fixture["calls"]) == 1


def test_source_memo_cannot_replay_a_fresh_original_check(ordinary):
    fixture, owner = ordinary
    memo = {}
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _lower(fixture, "one", memo=memo)
        _lower(fixture, "two", memo=memo)
        assert len(fixture["calls"]) == 0  # No actual ABI consumer, no formal credit.
        assert memo == {}
        assert len(X.active_emission_checks().records) == 2
    assert X.active_emission_checks() is None


@pytest.mark.parametrize("status", ["refuted", "unknown", "unavailable"])
def test_covered_failed_edge_refuses_ordinary_compilation(ordinary, monkeypatch, status):
    fixture, owner = ordinary
    monkeypatch.setattr(CommonTransformService, "check", lambda *_a, **_k: {"status": status})
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture)
        with pytest.raises(ValueError, match="fresh original live"):
            C.llvm_mlir_to_object(artifact, fixture["tmp"] / "build")
        assert X.requires_fresh_object(artifact)
        assert len(X.active_emission_checks().records) == 1
        with pytest.raises(ValueError, match="incomplete or already claimed"):
            C.llvm_mlir_to_object(artifact, fixture["tmp"] / "retry")
    assert not _reports(fixture["tmp"] / "build")
    assert not (fixture["tmp"] / "build/kernel.o").exists()
    assert not (fixture["tmp"] / "retry/kernel.o").exists()


def test_wrong_final_trace_cannot_attest_another_emitted_artifact(ordinary):
    fixture, owner = ordinary
    trace = fixture["trace"]
    trace["checkpoints"][-1]["ir"] = FINAL + "\n"
    (fixture["package"].directory / "trace.json").write_text(json.dumps(trace))
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture)
        with pytest.raises(ValueError, match="last checkpoint"):
            C.llvm_mlir_to_object(artifact, fixture["tmp"] / "build")


def test_candidate_helper_entry_cannot_replace_original_dispatch_entry(ordinary):
    fixture, owner = ordinary
    artifact = FINAL + "\n"
    artifact = artifact.replace("} }", "} llvm.func @helper() { llvm.return } }")
    (fixture["package"].directory / "emitted.mlir").write_text(artifact)
    fixture["trace"]["checkpoints"][-1] = {"ir": artifact, "entry": "helper"}
    (fixture["package"].directory / "trace.json").write_text(json.dumps(fixture["trace"]))
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, actual = _lower(fixture)
        with pytest.raises(ValueError, match="last checkpoint"):
            C.llvm_mlir_to_object(actual, fixture["tmp"] / "build")


def test_absent_trace_remains_independent_not_formally_qualified(ordinary):
    fixture, owner = ordinary
    fixture["package"].manifest["commands"].pop("emit_transformation_trace")
    memo = {}
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture, memo=memo)
        _lower(fixture, "repeated", memo=memo)
        assert X.active_emission_checks().records == {}
        assert len(memo) == 1 and not X.requires_fresh_object(artifact)
        C.llvm_mlir_to_object(artifact, fixture["tmp"] / "build")
    assert _reports(fixture["tmp"] / "generated")[0]["status"] == "not_declared"
    assert _reports(fixture["tmp"] / "build")[0]["status"] == "not_matched"
    assert fixture["calls"] == []


def test_live_second_consumer_rechecks_its_source_but_saved_summary_cannot_restore_it(ordinary):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture)
        C.llvm_mlir_to_object(artifact, fixture["tmp"] / "one")
        C.llvm_mlir_to_object(artifact, fixture["tmp"] / "two")
        assert len(fixture["calls"]) == 2
        assert not X.requires_fresh_object(artifact) and X.requires_original_object(artifact)
    assert _reports(fixture["tmp"] / "one")[0]["status"] == "partial"
    assert _reports(fixture["tmp"] / "two")[0]["status"] == "partial"
    with _scope(owner), P.scoped_package_executor(_Executor()):
        with pytest.raises(ValueError, match="no current original host issuance"):
            C.llvm_mlir_to_object(artifact, fixture["tmp"] / "three")
    assert not _reports(fixture["tmp"] / "three")


def test_expired_copied_context_cannot_reuse_original_custody(ordinary):
    fixture, owner = ordinary
    original_live = set(E._LIVE)
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _lower(fixture)
        copied = copy_context()
        scope = X.active_emission_checks()
        assert scope.records
    assert not scope.live and not scope.records
    assert set(E._LIVE) == original_live
    with pytest.raises(ValueError, match="current original host|host-issued"):
        copied.run(X.active_emission_checks)
    with pytest.raises(ValueError, match="live original emission owner"):
        with X._operation_snapshot(scope):
            pytest.fail("expired operation owner admitted")


def test_concurrent_same_byte_consumers_cannot_choose_another_source_obligation(ordinary):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture, "source-one")
        _lower(fixture, "source-two")
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [
                pool.submit(copy_context().run, C.llvm_mlir_to_object, str(artifact), fixture["tmp"] / name)
                for name in ("build-one", "build-two")
            ]
            for job in jobs:
                with pytest.raises(ValueError, match="ambiguous original emission ownership"):
                    job.result()
        assert len(X.active_emission_checks().records) == 2
        assert fixture["calls"] == []
    for name in ("build-one", "build-two"):
        assert not (fixture["tmp"] / name / "kernel.o").exists()
        assert not _reports(fixture["tmp"] / name)


def test_same_byte_ambiguity_refuses_before_a_covered_failed_source_is_skipped(ordinary, monkeypatch):
    fixture, owner = ordinary
    monkeypatch.setattr(CommonTransformService, "check", lambda *_a, **_k: {"status": "refuted"})
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture, "source-one")
        # A distinct original input can yield the same emitted product. Its
        # source obligation cannot be selected using only that product hash.
        fixture["original_source"].write_text(fixture["original_source"].read_text() + "\n")
        _lower(fixture, "source-two")
        assert X.requires_fresh_object(artifact)
        with pytest.raises(ValueError, match="ambiguous original emission ownership"):
            C.llvm_mlir_to_object(str(artifact), fixture["tmp"] / "build")
        assert len(X.active_emission_checks().records) == 2
        assert fixture["calls"] == []
    assert not (fixture["tmp"] / "build/kernel.o").exists()


def test_native_object_failure_keeps_the_actual_emission_claim_pending(ordinary, monkeypatch):
    from merlin.llvmlower import codegen

    fixture, owner = ordinary

    def failed_native(*_args, **_kwargs):
        raise ValueError("native object failed")

    monkeypatch.setattr(codegen, "compile_ll", failed_native)
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture)
        with pytest.raises(ValueError, match="native object failed"):
            C.llvm_mlir_to_object(artifact, fixture["tmp"] / "build")
        assert X.requires_fresh_object(artifact)
        assert len(X.active_emission_checks().records) == 1
        with pytest.raises(ValueError, match="incomplete or already claimed"):
            C.llvm_mlir_to_object(artifact, fixture["tmp"] / "retry")
    assert not (fixture["tmp"] / "retry/kernel.o").exists()


def test_inventory_overflow_cleans_failed_and_remaining_live_records(ordinary, monkeypatch):
    fixture, owner = ordinary
    monkeypatch.setattr(X, "_MAX_PENDING", 1)
    original_live = set(E._LIVE)
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _lower(fixture, "one")
        with pytest.raises(ValueError, match="bounded live"):
            _lower(fixture, "two")
        assert len(X.active_emission_checks().records) == 1
    assert set(E._LIVE) == original_live


def test_existing_explicit_compile_only_chain_is_checked_once(ordinary, monkeypatch):
    from test_lean_emission_chain import test_connected_compile_source_only_and_reopen_same_host_selection

    fixture, owner = ordinary
    fixture["selection"] = E.EmissionChainSelection(owner)
    with _scope(owner):
        test_connected_compile_source_only_and_reopen_same_host_selection(fixture, monkeypatch)
        assert not X.active_emission_checks().records
    assert len(fixture["calls"]) == 1


def test_original_scope_never_replays_legacy_elf_cache(ordinary, monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen import build_cache

    fixture, owner = ordinary
    monkeypatch.setattr(base, "get_backend", lambda target: object())

    def legacy_replay(*args, **kwargs):
        pytest.fail("legacy ELF cache cannot issue fresh original emission/object custody")

    monkeypatch.setattr(build_cache, "reuse", legacy_replay)
    calls = []
    monkeypatch.setattr(
        C, "llvm_mlir_to_object", lambda source, work, **options: calls.append("object") or work / "kernel.o"
    )
    monkeypatch.setattr(
        C, "link_elf", lambda buffer, obj, work, **options: calls.append("link") or work / "program.elf"
    )
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture)
        assert X.requires_fresh_object(artifact)
        elf = C.compile_lowered_to_elf({}, artifact, fixture["tmp"] / "build", target="transport_fixture", inputs={})
    assert elf.name == "program.elf" and calls == ["object", "link"]


def test_no_owner_or_different_owner_cannot_select_checks(ordinary):
    fixture, owner = ordinary
    with pytest.raises(ValueError, match="live original API"):
        with X.original_emission_checks(owner):
            pass
    with owner.compiler_imports():
        owner.repo = Path("/unselected-repository")
        with pytest.raises(ValueError, match="different original owner"):
            with X.original_emission_checks(owner):
                pass


def test_scope_reopens_snapshot_once_and_refuses_same_size_content_change(ordinary, monkeypatch):
    import os

    from merlin.common import bundle_snapshot

    _, owner = ordinary
    source = Path(owner.bundle["allowed"][0]["path"])
    payload = bundle_snapshot.snapshot_path(bundle_snapshot.bundle_snapshot_root(owner.workspace), source, owner.repo)
    original_bytes = payload.read_bytes()
    original_stat = payload.stat()
    with _scope(owner):
        scope = X.active_emission_checks()
        admission_type = type(scope.admission)
        original_verify = admission_type.verify
        calls = []

        def verify(admission):
            calls.append(admission)
            return original_verify(admission)

        monkeypatch.setattr(admission_type, "verify", verify)
        scope.verify()
        assert calls == [scope.admission]
        calls.clear()
        try:
            payload.chmod(0o600)
            payload.write_bytes(bytes([original_bytes[0] ^ 1]) + original_bytes[1:])
            os.utime(payload, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            with pytest.raises(RuntimeError, match="content differs"):
                scope.verify()
            assert calls == [scope.admission]
        finally:
            payload.write_bytes(original_bytes)
            payload.chmod(original_stat.st_mode & 0o777)
            os.utime(payload, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))


def test_scope_refuses_a_different_live_admission_for_the_same_snapshot(ordinary):
    _, owner = ordinary
    with _scope(owner):
        scope = X.active_emission_checks()
        with owner.compiler_imports():
            with pytest.raises(ValueError, match="current original host admission"):
                scope.verify()
            with pytest.raises(ValueError, match="current original host admission"):
                with X._operation_snapshot(scope):
                    pytest.fail("foreign operation owner admitted")
        scope.verify()


def test_operation_batches_custody_reads_but_reopens_all_bytes_at_boundaries(ordinary, monkeypatch):
    from merlin.common import bundle_snapshot

    _, owner = ordinary
    payload = bundle_snapshot.snapshot_path(
        bundle_snapshot.bundle_snapshot_root(owner.workspace), Path(owner.bundle["allowed"][0]["path"]), owner.repo
    )
    with _scope(owner):
        scope = X.active_emission_checks()
        original = bundle_snapshot.hashlib.file_digest
        original_scan = bundle_snapshot._scanned_members
        calls, scans = [], []

        def digest(stream, *args):
            if Path(stream.name) == payload:
                calls.append(payload)
            return original(stream, *args)

        monkeypatch.setattr(bundle_snapshot.hashlib, "file_digest", digest)

        def scan(root):
            scans.append(root)
            return original_scan(root)

        monkeypatch.setattr(bundle_snapshot, "_scanned_members", scan)
        with X._operation_snapshot(scope):
            for _ in range(4):
                scope.verify()
            assert calls == [payload]
        assert calls == [payload, payload]
        # Independent entry/exit and original admission boundaries remain;
        # four pure custody guards no longer each scan the complete SDK roster.
        assert len(scans) == 5
        scope.verify()
        assert calls == [payload, payload, payload]
        assert len(scans) == 6


def test_copied_thread_and_ended_operation_never_reuse_admission(ordinary, monkeypatch):
    _, owner = ordinary
    with _scope(owner):
        scope = X.active_emission_checks()
        original, calls = type(scope.admission).verify, []

        def verify(admission):
            calls.append(admission)
            return original(admission)

        monkeypatch.setattr(type(scope.admission), "verify", verify)
        with X._operation_snapshot(scope):
            calls.clear()
            scope.verify()
            assert not calls
            copied = copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(copied.run, scope.verify).result()
            assert calls == [scope.admission]
            scope.verify()
            assert len(calls) == 1
        calls.clear()
        copied.run(scope.verify)
        assert calls == [scope.admission]


def test_independent_or_nested_actual_snapshot_never_reuses_pure_custody(ordinary, monkeypatch):
    from merlin.common import bundle_snapshot

    _, owner = ordinary
    with _scope(owner):
        scope = X.active_emission_checks()
        original, calls = type(scope.admission).verify, []

        def verify(admission):
            calls.append(admission)
            return original(admission)

        monkeypatch.setattr(type(scope.admission), "verify", verify)
        with X._operation_snapshot(scope):
            calls.clear()
            scope.verify()
            assert not calls
            with bundle_snapshot.independent_snapshot_reads():
                scope.verify()
            with bundle_snapshot.verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo):
                scope.verify()
            assert calls == [scope.admission, scope.admission]
            scope.verify()
            assert len(calls) == 2


def test_constructed_or_altered_thread_token_cannot_borrow_issuance(ordinary, monkeypatch):
    import os
    import threading

    _, owner = ordinary
    with _scope(owner):
        scope = X.active_emission_checks()
        original, calls = type(scope.admission).verify, []

        def verify(admission):
            calls.append(admission)
            return original(admission)

        monkeypatch.setattr(type(scope.admission), "verify", verify)
        forged = X._CustodyOperation(scope, os.getpid(), threading.current_thread())
        selected = X._ACTIVE_CUSTODY_OPERATION.set(forged)
        try:
            scope.verify()
            assert calls == [scope.admission]
        finally:
            X._ACTIVE_CUSTODY_OPERATION.reset(selected)
        with X._operation_snapshot(scope):
            issued = X._ACTIVE_CUSTODY_OPERATION.get()
            thread = issued.thread
            other = threading.Thread()
            # Reusing an OS TID/ident never makes this the actual issued Thread.
            other._ident = thread.ident
            object.__setattr__(issued, "thread", other)
            try:
                with pytest.raises(ValueError, match="original process or thread issuer"):
                    scope.verify()
            finally:
                object.__setattr__(issued, "thread", thread)


def test_copied_fork_cannot_reuse_original_operation(ordinary):
    import os

    _, owner = ordinary
    with _scope(owner):
        scope = X.active_emission_checks()
        with X._operation_snapshot(scope):
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                try:
                    scope.verify()
                except ValueError:
                    os.write(write_fd, b"refused")
                else:
                    os.write(write_fd, b"accepted")
                os._exit(0)
            os.close(write_fd)
            try:
                assert os.read(read_fd, 32) == b"refused"
                assert os.waitpid(pid, 0)[1] == 0
            finally:
                os.close(read_fd)


@pytest.mark.parametrize("change", ["admission", "product_owner", "archive"])
def test_operation_rejects_changed_live_owner_binding(ordinary, change):
    from merlin.verify.lean_emission_products import EmissionProducts

    _, owner = ordinary
    with _scope(owner):
        scope = X.active_emission_checks()
        with X._operation_snapshot(scope):
            if change == "admission":
                with owner.compiler_imports(), pytest.raises(ValueError, match="exact original admission"):
                    scope.verify()
            elif change == "product_owner":
                original = scope.products
                scope.products = EmissionProducts(1)
                try:
                    with pytest.raises(ValueError, match="product owner"):
                        scope.verify()
                finally:
                    scope.products = original
            else:
                original = owner.bundle_manifest.read_bytes()
                owner.bundle_manifest.write_bytes(original + b"\n")
                try:
                    with pytest.raises(ValueError, match="exact original admission"):
                        scope.verify()
                finally:
                    owner.bundle_manifest.write_bytes(original)


def test_exceptional_operation_closes_final_mutation_check_before_publication(ordinary):
    import os

    from merlin.common import bundle_snapshot

    fixture, owner = ordinary
    payload = bundle_snapshot.snapshot_path(
        bundle_snapshot.bundle_snapshot_root(owner.workspace), Path(owner.bundle["allowed"][0]["path"]), owner.repo
    )
    contents, metadata = payload.read_bytes(), payload.stat()
    accepted = fixture["tmp"] / "accepted"
    with _scope(owner):
        scope = X.active_emission_checks()
        try:
            with pytest.raises(RuntimeError, match="content differs"):
                with X._operation_snapshot(scope):
                    scope.verify()
                    payload.chmod(0o600)
                    payload.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
                    os.utime(payload, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
                    raise RuntimeError("original operation interrupted")
                accepted.write_bytes(b"accepted")
            assert not accepted.exists()
        finally:
            payload.write_bytes(contents)
            payload.chmod(metadata.st_mode & 0o777)


@pytest.mark.parametrize("operation", ["emission", "object"])
def test_changed_snapshot_cannot_publish_operation_completion(ordinary, monkeypatch, operation):
    from merlin.common import bundle_snapshot
    from merlin.llvmlower import codegen

    fixture, owner = ordinary
    payload = bundle_snapshot.snapshot_path(
        bundle_snapshot.bundle_snapshot_root(owner.workspace), Path(owner.bundle["allowed"][0]["path"]), owner.repo
    )
    original_bytes, original_mode = payload.read_bytes(), payload.stat().st_mode & 0o777

    def mutate():
        payload.chmod(0o600)
        payload.write_bytes(bytes([original_bytes[0] ^ 1]) + original_bytes[1:])

    with _scope(owner), P.scoped_package_executor(_Executor()):
        scope = X.active_emission_checks()
        if operation == "emission":
            run = _Executor.run_entrypoint

            def changed(self, package, name, *args, **kwargs):
                result = run(self, package, name, *args, **kwargs)
                if name == "emit_target_artifact":
                    mutate()
                return result

            monkeypatch.setattr(_Executor, "run_entrypoint", changed)
            invoke = lambda: _lower(fixture)
            output = fixture["tmp"] / "generated"
        else:
            _, artifact = _lower(fixture)
            compile_ll = codegen.compile_ll

            def changed(*args, **kwargs):
                result = compile_ll(*args, **kwargs)
                mutate()
                return result

            monkeypatch.setattr(codegen, "compile_ll", changed)
            output = fixture["tmp"] / "build"
            invoke = lambda: C.llvm_mlir_to_object(artifact, output)
        try:
            with pytest.raises(RuntimeError, match="content differs"):
                invoke()
            assert not _reports(output)
            assert scope.completed == {}
            stage = "ordinary_source_emission" if operation == "emission" else "ordinary_emission_object_consumer"
            observations = [
                json.loads(path.read_text())
                for path in output.rglob("invocation.json")
                if json.loads(path.read_text()).get("stage") == stage
            ]
            assert len(observations) == 1 and observations[0]["status"] == "interrupted"
            assert "outcome" not in observations[0]
            if operation == "emission":
                assert scope.records == {}
            else:
                assert len(scope.records) == 1
        finally:
            payload.write_bytes(original_bytes)
            payload.chmod(original_mode)


def test_actual_proof_and_native_consumers_keep_independent_snapshot_reads(ordinary, monkeypatch):
    from merlin.common import bundle_snapshot
    from merlin.llvmlower import codegen

    fixture, owner = ordinary
    check, compile_ll = CommonTransformService.check, codegen.compile_ll
    calls = []

    def independent_check(*args, **kwargs):
        assert bundle_snapshot._ACTIVE_VERIFICATION.get() is None
        calls.append("proof")
        return check(*args, **kwargs)

    def independent_compile(*args, **kwargs):
        assert bundle_snapshot._ACTIVE_VERIFICATION.get() is None
        calls.append("object")
        return compile_ll(*args, **kwargs)

    monkeypatch.setattr(CommonTransformService, "check", independent_check)
    monkeypatch.setattr(codegen, "compile_ll", independent_compile)
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, artifact = _lower(fixture)
        C.llvm_mlir_to_object(artifact, fixture["tmp"] / "build")
    assert calls == ["proof", "object"]
    assert bundle_snapshot._ACTIVE_VERIFICATION.get() is None


def test_unmatched_object_uses_one_fresh_consumer_boundary(ordinary, monkeypatch):
    from merlin.common import bundle_snapshot
    from merlin.llvmlower import codegen

    fixture, owner = ordinary
    payload = bundle_snapshot.snapshot_path(
        bundle_snapshot.bundle_snapshot_root(owner.workspace), Path(owner.bundle["allowed"][0]["path"]), owner.repo
    )
    original, compile_ll = bundle_snapshot.hashlib.file_digest, codegen.compile_ll
    reads = []

    def digest(stream, *args):
        if Path(stream.name) == payload:
            reads.append(stream.name)
        return original(stream, *args)

    def native(*args, **kwargs):
        assert bundle_snapshot._ACTIVE_VERIFICATION.get() is None
        return compile_ll(*args, **kwargs)

    with _scope(owner), P.scoped_package_executor(_Executor()):
        scope = X.active_emission_checks()
        monkeypatch.setattr(bundle_snapshot.hashlib, "file_digest", digest)
        monkeypatch.setattr(codegen, "compile_ll", native)
        obj = C.llvm_mlir_to_object(FINAL, fixture["tmp"] / "unmatched")
        assert obj.is_file() and len(reads) == 3  # Source selection + native entry/exit; previously five.
        assert scope.records == scope.completed == {}
        assert _reports(fixture["tmp"] / "unmatched")[0]["status"] == "not_matched"


def test_unmatched_native_internal_reads_stay_fresh(ordinary, monkeypatch):
    from merlin.common import bundle_snapshot
    from merlin.llvmlower import codegen

    fixture, owner = ordinary
    original = codegen.compile_ll
    entered = []

    def native(*args, **kwargs):
        assert bundle_snapshot._ACTIVE_VERIFICATION.get() is None
        for _ in range(2):
            document = bundle_snapshot.verify_bundle_snapshot(owner.workspace, owner.bundle, repo=owner.repo)
            entered.append(document["content_sha256"])
        return original(*args, **kwargs)

    monkeypatch.setattr(codegen, "compile_ll", native)
    with _scope(owner), P.scoped_package_executor(_Executor()):
        C.llvm_mlir_to_object(FINAL, fixture["tmp"] / "unmatched")
    assert entered == [owner.snapshot_digest] * 2


@pytest.mark.parametrize("mutation_at", ["consumer", "final_boundary"])
def test_unmatched_mutation_cannot_publish_return_or_report(ordinary, monkeypatch, mutation_at):
    import os
    from contextlib import contextmanager

    from merlin.common import bundle_snapshot
    from merlin.llvmlower import codegen

    fixture, owner = ordinary
    payload = bundle_snapshot.snapshot_path(
        bundle_snapshot.bundle_snapshot_root(owner.workspace), Path(owner.bundle["allowed"][0]["path"]), owner.repo
    )
    contents, metadata = payload.read_bytes(), payload.stat()
    original_compile, original_scope = codegen.compile_ll, X._operation_snapshot
    work = fixture["tmp"] / "unmatched"

    def mutate():
        payload.chmod(0o600)
        payload.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
        os.utime(payload, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))

    def native(*args, **kwargs):
        output = original_compile(*args, **kwargs)
        if mutation_at == "consumer":
            mutate()
        return output

    @contextmanager
    def boundary(scope):
        with original_scope(scope):
            yield
            if mutation_at == "final_boundary":
                mutate()

    monkeypatch.setattr(codegen, "compile_ll", native)
    monkeypatch.setattr(X, "_operation_snapshot", boundary)
    with _scope(owner), P.scoped_package_executor(_Executor()):
        scope = X.active_emission_checks()
        try:
            with pytest.raises(RuntimeError, match="content differs"):
                C.llvm_mlir_to_object(FINAL, work)
            assert not _reports(work) and scope.records == scope.completed == {}
            observations = [json.loads(path.read_bytes()) for path in work.rglob("invocation.json")]
            consumer = [row for row in observations if row.get("stage") == "ordinary_emission_object_consumer"]
            assert len(consumer) == 1 and consumer[0]["status"] == "interrupted"
            assert "outcome" not in consumer[0]
        finally:
            payload.write_bytes(contents)
            payload.chmod(metadata.st_mode & 0o777)


def test_historical_versioned_profile_keeps_lowering_memo_and_no_chain(emission, monkeypatch):
    owner = _owner(emission, monkeypatch, COMMON_SUBSTRATE_V1)
    memo = {}
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _lower(emission, "one", memo=memo)
        _lower(emission, "two", memo=memo)
        assert X.active_emission_checks() is None
    assert len(memo) == 1 and emission["calls"] == []
    assert not _reports(emission["tmp"])
