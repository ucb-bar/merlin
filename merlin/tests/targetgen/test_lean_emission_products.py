"""Original identity wiring; numerical/native/formal edges remain explicit stubs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
from test_lean_emission_chain import _Executor  # noqa: F401
from test_lean_emission_chain import emission as emission
from test_lean_emission_context import _lower, _reports, _scope  # noqa: F401
from test_lean_emission_context import ordinary as ordinary
from test_stack_frame_preflight import _recipe

from merlin.targetgen import package_runtime as P
from merlin.targetgen.common_transforms import CommonTransformService
from merlin.targetgen.contract import compile as C
from merlin.verify import lean_emission_context as X
from merlin.verify.lean_emission_products import EmissionProducts, IssuedEmission


def _second(fixture):
    source = fixture["tmp"] / "another-original.mlir"
    source.write_bytes(fixture["original_source"].read_bytes())
    return {**fixture, "original_source": source}


def test_distinct_same_byte_originals_compile_their_own_observed_source(ordinary):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, first = _lower(fixture, "one")
        _, second = _lower(_second(fixture), "two")
        assert first == second and first is not second
        assert X.artifact_cache_key(first) != X.artifact_cache_key(second)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(copy_context().run, C.llvm_mlir_to_object, source, fixture["tmp"] / name)
                for source, name in ((first, "a"), (second, "b"))
            ]
            assert all(future.result().is_file() for future in futures)
        assert not X.active_emission_checks().records
    reports = _reports(fixture["tmp"] / "a") + _reports(fixture["tmp"] / "b")
    assert len(reports) == 2 and all(report["status"] == "partial" for report in reports)
    assert len({report["compiler_source_invocation"]["path"] for report in reports}) == 2
    assert len({report["source"]["path"] for report in reports}) == 2


@pytest.mark.parametrize(
    "copy_text",
    [
        str,
        copy.copy,
        copy.deepcopy,
        lambda x: json.loads(json.dumps(x)),
        lambda x: pickle.loads(pickle.dumps(x)),
        IssuedEmission,
    ],
)
def test_copied_or_reconstructed_text_cannot_issue_a_source_identity(ordinary, copy_text):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, source = _lower(fixture)
        copied = copy_text(source)
        assert copied == source and copied is not source
        with pytest.raises(ValueError, match="issued emission identity|no current original host issuance"):
            C.llvm_mlir_to_object(copied, fixture["tmp"] / "unissued")
        assert len(X.active_emission_checks().records) == 1
    assert not (fixture["tmp"] / "unissued/kernel.o").exists()


def test_host_copy_recovers_exact_origin_but_an_arbitrary_copy_does_not(ordinary):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, first = _lower(fixture, "one")
        _, second = _lower(_second(fixture), "two")
        alias = fixture["tmp"] / "owned-copy.mlir"
        X.write_emitted_artifact(alias, second)
        assert X.read_emitted_artifact(alias) is second
        arbitrary = fixture["tmp"] / "arbitrary-copy.mlir"
        arbitrary.write_text(second)
        assert type(X.read_emitted_artifact(arbitrary)) is str
        with pytest.raises(ValueError, match="ambiguous original emission ownership"):
            C.llvm_mlir_to_object(X.read_emitted_artifact(arbitrary), fixture["tmp"] / "unbound")
        C.llvm_mlir_to_object(X.read_emitted_artifact(alias), fixture["tmp"] / "bound")
        assert X.requires_fresh_object(first) and not X.requires_fresh_object(second)
    assert not (fixture["tmp"] / "unbound/kernel.o").exists()


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "two_origins"])
def test_known_alias_cannot_change_or_choose_an_ambiguous_origin(ordinary, mutation):
    fixture, owner = ordinary
    alias = fixture["tmp"] / "owned-copy.mlir"
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, first = _lower(fixture, "one")
        _, second = _lower(_second(fixture), "two")
        X.write_emitted_artifact(alias, first)
        if mutation == "bytes":
            alias.write_text(first + "\n")
        elif mutation == "symlink":
            alias.unlink()
            alias.symlink_to(fixture["tmp"] / "one/lowered.llvm.mlir")
        else:
            # A contradictory raw registration has no observed replacement
            # write. Genuine host writes instead establish a new current owner.
            X.active_emission_checks().products.alias(second, X.E._pin(alias))
        with pytest.raises(ValueError, match="custody|changed|ambiguous"):
            X.read_emitted_artifact(alias)
        assert len(X.active_emission_checks().records) == 2


def test_equal_byte_host_rewrites_replace_or_remove_the_paths_original_owner(ordinary):
    fixture, owner = ordinary
    alias = fixture["tmp"] / "owned-copy.mlir"
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, first = _lower(fixture, "one")
        _, second = _lower(_second(fixture), "two")
        X.write_emitted_artifact(alias, first)
        linked_parent = fixture["tmp"] / "linked-parent"
        linked_parent.symlink_to(fixture["tmp"], target_is_directory=True)
        (fixture["tmp"] / "ordinary-parent").mkdir()
        for spelling in (linked_parent / alias.name, fixture["tmp"] / "ordinary-parent/.." / alias.name):
            with pytest.raises(ValueError, match="canonical plain original paths"):
                X.write_emitted_artifact(spelling, str(first))
            assert X.read_emitted_artifact(alias) is first
        X.write_emitted_artifact(alias, second)
        assert X.read_emitted_artifact(alias) is second
        X.write_emitted_artifact(alias, str(second))
        assert type(X.read_emitted_artifact(alias)) is str
        assert not X.is_emitted_artifact(alias)
        with pytest.raises(ValueError, match="ambiguous original emission ownership"):
            C.llvm_mlir_to_object(X.read_emitted_artifact(alias), fixture["tmp"] / "unowned")
        C.llvm_mlir_to_object(first, fixture["tmp"] / "first-object")
        C.llvm_mlir_to_object(second, fixture["tmp"] / "second-object")
        assert not X.active_emission_checks().records


def test_equal_byte_untraced_emission_cannot_recover_an_old_path_owner(ordinary):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, original = _lower(fixture)
        artifact_path = fixture["tmp"] / "generated/lowered.llvm.mlir"
        assert X.read_emitted_artifact(artifact_path) is original
        fixture["package"].manifest["commands"].pop("emit_transformation_trace")
        _, untraced = _lower(fixture)
        assert untraced == original and type(untraced) is str
        assert type(X.read_emitted_artifact(artifact_path)) is str
        assert not X.is_emitted_artifact(artifact_path)
        with pytest.raises(ValueError, match="changed its original package declaration"):
            C.llvm_mlir_to_object(original, fixture["tmp"] / "old-object")


def test_overlapping_observed_writers_cannot_restore_an_old_issued_alias(ordinary):
    fixture, owner = ordinary
    started, release = threading.Event(), threading.Event()
    artifact = fixture["tmp"] / "overlap/lowered.llvm.mlir"

    @X.observe_emission
    def emit(pkg, interface, generated, *, contract, timeout, artifact_name="lowered.llvm.mlir", memo=None):
        (generated / "input.interface.mlir").write_bytes(interface.read_bytes())
        (generated / "command_buffer.json").write_text("{}")
        (generated / artifact_name).write_text("module {}")
        started.set()
        assert release.wait(5)
        return {}, "module {}"

    with _scope(owner), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            copy_context().run,
            emit,
            fixture["package"],
            fixture["original_source"],
            artifact.parent,
            contract=fixture["contract"],
            timeout=30,
        )
        assert started.wait(5)
        try:
            X.write_emitted_artifact(artifact, "module {}")
        finally:
            release.set()
        with pytest.raises(ValueError, match="another observed writer replaced"):
            future.result()
        assert type(X.read_emitted_artifact(artifact)) is str
        assert not X.active_emission_checks().records


@pytest.mark.parametrize("same_directory", [True, False])
def test_actual_elf_oracle_route_rechecks_the_same_emission_at_each_tier(ordinary, monkeypatch, same_directory):
    from types import SimpleNamespace

    from merlin.llvmlower import codegen
    from merlin.runtime.backends import base
    from merlin.targetgen import build_cache

    fixture, owner = ordinary
    calls = []
    native = codegen.compile_ll

    def compile_native(*args, **kwargs):
        calls.append("object")
        return native(*args, **kwargs)

    def link(buffer, obj, work, **options):
        assert obj.is_file()
        calls.append("link")
        elf = work / "fixture.elf"
        elf.write_bytes(b"link boundary stub; not executable")
        return elf

    monkeypatch.setattr(codegen, "compile_ll", compile_native)
    monkeypatch.setattr(C, "link_elf", link)
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace())
    for name in ("reuse", "store"):
        monkeypatch.setattr(build_cache, name, lambda *_a, **_k: pytest.fail("issued origins cannot use byte caches"))
    with _scope(owner), P.scoped_package_executor(_Executor()):
        buffer, source = _lower(fixture)
        for tier in range(2):
            work = fixture["tmp"] / ("tiers" if same_directory else f"tier-{tier}")
            assert C.compile_lowered_to_elf(buffer, source, work, target=None).is_file()
        assert len(fixture["calls"]) == 2
        assert len(X.active_emission_checks().completed) == 1
    assert calls == ["object", "link", "object", "link"]
    reports = _reports(fixture["tmp"])
    objects = [report for report in reports if "object_product" in report]
    assert len(objects) == 2 and all(report["status"] == "partial" for report in objects)
    assert len({report["object_consumer_invocation"]["path"] for report in objects}) == 2


def test_a_refuted_second_consumer_cannot_reuse_a_prior_success(ordinary, monkeypatch):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, source = _lower(fixture)
        C.llvm_mlir_to_object(source, fixture["tmp"] / "one")
        monkeypatch.setattr(CommonTransformService, "check", lambda *_a, **_k: {"status": "refuted"})
        with pytest.raises(ValueError, match="fresh original live"):
            C.llvm_mlir_to_object(source, fixture["tmp"] / "two")
        with pytest.raises(ValueError, match="incomplete or already claimed"):
            C.llvm_mlir_to_object(source, fixture["tmp"] / "retry")
        assert X.requires_fresh_object(source) and X.requires_original_object(source)
        assert not X.active_emission_checks().completed
    assert not (fixture["tmp"] / "two/kernel.o").exists()


def test_partial_reply_two_pass_statement_uses_the_latest_observed_owner(ordinary, monkeypatch):
    """Actual state orchestration and emissions; graph/cache boundary is stubbed."""
    from types import SimpleNamespace

    from merlin.common import artifacts, mlir_query
    from merlin.llvmlower import whole_program
    from merlin.perf import whole_model_build as W
    from merlin.xdsl_dialects.lowering import stream_plan

    fixture, owner = ordinary
    manifest = fixture["tmp"] / "weights.json"
    manifest.write_text("{}")
    path = fixture["tmp"] / "group-copy.mlir"
    emitted, calls = [], []

    class AskerBoundary:
        def __init__(self, *args, **kwargs):
            self.missing = []

        def recorded(self):
            self.missing.append("another group is absent from the reply cache")

        def invoke(self):
            pass

        def close(self):
            calls.append("close")

    def statement(module, target, **options):
        options["run"]()
        calls.append(options["run"].__name__)
        _, product = _lower(fixture, "same-group-generated")
        emitted.append(product)
        X.write_emitted_artifact(path, product)
        return {"whole_program": {"artifact": str(path)}}

    monkeypatch.setattr(W, "Asker", AskerBoundary)
    monkeypatch.setattr(
        W,
        "_ReplyCache",
        lambda digest, directory: SimpleNamespace(
            directory=directory, package_digest=digest, hits=0, misses=0, cache_root=None
        ),
    )
    monkeypatch.setattr(W, "_package_digest", lambda directory: "cache-boundary")
    monkeypatch.setattr(P, "load_package", lambda directory: fixture["package"])
    monkeypatch.setattr(artifacts, "cache_dir", lambda name: fixture["tmp"] / "reply-cache")
    monkeypatch.setattr(mlir_query, "parse", lambda path: object())
    monkeypatch.setattr(stream_plan, "weight_args_beside", lambda path: ())
    monkeypatch.setattr(whole_program, "whole_program_buffer", statement)
    capsule = SimpleNamespace(interface=fixture["original_source"], weights_manifest=manifest, name="public_fixture")
    with _scope(owner), P.scoped_package_executor(_Executor()):
        buffer = W.state(
            capsule,
            target="transport_fixture",
            package_dir=fixture["package"].directory,
            work=fixture["tmp"] / "statement",
            jobs=2,
        )
        assert calls == ["recorded", "invoke", "close"]
        assert emitted[0] == emitted[1] and emitted[0] is not emitted[1]
        assert X.read_emitted_artifact(buffer["whole_program"]["artifact"]) is emitted[1]
        C.llvm_mlir_to_object(emitted[1], fixture["tmp"] / "latest-object")
        assert X.requires_fresh_object(emitted[0])
        C.llvm_mlir_to_object(emitted[0], fixture["tmp"] / "earlier-object")
        assert not X.active_emission_checks().records


def test_superseded_original_carrier_refuses_changed_source_artifact(ordinary):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, original = _lower(fixture)
        (fixture["package"].directory / "emitted.mlir").write_text(original + "\n")
        _, latest = _lower(fixture)
        assert X.read_emitted_artifact(fixture["tmp"] / "generated/lowered.llvm.mlir") is latest
        with pytest.raises(ValueError, match="changed|digest|bytes"):
            C.llvm_mlir_to_object(original, fixture["tmp"] / "superseded-object")


def test_other_same_byte_success_cannot_skip_this_sources_covered_failure(ordinary, monkeypatch):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, first = _lower(fixture, "one")
        _, second = _lower(_second(fixture), "two")
        C.llvm_mlir_to_object(first, fixture["tmp"] / "a")
        monkeypatch.setattr(CommonTransformService, "check", lambda *_a, **_k: {"status": "refuted"})
        assert X.requires_fresh_object(second)
        with pytest.raises(ValueError, match="fresh original live"):
            C.llvm_mlir_to_object(second, fixture["tmp"] / "b")
        assert X.requires_fresh_object(second)
        with pytest.raises(ValueError, match="incomplete or already claimed"):
            C.llvm_mlir_to_object(second, fixture["tmp"] / "retry")
    assert not (fixture["tmp"] / "b/kernel.o").exists()


def test_expired_text_cannot_enter_native_compilation_or_cache(ordinary):
    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, source = _lower(fixture)
    for action in (X.requires_fresh_object, X.artifact_cache_key):
        with pytest.raises(ValueError, match="live original scope"):
            action(source)
    with pytest.raises(ValueError, match="live original scope"):
        C.llvm_mlir_to_object(source, fixture["tmp"] / "expired")


@pytest.fixture
def model_boundary(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.llvmlower import toolchain
    from merlin.perf import whole_model_build as W
    from merlin.targetgen.contract import harness_abi

    monkeypatch.setattr(W, "_object_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(W, "_object_cache_fingerprint", lambda target: "boundary-stub")
    monkeypatch.setattr(W, "prune_object_cache", lambda path: None)
    monkeypatch.setattr(harness_abi, "for_target", lambda target: SimpleNamespace(entry_symbol="fixture_entry"))
    monkeypatch.setattr(toolchain, "objcopy", lambda: "fixture-objcopy")
    monkeypatch.setattr(toolchain, "nm", lambda: "fixture-nm")
    original_process = subprocess.run

    def process(argv, **options):
        from pathlib import Path

        if argv[0] == "fixture-objcopy":
            destination = Path(argv[-1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(argv[-2], destination)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "fixture-nm":
            return subprocess.CompletedProcess(argv, 0, f"00 T fixture_entry_{Path(argv[-1]).stem}\n", "")
        return original_process(argv, **options)

    monkeypatch.setattr(W.subprocess, "run", process)


def test_whole_model_workers_keep_distinct_origins_and_share_only_same_origin(ordinary, model_boundary, monkeypatch):
    from merlin.llvmlower import codegen
    from merlin.perf import whole_model_build as W
    from merlin.perf import whole_model_object_cache as O
    from merlin.runtime.backends import base
    from merlin.targetgen import host_optimization_admission as H

    fixture, owner = ordinary

    def select_entry(admission, target, build_service):
        admission.verify()
        assert admission.workspace == owner.workspace and target == "transport_fixture" and build_service is None
        return "fixture_entry", None

    calls = []

    def native_stub(source, output, *_args, **_kwargs):
        calls.append(output.parent.name)
        output.write_bytes(b"native boundary stub; not executable: " + output.parent.name.encode())
        output.with_suffix(".su").write_text(f"{source}:fixture_entry\t0\tstatic\n")
        return output

    monkeypatch.setattr(H, "_original_entry", select_entry)
    monkeypatch.setattr(base, "harness_build_recipe", lambda target: _recipe())
    monkeypatch.setattr(codegen, "compile_ll", native_stub)
    monkeypatch.setattr(O, "load", lambda path: pytest.fail("an owned pending emission cannot use disk cache"))
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, first = _lower(fixture, "one")
        _, second = _lower(_second(fixture), "two")
        first_path, second_path = fixture["tmp"] / "first-copy.mlir", fixture["tmp"] / "second-copy.mlir"
        X.write_emitted_artifact(first_path, first)
        X.write_emitted_artifact(second_path, second)
        rows = [
            {"group": group, "on": W.ON_PACKAGE, "artifact": str(path)}
            for group, path in ((1, first_path), (2, second_path), (3, first_path))
        ]
        result = W._kernel_objects(rows, target="transport_fixture", out=fixture["tmp"] / "objects", jobs=2)
        assert not X.active_emission_checks().records
    assert sorted(calls) == ["g1", "g2"]
    assert [row["object_cache"] for row in rows] == ["miss", "miss", "shared"]
    assert result == {"unique_signatures": 1, "total_groups": 3}
    assert (fixture["tmp"] / "objects/g1.o").read_bytes() == (fixture["tmp"] / "objects/g3.o").read_bytes()
    assert (fixture["tmp"] / "objects/g1.o").read_bytes() != (fixture["tmp"] / "objects/g2.o").read_bytes()


def test_whole_model_cannot_hide_changed_owned_artifact_as_vendor_fallback(ordinary, model_boundary):
    from merlin.perf import whole_model_build as W

    fixture, owner = ordinary
    with _scope(owner), P.scoped_package_executor(_Executor()):
        _, source = _lower(fixture)
        path = fixture["tmp"] / "owned-copy.mlir"
        X.write_emitted_artifact(path, source)
        path.write_text(source + "\n")
        rows = [{"group": 1, "on": W.ON_PACKAGE, "artifact": str(path)}]
        with pytest.raises(W.WholeModelBuildError, match="traced object production did not complete"):
            W._kernel_objects(rows, target="transport_fixture", out=fixture["tmp"] / "objects", jobs=1)
        assert rows[0]["on"] == W.ON_PACKAGE


def test_product_limits_do_not_create_unbounded_empty_alias_rows(tmp_path):
    source = "original bytes"
    pin = {"path": str(tmp_path / "a"), "sha256": hashlib.sha256(source.encode()).hexdigest()}
    products = EmissionProducts(1)
    issued = products.issue(source, pin)
    for name in ("b", "c"):
        with pytest.raises(ValueError, match="bounded live inventory"):
            products.alias(issued, {**pin, "path": str(tmp_path / name)})
    assert len(products.aliases) == products.alias_count == 1
    constrained = EmissionProducts(2, maximum_bytes=1)
    with pytest.raises(ValueError, match="bounded retained memory"):
        constrained.issue(source, pin)
    assert not constrained.aliases and not constrained.issued and constrained.retained_bytes == 0
    products.close()
    assert not products.aliases and products.retained_bytes == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="actual original process-copy control requires fork")
def test_forked_registry_is_not_an_issued_host_process(tmp_path):
    products = EmissionProducts(1)
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            products.verify()
        except ValueError:
            os.write(writer, b"refused")
        finally:
            os.close(writer)
            os._exit(0)
    os.close(writer)
    try:
        assert os.read(reader, 16) == b"refused"
    finally:
        os.close(reader)
        assert os.waitpid(pid, 0)[1] == 0
