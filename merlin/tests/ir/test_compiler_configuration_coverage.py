"""Original configuration wrappers and host coverage; no SDK/native proof claim."""

import copy
import json
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from nested_configuration_hoist_inputs import actual as nested_actual
from test_compiler_configuration import actual, selection
from test_compiler_configuration_hoist_wide import native_iset
from xdsl.dialects import builtin

from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer
from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.ir import Var
from merlin.transforms.certificates import ir_text
from merlin.verify import compiler_configuration as elision
from merlin.verify import compiler_configuration_hoist as hoist
from merlin.verify import compiler_configuration_source_profile as profile
from merlin.verify import pass_validation
from merlin.verify.compiler_profile import SourceOutsideProfile
from merlin.verify.configuration_elision_bridge import canonical, file_ref
from merlin.verify.gate_errors import GateUnavailable
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate


def validation(work, selector):
    # Exact host type, deliberately without invoking tool discovery/builds.
    tools = object.__new__(pass_validation.CheckerTools)
    object.__setattr__(tools, "sdk_bin", work / "never-SDK")
    object.__setattr__(tools, "compiler_python", Path(sys.executable))
    worker = SpeculativeGate(work / "worker", interrupt=False)
    return pass_validation.PassValidation(worker, tools, frozenset({selector}))


def report(owner):
    return json.loads((owner.gate.directory / "coverage.json").read_text())


def forbid_job(monkeypatch, owner):
    monkeypatch.setattr(owner.gate, "submit", lambda *a, **k: pytest.fail("source exclusion submitted a job"))


def fail_job(monkeypatch, owner, verdict="not_established"):
    def fail(check, **parameters):
        raise GateFailure(
            dict(
                check=check,
                label=parameters.get("label"),
                verdict=verdict,
                reason="host sentinel",
                directory=str(owner.gate.directory),
            )
        )

    monkeypatch.setattr(owner.gate, "submit", fail)


def floating_source(work):
    chosen, raw, _ = actual(work)
    source = chosen.parse(raw)
    source.body.block.first_op.attributes["tag"] = builtin.FloatAttr(0.0, builtin.f32)
    return chosen, source


def test_excluded_instruction_source_keeps_actual_ordinary_optimization(tmp_path, monkeypatch, caplog):
    chosen, source = floating_source(tmp_path)
    before = ir_text(source)
    ordinary = chosen.apply(source.clone())
    owner = validation(tmp_path, elision.SELECTOR)
    forbid_job(monkeypatch, owner)
    result = chosen.apply(source, pass_validation=owner)
    result.revalidate()
    assert ir_text(source) == before
    assert ir_text(result.module) == ir_text(ordinary.module) != before
    assert owner.gate.verdicts == []
    document = report(owner)
    assert document["compiler_completed"] is True
    assert document["verified_boundaries"] == 0 and document["unverified_boundaries"] == 1
    (row,) = document["boundaries"]
    assert row["status"] == "unverified" and row["code"] == "floating" and row["proof_credit"] is False
    assert "Unverified compiler boundaries" in caplog.text


def indexed_outside_source(work):
    chosen, source, _, adapter = nested_actual(work, "indexed")
    (outer,) = source.body
    (inner,) = outer.body
    setup, call = inner.body
    changed = replace(
        call,
        args=tuple((key, Var("i") * (1 << 32) + Var("j") * 2 + 1 if key == "a" else value) for key, value in call.args),
    )
    return chosen, replace(source, body=(replace(outer, body=(replace(inner, body=(setup, changed)),)),)), adapter


def test_excluded_hoist_keeps_changed_original_c64_owner_and_emits_text_only(tmp_path, monkeypatch):
    chosen, source, adapter = indexed_outside_source(tmp_path)
    before = source.text()
    ordinary = hoist_loop_configuration(source, adapter)
    consumer = ConfigurationCConsumer(64)
    owner = validation(tmp_path, hoist.SELECTOR)
    forbid_job(monkeypatch, owner)
    result = consumer.hoist(source, adapter, selected_contracts=chosen.selected_contracts, pass_validation=owner)
    assert source.text() == before
    assert result.kernel.text() == ordinary.kernel.text() != before
    text = result.emit_c(native_iset(), symbol="actual_kernel")
    assert "for (int64_t" in text
    assert owner.gate.verdicts == []
    document = report(owner)
    assert document["compiler_completed"] is True
    assert document["verified_boundaries"] == 0 and document["unverified_boundaries"] == 1
    (row,) = document["boundaries"]
    assert row["status"] == "unverified" and row["proof_credit"] is False
    with pytest.raises(ValueError, match="no original live"):
        copy.copy(result).emit_c(native_iset(), symbol="actual_kernel")


@pytest.mark.parametrize("verdict", ["not_established", "unsupported", "unavailable"])
def test_supported_instruction_gate_failure_never_becomes_source_exclusion(tmp_path, monkeypatch, verdict):
    chosen, raw, _ = actual(tmp_path)
    source = chosen.parse(raw)
    assert profile.admit_elision_source(raw, selection=chosen, representation="instruction_mlir") == profile.ELISION_ID
    owner = validation(tmp_path, elision.SELECTOR)
    fail_job(monkeypatch, owner, verdict)
    with pytest.raises(GateFailure) as error:
        chosen.apply(source, pass_validation=owner)
    assert error.value.report["verdict"] == verdict
    document = report(owner)
    assert document["compiler_completed"] is False and document["unverified_boundaries"] == 0


def test_supported_hoist_target_failure_stays_blocking(tmp_path, monkeypatch):
    chosen, source, _, adapter = nested_actual(tmp_path, "indexed")
    assert profile.admit_hoist_source(source, selection=chosen) == profile.HOIST_ID
    owner = validation(tmp_path, hoist.SELECTOR)
    fail_job(monkeypatch, owner)
    with pytest.raises(GateFailure):
        hoist_loop_configuration(source, adapter, selected_contracts=chosen.selected_contracts, pass_validation=owner)
    assert report(owner)["unverified_boundaries"] == 0


@pytest.mark.parametrize("kind", ["resource", "wrong-profile"])
def test_source_controller_never_downgrades_nonprofile_failures(tmp_path, monkeypatch, kind):
    chosen, source = floating_source(tmp_path)
    owner = validation(tmp_path, elision.SELECTOR)
    forbid_job(monkeypatch, owner)
    failure = (
        GateUnavailable("host budget sentinel") if kind == "resource" else SourceOutsideProfile("foreign", "x", "x")
    )

    def reject(*args, **kwargs):
        raise failure

    monkeypatch.setattr(profile, "admit_elision_source", reject)
    with pytest.raises(GateUnavailable if kind == "resource" else ValueError):
        chosen.apply(source, pass_validation=owner)
    assert report(owner)["compiler_completed"] is False
    assert report(owner)["unverified_boundaries"] == 0


def test_stale_original_selection_is_not_unverified_fallback(tmp_path, monkeypatch):
    chosen, source = floating_source(tmp_path)
    owner = validation(tmp_path, elision.SELECTOR)
    forbid_job(monkeypatch, owner)
    chosen.selected_contracts.facts_path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        chosen.apply(source, pass_validation=owner)
    assert owner.gate.verdicts == []


def test_supported_unchanged_instruction_has_zero_credit(tmp_path, monkeypatch):
    chosen = selection(tmp_path)
    from configuration_elision_qualification_cases import cases

    source = chosen.parse(next(case.source for case in cases() if case.name == "multifield"))
    source = chosen.apply(source).module
    owner = validation(tmp_path, elision.SELECTOR)
    forbid_job(monkeypatch, owner)
    result = chosen.apply(source, pass_validation=owner)
    assert ir_text(result.module) == ir_text(source)
    (row,) = report(owner)["boundaries"]
    assert row["status"] == "unchanged" and row["proof_credit"] is False


def replay_protocol(work, monkeypatch, module, *, during_check=None):
    """Host-only private ledger fixture; no kernel, accepted worker or proof credit."""
    paths = tuple(work / name for name in ("source.mlir", "target.mlir"))
    raw = ("original\r\nsource\r\n", "ordinary\r\noutput\r\n")
    for path, text in zip(paths, raw, strict=True):
        path.write_bytes(text.encode())
    rows = tuple(file_ref(path) for path in paths)
    schema = (
        "merlin.compiler-configuration.checked.v1"
        if module is elision
        else "merlin.compiler-configuration-hoist.checked.v1"
    )
    saved = work / "saved-host-locator.json"
    saved.write_text(
        canonical(
            dict(schema=schema, buffers=rows, source_request={}, candidate={"representation": "instruction_mlir"})
        )
    )
    monkeypatch.setattr(module, "ready", lambda candidate: None)
    if module is elision:
        monkeypatch.setattr(module, "reopen", lambda request: object())
    else:
        monkeypatch.setattr(module, "reopen_request", lambda request: (object(), None))
    seen = []

    def prepare(before, after, **kwargs):
        seen.append((before, after))
        return SimpleNamespace(source=before, final=after, identity="host-only-protocol-not-a-proof")

    def check(candidate, *, destination, **kwargs):
        destination.mkdir(parents=True)
        library = destination / "empty-host-library"
        library.mkdir()
        marker = destination / "host-marker"
        marker.write_text("host protocol fixture, no Lean invocation")
        row = file_ref(marker)
        report = dict(
            host_only=True,
            library_root=str(library),
            library=[],
            buffers=[],
            mapping_records=[],
            query=row,
            object=row,
            tool=row,
            auditor=row,
            imports=[],
            process_evidence=[],
        )
        report_path = destination / "REPORT.json"
        report_path.write_text(canonical(report))
        completion = module.Completion(candidate.identity, str(report_path))
        module._LIVE[completion] = (os.getpid(), threading.get_ident(), candidate, file_ref(report_path), report, ())
        if during_check is not None:
            during_check(paths)
        return completion

    monkeypatch.setattr(module, "prepare", prepare)
    monkeypatch.setattr(module, "check_candidate", check)
    return saved, paths, raw, seen


@pytest.mark.parametrize("module", [elision, hoist], ids=["elision", "hoist"])
def test_replay_preserves_exact_crlf_and_binds_delivery_to_originals(tmp_path, monkeypatch, module):
    saved, paths, raw, seen = replay_protocol(tmp_path, monkeypatch, module)
    completion, candidate = module.replay(saved, lean=tmp_path / "never-SDK", destination=tmp_path / "proof")
    assert seen == [raw] and module.deliver(completion, candidate) == raw[1]
    paths[0].write_bytes(b"mutated after replay\r\n")
    with pytest.raises(module.Unavailable):
        module.deliver(completion, candidate)
    with pytest.raises(module.Unavailable, match="no authority"):
        module.deliver(copy.copy(completion), candidate)


@pytest.mark.parametrize("module", [elision, hoist], ids=["elision", "hoist"])
@pytest.mark.parametrize("side", [0, 1], ids=["source", "target"])
def test_replay_independent_read_cannot_differ_from_just_checked_reference(tmp_path, monkeypatch, module, side):
    saved, paths, _, seen = replay_protocol(tmp_path, monkeypatch, module)
    original_read = Path.read_bytes
    count = 0

    def read(path):
        nonlocal count
        if path == paths[side]:
            count += 1
            if count == 2:
                return b"different bytes during independent raw read\r\n"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(module.Unavailable, match="differ"):
        module.replay(saved, lean=tmp_path / "never-SDK", destination=tmp_path / "proof")
    assert seen == []


@pytest.mark.parametrize("module", [elision, hoist], ids=["elision", "hoist"])
@pytest.mark.parametrize("side", [0, 1], ids=["source", "target"])
def test_replay_rechecks_originals_after_host_check(tmp_path, monkeypatch, module, side):
    def mutate(paths):
        paths[side].write_bytes(b"mutated during host check\r\n")

    saved, _, raw, seen = replay_protocol(tmp_path, monkeypatch, module, during_check=mutate)
    with pytest.raises(module.Unavailable):
        module.replay(saved, lean=tmp_path / "never-SDK", destination=tmp_path / "proof")
    assert seen == [raw]


def test_c64_replay_initial_read_is_bound_before_any_projection(tmp_path, monkeypatch):
    chosen, source, expected, _ = nested_actual(tmp_path, "inner")
    consumer = ConfigurationCConsumer(64)
    paths = tuple(tmp_path / name for name in ("before.mlir", "after.mlir"))
    for path, text in zip(paths, (source.text(), expected.kernel.text()), strict=True):
        path.write_bytes(text.encode())
    saved = tmp_path / "saved-c64-host-locator.json"
    saved.write_text(
        canonical(
            dict(
                schema="merlin.compiler-configuration-hoist.checked.v2",
                buffers=[file_ref(path) for path in paths],
                source_request={},
            )
        )
    )
    monkeypatch.setattr(hoist, "reopen_request", lambda request: (chosen, consumer))
    monkeypatch.setattr(hoist, "prepare", lambda *a, **k: pytest.fail("foreign replay bytes reached projection"))
    original_read = Path.read_bytes
    count = 0

    def read(path):
        nonlocal count
        if path == paths[0]:
            count += 1
            if count == 2:
                return b"foreign C64 source"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(hoist.Unavailable, match="differ"):
        consumer.replay(saved, lean=tmp_path / "never-SDK", destination=tmp_path / "proof")
