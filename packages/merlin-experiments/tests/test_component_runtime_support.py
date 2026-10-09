"""Actual private upstream controls refuse unknown physical runtime authority.

Preparation tests isolate the context's private input bookkeeping. They do not
mint RTL or runtime authority. Defect tests invoke real compiler subprocesses
through the shared lowering/binding path and inspect the actual refusal cause.
"""

from __future__ import annotations

import inspect
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_runtime_support as support
from merlin_experiments.phase2.component_runtime_qualification import RuntimeControlRefusal
from merlin_experiments.phase2.contracts import StageGateError, mapping_file

from merlin.common import invocation_record
from merlin.common.paths import data_path
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.execution_service import FunctionalExecutionService


def unused_runner(*_args, **_kwargs):
    raise AssertionError("defective source reached native execution before its ordinary gate")


def unused_renderer(_cb, *, inputs, readback_policy):
    raise AssertionError("defective source reached native building before its ordinary gate")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    context_type = support.PreparedIndependentRuntimeContext
    target = tmp_path / "target.yaml"
    target.write_text("target: private_control\n")
    shared = data_path("contract")
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    for name in ("manifest.schema.json", "capsule.schema.json", "command_buffer.schema.json"):
        shutil.copyfile(shared / "schemas" / name, contract / "schemas" / name)
    owner = Path(__file__).resolve()
    pins = ((str(owner), file_digest(owner)),)
    recipe = HarnessBuildRecipe(
        tmp_path / "unused-gcc",
        (),
        (),
        tmp_path / "unused.ld",
        0,
        kernel_stack_frame=KernelStackFramePolicy("control_entry", 1024),
    )
    builder = BuildOnlyService("private_control", recipe, unused_renderer, pins)
    execution = FunctionalExecutionService(
        "private_control",
        "private_control",
        unused_runner,
        unused_runner,
        pins,
        '{"scope":"private source controls only"}',
    )
    context = context_type(
        SimpleNamespace(sha256="not_an_intake"), target, contract, builder, execution, ((owner, file_digest(owner)),)
    )
    # Explicit private unit isolation, never a valid experiment prerequisite.
    monkeypatch.setattr(context_type, "verify", lambda _self: "unit_source_preparation_only")
    return context


@pytest.mark.parametrize(
    "case",
    [
        "source_correspondence.negative",
        "original_output_roster.negative",
        "original_numeric_gate.negative",
    ],
)
def test_actual_lowering_rejects_each_predeclared_source_defect(prepared, tmp_path, case, monkeypatch):
    from merlin.runtime.backends import base

    monkeypatch.setattr(base, "get_backend", lambda *_args: pytest.fail("private control discovered a backend"))
    fixture = prepared.prepare_control(case, tmp_path / "control")
    with pytest.raises(RuntimeControlRefusal) as rejected:
        prepared.services.grade(**fixture.grade_arguments)
    assert rejected.value.case_id == case
    assert rejected.value.mechanism == case.partition(".")[0]
    evidence = dict(rejected.value.evidence_files)
    assert evidence and all(file_digest(path) == digest for path, digest in evidence.items())
    records = [invocation_record.verify(path) for path in fixture.evidence_root.rglob("invocation.json")]
    assert any(row["kind"] == "subprocess" for row in records)
    assert set(fixture.required_invocation_stages) <= {row["stage"] for row in records}
    prepared.verify_control(fixture)


def test_original_numeric_gate_is_retained_separately(prepared, tmp_path):
    fixture = prepared.prepare_control("original_numeric_gate.negative", tmp_path / "control")
    original = mapping_file(fixture.evidence_root / "original_policy.json")
    actual = mapping_file(fixture.capsule_root / "capsule.yaml", yaml_file=True)["numeric_policy"]
    assert original["atol"] == 0.0 and actual["atol"] == 100.0
    assert "golden" not in (fixture.grade_arguments["package_dir"] / "driver.py").read_text().lower()


def test_mutable_fixture_arguments_and_source_changes_are_not_admitted(prepared, tmp_path):
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "control")
    fixture.grade_arguments["timeout"] = 1
    with pytest.raises(StageGateError, match="products changed"):
        prepared.verify_control(fixture)


@pytest.mark.parametrize(
    "mechanism", ["instruction_audit", "ownership_lifetime", "host_device_synchronization", "hardware_runtime_binding"]
)
def test_physical_controls_remain_unknown_without_independent_semantics(prepared, tmp_path, mechanism):
    with pytest.raises(StageGateError, match="remains UNKNOWN"):
        prepared.prepare_control(mechanism + ".positive", tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_numeric_pass_does_not_issue_a_physical_witness(prepared, tmp_path):
    result = tmp_path / "capsule_result.json"
    result.write_text('{"status":"pass","numeric":"pass"}')
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        prepared.stage_verifier(result_path=result)


def test_constructed_hardware_metadata_cannot_admit_runtime(tmp_path):
    context = support.PreparedIndependentRuntimeContext(
        SimpleNamespace(sha256="caller_hash"), tmp_path / "target.yaml", tmp_path / "contract", None, None, ()
    )
    with pytest.raises(StageGateError, match="live independently produced hardware intake"):
        context.verify()


def test_real_stock_structural_intake_reopens_complete_context_and_minimal_contract(tmp_path):
    selected = os.environ.get("MERLIN_TEST_FIRTOOL")
    if not selected:
        pytest.skip("requires explicitly selected stock FIRRTL producer")
    from merlin_experiments.phase0.rtl_intake import issue_independent_hardware_intake

    from merlin.targetgen.rtl import source_selection

    firrtl = tmp_path / "unit.fir"
    firrtl.write_text(
        "FIRRTL version 3.2.0\ncircuit Unit :\n"
        "  module Unit : @[generators/private_control/src/Unit.scala 1:1]\n"
        "    input clock : Clock\n"
        "    smem bank : UInt<8>[4] @[generators/private_control/src/Unit.scala 3:1]\n"
    )
    target = tmp_path / "target.yaml"
    target.write_text("target: private_control\n")
    bundle = source_selection.produce_selection(
        target="private_control",
        firrtl=firrtl,
        generator="private_control",
        config="TestConfiguration",
        core_root="Unit",
        firtool=Path(selected).resolve(strict=True),
        output=tmp_path / "production",
    )
    protected = tmp_path / "private-golden"
    protected.mkdir()
    intake = issue_independent_hardware_intake(
        target="private_control",
        descriptor=target,
        source_bundle=bundle,
        forbidden_roots=(protected,),
        output=tmp_path / "intake",
    )
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    shared = data_path("contract")
    for name in ("manifest.schema.json", "capsule.schema.json", "command_buffer.schema.json"):
        shutil.copyfile(shared / "schemas" / name, contract / "schemas" / name)
    owner = Path(__file__).resolve()
    service_pins = ((str(owner), file_digest(owner)),)
    recipe = HarnessBuildRecipe(
        tmp_path / "not-used-compiler",
        (),
        (),
        tmp_path / "not-used.ld",
        0,
        kernel_stack_frame=KernelStackFramePolicy("control_entry", 1024),
    )
    builder = BuildOnlyService("private_control", recipe, unused_renderer, service_pins)
    execution = FunctionalExecutionService(
        "private_control",
        "not-run",
        unused_runner,
        unused_runner,
        service_pins,
        '{"scope":"stock structural replay plus source control preparation only"}',
    )
    files = {owner, target, *contract.rglob("*.json")}
    files.update(
        Path(inspect.getsourcefile(value)).resolve()
        for value in (
            support.PreparedIndependentRuntimeContext,
            support.controls.parse_primitive,
            support.execute_component,
            support.PrivateRuntimeControlExecutor,
            support.prepare_source_control,
        )
    )
    context = support.PreparedIndependentRuntimeContext(
        intake,
        target,
        contract,
        builder,
        execution,
        tuple((path, file_digest(path)) for path in sorted(files)),
    )
    assert context.verify() == context.sha256
    readback_source = Path(support.RB.__file__).resolve()
    coherent = replace(
        context,
        readback_policy=support.RB.ReadbackPolicy(support.RB.COHERENT_DUMP_V1),
        memory_readback=SelectedMemoryObserver(),
        source_pins=(*context.source_pins, (readback_source, file_digest(readback_source))),
    )
    coherent_identity = coherent.verify()
    assert coherent_identity != context.verify()
    coherent.memory_readback.prepared_state["current_elf"] = "test data only"
    assert coherent.verify() == coherent_identity
    coherent.memory_readback.selection["selected_format"] += 1
    with pytest.raises(StageGateError, match="immutable selection changed"):
        coherent.verify()
    changed_recipe = replace(recipe, load_address=recipe.load_address + 1)
    changed_context = replace(context, build_service=replace(builder, recipe=changed_recipe))
    assert changed_context.verify() != context.verify()
    fixture = context.prepare_control("source_correspondence.negative", tmp_path / "control")
    context.verify_control(fixture)
    assert "simulator_equivalence" in intake.public_facts()["unknowns"]
    with pytest.raises(StageGateError, match="remains UNKNOWN"):
        context.prepare_control("hardware_runtime_binding.positive", tmp_path / "unknown")
    (contract / "development-answer.json").write_text("{}")
    with pytest.raises(StageGateError, match="only the three shared ABI schemas"):
        context.verify()


class SelectedMemoryObserver:
    """Test source selection only; this object issues no memory/runtime proof."""

    def __init__(self):
        self.selection = {"scope": "unit source selection only", "selected_format": 1}
        self.prepared_state = {}

    def prepare(self, **kwargs):
        raise AssertionError("observer selection tests must not prepare guest execution")

    def decode(self, console):
        raise AssertionError("observer selection tests must not decode guest output")

    def verify(self):
        assert self.selection["scope"] == "unit source selection only"

    def record(self):
        return dict(self.selection)


def replacement_decode(console):
    raise AssertionError("changed decoder must refuse before guest execution")


def coherent_context(prepared, **changes):
    return replace(
        prepared,
        readback_policy=support.RB.ReadbackPolicy(support.RB.COHERENT_DUMP_V1),
        memory_readback=SelectedMemoryObserver(),
        **changes,
    )


def test_explicit_coherent_readback_is_bound_without_runtime_authority(prepared):
    context = coherent_context(prepared)
    selected = context._readback_selection()
    assert selected["policy"] == support.RB.ReadbackPolicy(support.RB.COHERENT_DUMP_V1).record()
    assert [row[3] for row in selected["callbacks"]] == [
        "SelectedMemoryObserver.prepare",
        "SelectedMemoryObserver.decode",
        "SelectedMemoryObserver.verify",
        "SelectedMemoryObserver.record",
    ]
    assert prepared._readback_selection()["policy"]["transport"] == support.RB.FULL_VALUES_B64
    assert context.sha256 != prepared.sha256
    with pytest.raises(StageGateError, match="remains UNKNOWN"):
        context.prepare_control("hardware_runtime_binding.positive", Path(context.target_descriptor.parent / "unknown"))


@pytest.mark.parametrize(
    "policy,observer",
    [
        (support.RB.ReadbackPolicy(support.RB.FULL_VALUES_B64), SelectedMemoryObserver()),
        (support.RB.ReadbackPolicy(support.RB.COHERENT_DUMP_V1), None),
        (None, None),
    ],
)
def test_readback_selection_refuses_inconsistent_policy_observer(prepared, policy, observer):
    context = replace(prepared, readback_policy=policy, memory_readback=observer)
    with pytest.raises(StageGateError, match="policy|observer"):
        context._readback_selection()


@pytest.mark.parametrize("omit", ["execution", "context"])
def test_coherent_observer_requires_both_actual_source_memberships(prepared, omit):
    observer = SelectedMemoryObserver()
    changed = (
        {"execution_service": replace(prepared.execution_service, source_pins=())}
        if omit == "execution"
        else {"source_pins": ()}
    )
    context = replace(
        prepared,
        readback_policy=support.RB.ReadbackPolicy(support.RB.COHERENT_DUMP_V1),
        memory_readback=observer,
        **changed,
    )
    with pytest.raises(StageGateError, match="source membership"):
        context._readback_selection()


def test_coherent_observer_callback_substitution_refuses(prepared):
    context = coherent_context(prepared)
    context._readback_selection()
    context.memory_readback.decode = replacement_decode
    with pytest.raises(StageGateError, match="callbacks changed"):
        context._readback_selection()


def test_coherent_selection_drift_refuses_but_per_elf_state_does_not(prepared):
    context = coherent_context(prepared)
    original = context._readback_selection()
    context.memory_readback.prepared_state["current_elf"] = "unit preparation data only"
    assert context._readback_selection() == original
    context.memory_readback.selection["selected_format"] += 1
    with pytest.raises(StageGateError, match="immutable selection changed"):
        context._readback_selection()


def test_prepare_decode_only_reader_is_unsupported_without_stable_selection(prepared):
    class IncompleteObserver:
        prepare = SelectedMemoryObserver.prepare
        decode = SelectedMemoryObserver.decode

    with pytest.raises(StageGateError, match="inspected source owner"):
        replace(
            prepared,
            readback_policy=support.RB.ReadbackPolicy(support.RB.COHERENT_DUMP_V1),
            memory_readback=IncompleteObserver(),
        )


def test_grade_forwards_exact_selected_memory_observer(prepared, tmp_path, monkeypatch):
    context = coherent_context(prepared)
    fixture = context.prepare_control("source_correspondence.positive", tmp_path / "coherent-control")
    selected = []

    def capture(**arguments):
        selected.append(arguments)
        return {"numeric_report": {"status": "pass"}}

    # This test checks ordinary context forwarding only, not native output or a
    # runtime witness. Actual transport/build tests run with explicit stock tools.
    monkeypatch.setattr(support, "execute_component", capture)
    score = context.services.grade(**fixture.grade_arguments)
    assert score["per_capsule"][0]["numeric"] == "pass"
    assert len(selected) == 1
    assert selected[0]["readback_policy"] is context.readback_policy
    assert selected[0]["memory_readback"] is context.memory_readback
    assert selected[0]["execution_service"] is context.execution_service
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        context.stage_verifier(result_path=fixture.witness_arguments["result_path"])
