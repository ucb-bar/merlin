"""Versioned source consumers preserve structural cuts and grant no mappings."""

import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import test_declared_phase0_run as DECLARATIONS
from merlin_experiments.phase0 import declared_run as D
from merlin_experiments.phase0 import packing_intake as P
from merlin_experiments.phase0.component_source_performance import SCHEMA as PERFORMANCE_SCHEMA
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal

from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
from merlin.targetgen.rtl.hw_partition_memory_bindings import PartitionMemoryLimits

declared = DECLARATIONS.declared


def selection():
    return {
        "schema": P.MEMORY_SELECTION_SCHEMA,
        "source_bytes": 1048576,
        "memory_limits": asdict(MemoryPortLimits(16, 16, 32, 1024, 64, 65536, 64)),
        "hierarchy_limits": asdict(HierarchyBindingLimits(16, 16, 16, 32, 256, 2048, 64, 65536, 16, 64)),
        "binding_limits": asdict(PartitionMemoryLimits(512, 128, 64, 128, 65536, 8192, 8192, 8192, 8192, 64)),
    }


def source():
    body = '%lo = "comb.extract"(%word) {lowBit=0:i32} : (i8) -> i4\n'
    body += '%hi = "comb.extract"(%word) {lowBit=4:i32} : (i8) -> i4\n'
    for index, value in enumerate(("lo", "hi")):
        body += (
            f'%m{index} = "seq.firmem"() {{name="m{index}",readLatency=0:i32,writeLatency=1:i32,'
            "ruw=0:i32,wuw=1:i32} : () -> !seq.firmem<3 x 4>\n"
            f'"seq.firmem.write_port"(%m{index},%index,%clock,%{value}) '
            "{operandSegmentSizes=array<i32:1,1,1,0,1,0>} : (!seq.firmem<3 x 4>,i2,!seq.clock,i4) -> ()\n"
            f'%r{index} = "seq.firmem.read_port"(%m{index},%index,%clock) '
            ": (!seq.firmem<3 x 4>,i2,!seq.clock) -> i4\n"
        )
    return (
        'builtin.module { "hw.module"() ({ ^bb0(%clock:!seq.clock,%index:i2,%word:i8):\n'
        + body
        + '"hw.output"(%r0,%r1) : (i4,i4) -> ()\n}) {sym_name="Unit",'
        "module_type=!hw.modty<input clock:!seq.clock,input index:i2,input word:i8,output a:i4,output b:i4>,"
        "parameters=[]} : () -> () }\n"
    )


def test_actual_typed_source_binding_retains_all_ports_and_undefined_memory_domains(tmp_path):
    path = tmp_path / "original.mlir"
    path.write_text(source())
    facts = P._memory_bindings(path, {"production": {"production": {"core_root": "Unit"}}}, selection())
    assert len(facts["ports"]) == 4
    assert [row["data_status"] for row in facts["ports"]].count("conditional_bit_relation") == 2
    assert [row["data_status"] for row in facts["ports"]].count("no_write_data_operand") == 2
    assert [
        (memory["declaration"]["depth"], memory["declaration"]["read_under_write"])
        for memory in facts["hierarchy"]["memories"]
    ] == [(3, "undefined"), (3, "undefined")]
    assert facts["source_values_evaluated"] is False
    assert facts["packing_mapping_admission"] is False
    assert facts["command_capacity_axis_or_temporal_admission"] is False


def test_selection_snapshot_does_not_alias_mutable_caller_budgets():
    original = selection()
    selected = P.validate_memory_selection(original)
    original["binding_limits"]["operations"] = 1
    assert selected["binding_limits"]["operations"] == 512


@pytest.mark.parametrize("kind", ["state_result", "memory_read_result", "opaque_instance_result", "unsupported_result"])
def test_complete_source_consumer_retains_state_read_opaque_and_unsupported_cuts(tmp_path, kind):
    text = source()
    if kind == "memory_read_result":
        text = text.replace("(%m1,%index,%clock,%hi)", "(%m1,%index,%clock,%r0)")
    else:
        if kind == "state_result":
            prefix = '%held = "seq.firreg"(%word,%clock) : (i8,!seq.clock) -> i8\n'
        elif kind == "unsupported_result":
            prefix = '%held = "comb.divu"(%word,%word) : (i8,i8) -> i8\n'
        else:
            text = text.replace(
                "builtin.module {",
                'builtin.module { "hw.module.extern"() '
                '{sym_name="Opaque",module_type=!hw.modty<output word:i8>,parameters=[]} : () -> ()\n',
            )
            prefix = '%held = "hw.instance"() {instanceName="absent",moduleName=@Opaque,'
            prefix += 'argNames=[],resultNames=["word"]} : () -> i8\n'
        text = text.replace("%lo =", prefix + "%lo =").replace('"comb.extract"(%word)', '"comb.extract"(%held)')
    path = tmp_path / "original.mlir"
    path.write_text(text)
    facts = P._memory_bindings(path, {"production": {"production": {"core_root": "Unit"}}}, selection())
    assert len(facts["ports"]) == 4
    expressions = {row["id"]: row for row in facts["hierarchy"]["expressions"]}
    assert kind in {expressions[identity]["kind"] for row in facts["ports"] for identity in row["cuts"]}
    assert facts["source_values_evaluated"] is facts["packing_mapping_admission"] is False


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "saved_facts",
        "root",
        "source_bool",
        "source_zero",
        "missing_limits",
        "missing_field",
        "extra_field",
        "bool_limit",
        "float_limit",
        "zero_limit",
    ],
)
def test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection(change):
    selected = selection()
    if change == "schema":
        selected["schema"] = P.SCHEMA
    elif change in {"saved_facts", "root"}:
        selected[change] = "candidate-selected-answer"
    elif change in {"source_bool", "source_zero"}:
        selected["source_bytes"] = True if change == "source_bool" else 0
    elif change == "missing_limits":
        del selected["memory_limits"]
    elif change == "missing_field":
        del selected["hierarchy_limits"]["port_bindings"]
    elif change == "extra_field":
        selected["binding_limits"]["axis"] = 1
    else:
        selected["binding_limits"]["operations"] = {"bool_limit": True, "float_limit": 512.0, "zero_limit": 0}[change]
    with pytest.raises(ValueError):
        P.validate_memory_selection(selected)


def test_source_budget_precedes_parser_and_geometry_work(tmp_path, monkeypatch):
    path = tmp_path / "original.mlir"
    path.write_text(source())
    selected = selection()
    selected["source_bytes"] = 1
    monkeypatch.setattr(P, "parse_generic_hw", lambda *args, **kwargs: pytest.fail("oversized source reached parser"))
    with pytest.raises(RtlIntakeRefusal, match="preparse"):
        P._memory_bindings(path, {}, selected)


def test_occurrence_budget_precedes_partition_expansion(tmp_path):
    path = tmp_path / "original.mlir"
    path.write_text(source())
    selected = selection()
    selected["binding_limits"]["operations"] = 1
    with pytest.raises(ValueError, match="operation budget"):
        P._memory_bindings(path, {"production": {"production": {"core_root": "Unit"}}}, selected)


def test_ordinary_declared_caller_passes_only_explicit_v2_snapshot(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(D, "issue_independent_packing_intake", lambda **kwargs: calls.append(kwargs))
    original = selection()
    hardware = object()
    arguments = {
        "hardware": hardware,
        "circt_opt": Path("/explicit/tool"),
        "forbidden_roots": (tmp_path / "excluded",),
        "output": tmp_path / "new",
    }
    D._issue_packing({"schema": D.PACKING_SCHEMA, "packing": original}, **arguments)
    original["memory_limits"]["nodes"] = 1
    assert calls[0]["memory_selection"]["memory_limits"]["nodes"] == 1024
    assert calls[0]["hardware"] is hardware
    D._issue_packing({"schema": D.REFERENCE_SCHEMA}, **arguments)
    assert "memory_selection" not in calls[1]


@pytest.mark.parametrize("change", [None, "missing", "legacy", "answer", "missing_reference", "missing_performance"])
def test_new_declared_selector_preserves_every_legacy_requirement(declared, change):
    request, _ = declared
    request = copy.deepcopy(request)
    request.update(
        schema=D.PACKING_SCHEMA,
        packing=selection(),
        release_purpose="performance_campaign",
        source_performance={
            "schema": PERFORMANCE_SCHEMA,
            "objectives": request["inputs"]["descriptor"],
            "sweeps": request["inputs"]["descriptor"],
        },
        original_references={
            "reference": request["inputs"]["descriptor"],
            "standard_ir": request["inputs"]["descriptor"],
        },
    )
    if change == "missing":
        del request["packing"]
    elif change == "legacy":
        request["schema"] = D.REFERENCE_SCHEMA
    elif change == "answer":
        request["packing"]["command_axis"] = "candidate-selected-role"
    elif change == "missing_reference":
        del request["original_references"]
    elif change == "missing_performance":
        del request["source_performance"]
    if change is None:
        assert D.validate(request) is request
    else:
        with pytest.raises(ValueError):
            D.validate(request)


@pytest.mark.parametrize("change", ["v1_extra", "v2_missing", "unknown_removed"])
def test_exported_record_version_and_unknown_membership_refuse_before_native_replay(change, monkeypatch):
    record = {
        "schema": P.SCHEMA,
        "hardware_intake_sha256": "a" * 64,
        "source_pins": [],
        "invocation": "/unavailable",
        "facts": {},
        "unknowns": list(P._UNKNOWN),
    }
    if change == "v1_extra":
        record["memory_bindings"] = {}
    elif change == "v2_missing":
        record["schema"] = P.MEMORY_SCHEMA
    else:
        record["unknowns"].pop()
    monkeypatch.setattr(
        P.I, "require_environment", lambda *args, **kwargs: pytest.fail("invalid record reached replay")
    )
    with pytest.raises(RtlIntakeRefusal, match="closed original record"):
        P.verify_record(copy.deepcopy(record))
