"""Admission derives evidence obligations from every selected capsule, before authoring."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import trace_readiness as R


@pytest.fixture
def selected(tmp_path):
    root = Path(__file__).resolve().parents[3]
    fixture = runpy.run_path(str(root / "merlin/tests/targetgen/test_selected_operand_protocol.py"))
    original, facts, blob, sidecar, *_ = fixture["selected"](tmp_path)
    return original, facts, blob, sidecar, fixture["load"]


def cap(name, **modes):
    return {"name": name, "kind": "isa", "expected": {"modes": modes}}


def test_simple_movement_probe_does_not_mask_missing_complex_mode_evidence(selected):
    original, *_ = selected
    contract = {"corpus_instruction_roles": {"by_role": {"memory": ["INPUT"], "acc_readout": ["OUTPUT"]}}}
    report = R.inspect(
        [
            cap("simple", movement=True),
            cap("convolution", conv2d=True, k_accumulate=True),
            cap("activation", relu=True),
            cap("retained", resident_reuse=True),
        ],
        semantics=original,
        contract=contract,
    )
    assert report["status"] == "blocked"
    assert {row["mode"] for row in report["blockers"]} == {"conv2d", "k_accumulate", "relu", "resident_reuse"}
    assert all(row["reason"] == "missing_selected_operand_protocol" for row in report["blockers"])
    assert report["requirements"]["movement"] == ["simple"]


def test_bound_selected_protocol_does_not_hide_unresolved_field_widths(selected):
    original, _facts, blob, sidecar, load = selected
    original.operand_observer = load(original, blob, sidecar)
    report = R.inspect([cap("activation", i8=True, relu=True)], semantics=original, contract={})
    assert report["status"] == "blocked"
    assert all(row["reason"] == "unresolved_selected_operand_fields" for row in report["blockers"])


def test_source_derived_observer_support_is_available_without_executing_a_candidate(selected):
    original, _facts, blob, sidecar, load = selected
    sidecar = copy.deepcopy(sidecar)
    sidecar["protocol"]["encoding"]["semantic_class"] = {4: "MVOUT"}
    sidecar["protocol"]["roles"]["classes"] = {
        "MVOUT": {"fields": [{"name": "readout", "operand": "rs2", "bundle": "Extent", "field": "rows"}]}
    }
    original.operand_observer = load(original, blob, sidecar)
    before = copy.deepcopy(original._facts)
    report = R.inspect([cap("output", i8=True)], semantics=original, contract={})
    assert report["status"] == "ready" and not report["blockers"]
    assert original._facts == before


def test_unknown_derived_fields_do_not_become_available_from_dependent_predicates():
    program = {
        "fields": [
            {"name": "unknown", "as": "int", "offset": None, "width": None},
            {"name": "predicate", "as": "equals", "from": "unknown"},
            {"name": "conjunction", "as": "all_equals", "predicates": [{"from": "predicate"}]},
        ]
    }
    assert R._observable_fields(program) == set()


def test_capsule_requirements_are_not_manually_selected_or_inferred_from_names(selected):
    original, *_ = selected
    report = R.inspect(
        [cap("new_case_unrelated_name", acc_scale=True), cap("irrelevant", relu=False)], semantics=original, contract={}
    )
    assert report["requirements"] == {"acc_scale": ["new_case_unrelated_name"]}
    assert report["blockers"][0]["capsules"] == ["new_case_unrelated_name"]


def test_other_backend_decoder_admission_is_unchanged():
    assert R.inspect([cap("example", i8=True)], semantics=None, contract={})["applicability"] == "legacy_decoder"


def test_unresolved_selected_load_identity_and_load_states_refuse(selected):
    original, _facts, blob, sidecar, load = selected
    original.operand_observer = load(original, blob, sidecar)
    report = R.inspect(
        [cap("retained", resident_reuse=True, load_state_resident=True)], semantics=original, contract={}
    )
    assert {row["reason"] for row in report["blockers"]} == {
        "unresolved_selected_load_identity",
        "missing_selected_load_state_evidence",
    }


def test_admission_checks_hidden_requirements_and_writes_blockers_before_authoring(selected, tmp_path, monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen import capsule_runner

    original, *_ = selected
    monkeypatch.setattr(
        base, "get_backend", lambda _target: SimpleNamespace(rocc_semantics=original, _data=lambda: {"contract": {}})
    )

    def discover(roots, *, labels, contract):
        return [cap("held_out", relu=True)] if labels == {"hidden"} else [cap("simple")]

    monkeypatch.setattr(capsule_runner, "discover_capsules", discover)
    report = tmp_path / "trace_readiness.json"
    with pytest.raises(ValueError, match="trace evidence is unavailable"):
        R.require_ready(tmp_path, target="array", contract=tmp_path, output=report, hidden_roots=tmp_path)
    data = json.loads(report.read_text())
    assert data["n_capsules"] == 2
    assert data["blockers"] == [
        {"mode": "relu", "reason": "missing_selected_operand_protocol", "capsules": ["held_out"]}
    ]
