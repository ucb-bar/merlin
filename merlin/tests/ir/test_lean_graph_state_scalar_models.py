"""Actual changed chains and initialized nonzero/zero-trip/partial executions."""

import json

import pytest
from test_lean_graph_state_scalar import CASES, OPAQUE, REFERENCE, looping, produce

from merlin.verify import lean_graph_state as stable
from merlin.verify import lean_graph_state_scalar as checker
from merlin.verify.lean_graph_state_scalar_chain import emit_chain_query
from merlin.verify.lean_graph_state_scalar_kernel import check_query
from merlin.verify.lean_graph_state_scalar_models import emit_witness


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("cse-i8", [2, 3]),
        ("constants-casts", [200]),
        ("licm-nonunit", [2, 3]),
        ("licm-zero", [2, 3]),
        ("licm-two-carries", [2, 3]),
        ("licm-dynamic-bound", [7, 2, 3]),
        ("chain", [2, 3]),
        ("nested-chain", [2, 3]),
    ],
)
def test_actual_bound_initialized_success_model(name, arguments, tmp_path):
    mode, text = CASES[name]
    _, _, projection = produce(tmp_path, text, mode)
    assert projection["projection_checked"], projection
    query, roots, expected = emit_witness(projection, arguments)
    report = check_query(query, tmp_path / "model", roots)
    assert report["status"] == "conditional_graph_checked", report
    assert expected["status"] == "ok" and any(expected["values"])
    assert expected["world"][0] == 23
    stable._write(
        tmp_path / "model-interpretation.json",
        {
            **expected,
            "actual_primitive_fidelity_proved": False,
            "counterexamples_preserved": True,
            "numerical_gates_discharged": False,
            "kind": "explicit_initialized_Int_operational_model_only",
        },
    )


def test_typing_does_not_imply_partial_call_success(tmp_path):
    _, _, projection = produce(tmp_path, OPAQUE)
    failure = next(
        index for index, row in enumerate(projection["registry"]) if row["descriptor"]["name"] == "arith.divsi"
    )
    query, roots, expected = emit_witness(projection, [2, 0], failure=failure)
    report = check_query(query, tmp_path / "partial-model", roots)
    assert report["status"] == "conditional_graph_checked", report
    assert expected["status"] == "error" and expected["values"] == 17
    assert expected["world"][0] == 23 and len(expected["world"]) > 1
    assert not report["source_success_proved"]
    stable._write(
        tmp_path / "partial-model.json",
        {
            **expected,
            "kind": "typed_program_can_fail",
            "opaque_call_approval_issued": False,
            "actual_division_poison_semantics_proved": False,
        },
    )


@pytest.mark.parametrize("zero", [False, True])
def test_actual_two_changed_stages_and_distinct_composed_direct_proofs(zero, tmp_path):
    text = looping(upper=0 if zero else 7, observe=True, redundant_body=True)
    before, after, data = produce(tmp_path, text, "chain")
    assert data["projection_checked"] and data["stage_changed"] == [True, True]
    middle = tmp_path / "actual/middle.mlir"
    assert middle.read_bytes() == (tmp_path / "actual/stage-000.mlir").read_bytes()
    original = [stable.file_ref(path) for path in (before, middle, after)]
    result = checker.check_chain(
        before, middle, after, tmp_path / "chain", reference_root=REFERENCE, expected_originals=original
    )
    assert result["status"] == "conditional_graph_checked", result
    assert len(result["proof_routes"]) == 4 and result["stage_changed"] == [True, True]
    audit = json.loads((tmp_path / "chain/kernel/report.json").read_text())["audit"]
    for name in ("F0.first_actual", "F0.second_actual", "F0.middle_join", "F0.independently_composed", "F0.actual"):
        assert "Merlin.GraphStateScalarInstance." + name in audit["statements"]


@pytest.mark.parametrize("kind", ["join", "registry", "type-table", "source", "final"])
def test_actual_chain_checked_semantic_and_middle_joins_refuse_forgeries(kind, tmp_path):
    _, _, data = produce(tmp_path, CASES["chain"][1], "chain")
    data = json.loads(json.dumps(data))
    if kind == "join":
        data["chain_stages"]["second"]["functions"][0]["before"][-1]["ret"] = [1]
    elif kind == "registry":
        data["chain_stages"]["second"]["registry"].reverse()
    elif kind == "type-table":
        data["chain_stages"]["second"]["types"].reverse()
    elif kind == "source":
        data["chain_stages"]["first"]["functions"][0]["before"][-1]["ret"] = [1]
    else:
        data["chain_stages"]["second"]["functions"][0]["after"][-1]["ret"] = [1]
    with pytest.raises(ValueError):
        emit_chain_query(data)
