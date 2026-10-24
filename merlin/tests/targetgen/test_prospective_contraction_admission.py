"""Exact source-operation constraints survive prospective integer precision."""

import copy

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.core_aten_spec_int8 import source_declaration_continuity
from merlin.targetgen.software_spec import admit_contraction_operation, load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec


@pytest.fixture
def spec():
    return resolve_spec(load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml"), {})[0]


@pytest.fixture
def signature():
    return dict(
        family="contraction",
        ordered_operand_dtypes=["i8", "i8", "i32"],
        ordered_result_dtypes=["i32"],
        operand_dtype="i8",
        accum_dtype="i32",
        readout_dtype="i32",
        rank=3,
        layout="row_major_contiguous",
        tails="zero_pad_valid_window",
        broadcasting="independent_batches",
        aliasing="disjoint_inputs_outputs",
    )


def test_only_disjoint_reviewed_host_f32_row_is_ignored(spec, signature):
    before = copy.deepcopy(spec)
    result = admit_contraction_operation(spec, "aten.bmm.default", signature, "accelerator")
    assert result["status"] == "admitted" and result["declaration"] == "contraction"
    assert result["ignored_disjoint_host_precision_declarations"] == ["host_f32_bmm"]
    assert not result["applicable_exact_source_declaration"] and spec == before


@pytest.mark.parametrize(
    "kind",
    ["rank", "accumulator", "unreviewed", "unknown", "host_i8", "host_untyped", "device_f32", "unreviewed_host_f32"],
)
def test_exact_restrictions_cannot_fall_through_to_generic(spec, signature, kind):
    row = dict(
        id="exact_bmm",
        ops=["aten.bmm.default"],
        placement="accelerator",
        status="reviewed",
        signature=dict(ordered_operand_dtypes=["i8", "i8", "i32"], ordered_result_dtypes=["i32"]),
    )
    if kind == "rank":
        row["signature"]["ranks"] = [2]
    elif kind == "accumulator":
        row["signature"]["accumulator_dtype"] = "i64"
    elif kind == "unreviewed":
        row["status"] = "unreviewed"
    elif kind == "unknown":
        row["signature"]["restriction"] = "not independently proved"
    elif kind == "host_i8":
        row["placement"] = "host"
    elif kind == "host_untyped":
        row.update(placement="host", signature={})
    else:
        row["signature"].update(ordered_operand_dtypes=["f32", "f32", "f32"], ordered_result_dtypes=["f32"])
        if kind == "unreviewed_host_f32":
            row.update(placement="host", status="unreviewed")
    spec["operations"].append(row)
    result = admit_contraction_operation(spec, "aten.bmm.default", signature, "accelerator")
    assert result["status"] != "admitted" and result["declaration"] == "exact_bmm"
    assert result["applicable_exact_source_declaration"]
    assert "exact_bmm" not in result["ignored_disjoint_host_precision_declarations"]


def test_explicit_frontend_roster_stays_narrow(spec, signature):
    contraction = next(row for row in spec["operations"] if row["id"] == "contraction")
    contraction["ops"] = ["aten.mm.default"]
    assert admit_contraction_operation(spec, "aten.bmm.default", signature, "accelerator")["status"] == "unsupported"


@pytest.mark.parametrize(
    "exact,operator,expected",
    [
        (True, "aten._int_mm.default", "refused"),
        (False, "aten._int_mm.default", "preserved"),
        (True, "aten.mm.default", "preserved"),
    ],
)
def test_rewritten_overload_cannot_lose_exact_source_obligation(exact, operator, expected):
    initial = {
        "sites": [
            {
                "steps": [
                    {
                        "step": 3,
                        "source_operator": "aten.mm.default",
                        "software": {"applicable_exact_source_declaration": exact},
                    }
                ]
            }
        ]
    }
    captured = {"sites": [{"steps": [{"step": 3, "source_operator": operator}]}]}
    assert source_declaration_continuity(initial, captured)["status"] == expected
