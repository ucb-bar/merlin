"""An exact-op host row narrows admission only for the typed instance it declares.

WHY THIS EXISTS. ``admit_operation`` used to select every row whose ``ops`` named the observed
operation and consult the family rows only when none did. The gemmini software spec names
``aten.add.Tensor``, ``aten.relu.default``, ``aten.neg.default``, ``aten.sub.Tensor``,
``aten.mean.dim`` and ``aten._to_copy.default`` in reviewed f32/i64 HOST rows, so an int8 instance
of any of them was screened only against those rows and refused ("placement 'accelerator' differs
from declared 'host'; operand_dtype 'int8' not in ['fp32']") -- the accelerator family rows were
never reached. The same rule was copied into Phase-0 operation accounting, and a row whose id equals
a family name (``elementwise_map``) shadowed that family's other rows in a family-name query.

These tests run the working-tree gemmini spec resolved against FIXTURE facts (no facts bundle is
available offline), so they pin the screening rule, not the gemmini fact derivation.
"""

from __future__ import annotations

import copy

import pytest
from phase0_fixture_selection import fixture_resolved, real_selection

from merlin.common.paths import repo_root
from merlin.targetgen.operation_accounting import admit_operation_row
from merlin.targetgen.software_spec import admit_contraction_operation, admit_operation

pytestmark = pytest.mark.target("gemmini")

_BASE = repo_root() / "examples/gemmini/target"


@pytest.fixture(scope="module")
def resolved():
    return fixture_resolved(_BASE, "gemmini")


def _int8(family: str, **extra) -> dict:
    return {
        "family": family,
        "operand_dtype": "int8",
        "rank": 4,
        "layout": "row_major_contiguous",
        "aliasing": "disjoint_inputs_outputs",
        "scale_granularity": "tensor",
        **extra,
    }


@pytest.mark.parametrize(
    "op, family, extra",
    [
        ("aten.add.Tensor", "elementwise_map", {}),
        ("aten.add.Tensor", "operand_sum", {}),
        ("aten.sub.Tensor", "elementwise_map", {}),
        ("aten.relu.default", "elementwise_map", {}),
        ("aten.neg.default", "elementwise_map", {}),
        (
            "aten._to_copy.default",
            "elementwise_map",
            {"ordered_operand_dtypes": ["int8", "int32"], "ordered_result_dtypes": ["int32"]},
        ),
    ],
)
def test_int8_instance_falls_through_reviewed_host_rows_to_the_family(resolved, op, family, extra):
    decision = admit_operation(resolved, op, _int8(family, **extra), "accelerator")
    assert decision["status"] == "admitted", decision
    assert decision["declaration"] == "elementwise_map_standalone"
    assert decision["ignored_disjoint_declarations"], "the shadowing host rows are reported, not silently dropped"


def test_mean_dim_reaches_its_accelerator_families(resolved):
    window_mean = _int8("window_mean", accum_dtype="i32", tails="zero_pad_valid_window", broadcasting="none")
    decision = admit_operation(resolved, "aten.mean.dim", window_mean, "accelerator")
    assert decision["status"] == "admitted" and decision["declaration"] == "contraction"
    assert decision["ignored_disjoint_declarations"] == ["host_f32_reduction_ops"]
    # The standalone sum-reduction row is now reached instead of the host row: admitted once the
    # operator has reviewed it, unknown (review pending) before that -- never a host-row refusal.
    row = next(r for r in resolved["operations"] if r["id"] == "reduction_standalone")
    standalone = admit_operation(resolved, "aten.mean.dim", _int8("reduction", kind="sum"), "accelerator")
    expected = "admitted" if row.get("status", "reviewed") == "reviewed" else "unknown"
    assert standalone["status"] == expected and standalone["declaration"] == "reduction_standalone"


@pytest.mark.parametrize(
    "op, signature, declaration",
    [
        (
            "aten.add.Tensor",
            {"ordered_operand_dtypes": ["f32", "f32", "f32"], "ordered_result_dtypes": ["f32"], "rank": 1},
            "host_review_f32_add_r1",
        ),
        ("aten.relu.default", {"operand_dtype": "f32"}, "host_f32_elementwise"),
        ("aten.mean.dim", {"operand_dtype": "f32"}, "host_f32_reduction_ops"),
        (
            "aten.neg.default",
            {"ordered_operand_dtypes": ["f32", "f32"], "ordered_result_dtypes": ["f32"], "rank": 4},
            "host_f32_neg_rank4",
        ),
        (
            "aten._to_copy.default",
            {"ordered_operand_dtypes": ["i32", "f32"], "ordered_result_dtypes": ["f32"]},
            "host_cast",
        ),
    ],
)
def test_matching_host_rows_still_govern_f32_host_ops(resolved, op, signature, declaration):
    family = {"aten.mean.dim": "reduction"}.get(op, "elementwise_map")
    observed = {"family": family, **signature}
    host = admit_operation(resolved, op, observed, "host")
    assert host["status"] == "admitted" and host["declaration"] == declaration, host
    # The typed-compatible host row keeps its exact-roster priority: an f32 instance is not
    # re-screened against the int8 accelerator family rows.
    device = admit_operation(resolved, op, observed, "accelerator")
    assert device["status"] == "unsupported"
    assert device["declaration"] != "elementwise_map_standalone"


def test_int8_instance_is_not_admitted_on_the_host(resolved):
    decision = admit_operation(resolved, "aten.relu.default", _int8("elementwise_map"), "host")
    assert decision["status"] != "admitted"
    assert decision["declaration"] not in {"host_f32_elementwise", "elementwise_map_standalone"}


def test_unobserved_precision_keeps_exact_priority(resolved):
    # Nothing observed about precision: the host rows cannot be shown disjoint and stay in force.
    decision = admit_operation(resolved, "aten.relu.default", {"family": "elementwise_map", "rank": 4}, "accelerator")
    assert decision["status"] == "unsupported" and decision["declaration"] == "host_f32_elementwise"
    assert "ignored_disjoint_declarations" not in decision


def test_unreviewed_or_device_exact_rows_are_never_set_aside(resolved):
    spec = copy.deepcopy(resolved)
    for row in spec["operations"]:
        if row["id"] == "host_f32_neg_rank4":
            row["status"] = "unreviewed"
    decision = admit_operation(spec, "aten.neg.default", _int8("elementwise_map"), "accelerator")
    assert decision["status"] == "unsupported" and decision["declaration"] == "host_f32_neg_rank4"
    spec = copy.deepcopy(resolved)
    spec["operations"].append(
        {
            "id": "exact_relu_f32_device",
            "ops": ["aten.relu.default"],
            "placement": "accelerator",
            "signature": {"dtypes": ["f32"]},
        }
    )
    decision = admit_operation(spec, "aten.relu.default", _int8("elementwise_map"), "accelerator")
    assert decision["status"] == "unsupported" and decision["declaration"] == "exact_relu_f32_device"


def test_family_name_query_is_not_shadowed_by_a_row_named_after_the_family(resolved):
    # The fused row's id is `elementwise_map`; a family-name query must still reach the standalone row.
    decision = admit_operation(resolved, "elementwise_map", _int8("elementwise_map"), "accelerator")
    assert decision["status"] == "admitted" and decision["declaration"] == "elementwise_map_standalone"
    fused = admit_operation(
        resolved,
        "reduction",
        _int8("reduction", composed_with=["contraction"]),
        "fused_accelerator",
    )
    assert fused["status"] == "admitted" and fused["declaration"] == "pooling_on_readout"


def test_contraction_wrapper_reports_the_same_set_aside_rows(resolved):
    signature = dict(
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
    decision = admit_contraction_operation(resolved, "aten.bmm.default", signature, "accelerator")
    assert decision["status"] == "admitted"
    assert decision["ignored_disjoint_host_precision_declarations"] == ["host_f32_bmm"]
    assert decision["ignored_disjoint_declarations"] == ["host_f32_bmm"]


def test_phase0_accounting_uses_the_same_rule(resolved):
    row = {
        "operation": "aten.relu.default",
        "frontend_op": "aten.relu.default",
        "mlir_operation": "linalg.generic",
        "semantic_family": "elementwise_map",
        "disposition": "unclassified",
        "operand_format": "int8",
        "layout": "row_major_contiguous",
        "aliasing": "disjoint_inputs_outputs",
        "composed_observation": {"scale_granularity": "tensor"},
        "ordered_operand_types": [{"shape": [1, 2, 3, 4], "dtype": "i8"}, {"shape": [1, 2, 3, 4], "dtype": "i8"}],
        "ordered_result_types": [{"shape": [1, 2, 3, 4], "dtype": "i8"}],
    }
    result = admit_operation_row(row, software_spec=resolved, capability_contract=None)
    assert "host_f32_elementwise" not in result["matching_declarations"]
    admitted = [d for d in result["software_admissions"] if d["status"] == "admitted"]
    assert [d["declaration"] for d in admitted] == ["elementwise_map_standalone"]
    host = dict(row, operand_format="f32")
    host["ordered_operand_types"] = [dict(t, dtype="f32") for t in row["ordered_operand_types"]]
    host["ordered_result_types"] = [dict(t, dtype="f32") for t in row["ordered_result_types"]]
    result = admit_operation_row(host, software_spec=resolved, capability_contract=None)
    assert result["matching_declarations"] == ["host_f32_elementwise"]


_SHADOWED = [
    ("aten.add.Tensor", "elementwise_map", {}),
    ("aten.sub.Tensor", "elementwise_map", {}),
    ("aten.relu.default", "elementwise_map", {}),
    ("aten.neg.default", "elementwise_map", {}),
    ("aten._to_copy.default", "elementwise_map", {"ordered_operand_dtypes": ["int8", "int32"]}),
]


def test_the_same_fall_through_holds_against_a_selected_real_facts_bundle():
    """Run only when MERLIN_RTL_FACTS names a real bundle; then the production select_spec resolves it."""
    try:
        selection = real_selection(_BASE, "gemmini", repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")
    except (KeyError, LookupError, ValueError) as exc:
        pytest.skip(f"the selected bundle cannot be resolved here: {exc}")
    if selection is None:
        pytest.skip("MERLIN_RTL_FACTS names no facts bundle")
    for op, family, extra in _SHADOWED:
        decision = admit_operation(selection["spec"], op, _int8(family, **extra), "accelerator")
        assert decision["status"] == "admitted", (op, decision)
        assert decision["ignored_disjoint_declarations"]
