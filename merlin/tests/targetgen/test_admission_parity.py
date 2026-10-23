"""The three software-admission entry points give one answer for the same observed operation.

WHY THIS EXISTS. The exact-roster rule was implemented three times -- ``admit_operation``,
``admit_contraction_operation`` and Phase-0 ``operation_accounting._software_admissions`` -- and the
copies diverged: an int8 bmm was admitted by the contraction path and refused by the other two, so the
quantizer integerized a bmm that operation accounting reported as SW-unsupported. All three now use
``software_spec.exact_declarations``. This pins that they agree, on the 39 support-set operations and on
host-only typed instances, at both placements.
"""

from __future__ import annotations

import pytest
from phase0_fixture_selection import fixture_resolved

from merlin.common.paths import repo_root
from merlin.targetgen.operation_accounting import admit_operation_row
from merlin.targetgen.software_spec import admit_contraction_operation, admit_operation

pytestmark = pytest.mark.target("gemmini")
_BASE = repo_root() / "examples/gemmini/target"
C, E, R, M = "contraction", "elementwise_map", "reduction", "movement"
_SUPPORT = {
    "aten.mm.default": (C, 2), "aten.bmm.default": (C, 3), "aten.addmm.default": (C, 2),
    "aten.convolution.default": (C, 4), "aten.convolution_backward.default": (C, 4),
    "aten.relu.default": (E, 4), "aten.neg.default": (E, 4), "aten.mul.Scalar": (E, 4),
    "aten.div.Scalar": (E, 4), "aten.add.Tensor": (E, 4), "aten.sub.Tensor": (E, 4),
    "aten.add.Scalar": (E, 4), "aten.sub.Scalar": (E, 4), "aten._to_copy.default": (E, 4),
    "aten.mean.dim": (R, 4), "aten.avg_pool1d.default": (R, 3), "aten.avg_pool2d.default": (R, 4),
    "aten.avg_pool3d.default": (R, 5), "aten.adaptive_avg_pool1d.default": (R, 3),
    "aten._adaptive_avg_pool2d.default": (R, 4), "aten._adaptive_avg_pool3d.default": (R, 5),
    "aten.avg_pool2d_backward.default": (R, 4), "aten._adaptive_avg_pool2d_backward.default": (R, 4),
    "aten.col2im.default": (R, 3), "aten.clone.default": (M, 4), "aten.permute.default": (M, 4),
    "aten.cat.default": (M, 4), "aten.repeat.default": (M, 4), "aten.slice_scatter.default": (M, 4),
    "aten.select_scatter.default": (M, 4), "aten.fill.Scalar": (M, 4), "aten.full.default": (M, 4),
    "aten.full_like.default": (M, 4), "aten.constant_pad_nd.default": (M, 4),
    "aten.reflection_pad1d.default": (M, 3), "aten.reflection_pad2d.default": (M, 4),
    "aten.reflection_pad3d.default": (M, 5), "aten.replication_pad2d.default": (M, 4),
    "aten.replication_pad3d.default": (M, 5),
}  # fmt: skip


def _int8(op: str) -> dict:
    family, rank = _SUPPORT[op]
    signature = {
        "family": family,
        "operand_dtype": "int8",
        "rank": rank,
        "layout": "row_major_contiguous",
        "aliasing": "disjoint_inputs_outputs",
        "scale_granularity": "tensor",
    }
    if family == C:
        signature.update(
            ordered_operand_dtypes=["int8", "int8", "int32"],
            ordered_result_dtypes=["int32"],
            accum_dtype="i32",
            readout_dtype="i32",
            tails="zero_pad_valid_window",
            broadcasting="independent_batches" if rank == 3 else "none",
        )
    elif op == "aten._to_copy.default":
        signature.update(ordered_operand_dtypes=["int8", "int32"], ordered_result_dtypes=["int32"])
    elif family == R:
        signature["kind"] = "sum"
    return signature


_HOST_ONLY = [
    ("aten.bmm.default", {"family": C, "ordered_operand_dtypes": ["f32", "f32", "f32"],
                          "ordered_result_dtypes": ["f32"], "operand_dtype": "f32", "accum_dtype": "f32", "rank": 3}),
    ("aten.add.Tensor", {"family": E, "ordered_operand_dtypes": ["f32", "f32", "f32"],
                         "ordered_result_dtypes": ["f32"], "rank": 1}),
    ("aten.relu.default", {"family": E, "operand_dtype": "f32", "rank": 4}),
    ("aten.mean.dim", {"family": R, "operand_dtype": "f32", "rank": 4}),
    ("aten._to_copy.default", {"family": E, "ordered_operand_dtypes": ["i32", "f32"],
                               "ordered_result_dtypes": ["f32"], "rank": 2}),
    ("aten.neg.default", {"family": E, "ordered_operand_dtypes": ["f32", "f32"],
                          "ordered_result_dtypes": ["f32"], "rank": 4}),
    ("quantized_decomposed.dequantize_per_tensor.default",
     {"family": E, "ordered_operand_dtypes": ["i8", "f32", "i64"], "ordered_result_dtypes": ["f32"],
      "operand_dtype": "i8", "rank": 3, "quantization_parameters": {"zero_point": 0}}),
]  # fmt: skip
_CASES = [(op, _int8(op)) for op in _SUPPORT] + _HOST_ONLY


@pytest.fixture(scope="module")
def resolved():
    return fixture_resolved(_BASE, "gemmini")


@pytest.mark.parametrize("op, signature", _CASES, ids=[f"{op}-{i}" for i, (op, _) in enumerate(_CASES)])
def test_the_three_entry_points_agree(resolved, op, signature):
    row = {"operation": op, "mlir_operation": "linalg.generic", "disposition": "unclassified"}
    accounted = admit_operation_row(row, software_spec=resolved, capability_contract=None, observed=signature)
    for placement in ("accelerator", "host"):
        direct = admit_operation(resolved, op, signature, placement)
        contraction = admit_contraction_operation(resolved, op, signature, placement)
        assert contraction["status"] == direct["status"]
        assert contraction.get("declaration") == direct.get("declaration")
        placements = {"accelerator", "fused_accelerator"} if placement == "accelerator" else {"host"}
        accounting_admitted = any(
            d["status"] == "admitted" and d["placement"] in placements for d in accounted["software_admissions"]
        )
        assert accounting_admitted == (direct["status"] == "admitted"), (placement, direct, accounted)


def test_the_int8_bmm_that_used_to_split_the_three_is_admitted_by_all(resolved):
    signature = _int8("aten.bmm.default")
    assert admit_operation(resolved, "aten.bmm.default", signature, "accelerator")["status"] == "admitted"
    row = {"operation": "aten.bmm.default", "mlir_operation": "linalg.batch_matmul", "disposition": "unclassified"}
    accounted = admit_operation_row(row, software_spec=resolved, capability_contract=None, observed=signature)
    assert [d["declaration"] for d in accounted["software_admissions"] if d["status"] == "admitted"] == ["contraction"]
