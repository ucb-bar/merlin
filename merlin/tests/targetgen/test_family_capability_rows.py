"""One unit's rows of a family stay distinct, and a reduction row can say which KIND it computes.

WHY THIS EXISTS. ``semantic_capability_map`` folded every same-family row into one capability:
fields union and ``composed_with`` intersects. A unit that runs max pooling only fused behind a
contraction, and SUM reductions standalone, therefore read as "standalone reductions of every kind":
the fold could not keep "standalone for sums, fused for max". The merged view keeps each declared
row (``SemanticCapability.alternatives``) and eligibility checks them any-of; a reduction row may
declare ``kinds`` (sum / max / absmax / other), checked against the kind observed in the region body.
A fold ACROSS units keeps the legacy union view, which other targets' verdicts rely on.
"""

from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.targetgen import compute_units as cu
from merlin.targetgen import eligibility as el
from merlin.targetgen.software_spec import admit_operation
from merlin.xdsl_dialects.lowering.contraction_coverage import reduction_kind


def _contract(*rows, units=1):
    return {
        "compute_units": [
            {"name": f"u{i}", "kind": "systolic", "dtypes": ["int8"], "semantic_capabilities": list(rows)}
            for i in range(units)
        ]
    }


_FUSED = {"family": "reduction", "dtypes": ["int8"], "composed_with": ["contraction", "movement"]}
_SUM = {"family": "reduction", "dtypes": ["int8"], "kinds": ["sum"]}


def _verdict(cmap, kind, fused_with=None):
    region = el.RegionDescriptor(op="reduce", family="reduction", in_dtype="int8", kind=kind)
    return el.is_eligible(region, cmap, fused_with=fused_with)


def test_standalone_sum_and_fused_max_coexist_on_one_unit():
    cmap = el.capability_map_from_contract(_contract(_FUSED, _SUM))
    assert len(cmap["reduction"].rows()) == 2
    assert _verdict(cmap, "sum").eligible
    standalone_max = _verdict(cmap, "max")
    assert not standalone_max.eligible and standalone_max.refusal in {"fused_only", "reduction_kind"}
    assert _verdict(cmap, "max", fused_with=("contraction",)).eligible
    unknown = _verdict(cmap, None)
    assert not unknown.eligible and unknown.undetermined and unknown.refusal == "reduction_kind_unknown"


def test_without_kinds_the_merged_family_is_standalone_for_every_kind():
    """The pre-existing reading, unchanged for a contract that declares no kinds."""
    unrestricted = {k: v for k, v in _SUM.items() if k != "kinds"}
    cmap = el.capability_map_from_contract(_contract(_FUSED, unrestricted))
    assert _verdict(cmap, "max").eligible and _verdict(cmap, None).eligible


def test_a_fold_across_units_keeps_the_union_view():
    cmap = el.capability_map_from_contract(_contract(_FUSED, _SUM, units=2))
    assert cmap["reduction"].alternatives == ()
    assert cmap["reduction"].kinds == ()  # the fused row restricts no kind, so the union does not either
    assert cmap["reduction"].composed_with == ()


def test_the_merged_rows_do_not_change_capability_equality():
    one = el.capability_map_from_contract(_contract(_FUSED, _SUM))["reduction"]
    two = el.capability_map_from_contract(_contract(_FUSED, _SUM))["reduction"]
    assert one == two and "alternatives" not in repr(one)


@pytest.mark.parametrize(
    "row",
    [
        {"family": "movement", "dtypes": ["int8"], "kinds": ["sum"]},
        {"family": "reduction", "dtypes": ["int8"], "kinds": ["mean"]},
        {"family": "reduction", "dtypes": ["int8"], "kinds": []},
    ],
)
def test_kinds_is_a_closed_reduction_vocabulary(row):
    with pytest.raises(ValueError, match="kinds"):
        cu.compute_units(_contract(row))


_SUM_GENERIC = """
module {
  func.func @f(%x: tensor<4x8xi8>, %o: tensor<4xi32>) -> tensor<4xi32> {
    %r = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
                         iterator_types = ["parallel", "reduction"]}
         ins(%x : tensor<4x8xi8>) outs(%o : tensor<4xi32>) {
    ^bb0(%a: i8, %c: i32):
      %w = arith.extsi %a : i8 to i32
      %s = arith.addi %c, %w : i32
      linalg.yield %s : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
}
"""
_MAX_REDUCE = """
module {
  func.func @f(%x: tensor<4x8xf32>, %o: tensor<4xf32>) -> tensor<4xf32> {
    %r = linalg.reduce ins(%x : tensor<4x8xf32>) outs(%o : tensor<4xf32>) dimensions = [1]
      (%a: f32, %c: f32) {
        %m = arith.maximumf %a, %c : f32
        linalg.yield %m : f32
      }
    return %r : tensor<4xf32>
  }
}
"""
_MAP = """
module {
  func.func @f(%x: tensor<4xi32>, %o: tensor<4xi32>) -> tensor<4xi32> {
    %r = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
                         iterator_types = ["parallel"]}
         ins(%x : tensor<4xi32>) outs(%o : tensor<4xi32>) {
    ^bb0(%a: i32, %c: i32):
      %s = arith.addi %a, %a : i32
      linalg.yield %s : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
}
"""


@pytest.mark.parametrize(
    "source, carrier, kind",
    [(_SUM_GENERIC, "linalg.generic", "sum"), (_MAX_REDUCE, "linalg.reduce", "max"), (_MAP, "linalg.generic", None)],
)
def test_the_reduction_kind_is_read_from_the_body(source, carrier, kind):
    with IR_LOCK:
        op = next(iter(mq.walk(mq.parse(source), carrier)))
        assert reduction_kind(op) == kind


def test_a_software_row_can_constrain_the_kind():
    spec = {
        "status": "reviewed",
        "operations": [
            {"id": "sums", "families": ["reduction"], "placement": "accelerator", "signature": {"kinds": ["sum"]}}
        ],
    }
    observed = {"family": "reduction"}
    assert admit_operation(spec, "aten.mean.dim", {**observed, "kind": "sum"}, "accelerator")["status"] == "admitted"
    assert admit_operation(spec, "aten.amax.default", {**observed, "kind": "max"}, "accelerator")["status"] == (
        "unsupported"
    )
    assert admit_operation(spec, "aten.mean.dim", observed, "accelerator")["status"] == "unknown"


@pytest.mark.target("gemmini")
def test_gemmini_working_tree_contract_is_sum_only_standalone():
    from merlin.common.paths import repo_root
    from merlin.common.yaml import load_yaml

    contract = load_yaml(repo_root() / "examples/gemmini/target/contracts/target_contract.yaml")
    rows = [
        row
        for unit in contract["compute_units"]
        for row in unit.get("semantic_capabilities") or ()
        if row["family"] == "reduction"
    ]
    if not any(row.get("kinds") for row in rows):
        pytest.skip("the selected contract declares no kind-restricted reduction row")
    cmap = el.capability_map_from_contract(contract)
    assert _verdict(cmap, "sum").eligible
    assert not _verdict(cmap, "max").eligible
    assert _verdict(cmap, "max", fused_with=("contraction",)).eligible
