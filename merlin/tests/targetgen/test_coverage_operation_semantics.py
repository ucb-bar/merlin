"""Coverage uses an operation's own computation and observed result type."""

from merlin.common import mlir_query as mq
from merlin.targetgen import coverage_certificate as C
from merlin.targetgen.capsule_source import model_op_demands_checked
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.routing import OpDemand, RouteResult, route_plan_on
from merlin.xdsl_dialects._common import text as module_text

GATHER = """module {
  func.func @forward(%x: tensor<4x4xi8>) -> tensor<2x2xi8> {
    %e = "tensor.empty"() {prov.op = "convolution_im2col_matmul", prov.family = "contraction"}
      : () -> tensor<2x2xi8>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<4x4xi8>) outs(%e : tensor<2x2xi8>)
      attrs = {prov.op = "convolution_im2col_matmul", prov.family = "contraction", prov.region_id = "conv_0"} {
      ^bb0(%a: i8, %b: i8):
        linalg.yield %a : i8
    } -> tensor<2x2xi8>
    func.return %r : tensor<2x2xi8>
  }
}"""

CAST = """module {
  func.func @forward(%x: tensor<2xi8>) -> tensor<2xi32> {
    %e = "tensor.empty"() {prov.op = "dtype_cast", prov.family = "cast"} : () -> tensor<2xi32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x : tensor<2xi8>) outs(%e : tensor<2xi32>)
      attrs = {prov.op = "dtype_cast", prov.family = "cast", prov.region_id = "cast_0"} {
      ^bb0(%a: i8, %b: i32):
        %v = arith.extsi %a : i8 to i32
        linalg.yield %v : i32
    } -> tensor<2xi32>
    func.return %r : tensor<2xi32>
  }
}"""


def certificate(source, capabilities):
    source = module_text(mq.parse(source), generic=True)
    demands = model_op_demands_checked(source, "int8")
    return C.build(route_plan_on(demands, []), capabilities, linalg_mlir=source)


def test_nonreducing_gather_cannot_recover_the_enclosing_contraction_name():
    cap = {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))}
    cert = certificate(GATHER, cap)
    row = next(r for r in cert["regions"] if r["carrier_op"] == "linalg.generic")
    assert row["op"] == "convolution_im2col_matmul"  # Original provenance is preserved.
    assert row["semantic_family"] is None and row["target_eligible"] is False
    assert "unrecognized semantic family" in row["eligibility_reason"]
    assert cert["n_eligible"] == 0 and cert["n_unknown_capture_formats"] == 0
    assert cert["denominator_completeness"]["matched_contraction_macs"] == 0


def test_accelerating_a_nonreducing_region_tag_still_records_ineligible_acceleration():
    source = module_text(mq.parse(GATHER), generic=True)
    (demand,) = [d for d in model_op_demands_checked(source, "int8") if d.carrier_op == "linalg.generic"]
    result = RouteResult(demand, unit="unproved", acc=None, gap=None)
    cert = C.build(
        {"results": [result], "mesh": [result]},
        {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    assert cert["n_eligible"] == 0 and cert["accelerated_ineligible_count"] == 1


def test_direct_non_capture_family_inference_remains_available():
    result = RouteResult(OpDemand(op="matmul", in_fmt="int8"), unit=None, acc=None, gap=None)
    cert = C.build({"results": [result]}, {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))})
    assert cert["n_eligible"] == 1 and cert["regions"][0]["semantic_family"] == "contraction"


def test_cast_result_type_requires_independent_result_capability():
    family = "elementwise_map"
    unknown = certificate(CAST, {family: SemanticCapability(family=family, dtypes=("int8",))})
    row = next(r for r in unknown["regions"] if r["carrier_op"] == "linalg.generic")
    assert row["captured_result_format"] == "i32" and row["target_eligible"] is False
    assert row["eligibility_undetermined"] is True and "result" in row["eligibility_reason"]
    assert unknown["n_unknown_capture_formats"] == 0
    narrow = certificate(CAST, {family: SemanticCapability(family=family, dtypes=("int8",), result_dtypes=("int8",))})
    row = next(r for r in narrow["regions"] if r["carrier_op"] == "linalg.generic")
    assert row["target_eligible"] is False and row["eligibility_undetermined"] is False
    wide = certificate(CAST, {family: SemanticCapability(family=family, dtypes=("int8",), result_dtypes=("int32",))})
    assert wide["n_eligible"] == 1
