"""A bit-faithful add (fma plus NaN-payload selects) is a residual add exactly when that is provable."""

from __future__ import annotations

from fake_quant_layer import Oracle, residual_module

from merlin.common import mlir_query as mq
from merlin.xdsl_dialects.lowering import compute_groups as CG

_PLAIN = "      %r = arith.addf %p, %q0 : f32\n      linalg.yield %r : f32"


def _faithful(*, alpha: int = 1, lane_test: str = "arith.cmpi ult, %i, %c8 : index", lane_arm: str = "%s1") -> str:
    """The body a CPU-bit-faithful frontend emits for ``p + alpha*q0``: a fused lane path, a
    multiply-then-add tail path, each followed by the selects that choose a NaN payload."""
    return f"""      %one = arith.constant {alpha} : i64
      %alpha = arith.sitofp %one : i64 to f32
      %f = math.fma %q0, %alpha, %p : f32
      %nanbits = arith.constant -4194304 : i32
      %nan = arith.bitcast %nanbits : i32 to f32
      %fn = arith.cmpf uno, %f, %f : f32
      %s0 = arith.select %fn, %nan, %f : f32
      %quiet = arith.constant 4194304 : i32
      %pb = arith.bitcast %p : f32 to i32
      %pq = arith.ori %pb, %quiet : i32
      %pn = arith.bitcast %pq : i32 to f32
      %pt = arith.cmpf uno, %p, %p : f32
      %s1 = arith.select %pt, %pn, %s0 : f32
      %m = arith.mulf %q0, %alpha : f32
      %mt = arith.cmpf uno, %m, %m : f32
      %s2 = arith.select %mt, %nan, %m : f32
      %t = arith.addf %s2, %p : f32
      %tt = arith.cmpf uno, %t, %t : f32
      %s3 = arith.select %tt, %nan, %t : f32
      %i = linalg.index 1 : index
      %c8 = arith.constant 8 : index
      %lane = {lane_test}
      %r = arith.select %lane, {lane_arm}, %s3 : f32
      linalg.yield %r : f32"""


def _summing() -> Oracle:
    from merlin.targetgen import readout_facet as RF

    facet = RF.ReadoutFacet(
        target="synthetic",
        unit="unit0",
        accumulator_kind="addressable",
        scale_granularities=("tensor",),
        operand_sum={"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True},
    )
    return Oracle(readout=RF.TargetReadout((facet,)))


def _groups(body: str, *, scales=(1.0, 1.0, 1.0)):
    text = residual_module(lhs_scale=scales[0], rhs_scale=scales[1], out_scale=scales[2])
    assert _PLAIN in text
    return CG.form_groups(mq.parse(text.replace(_PLAIN, body, 1)), "synthetic", oracle=_summing())


def _add_of(text: str):
    return next(op for op in mq.parse(text).walk() if op.name == "linalg.generic")


def test_a_faithful_add_of_dequantized_operands_is_the_residual_add_it_computes() -> None:
    (plain,) = _groups(_PLAIN)
    (faithful,) = _groups(_faithful())
    assert faithful.stages == plain.stages == ["dequantize", "dequantize", "residual_add", "relu", "quantize"]
    assert faithful.placement == plain.placement == "unit0"
    assert faithful.operand_sum == plain.operand_sum


def test_a_faithful_add_whose_sum_is_not_provable_stays_elementwise() -> None:
    # alpha 2 is a scaled add, not the sum: nothing is claimed for it.
    (scaled,) = _groups(_faithful(alpha=2))
    assert "residual_add" not in scaled.stages and "elementwise" in scaled.stages
    assert scaled.placement == CG.HOST and scaled.operand_sum is None
    # Two equal sums chosen by lane are the sum; a choice on a compare that is no NaN test, between
    # the sum and something else, is not.
    (ordered,) = _groups(_faithful(lane_test="arith.cmpf olt, %p, %q0 : f32"))
    assert "residual_add" in ordered.stages
    (ordered,) = _groups(_faithful(lane_test="arith.cmpf olt, %p, %q0 : f32", lane_arm="%p"))
    assert "residual_add" not in ordered.stages and ordered.operand_sum is None
    # The same body over float model inputs: nothing proves them finite, so the NaN branches may be live.
    text = residual_module(lhs_scale=1.0, rhs_scale=1.0).replace(_PLAIN, _faithful(), 1)
    text = text.replace("%ad, %bd : tensor<4x16xf32>", "%sx, %sy : tensor<4x16xf32>", 1)
    text = text.replace(
        "func.func @forward(%a: tensor<4x16xi8>, %b: tensor<4x16xi8>)",
        "func.func @forward(%a: tensor<4x16xi8>, %b: tensor<4x16xi8>, %sx: tensor<4x16xf32>, %sy: tensor<4x16xf32>)",
        1,
    )
    from merlin.xdsl_dialects.lowering import faithful_sum as FS

    add = _add_of(text)
    assert all(getattr(v.owner, "name", None) != "quant_ext.dequantize_per_tensor" for v in add.inputs)
    assert FS.is_finite_input_unit_sum(add)  # the body IS the sum for finite inputs ...
    assert CG.classify(add).kind == CG.ELEMENTWISE  # ... and nothing proves these inputs finite
