"""A derived output epilogue joins its stages to the contraction's group, and only exact ones."""

from __future__ import annotations

import pytest
from fake_quant_layer import Oracle as _Oracle
from fake_quant_layer import module as _module

from merlin.common import mlir_query as mq
from merlin.targetgen import readout_facet as RF
from merlin.xdsl_dialects.lowering import compute_groups as CG

_ONE_INPUT = """builtin.module {{
  func.func @forward(%x: tensor<4x16x{dtype}>) -> tensor<4x16x{dtype}> {{
    %e = tensor.empty() : tensor<4x16x{dtype}>
    %r = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}} ins(%x : tensor<4x16x{dtype}>) outs(%e : tensor<4x16x{dtype}>) {{
      ^bb0(%v: {dtype}, %o: {dtype}):
{body}
      }} -> tensor<4x16x{dtype}>
    func.return %r : tensor<4x16x{dtype}>
  }}
}}"""


def _stage(body: str, dtype: str = "f32") -> str:
    module = mq.parse(_ONE_INPUT.format(dtype=dtype, body=body))
    (op,) = [op for op in module.walk() if mq.op_name(op) == "linalg.generic"]
    return CG.classify(op).kind


_ZERO_F = "        %i = arith.constant 0 : i64\n        %zero = arith.sitofp %i : i64 to f32\n"


@pytest.mark.parametrize(
    "body",
    [
        _ZERO_F + "        %c = arith.cmpf olt, %v, %zero : f32\n        %s = arith.select %c, %zero, %v : f32",
        _ZERO_F + "        %c = arith.cmpf ogt, %v, %zero : f32\n        %s = arith.select %c, %v, %zero : f32",
        _ZERO_F + "        %c = arith.cmpf ogt, %zero, %v : f32\n        %s = arith.select %c, %zero, %v : f32",
    ],
)
def test_a_rectifier_written_as_compare_and_select_is_a_relu(body):
    assert _stage(body + "\n        linalg.yield %s : f32") == CG.RELU


def test_an_integer_rectifier_is_a_relu_too():
    body = (
        "        %zero = arith.constant 0 : i32\n        %c = arith.cmpi slt, %v, %zero : i32\n"
        "        %s = arith.select %c, %zero, %v : i32\n        linalg.yield %s : i32"
    )
    assert _stage(body, "i32") == CG.RELU


@pytest.mark.parametrize(
    "body",
    [
        # An unordered predicate decides NaN the other way.
        _ZERO_F + "        %c = arith.cmpf ult, %v, %zero : f32\n        %s = arith.select %c, %zero, %v : f32",
        # The arms in the order that computes min(x, 0).
        _ZERO_F + "        %c = arith.cmpf olt, %v, %zero : f32\n        %s = arith.select %c, %v, %zero : f32",
        # A threshold that is not zero.
        "        %one = arith.constant 1.0 : f32\n        %c = arith.cmpf olt, %v, %one : f32\n"
        "        %s = arith.select %c, %one, %v : f32",
    ],
)
def test_a_compare_and_select_that_is_not_max_with_zero_is_not_a_relu(body):
    assert _stage(body + "\n        linalg.yield %s : f32") == CG.ELEMENTWISE


class _Derived(_Oracle):
    """A synthetic target whose readout is a FACET, asked the way the real oracle asks it."""

    def __init__(self, facet: RF.ReadoutFacet):
        super().__init__()
        self.readout, self.routes = RF.TargetReadout((facet,)), ()

    def absorbs(self, kind: str):
        return CG.TargetOracle.absorbs(self, kind)


def _groups(text: str, oracle):
    return CG.form_groups(mq.parse(text), "synthetic", oracle=oracle)


def test_a_derived_stage_is_absorbed_where_no_readout_declares_it():
    declared_bias = ({"selector": "out", "applies": ["bias_add"]},)
    derived = RF.ReadoutFacet(target="synthetic", readouts=declared_bias, derived_stages=("acc_scale", "relu"))
    (device,) = _groups(_module(weight_dequantize="per_tensor"), _Derived(derived))
    assert device.stages == ["dequantize", "dequantize", "contraction", "bias_add", "relu", "quantize"]
    assert device.stopped_by is None

    undeclared = RF.ReadoutFacet(target="synthetic", readouts=declared_bias)
    device, host = _groups(_module(weight_dequantize="per_tensor"), _Derived(undeclared))
    assert (device.stopped_by, device.refusal) == ("relu", CG.READOUT_DOES_NOT_APPLY)
    assert host.stages == ["relu", "quantize"]


def test_a_residual_the_unit_could_accumulate_stays_on_the_host_until_it_is_proved_exact():
    applies = ({"selector": "out", "applies": ["bias_add", "relu", "acc_scale"]},)
    facet = RF.ReadoutFacet(target="synthetic", readouts=applies, accumulates=True)
    device, host = _groups(_module(weight_dequantize="per_tensor", residual=True), _Derived(facet))
    assert (device.stopped_by, device.refusal) == ("residual_add", CG.ACCUMULATE_UNPROVED)
    assert "no proof" in device.reason
    assert CG.gap_class(CG.ACCUMULATE_UNPROVED) == "OG6"
    assert host.stages == ["residual_add", "quantize"]

    plain = RF.ReadoutFacet(target="synthetic", readouts=applies)
    device, _host = _groups(_module(weight_dequantize="per_tensor", residual=True), _Derived(plain))
    assert device.refusal == CG.READOUT_DOES_NOT_APPLY


def _record(**stages):
    from merlin.targetgen.rtl import output_epilogue as OE

    record = OE.underived("synthetic")
    record["unknown"] = {}
    for stage, applies in stages.items():
        capability = OE.STAGE_CAPABILITY[stage]
        record["stages"][stage] = applies
        if applies:
            record[capability]["status"] = OE.DERIVED
    if stages.get("acc_scale"):
        record["scale"].update({"dtype": "f32", "format": {"format": "fp32"}, "carriers": [{"field": "s"}]})
    if stages.get("relu"):
        record["activation"]["functions"] = {"relu": {"codes": [1]}}
    return OE.validate(record)


def test_a_contraction_unit_facet_carries_the_derived_stages():
    record = _record(acc_scale=True, relu=True, maxpool=False)
    facet = RF.derive("synthetic", unit={"name": "array", "ops": ["matmul"]}, output_epilogue=record)
    assert facet.derived_stages == ("acc_scale", "relu")
    assert facet.applies_stage("relu") is True and facet.applies_stage("maxpool") is None
    assert facet.scale_dtype == "f32" and "rtl_derived" in facet.rungs_for("derived_stages")
    other = RF.derive("synthetic", unit={"name": "dma", "ops": ["copy"]}, output_epilogue=record)
    assert other.derived_stages == ()
