"""Group formation over a static integerization's readout chain.

A static integerization writes every contraction's readout as ``float(acc) * s_in * s_w``, an
optional layout permutation, a per-channel float bias, the activation and the output quantize. The
pass used to stop at the first operation of that chain -- the accumulator cast, which comes before
any scale -- so no group ever carried a bias, scale or activation, and nothing downstream demanded
the fused readout every quantized layer needs. These tests pin the chain on a target whose readout
declares scale and activation and whose stage route seeds the bias into the accumulator.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.xdsl_dialects.lowering import compute_groups as CG

pytestmark = pytest.mark.target("gemmini")

_TARGET = "gemmini"

_PREFIX = """
builtin.module {
  func.func @forward(%x: tensor<4x8xf32>, %w: tensor<8x16xi8>, %b: tensor<16xf32>) -> tensor<16x4xi8> {
    %s = arith.constant dense<5.000000e-01> : tensor<f32>
    %z = arith.constant dense<0> : tensor<i64>
    %c0 = arith.constant 0 : i32
    %qx = "quant_ext.quantize_per_tensor"(%x, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64,
      output_dtype = "int8"}> : (tensor<4x8xf32>, tensor<f32>, tensor<i64>) -> tensor<4x8xi8>
    %e0 = tensor.empty() : tensor<4x16xi32>
    %f = linalg.fill ins(%c0 : i32) outs(%e0 : tensor<4x16xi32>) -> tensor<4x16xi32>
    %mm = linalg.matmul ins(%qx, %w : tensor<4x8xi8>, tensor<8x16xi8>)
                        outs(%f : tensor<4x16xi32>) -> tensor<4x16xi32>
    %e1 = tensor.empty() : tensor<4x16xf32>
    %c = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<4x16xi32>) outs(%e1 : tensor<4x16xf32>) {
    ^bb0(%p: i32, %o: f32):
      %r = arith.sitofp %p : i32 to f32
      linalg.yield %r : f32
    } -> tensor<4x16xf32>
"""

_SCALED_TAIL = """
    %k = arith.constant 7.812500e-03 : f32
    %ks = tensor.splat %k : tensor<4x16xf32>
    %e2 = tensor.empty() : tensor<4x16xf32>
    %m = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%c, %ks : tensor<4x16xf32>, tensor<4x16xf32>) outs(%e2 : tensor<4x16xf32>) {
    ^bb0(%p: f32, %q: f32, %o: f32):
      %r = arith.mulf %p, %q : f32
      linalg.yield %r : f32
    } -> tensor<4x16xf32>
    %e3 = tensor.empty() : tensor<16x4xf32>
    %t = linalg.transpose ins(%m : tensor<4x16xf32>) outs(%e3 : tensor<16x4xf32>) permutation = [1, 0]
    %e4 = tensor.empty() : tensor<16x4xf32>
    %a = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%t, %b : tensor<16x4xf32>, tensor<16xf32>) outs(%e4 : tensor<16x4xf32>) {
    ^bb0(%p: f32, %q: f32, %o: f32):
      %r = arith.addf %p, %q : f32
      linalg.yield %r : f32
    } -> tensor<16x4xf32>
    %e5 = tensor.empty() : tensor<16x4xf32>
    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]} ins(%a : tensor<16x4xf32>) outs(%e5 : tensor<16x4xf32>) {
    ^bb0(%p: f32, %o: f32):
      %zero = arith.constant 0.000000e+00 : f32
      %r = arith.maximumf %p, %zero : f32
      linalg.yield %r : f32
    } -> tensor<16x4xf32>
    %so = arith.constant dense<2.500000e-01> : tensor<f32>
    %y = "quant_ext.quantize_per_tensor"(%relu, %so, %z) <{quant_min = -128 : i64, quant_max = 127 : i64,
      output_dtype = "int8"}> : (tensor<16x4xf32>, tensor<f32>, tensor<i64>) -> tensor<16x4xi8>
    func.return %y : tensor<16x4xi8>
  }
}
"""


#: A readout that applies scale and activation on its narrowing store, as a target declares it, and
#: a stage route that seeds a bias into the accumulator before the contraction.
_READOUT = SimpleNamespace(
    applies_stage=lambda stage: stage in {"acc_scale", "relu", "maxpool"},
    admits_granularity=lambda granularity: granularity == "tensor",
    unknown={},
    scale_granularities=("tensor",),
)
_SEED = SimpleNamespace(stage="bias_add", composed_with="contraction")


def _oracle(**over):
    over.setdefault("readout", _READOUT)
    over.setdefault("routes", (_SEED,))
    try:
        return CG.TargetOracle(_TARGET, **over)
    except Exception as exc:  # noqa: BLE001 -- no selected capability contract, nothing to ask
        pytest.skip(f"no {_TARGET} capability contract is selected: {type(exc).__name__}: {exc}")


def _contraction_group(oracle):
    with IR_LOCK:
        groups = CG.form_groups(mq.parse(_PREFIX + _SCALED_TAIL), _TARGET, oracle=oracle)
    return next(g for g in groups if g.root is not None and CG.CONTRACTION in g.stages)


def test_a_scalar_spread_to_the_full_shape_is_a_per_tensor_scale():
    with IR_LOCK:
        module = mq.parse(_PREFIX + _SCALED_TAIL)
    multiplies = [op for op in module.walk() if CG.classify(op) is not None and CG.classify(op).kind == CG.SCALE]
    assert multiplies and all(CG.classify(op).varies_over == () for op in multiplies)


def test_the_whole_integerized_readout_chain_is_one_device_group():
    group = _contraction_group(_oracle())
    assert group.placement != CG.HOST, group.reason
    for kind in (CG.CAST, CG.SCALE, CG.MOVEMENT, CG.BIAS_ADD, CG.RELU, CG.QUANTIZE):
        assert kind in group.stages, (kind, group.stages)
    assert group.refusal is None


def test_without_a_declared_bias_route_the_unclosed_chain_stays_on_the_host():
    """Refused at the bias, the chain never reaches its quantize: a converted and scaled accumulator is
    a float tensor no integer readout writes, so the conversion and scales stay host work."""
    group = _contraction_group(_oracle(routes=()))
    assert group.stages == [CG.CONTRACTION]
    assert group.stopped_by == CG.CAST and group.refusal == CG.READOUT_REQUIRES_SCALE
    assert "bias_add" in group.reason, "the stage that ended growth is still named"


def test_a_route_for_another_composition_does_not_license_the_stage():
    other = SimpleNamespace(stage="bias_add", composed_with="residual_add")
    group = _contraction_group(_oracle(routes=(other,)))
    assert CG.BIAS_ADD not in group.stages and group.stages == [CG.CONTRACTION]


def test_the_converted_store_states_one_multiplier_and_an_accumulator_domain_bias():
    """``(float(acc) * s + b)`` quantized at ``s_out``: the readout multiplier is ``s / s_out`` and the
    bias folds into the accumulator in units of ``s``, read from the model argument behind it."""
    from merlin.xdsl_dialects.lowering import group_command as GC
    from merlin.xdsl_dialects.lowering import group_numerics as GN

    group = _contraction_group(_oracle())
    numerics = GN.numerics_of(group)
    assert numerics.multiplier == pytest.approx(7.8125e-03 / 2.5e-01)
    assert numerics.bias_arg_index == 2 and numerics.bias_divisor == pytest.approx(7.8125e-03)
    assert numerics.activation == "relu" and numerics.clamp == (-128, 127)
    stated = GC.program(group, weight_args={1, 2})
    assert stated.entry["epilogue"] == ["bias_add", "acc_scale", "relu"]
    assert stated.entry["acc_scale"] == pytest.approx(7.8125e-03 / 2.5e-01)
