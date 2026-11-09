"""Neutral public input and isolated compiler probe for author prelaunch readiness.

This is one framework smoke program with a synthetic capability declaration.
It grants no target facts, candidate acceptance policy or physical authority.
The operator reads this owner from the explicitly selected frozen framework;
the probe then imports only the author's staged compiler implementation.
"""

from __future__ import annotations

import json

FUSION_SOURCE = r"""builtin.module {
  func.func @forward(%x: tensor<4x8xi8>, %rq: tensor<4x8xi8>) -> tensor<4x8xi8> {
    %w = arith.constant dense<[[-2, -1, 0, 1, 2, -2, -1, 0], [1, 2, -2, -1, 0, 1, 2, -2], [-1, 0, 1, 2, -2, -1, 0, 1], [2, -2, -1, 0, 1, 2, -2, -1], [0, 1, 2, -2, -1, 0, 1, 2], [-2, -1, 0, 1, 2, -2, -1, 0], [1, 2, -2, -1, 0, 1, 2, -2], [-1, 0, 1, 2, -2, -1, 0, 1]]> : tensor<8x8xi8>
    %zero = arith.constant 0 : i32
    %e0 = tensor.empty() : tensor<4x8xi32>
    %init = linalg.fill ins(%zero : i32) outs(%e0 : tensor<4x8xi32>) -> tensor<4x8xi32>
    %mm = linalg.matmul ins(%x, %w : tensor<4x8xi8>, tensor<8x8xi8>) outs(%init : tensor<4x8xi32>) -> tensor<4x8xi32>
    %e_f = tensor.empty() : tensor<4x8xf32>
    %f = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<4x8xi32>) outs(%e_f : tensor<4x8xf32>) {
    ^bb0(%a: i32, %o: f32):
      %c = arith.sitofp %a : i32 to f32
      linalg.yield %c : f32
    } -> tensor<4x8xf32>

    %sin = arith.constant 2.500000e-01 : f32
    %s0 = tensor.splat %sin : tensor<4x8xf32>
    %e_m1 = tensor.empty() : tensor<4x8xf32>
    %m1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%f, %s0 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%e_m1 : tensor<4x8xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %c = arith.mulf %a, %b : f32
      linalg.yield %c : f32
    } -> tensor<4x8xf32>

    %sw = arith.constant dense<[5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01]> : tensor<8xf32>
    %e_m2 = tensor.empty() : tensor<4x8xf32>
    %m2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%m1, %sw : tensor<4x8xf32>, tensor<8xf32>) outs(%e_m2 : tensor<4x8xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %c = arith.mulf %a, %b : f32
      linalg.yield %c : f32
    } -> tensor<4x8xf32>

    %bias = arith.constant dense<[0.000000e+00, 8.000000e+00, -8.000000e+00, 1.600000e+01, 1.000000e+00, -1.000000e+00, 0.000000e+00, 3.200000e+01]> : tensor<8xf32>
    %e_ba = tensor.empty() : tensor<4x8xf32>
    %ba = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%m2, %bias : tensor<4x8xf32>, tensor<8xf32>) outs(%e_ba : tensor<4x8xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %c = arith.addf %a, %b : f32
      linalg.yield %c : f32
    } -> tensor<4x8xf32>

    %e_r = tensor.empty() : tensor<4x8xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%ba : tensor<4x8xf32>) outs(%e_r : tensor<4x8xf32>) {
    ^bb0(%a: f32, %o: f32):
      %z = arith.constant 0.000000e+00 : f32
      %c = arith.maximumf %a, %z : f32
      linalg.yield %c : f32
    } -> tensor<4x8xf32>

    %qs = arith.constant dense<8.000000e+00> : tensor<f32>
    %qz = arith.constant dense<0> : tensor<i64>
    %q = "quant_ext.quantize_per_tensor"(%r, %qs, %qz) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<4x8xf32>, tensor<f32>, tensor<i64>) -> tensor<4x8xi8>
    func.return %q : tensor<4x8xi8>
  }
}"""

FUSION_OUTPUT_PATH_JSON = '{"accumulator": {"dtype": "i32"}, "accumulator_stages": [{"dtype": "i32", "kind": "preload", "range": [-2147483648, 2147483647], "rounding": "none"}, {"dtype": "i32", "kind": "accumulate", "operand_scale": "none"}], "exits": {"accumulator": true, "requant": true}, "readout": [{"domain": "accumulator", "function": "relu", "kind": "activation"}, {"kind": "convert", "rounding": "half_even", "to": "f32"}, {"columns_per_scale": 4, "dtype": "f32", "granularity": "per_command", "kind": "scale", "rounding": "half_even"}, {"clamp": [-128, 127], "kind": "requant", "rounding": "half_even", "to": "i8"}, {"kind": "pool", "padding": "unknown", "reductions": ["max"]}], "schema": "output_path_v1"}'

AUTHOR_PROBE = r"""
payload = json.loads(sys.argv[1])
from merlin.common import mlir_query as mq
from merlin.compile.analysis.contraction_shapes import contraction_extents
from merlin.targetgen import output_path as OP
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import output_fusion as OF

class _SyntheticFusionCapability:

    def __init__(self, holds=('tensor',), families=('contraction', 'elementwise_map'), readout=None, applies=('bias_add', 'bias', 'requant', 'acc_scale', 'relu', 'maxpool'), standalone=()):
        (self.holds, self.families, self.target, self.readout) = (holds, families, 'synthetic', readout)
        self.standalone = tuple(standalone)
        self.asked: list[dict] = []
        self.applies = applies

    def unit_for(self, *_args) -> str:
        return 'unit0'

    def absorbs(self, kind: str):
        return CG.readout_absorbs(kind, self.declared_readout(), target=self.target)

    def declared_readout(self):
        from merlin.targetgen import readout_facet as RF
        declared = ({'selector': 'out', 'applies': list(self.applies)},) if self.applies is not None else ()
        return RF.TargetReadout((RF.ReadoutFacet(target=self.target, readouts=declared),))

    def ask(self, *, op, family, in_dtype, weight_dtype=None, rank=None, attached, granularity=None, form=None):
        self.asked.append({'op': op, 'family': family, 'in_dtype': in_dtype, 'attached': attached})
        if family not in self.families:
            return CG.Admission(False, 'undeclared_family', f'no capability for {family}')
        if in_dtype != 'int8':
            return CG.Admission(False, 'input_dtype', f'{in_dtype} is not an integer format here')
        if family != 'contraction' and family not in self.standalone and (not attached):
            return CG.Admission(False, 'fused_only', 'available only fused with a contraction')
        if granularity is not None and granularity not in self.holds:
            return CG.Admission(False, 'scale_granularity', f'scales per {granularity}; the readout holds {list(self.holds)}')
        return CG.Admission(True, units=('unit0',))

module = mq.parse(payload["source"])
extents = contraction_extents(module)
first = CG.form_groups(module, "synthetic", oracle=_SyntheticFusionCapability())
decided = OF.decisions(first, module, OP.validate(payload["output_path"]), OF.Values())
groups = CG.form_groups(module, "synthetic", oracle=_SyntheticFusionCapability(), fusion=decided)
device = [g for g in groups if g.placement != CG.HOST]
_emit({"extents": [list(v) for v in extents.values()], "cuts": [g.fusion["cut"] for g in device],
       "refusals": [g.fusion.get("refusals") for g in device], "outside": _outside(_ROOTS)})
"""


def probe_request() -> dict:
    """Return fresh inputs and the probe from this selected framework owner."""
    return {
        "probe": AUTHOR_PROBE,
        "input": {"source": FUSION_SOURCE, "output_path": json.loads(FUSION_OUTPUT_PATH_JSON)},
    }
