"""Region provenance cannot turn an explicit permutation into a contraction."""

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen.capsule_source import ModelDemandIncomplete, model_op_demands_checked
from merlin.xdsl_dialects._common import text as module_text

SOURCE = '''module {
  func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<4x2xi32> {
    %e = tensor.empty() : tensor<2x4xi32>
    %d = linalg.matmul {prov.op = "convolution_im2col_matmul", prov.family = "contraction"}
      ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%e : tensor<2x4xi32>) -> tensor<2x4xi32>
    %t = tensor.empty() : tensor<4x2xi32>
    %r = "linalg.transpose"(%d, %t)
      <{permutation = array<i64: 1, 0>, operandSegmentSizes = array<i32: 1, 1>}>
      {prov.op = "convolution_im2col_matmul", prov.family = "contraction"}
      : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
    func.return %r : tensor<4x2xi32>
  }
}'''


@pytest.mark.parametrize("generic", [False, True])
def test_region_tagged_transpose_keeps_one_exact_contraction_and_movement(generic):
    source = module_text(mq.parse(SOURCE), generic=True) if generic else SOURCE
    demands = model_op_demands_checked(source, "int8")
    (dot,) = [d for d in demands if d.family == "contraction"]
    assert (dot.batch, dot.m, dot.k, dot.n) == (1, 2, 3, 4)
    (transpose,) = [d for d in demands if d.carrier_op == "linalg.transpose"]
    assert transpose.family == "movement" and transpose.form == "permutation"
    assert transpose.elem_fmt == "int32" and transpose.has_shape is False


def test_transpose_correction_cannot_hide_an_untagged_real_contraction():
    source = SOURCE.replace('prov.op = "convolution_im2col_matmul"', 'prov.unattributed = "dot"', 1)
    with pytest.raises(ModelDemandIncomplete, match="untagged=.*linalg.matmul"):
        model_op_demands_checked(source, "int8")
