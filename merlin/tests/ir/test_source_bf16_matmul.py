"""Authored BF16 contractions retain unresolved order and immutable words."""
import pytest

from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


SOURCE = '''module { func.func @work(%a: tensor<2x3xbf16>, %b: tensor<3x4xbf16>,
  %init: tensor<2x4xbf16>) -> tensor<2x4xbf16> {
  %result = linalg.matmul ins(%a, %b : tensor<2x3xbf16>, tensor<3x4xbf16>)
    outs(%init : tensor<2x4xbf16>) -> tensor<2x4xbf16>
  func.return %result : tensor<2x4xbf16>
} }'''


def test_source_bf16_matmul_keeps_parameters_and_unqualified_order():
    rhs = bytes.fromhex('803f00c000007f7f') * 3
    translated = translate_linalg_text(SOURCE, entry='work', target_identity='test',
                                       argument_constants={1: rhs})
    request = translated.request
    node = request.nodes[-1]
    assert node.op == 'source_bf16_matmul_unqualified'
    assert node.inputs == ('arg0', 'arg1', 'arg2')
    assert node.type.dtype == 'bf16'
    assert dict(node.attrs) == {'reduction_order': 'unresolved',
                               'numeric_semantics': 'source-bf16-mul-add-unqualified-v1'}
    assert request.constants[0].data_hex == rhs.hex()
    assert request.constants[0].encoding == 'bf16-le-raw'
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    values = {node.id: TensorValue(node.type, (0,) * 6) for node in request.nodes if node.id == 'arg0'}
    values['arg2'] = TensorValue(request.node('arg2').type, (0,) * 8)
    with pytest.raises(ValueError, match='unresolved reduction order'):
        evaluate_graph(request, values, constants=translated.constants)


def test_source_bf16_matmul_rejects_shape_and_type_changes():
    for changed in (SOURCE.replace('3x4xbf16', '5x4xbf16'),
                    SOURCE.replace('tensor<2x4xbf16>', 'tensor<2x4xf32>')):
        with pytest.raises(LinalgBridgeError):
            translate_linalg_text(changed, entry='work', target_identity='test')


def test_source_bf16_zero_filled_matmul_materializes_authored_positive_zero():
    source = SOURCE.replace('%init: tensor<2x4xbf16>', '%unused: tensor<2x4xbf16>', 1)
    source = source.replace('  %result =', '''  %empty = tensor.empty() : tensor<2x4xbf16>
  %zero = arith.constant {prov.role = "source-support"} 0.000000e+00 : bf16
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<2x4xbf16>) -> tensor<2x4xbf16>
  %result =''')
    request = translate_linalg_text(source, entry='work', target_identity='test').request
    node = request.nodes[-1]
    assert node.op == 'source_bf16_matmul_unqualified'
    seed = next(item for item in request.constants if item.node_id == node.inputs[2])
    assert seed.encoding == 'bf16-le-raw'
    assert seed.data_hex == '0000' * 8
    assert request.node(node.inputs[2]).type.shape == (2, 4)
