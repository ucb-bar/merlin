"""Preserve scalar tensor and boolean ports without assigning device semantics."""

import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.model import KernelRequest, TensorType
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def test_scalar_and_mask_ports_round_trip_and_keep_output_order():
    source = """module {
      func.func @forward(%scale: tensor<f32>, %mask: tensor<2xi1>) -> (tensor<2xi1>, tensor<f32>) {
        return %mask, %scale : tensor<2xi1>, tensor<f32>
      }
    }"""
    request = translate_linalg_text(source, entry="forward", target_identity="synthetic", lowering_policy="diagnostic").request
    assert request.nodes[0].type.shape == ()
    assert request.nodes[1].type == TensorType((2,), "i1", "boolean")
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    arguments = {
        request.nodes[0].id: TensorValue(request.nodes[0].type, (0x3F800000,)),
        request.nodes[1].id: TensorValue(request.nodes[1].type, (1, 0)),
    }
    assert tuple(value.elements for value in evaluate_graph(request, arguments)) == ((1, 0), (0x3F800000,))
    with pytest.raises(ValueError, match="zero or one"):
        TensorValue(request.nodes[1].type, (1, 2))


@pytest.mark.parametrize("shape", ((0,), (-1,), (1, 0)))
def test_rank_zero_admission_does_not_admit_dynamic_or_empty_dimensions(shape):
    with pytest.raises(ValueError, match="positive static"):
        TensorType(shape, "f32", "raw-bit-pattern")
