"""Preserve strict captured negf without admitting relaxed or captured scalar SSA."""

import pytest

from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text


def source(dtype="f32", flags="none", operand="%v"):
    return f"""module {{
      func.func @forward(%x: tensor<2x3x{dtype}>) -> tensor<2x3x{dtype}> {{
        %empty = tensor.empty() : tensor<2x3x{dtype}>
        %y = linalg.generic {{indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
          iterator_types = ["parallel","parallel"]}}
          ins(%x : tensor<2x3x{dtype}>) outs(%empty : tensor<2x3x{dtype}>) {{
          ^bb0(%v: {dtype}, %old: {dtype}):
            %negative = arith.negf {operand} fastmath<{flags}> : {dtype}
            linalg.yield %negative : {dtype}
        }} -> tensor<2x3x{dtype}>
        return %y : tensor<2x3x{dtype}>
      }}
    }}"""


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_source_has_explicit_sign_flip_semantics(dtype):
    request = translate_linalg_text(source(dtype), entry="forward", target_identity="synthetic").request
    result = request.node(request.outputs[0])
    assert result.op == dtype + "_neg" and result.inputs == ("arg0",)
    assert dict(result.attrs) == {"numeric_semantics": dtype + "-sign-bit-flip-v1"}


@pytest.mark.parametrize("change", [{"flags": "nnan"}, {"operand": "%old"}])
def test_relaxed_or_uninitialized_sign_flip_is_rejected(change):
    with pytest.raises(LinalgBridgeError):
        translate_linalg_text(source(**change), entry="forward", target_identity="synthetic")
