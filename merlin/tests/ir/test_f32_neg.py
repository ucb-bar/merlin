"""Exact source negation, including values multiplication cannot preserve."""
from dataclasses import replace

import pytest

from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def program(body="%v = arith.negf %a : f32", yielded="%v", maps="(d0)", iterator="parallel"):
    return f'''module {{
func.func @forward(%x: tensor<12xf32>) -> tensor<12xf32> {{
 %empty = tensor.empty() : tensor<12xf32>
 %out = linalg.generic {{indexing_maps = [affine_map<(d0)->{maps}>, affine_map<(d0)->(d0)>],
 iterator_types = ["{iterator}"]}} ins(%x : tensor<12xf32>) outs(%empty : tensor<12xf32>) {{
 ^bb0(%a: f32, %init: f32):
 {body}
 linalg.yield {yielded} : f32
 }} -> tensor<12xf32>
 func.return %out : tensor<12xf32>
}}
}}'''


def translate(text):
    return translate_linalg_text(text, entry="forward", target_identity="portable-test", lowering_policy="diagnostic").request


def test_neg_preserves_all_payload_bits_and_inverts_only_sign():
    source = translate(program())
    node = next(n for n in source.nodes if n.op == "f32_neg")
    assert dict(node.attrs) == {"numeric_semantics": "f32-sign-bit-flip-v1"}
    # Both zeros, smallest subnormals, finite extrema, infinities, quiet/signaling
    # NaNs and signed payloads: no host float conversion may quiet the payload.
    words = (0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff,
             0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x7f812345, 0xff812345)
    input_node = next(n for n in source.nodes if n.effect == "input")
    actual, = evaluate_graph(source, {input_node.id: TensorValue(input_node.type, words)})
    assert actual.elements == tuple(w ^ (1 << 31) for w in words)
    invalid = replace(source, nodes=tuple(replace(n, attrs=(("numeric_semantics", "ieee754-f32-mul-rne-v1"),))
                                         if n.id == node.id else n for n in source.nodes))
    with pytest.raises(ValueError, match="unsupported type or policy"):
        evaluate_graph(invalid, {input_node.id: TensorValue(input_node.type, words)})


@pytest.mark.parametrize("changes", [
    {"body": "%v = arith.negf %init : f32"},
    {"yielded": "%a"},
    {"body": "%v = arith.negf %a fastmath<fast> : f32"},
    {"body": "%u = arith.negf %a : f32\n%v = arith.negf %u : f32"},
    {"maps": "(0)"},
    {"iterator": "reduction"},
])
def test_changed_neg_body_or_shell_is_refused(changes):
    with pytest.raises((LinalgBridgeError, ValueError)):
        translate(program(**changes))
