"""Exact BF16 sign flips and mask selection preserve authored bit patterns."""
from dataclasses import replace

import pytest

from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.semantic_compiler.model import pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def translate(text):
    return translate_linalg_text(text, entry="forward", target_identity="portable-test", lowering_policy="diagnostic")


def negate(body="%v = arith.negf %a : bf16", yielded="%v", iterator="parallel", size=65536):
    return f'''module {{
 func.func @forward(%x: tensor<{size}xbf16>) -> tensor<{size}xbf16> {{
  %empty = tensor.empty() : tensor<{size}xbf16>
  %out = linalg.generic {{indexing_maps = [affine_map<(d)->(d)>,affine_map<(d)->(d)>],
    iterator_types = ["{iterator}"]}} ins(%x : tensor<{size}xbf16>) outs(%empty : tensor<{size}xbf16>) {{
  ^bb0(%a: bf16, %init: bf16):
    {body}
    linalg.yield {yielded} : bf16
  }} -> tensor<{size}xbf16>
  func.return %out : tensor<{size}xbf16>
 }}
}}'''


def mask(body=None, yielded="%v", iterator="parallel", condition_map="(d0, 0)"):
    if body is None:
        body = "%limit = arith.constant -3.3895313892515355e+38 : bf16\n%v = arith.select %c, %s, %limit : bf16"
    return f'''module {{
 func.func @forward(%condition:tensor<2x1xi1>, %scalar:tensor<bf16>) -> tensor<2x3xbf16> {{
  %empty = tensor.empty() : tensor<2x3xbf16>
  %out = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->{condition_map}>,
    affine_map<(d0,d1)->()>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "{iterator}"]}}
    ins(%condition, %scalar : tensor<2x1xi1>, tensor<bf16>) outs(%empty : tensor<2x3xbf16>) {{
  ^bb0(%c:i1, %s:bf16, %init:bf16):
    {body}
    linalg.yield {yielded} : bf16
  }} -> tensor<2x3xbf16>
  func.return %out : tensor<2x3xbf16>
 }}
}}'''


def execute(translated, *values):
    request=translated.request
    inputs={f"arg{i}":TensorValue(request.node(f"arg{i}").type, tuple(value)) for i,value in enumerate(values)}
    return evaluate_graph(request,inputs,constants=translated.constants)[0].elements


def test_bf16_neg_all_65536_patterns_are_exact_sign_bit_flips():
    result=translate(negate())
    node=result.request.nodes[-1]
    assert node.op == "bf16_neg"
    assert node.type.dtype == "bf16" and node.type.numerical_policy == "raw-bit-pattern"
    assert node.attrs == (("numeric_semantics","bf16-sign-bit-flip-v1"),)
    assert node.index_maps == pointwise_identity_maps(1,1)
    assert execute(result,range(65536)) == tuple(word ^ 0x8000 for word in range(65536))
    changed=replace(node,attrs=(("numeric_semantics","bf16-multiply-minus-one"),))
    invalid=replace(result,request=replace(result.request,nodes=(*result.request.nodes[:-1],changed)))
    with pytest.raises(ValueError,match="unsupported type or policy"):
        execute(invalid,range(65536))


def test_bf16_mask_literal_and_singleton_projections_keep_exact_bits():
    translated=translate(mask())
    assert [node.op for node in translated.request.nodes] == ["input","input","gather","gather","constant","gather","select"]
    constant=translated.request.constants[0]
    assert constant.encoding == "bf16-le-raw"
    assert translated.request.node(constant.node_id).type.shape == ()
    assert bytes.fromhex(constant.data_hex) == b"\x7f\xff"
    for payload in (0,0x8000,1,0x7f80,0xff80,0x7f81,0x7fc1,0xffc3):
        assert execute(translated,(1,0),(payload,)) == (payload,)*3 + (0xff7f,)*3


def test_bf16_select_three_tensor_inputs_preserves_nan_payloads():
    source='''module {func.func @forward(%c:tensor<4xi1>,%a:tensor<4xbf16>,%b:tensor<4xbf16>) -> tensor<4xbf16> {
     %empty=tensor.empty():tensor<4xbf16>
     %v=linalg.generic {indexing_maps=[affine_map<(d)->(d)>,affine_map<(d)->(d)>,affine_map<(d)->(d)>,affine_map<(d)->(d)>],iterator_types=["parallel"]}
      ins(%c,%a,%b:tensor<4xi1>,tensor<4xbf16>,tensor<4xbf16>) outs(%empty:tensor<4xbf16>) {
      ^bb0(%cc:i1,%aa:bf16,%bb:bf16,%unused:bf16):
       %r=arith.select %cc,%aa,%bb:bf16
       linalg.yield %r:bf16
      } -> tensor<4xbf16>
      return %v:tensor<4xbf16>
    }}'''
    translated=translate(source)
    assert execute(translated,(1,0,1,0),(0x7f81,1,0x8000,0),(0,0xffc3,0,0x7f80)) == (0x7f81,0xffc3,0x8000,0x7f80)


@pytest.mark.parametrize("changes",[
    {"body":"%v = arith.negf %init : bf16"},
    {"yielded":"%a"},
    {"body":"%v = arith.negf %a fastmath<fast> : bf16"},
    {"body":"%u=arith.negf %a:bf16\n%v=arith.negf %u:bf16"},
    {"iterator":"reduction"},
])
def test_bf16_neg_rejects_changed_body_and_shell(changes):
    with pytest.raises((LinalgBridgeError,ValueError)):
        translate(negate(**changes))


@pytest.mark.parametrize("changes",[
    {"body":"%limit=arith.constant 0.0:bf16\n%v=arith.addf %s,%limit:bf16"},
    {"body":"%limit=arith.constant 0.0:bf16\n%v=arith.select %c,%s,%init:bf16"},
    {"body":"%limit=arith.constant 0.0:bf16\n%v=arith.select %c,%s,%limit:bf16", "yielded":"%init"},
    {"body":"%limit=arith.constant 0.0:bf16\n%u=arith.select %c,%s,%limit:bf16\n%v=arith.negf %u:bf16"},
    {"condition_map":"(d0,d1)"},
    {"iterator":"reduction"},
])
def test_bf16_mask_refuses_nonselect_bodies_and_invalid_projections(changes):
    with pytest.raises((LinalgBridgeError,ValueError)):
        translate(mask(**changes))
