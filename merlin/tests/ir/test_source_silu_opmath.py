"""Direct-division SiLU preserves f32 opmath and leaves exp unqualified."""
import pytest

from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text


def program(body=None):
    if body is None:
        body='''%one=arith.constant 1.0:f32
        %neg=arith.negf %x:f32
        %exp=math.exp %neg:f32
        %den=arith.addf %one,%exp:f32
        %value=arith.divf %x,%den:f32'''
    return '''module {func.func @forward(%a:tensor<4xbf16>) -> tensor<4xbf16> {
      %e0=tensor.empty():tensor<4xf32>
      %wide=linalg.generic {indexing_maps=[affine_map<(d)->(d)>,affine_map<(d)->(d)>],iterator_types=["parallel"]}
       ins(%a:tensor<4xbf16>) outs(%e0:tensor<4xf32>) {
       ^bb0(%x:bf16,%unused:f32):
        %v=arith.extf %x:bf16 to f32
        linalg.yield %v:f32
       } -> tensor<4xf32>
      %e1=tensor.empty():tensor<4xf32>
      %silu=linalg.generic {indexing_maps=[affine_map<(d)->(d)>,affine_map<(d)->(d)>],iterator_types=["parallel"]}
       ins(%wide:tensor<4xf32>) outs(%e1:tensor<4xf32>) {
       ^bb0(%x:f32,%unused:f32):
       BODY
        linalg.yield %value:f32
       } -> tensor<4xf32>
      %e2=tensor.empty():tensor<4xbf16>
      %result=linalg.generic {indexing_maps=[affine_map<(d)->(d)>,affine_map<(d)->(d)>],iterator_types=["parallel"]}
       ins(%silu:tensor<4xf32>) outs(%e2:tensor<4xbf16>) {
       ^bb0(%x:f32,%unused:bf16):
        %v=arith.truncf %x:f32 to bf16
        linalg.yield %v:bf16
       } -> tensor<4xbf16>
       return %result:tensor<4xbf16>
    }}'''.replace('BODY',body)


def translate(source):
    return translate_linalg_text(source,entry="forward",target_identity="portable-test").request


def test_silu_exp_is_preserved_as_unqualified_source_between_exact_rounding_boundaries():
    request=translate(program())
    assert [node.op for node in request.nodes] == ["input","bf16_to_f32","constant","gather","f32_neg",
        "source_f32_exp_unqualified","f32_add","f32_div","f32_to_bf16"]
    assert dict(request.nodes[5].attrs) == {"numeric_semantics":"source-f32-exp-unqualified-v1"}
    assert request.nodes[2].type.shape == ()
    assert request.constants[0].data_hex == "0000803f"
    assert request.nodes[-2].inputs == (request.nodes[1].id,request.nodes[-3].id)
    assert request.nodes[-1].type.dtype == "bf16"
    assert all(node.type.dtype == "f32" for node in request.nodes[1:-1])
    assert type(request).from_record(request.record()).digest() == request.digest()


@pytest.mark.parametrize("before,after",[
    ('1.0:f32','2.0:f32'),
    ('arith.negf %x:f32','arith.negf %unused:f32'),
    ('math.exp %neg:f32','math.exp %x:f32'),
    ('arith.addf %one,%exp:f32','arith.addf %exp,%one:f32'),
    ('arith.divf %x,%den:f32','arith.divf %unused,%den:f32'),
    ('arith.divf %x,%den:f32','arith.divf %x,%den fastmath<fast>:f32'),
])
def test_changed_direct_division_silu_dag_is_refused(before,after):
    with pytest.raises((LinalgBridgeError,ValueError)):
        translate(program().replace(before,after))
