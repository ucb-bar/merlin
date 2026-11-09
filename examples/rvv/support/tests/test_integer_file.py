"""Source normalization closes exact arithmetic before a provider may stream."""

from merlin.common import mlir_query as mq
from test_backend import BACKEND

normalize = BACKEND.integer_file_api().normalize


def matrix_source(m=2, n=7, k=65, rhs_transposed=True):
    shape = f"{n}x{k}" if rhs_transposed else f"{k}x{n}"
    indexing = "(n,k)" if rhs_transposed else "(k,n)"
    return f"""module {{
func.func @forward(%a:tensor<{m}x{k}xi8>,%b:tensor<{shape}xi8>)->tensor<{m}x{n}xi32> {{
%z=arith.constant 0:i32
%e=tensor.empty():tensor<{m}x{n}xi32>
%c=linalg.fill ins(%z:i32)outs(%e:tensor<{m}x{n}xi32>)->tensor<{m}x{n}xi32>
%r=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->{indexing}>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]}}
ins(%a,%b:tensor<{m}x{k}xi8>,tensor<{shape}xi8>)outs(%c:tensor<{m}x{n}xi32>){{
^bb0(%x:i8,%y:i8,%acc:i32):
%xx=arith.extsi %x:i8 to i32
%yy=arith.extsi %y:i8 to i32
%p=arith.muli %xx,%yy:i32
%s=arith.addi %acc,%p:i32
linalg.yield %s:i32
}}->tensor<{m}x{n}xi32>
func.return %r:tensor<{m}x{n}xi32>
}}}}"""


SOURCE = """builtin.module {
 func.func @forward(%a:tensor<1x2x5x9xi8>,%b:tensor<5x2x3x3xi8>)->tensor<1x5x3x7xi32> {
 %z=arith.constant 0 : i32
 %e=tensor.empty():tensor<1x5x3x7xi32>
 %c=linalg.fill ins(%z:i32) outs(%e:tensor<1x5x3x7xi32>)->tensor<1x5x3x7xi32>
 %r=linalg.generic {indexing_maps=[
 affine_map<(n,f,h,w,c,kh,kw)->(n,c,h+kh,w+kw)>,
 affine_map<(n,f,h,w,c,kh,kw)->(f,c,kh,kw)>,
 affine_map<(n,f,h,w,c,kh,kw)->(n,f,h,w)>],
 iterator_types=["parallel","parallel","parallel","parallel","reduction","reduction","reduction"]}
 ins(%a,%b:tensor<1x2x5x9xi8>,tensor<5x2x3x3xi8>) outs(%c:tensor<1x5x3x7xi32>) {
 ^bb0(%x:i8,%y:i8,%acc:i32):
 %xx=arith.extsi %x:i8 to i32
 %yy=arith.extsi %y:i8 to i32
 %p=arith.muli %xx,%yy:i32
 %s=arith.addi %acc,%p:i32
 linalg.yield %s:i32
 }->tensor<1x5x3x7xi32>
 func.return %r:tensor<1x5x3x7xi32>
 }
} """


def test_window_normalization_keeps_public_order_and_source_integer_arithmetic():
    module = mq.parse(SOURCE)
    before = next(mq.walk(module, "func.func")).function_type
    report = normalize(module)
    assert report["normalized"] == 1 and not report["refusals"]
    assert next(mq.walk(module, "func.func")).function_type == before
    assert (
        report["public_output_layout_preserved"]
        and report["quantization_changed"] is False
    )
    assert len(list(mq.walk(module, "arith.extsi"))) == 2
    assert len(list(mq.walk(module, "arith.muli"))) == 1
    gathers = [op for op in mq.walk(module, "linalg.generic") if len(op.inputs) == 1]
    assert len(gathers) == 1 and len(gathers[0].results[0].type.get_shape()) == 6
    assert any(
        op.result.type.get_shape() == (378,)
        for op in mq.walk(module, "tensor.collapse_shape")
    )
    assert any(
        op.result.type.get_shape() == (18, 21)
        for op in mq.walk(module, "tensor.expand_shape")
    )
    module.verify()


def test_nonzero_source_accumulator_refuses_without_changing_source():
    module = mq.parse(SOURCE.replace("constant 0", "constant 1"))
    before = str(module)
    report = normalize(module)
    assert report["normalized"] == 0 and report["refusals"] == {"requires_zero_fill": 1}
    assert str(module) == before


def test_unrelated_scalar_wiring_is_not_a_mac():
    module = mq.parse(SOURCE.replace("%xx,%yy", "%xx,%xx"))
    before = str(module)
    report = normalize(module)
    assert report["refusals"] == {"requires_exact_mac_wiring": 1}
    assert str(module) == before


def test_integer_poison_flags_cannot_be_replaced_by_modular_arithmetic():
    from xdsl.dialects import arith

    module = mq.parse(SOURCE)
    multiply = next(mq.walk(module, "arith.muli"))
    multiply.properties["overflowFlags"] = arith.IntegerOverflowAttr(
        [arith.IntegerOverflowFlag.NSW]
    )
    before = str(module)
    report = normalize(module)
    assert report["refusals"] == {"nondefault_integer_overflow_flags": 1}
    assert str(module) == before


def test_unsigned_window_is_outside_the_signed_contract():
    module = mq.parse(SOURCE.replace("arith.extsi %x", "arith.extui %x"))
    before = str(module)
    assert normalize(module)["normalized"] == 0
    assert str(module) == before


def test_both_rhs_layouts_preserve_entry_types_and_get_exact_tail_blocks():
    import pytest

    api = BACKEND.integer_file_api()
    for vlen in (128, 256, 512, 1024):
        target = BACKEND.im2col_stream_api().PanelTarget(vlen, 64, 8, 32, 1, 65536)
        for transposed in (False, True):
            module = mq.parse(matrix_source(rhs_transposed=transposed))
            signature = next(mq.walk(module, "func.func")).function_type
            report = api.normalize(module, windows=False, contractions=True)
            assert report["normalized"] == 1
            blocks = api.register_blocks(module, target)
            assert [(e["mr"], e["nr"]) for e in blocks["entries"]] == [(2, 1)]
            assert next(mq.walk(module, "func.func")).function_type == signature
            assert api.lowering_schedule(blocks["table"])
    with pytest.raises(ValueError):
        api.exact_divisor(0, 4)


def test_large_minor_geometry_tracks_selected_vlen_without_a_fixed_default():
    api = BACKEND.integer_file_api()
    widths = []
    for vlen in (128, 256, 512, 1024):
        target = BACKEND.im2col_stream_api().PanelTarget(vlen, 64, 8, 32, 1, 65536)
        module = mq.parse(matrix_source(m=4, n=128, k=7, rhs_transposed=False))
        api.normalize(module, windows=False, contractions=True)
        entry = api.register_blocks(module, target)["entries"][0]
        widths.append(entry["nr"])
        assert entry["mr"] == 4
        assert entry["performance_optimality_claimed"] is False
    assert widths == [16, 32, 64, 128]


def test_unselected_file_api_performs_no_parse_or_rewrite(tmp_path):
    source = tmp_path / "opaque.mlir"
    source.write_text("opaque bytes")
    target = BACKEND.im2col_stream_api().PanelTarget(256, 64, 8, 32, 1, 65536)
    selected, report = BACKEND.integer_file_api().prepare_file(source, tmp_path, target)
    assert selected == source and report["normalization"]["normalized"] == 0
    assert list(tmp_path.iterdir()) == [source]
