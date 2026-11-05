"""Independent adversarial IR evidence for guarded zero-padding gathers.

The fixtures here are constructed without the realizer or its matcher. These
tests establish logical conformance, not physical allocation or target support.
"""

import pytest
from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.ops import GenericOp, IndexOp
from xdsl.rewriter import InsertPoint, Rewriter

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.im2col_padding_check import verify_padding_gather_rewrite


def pair(*, width=8, shape=(1, 1, 3, 4), padding=(1, 1, 1, 1), kernel=(3, 3), steps=(1, 1, 1, 1)):
    n, c, height, columns = shape
    top, bottom, left, right = padding
    kh, kw = kernel
    sh, dh, sw, dw = steps
    ph, pw = height + top + bottom, columns + left + right
    oh, ow = (ph - 1 - (kh - 1) * dh) // sh + 1, (pw - 1 - (kw - 1) * dw) // sw + 1
    elem = f"i{width}"
    image = f"tensor<{n}x{c}x{height}x{columns}x{elem}>"
    padded = f"tensor<{n}x{c}x{ph}x{pw}x{elem}>"
    gathered = f"tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}x{elem}>"
    header = f"""module {{
func.func private @before()
func.func private @after()
func.func @forward(%image:{image}, %other:{image})->({gathered},{gathered}) {{
func.call @before():()->()
%zero=arith.constant 0:{elem}
"""
    ge = f"%ge=tensor.empty():{gathered}\n"
    footer = f"""func.call @after():()->()
func.return %g,%g:{gathered},{gathered}
}} }}"""
    attrs = 'iterator_types=["parallel","parallel","parallel","parallel","parallel","parallel"]'
    identity = "affine_map<(c,kh,kw,n,oh,ow)->(c,kh,kw,n,oh,ow)>"
    before = (
        header
        + f"""%pe=tensor.empty():{padded}
%fill=linalg.fill ins(%zero:{elem}) outs(%pe:{padded})->{padded}
%pad="tensor.insert_slice"(%image,%fill) <{{
 operandSegmentSizes=array<i32:1,1,0,0,0>, static_offsets=array<i64:0,0,{top},{left}>,
 static_sizes=array<i64:{n},{c},{height},{columns}>,static_strides=array<i64:1,1,1,1>
}}> :({image},{padded})->{padded}
"""
        + ge
        + f"""%g=linalg.generic {{indexing_maps=[
affine_map<(c,kh,kw,n,oh,ow)->(n,c,oh*{sh}+kh*{dh},ow*{sw}+kw*{dw})>,{identity}],{attrs}}}
ins(%pad:{padded}) outs(%ge:{gathered}) {{
^bb0(%v:{elem},%old:{elem}):
linalg.yield %v:{elem}
}}->{gathered}
"""
        + footer
    )
    after = (
        header
        + ge
        + f"""%g=linalg.generic {{indexing_maps=[{identity}],{attrs}}}
outs(%ge:{gathered}) {{
^bb0(%old:{elem}):
%i0=linalg.index 0:index
%i1=linalg.index 1:index
%i2=linalg.index 2:index
%i3=linalg.index 3:index
%i4=linalg.index 4:index
%i5=linalg.index 5:index
%sh=arith.constant {sh}:index
%dh=arith.constant {dh}:index
%sw=arith.constant {sw}:index
%dw=arith.constant {dw}:index
%top=arith.constant {top}:index
%left=arith.constant {left}:index
%bottom=arith.constant {top + height}:index
%right=arith.constant {left + columns}:index
%ha=arith.muli %i4,%sh:index
%hb=arith.muli %i1,%dh:index
%h=arith.addi %ha,%hb:index
%wa=arith.muli %i5,%sw:index
%wb=arith.muli %i2,%dw:index
%w=arith.addi %wa,%wb:index
%a=arith.cmpi uge,%h,%top:index
%b=arith.cmpi ult,%h,%bottom:index
%c=arith.cmpi uge,%w,%left:index
%d=arith.cmpi ult,%w,%right:index
%ab=arith.andi %a,%b:i1
%cd=arith.andi %c,%d:i1
%both=arith.andi %ab,%cd:i1
%v=scf.if %both ->({elem}) {{
%ih=arith.subi %h,%top:index
%iw=arith.subi %w,%left:index
%read=tensor.extract %image[%i3,%i0,%ih,%iw]:{image}
scf.yield %read:{elem}
}} else {{
scf.yield %zero:{elem}
}}
linalg.yield %v:{elem}
}}->{gathered}
"""
        + footer
    )
    return before, after


def check(before, after, bits=64):
    return verify_padding_gather_rewrite(parse_mlir_text(before), parse_mlir_text(after), index_bits=bits)


@pytest.mark.parametrize("width", [8, 16, 32])
@pytest.mark.parametrize(
    "geometry", [dict(), dict(shape=(2, 2, 3, 5), padding=(1, 2, 2, 1), kernel=(2, 3), steps=(2, 2, 2, 1))]
)
def test_independent_guarded_gather_with_complete_output_fanout(width, geometry):
    before, after = pair(width=width, **geometry)
    records = check(before, after)
    assert len(records) == 1
    assert records[0]["eliminated_padding_payload_bytes"] > 0
    assert records[0]["gather_payload_bytes"] > 0
    assert check(before, before) == ()
    assert check(after, after) == ()


@pytest.mark.parametrize(
    "old,new",
    [
        ("cmpi uge,%h,%top", "cmpi ugt,%h,%top"),
        ("cmpi ult,%w,%right", "cmpi ule,%w,%right"),
        ("%both=arith.andi %ab,%cd", "%both=arith.ori %ab,%cd"),
        ("%both=arith.andi %ab,%cd", "%both=arith.andi %ab,%ab"),
        ("arith.subi %h,%top", "arith.subi %h,%left"),
        ("tensor.extract %image[", "tensor.extract %other["),
        ("[%i3,%i0,%ih,%iw]", "[%i0,%i3,%ih,%iw]"),
        ("%hb=arith.muli %i1,%dh", "%hb=arith.muli %i2,%dh"),
        ("scf.yield %zero:i8", "scf.yield %old:i8"),
        ("linalg.yield %v:i8", "linalg.yield %zero:i8"),
        ("func.call @before()", "func.call @after()"),
    ],
)
def test_actual_replacement_mutations_refuse(old, new):
    # Different top/left and N/C dimensions make swaps observably distinct.
    before, after = pair(shape=(2, 1, 3, 4), padding=(1, 1, 2, 1))
    assert old in after
    with pytest.raises(ValueError):
        check(before, after.replace(old, new))


def test_subtraction_outside_guard_refuses_even_when_rest_is_unchanged():
    before, after = pair()
    after = after.replace("%ih=arith.subi %h,%top:index\n", "")
    after = after.replace("%v=scf.if", "%ih=arith.subi %h,%top:index\n%v=scf.if")
    with pytest.raises(ValueError, match="eager"):
        check(before, after)


def test_eager_read_before_guard_refuses():
    before, after = pair()
    # Move the entire read chain before the guard; all IR types remain valid.
    body = (
        "%ih=arith.subi %h,%top:index\n%iw=arith.subi %w,%left:index\n"
        "%read=tensor.extract %image[%i3,%i0,%ih,%iw]:tensor<1x1x3x4xi8>\n"
    )
    assert body in after
    after = after.replace(body, "").replace("%v=scf.if", body + "%v=scf.if")
    with pytest.raises(ValueError, match="eager"):
        check(before, after)


@pytest.mark.parametrize(
    "source_change",
    [
        lambda s: s.replace("arith.constant 0:i8", "arith.constant 1:i8"),
        lambda s: s.replace("static_offsets=array<i64:0,0,1,1>", "static_offsets=array<i64:0,0,0,1>"),
        lambda s: s.replace("linalg.yield %v:i8", "linalg.yield %old:i8"),
        lambda s: s.replace(
            "%ge=tensor.empty()", '"opaque.observe"(%pad):(tensor<1x1x5x6xi8>)->()\n%ge=tensor.empty()'
        ),
        lambda s: s.replace("%ge=tensor.empty()", "func.call @after():()->()\n%ge=tensor.empty()"),
    ],
)
def test_source_must_establish_exact_closed_zero_padding(source_change):
    before, after = pair()
    with pytest.raises(ValueError):
        check(source_change(before), after)


def test_current_index_width_checks_bytes_and_unsupported_unchanged_fallback():
    before, after = pair(width=32)
    with pytest.raises(ValueError):
        check(before, after, bits=8)
    assert len(check(before, after, bits=16)) == 1
    opaque = before.replace(
        "func.call @before()", '"unknown.region"() ({\n"unknown.yield"():()->()\n}):()->()\nfunc.call @before()'
    )
    assert check(opaque, opaque) == ()


def test_partial_padding_retention_cannot_be_reported_as_eliminated():
    before, after = pair()
    after = after.replace("%ge=tensor.empty()", "%pe=tensor.empty():tensor<1x1x5x6xi8>\n%ge=tensor.empty()")
    with pytest.raises(ValueError, match="retained"):
        check(before, after)


def test_unknown_scalar_attributes_and_extra_effect_refuse():
    before, text = pair()
    after = parse_mlir_text(text)
    next(op for op in after.walk() if type(op) is IndexOp).attributes["unknown.permission"] = builtin.UnitAttr()
    with pytest.raises(ValueError, match="unknown"):
        verify_padding_gather_rewrite(parse_mlir_text(before), after, index_bits=64)
    after = parse_mlir_text(text)
    writer = next(op for op in after.walk() if type(op) is GenericOp)
    call = parse_mlir_text(
        "module { func.func @x() { func.call @after():()->() func.return } }"
    ).body.block.first_op.body.block.first_op
    call.detach()
    Rewriter.insert_op(call, InsertPoint.before(writer.body.block.last_op))
    with pytest.raises(ValueError):
        verify_padding_gather_rewrite(parse_mlir_text(before), after, index_bits=64)


def test_same_typed_image_swap_cannot_rebind_the_original_proof():
    before, after = pair()
    with pytest.raises(ValueError, match="coordinates|dataflow"):
        check(before, after.replace("tensor.extract %image[", "tensor.extract %other["))


def test_unchanged_nested_dense_nan_metadata_remains_bit_exact():
    before, after = pair()
    attr = ' attributes {prov.bits = {payload = dense<"0x0100C07F"> : tensor<1xf32>}} '
    before = before.replace("module {", "module" + attr + "{")
    after = after.replace("module {", "module" + attr + "{")
    assert len(check(before, after)) == 1
    with pytest.raises(ValueError):
        check(before, after.replace("0100C07F", "0200C07F"))


def test_dead_nested_value_type_change_is_not_hidden_by_structural_equivalence():
    before, after = pair()
    prefix = '"opaque.container"() ({\n%dead=arith.constant 3:i8\n"opaque.end"():()->()\n}):()->()\n'
    before = before.replace("func.call @before()", prefix + "func.call @before()")
    after = after.replace("func.call @before()", prefix + "func.call @before()")
    assert len(check(before, after)) == 1
    with pytest.raises(ValueError):
        check(before, after.replace("%dead=arith.constant 3:i8", "%dead=arith.constant 3:i16"))


def test_actual_no_overflow_flags_are_part_of_the_proof():
    before, text = pair()
    after = parse_mlir_text(text)
    add = next(op for op in after.walk() if type(op) is arith.AddiOp)
    # A poison-producing overflow promise must not be erased by symbolic algebra.
    add.properties["overflowFlags"] = arith.IntegerOverflowAttr(frozenset((arith.IntegerOverflowFlag.NSW,)))
    with pytest.raises(ValueError, match="overflow"):
        verify_padding_gather_rewrite(parse_mlir_text(before), after, index_bits=64)


def test_surviving_output_empty_keeps_its_original_identity():
    before, text = pair()
    after = parse_mlir_text(text)
    output = next(op for op in after.walk() if type(op) is tensor.EmptyOp)
    output.attributes["unreviewed"] = builtin.UnitAttr()
    with pytest.raises(ValueError):
        verify_padding_gather_rewrite(parse_mlir_text(before), after, index_bits=64)


def test_unchanged_unknown_enclosing_contract_is_not_a_rewrite_permission():
    before, after = pair()
    before = before.replace("module {", "module attributes {unknown.ordering} {")
    after = after.replace("module {", "module attributes {unknown.ordering} {")
    assert check(before, before) == ()
    with pytest.raises(ValueError, match="enclosing"):
        check(before, after)


def test_scalar_then_branch_must_not_contain_an_extra_hidden_observer():
    before, after = pair()
    after = after.replace("%ih=arith.subi", '"unknown.observe"():()->()\n%ih=arith.subi')
    with pytest.raises(ValueError):
        check(before, after)


@pytest.mark.parametrize("bits", [True, None, 1, 129])
def test_index_width_is_an_original_explicit_bounded_integer(bits):
    before, after = pair()
    with pytest.raises(ValueError, match="index width"):
        check(before, after, bits)
