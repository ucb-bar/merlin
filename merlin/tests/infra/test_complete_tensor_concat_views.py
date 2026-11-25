"""Independent value/order/refusal cases for complete contiguous tensor-view folding."""

import numpy as np
import pytest
from xdsl.dialects import builtin, tensor

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.sibling_source_route import _text
from merlin.transforms import tensor_views as V


def fixture(
    *,
    g=2,
    r=3,
    m=2,
    n=5,
    order=None,
    partial=False,
    other=False,
    stride=1,
    observer=True,
    root_call=False,
    floating=False,
):
    rows = r * m * stride
    root = f"tensor<{g}x{rows}x{n}xi32>"
    parts = []
    lines = [
        f"module {{ func.func @forward(%x:{root},%other:{root}) -> (tensor<{g * r * m}x{n}xi32>,tensor<{m}x{n}xi32>) {{"
    ]
    rootname = "%x"
    if root_call:
        lines.append(f"%opaque=func.call @opaque(%x):({root})->{root}")
        rootname = "%opaque"
    for a in range(g):
        src = "%other" if other and a == g - 1 else rootname
        lines.append(
            f'%group{a}="tensor.extract_slice"({src}) <{{static_offsets=array<i64:{a},0,0>, '
            f"static_sizes=array<i64:1,{rows},{n}>,static_strides=array<i64:1,1,1>,"
            f"operandSegmentSizes=array<i32:1,0,0,0>}}> : ({root})->tensor<1x{rows}x{n}xi32>"
        )
        lines.append(
            f"%matrix{a}=tensor.collapse_shape %group{a} [[0,1],[2]]:"
            f"tensor<1x{rows}x{n}xi32> into tensor<{rows}x{n}xi32>"
        )
        for b in range(r):
            name = f"%part{a}_{b}"
            parts.append(name)
            lines.append(
                f'{name}="tensor.extract_slice"(%matrix{a}) <{{static_offsets=array<i64:{b * m * stride},0>,'
                f"static_sizes=array<i64:{m},{n}>,static_strides=array<i64:{stride},1>,"
                f"operandSegmentSizes=array<i32:1,0,0,0>}}>:(tensor<{rows}x{n}xi32>)->tensor<{m}x{n}xi32>"
            )
    selected = parts if order is None else [parts[i] for i in order]
    if partial:
        selected = selected[:-1]
    target = f"tensor<{len(selected) * m}x{n}xi32>"
    # Keep valid function types even in partial/mixed/refused fixtures.
    lines[0] = f"module {{ func.func @forward(%x:{root},%other:{root}) -> ({target},tensor<{m}x{n}xi32>) {{"
    lines.append(
        "%joined=tensor.concat dim(0) "
        + ",".join(selected)
        + " : ("
        + ",".join([f"tensor<{m}x{n}xi32>"] * len(selected))
        + f")->{target}"
    )
    if floating:
        ftype = f"tensor<{len(selected) * m}x{n}xf32>"
        lines[0] = lines[0].replace(") {", f",{ftype}) {{")
        lines.append(f"""%finit=tensor.empty():{ftype}
%float=linalg.generic {{indexing_maps=[affine_map<(a,b)->(a,b)>,affine_map<(a,b)->(a,b)>],
iterator_types=["parallel","parallel"]}} ins(%joined:{target}) outs(%finit:{ftype}) {{
^bb0(%a:i32,%unused:f32):
%f=arith.sitofp %a:i32 to f32
%c=arith.constant 0.33333334:f32
%d=arith.constant 0.000001:f32
%e=arith.constant -0.125:f32
%mul=arith.mulf %f,%c:f32
%sum=arith.addf %mul,%d:f32
%out=arith.mulf %sum,%e:f32
linalg.yield %out:f32
}} -> {ftype}""")
        lines.append(f"return %joined,{parts[1]},%float:{target},tensor<{m}x{n}xi32>,{ftype}")
    else:
        lines.append(f"return %joined,{parts[1]}:{target},tensor<{m}x{n}xi32>")
    lines.append("}")
    if root_call:
        lines.append(f"func.func private @opaque({root})->{root}")
    lines.append("}")
    module = parse_mlir_text("\n".join(lines))
    module.verify()
    return module


@pytest.mark.parametrize("shape", [(2, 3, 2, 5), (3, 2, 1, 7), (1, 4, 3, 2)])
def test_nested_complete_coverage_preserves_individual_observers(shape):
    source = fixture(g=shape[0], r=shape[1], m=shape[2], n=shape[3])
    old = _text(source)
    call = source.body.block.first_op
    observed = call.body.block.last_op.arguments[1]
    assert V.fold_complete_matrix_slice_concats(source) == 0
    assert _text(source) == old
    assert V.fold_complete_matrix_slice_concats(source, compose_contiguous_views=True) == 1
    assert call.body.block.last_op.arguments[1] is observed
    assert not any(type(op) is tensor.ConcatOp for op in source.walk())
    source.verify()


@pytest.mark.parametrize(
    "kwargs",
    [{"partial": True}, {"order": [1, 0, 2, 3, 4, 5]}, {"order": [0, 1, 2, 3, 4, 4]}, {"other": True}, {"stride": 2}],
)
def test_partial_reordered_duplicate_mixedroot_strided_refuse(kwargs):
    source = fixture(**kwargs)
    before = _text(source)
    assert V.fold_complete_matrix_slice_concats(source, compose_contiguous_views=True) == 0
    assert _text(source) == before


@pytest.mark.parametrize("budget", [0, 1, 4])
def test_budget_refusal_is_nonmutating(budget):
    source = fixture()
    before = _text(source)
    assert V.fold_complete_matrix_slice_concats(source, compose_contiguous_views=True, max_view_operations=budget) == 0
    assert _text(source) == before


@pytest.mark.parametrize("kind", ["view", "concat"])
def test_unknown_attributes_refuse(kind):
    source = fixture()
    op = next(op for op in source.walk() if type(op) is (tensor.ExtractSliceOp if kind == "view" else tensor.ConcatOp))
    op.attributes["unknown.policy"] = builtin.UnitAttr()
    before = _text(source)
    assert V.fold_complete_matrix_slice_concats(source, compose_contiguous_views=True) == 0
    assert _text(source) == before


def test_opaque_effectful_producer_is_not_modified_moved_or_rebound():
    source = fixture(root_call=True)
    fn = source.body.block.first_op
    before = list(fn.body.block.ops)
    call = before[0]
    arguments = tuple(call.operands)
    text = _text(call)
    assert V.fold_complete_matrix_slice_concats(source, compose_contiguous_views=True) == 1
    assert list(fn.body.block.ops)[0] is call and tuple(call.operands) == arguments and _text(call) == text


@pytest.mark.parametrize("selection", [1, None, "yes"])
def test_selection_requires_exact_boolean(selection):
    with pytest.raises(ValueError):
        V.fold_complete_matrix_slice_concats(fixture(), compose_contiguous_views=selection)


def test_native_original_and_folded_preserve_i32_bits_and_observed_head(tmp_path):
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    original = fixture(floating=True)
    optimized = original.clone()
    assert V.fold_complete_matrix_slice_concats(optimized, compose_contiguous_views=True) == 1
    data = np.array([0, -1, 2147483647, -2147483648, 16777217, -16777217] * 10, dtype=np.int32).reshape(2, 6, 5)
    original_copy = data.copy()
    wanted = [
        data.reshape(12, 5),
        data[0, 2:4, :],
        ((data.reshape(12, 5).astype(np.float32) * np.float32(0.33333334)) + np.float32(0.000001)) * np.float32(-0.125),
    ]
    for label, source in [("original", original), ("folded", optimized)]:
        lowered = lower_model(_text(source), tmp_path / label)
        invoke = HostModel.load(
            str(lowered.host_so), image_policy=PrivateHostImagePolicy(lowered.host_so.parent.resolve())
        )
        buffers = [np.full(v.size + 16, 91, dtype=v.dtype) for v in wanted]
        outputs = [b[8:-8].reshape(v.shape) for b, v in zip(buffers, wanted, strict=True)]
        invoke([(v.ctypes.data, v.shape) for v in (data, data, *outputs)])
        for got, want, buf in zip(outputs, wanted, buffers, strict=True):
            np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))
            assert np.all(buf[:8] == 91) and np.all(buf[-8:] == 91)
    np.testing.assert_array_equal(data, original_copy)


def test_flatten_expand_reassembly_preserves_row_major_value():
    source = parse_mlir_text("""module {func.func @forward(%x:tensor<2x3x4xi32>)->tensor<3x8xi32> {
 %v=tensor.collapse_shape %x [[0,1,2]]:tensor<2x3x4xi32> into tensor<24xi32>
 %a="tensor.extract_slice"(%v) <{static_offsets=array<i64:0>,static_sizes=array<i64:8>,
 static_strides=array<i64:1>,operandSegmentSizes=array<i32:1,0,0,0>}>:(tensor<24xi32>)->tensor<8xi32>
 %b="tensor.extract_slice"(%v) <{static_offsets=array<i64:8>,static_sizes=array<i64:8>,
 static_strides=array<i64:1>,operandSegmentSizes=array<i32:1,0,0,0>}>:(tensor<24xi32>)->tensor<8xi32>
 %c="tensor.extract_slice"(%v) <{static_offsets=array<i64:16>,static_sizes=array<i64:8>,
 static_strides=array<i64:1>,operandSegmentSizes=array<i32:1,0,0,0>}>:(tensor<24xi32>)->tensor<8xi32>
 %aa=tensor.expand_shape %a [[0,1]] output_shape [1,8]:tensor<8xi32> into tensor<1x8xi32>
 %bb=tensor.expand_shape %b [[0,1]] output_shape [1,8]:tensor<8xi32> into tensor<1x8xi32>
 %cc=tensor.expand_shape %c [[0,1]] output_shape [1,8]:tensor<8xi32> into tensor<1x8xi32>
 %j=tensor.concat dim(0) %aa,%bb,%cc:(tensor<1x8xi32>,tensor<1x8xi32>,tensor<1x8xi32>)->tensor<3x8xi32>
 return %j:tensor<3x8xi32>
}}""")
    assert V.fold_complete_matrix_slice_concats(source, compose_contiguous_views=True) == 1
    source.verify()
    assert type(source.body.block.first_op.body.block.last_op.arguments[0].owner) is tensor.ExpandShapeOp
