"""Element-precise static views preserve type, fanout and bounded row tails."""

import numpy as np
import pytest
from xdsl.dialects import tensor
from xdsl.dialects.builtin import DenseArrayBase, i64
from xdsl.parser import Parser

from merlin.llvmlower.segmented_matrix_view import SegmentedRows, prove_segmented_matrix
from merlin.xdsl_dialects._common import make_context, text


def fixture(h=9, w=13, c=5, oh=3, ow=4, sh=2, sw=3, y=1, x=2, dtype="i8", nested=False):
    source = f"tensor<1x{h}x{w}x{c}x{dtype}>"
    sliced = f"tensor<1x{oh}x{ow}x{c}x{dtype}>"
    flat = f"tensor<{oh * ow * c}x{dtype}>"
    matrix = f"tensor<{oh * ow}x{c}x{dtype}>"
    outer = f"tensor<1x17x29x{c}x{dtype}>" if nested else source
    prefix = (
        f"""%p = "tensor.extract_slice"(%a) <{{
        static_offsets = array<i64: 0, 2, 3, 0>, static_sizes = array<i64: 1, {h}, {w}, {c}>,
        static_strides = array<i64: 1, 1, 2, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
      }}> : ({outer}) -> {source}"""
        if nested
        else ""
    )
    operand = "%p" if nested else "%a"
    module = Parser(
        make_context(tensor.Tensor),
        f"""module {{
      func.func @f(%a: {outer}) -> ({outer}, {matrix}) {{
        {prefix}
        %s = "tensor.extract_slice"({operand}) <{{
          static_offsets = array<i64: 0, {y}, {x}, 0>, static_sizes = array<i64: 1, {oh}, {ow}, {c}>,
          static_strides = array<i64: 1, {sh}, {sw}, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
        }}> : ({source}) -> {sliced}
        %f = "tensor.collapse_shape"(%s) <{{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}}>
          : ({sliced}) -> {flat}
        %v = "tensor.expand_shape"(%f) <{{reassociation = [[0 : i64, 1 : i64]],
          static_output_shape = array<i64: {oh * ow}, {c}>}}> : ({flat}) -> {matrix}
        return %a, %v : {outer}, {matrix}
      }}
    }}""",
    ).parse_module()
    module.verify()
    value = next(op.results[0] for op in module.walk() if op.name == "tensor.expand_shape")
    return module, value


@pytest.mark.parametrize("dtype,width", [("i8", 1), ("i32", 4), ("f32", 4)])
def test_non_square_offset_view_matches_every_numpy_element_without_edits(dtype, width):
    module, value = fixture(dtype=dtype)
    before = text(module, generic=True)
    proof = prove_segmented_matrix(value)
    assert proof.address.element_bytes == width
    assert proof.address.dtype == dtype
    assert proof.owner is value.owner.operands[0].owner.operands[0].owner.operands[0]
    a = np.arange(9 * 13 * 5).reshape(1, 9, 13, 5)
    expected = a[:, 1:7:2, 2:14:3, :].reshape(12, 5)
    for m in range(12):
        for k in range(5):
            assert a.flat[proof.address.offset(m, k)] == expected[m, k]
    assert text(module, generic=True) == before
    assert proof.to_dict()["physical_storage"].startswith("Requires")


def test_split_rows_handles_panels_spanning_many_boundaries_and_last_tail():
    view = SegmentedRows(33, 65, 7, 131, 2000, 10000, 1, "i8", 5)
    assert list(view.split_rows(0, 16)) == [(0, 7, 5), (7, 7, 2005), (14, 2, 4005)]
    assert list(view.split_rows(30, 3)) == [(30, 3, 8267)]
    with pytest.raises(ValueError, match="exceeds"):
        list(view.split_rows(32, 2))


def test_uniform_physical_pitch_coalesces_logical_segment_boundaries():
    view = SegmentedRows(33, 65, 7, 131, 917, 5000, 1, "i8", 5)
    assert list(view.split_rows(3, 16)) == [(3, 16, 398)]


def test_nested_slice_offsets_and_strides_compose_in_source_elements():
    _, value = fixture(nested=True)
    proof = prove_segmented_matrix(value)
    a = np.arange(17 * 29 * 5).reshape(1, 17, 29, 5)
    expected = a[:, 2:11, 3:29:2, :][:, 1:7:2, 2:14:3, :].reshape(12, 5)
    assert proof.address.row_stride == 30
    assert proof.address.segment_stride == 290
    for m in range(12):
        for k in range(5):
            assert a.flat[proof.address.offset(m, k)] == expected[m, k]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"source_elements": 100}, "endpoint"),
        ({"row_stride": 3}, "rows overlap"),
        ({"segment_stride": 3}, "segments overlap"),
        ({"element_bytes": 0}, "positive"),
        ({"origin": -1}, "nonnegative"),
    ],
)
def test_address_contract_refuses_overlap_or_out_of_bounds(changes, reason):
    values = dict(
        rows=12,
        cols=5,
        segment_rows=4,
        row_stride=15,
        segment_stride=130,
        source_elements=585,
        element_bytes=1,
        dtype="i8",
        origin=15,
    )
    values.update(changes)
    with pytest.raises(ValueError, match=reason):
        SegmentedRows(**values)


def test_column_stride_and_invalid_slice_endpoint_refuse_without_mutation():
    module, value = fixture()
    slice_op = next(op for op in module.walk() if op.name == "tensor.extract_slice")
    slice_op.properties["static_strides"] = DenseArrayBase.from_list(i64, [1, 2, 3, 2])
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="endpoint"):
        prove_segmented_matrix(value)
    assert text(module, generic=True) == before
    slice_op.properties["static_strides"] = DenseArrayBase.from_list(i64, [1, 2, 3, 1])
    slice_op.properties["static_offsets"] = DenseArrayBase.from_list(i64, [0, 9, 2, 0])
    with pytest.raises(ValueError, match="endpoint"):
        prove_segmented_matrix(value)


def test_dynamic_rank_and_plain_matrix_refuse():
    _, value = fixture()
    with pytest.raises(ValueError, match="flat-to-matrix"):
        prove_segmented_matrix(value.owner.operands[0])


def grouped_query_view(m=16, groups=4, heads=8, k=64, group=2):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    owner = f"tensor<{m}x{groups}x{heads}x{k}xi8>"
    sliced = f"tensor<{m}x1x{heads}x{k}xi8>"
    permuted = f"tensor<1x{heads}x{m}x{k}xi8>"
    flat = f"tensor<{heads*m*k}xi8>"
    matrix = f"tensor<{heads*m}x{k}xi8>"
    module = parse_mlir_text(f'''module {{func.func @forward(%q:{owner})->{matrix} {{
      %s=tensor.extract_slice %q[0,{group},0,0][{m},1,{heads},{k}][1,1,1,1]
        :{owner} to {sliced}
      %e=tensor.empty():{permuted}
      %t=linalg.transpose ins(%s:{sliced}) outs(%e:{permuted}) permutation=[1,2,0,3]
      %f=tensor.collapse_shape %t [[0,1,2,3]]:{permuted} into {flat}
      %v=tensor.expand_shape %f [[0,1]] output_shape [{heads*m},{k}]:{flat} into {matrix}
      return %v:{matrix}
    }}}}''')
    module.verify()
    value = next(op.results[0] for op in module.walk() if op.name == "tensor.expand_shape")
    return module, value


@pytest.mark.parametrize("m,groups,heads,k,group", [(16, 4, 8, 64, 2), (19, 2, 3, 33, 1), (1, 3, 2, 5, 0)])
def test_source_permuted_query_view_has_exact_disjoint_interleaved_rows(m, groups, heads, k, group):
    module, value = grouped_query_view(m, groups, heads, k, group)
    original = text(module, generic=True)
    proof = prove_segmented_matrix(value)
    source = np.arange(m * groups * heads * k).reshape(m, groups, heads, k)
    expected = source[:, group:group+1, :, :].transpose(1, 2, 0, 3).reshape(heads*m, k)
    offsets = [proof.address.offset(row, col) for row in range(heads*m) for col in range(k)]
    assert len(set(offsets)) == expected.size
    np.testing.assert_array_equal(source.ravel()[offsets].reshape(expected.shape), expected)
    assert proof.address.row_stride == groups * heads * k
    assert proof.address.segment_stride == k
    assert proof.address.origin == group * heads * k
    assert proof.owner is next(iter(module.body.block.ops)).body.block.args[0]
    assert text(module, generic=True) == original
    runs = list(proof.address.split_rows(max(0, m-1), min(m+2, heads*m-max(0, m-1))))
    for start, count, offset in runs:
        for step in range(count):
            assert offset + step * proof.address.row_stride == proof.address.offset(start+step)


def test_interleaved_partial_last_segment_checks_earlier_physical_endpoint():
    # Logical last row is physical offset4; the preceding full segment ends103.
    with pytest.raises(ValueError, match="endpoint"):
        SegmentedRows(4, 4, 3, 50, 4, 103, 1, "i8")
    valid = SegmentedRows(4, 4, 3, 50, 4, 108, 1, "i8")
    assert list(valid.split_rows(0, 4)) == [(0, 3, 0), (3, 1, 4)]
    with pytest.raises(ValueError, match="segments overlap"):
        SegmentedRows(6, 5, 3, 12, 3, 100, 1, "i8")


def test_non_identity_transpose_scalar_region_refuses():
    from xdsl.dialects.linalg.ops import TransposeOp

    module, value = grouped_query_view()
    transpose = next(op for op in module.walk() if isinstance(op, TransposeOp))
    transpose.body.block.last_op.operands = (transpose.body.block.args[1],)
    with pytest.raises(ValueError, match="changes source values"):
        prove_segmented_matrix(value)
