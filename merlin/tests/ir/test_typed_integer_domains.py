import pytest
from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IntegerAttr, TensorType, i8, i64
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.ir import Block, Region

from merlin.llvmlower.typed_integer_domains import SignedByteDomain, trace_signed_byte_domain


def argument():
    block = Block(arg_types=[TensorType(i8, [16, 64])])
    function = func.FuncOp("independent", ([block.args[0].type], []), Region(block))
    return function, block.args[0]


def fact(value):
    return SignedByteDomain(value, 0, 127, dict(minimum=0, maximum=127, source="caller proof"))


def reassociation(axes):
    return ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in group]) for group in axes])


def test_unknown_source_retains_entire_type_even_with_unrelated_facts():
    _, value = argument()
    _, other = argument()
    assert trace_signed_byte_domain(value, [fact(other)])["minimum"] == -128


def test_actual_ssa_fact_survives_verified_views_and_transpose():
    _, value = argument()
    collapse = tensor.CollapseShapeOp(
        operands=[value], result_types=[TensorType(i8, [1024])], properties={"reassociation": reassociation([[0, 1]])}
    )
    expand = tensor.ExpandShapeOp(collapse.results[0], [], reassociation([[0, 1]]), [32, 32], TensorType(i8, [32, 32]))
    init = tensor.EmptyOp([], TensorType(i8, [32, 32]))
    transpose = TransposeOp(expand.result, init.tensor, DenseArrayBase.from_list(i64, [1, 0]))
    result = trace_signed_byte_domain(transpose.results[0], [fact(value)])
    assert (result["minimum"], result["maximum"]) == (0, 127)
    assert [row["operation"] for row in result["path"]] == [
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "linalg.transpose",
    ]


def test_contradictory_fact_and_duplicates_refuse():
    _, value = argument()
    with pytest.raises(ValueError, match="contradicts"):
        SignedByteDomain(value, 0, 127, dict(minimum=-128, maximum=127))
    with pytest.raises(ValueError, match="duplicate"):
        trace_signed_byte_domain(value, [fact(value), fact(value)])


def test_invalid_view_cannot_grant_tighter_domain():
    _, value = argument()
    invalid = tensor.CollapseShapeOp(
        operands=[value], result_types=[TensorType(i8, [1000])], properties={"reassociation": reassociation([[0, 1]])}
    )
    with pytest.raises(Exception):
        trace_signed_byte_domain(invalid.results[0], [fact(value)])


@pytest.mark.parametrize("bounds", [(False, 127), (-129, 127), (0, -1), (0.0, 127)])
def test_invalid_intervals_refuse(bounds):
    _, value = argument()
    with pytest.raises(ValueError):
        SignedByteDomain(value, *bounds, dict(minimum=bounds[0], maximum=bounds[1]))
