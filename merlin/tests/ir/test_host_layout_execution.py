"""Selected physical layouts preserve complete integer tensors and scalar work.

These are typed-IR/interpreter tests. Materialization bytes count logical tensor
payloads, not executed machine instructions, physical traffic, or latency.
"""

import hashlib
from math import prod

import numpy as np
import pytest
from xdsl.dialects.builtin import i32
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.interpreters.utils.ptr import TypedPtr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.graph_optimizations import GraphOptimizationContract, optimize_host_graph


def _type(shape):
    return "tensor<" + "x".join((*map(str, shape), "i32")) + ">"


def _source(shape, permutation, bias_shape=None, *, shared=False):
    """An explicit logical transpose/add/inverse chain with independent outputs."""
    rank = len(shape)
    moved = tuple(shape[i] for i in permutation)
    original, transposed = _type(shape), _type(moved)
    dimensions = [f"d{i}" for i in range(rank)]

    def mapping(axes):
        return "affine_map<(" + ",".join(dimensions) + ")->(" + ",".join(axes) + ")>"

    same = mapping(dimensions)
    iterators = ",".join(['"parallel"'] * rank)
    inverse = ",".join(str(permutation.index(i)) for i in range(rank))
    extra_arguments = ""
    inputs, input_types, maps = "%t", transposed, [same, same]
    block_arguments = "%v:i32,%unused:i32"
    body = "%sum=arith.addi %v,%one:i32"
    if not shared:
        extra_arguments = f",%bias:{_type(bias_shape)},%scalar:tensor<i32>"
        if len(bias_shape) < rank:
            bias_map = mapping(dimensions[-len(bias_shape) :])
        else:
            bias_map = mapping([f"d{i}" if size != 1 else "0" for i, size in enumerate(bias_shape)])
        inputs, input_types = "%t,%bias,%scalar", f"{transposed},{_type(bias_shape)},tensor<i32>"
        maps = [same, bias_map, mapping([]), same]
        block_arguments = "%v:i32,%biasv:i32,%scalarv:i32,%unused:i32"
        body = "%partial=arith.addi %v,%biasv:i32\n%sum=arith.addi %partial,%scalarv:i32"
    second = ""
    result_types, returned = original, "%back"
    if shared:
        second = f"""
%right_init=tensor.empty():{transposed}
%right=linalg.generic {{indexing_maps=[{same},{same}],iterator_types=[{iterators}]}}
 ins(%a:{transposed}) outs(%right_init:{transposed}) {{
 ^bb0(%right_v:i32,%right_unused:i32):
 %right_sum=arith.addi %right_v,%one:i32
 linalg.yield %right_sum:i32
}}->{transposed}
%back2_init=tensor.empty():{original}
%back2=linalg.transpose ins(%right:{transposed}) outs(%back2_init:{original}) permutation=[{inverse}]
"""
        result_types, returned = f"({original},{original})", "%back,%back2"
    return f"""module {{
func.func @kernel(%x:{original}{extra_arguments})->{result_types} {{
%one=arith.constant 2147483647:i32
%init=tensor.empty():{transposed}
%t=linalg.transpose ins(%x:{original}) outs(%init:{transposed}) permutation=[{",".join(map(str, permutation))}]
%out=tensor.empty():{transposed}
%a=linalg.generic {{indexing_maps=[{",".join(maps)}],iterator_types=[{iterators}]}}
 ins({inputs}:{input_types}) outs(%out:{transposed}) {{
 ^bb0({block_arguments}):
 {body}
 linalg.yield %sum:i32
}}->{transposed}
%back_init=tensor.empty():{original}
%back=linalg.transpose ins(%a:{transposed}) outs(%back_init:{original}) permutation=[{inverse}]
{second}
func.return {returned}:{original + "," + original if shared else original}
}}
}}"""


def _execute(module, values):
    module.verify()
    interpreter = Interpreter(module)
    for implementation in (ArithFunctions(), FuncFunctions(), LinalgFunctions(), TensorFunctions()):
        interpreter.register_implementations(implementation)
    arguments = [ShapedArray(TypedPtr.new(value.flatten().tolist(), xtype=i32), list(value.shape)) for value in values]
    return tuple(
        np.asarray(value.data, dtype=np.int32).reshape(value.shape)
        for value in interpreter.call_op("kernel", arguments)
    )


def _wrap_i32(values):
    # Python integer arithmetic is independent of the interpreter's scalar rules.
    values = np.asarray(values, dtype=object)
    return np.asarray([((int(v) + (1 << 31)) % (1 << 32)) - (1 << 31) for v in values.flat], dtype=np.int32).reshape(
        values.shape
    )


def _observed_work(module):
    transposes = [op for op in module.walk() if type(op) is TransposeOp]
    generics = [op for op in module.walk() if type(op) is GenericOp]
    return (
        len(transposes),
        sum(prod(op.results[0].type.get_shape()) * 4 for op in transposes),
        sum(
            prod(op.results[0].type.get_shape()) * sum(scalar.name == "arith.addi" for scalar in op.body.block.ops)
            for op in generics
        ),
        len(generics),
    )


@pytest.mark.parametrize(
    "shape,permutation,bias_rank,shared",
    [
        ((2, 3, 4), (1, 2, 0), None, False),
        ((2, 2, 2), (1, 2, 0), None, False),
        ((2, 3, 4), (2, 0, 1), 1, False),
        ((2, 3, 4), (1, 2, 0), 2, False),
        ((1, 2, 3, 4), (0, 2, 3, 1), None, False),
        ((2, 3, 4), (1, 2, 0), None, True),
    ],
    ids=["cyclic-rectangular", "cyclic-equal-extents", "rank1-bias", "rank2-bias", "unit-zero-broadcast", "shared"],
)
def test_selected_layout_executes_complete_modular_outputs(shape, permutation, bias_rank, shared):
    moved = tuple(shape[i] for i in permutation)
    bias_shape = (
        tuple(size if i == len(shape) - 1 else 1 for i, size in enumerate(moved))
        if bias_rank is None
        else moved[-bias_rank:]
    )
    source = _source(shape, permutation, bias_shape, shared=shared)
    selected, report = optimize_host_graph(
        source,
        GraphOptimizationContract(
            hashlib.sha256(source.encode()).hexdigest(),
            "kernel",
            propagate_layout=True,
            physical_layout=True,
            independent_element_order_unobserved=True,
        ),
    )
    original, optimized = parse_mlir_text(source), parse_mlir_text(selected)
    x = np.resize(np.array([-(1 << 31), (1 << 31) - 1, -1, 0, 1, 12345], dtype=np.int32), shape)
    if shared:
        values = [x]
        first = _wrap_i32(x.astype(object) + ((1 << 31) - 1))
        expected = (first, _wrap_i32(first.astype(object) + ((1 << 31) - 1)))
    else:
        bias = np.resize(np.array([(1 << 31) - 1, -11, -(1 << 31), 23], dtype=np.int32), bias_shape)
        scalar = np.array((1 << 31) - 1, dtype=np.int32)
        values = [x, bias, scalar]
        inverse = tuple(permutation.index(i) for i in range(len(shape)))
        expected = (
            _wrap_i32(x.transpose(permutation).astype(object) + bias.astype(object) + int(scalar)).transpose(inverse),
        )
    for module in (original, optimized):
        actual = _execute(module, values)
        assert len(actual) == len(expected)
        for output, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(output, reference)
    before, after = _observed_work(original), _observed_work(optimized)
    assert before[0] == (3 if shared else 2) and after[0] == 0
    assert before[2] == after[2] == 2 * prod(shape)
    assert before[3] == after[3] == (2 if shared else 1)
    assert report["layout"]["before_transpose_bytes"] == before[1]
    assert report["layout"]["after_transpose_bytes"] == after[1]
    assert any(row["status"] == "applied" for row in report["layout"]["rewrite"]["islands"])
    assert report["timing"] == "UNKNOWN" and report["hardware_conformance"] == "not_established"
