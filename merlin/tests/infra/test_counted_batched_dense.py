"""Counted product offsets drive real emission, including shared plane views."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from merlin.compile.scheduling.batched_dense import (
    BatchedDenseProblem,
    BatchMatrixView,
    counted_batched_dense_products,
)
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.llvmlower.host_builder import FnBuilder
from merlin.sched.ir import check_structure, concretize, instances
from merlin.sched.ir.parse import parse_kernel

CASES = (
    BatchedDenseProblem((5,), BatchMatrixView((5, 3, 7), (0,)),
                        BatchMatrixView((5, 7, 19), (0,))),
    # Independent planes share the other operand and retain every combination.
    BatchedDenseProblem((2, 3, 5), BatchMatrixView((2, 5, 3, 7), (0, 2)),
                        BatchMatrixView((3, 5, 7, 4), (1, 2))),
    # Physical batch dimensions need not have logical order. RHS is broadcast.
    BatchedDenseProblem((2, 3), BatchMatrixView((3, 2, 7, 1), (1, 0), True),
                        BatchMatrixView((5, 7), (), True)),
    # Neither read physically has either batch axis; outputs remain distinct.
    BatchedDenseProblem((3, 2), BatchMatrixView((19, 13), ()),
                        BatchMatrixView((13, 7), ())),
)


class ExecutingBuilder:
    def loop(self, start, stop, step, body):
        for index in range(start, stop, step):
            body(index)

    def op(self, opcode, lhs, rhs):
        assert opcode in ("add", "mul")
        return lhs + rhs if opcode == "add" else lhs * rhs


def reference_offsets(problem):
    for at in product(*(range(n) for n in problem.batch_shape)):
        projected = []
        for view in (problem.lhs, problem.rhs):
            physical = (*tuple(at[axis] for axis in view.batch_axes), 0, 0)
            projected.append(int(np.ravel_multi_index(physical, view.shape)))
        logical = int(np.ravel_multi_index(at, problem.batch_shape))
        yield {"product": logical, "lhs_offset": projected[0],
               "rhs_offset": projected[1], "output_offset": logical * problem.m * problem.n}


@pytest.mark.parametrize("problem", CASES)
def test_emitted_element_offsets_cover_every_original_product_and_output(problem):
    kernel = counted_batched_dense_products(problem, name="batch.products", instruction="matrix")
    assert not check_structure(kernel)
    assert parse_kernel(kernel.text()) == kernel
    assert dict(kernel.attrs)["offset_unit"] == "elements"
    original = list(reference_offsets(problem))
    assert [concretize(call, env) for call, env in instances(kernel)] == original
    a = (np.arange(np.prod(problem.lhs.shape)).reshape(problem.lhs.shape) % 13 - 6).astype(np.int8)
    b = (np.arange(np.prod(problem.rhs.shape)).reshape(problem.rhs.shape) % 11 - 5).astype(np.int8)
    output = np.full(np.prod(problem.output_shape) + 6, -777, np.int32)
    written = np.zeros(output.size, np.int32)
    emitted = []

    def lower(instruction, values):
        assert instruction == "matrix"
        emitted.append(values)
        operands = []
        for array, view, key in ((a, problem.lhs, "lhs_offset"), (b, problem.rhs, "rhs_offset")):
            offset = values[key]
            size = np.prod(view.shape[-2:])
            matrix = array.reshape(-1)[offset:offset + size].reshape(view.shape[-2:])
            operands.append(matrix.T if view.transposed else matrix)
        offset = values["output_offset"]
        size = problem.m * problem.n
        output[offset:offset + size] = (operands[0].astype(np.int64) @ operands[1].astype(np.int64)).reshape(-1)
        written[offset:offset + size] += 1

    receipt = emit_index_program(kernel, ExecutingBuilder(), lower)
    assert emitted == original
    assert receipt["kernel_sha256"] == kernel.digest()
    expected = np.empty(problem.output_shape, np.int32)
    for at in product(*(range(n) for n in problem.batch_shape)):
        aa = a[tuple(at[axis] for axis in problem.lhs.batch_axes)]
        bb = b[tuple(at[axis] for axis in problem.rhs.batch_axes)]
        expected[at] = (aa.T if problem.lhs.transposed else aa).astype(np.int64) @ (
            bb.T if problem.rhs.transposed else bb).astype(np.int64)
    np.testing.assert_array_equal(output[:-6].reshape(problem.output_shape), expected)
    np.testing.assert_array_equal(written[:-6], 1)
    np.testing.assert_array_equal(output[-6:], -777)
    np.testing.assert_array_equal(written[-6:], 0)


@pytest.mark.parametrize("problem", CASES)
def test_actual_llvm_consumer_keeps_batch_axes_counted(problem):
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp, i64

    kernel = counted_batched_dense_products(problem, name="batch", instruction="matrix")
    builder = FnBuilder([])
    bridge = FnBuilderIndexAdapter(builder)

    def lower(instruction, values):
        assert instruction == "matrix"
        builder.add(llvm.CallOp("original_product", *(bridge.value(values[key]) for key in
            ("product", "lhs_offset", "rhs_offset", "output_offset"))))

    emitted = emit_index_program(kernel, bridge, lower)
    module = ModuleOp([
        llvm.FuncOp("original_product", llvm.LLVMFunctionType([i64] * 4, llvm.LLVMVoidType())),
        llvm.FuncOp("batch", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish()),
    ])
    module.verify()
    assert emitted["kernel_sha256"] == kernel.digest()
    assert sum(isinstance(op, llvm.CallOp) for op in module.walk()) == 1
    assert sum(isinstance(op, llvm.CondBrOp) for op in module.walk()) == len(problem.batch_shape)


def test_fresh_problem_validation_and_explicit_names_are_required():
    original = CASES[0]
    for values in ({"name": ""}, {"instruction": "bad\ncall"}, {"name": 3}):
        with pytest.raises(ValueError, match="explicit"):
            counted_batched_dense_products(original, **(dict(name="batch", instruction="matrix") | values))
    changed = replace(original, lhs=replace(original.lhs))
    object.__setattr__(changed.lhs, "batch_axes", (1,))
    with pytest.raises(ValueError, match="outside"):
        counted_batched_dense_products(changed, name="batch", instruction="matrix")
