"""Pointwise views use the existing dense scheduler and actual LLVM emitter."""
from dataclasses import replace

import numpy as np
import pytest
from test_compact_dense import ADAPTER, fixture
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i64

from merlin.compile.scheduling.compact_dense import schedule_compact_dense
from merlin.compile.scheduling.pointwise_convolution import pointwise_dense_view
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.llvmlower.host_builder import FnBuilder


@pytest.mark.parametrize('shape,columns,dilation', [
    ((2, 3, 5, 7), 9, (1, 1)),
    ((1, 1, 1, 5), 3, (3, 2)),
    ((1, 2, 3, 3), 17, (1, 1)),
])
def test_batch_and_tail_views_reach_same_dense_effect_emitter(shape, columns, dilation):
    window = Window2D(shape, (1, 1), dilation=dilation)
    view = pointwise_dense_view(window, columns, output_shape=(*shape[:3], columns))
    m, k, n = view.dimensions
    assert view.matrix_shapes == ((m, k), (k, n), (m, n))
    inputs = (np.arange(np.prod(shape)) % 11-5).astype(np.int8).reshape(shape)
    weights = (np.arange(k*n) % 7-3).astype(np.int8).reshape(k, n)
    expected = np.empty((*shape[:3], n), dtype=np.int32)
    for b in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                expected[b, y, x] = inputs[b, y, x].astype(np.int32) @ weights.astype(np.int32)
    actual = inputs.reshape(m, k).astype(np.int32) @ weights.astype(np.int32)
    assert np.array_equal(actual.reshape(expected.shape), expected)

    problem, resources, storage = fixture(shape=view.dimensions)
    result = schedule_compact_dense(problem, resources, storage, ADAPTER, name='pointwise',
        dtypes=('i8', 'i8', 'i32'), external_buffers_disjoint=True, static_budget=128)
    assert result is not None
    builder, declarations, callbacks = FnBuilder([]), {}, []

    def lower(instruction, operands):
        declarations[instruction] = len(operands)
        callbacks.append(instruction)
        builder.add(llvm.CallOp('actual_' + instruction, *operands.values()))

    receipt = result.emit_llvm(builder, lower)
    function = llvm.FuncOp('pointwise', llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())
    module = ModuleOp([*(llvm.FuncOp('actual_' + name,
        llvm.LLVMFunctionType([i64]*count, llvm.LLVMVoidType()))
        for name, count in declarations.items()), function])
    module.verify()
    assert callbacks and callbacks[-1] == 'store'
    assert {'lhs_load', 'weight_load', 'preload', 'compute', 'store'} <= set(callbacks)
    assert receipt['machine_emission'] == 'requires_independent_target_qualification'
    assert view.record()['payload_copies'] == 0


@pytest.mark.parametrize('window', [
    Window2D((1, 5, 7, 3), (1, 1), stride=(2, 2)),
    Window2D((1, 5, 7, 3), (1, 1), padding=(1, 0, 0, 0)),
    Window2D((1, 5, 7, 3), (3, 3)),
])
def test_gather_sources_keep_original_convolution_route(window):
    oh, ow = window.output_spatial
    assert pointwise_dense_view(window, 4, output_shape=(oh*ow, 4)) is None


def test_bad_output_shape_and_mutated_current_source_refuse():
    source = Window2D((1, 2, 3, 5), (1, 1))
    with pytest.raises(ValueError, match='result shape'):
        pointwise_dense_view(source, 4, output_shape=(5, 4))
    with pytest.raises(ValueError, match='positive original'):
        pointwise_dense_view(source, True, output_shape=(6, 1))
    view = pointwise_dense_view(source, 4, output_shape=(6, 4))
    object.__setattr__(source, 'input_shape', (1, 3, 2, 5))
    with pytest.raises(ValueError, match='original pointwise source changed'):
        view.record()
    rebuilt = pointwise_dense_view(replace(source), 4, output_shape=(6, 4))
    assert rebuilt.dimensions == (6, 5, 4)
