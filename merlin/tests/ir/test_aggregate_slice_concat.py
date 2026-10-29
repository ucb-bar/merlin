"""Complete matrix slices reuse their aggregate without losing live observers."""

import numpy as np
import pytest
from xdsl.dialects import tensor

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.tensor_views import fold_complete_matrix_slice_concats
from merlin.xdsl_dialects._common import text


def source(order=(0, 1, 2, 3), *, rows=3, observer=True):
    statements = []
    for index in range(4):
        statements += [f"""
          %s{index}=tensor.extract_slice %x[{index},0,0][1,{rows},5][1,1,1]
            :tensor<4x3x5xi32> to tensor<1x{rows}x5xi32>
          %c{index}=tensor.collapse_shape %s{index} [[0,1],[2]]
            :tensor<1x{rows}x5xi32> into tensor<{rows}x5xi32>"""]
    inputs = ','.join(f'%c{i}' for i in order)
    types = ','.join(f'tensor<{rows}x5xi32>' for _ in order)
    result = f'tensor<{len(order)*rows}x5xi32>'
    returned = f'%cat,%c1:{result},tensor<{rows}x5xi32>' if observer else f'%cat:{result}'
    return_types = f'({result},tensor<{rows}x5xi32>)' if observer else result
    return (f'module {{func.func @forward(%x:tensor<4x3x5xi32>)->{return_types} {{'
            + '\n'.join(statements)
            + f'\n%cat=tensor.concat dim(0) {inputs}:({types})->{result}\nreturn {returned}\n}}}}')


def test_complete_order_and_independent_observer_are_preserved():
    module = parse_mlir_text(source())
    observer = next(op for op in module.walk() if op.name == 'func.return').operands[1]
    assert fold_complete_matrix_slice_concats(module) == 1
    returned = next(op for op in module.walk() if op.name == 'func.return')
    assert returned.operands[1] is observer
    assert returned.operands[0].owner.operands[0] is next(
        op for op in module.walk() if op.name == 'func.func').body.block.args[0]
    assert not any(type(op) is tensor.ConcatOp for op in module.walk())
    assert fold_complete_matrix_slice_concats(module) == 0


@pytest.mark.parametrize('kwargs', [
    {'order': (1, 0, 2, 3)}, {'order': (0, 0, 2, 3)},
    {'order': (0, 1, 2)}, {'rows': 2},
])
def test_reordered_duplicated_incomplete_or_partial_coverage_keeps_concat(kwargs):
    module = parse_mlir_text(source(**kwargs))
    assert fold_complete_matrix_slice_concats(module) == 0
    assert any(type(op) is tensor.ConcatOp for op in module.walk())


def test_distinct_aggregate_and_unknown_attributes_keep_concat():
    from xdsl.dialects import builtin

    for fault in ('aggregate', 'attribute'):
        module = parse_mlir_text(source())
        view = next(op for op in module.walk() if type(op) is tensor.ExtractSliceOp)
        if fault == 'aggregate':
            empty = tensor.EmptyOp([], view.source.type)
            view.parent.insert_op_before(empty, view)
            view.operands = [empty.tensor]
        else:
            view.attributes['unknown_layout'] = builtin.UnitAttr()
        assert fold_complete_matrix_slice_concats(module) == 0


def test_separate_plane_aggregates_and_all_plane_outputs_survive():
    module = parse_mlir_text(source())
    function = next(op for op in module.walk() if op.name == 'func.func')
    copied = function.clone()
    from xdsl.dialects import builtin

    copied.properties['sym_name'] = builtin.StringAttr('another_plane')
    module.body.block.add_op(copied)
    assert fold_complete_matrix_slice_concats(module) == 2
    for fn in (function, copied):
        assert len(fn.body.block.last_op.operands) == 2
        assert fn.body.block.last_op.operands[0].owner.operands[0] is fn.body.block.args[0]


def test_ordinary_head_projection_folds_aggregate_and_keeps_slice_observer():
    from test_head_source_route import original
    from xdsl.dialects import builtin

    from merlin.llvmlower.head_source_route import project_heads

    module, _ = original()
    function = module.body.block.first_op
    returned = function.body.block.last_op
    observer = returned.operands[1]
    concat = tensor.ConcatOp(tuple(returned.operands), 0, builtin.TensorType(builtin.i32, (6, 5)))
    function.body.block.insert_op_before(concat, returned)
    returned.operands = [concat.result, observer]
    function.properties['function_type'] = builtin.FunctionType.from_lists(
        function.function_type.inputs, [value.type for value in returned.operands])
    module.verify()
    before = text(module, generic=True)
    projection = project_heads(module)
    assert len(projection.members) == 1 and projection.members[0]['batch_extent'] == 3
    assert not any(type(op) is tensor.ConcatOp for op in projection.module.walk())
    assert len(projection.module.body.block.first_op.body.block.last_op.operands) == 2
    assert text(module, generic=True) == before


def test_native_complete_output_and_independent_slice_are_bit_exact(tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    if not toolchain.available():
        pytest.skip('selected native lowering toolchain unavailable')
    module = parse_mlir_text(source())
    before = text(module, generic=True)
    assert fold_complete_matrix_slice_concats(module) == 1
    after = text(module, generic=True)
    x = (np.arange(60, dtype=np.int32) * -1234567).reshape(4, 3, 5)
    for arm, program in (('baseline', before), ('folded', after)):
        built = lower_model(program, tmp_path / arm, targets=('host',))
        invoke = HostModel.load(str(built.host_so),
                               image_policy=PrivateHostImagePolicy(built.host_so.parent.resolve()))
        aggregate = np.empty((12, 5), dtype=np.int32)
        observed = np.empty((3, 5), dtype=np.int32)
        invoke([(a.ctypes.data, a.shape) for a in (x, aggregate, observed)])
        np.testing.assert_array_equal(aggregate, x.reshape(12, 5))
        np.testing.assert_array_equal(observed, x[1])
