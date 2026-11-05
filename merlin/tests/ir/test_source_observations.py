import numpy as np
import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_region import append_source_observations
from merlin.runtime.linalg_numpy import evaluate
from merlin.xdsl_dialects._common import text


def source():
    module = parse_mlir_text('''module {
      func.func @forward(%a:tensor<2x3xi32>,%b:tensor<2x3xi32>) -> tensor<2x3xi32> {
        %x=arith.addi %a,%b:tensor<2x3xi32>
        %y=arith.subi %a,%b:tensor<2x3xi32>
        %z=arith.muli %x,%y:tensor<2x3xi32>
        return %z:tensor<2x3xi32>
      }}''')
    owner = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    return module, owner, tuple(owner.body.block.ops)


def test_original_results_operations_and_requested_observation_order_remain_exact():
    module, owner, ops = source()
    before = text(module, generic=True)
    result = append_source_observations(module, (ops[1].result, ops[0].result), max_elements=12)
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    b = np.full((2, 3), 3, dtype=np.int32)
    got = evaluate(result.module, [a, b])
    assert len(got) == 3
    assert np.array_equal(got[0], evaluate(module, [a, b])[0])
    assert np.array_equal(got[1], a-b)
    assert np.array_equal(got[2], a+b)
    assert text(module, generic=True) == before
    copied = next(op for op in result.module.body.block.ops if isinstance(op, func.FuncOp))
    assert [text(op, generic=True) for op in tuple(copied.body.block.ops)[:-1]] == [
        text(op, generic=True) for op in ops[:-1]]
    assert result.receipt['observed_elements'] == 12
    assert result.receipt['original_result_count'] == 1
    assert result.receipt['intermediate_lifetimes_extended']
    assert not result.receipt['effect_or_target_admission']
    assert not result.receipt['timing_equivalence']


@pytest.mark.parametrize('case', ['foreign', 'returned', 'input', 'duplicate', 'mutable', 'bound'])
def test_invalid_observations_refuse_without_source_changes(case):
    module, owner, ops = source()
    values = (ops[0].result,)
    bound = 12
    if case == 'foreign':
        values = (source()[2][0].result,)
    elif case == 'returned':
        values = (ops[2].result,)
    elif case == 'input':
        values = (owner.body.block.args[0],)
    elif case == 'duplicate':
        values = (ops[0].result, ops[0].result)
    elif case == 'mutable':
        values = [ops[0].result]
    else:
        bound = 5
    before = text(module, generic=True)
    with pytest.raises(ValueError, match='observation'):
        append_source_observations(module, values, max_elements=bound)
    assert text(module, generic=True) == before


def test_calls_are_retained_in_order_without_new_effect_authority():
    module = parse_mlir_text('''module {
      func.func private @external(tensor<2xi32>)->tensor<2xi32>
      func.func @forward(%a:tensor<2xi32>)->tensor<2xi32> {
        %x=func.call @external(%a):(tensor<2xi32>)->tensor<2xi32>
        %y=arith.addi %x,%a:tensor<2xi32>
        return %y:tensor<2xi32>
      }}''')
    owner = tuple(module.body.block.ops)[1]
    result = append_source_observations(module, (owner.body.block.first_op.results[0],))
    assert len(tuple(result.module.body.block.ops)) == 2
    assert tuple(result.module.body.block.ops)[1].body.block.first_op.name == 'func.call'
    assert not result.receipt['effect_or_target_admission']
