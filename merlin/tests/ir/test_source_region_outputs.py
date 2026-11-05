"""Multi-output source replacement preserves original SSA observers and order."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith, func, scf, tensor
from xdsl.parser import Parser

from merlin.llvmlower.source_model import SourceRegion, _region_results
from merlin.xdsl_dialects._common import make_context, text

SOURCE = """module {
func.func @forward(%a:tensor<2x3xi32>,%b:tensor<2x3xi32>)->
 (tensor<2x3xi32>,tensor<2x3xi32>,tensor<2x3xi32>) {
 %first=arith.addi %a,%b:tensor<2x3xi32>
 %second=arith.subi %a,%b:tensor<2x3xi32>
 return %first,%second,%first:tensor<2x3xi32>,tensor<2x3xi32>,tensor<2x3xi32>
}
}"""


def fixture(source=SOURCE):
    module = Parser(make_context(arith.Arith, tensor.Tensor), source).parse_module()
    module.verify()
    function = next(operation for operation in module.body.block.ops if isinstance(operation, func.FuncOp))
    first, second = tuple(function.body.block.ops)[:2]
    region = SourceRegion(
        None,
        (first, second),
        tuple(function.body.block.args),
        (2, 3),
        outputs=(second.results[0], first.results[0]),
    )
    return module, region


def test_explicit_output_order_preserves_all_observers_without_editing():
    module, region = fixture()
    before = text(module, generic=True)
    assert _region_results(region) == region.outputs
    assert len(tuple(region.outputs[1].uses)) == 2
    assert text(module, generic=True) == before


@pytest.mark.parametrize(
    "mutation", ["duplicate", "omit", "foreign", "foreign_input", "mutable", "reordered_members", "dependency"]
)
def test_bad_plural_ssa_correspondence_refuses_without_editing(mutation):
    module, region = fixture()
    if mutation == "duplicate":
        region = replace(region, outputs=(region.outputs[0], region.outputs[0]))
    elif mutation == "omit":
        third = arith.AddiOp(*region.inputs)
        block = region.operations[-1].parent
        block.insert_op_after(third, region.operations[-1])
        region = replace(region, operations=(*region.operations, third), outputs=(region.outputs[0], third.result))
    elif mutation == "foreign":
        _, other = fixture()
        region = replace(region, outputs=(region.outputs[0], other.outputs[1]))
    elif mutation == "foreign_input":
        _, other = fixture()
        region = replace(region, inputs=(other.inputs[0], region.inputs[1]))
    elif mutation == "mutable":
        region = replace(region, outputs=list(region.outputs))
    elif mutation == "reordered_members":
        region = replace(region, operations=tuple(reversed(region.operations)))
    else:
        region = replace(region, inputs=(*region.inputs, region.outputs[1]))
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="source"):
        _region_results(region)
    assert text(module, generic=True) == before


def test_unknown_intervening_call_is_a_barrier_even_when_no_result_is_used():
    module, region = fixture()
    barrier = func.CallOp("opaque", [], [])
    region.operations[0].parent.insert_op_after(barrier, region.operations[0])
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="intervening"):
        _region_results(region)
    assert text(module, generic=True) == before


def test_early_observer_cannot_move_after_the_composed_call():
    module, region = fixture()
    first, second = region.operations
    observer = arith.AddiOp(first.results[0], first.results[0])
    first.parent.insert_op_after(observer, first)
    # Including the observer in the proposed erased span cannot hide its
    # externally returned result from the complete output check.
    returned = tuple(second.parent.ops)[-1]
    returned.operands = (observer.result, second.results[0], first.results[0])
    region = replace(region, operations=(first, observer, second))
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="omits an observed"):
        _region_results(region)
    assert text(module, generic=True) == before


def test_singleton_convention_remains_last_operation_first_result():
    _, region = fixture()
    legacy = replace(region, outputs=())
    assert _region_results(legacy) == (region.operations[-1].results[0],)


def test_loop_carried_input_and_captured_weights_dominate_local_composition():
    source = """module {
func.func @forward(%a:tensor<2x3xi32>,%weight:tensor<2x3xi32>)->(tensor<2x3xi32>,tensor<2x3xi32>) {
 %zero=arith.constant 0:index
 %one=arith.constant 1:index
 %result:2=scf.for %i=%zero to %one step %one iter_args(%carry=%a,%prior=%weight)->
 (tensor<2x3xi32>,tensor<2x3xi32>) {
  %first=arith.addi %carry,%weight:tensor<2x3xi32>
  %second=arith.subi %carry,%weight:tensor<2x3xi32>
  scf.yield %first,%second:tensor<2x3xi32>,tensor<2x3xi32>
 }
 return %result#0,%result#1:tensor<2x3xi32>,tensor<2x3xi32>
}
}"""
    module = Parser(make_context(arith.Arith, tensor.Tensor, scf.Scf), source).parse_module()
    module.verify()
    function = next(operation for operation in module.body.block.ops if isinstance(operation, func.FuncOp))
    loop = next(operation for operation in function.body.block.ops if isinstance(operation, scf.ForOp))
    operations = tuple(loop.body.block.ops)[:2]
    outputs = tuple(operation.results[0] for operation in operations)
    region = SourceRegion(None, operations, (loop.body.block.args[1], function.body.block.args[1]), (2, 3), outputs)
    before = text(module, generic=True)
    assert _region_results(region) == outputs
    assert text(module, generic=True) == before
