"""Compiler-owned readout membership prevents duplicate pointwise mapping."""

from test_pointwise_source import source_program
from test_source_pointwise_lowering import fixture

from merlin.llvmlower.source_pointwise import discover_pointwise


def test_claimed_operation_is_not_mapped_and_other_source_still_is(tmp_path):
    selected, mapped, _ = fixture(tmp_path)
    module, claimed = source_program(m=1, n=1)
    other, _ = source_program(m=1, n=1)
    function = next(iter(other.body.block.ops)).clone()
    function.properties["sym_name"] = type(function.sym_name)("other")
    module.body.block.add_op(function)
    module.verify()
    regions, declined = discover_pointwise(module, selected, lambda: None, excluded_operations=frozenset((claimed,)))
    assert not declined and len(regions) == len(mapped) == 1
    assert regions[0].operations != (claimed,)
    assert regions[0].operations[0].parent_op() is function
