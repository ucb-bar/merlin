"""Source-bound complete readout proofs retain ordinary ownership and observers."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_connected_spatial_host import fixture as staged_fixture
from xdsl.dialects import func, memref
from xdsl.dialects.builtin import FunctionType

from merlin.compile.scheduling.spatial_readout import RoundedReadout
from merlin.llvmlower.connected_spatial import group_connected_spatial
from merlin.llvmlower.connected_spatial_host import prepare_connected_spatial_host
from merlin.llvmlower.requantization import f32
from merlin.xdsl_dialects._common import text


def fixture():
    source, rewrite, original, contract, workspace, _ = staged_fixture()
    # A synthetic nonunit Q2 that is identity on every signed-i8 code. The
    # original two stages still travel in the source statement and proof.
    rewrite.entries["pool"]["acc_scale"] = 0.999
    rewrite.programs["pool"]["entry"] = deepcopy(rewrite.entries["pool"])
    readouts = tuple(
        replace(chain, pool=RoundedReadout((0.999,)), replacement=chain.producer)
        for chain in original.readouts
    )
    plan = replace(original, readouts=readouts, readout_mode="single_readout", intermediate=None)
    selected = group_connected_spatial(source, rewrite)
    contract = replace(contract, symbol=selected.receipt["groups"][0]["symbol"])
    return source, rewrite, plan, contract, replace(workspace, bytes=1, alignment=1)


def prepare(source, rewrite, plan, contract, workspace):
    return prepare_connected_spatial_host(
        source, rewrite, ("producer", "pool"), plan, contract, workspace,
        kernel_symbol="private_spatial_kernel", pointer_alignments=(1, 1, 4, 1),
        stack_budget_bytes=1,
    )


def test_exact_nonunit_composition_keeps_source_and_public_abi_with_one_stack_byte():
    source, rewrite, plan, contract, workspace = fixture()
    before = text(source, generic=True)
    host = prepare(source, rewrite, plan, contract, workspace)
    assert text(source, generic=True) == before
    assert host.source_entry["components"][1]["acc_scale"] == 0.999
    assert host.scratch.module.body.block.first_op.function_type == source.body.block.first_op.function_type
    assert host.scratch.receipt["required_stripe_bytes"] == 0
    assert host.receipt["private_helper_pointer_count"] == 5
    assert all(p["exact"] and p["domain"] == [-(1 << 31), (1 << 31) - 1]
               and p["second_readout_identity"] and p["pool_scales"] == [f32(0.999)]
               for p in host.receipt["source_readout_proofs"])
    allocations = [op for op in host.scratch.module.walk() if isinstance(op, memref.AllocOp)
                   and op.memref.type.get_shape() == (1,)]
    stack = [op for op in host.scratch.module.walk() if isinstance(op, memref.AllocaOp)]
    assert not allocations and len(stack) == 1
    assert stack[0].memref.type.get_shape() == (1,)
    assert "scratch->size[0] < INT64_C(1)" in host.scratch.bridge_source
    host.scratch.module.verify()


def test_even_an_equivalent_changed_source_coefficient_cannot_reuse_the_old_binding():
    source, rewrite, plan, contract, workspace = fixture()
    rewrite.entries["pool"]["acc_scale"] = 0.998
    rewrite.programs["pool"]["entry"] = deepcopy(rewrite.entries["pool"])
    selected = group_connected_spatial(source, rewrite)
    contract = replace(contract, symbol=selected.receipt["groups"][0]["symbol"])
    before = text(source, generic=True)
    with pytest.raises(ValueError, match="Q1/Q2"):
        prepare(source, rewrite, plan, contract, workspace)
    assert text(source, generic=True) == before


def test_non_equivalent_replacement_and_missing_workspace_effects_still_refuse():
    source, rewrite, plan, contract, workspace = fixture()
    first = plan.readouts[0]
    with pytest.raises(ValueError, match="composition differs"):
        replace(plan, readouts=(replace(first, replacement=RoundedReadout((0.25,), True)),
                                *plan.readouts[1:]))
    before = text(source, generic=True)
    with pytest.raises(ValueError):
        prepare(source, rewrite, plan, contract, replace(workspace, synchronous_before_return=False))
    assert text(source, generic=True) == before


def test_single_readout_cannot_eliminate_a_separately_observed_producer():
    source, rewrite, plan, contract, workspace = fixture()
    owner = source.body.block.first_op
    calls = [op for op in owner.body.block.ops if isinstance(op, func.CallOp)]
    returned = owner.body.block.last_op
    result = returned.arguments[0]
    owner.body.block.erase_op(returned, safe_erase=False)
    owner.body.block.add_op(func.ReturnOp(result, calls[0].results[0]))
    owner.properties["function_type"] = FunctionType.from_lists(
        owner.function_type.inputs, [result.type, calls[0].results[0].type]
    )
    source.verify()
    before = text(source, generic=True)
    with pytest.raises(ValueError, match="unique closed"):
        prepare(source, rewrite, plan, contract, workspace)
    assert text(source, generic=True) == before
