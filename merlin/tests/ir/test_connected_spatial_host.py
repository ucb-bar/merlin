"""Original host owns spatial scratch; only a closed private device CFG crosses."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_connected_spatial import fixture as graph_fixture
from test_spatial_scratch_abi import fixture as ownership_fixture
from xdsl.dialects import func, llvm, memref
from xdsl.dialects.builtin import FunctionType, ModuleOp, i64

from merlin.llvmlower.connected_spatial import group_connected_spatial
from merlin.llvmlower.connected_spatial_host import prepare_connected_spatial_host, select_closed_spatial_device_helper
from merlin.llvmlower.host_builder import PTR, FnBuilder
from merlin.xdsl_dialects._common import text


def fixture():
    source, rewrite = graph_fixture()
    _, plan, contract, workspace = ownership_fixture()
    rewrite.entries["producer"].update(
        batch=2,
        Himg=5,
        Wimg=5,
        ci=3,
        N=17,
        kh=3,
        kw=3,
        stride=[1, 1],
        dilation=[1, 1],
        padding=[1, 1, 1, 1],
        output_dtype="i8",
    )
    rewrite.entries["pool"].update(
        M=50,
        N=17,
        pool_in_dims=[5, 5],
        pool_size=[3, 3],
        pool_stride=[2, 2],
        pool_padding=[1, 1, 1, 1],
        pool_pad_value=-128,
        output_dtype="i8",
    )
    for symbol in rewrite.entries:
        rewrite.programs[symbol]["entry"] = deepcopy(rewrite.entries[symbol])
    selected = group_connected_spatial(source, rewrite)
    contract = replace(contract, symbol=selected.receipt["groups"][0]["symbol"])
    host = prepare_connected_spatial_host(
        source,
        rewrite,
        ("producer", "pool"),
        plan,
        contract,
        workspace,
        kernel_symbol="private_spatial_kernel",
        pointer_alignments=(1, 1, 4, 1),
    )
    return source, rewrite, plan, contract, workspace, host


def test_current_two_operation_host_owns_private_scratch_and_keeps_its_public_abi():
    source, _, _, _, _, host = fixture()
    assert host.scratch.module.body.block.first_op.function_type == source.body.block.first_op.function_type
    allocations = [
        op for op in host.scratch.module.body.block.first_op.body.block.ops if isinstance(op, memref.AllocOp)
    ]
    assert len(allocations) == 1 and allocations[0].memref.type.get_shape() == (240,)
    assert host.scratch.owner_symbol == "forward"
    assert host.receipt["private_helper_pointer_count"] == 5
    assert len(host.receipt["logical_buffers"]) == 4
    assert "private_spatial_kernel" in host.scratch.bridge_source
    assert "malloc" not in host.scratch.bridge_source and "free(" not in host.scratch.bridge_source
    host.scratch.module.verify()


def test_same_signature_cannot_hide_a_changed_source_rounding():
    source, rewrite, plan, contract, workspace, _ = fixture()
    first = plan.readouts[0]
    bad = replace(plan, readouts=(replace(first, producer=replace(first.producer, scales=(0.25,))), plan.readouts[1]))
    before = text(source)
    with pytest.raises(ValueError, match="Q1/Q2"):
        prepare_connected_spatial_host(
            source,
            rewrite,
            ("producer", "pool"),
            bad,
            contract,
            workspace,
            kernel_symbol="private_spatial_kernel",
            pointer_alignments=(1, 1, 4, 1),
        )
    assert text(source) == before


def test_packed_plan_binds_same_actual_ssa_and_readouts_with_separate_private_capacity():
    from merlin.compile.scheduling.program_limits import AddressWindow
    from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan
    from merlin.llvmlower.connected_spatial_route import ConnectedSpatialRoute

    source, rewrite, original, contract, workspace, _ = fixture()
    plan = PackedSpatialProducerPlan(original, 7, AddressWindow("private_packed", 0, 512), 64)
    workspace = replace(workspace, bytes=512)
    host = prepare_connected_spatial_host(
        source,
        rewrite,
        ("producer", "pool"),
        plan,
        contract,
        workspace,
        kernel_symbol="private_spatial_kernel",
        pointer_alignments=(1, 1, 4, 1),
    )
    route = ConnectedSpatialRoute(("producer", "pool"), plan, contract, workspace, (1, 1, 4, 1), "raw_packed")
    record = route.record()
    assert record["plan"]["panel_rows"] == 7 and record["workspace"]["bytes"] == 512
    assert record["plan"]["source"]["readouts"][0]["producer"]["scales"] == [0.5]
    assert host.scratch.module.body.block.first_op.function_type == source.body.block.first_op.function_type
    assert host.scratch.plan_sha256 == plan.identity != original.identity
    first = original.readouts[0]
    wrong_q = replace(
        plan,
        source=replace(
            original, readouts=(replace(first, producer=replace(first.producer, scales=(0.25,))), original.readouts[1])
        ),
    )
    with pytest.raises(ValueError, match="Q1/Q2"):
        prepare_connected_spatial_host(
            source,
            rewrite,
            ("producer", "pool"),
            wrong_q,
            contract,
            workspace,
            kernel_symbol="private_spatial_kernel",
            pointer_alignments=(1, 1, 4, 1),
        )


def helper(*, arity=5, imports=False):
    builder = FnBuilder([PTR] * arity)
    if imports:
        builder.add(llvm.CallOp("malloc", builder.const(64), return_type=PTR))
    else:
        builder.add(llvm.InlineAsmOp("", "~{memory}", [], has_side_effects=True))
    function = llvm.FuncOp(
        "selected_raw_helper", llvm.LLVMFunctionType([PTR] * arity, llvm.LLVMVoidType()), body=builder.finish()
    )
    declarations = [llvm.FuncOp("malloc", llvm.LLVMFunctionType([i64], PTR))] if imports else []
    return ModuleOp([*declarations, function])


def test_closed_private_helper_has_no_runtime_imports_and_retains_all_effects():
    _, _, plan, _, _, host = fixture()
    module = helper()
    before = text(module)
    selected, receipt = select_closed_spatial_device_helper(
        module, host, plan, helper_symbol="selected_raw_helper", entry_symbol="private_spatial_kernel"
    )
    assert text(module) == before
    assert len(selected.body.block.ops) == 1
    assert selected.body.block.first_op.sym_name.data == "private_spatial_kernel"
    assert len(next(iter(selected.body.block.first_op.body.blocks)).args) == 5
    assert len([op for op in selected.walk() if isinstance(op, llvm.InlineAsmOp)]) == 1
    assert receipt["allocator_imports"] == [] and receipt["region"]["calls_added"] == 0
    selected.verify()


@pytest.mark.parametrize("mutation", ["arity", "allocator", "host", "source_entry"])
def test_private_device_selection_refuses_missing_closed_current_binding(mutation):
    _, _, plan, _, _, host = fixture()
    module = helper(arity=4 if mutation == "arity" else 5, imports=mutation == "allocator")
    if mutation == "host":
        host.scratch.module.attributes["mutation"] = __import__(
            "xdsl.dialects.builtin", fromlist=["UnitAttr"]
        ).UnitAttr()
    elif mutation == "source_entry":
        host.source_entry["components"][0]["acc_scale"] = 0.25
    before = text(module)
    with pytest.raises(ValueError):
        select_closed_spatial_device_helper(
            module, host, plan, helper_symbol="selected_raw_helper", entry_symbol="private_spatial_kernel"
        )
    assert text(module) == before


def test_independent_public_observation_survives_host_scratch_preparation():
    source, rewrite, plan, _, workspace, _ = fixture()
    forward = source.body.block.first_op
    original = forward.function_type
    retained_input = forward.body.block.args[0]
    terminator = forward.body.block.last_op
    results = tuple(terminator.arguments)
    forward.body.block.erase_op(terminator, safe_erase=False)
    forward.body.block.add_op(func.ReturnOp(*results, retained_input))
    forward.properties["function_type"] = FunctionType.from_lists(
        original.inputs, [*original.outputs, retained_input.type]
    )
    source.verify()
    selection = group_connected_spatial(source, rewrite)
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract

    contract = FreshTensorWriterContract(
        selection.receipt["groups"][0]["symbol"],
        3,
        (3,),
        "borrowed_pair",
        64,
        borrowed_effects=BorrowedBufferEffects("synthetic source ownership only", True, True, True),
    )
    host = prepare_connected_spatial_host(
        source,
        rewrite,
        ("producer", "pool"),
        plan,
        contract,
        workspace,
        kernel_symbol="private_spatial_kernel",
        pointer_alignments=(1, 1, 4, 1),
    )
    assert host.scratch.module.body.block.first_op.function_type == forward.function_type
    assert len(host.scratch.module.body.block.first_op.function_type.outputs) == 2
    host.scratch.module.verify()
