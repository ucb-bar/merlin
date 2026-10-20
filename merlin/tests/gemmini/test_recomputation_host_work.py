"""Typed controls for the actual-host-work witness; no compiler or execution."""

import pytest
from recomputation_execution_inputs import original_case
from recomputation_host_work import argument_slots, functions, implementation_paths, verify_work
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i8, i32, i64
from xdsl.ir import Block, Region

pytestmark = pytest.mark.target("gemmini")
CALLEES = ("first_original_contraction", "second_original_contraction")


def test_observer_direct_helpers_have_freezable_code_and_current_source_paths():
    assert all(callable(function) and hasattr(function, "__code__") for function in functions())
    assert all(path.is_file() for path in implementation_paths())


def fixture(*, candidate, damage=None):
    case = original_case()
    declarations = {str(i): {"kind": "input", "name": name} for i, (name, _, _) in enumerate(case.reference.INPUTS)}
    arguments = [
        dict(original_index=i, runtime_index=i, kind="MERLIN_INPUT", shape=list(shape), dtype=dtype)
        for i, (_, shape, dtype) in enumerate(case.reference.INPUTS)
    ]
    roles, fields = argument_slots(
        case.plan.source, arguments, declarations, {row["name"]: int(i) for i, row in declarations.items()}
    )
    entry, region = Block(arg_types=fields), Region()
    region.add_block(entry)

    def append(block, op):
        block.add_op(op)
        return op.results[0]

    constants = {n: append(entry, llvm.ConstantOp(IntegerAttr(n, i64), i64)) for n in (0, 1, 2, 3, 5, 10, 63, 64, 74)}
    scratch = append(entry, llvm.AllocaOp(constants[10], i8))
    # Synthetic output descriptor pointers make the two consumers observable.
    destinations = [entry.args[7 * (6 + i) + 1] for i in (3, 4)]
    raw_allocations = []
    if damage in ("aligned_allocations", "free_raw_before_call"):
        destinations = []
        for _ in range(2):
            raw = append(entry, llvm.CallOp("malloc", constants[74], return_type=llvm.LLVMPointerType()))
            raw_allocations.append(raw)
            integer = append(entry, llvm.PtrToIntOp(raw))
            shifted = append(entry, llvm.AddOp(integer, constants[63]))
            remainder = append(entry, llvm.URemOp(shifted, constants[64]))
            aligned = append(entry, llvm.SubOp(shifted, remainder))
            destinations.append(append(entry, llvm.IntToPtrOp(aligned)))
    if damage == "early_P_overwrite":
        destinations[0] = scratch

    def address(block, pointer, index, ty=i8):
        return append(block, llvm.GEPOp.from_mixed_indices(pointer, [index], ty))

    def load(block, role, index):
        pointer = entry.args[roles[role]["aligned"]]
        if damage == "wrong_root" and role == "x":
            pointer = entry.args[roles["bias0"]["aligned"]]
        if damage == "wrong_offset" and role == "state":
            index = append(block, llvm.AddOp(index, constants[1]))
        ty = i32 if role == "state" else i8
        return append(block, llvm.LoadOp(address(block, pointer, index, ty), ty, alignment=1))

    start, producers = entry, []
    order = ("bias1", "bias0") if damage == "wrong_phase" else ("bias0", "bias1")
    for role in (() if candidate else ("producer",)) + order:
        if damage == "input_clobber_between" and role == "bias1":
            value = append(start, llvm.ConstantOp(IntegerAttr(0, i32), i32))
            pointer = address(start, entry.args[roles["state"]["aligned"]], constants[0], i32)
            start.add_op(llvm.StoreOp(value, pointer, alignment=4))
        outer, pre, inner, work, latch, after = (
            Block(arg_types=[i64]),
            Block(),
            Block(arg_types=[i64]),
            Block(),
            Block(),
            Block(),
        )
        region.add_block([outer, pre, inner, work, latch, after])
        start.add_op(llvm.BrOp(outer, constants[0]))
        bound = entry.args[roles["state"]["aligned"] + 1] if damage == "unknown_control" else constants[2]
        cond = append(outer, llvm.ICmpOp(outer.args[0], bound, IntegerAttr(2, i64)))
        outer.add_op(llvm.CondBrOp(cond, pre, [], after, []))
        pre.add_op(llvm.BrOp(inner, constants[0]))
        cond = append(inner, llvm.ICmpOp(inner.args[0], constants[5], IntegerAttr(2, i64)))
        inner.add_op(llvm.CondBrOp(cond, work, [], latch, []))
        index = append(work, llvm.AddOp(append(work, llvm.MulOp(outer.args[0], constants[5])), inner.args[0]))
        if candidate or role == "producer":
            narrow = append(work, llvm.TruncOp(load(work, "state", index), i8))
            p = append(work, llvm.AddOp(load(work, "x", index), narrow))
            producers.append(p.owner)
            if damage == "overflow_flag":
                p.owner.properties["overflowFlags"] = IntegerAttr(1, i32)
        else:
            p = append(work, llvm.LoadOp(address(work, scratch, index), i8, alignment=1))
        if role == "producer":
            value, destination = p, scratch
        else:
            operation = llvm.AddOp if role == "bias0" else llvm.MulOp
            value = append(work, operation(p, load(work, role, index)))
            destination = destinations[role == "bias1"]
            if damage == "unconsumed_store":
                destination = scratch
        work.add_op(llvm.StoreOp(value, address(work, destination, index), alignment=1))
        if damage == "unknown_call_in_work":
            work.add_op(llvm.CallOp("unregistered_observer"))
        if damage == "runtime_call_in_work":
            work.add_op(llvm.CallOp("free", scratch))
        update = append(work, llvm.AddOp(inner.args[0], constants[1]))
        work.add_op(llvm.BrOp(inner, update))
        update = append(latch, llvm.AddOp(outer.args[0], constants[1]))
        latch.add_op(llvm.BrOp(outer, update))
        start = after
    if damage == "unknown_call_outside":
        start.add_op(llvm.CallOp("unregistered_observer"))
    if damage == "clobber_before_call":
        zero_byte = append(start, llvm.ConstantOp(IntegerAttr(0, i8), i8))
        start.add_op(llvm.StoreOp(zero_byte, address(start, destinations[0], constants[0]), alignment=1))
    if damage == "free_before_call":
        start.add_op(llvm.CallOp("free", destinations[0]))
    if damage == "free_raw_before_call":
        start.add_op(llvm.CallOp("free", raw_allocations[0]))
    members = list(enumerate(zip(CALLEES, destinations, strict=True)))
    for i, (callee, pointer) in reversed(members) if damage == "swapped_calls" else members:
        descriptor = (pointer, pointer, constants[0], constants[2], constants[5], constants[5], constants[1])
        rhs, output, n = (
            entry.args[roles[f"w{i}"]["aligned"]],
            entry.args[7 * (6 + 1 + i) + 1],
            constants[5 if i == 0 else 3],
        )
        right = (rhs, rhs, constants[0], constants[5], n, n, constants[1])
        result = (output, output, constants[0], constants[2], n, n, constants[1])
        start.add_op(llvm.CallOp(callee, *descriptor, *right, *result))
    start.add_op(llvm.ReturnOp())
    function = llvm.FuncOp("forward", llvm.LLVMFunctionType(fields), body=region)
    signatures = {
        op.callee.root_reference.data: (
            tuple(value.type for value in op.operands),
            op.results[0].type if op.results else None,
        )
        for op in function.walk()
        if op.name == "llvm.call"
    }
    ModuleOp(
        [
            *(llvm.FuncOp(name, llvm.LLVMFunctionType(types, result)) for name, (types, result) in signatures.items()),
            function,
        ]
    ).verify()
    return function, roles, fields


@pytest.mark.parametrize("candidate", (False, True))
def test_complete_same_coordinate_work_is_bound_to_actual_consumers(candidate):
    function, roles, fields = fixture(candidate=candidate)
    result = verify_work(function, roles, fields, CALLEES, candidate=candidate)
    assert result["counts"]["producer_evaluations"] == (20 if candidate else 10)
    assert result["counts"]["bias0"] == result["counts"]["bias1"] == 10


def test_original_emitted_body_cannot_be_reported_as_realized_recomputation():
    function, roles, fields = fixture(candidate=False)
    with pytest.raises(AssertionError):
        verify_work(function, roles, fields, CALLEES, candidate=True)


def test_actual_alignment_chain_retains_the_same_allocation_owner():
    function, roles, fields = fixture(candidate=True, damage="aligned_allocations")
    assert verify_work(function, roles, fields, CALLEES, candidate=True)["counts"]["producer_evaluations"] == 20


def test_first_epilogue_cannot_overwrite_stored_P_before_second_consumer():
    function, roles, fields = fixture(candidate=False, damage="early_P_overwrite")
    with pytest.raises(AssertionError):
        verify_work(function, roles, fields, CALLEES, candidate=False)


@pytest.mark.parametrize(
    "damage",
    (
        "wrong_root",
        "wrong_offset",
        "overflow_flag",
        "unknown_control",
        "unconsumed_store",
        "unknown_call_in_work",
        "runtime_call_in_work",
        "unknown_call_outside",
        "clobber_before_call",
        "free_before_call",
        "free_raw_before_call",
        "wrong_phase",
        "input_clobber_between",
        "swapped_calls",
    ),
)
def test_wrong_dataflow_or_unqualified_control_effects_refuse(damage):
    function, roles, fields = fixture(candidate=True, damage=damage)
    with pytest.raises(AssertionError):
        verify_work(function, roles, fields, CALLEES, candidate=True)


def test_runtime_projection_is_checked_instead_of_raw_source_slot_assumption():
    case = original_case()
    module = case.plan.source.clone()
    entry = module.body.block.first_op
    types = tuple(entry.function_type.inputs)
    # A wrong runtime map contradicts the actual current typed ABI.
    arguments = [
        dict(original_index=i, runtime_index=5 - i, kind="MERLIN_INPUT", shape=list(shape), dtype=dtype)
        for i, (_, shape, dtype) in enumerate(case.reference.INPUTS)
    ]
    declared = {str(i): {"kind": "input", "name": name} for i, (name, _, _) in enumerate(case.reference.INPUTS)}
    assert len(types) == 6
    with pytest.raises(AssertionError):
        argument_slots(module, arguments, declared, {row["name"]: int(i) for i, row in declared.items()})
