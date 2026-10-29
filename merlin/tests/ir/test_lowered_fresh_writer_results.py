"""Independent final LLVM evidence for every plural out-parameter owner."""

from dataclasses import replace

import pytest
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, prove_lowered_writer_results

SHAPES = ((2, 3), (3,), (2, 3), (3,))
WIDTHS = (1, 2, 4, 2)
CONTRACT = FreshTensorWriterContract(
    "writer",
    3,
    (2, 3),
    "borrowed",
    32,
    borrowed_effects=BorrowedBufferEffects("neutral-plural-final-LLVM-ABI", True, True, True),
    result_arguments=(3, 2),
)


def fixture(*, changed=None, old_destination=False, extra_body="", caller_inputs=False):
    counts = [3 + 2 * len(shape) for shape in SHAPES]
    output_indices = (3, 2)
    all_counts = [*counts, *(counts[i] for i in output_indices)]
    types = [t for count in all_counts for t in ["!llvm.ptr", "!llvm.ptr", *(["i64"] * (count - 2))]]
    args = [f"%x{i}" for i in range(len(types))]
    starts, size = [], 0
    for count in all_counts:
        starts.append(size)
        size += count
    replacements = {index: starts[4 + position] for position, index in enumerate(output_indices)}
    if old_destination:
        replacements[2] = starts[2]
    forwarded = [
        value
        for i, count in enumerate(counts)
        for value in args[replacements.get(i, starts[i]) : replacements.get(i, starts[i]) + count]
    ]
    owners = ("a", "state", "old0", "old1", "out1", "out0")
    descriptors = []
    for name, shape in zip(owners, (*SHAPES, SHAPES[3], SHAPES[2]), strict=True):
        strides, running = [], 1
        for dimension in reversed(shape):
            strides.insert(0, running)
            running *= dimension
        descriptors.extend([f"%{name}", f"%{name}", "%c0", *(f"%c{d}" for d in (*shape, *strides))])
    caller_arguments = ",".join(f"%in{i}:{typ}" for i, typ in enumerate(types[:12])) if caller_inputs else ""
    if caller_inputs:
        descriptors[:12] = [f"%in{i}" for i in range(12)]
    for index, value in (changed or {}).items():
        descriptors[index] = value
    constants = "\n".join(f"%c{i}=llvm.mlir.constant({i}:i64):i64" for i in (0, 1, 2, 3, 64))
    allocations = "\n".join(f"%{name}=llvm.call @malloc(%c64):(i64)->!llvm.ptr" for name in owners)
    text = f"""builtin.module {{
llvm.func @malloc(i64)->!llvm.ptr
llvm.func @borrowed({",".join(types[: sum(counts)])})
llvm.func @writer({",".join(f"{a}:{t}" for a, t in zip(args, types, strict=True))}) {{
 llvm.call @borrowed({",".join(forwarded)}):({",".join(types[: sum(counts)])})->()
 {extra_body}
 llvm.return
}}
llvm.func @caller({caller_arguments}) {{
 {constants}
 {allocations}
 llvm.call @writer({",".join(descriptors)}):({",".join(types)})->()
 llvm.return
}}
}}"""
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(llvm.LLVM)
    module = Parser(context, text).parse_module()
    module.verify()
    functions = {op.sym_name.data: op for op in module.body.block.ops}
    call = next(
        op
        for op in functions["caller"].walk()
        if isinstance(op, llvm.CallOp) and op.callee.root_reference.data == "writer"
    )
    return module, functions["writer"], call


def prove(wrapper, call, contract=CONTRACT):
    return prove_lowered_writer_results(
        wrapper, call, contract, argument_shapes=SHAPES, element_bytes=WIDTHS, allocator_alignment=32
    )


def test_actual_appended_outputs_have_distinct_complete_descriptors():
    module, wrapper, call = fixture()
    original = str(module)
    report = prove(wrapper, call)
    assert report["result_arguments"] == [3, 2]
    assert report["actual_output_parameter_starts"] == [24, 29]
    assert report["distinct_direct_output_owners"] is True
    assert [record["bytes"] for record in report["descriptors"]] == [6, 6, 24, 6]
    assert report["borrowed_effects"]["qualification_identity"] == CONTRACT.borrowed_effects.qualification_identity
    assert report["allocator_runtime_closure"] == "REQUIRED_SEPARATELY"
    assert str(module) == original


def test_exact_dynamic_readonly_input_descriptors_are_forwarded_but_not_qualified():
    _, wrapper, call = fixture(caller_inputs=True)
    report = prove(wrapper, call)
    assert report["descriptors"][0]["caller_descriptor_arguments"] == list(range(7))
    assert report["descriptors"][1]["caller_descriptor_arguments"] == list(range(7, 12))
    assert "shape" not in report["descriptors"][0]
    assert report["source_types_and_runtime_caller_descriptors"] == "REQUIRED_SEPARATELY"


@pytest.mark.parametrize("changes", [{1: "%in0"}, {2: "%in3"}, {9: "%in10"}, {24: "%in0", 25: "%in1"}])
def test_mixed_caller_descriptor_components_and_output_alias_refuse(changes):
    _, wrapper, call = fixture(changed=changes, caller_inputs=True)
    with pytest.raises(ValueError, match="owner/ABI"):
        prove(wrapper, call)


def test_changed_output_byte_width_cannot_exceed_actual_malloc_extent():
    _, wrapper, call = fixture()
    with pytest.raises(ValueError, match="owner/ABI"):
        prove_lowered_writer_results(
            wrapper, call, CONTRACT, argument_shapes=SHAPES, element_bytes=(1, 2, 1024, 2), allocator_alignment=32
        )


@pytest.mark.parametrize(
    "changes",
    [
        {24: "%a", 25: "%a"},
        {29: "%state", 30: "%state"},
        {29: "%out1", 30: "%out1"},
        {25: "%old1"},
        {30: "%old0"},
        {26: "%c1"},
        {31: "%c1"},
        {27: "%c2"},
        {32: "%c3"},
        {35: "%c2"},
    ],
)
def test_either_output_alias_offset_shape_or_stride_mutation_refuses(changes):
    _, wrapper, call = fixture(changed=changes)
    with pytest.raises(ValueError, match="owner/ABI"):
        prove(wrapper, call)


def test_valid_but_unused_original_writer_copy_cannot_replace_final_result():
    _, wrapper, call = fixture(old_destination=True)
    with pytest.raises(ValueError, match="owner/ABI"):
        prove(wrapper, call)


def test_missing_second_result_and_changed_forwarding_order_refuse():
    _, wrapper, call = fixture()
    borrowed = next(op for op in wrapper.walk() if isinstance(op, llvm.CallOp))
    operands = list(borrowed.operands)
    # Swapping one same-typed descriptor pointer must still change exact SSA identity.
    operands[12], operands[19] = operands[19], operands[12]
    borrowed.operands = operands
    with pytest.raises(ValueError, match="owner/ABI"):
        prove(wrapper, call)
    _, wrapper, call = fixture()
    wrapper.body.block.erase_arg(wrapper.body.block.args[-1], safe_erase=False)
    with pytest.raises(ValueError, match="owner/ABI"):
        prove(wrapper, call)


@pytest.mark.parametrize(
    "changes",
    [
        {"borrowed_effects": None},
        {"result_arguments": (3,)},
        {"fully_written_arguments": (3,)},
        {"result_arguments": (3, 3)},
    ],
)
def test_original_effect_and_complete_output_premises_are_required(changes):
    _, wrapper, call = fixture()
    with pytest.raises(ValueError):
        prove(wrapper, call, replace(CONTRACT, **changes))
