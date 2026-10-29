"""Actual generated transport recognition and lifetime-boundary refusals."""

from math import prod

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i8
from xdsl.parser import Parser

from merlin.compile.scheduling.window_transfer import Window2D
from merlin.llvmlower.convolution_packing import emit_nhwc_window_panel
from merlin.llvmlower.generated_byte_transport import recognize_nhwc_halo_transport
from merlin.llvmlower.host_builder import PTR, FnBuilder
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.xdsl_dialects._common import text


def emitted(window):
    builder = FnBuilder([PTR, PTR])
    builder.add(llvm.InlineAsmOp("", "~{memory}", [], has_side_effects=True))
    capacity = window.input_shape[0] * prod(window.output_spatial) * window.input_shape[-1]
    private = builder.add(llvm.AllocaOp(builder.const(capacity), i8, alignment=16)).res
    emit_nhwc_window_panel(
        builder, window, builder.entry.args[0], private, builder.const(0),
        window.input_shape[0] * prod(window.output_spatial),
        source_destination_disjoint=True, maximum_panel_bytes=capacity,
    )
    address = builder.add(llvm.PtrToIntOp(builder.gep(private, builder.const(0), i8))).output
    # This opaque use needs a separate target effect owner. Recognition gives
    # it no completion/borrowing permission.
    builder.add(llvm.InlineAsmOp("", "r,~{memory}", [address], has_side_effects=True))
    function = llvm.FuncOp("entry", llvm.LLVMFunctionType([PTR, PTR], llvm.LLVMVoidType()),
                           body=builder.finish())
    module = ModuleOp([function])
    module.verify()
    return module, function, capacity


def recognize(module, original_window, capacity, **changes):
    options = dict(entry_symbol="entry", source_argument=0, window=original_window,
                   maximum_private_bytes=capacity)
    options.update(changes)
    return recognize_nhwc_halo_transport(module, **options)


WINDOW = Window2D((2, 5, 7, 3), (1, 1), padding=(1, 2, 3, 4))


@pytest.mark.parametrize("roundtrip", [False, True])
def test_actual_complete_halo_and_parser_transport(roundtrip):
    module, _, capacity = emitted(WINDOW)
    if roundtrip:
        module = Parser(make_llvm_context(), text(module, generic=True)).parse_module()
    proof = recognize(module, WINDOW, capacity)
    assert proof.private_bytes == 2 * 9 * 13 * 3
    assert proof.matched_operations > 20
    assert proof.revalidate().record() == proof.record()
    assert "caller-owned" in proof.record()["scope"]


@pytest.mark.parametrize("change", ["source", "budget", "geometry", "nonunit", "unbounded"])
def test_original_source_and_capacity_are_required(change):
    module, _, capacity = emitted(WINDOW)
    options = {
        "source": {"source_argument": 1},
        "budget": {"maximum_private_bytes": capacity - 1},
        "geometry": {"window": Window2D((2, 5, 7, 3), (1, 1), padding=(2, 1, 3, 4))},
        "nonunit": {"window": Window2D((2, 5, 7, 3), (3, 3), padding=(1, 2, 3, 4))},
        "unbounded": {"maximum_private_bytes": 1 << 63},
    }[change]
    with pytest.raises(ValueError):
        recognize(module, WINDOW, capacity, **options)


@pytest.mark.parametrize("change", ["copy_count", "zero_value", "source_pointer"])
def test_changed_byte_effect_refuses(change):
    module, function, capacity = emitted(WINDOW)
    intrinsic = next(op for op in function.walk()
                     if op.name == "llvm.call_intrinsic" and
                     ("memset" if change == "zero_value" else "memcpy") in str(op))
    operands = list(intrinsic.operands)
    if change == "source_pointer":
        operands[1] = tuple(function.body.blocks)[0].args[1]
    else:
        from merlin.llvmlower.host_builder import iconst

        constant = iconst(1, i8) if change == "zero_value" else iconst(capacity)
        tuple(function.body.blocks)[0].insert_op_before(constant, tuple(function.body.blocks)[0].last_op)
        operands[1 if change == "zero_value" else 2] = constant.result
    intrinsic.operands = operands
    with pytest.raises(ValueError):
        recognize(module, WINDOW, capacity)


@pytest.mark.parametrize("change", ["allocation", "store", "call", "private_load", "reentry"])
def test_other_host_effects_and_lifetime_escapes_refuse(change):
    module, function, capacity = emitted(WINDOW)
    entry = tuple(function.body.blocks)[0]
    private = next(op.res for op in entry.ops if isinstance(op, llvm.AllocaOp))
    exit_block = next(op.parent for op in function.walk() if isinstance(op, llvm.ReturnOp))
    if change == "allocation":
        operation = llvm.AllocaOp(next(op.size for op in entry.ops if isinstance(op, llvm.AllocaOp)), i8)
    elif change == "store":
        from merlin.llvmlower.host_builder import iconst

        value = iconst(0, i8)
        entry.insert_op_before(value, entry.last_op)
        operation = llvm.StoreOp(value.result, entry.args[1])
    elif change == "call":
        operation = llvm.CallOp("unknown", private)
    elif change == "private_load":
        operation = llvm.LoadOp(private, i8)
    else:
        operation = llvm.BrOp(tuple(function.body.blocks)[1], entry.args[0])
    exit_block.insert_op_before(operation, exit_block.last_op)
    with pytest.raises(ValueError):
        recognize(module, WINDOW, capacity)


def test_live_witness_reopens_actual_source():
    module, function, capacity = emitted(WINDOW)
    proof = recognize(module, WINDOW, capacity)
    first = next(op for op in function.walk() if isinstance(op, llvm.InlineAsmOp))
    first.detach()
    with pytest.raises(ValueError, match="source changed"):
        proof.revalidate()
