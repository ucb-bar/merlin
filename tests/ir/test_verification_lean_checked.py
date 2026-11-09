from __future__ import annotations

from verification_witnesses.lean_checked import (
    CHECK,
    EffectScheduleContract,
    LeanToolchain,
    LeanTransformReceipt,
    Path,
    _contract,
    blocks_in,
    coverage_policy_record,
    ir_text,
    os,
    parse_ir,
    pytest,
    replace,
    selected_toolchain,
)


def test_real_state_schedule_handles_static_fresh_memory_and_rejects_aliases(selected_toolchain):
    source = "builtin.module { func.func @f(%a:i8,%b:i8) {\n      %index = arith.constant 0 : index\n      %left = memref.alloc() : memref<1xi8>\n      %right = memref.alloc() : memref<1xi8>\n      memref.store %a,%left[%index] : memref<1xi8>\n      memref.store %b,%right[%index] : memref<1xi8>\n      memref.dealloc %left : memref<1xi8>\n      memref.dealloc %right : memref<1xi8>\n      func.return\n    }}"
    module = parse_ir(source)
    block = blocks_in(module)[1]
    ops = tuple(block.ops)
    for op in ops:
        block.detach_op(op)
    block.add_ops((ops[index] for index in (0, 1, 2, 4, 3, 5, 6, 7)))
    emitted = ir_text(module)
    parameters = dict(entry="f", contract=_contract(), toolchain=selected_toolchain)
    receipt = CHECK.verify_effect_schedule(source, emitted, **parameters)
    assert receipt.verified, receipt.reason
    assert CHECK.accepted_effect_receipt(receipt, source, emitted, **parameters)
    assert not CHECK.accepted_effect_receipt(
        receipt,
        source,
        emitted,
        **{**parameters, "contract": replace(_contract(), standard_allocations_successful=False)},
    )
    source_alias = source.replace("%b,%right[%index]", "%b,%left[%index]")
    emitted_alias = emitted.replace("%b, %right", "%b, %left")
    alias_module = parse_ir(source_alias)
    alias_block = blocks_in(alias_module)[1]
    alias_ops = tuple(alias_block.ops)
    for op in alias_ops:
        alias_block.detach_op(op)
    alias_block.add_ops((alias_ops[index] for index in (0, 1, 2, 4, 3, 5, 6, 7)))
    emitted_alias = ir_text(alias_module)
    wrong = CHECK.verify_effect_schedule(source_alias, emitted_alias, **parameters)
    assert wrong.status == "unsupported" and (not wrong.verified)
