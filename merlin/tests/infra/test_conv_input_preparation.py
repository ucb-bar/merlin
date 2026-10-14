from types import SimpleNamespace

import pytest

from merlin.perf.host_cfg_activity import analyze_host_cfg_activity
from merlin.targetgen.rocc.decode import _parse_module
from merlin.verify.conv_input_preparation import prove_input_preparation
from merlin.verify.host_compute_audit import _functions

BODY = """
builtin.module {
 llvm.func @kernel(%input: !llvm.ptr, %weight: !llvm.ptr, %output: !llvm.ptr) {
 %one = llvm.mlir.constant(1) : i64
 %scratch = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
 %value = llvm.load %input : !llvm.ptr -> i8
 REPLACE
 %address = llvm.ptrtoint %scratch : !llvm.ptr to i64
 "llvm.inline_asm"(%one, %one) <{asm_string = ".insn r 0x7b, 0x3, 0x0, x0, $0, $1",
 constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
 "llvm.inline_asm"(%address, %one) <{asm_string = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1",
 constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
 llvm.return
 }
}
"""
CONTEXT = {
    "call_buffers": [
        {"role": role, "shape": [1], "dtype": dtype}
        for role, dtype in zip(("input_0", "weight_0", "out_0"), ("i8", "i8", "i32"))
    ]
}


def _proof(monkeypatch, replacement, transform=lambda text: text, observer_patch=lambda observer: observer):
    from merlin.targetgen.rocc import decode

    def instruction(funct, rs1, rs2, isa):
        return (
            ("CONFIG_LD", {"stride": 1, "state_id": (rs1["raw"] >> 3) & 3})
            if funct == 0
            else ("MVIN2" if funct == 1 else "MVIN", {"dram": rs1, "rows": 1, "cols": 1})
        )

    semantics = SimpleNamespace(
        word_bits=64,
        isa_constants=lambda target: {
            "CUSTOM_OPCODE": 0x7B,
        },
        decode_instruction=instruction,
    )
    monkeypatch.setattr(
        decode, "_semantics", lambda target: SimpleNamespace(operand_observer=observer_patch(semantics))
    )
    function = _functions(_parse_module(transform(BODY.replace("REPLACE", replacement))))["kernel"]
    return prove_input_preparation(function, analyze_host_cfg_activity(function), CONTEXT, target="test")


def test_closed_input_copy_proves_actual_transfer_and_keeps_counts(monkeypatch):
    proof = _proof(monkeypatch, "llvm.store %value, %scratch : i8, !llvm.ptr")
    assert proof["status"] == "proven"
    assert proof["payload_bytes"] == 2
    assert proof["scratch_input_transfers"] == 1


def test_loaded_value_arithmetic_output_store_and_unknown_pointer_refuse(monkeypatch):
    bad = [
        "%changed = llvm.add %value, %value : i8\n llvm.store %changed, %scratch : i8, !llvm.ptr",
        "llvm.store %value, %output : i8, !llvm.ptr",
        "%unknown = llvm.inttoptr %one : i64 to !llvm.ptr\n llvm.store %value, %unknown : i8, !llvm.ptr",
        "",
    ]
    for replacement in bad:
        assert _proof(monkeypatch, replacement)["status"] == "unproven"


def test_unbound_dma_secondary_port_and_nonzero_load_state_refuse(monkeypatch):
    copy = "llvm.store %value, %scratch : i8, !llvm.ptr"
    changes = [
        lambda text: text.replace('"llvm.inline_asm"(%address, %one)', '"llvm.inline_asm"(%one, %one)'),
        lambda text: text.replace("0x2, x0", "0x1, x0"),
        lambda text: text.replace(
            "llvm.store %value", "%port = llvm.mlir.constant(9) : i64\n llvm.store %value"
        ).replace('"llvm.inline_asm"(%one, %one)', '"llvm.inline_asm"(%port, %one)'),
    ]
    for change in changes:
        assert _proof(monkeypatch, copy, change)["status"] == "unproven"


def test_pointer_add_requires_full_width_and_no_poison_flags():
    import pytest
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    from merlin.verify.conv_input_preparation import _unpoisoned_pointer_add

    value = SimpleNamespace(type=IntegerType(64))
    op = SimpleNamespace(results=[value], operands=[value, value], properties={})
    _unpoisoned_pointer_add(op)
    op.properties["overflowFlags"] = IntegerAttr(1, 32)
    with pytest.raises(ValueError, match="poison"):
        _unpoisoned_pointer_add(op)
    op.properties.clear()
    op.results = [SimpleNamespace(type=IntegerType(32))]
    with pytest.raises(ValueError, match="truncates"):
        _unpoisoned_pointer_add(op)


def test_observer_absence_width_and_unknown_state_refuse(monkeypatch):
    copy = "llvm.store %value, %scratch : i8, !llvm.ptr"
    assert _proof(monkeypatch, copy, observer_patch=lambda observer: None)["status"] == "unproven"

    def narrow(observer):
        observer.word_bits = 32
        return observer

    def unknown_state(observer):
        original = observer.decode_instruction

        def decode(*args):
            name, fields = original(*args)
            fields.pop("state_id", None)
            return name, fields

        observer.decode_instruction = decode
        return observer

    for patch in (narrow, unknown_state):
        assert _proof(monkeypatch, copy, observer_patch=patch)["status"] == "unproven"


ZERO_GLOBAL = """llvm.mlir.global private constant @padding(dense<0> : tensor<3xi8>)
 {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.array<3 x i8>"""


def _zero_global(text, declaration=ZERO_GLOBAL):
    return (
        text.replace("builtin.module {", "builtin.module {\n" + declaration)
        .replace(
            "%scratch = llvm.alloca %one x i8 : (i64) -> !llvm.ptr",
            "%scratch = llvm.mlir.addressof @padding : !llvm.ptr",
        )
        .replace("%value = llvm.load %input : !llvm.ptr -> i8", "")
    )


@pytest.mark.parametrize("offset,linkage", [(0, "private"), (2, "internal")])
def test_constant_zero_global_proves_only_bounded_address_dma(monkeypatch, offset, linkage):
    def transform(text):
        return _zero_global(text, ZERO_GLOBAL.replace("private", linkage)).replace(
            "%address = llvm.ptrtoint %scratch",
            f"%offset = llvm.getelementptr %scratch[{offset}] : (!llvm.ptr) -> !llvm.ptr, i8\n"
            "%address = llvm.ptrtoint %offset",
        )

    proof = _proof(monkeypatch, "", transform)
    assert proof["status"] == "proven"
    assert proof["readonly_zero_globals"] == 1
    assert proof["input_transfers"] == 1
    assert proof["scratch_input_transfers"] == 1
    assert proof["payload_bytes"] == 0


@pytest.mark.parametrize(
    "declaration",
    [
        "",
        ZERO_GLOBAL.replace(" constant", ""),
        ZERO_GLOBAL.replace("dense<0>", "dense<1>"),
        ZERO_GLOBAL.replace("dense<0> : tensor<3xi8>", ""),
        ZERO_GLOBAL.replace("!llvm.array<3 x i8>", "!llvm.array<4 x i8>"),
        ZERO_GLOBAL.replace("tensor<3xi8>", "tensor<3xi16>"),
        ZERO_GLOBAL.replace("addr_space = 0", "addr_space = 1"),
        ZERO_GLOBAL.replace("private", "external"),
        ZERO_GLOBAL.replace("private", "linkonce_odr"),
        ZERO_GLOBAL.replace("private", "weak_odr"),
        ZERO_GLOBAL + "\n" + ZERO_GLOBAL,
    ],
)
def test_unknown_mutable_nonzero_and_mismatched_globals_refuse(monkeypatch, declaration):
    proof = _proof(monkeypatch, "", lambda text: _zero_global(text, declaration))
    assert proof["status"] == "unproven"


@pytest.mark.parametrize(
    "effect",
    [
        "llvm.store %one, %scratch : i64, !llvm.ptr",
        "%byte = llvm.mlir.constant(0) : i8\n llvm.store %byte, %scratch : i8, !llvm.ptr",
        "llvm.call @escape(%scratch) : (!llvm.ptr) -> ()",
        '"llvm.inline_asm"(%scratch) <{asm_string = "fence", constraints = "r,~{memory}",'
        "has_side_effects}> : (!llvm.ptr) -> ()",
        "%outside = llvm.getelementptr %scratch[3] : (!llvm.ptr) -> !llvm.ptr, i8\n"
        "%bad = llvm.ptrtoint %outside : !llvm.ptr to i64\n"
        '"llvm.inline_asm"(%bad, %one) <{asm_string = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1",'
        'constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()',
    ],
)
def test_global_writes_escapes_and_outside_dma_refuse(monkeypatch, effect):
    assert _proof(monkeypatch, effect, _zero_global)["status"] == "unproven"


def test_zero_global_cannot_escape_through_another_function(monkeypatch):
    def transform(text):
        return _zero_global(text).replace(
            "builtin.module {",
            """builtin.module {
 llvm.func @other() {
  %escaped = llvm.mlir.addressof @padding : !llvm.ptr
  llvm.return
 }
""",
        )

    assert _proof(monkeypatch, "", transform)["status"] == "unproven"


def test_zero_global_unknown_module_alias_refuses(monkeypatch):
    def transform(text):
        return _zero_global(text).replace(
            "builtin.module {", 'builtin.module {\n "unknown.alias"() {referenced = @padding} : () -> ()'
        )

    assert _proof(monkeypatch, "", transform)["status"] == "unproven"
