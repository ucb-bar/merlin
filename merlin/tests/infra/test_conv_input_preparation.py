from types import SimpleNamespace

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


def _proof(monkeypatch, replacement):
    from merlin.targetgen.rocc import decode

    def instruction(funct, rs1, rs2, isa):
        return ("CONFIG_LD", {"stride": 1}) if funct == 0 else ("MVIN", {"dram": rs1, "rows": 1, "cols": 1})

    semantics = SimpleNamespace(isa_constants=lambda target: {"CUSTOM_OPCODE": 0x7B}, decode_instruction=instruction)
    monkeypatch.setattr(decode, "_semantics", lambda target: semantics)
    function = _functions(_parse_module(BODY.replace("REPLACE", replacement)))["kernel"]
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
