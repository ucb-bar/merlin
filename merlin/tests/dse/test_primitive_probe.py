"""Completion comes from the decoded source contract, never a shared host-ISA fallback."""
import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region

from merlin.perf import primitive_probe
from merlin.targetgen.rocc import decode


def source(monkeypatch, *, completion=True, ordered=True):
    block = Block(arg_types=[])
    names = ["initialize", "stage", "execute", "configure_output", "copy_output"]
    if completion:
        names.append("target_completion_instruction")
    for name in names:
        constraints = "~{memory}" if ordered else ""
        block.add_op(llvm.InlineAsmOp(name, constraints, [], [], has_side_effects=True))
    block.add_op(llvm.ReturnOp())
    module = ModuleOp([llvm.FuncOp("original", llvm.LLVMFunctionType([]), body=Region([block]))])
    rows = [{"class": "init"}, {"class": "stage"}, {"class": "execute"},
            {"class": "store_config", "decoded": {"subtype": "ST"}},
            {"class": "readback", "decoded": {"acc_addr": 0}}]
    if completion:
        rows.append({"class": "FENCE"})
    # This unit tests the generic IR transform's completion handling; the target decoder's
    # independent correctness is not claimed. A real emitted-artifact witness accompanies it.
    monkeypatch.setattr(decode, "decode_module", lambda *a, **kw: {"instructions": rows})
    monkeypatch.setattr(decode, "isa_constants", lambda target: {"CONFIG_SUBTYPE": {0: "store_config"}})
    monkeypatch.setattr(primitive_probe, "initialized_compute_primitives", lambda *a, **kw: [
        {"missing": [], "instruction_indices": [1, 2], "domain_digest": "1" * 64}])
    return module


def test_source_completion_instruction_and_clobbers_are_cloned(monkeypatch):
    result = primitive_probe.extract_primitive_program(source(monkeypatch), target="fixture")
    assert result.source_completion_index == 5
    for function in result.module.body.block.ops:
        instructions = [op for op in function.body.block.ops if op.name == "llvm.inline_asm"]
        assert instructions[-1].asm_string.data == "target_completion_instruction"
        assert instructions[-1].constraints.data == "~{memory}"
        assert instructions[-1].has_side_effects is not None


def test_missing_completion_does_not_invent_a_host_instruction(monkeypatch):
    with pytest.raises(ValueError, match="no decoded post-readback completion"):
        primitive_probe.extract_primitive_program(source(monkeypatch, completion=False), target="fixture")


def test_unordered_completion_is_not_silently_strengthened(monkeypatch):
    with pytest.raises(ValueError, match="host-memory ordering"):
        primitive_probe.extract_primitive_program(source(monkeypatch, ordered=False), target="fixture")


def test_optional_counters_do_not_enter_compute_timer(monkeypatch):
    program = primitive_probe.extract_primitive_program(source(monkeypatch), target="fixture")
    text = primitive_probe.render_primitive_host_wrapper(
        program, declarations="/* host-owned declarations */", argument_expressions=[],
        cycle_reader="host_timer", verify_call="host_verify()",
        before_measurement="  counters_reset();\n", after_measurement="  counters_snapshot();\n")
    warm = text.index(f"  {program.body_symbol}();")
    begin, end = text.index("const uint64_t begin"), text.index("const uint64_t end")
    assert warm < text.index("counters_reset();") < begin < end
    assert end < text.index("counters_snapshot();") < text.index(f"  {program.readback_symbol}();")
    assert "counters_" not in text[begin:end]


def test_context_keeps_loads_in_body_and_restores_setup_after_warmup(monkeypatch):
    module = source(monkeypatch)
    original_decode = decode.decode_module
    rows = original_decode(module, target="fixture")["instructions"]
    rows[0]["decoded"] = {"spad_addr": 0}
    program = primitive_probe.extract_primitive_program(module, target="fixture", include_operand_movement=True)
    functions = list(program.module.body.block.ops)
    names = lambda fn: [op.asm_string.data for op in fn.body.block.ops if op.name == "llvm.inline_asm"]
    assert names(functions[0]) == ["target_completion_instruction"]
    assert names(functions[1]) == ["initialize", "stage", "execute", "target_completion_instruction"]
    assert program.timed_instruction_indices == (0, 1, 2)
    text = primitive_probe.render_primitive_host_wrapper(
        program, declarations="/* host-owned declarations */", argument_expressions=[],
        cycle_reader="host_timer", verify_call="host_verify()", before_measurement="  counters_reset();\n")
    warm = text.index(f"  {program.body_symbol}();")
    restored = text.index(f"  {program.setup_symbol}();", warm)
    assert warm < restored < text.index("counters_reset();") < text.index("const uint64_t begin")


@pytest.mark.parametrize("unsupported", [False, True])
def test_fixed_work_drains_trailing_loads_inside_body(monkeypatch, unsupported):
    module = source(monkeypatch)
    rows = decode.decode_module(module, target="fixture")["instructions"]
    rows[0]["decoded"] = {"spad_addr": 0}
    block = next(iter(module.body.block.ops)).body.block
    execute = list(block.ops)[2]
    trailing = llvm.InlineAsmOp("retained_competing_load", "~{memory}", [], [], has_side_effects=True)
    block.insert_op_after(trailing, execute)
    rows.insert(3, {"class": "load", "decoded": {"spad_addr": 16}})
    if unsupported:
        extra = llvm.InlineAsmOp("omitted_future_compute", "~{memory}", [], [], has_side_effects=True)
        block.insert_op_after(extra, execute)
        rows.insert(3, {"class": "execute", "decoded": {}})
        with pytest.raises(ValueError, match="another compute or unsupported effect"):
            primitive_probe.extract_primitive_program(module, target="fixture",
                include_operand_movement=True, include_trailing_operand_movement=True)
        return
    program = primitive_probe.extract_primitive_program(module, target="fixture",
        include_operand_movement=True, include_trailing_operand_movement=True)
    body = list(program.module.body.block.ops)[1]
    names = [op.asm_string.data for op in body.body.block.ops if op.name == "llvm.inline_asm"]
    assert names == ["initialize", "stage", "execute", "retained_competing_load", "target_completion_instruction"]
    assert program.timed_instruction_indices == (0, 1, 2, 3)
