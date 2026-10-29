"""Markers must be on one-time task entries, outside nested loop cycles."""

import copy

import pytest
from merlin_experiments.phase2.task_window_profile import instrument_task_windows
from xdsl.parser import Parser

from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

NESTED = '''builtin.module {
  llvm.func @kernel_entry() {
    %zero = llvm.mlir.constant(0) : i64
    %one = llvm.mlir.constant(1) : i64
    %two = llvm.mlir.constant(2) : i64
    %three = llvm.mlir.constant(3) : i64
    llvm.br ^outer(%zero : i64) {merlin.global_task = 0 : i64}
  ^outer(%oi: i64):
    %oc = llvm.icmp "slt" %oi, %two {merlin.global_task = 0 : i64} : i64
    llvm.cond_br %oc, ^inner_entry, ^done {merlin.global_task = 0 : i64}
  ^inner_entry:
    llvm.br ^inner(%zero : i64) {merlin.global_task = 0 : i64}
  ^inner(%ii: i64):
    %ic = llvm.icmp "slt" %ii, %three {merlin.global_task = 0 : i64} : i64
    llvm.cond_br %ic, ^body, ^inner_exit {merlin.global_task = 0 : i64}
  ^body:
    %next_i = llvm.add %ii, %one {merlin.global_task = 0 : i64, prov.region_id = "original"} : i64
    llvm.br ^inner(%next_i : i64) {merlin.global_task = 0 : i64}
  ^inner_exit:
    %next_o = llvm.add %oi, %one {merlin.global_task = 0 : i64} : i64
    llvm.br ^outer(%next_o : i64) {merlin.global_task = 0 : i64}
  ^done:
    %last = llvm.add %zero, %one {merlin.global_task = 1 : i64} : i64
    llvm.return {merlin.global_task = -2 : i64}
  }
}'''


def plan(count=2):
    return dict(params=dict(global_program_plan=dict(tasks=[dict(task_index=i, kind="host") for i in range(count)])))


def test_nested_loop_marks_are_outside_cycles_and_original_metadata_survives():
    cb = plan()
    original = copy.deepcopy(cb)
    result = instrument_task_windows(NESTED, cb, symbol_prefix="case0")
    module = Parser(make_llvm_context(), result["lowered_mlir_text"]).parse_module()
    module.verify()
    function = next(op for op in module.body.block.ops if op.name == "llvm.func" and op.body.blocks)
    calls = [(index, op) for index, block in enumerate(function.body.blocks)
             for op in block.ops if op.name == "llvm.call"]
    assert [index for index, _ in calls] == [0, 6]
    assert [call.args[0].owner.value.value.data for _, call in calls] == [0, 1]
    assert result["manifest"]["markers"] == [
        dict(task_id=0, block_index=0, first_operation="llvm.br"),
        dict(task_id=1, block_index=6, first_operation="llvm.add"),
    ]
    assert 'prov.region_id = "original"' in result["lowered_mlir_text"]
    assert "".join(line for line in result["lowered_mlir_text"].splitlines(keepends=True)
                   if "case0" not in line) == NESTED
    assert result["completion_statement"] == "case0_mark(2);" and result["dump_statement"] == "case0_dump();"
    assert cb == original and result["manifest"]["tasks"] == original["params"]["global_program_plan"]["tasks"]
    assert result["c_support"].count("aligned(64)") == 4
    hook = result["c_support"].split("void case0_dump")[0]
    assert "printf(" not in hook and "mcycle" in hook and "minstret" in hook and hook.count('"memory"') == 2
    assert result["manifest"]["qualification"] == "UNQUALIFIED"


def test_loop_header_first_task_operation_is_refused():
    text = NESTED.replace('llvm.br ^outer(%zero : i64) {merlin.global_task = 0 : i64}',
                          'llvm.br ^outer(%zero : i64) {merlin.global_task = -1 : i64}')
    with pytest.raises(ValueError, match="marker would repeat"):
        instrument_task_windows(text, plan())


def test_noncontiguous_disjoint_task_body_is_refused():
    text = NESTED.replace('merlin.global_task = 0 : i64, prov.region_id',
                          'merlin.global_task = 1 : i64, prov.region_id')
    with pytest.raises(ValueError, match="task.*order|noncontiguous|disjoint"):
        instrument_task_windows(text, plan())


def test_missing_declared_task_is_refused():
    with pytest.raises(ValueError, match="planned task|entire planned task"):
        instrument_task_windows(NESTED, plan(3))


def test_symbol_collision_and_invalid_prefix_are_refused():
    result = instrument_task_windows(NESTED, plan(), symbol_prefix="case0")
    with pytest.raises(ValueError, match="overlap"):
        instrument_task_windows(result["lowered_mlir_text"], plan(), symbol_prefix="case0")
    with pytest.raises(ValueError, match="identifier"):
        instrument_task_windows(NESTED, plan(), symbol_prefix="bad-prefix")


def test_multiline_tagged_instruction_is_refused_instead_of_guessing_an_insertion():
    text = NESTED.replace('llvm.br ^outer(%zero : i64) {merlin.global_task = 0 : i64}',
                          'llvm.br ^outer(%zero : i64) {\nmerlin.global_task = 0 : i64}')
    with pytest.raises(ValueError, match="one complete tagged LLVM operation per line"):
        instrument_task_windows(text, plan())
