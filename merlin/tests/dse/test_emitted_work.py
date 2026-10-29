"""Exact emitted work summaries remain cheap for large counted programs."""

import pytest
from xdsl.parser import Parser

from merlin.perf.emitted_work import summarize_emitted_work
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context


def function(rows=3, columns=7, *, control=None):
    extra = "" if control is None else f"%value = {control}\nllvm.cond_br %value, ^done, ^start\n^start:"
    source = f'''module {{ llvm.func @work(%input: !llvm.ptr) {{
      {extra}
      %zero = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %rows = llvm.mlir.constant({rows} : i64) : i64
      %columns = llvm.mlir.constant({columns} : i64) : i64
      llvm.br ^outer(%zero : i64)
    ^outer(%i: i64):
      %more = llvm.icmp "ult" %i, %rows : i64
      llvm.cond_br %more, ^tile, ^done
    ^tile:
      llvm.br ^inner(%zero : i64)
    ^inner(%j: i64):
      %again = llvm.icmp "ult" %j, %columns : i64
      llvm.cond_br %again, ^body, ^latch
    ^body:
      llvm.inline_asm has_side_effects asm_dialect = att "command $0", "r" %j : (i64) -> ()
      %next = llvm.add %j, %one : i64
      llvm.br ^inner(%next : i64)
    ^latch:
      %following = llvm.add %i, %one : i64
      llvm.br ^outer(%following : i64)
    ^done:
      llvm.return
    }} }}'''
    module = Parser(make_llvm_context(), source).parse_module()
    return next(op for op in module.walk() if op.name == "llvm.func")


def classify(op):
    return "issue" if op.name == "llvm.inline_asm" else None


@pytest.mark.parametrize("rows,columns", [(3, 7), (0, 7), (3, 0), (1, 1)])
def test_counts_match_original_dynamic_control(rows, columns):
    f = function(rows, columns)
    result = summarize_emitted_work(f, command_kind=classify)
    assert result["command_counts"]["issue"] == rows * columns
    assert result["operation_counts"]["llvm.cond_br"] == rows + 1 + rows * (columns + 1)
    assert result["command_classification_complete"]
    assert result["requested_transfer_bytes"] is result["physical_cycles"] is None
    assert not result["performance_qualification"]


def test_billion_command_domain_uses_static_proof_budget():
    result = summarize_emitted_work(function(100_000, 100_000), command_kind=classify,
                                    max_operations=128, max_blocks=32)
    assert result["command_counts"] == {"issue": 10_000_000_000}
    assert result["static_ir_operations"] < 32


def test_unclassified_commands_remain_visible():
    result = summarize_emitted_work(function())
    assert result["unclassified_command_instances"] == 21
    assert result["command_counts"] == {}
    assert not result["command_classification_complete"]


def test_unknown_control_does_not_return_partial_counts():
    with pytest.raises(ValueError, match="tensor data|opaque"):
        summarize_emitted_work(function(control="llvm.load %input : !llvm.ptr -> i1"), command_kind=classify)


def test_unreasonable_budget_and_invalid_classifier_refuse():
    with pytest.raises(ValueError, match="bounded analysis"):
        summarize_emitted_work(function(), max_operations=True)
    with pytest.raises(ValueError, match="classification"):
        summarize_emitted_work(function(), command_kind=lambda op: 3)


def test_ordinary_candidate_diagnostics_expose_selected_entry_work():
    from merlin.perf.candidate_diagnostics import observe_candidate_artifacts

    result = observe_candidate_artifacts(emitted_mlir=str(function()), entry_symbol="work")
    row = result["host"]["functions"][0]
    work = row["emitted_work"]
    assert row["selected_entry"]
    assert work["status"] == "observed"
    assert work["command_counts"] == {"asm:command $0": 21}
    assert work["physical_cycles"] is None
