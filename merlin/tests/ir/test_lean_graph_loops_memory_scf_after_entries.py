"""SDK-free complete-roster preparation. No Lean outcome is observed here."""

import hashlib
import io
import json
import os
from pathlib import Path

import pytest
from xdsl.dialects import arith, builtin, func, memref
from xdsl.printer import Printer

from merlin.verify.lean_graph_loops_memory_scf_after_entries import prepare_entries
from merlin.verify.lean_graph_loops_shape_projection import Decoder
from test_lean_graph_loops_memory_scf_after_whole import parser


def _function(module, name):
    return next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == name)


def _print(module):
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _mutate(module, case):
    untouched = _function(module, "untouched")
    touch = _function(module, "touch")
    kernel = _function(module, "kernel")
    if case == "untouched_return":
        untouched.body.block.last_op.operands = (untouched.body.block.args[0],)
    elif case == "literal":
        constant = next(op for op in untouched.body.block.ops if isinstance(op, arith.ConstantOp))
        constant.properties["value"] = builtin.IntegerAttr(2, builtin.i32)
    elif case == "callee":
        replacement = parser('builtin.module { func.func @untouched(%input: i32) -> i32 { '
                             '%result = func.call @untouched(%input) : (i32) -> i32 '
                             'func.return %result : i32 } }')
        body = replacement.body.block.first_op
        replacement.body.block.detach_op(body)
        module.body.block.insert_op_before(body, untouched)
        module.body.block.erase_op(untouched)
    elif case == "arity":
        replacement = parser('builtin.module { func.func @untouched(%input: i32, %extra: i32) -> i32 { '
                             '%one = arith.constant 1 : i32 %result = arith.addi %input, %one : i32 '
                             'func.return %result : i32 } }')
        body = replacement.body.block.first_op
        replacement.body.block.detach_op(body)
        module.body.block.insert_op_before(body, untouched)
        module.body.block.erase_op(untouched)
    elif case == "caller_buffer":
        block = touch.body.block
        load = next(op for op in block.ops if isinstance(op, memref.LoadOp))
        allocation = memref.AllocOp.get(builtin.i32, shape=[1])
        block.insert_op_before(allocation, load)
        load.operands = (allocation.memref, *load.operands[1:])
    elif case == "interface":
        untouched.properties["sym_visibility"] = builtin.StringAttr("private")
    elif case == "observer":
        addition = next(op for op in touch.body.block.ops if isinstance(op, arith.AddiOp))
        addition.operands = (addition.operands[0], addition.operands[0])
    elif case == "kernel_return":
        returned = kernel.body.block.last_op
        returned.operands = (kernel.body.block.args[0], returned.operands[1])
    elif case in ("missing_entry", "declaration"):
        module.body.block.erase_op(untouched)
        if case == "declaration":
            declaration = parser('builtin.module { func.func private @untouched(i32) -> i32 }').body.block.first_op
            declaration.parent_block().detach_op(declaration)
            module.body.block.add_op(declaration)
    elif case == "wrapper":
        added = parser('builtin.module { func.func @wrapper(%a: i32, %b: i32, %caller: memref<1xi32>) '
                       '-> (i32, index) { %x, %y = func.call @kernel(%a, %b, %caller) '
                       ': (i32, i32, memref<1xi32>) -> (i32, index) func.return %x, %y : i32, index } }')
        wrapper = added.body.block.first_op
        added.body.block.detach_op(wrapper)
        module.body.block.add_op(wrapper)
    else:
        raise AssertionError(case)


@pytest.mark.parametrize("case", (
    "positive", "untouched_return", "literal", "callee", "caller_buffer", "interface",
    "observer", "kernel_return", "missing_entry", "declaration", "wrapper", "arity",
))
def test_complete_roster_actual_binding(case, monkeypatch, tmp_path):
    selected = {key: os.environ.get(key) for key in (
        "MERLIN_GRAPH_MEMORY_AFTER_SOURCE", "MERLIN_GRAPH_MEMORY_AFTER_ACTUAL", "MERLIN_GRAPH_MEMORY_AFTER_HINT",
    )}
    if not all(selected.values()):
        pytest.skip("select retained actual raw boundaries and untrusted plan")
    raw_source = Path(selected["MERLIN_GRAPH_MEMORY_AFTER_SOURCE"]).read_text()
    raw_final = Path(selected["MERLIN_GRAPH_MEMORY_AFTER_ACTUAL"]).read_text()
    original_final = raw_final
    hint = json.loads(Path(selected["MERLIN_GRAPH_MEMORY_AFTER_HINT"]).read_text())
    if case != "positive":
        final = parser(raw_final)
        _mutate(final, case)
        raw_final = _print(final)
        if case == "wrapper":
            source = parser(raw_source)
            _mutate(source, case)
            raw_source = _print(source)
    parser(raw_source).verify()
    parser(raw_final).verify()

    def forbidden(*args, **kwargs):
        raise AssertionError("old whole-tree pair matcher cannot supply either boundary")

    for name in ("block", "project", "operands"):
        monkeypatch.setattr(Decoder, name, forbidden)
    result = prepare_entries(hint["adapter"], hint["pair"]["afterFrame"], raw_source, raw_final,
                             parser=parser, index_bits=64, max_nodes=5000)
    roster = result["entry_roster"]
    assert [row["name"] for row in roster["rows"]] == [row["name"] for row in roster["source_headers"]]
    assert result["lean_status"] == "UNCOMPILED"
    assert not result["all_entries_proved"] and not result["kernel_checked"] and not result["qualified"]
    assert result["raw_source_sha256"] == hashlib.sha256(raw_source.encode()).hexdigest()
    assert result["raw_final_sha256"] == hashlib.sha256(raw_final.encode()).hexdigest()
    if case != "positive":
        assert raw_final != original_final
    assert result["execution_witness_roots"][-1] == "entry2_final_outcome"
    assert result["query_sha256"] == hashlib.sha256(result["query"].encode()).hexdigest()
    assert result["final_root"] == "GraphMemoryAfterEntries.every_definition"
    assert result["final_root"] not in result["support_roots"]
    assert not any(line.lstrip().startswith("#") for line in result["query"].splitlines())
    if case == "positive":
        assert not roster["refusals"]
        assert roster["kernel_reachable"] == ["kernel", "touch"]
        assert len(roster["source_callgraph"]["kernel"]) == 2
        assert not roster["source_callgraph"]["touch"] and not roster["source_callgraph"]["untouched"]
        assert [row["name"] for row in roster["source_headers"]] == ["kernel", "touch", "untouched"]
        assert len(roster["literals"]) == 1
    if case == "wrapper":
        assert any(row["code"] == "unproved_changed_kernel_caller_context" for row in roster["refusals"])
    if case in ("missing_entry", "interface", "declaration", "arity"):
        assert any(row["code"] == "boundary_roster_or_interface_changed" for row in roster["refusals"])
    if case == "declaration":
        assert any(row["code"] == "declaration_not_supported" for row in roster["refusals"])
    (tmp_path / "source.mlir").write_text(raw_source)
    (tmp_path / "final.mlir").write_text(raw_final)
    (tmp_path / "Entries.lean").write_text(result["query"])
    (tmp_path / "PREPARATION.json").write_text(json.dumps(dict(
        case=case, xdsl_verified=True, llvm_verified=False, kernel_outcome="UNOBSERVED",
        expected_finite_binding_acceptance=case == "positive", semantic_counterexample_proved=False,
        mutation_kind="binding change; no semantic rejection or wrong-result claim",
        prepared={key: value for key, value in result.items() if key != "query"},
    ), indent=2) + "\n")
