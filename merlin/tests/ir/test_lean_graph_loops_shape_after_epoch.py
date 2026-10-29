"""SDK-free checked-middle preparation; final memory matcher stays in the TCB."""

import json
import os
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser

from merlin.verify.lean_graph_loops_shape_after import prepare_pair
from merlin.verify.lean_graph_loops_shape_after_epoch import obligation, prepare_chain


def parser(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


def buffers():
    selected = os.environ.get("MERLIN_GRAPH_SHAPE_AFTER_CASES")
    if not selected:
        pytest.skip("select the retained actual four-buffer cases explicitly")
    root = Path(selected) / "two_epochs"
    return [
        (root / relative).read_text()
        for relative in ("source.mlir", "shape/after.mlir", "memory/middle.mlir", "memory/after.mlir")
    ]


def test_actual_middle_is_checked_against_existing_complete_heap_adapter(tmp_path):
    result = prepare_chain(*buffers(), parser=parser, index_bits=64, max_nodes=5000)
    (tmp_path / "Obligation.lean").write_text(result["query"])
    (tmp_path / "PREPARATION.json").write_text(json.dumps({k: v for k, v in result.items() if k != "query"}, indent=2))
    assert "theorem middle_binding : plan.module = After.permuteModule" in result["query"]
    assert "After.actual_middle_complete GraphShapeAfterObligation.actualAfter" in result["query"]
    assert result["actual_shape_after_independently_decoded"]
    assert not result["actual_memory_final_independently_interpreted"]
    assert result["memory_final_python_matcher_is_trusted"]
    assert not any(result[key] for key in ("lean_invoked", "lean_accepted", "qualified", "pipeline_receipt_issued"))
    printed = [
        line.removeprefix("#print axioms ")
        for line in result["query"].splitlines()
        if line.startswith("#print axioms ")
    ]
    assert set(printed) == set(result["expected_roots"])


def test_wrong_actual_middle_cannot_hide_behind_unchanged_epoch_blueprint(tmp_path):
    source, normalized, reused, segmented = buffers()
    good = prepare_chain(source, normalized, reused, segmented, parser=parser, index_bits=64, max_nodes=5000)
    module = parser(normalized)
    ret = next(op for op in module.body.block.first_op.walk() if isinstance(op, func.ReturnOp))
    ret.operands = (module.body.block.first_op.body.block.args[0], ret.operands[1])
    module.verify()
    from io import StringIO

    from xdsl.printer import Printer

    output = StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    wrong = prepare_pair(source, output.getvalue(), parser=parser, index_bits=64, max_nodes=5000)
    query, _ = obligation(good["adapter"], wrong["model"])
    (tmp_path / "WrongMiddle.lean").write_text(query)
    assert query != good["query"]
    assert "GraphShapeAfterObligation.actualAfter" in query
    # This is only a changed proof obligation. No native rejection has run.
    assert not wrong["lean_invoked"]
