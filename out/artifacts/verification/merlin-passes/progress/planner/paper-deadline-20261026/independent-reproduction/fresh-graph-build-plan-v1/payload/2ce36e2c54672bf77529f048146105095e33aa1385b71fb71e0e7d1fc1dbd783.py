"""Prepare full graph-chain candidates without Lean/native execution."""

import io
import json
import os
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.verify.lean_graph_loops_memory_scf_after_whole import prepare_whole
from merlin.verify.lean_graph_loops_shape_projection import Decoder


def parser(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


@pytest.mark.parametrize("case", (
    "positive", "source_return", "buffer", "address", "initializer", "initializer_address",
    "control", "return", "free", "observer",
))
def test_complete_common_interpretation_obligation(case, monkeypatch, tmp_path):
    selected = {key: os.environ.get(key) for key in (
        "MERLIN_GRAPH_MEMORY_AFTER_SOURCE", "MERLIN_GRAPH_MEMORY_AFTER_ACTUAL",
        "MERLIN_GRAPH_MEMORY_AFTER_HINT", "MERLIN_GRAPH_MEMORY_AFTER_MUTATIONS",
    )}
    if not all(selected.values()):
        pytest.skip("select retained raw source/final, untrusted plan and source-level mutations")
    raw_source = Path(selected["MERLIN_GRAPH_MEMORY_AFTER_SOURCE"]).read_text()
    raw_final = Path(selected["MERLIN_GRAPH_MEMORY_AFTER_ACTUAL"]).read_text()
    hint = json.loads(Path(selected["MERLIN_GRAPH_MEMORY_AFTER_HINT"]).read_text())
    if case == "source_return":
        source = parser(raw_source)
        kernel = next(op for op in source.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "kernel")
        returned = next(op for op in kernel.body.block.ops if isinstance(op, func.ReturnOp))
        returned.operands = (kernel.body.block.args[0], returned.operands[1])
        source.verify()
        stream = io.StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(source)
        raw_source = stream.getvalue()
    elif case != "positive":
        matches = [
            path for path in Path(selected["MERLIN_GRAPH_MEMORY_AFTER_MUTATIONS"]).rglob("PREPARATION.json")
            if json.loads(path.read_text()).get("case") == case
        ]
        assert len(matches) == 1
        raw_final = (matches[0].parent / "changed-final.mlir").read_text()
    parser(raw_source).verify()
    parser(raw_final).verify()

    def forbidden(*args, **kwargs):
        raise AssertionError("old pair matcher is not permitted to supply either boundary")

    for method in ("block", "project", "operands"):
        monkeypatch.setattr(Decoder, method, forbidden)
    result = prepare_whole(
        hint["adapter"], hint["pair"]["afterFrame"], raw_source, raw_final,
        parser=parser, index_bits=64, max_nodes=5000,
    )
    source_equal = result["independent_source"]["module"] == hint["adapter"]["shape"]["module"]
    assert source_equal == (case != "source_return")
    assert result["lean_status"] == "UNCOMPILED"
    assert not result["whole_run_equality_is_premise"] and not result["local_simulation_is_premise"]
    assert not result["kernel_checked"] and not result["qualified"] and not result["complete_memory_bridge_proved"]
    assert result["final_root"] == "GraphMemoryAfterWhole.complete_actual_chain"
    assert result["final_root"] not in result["support_roots"]
    assert len(result["support_roots"]) == len(set(result["support_roots"]))
    assert "source_index_absent" in result["query"] and "boundary_facts" in result["query"]
    assert not any(line.strip().startswith("#") for line in result["query"].splitlines())
    (tmp_path / "source.mlir").write_text(raw_source)
    (tmp_path / "final.mlir").write_text(raw_final)
    (tmp_path / "Whole.lean").write_text(result["query"])
    (tmp_path / "PREPARATION.json").write_text(json.dumps({
        "case": case,
        "expected_acceptance": case == "positive",
        "xdsl_verified": True,
        "llvm_verified": False,
        "kernel_outcome": "UNOBSERVED",
        "semantic_counterexample_proved": False,
        "source_term_equality_python_diagnostic": source_equal,
        "prepared": {key: value for key, value in result.items() if key != "query"},
    }, indent=2) + "\n")
