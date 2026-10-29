"""Focused SDK-free actual-final binding; no claim of a completed bridge."""

import io
import json
import os
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.verify.lean_graph_loops_memory_scf_after import prepare_final
from merlin.verify.lean_graph_loops_shape_projection import Decoder


def parser(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


def actual():
    selected = os.environ.get("MERLIN_GRAPH_MEMORY_AFTER_ACTUAL")
    if not selected:
        pytest.skip("select the exact retained two-epoch actual final IR")
    return Path(selected).read_text()


def prepare(raw):
    return prepare_final(raw, parser=parser, index_bits=64, max_nodes=5000)


def test_complete_actual_final_has_dynamic_controls_and_all_functions(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("source/target matching must not supply the final term")

    for method in ("block", "project", "operands"):
        monkeypatch.setattr(Decoder, method, forbidden)
    result = prepare(actual())
    assert [row["name"] for row in result["model"]["module"]["functions"]] == ["kernel", "touch", "untouched"]
    assert result["model"]["dynamic_regions"] == 2
    assert {opcode for _, opcode in result["model"]["index_opcodes"]} == {"add", "minimum"}
    assert len(result["initializers"]) == 2
    assert "actualFinal : Module" in result["query"] and "initializer1_syntax" in result["query"]
    assert "initializer1_control_facts" in result["query"]
    assert result["query_version"] == "actual-final-local-tiled-v2"
    assert not result["local_initializer_root_kernel_checked"]
    assert not any(line.startswith("#") for line in result["query"].splitlines())
    assert not result["complete_memory_bridge_proved"] and not result["kernel_checked"]
    (tmp_path / "ActualFinal.lean").write_text(result["query"])
    (tmp_path / "PREPARATION.json").write_text(json.dumps({k: v for k, v in result.items() if k != "query"}, indent=2))


@pytest.mark.parametrize("mutation", ("add_operand", "minimum_operand", "inner_bound", "return", "observer"))
def test_valid_wrong_final_changes_actual_operational_term(mutation, tmp_path):
    raw = actual()
    good = prepare(raw)
    module = parser(raw)
    kernel = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    controls = [
        op for op in kernel.walk() if isinstance(op, arith.ConstantOp) and isinstance(op.result.type, builtin.IndexType)
    ]
    if mutation == "add_operand":
        op = next(
            op for op in kernel.walk() if isinstance(op, arith.AddiOp) and isinstance(op.result.type, builtin.IndexType)
        )
        op.operands = (op.operands[0], controls[1].result)
    elif mutation == "minimum_operand":
        op = next(op for op in kernel.walk() if isinstance(op, arith.MinUIOp))
        op.operands = (op.operands[0], controls[3].result)
    elif mutation == "inner_bound":
        loops = [op for op in kernel.walk() if isinstance(op, scf.ForOp)]
        loops[1].operands = (loops[1].operands[0], controls[2].result, *loops[1].operands[2:])
    elif mutation == "return":
        op = next(op for op in kernel.walk() if isinstance(op, func.ReturnOp))
        op.operands = (kernel.body.block.args[0], op.operands[1])
    else:
        observer = next(
            op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "touch"
        )
        op = next(op for op in observer.walk() if isinstance(op, arith.AddiOp))
        op.operands = (op.operands[0], op.operands[0])
    module.verify()
    text = io.StringIO()
    Printer(stream=text, print_generic_format=True).print_op(module)
    changed = text.getvalue()
    bad = prepare(changed)
    assert bad["model"]["module"] != good["model"]["module"]
    assert bad["query"] != good["query"]
    assert not bad["kernel_checked"] and not bad["qualified"]
    (tmp_path / "wrong-final.mlir").write_text(changed)
    (tmp_path / "WrongFinal.lean").write_text(bad["query"])
