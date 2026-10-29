"""Source/data regressions for the complete independently decoded final IR.

Python/xDSL verification is not an LLVM verifier or Lean rejection result. Every
case retains a complete UNCOMPILED obligation; no runtime context is constructed.
"""

import copy
import io
import json
import os
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.verify.lean_graph_loops_memory_scf_after_plan import prepare_plan
from merlin.verify.lean_graph_loops_shape_after_projection import normalized_code
from merlin.verify.lean_graph_loops_shape_projection import Decoder


def parser(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


def _permuted(value, swaps):
    """Diagnostic only. The actual Lean term is never built from this result."""
    if isinstance(value, list):
        return [_permuted(item, swaps) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"name", "kind", "known"}:
        name = value["name"]
        for left, right in reversed(swaps):
            name = right if name == left else left if name == right else name
        return dict(value, name=name)
    return {key: _permuted(item, swaps) for key, item in value.items()}


def _mutate(module, case):
    kernel = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "kernel")
    operations = list(kernel.body.block.ops)
    loads = [op for op in operations if isinstance(op, memref.LoadOp)]
    regions = [op for op in operations if isinstance(op, scf.ForOp)]
    zero, one = [
        op.result for op in operations
        if isinstance(op, arith.ConstantOp) and isinstance(op.result.type, builtin.IndexType)
    ][:2]
    if case == "buffer":
        original = next(op for op in operations if isinstance(op, memref.AllocOp))
        additional = original.clone()
        kernel.body.block.insert_op_before(additional, loads[0])
        loads[0].operands = (additional.results[0], *loads[0].operands[1:])
    elif case == "address":
        # The read after the update now selects a different, still in-bounds cell.
        loads[1].operands = (loads[1].operands[0], one, zero)
    elif case == "initializer":
        store = next(op for op in regions[0].walk() if isinstance(op, memref.StoreOp))
        store.operands = (kernel.body.block.args[1], *store.operands[1:])
    elif case == "initializer_address":
        store = next(op for op in regions[0].walk() if isinstance(op, memref.StoreOp))
        store.operands = (*store.operands[:2], zero, store.operands[3])
    elif case == "control":
        inner = [op for op in regions[0].walk() if isinstance(op, scf.ForOp)][1]
        inner.operands = (inner.operands[0], zero, *inner.operands[2:])
    elif case == "return":
        returned = next(op for op in operations if isinstance(op, func.ReturnOp))
        returned.operands = (kernel.body.block.args[0], returned.operands[1])
    elif case == "free":
        release = next(op for op in operations if isinstance(op, memref.DeallocOp))
        release.detach()
        kernel.body.block.insert_op_before(release, regions[1])
    elif case == "observer":
        touch = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "touch")
        addition = next(op for op in touch.walk() if isinstance(op, arith.AddiOp))
        addition.operands = (addition.operands[0], addition.operands[0])
    else:
        raise AssertionError("unknown fixed mutation case")


@pytest.mark.parametrize("case", (
    "buffer", "address", "initializer", "initializer_address", "control", "return", "free", "observer",
))
def test_complete_wrong_final_obligation(case, monkeypatch, tmp_path):
    raw_path = os.environ.get("MERLIN_GRAPH_MEMORY_AFTER_ACTUAL")
    hint_path = os.environ.get("MERLIN_GRAPH_MEMORY_AFTER_HINT")
    if not raw_path or not hint_path:
        pytest.skip("select exact retained actual final and source-plan hint")

    def forbidden(*args, **kwargs):
        raise AssertionError("old whole-tree matcher cannot supply actual final correspondence")

    for method in ("block", "project", "operands"):
        monkeypatch.setattr(Decoder, method, forbidden)
    raw = Path(raw_path).read_text()
    hint = json.loads(Path(hint_path).read_text())
    original_hint = copy.deepcopy(hint)
    module = parser(raw)
    _mutate(module, case)
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    changed = stream.getvalue()
    good = prepare_plan(hint["adapter"], hint["pair"]["afterFrame"], raw, parser=parser, index_bits=64, max_nodes=5000)
    bad = prepare_plan(hint["adapter"], hint["pair"]["afterFrame"], changed, parser=parser, index_bits=64, max_nodes=5000)
    assert hint == original_hint
    assert bad["expansion"] == good["expansion"]  # mutation cannot edit the source plan
    assert bad["actual_final"]["model"]["module"] != good["actual_final"]["model"]["module"]
    expanded = bad["expansion"]["module"]
    normalized = dict(expanded, functions=[
        dict(function, code=normalized_code(function["code"], {})) for function in expanded["functions"]
    ])
    assert normalized != _permuted(bad["actual_final"]["model"]["module"], bad["swaps"])
    assert "GraphMemoryAfterMaterialization.actualFinal" in bad["query"]
    assert "expansions.map Expansion.source = kernel.commands" in bad["query"]
    assert not bad["source_plan_semantics_proved"] and not bad["complete_memory_bridge_proved"]
    assert not bad["kernel_checked"] and not bad["qualified"] and not bad["pipeline_receipt_issued"]
    (tmp_path / "original-final.mlir").write_text(raw)
    (tmp_path / "changed-final.mlir").write_text(changed)
    (tmp_path / "Positive.lean").write_text(good["query"])
    (tmp_path / "Changed.lean").write_text(bad["query"])
    (tmp_path / "PREPARATION.json").write_text(json.dumps({
        "case": case,
        "xdsl_verified": True,
        "llvm_verified": False,
        "expected_kernel_acceptance": False,
        "kernel_outcome": "UNOBSERVED",
        "lean_status": "UNCOMPILED",
        "source_hint_unchanged": True,
        "python_normalized_syntax_differs": True,
        "semantic_counterexample_proved": False,
        "prepared": {key: value for key, value in bad.items() if key != "query"},
    }, indent=2) + "\n")
