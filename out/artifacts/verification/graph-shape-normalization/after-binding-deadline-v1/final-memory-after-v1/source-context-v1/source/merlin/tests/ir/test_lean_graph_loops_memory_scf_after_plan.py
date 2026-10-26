"""SDK-free preparation of the complete finite expansion/IR obligations."""

import json
import os
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser

from merlin.verify.lean_graph_loops_memory_scf_after_plan import prepare_plan


def parser(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


def test_actual_plan_emits_all_commands_and_computed_target(tmp_path):
    raw_path = os.environ.get("MERLIN_GRAPH_MEMORY_AFTER_ACTUAL")
    hint_path = os.environ.get("MERLIN_GRAPH_MEMORY_AFTER_HINT")
    if not raw_path or not hint_path:
        pytest.skip("select exact retained actual final and source-plan hint")
    hint = json.loads(Path(hint_path).read_text())
    prepared = prepare_plan(
        hint["adapter"], hint["pair"]["afterFrame"], Path(raw_path).read_text(),
        parser=parser, index_bits=64, max_nodes=5000,
    )
    assert len(prepared["expansion"]["sites"]) == 27
    assert prepared["finite_expansion_binding_emitted"]
    assert not prepared["finite_expansion_binding_kernel_checked"]
    assert not prepared["source_plan_semantics_proved"]
    assert not prepared["complete_memory_bridge_proved"] and not prepared["qualified"]
    assert "expansions.map Expansion.source = kernel.commands" in prepared["query"]
    assert "expandedModule plan kernel expansions" in prepared["query"]
    assert "GraphMemoryAfterMaterialization.actualFinal" in prepared["query"]
    assert not any(line.strip().startswith("#") for line in prepared["query"].splitlines())
    assert len(prepared["support_roots"]) == len(set(prepared["support_roots"]))
    assert prepared["final_root"] not in prepared["support_roots"]
    (tmp_path / "CompleteExpansion.lean").write_text(prepared["query"])
    (tmp_path / "PREPARATION.json").write_text(json.dumps({k: v for k, v in prepared.items() if k != "query"}, indent=2))
