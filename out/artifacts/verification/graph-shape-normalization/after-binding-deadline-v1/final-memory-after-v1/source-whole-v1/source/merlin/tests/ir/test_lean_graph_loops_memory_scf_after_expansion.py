"""SDK-free actual materialization preparation, never kernel acceptance."""

import json
import os
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser

from merlin.verify.lean_graph_loops_memory_scf_after_expansion import prepare_expansion
from merlin.verify.lean_graph_loops_shape_after_projection import normalized_code
from merlin.verify.lean_graph_loops_shape_projection import Decoder


def parser(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


def permuted(value, swaps):
    """Diagnostic comparison only; a passing comparison is no proof receipt."""
    if isinstance(value, list):
        return [permuted(item, swaps) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"name", "kind", "known"}:
        name = value["name"]
        for left, right in reversed(swaps):
            name = right if name == left else left if name == right else name
        return dict(value, name=name)
    return {key: permuted(item, swaps) for key, item in value.items()}


def test_actual_final_matches_complete_expanded_target(monkeypatch, tmp_path):
    raw_path = os.environ.get("MERLIN_GRAPH_MEMORY_AFTER_ACTUAL")
    hint_path = os.environ.get("MERLIN_GRAPH_MEMORY_AFTER_HINT")
    if not raw_path or not hint_path:
        pytest.skip("select exact retained actual final and source-plan hint")

    def forbidden(*args, **kwargs):
        raise AssertionError("old whole-tree matcher must not bind actual final")

    for name in ("block", "project", "operands"):
        monkeypatch.setattr(Decoder, name, forbidden)
    hint = json.loads(Path(hint_path).read_text())
    result = prepare_expansion(
        hint["adapter"], hint["pair"]["afterFrame"], Path(raw_path).read_text(),
        parser=parser, index_bits=64, max_nodes=5000,
    )
    sites = result["expansion"]["sites"]
    assert len(sites) == len(hint["adapter"]["step_binding"]["steps"])
    assert [site["kind"] for site in sites].count("alias") == 1
    assert [site["kind"] for site in sites].count("empty-release") == 1
    assert [site["kind"] for site in sites].count("tiled-initializer") == 2
    assert result["expansion"]["final_physical_clock"] == 1
    expanded = result["expansion"]["module"]
    expected = dict(
        expanded,
        functions=[dict(function, code=normalized_code(function["code"], {})) for function in expanded["functions"]],
    )
    assert expected == permuted(result["actual_final"]["model"]["module"], result["swaps"])
    assert result["source_frame"] == result["actual_final"]["model"]["frame"]
    assert not result["source_plan_semantics_proved"] and not result["complete_memory_bridge_proved"]
    assert not result["kernel_checked"] and not result["qualified"]
    assert "expanded_to_actual" in result["query"]
    assert not any(line.startswith("#") for line in result["query"].splitlines())
    (tmp_path / "Materialization.lean").write_text(result["query"])
    (tmp_path / "PREPARATION.json").write_text(json.dumps(result, indent=2))
