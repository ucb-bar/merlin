"""Complete-module unroll composition across different entry modes."""

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import compiler_unroll_composition as composition


def mixed_source():
    from test_compiler_loop_generalization import source

    entries = [
        source(3, 2, name=name, trips=trips)
        for name, trips in (("empty_entry", 0), ("inline_entry", 2), ("tail_entry", 5), ("bulk_entry", 6))
    ]
    entries.append("""func.func @unchanged_entry(%upper:index,%x:i32,%y:i32) -> i32 {
      %lb = arith.constant 0 : index
      %st = arith.constant 1 : index
      %r = scf.for %i = %lb to %upper step %st iter_args(%c = %x) -> (i32) {
        %n = arith.addi %c, %y : i32
        scf.yield %n : i32
      }
      func.return %r : i32
    }""")
    return "builtin.module {" + "\n".join(entries) + "}"


def pair():
    module = parse_ir(mixed_source())
    before = ir_text(module)
    unroll_ordered_loops(module, (2,), index_bits=64)
    return before, ir_text(module)


def test_projection_keeps_complete_actual_roster_and_one_registry(monkeypatch):
    buffers = pair()
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("producer replay")
    )
    result = composition.prepare_query(*buffers, factor=2)
    capture = result["capture"]
    assert [row["mode"] for row in capture["fragments"]] == ["empty", "inline", "tail", "bulk", "unchanged"]
    assert len(capture["pipeline"]["stages"][2]["functions"]) == 5
    assert not any(row["declaration"] for row in capture["pipeline"]["stages"][2]["functions"])
    assert all(
        stage["registry"] == capture["pipeline"]["stages"][0]["registry"] for stage in capture["pipeline"]["stages"]
    )
    assert result["final_root"] == intake.ROOT
    assert "Merlin.GraphGateProfileAdjacent.complete2" in result["roots"]
    assert len(result["roots"]) == len(set(result["roots"]))
    assert capture["auxiliary_delivered"] is False


def test_mutated_dynamic_context_cannot_be_hidden_as_an_unchanged_fragment():
    from xdsl.dialects import arith, func
    from xdsl.rewriter import Rewriter

    before, after = pair()
    module = parse_ir(after)
    function = next(
        op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "unchanged_entry"
    )
    op = next(op for op in function.walk() if type(op) is arith.AddiOp)
    Rewriter.replace_op(op, arith.SubiOp(op.lhs, op.rhs))
    module.verify()
    with pytest.raises(ValueError, match="static"):
        composition.prepare_query(before, ir_text(module), factor=2)
