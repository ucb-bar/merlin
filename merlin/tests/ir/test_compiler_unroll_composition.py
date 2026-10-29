"""Complete-module unroll composition across different entry modes."""

import json
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import compiler_unroll_composition as composition
from merlin.verify import loop_transform_gate


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


def test_mixed_entries_check_and_deliver_complete_actual_output(tmp_path, pass_gate_tools):
    buffers = pair()
    prepared = intake.prepare(buffers, kinds=(intake.UNROLL,))
    assert intake.refresh(prepared) == prepared
    token = loop_transform_gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "proof", timeout=300
    )
    delivered = []
    loop_transform_gate.deliver(token, prepared, delivered.append)
    assert delivered == [buffers[-1]]
    report = json.loads(Path(token.report).read_text())
    assert set(report["audit"]["axioms_used"]) <= loop_transform_gate.STD3
    assert intake.ROOT in report["statements"]
    assert "PreservesEntries" in report["statements"][intake.ROOT]


@pytest.mark.parametrize("entry", ["inline_entry", "bulk_entry"])
def test_corruption_of_any_actual_entry_blocks_complete_gate(tmp_path, pass_gate_tools, entry):
    from xdsl.dialects import arith, func
    from xdsl.rewriter import Rewriter

    before, after = pair()
    module = parse_ir(after)
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == entry)
    op = next(op for op in function.walk() if type(op) is arith.AddiOp and str(op.result.type) == "i32")
    Rewriter.replace_op(op, arith.SubiOp(op.lhs, op.rhs))
    module.verify()
    prepared = intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))
    with pytest.raises(loop_transform_gate.NotEstablished):
        loop_transform_gate.check(
            prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300
        )


def test_compiler_gate_and_native_results_cover_every_entry(tmp_path, pass_gate_tools):
    import ctypes

    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms, pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    text = mixed_source()
    options = compiler_transforms.Options(unroll_factor=2)
    baseline = lower.lower_model(text, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({intake.UNROLL}))
    checked = lower.lower_model(
        text, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    library = shared_library(checked, tmp_path / "checked")
    for name, trips in (("empty_entry", 0), ("inline_entry", 2), ("tail_entry", 5), ("bulk_entry", 6)):
        function = getattr(library, name)
        function.argtypes, function.restype = [ctypes.c_int32] * 3, ctypes.c_int32
        for x, unused, y in ((7, 999, -3), (2**31 - 1, 0, 1)):
            assert function(x, unused, y) == ctypes.c_int32(2 * (x + trips * y)).value
    function = library.unchanged_entry
    function.argtypes, function.restype = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32], ctypes.c_int32
    assert function(5, 7, -3) == -8
