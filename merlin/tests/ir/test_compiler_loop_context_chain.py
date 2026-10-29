"""Full actual sibling-loop tiling and explicit unroll-tail/tiling composition."""

import json
from pathlib import Path

import pytest
from test_compiler_loop_context import pair, source

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.loops import segment_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import loop_transform_gate as gate


def buffers(*, chain=True, effects=True):
    if chain:
        before, middle = pair(effects=effects)
    else:
        middle = ir_text(parse_ir(source(effects=effects)))
        before = middle
    module = parse_ir(middle)
    segment_loops(module, (3,), index_bits=64)
    after = ir_text(module)
    return (before, middle, after) if chain else (before, after)


@pytest.mark.parametrize("chain", [False, True])
def test_actual_context_tile_and_chain_reconstruct_without_producer_replay(monkeypatch, chain):
    actual = buffers(chain=chain)
    monkeypatch.setattr("merlin.transforms.loops.segment_loops", lambda *a, **k: pytest.fail("producer replay"))
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("producer replay")
    )
    prepared = intake.prepare(actual, kinds=(intake.UNROLL, intake.TILE) if chain else (intake.TILE,))
    assert intake.refresh(prepared) == prepared
    assert prepared.actual_target == actual[-1]
    assert "Merlin.GraphGateProfileAdjacent.complete3" in prepared.roots
    capture = json.loads(prepared.capture_json)
    assert [row["boundary"] for row in capture["stages"]] == ([2, 2, 3, 3] if chain else [3, 3])
    if chain:
        assert "Merlin.GraphGateProfileAdjacent.complete2" in prepared.roots
        assert "middleEntries" in prepared.query
        assert len(prepared.roots) == 410
    else:
        assert len(prepared.roots) == 201


def test_generated_index_input_requires_complete_prior_actual_boundary():
    from merlin.verify.compiler_loop_context import prepare_tile_query

    _, middle, after = buffers()
    with pytest.raises(ValueError, match="source index arithmetic"):
        prepare_tile_query(middle, after, tile=3)


@pytest.mark.parametrize("chain", [False, True])
def test_full_actual_context_tile_and_chain_gate(tmp_path, pass_gate_tools, chain):
    actual = buffers(chain=chain)
    prepared = intake.prepare(actual, kinds=(intake.UNROLL, intake.TILE) if chain else (intake.TILE,))
    token = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "check", timeout=300
    )
    received = []
    gate.deliver(token, prepared, received.append)
    assert received == [actual[-1]]
    report = json.loads(Path(token.report).read_text())
    assert set(report["audit"]["axioms_used"]) <= gate.STD3
    if chain:
        for root in ("complete2", "complete3"):
            statement = report["statements"]["Merlin.GraphGateProfileAdjacent." + root]
            assert "PreservesEntries" in statement
            assert "middleEntries" in statement
        assert "originalEntries" in report["statements"]["Merlin.GraphGateProfileAdjacent.complete2"]
        assert "finalEntries" in report["statements"]["Merlin.GraphGateProfileAdjacent.complete3"]


@pytest.mark.parametrize("mutation", ["inner-step", "body", "middle"])
def test_actual_chain_malformed_control_body_and_prior_output_are_lean_refused(tmp_path, pass_gate_tools, mutation):
    from xdsl.dialects import arith, scf
    from xdsl.rewriter import Rewriter

    before, middle, after = buffers()
    module = parse_ir(middle if mutation == "middle" else after)
    if mutation == "inner-step":
        outer = next(op for op in module.walk() if type(op) is scf.ForOp)
        inner = next(op for op in outer.body.block.ops if type(op) is scf.ForOp)
        inner.operands = (inner.lb, inner.ub, outer.step, *inner.iter_args)
    else:
        region = module if mutation == "middle" else next(op for op in module.walk() if type(op) is scf.ForOp)
        addition = next(op for op in region.walk() if type(op) is arith.AddiOp and str(op.result.type) == "i32")
        Rewriter.replace_op(addition, arith.SubiOp(addition.lhs, addition.rhs))
    module.verify()
    if mutation == "middle":
        middle = ir_text(module)
        segment_loops(module, (3,), index_bits=64)
    after = ir_text(module)
    prepared = intake.prepare((before, middle, after), kinds=(intake.UNROLL, intake.TILE))
    with pytest.raises(gate.NotEstablished):
        gate.check(prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300)


def test_real_compiler_uses_one_complete_chain_gate_and_native_output(tmp_path, pass_gate_tools):
    import ctypes

    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms, pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    text = source(effects=False)
    options = compiler_transforms.Options(unroll_factor=2, tile_size=3)
    baseline = lower.lower_model(text, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({intake.UNROLL, intake.TILE}))
    checked = lower.lower_model(
        text, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    native = tmp_path / "context-chain-native"
    native.mkdir()
    library = shared_library(checked, native)
    library.forward.argtypes = (ctypes.c_int32, ctypes.c_int32)
    library.forward.restype = ctypes.c_int32
    for x, y in ((3, 5), (2**31 - 4, 17), (-(2**31), -23)):
        assert library.forward(x, y) == ctypes.c_int32(x + 8 * y).value


def test_empty_sibling_tiling_keeps_complete_context(tmp_path, pass_gate_tools):
    module = parse_ir(source().replace("%small=arith.constant 2:index", "%small=arith.constant 0:index"))
    before = ir_text(module)
    segment_loops(module, (3,), index_bits=64)
    after = ir_text(module)
    prepared = intake.prepare((before, after), kinds=(intake.TILE,))
    token = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "check", timeout=300
    )
    received = []
    gate.deliver(token, prepared, received.append)
    assert received == [after]
