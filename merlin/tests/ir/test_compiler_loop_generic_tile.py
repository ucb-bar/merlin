"""Actual singleton generic TILE boundaries, complete errors/world and custody."""

import json
from pathlib import Path

import pytest
from test_compiler_loop_generic import SOURCE

from merlin.transforms.certificates import ir_text
from merlin.transforms.loops import segment_loops
from merlin.verify import compiler_loop_generic as gate
from merlin.verify.compiler_loop_generic_profile import OutsideProfile, source_profile
from merlin.verify.lean_graph_generic_parser import parse_generic


def pair(tile=32, upper=257):
    module = parse_generic(SOURCE.replace("5:index", str(upper) + ":index"))
    before = ir_text(module)
    assert segment_loops(module, (tile,), index_bits=64)
    return before, ir_text(module)


@pytest.mark.parametrize("tile,upper", [(2, 5), (32, 257), (128, 65537), (65536, 5)])
def test_supported_tile_source_and_actual_target_are_prepared_without_producer_replay(tile, upper, monkeypatch):
    before, after = pair(tile, upper)
    monkeypatch.setattr("merlin.transforms.loops.segment_loops", lambda *a, **k: pytest.fail("producer replayed"))
    assert source_profile(before, kind=gate.TILE, tile=tile)["supported"]
    value = gate.prepare(before, after, kind=gate.TILE, tile=tile)
    gate.ready(value)
    assert gate.adjacent(gate.TILE) in value.roots
    assert value.describe()["kind"] == gate.TILE
    assert "PreservesEntries" in value.query
    assert len(value.query.encode()) < 32768


@pytest.mark.parametrize("tile", [True, 1, 65537])
def test_explicit_tile_policy_excludes_before_producer(tile):
    with pytest.raises(OutsideProfile):
        source_profile(SOURCE, kind=gate.TILE, tile=tile)


def test_generic_tile_remains_singleton_and_does_not_claim_unroll_tile_chain():
    from merlin.transforms.unrolling import unroll_ordered_loops

    module = parse_generic(SOURCE)
    assert unroll_ordered_loops(module, (2,), index_bits=64)
    with pytest.raises(OutsideProfile):
        source_profile(ir_text(module), kind=gate.TILE, tile=32)


def test_public_compiler_tile32_canonical_worker_and_fresh_replay(tmp_path, pass_gate_tools):
    from merlin.verify import compiler_transforms as formal
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    before, after = pair()
    expected = gate.prepare(before, after, kind=gate.TILE, tile=32)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=900)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.TILE}))
    with pass_validation.validating(selected):
        actual, _ = formal.apply(before, tmp_path / "compiler", options=formal.Options(tile_size=32))
    assert actual == after
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["kind"] == gate.TILE and detail["compiler_enforcement"] is True
    assert Path(detail["delivered"]["path"]).read_bytes() == after.encode()
    saved = json.loads(Path(detail["receipt"]).read_bytes())
    assert saved["candidate"] == expected.describe()
    assert gate.adjacent(gate.TILE) in saved["statements"]
    assert set(saved["audit"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    token, value = gate.replay(
        detail["receipt"], lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "replay"
    )
    assert value == expected and gate.deliver(token, value) == after


def test_supported_generic_tile_wrong_inner_bound_blocks_actual_worker(tmp_path, pass_gate_tools):
    from xdsl.dialects import scf

    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    before, after = pair()
    assert source_profile(before, kind=gate.TILE, tile=32)["supported"]
    module = parse_generic(after)
    outer, inner = [op for op in module.walk() if type(op) is scf.ForOp]
    inner.operands = (inner.lb, outer.ub, inner.step, *inner.iter_args)
    module.verify()
    wrong = ir_text(module)
    value = gate.prepare(before, wrong, kind=gate.TILE, tile=32)
    assert gate.ROOT in value.roots
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=900)
    try:
        worker.submit(
            gate.CHECK,
            before=before,
            after=wrong,
            params=gate.job_params(sdk_bin=pass_gate_tools.sdk_bin, kind=gate.TILE, tile=32),
            label="wrong-actual-inner-upper",
        )
        with pytest.raises(GateFailure):
            worker.join()
    finally:
        worker.close()
    assert len(worker.verdicts) == 1
    assert worker.verdicts[0]["verdict"] in {"not_established", "unavailable"}
    proofs = list((tmp_path / "worker").glob("*/generic-loop-proof"))
    assert len(proofs) == 1
    proof = proofs[0]
    assert json.loads((proof / "query.process.json").read_text())["returncode"] == 1
    assert all(json.loads(path.read_text())["returncode"] == 0 for path in proof.glob("import-*.process.json"))
    diagnostic = (proof / "query.stdout").read_text()
    assert ": error:" in diagnostic and "maximum number of heartbeats" not in diagnostic
    assert not (proof / "REPORT.json").exists()
