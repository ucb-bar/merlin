"""Bounded generic-value canonical intake, authority and actual worker controls."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_generic as gate
from merlin.verify.gate_errors import GateRejected, GateUnsupported
from merlin.verify.lean_graph_generic_parser import parse_generic

SOURCE = """builtin.module {
func.func private @prefix(tensor<4xf32>)->tensor<4xf32>
func.func private @tick(index,tensor<4xf32>)
func.func @forward(%left:tensor<4xf32>,%right:tensor<4xf32>)->tensor<4xf32> {
%r=func.call @prefix(%left):(tensor<4xf32>)->tensor<4xf32>
%l=arith.constant 0:index
%u=arith.constant 5:index
%s=arith.constant 1:index
scf.for %i=%l to %u step %s {func.call @tick(%i,%right):(index,tensor<4xf32>)->() scf.yield}
func.return %r:tensor<4xf32>
}}
"""


def pair():
    module = parse_generic(SOURCE)
    before = ir_text(module)
    assert unroll_ordered_loops(module, (2,), index_bits=64)
    return before, ir_text(module)


def test_canonical_full_pair_preserves_generic_domain_and_proof_closure(monkeypatch):
    before, after = pair()
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("replayed producer")
    )
    value = gate.prepare(before, after)
    gate.ready(value)
    assert value.domain == gate.DOMAIN
    assert gate.ROOT in value.roots and gate.ADJACENT in value.roots
    assert "Value" in value.query and "Codec Value" in value.query
    source_refs = {row["path"] for row in json.loads(value.sources_json)}
    proofs = gate.proof_sources(value.query)
    assert len(proofs) == 28 and {r["path"] for r in proofs} <= source_refs
    assert value.describe()["entries"] == ["forward"]


@pytest.mark.parametrize("field", ["query", "roots", "projection_json", "resolution_json"])
def test_changed_preparation_cannot_be_reopened_as_canonical(field):
    value = gate.prepare(*pair())
    altered = replace(value, **{field: value.roots[:-1] if field == "roots" else getattr(value, field) + " "})
    with pytest.raises(gate.Unavailable, match="canonical generic loop"):
        gate.ready(altered)


@pytest.mark.parametrize("unused_tile", [32, 65537])
def test_unroll_v1_capture_and_replay_request_ignore_unused_tile(unused_tile):
    before, after = pair()
    ordinary = gate.prepare(before, after)
    compatibility = gate.prepare(before, after, tile=unused_tile)
    assert compatibility == ordinary
    assert "tile" not in json.loads(compatibility.request_json)
    gate.ready(compatibility)


def test_saved_constructed_or_copied_tokens_have_no_delivery_authority(tmp_path):
    value = gate.prepare(*pair())
    fake = gate.Completion(value.identity, str(tmp_path / "REPORT.json"))
    for token in (fake, copy.copy(fake), copy.deepcopy(fake)):
        with pytest.raises(gate.Unavailable, match="no authority"):
            gate.deliver(token, value)


def test_supported_source_bad_target_is_blocking_before_sdk(tmp_path, monkeypatch):
    before, after = pair()
    # Valid generic IR with one complex literal is deliberately outside the
    # target projection, while the original source remains admitted.
    from xdsl.dialects import builtin, func

    module = parse_generic(after)
    forward = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.body.blocks)
    forward.attributes["test.tag"] = builtin.DenseIntOrFPElementsAttr(
        builtin.TensorType(builtin.ComplexType(builtin.f32), [1]), builtin.BytesAttr(bytes.fromhex("0000803F00000000"))
    )
    (tmp_path / "before.mlir").write_text(before)
    (tmp_path / "after.mlir").write_text(ir_text(module))
    monkeypatch.setattr(gate, "check_candidate", lambda *a, **k: pytest.fail("bad literal reached SDK"))
    job = {
        "before": str(tmp_path / "before.mlir"),
        "after": str(tmp_path / "after.mlir"),
        "directory": str(tmp_path),
        "params": gate.job_params(sdk_bin=tmp_path),
    }
    with pytest.raises(GateRejected, match="complex-floating-dense"):
        gate.check(job)


def test_job_schema_cannot_accept_saved_status_instead_of_original_selection(tmp_path):
    params = gate.job_params(sdk_bin=tmp_path)
    params["saved_status"] = "accepted"
    with pytest.raises(GateUnsupported, match="exact bounded generic"):
        gate.check({"params": params})


def test_bound_replay_reads_preserve_crlf_and_reject_transient_good_bytes(tmp_path, monkeypatch):
    paths = [tmp_path / "source.mlir", tmp_path / "final.mlir"]
    raw = [b"bad-source\r\n", b"target\r\n"]
    for path, data in zip(paths, raw, strict=True):
        path.write_bytes(data)
    refs = [gate.file_ref(path) for path in paths]
    assert tuple(value.encode() for value in gate._read_bound_buffers(refs)) == tuple(raw)
    original = Path.read_bytes
    calls = [0]

    def transient(path):
        if path == paths[0]:
            calls[0] += 1
            if calls[0] == 2:
                return b"transient-good-source\r\n"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", transient)
    with pytest.raises(gate.Unavailable, match="decoded bytes differ"):
        gate._read_bound_buffers(refs)


def test_replay_original_mutation_during_proof_blocks_before_delivery(tmp_path, monkeypatch):
    candidate = gate.prepare(*pair())
    originals = [tmp_path / "source.mlir", tmp_path / "final.mlir"]
    for path, text in zip(originals, (candidate.source, candidate.final), strict=True):
        path.write_bytes(text.encode())
    saved = {
        "schema": "merlin.compiler-generic-loop.checked.v1",
        "status": "accepted",
        "buffers": [gate.file_ref(path) for path in originals],
        "source_request": json.loads(candidate.request_json),
    }
    locator = tmp_path / "saved.json"
    locator.write_text(gate.canonical(saved))
    called = []

    def proof_control(actual, **kwargs):
        called.append(True)
        assert actual == candidate
        originals[0].write_bytes(originals[0].read_bytes() + b" ")
        return gate.Completion(actual.identity, str(tmp_path / "never-authoritative.json"))

    monkeypatch.setattr(gate, "check_candidate", proof_control)
    with pytest.raises(gate.Unavailable, match="bound configuration"):
        gate.replay(locator, lean=tmp_path / "lean", destination=tmp_path / "unused")
    assert called == [True]


def test_actual_fresh_kernel_live_delivery_and_replay(tmp_path, pass_gate_tools):
    value = gate.prepare(*pair())
    lean = Path(pass_gate_tools.sdk_bin) / "lean"
    from merlin.verify import compiler_transforms as formal
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    worker = SpeculativeGate(tmp_path / "worker", job_timeout=900)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.UNROLL}))
    with pass_validation.validating(selected):
        result, compiler_report = formal.apply(
            value.source, tmp_path / "compiler", options=formal.Options(unroll_factor=2)
        )
    assert result == value.final
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["domain"] == gate.DOMAIN and detail["compiler_enforcement"] is True
    assert detail["native_fidelity_proved"] is False
    assert Path(detail["delivered"]["path"]).read_bytes() == value.final.encode()
    report = json.loads(Path(detail["receipt"]).read_bytes())
    assert report["candidate"] == value.describe()
    assert set(report["audit"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    assert set(report["statements"]) == set(value.roots)
    assert "Value" in report["statements"][gate.ROOT]
    # A worker's saved report cannot produce a live token in this process.
    with pytest.raises(gate.Unavailable, match="no authority"):
        gate.deliver(gate.Completion(value.identity, detail["receipt"]), value)
    locators = tmp_path / "replay-inputs"
    locators.mkdir()
    rows = []
    for label, text in (("source", value.source), ("final", value.final)):
        path = locators / (label + ".mlir")
        path.write_bytes(text.encode())
        rows.append(gate.file_ref(path))
    saved = json.loads(Path(detail["receipt"]).read_bytes())
    saved.update(buffers=rows, status="rejected", statements={}, audit={})
    locator = locators / "LOCATOR.json"
    locator.write_text(gate.canonical(saved))
    other, reconstructed = gate.replay(locator, lean=lean, destination=tmp_path / "replay", timeout=300)
    assert reconstructed == value
    assert gate.deliver(other, reconstructed) == value.final
    # Retain the real proof artifacts; only copied replay-input locators mutate.
    path = Path(rows[0]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(gate.Unavailable, match="bound configuration"):
        gate.deliver(other, reconstructed)


def test_valid_wrong_capture_is_refused_by_lean_not_reported_unavailable(tmp_path, pass_gate_tools, monkeypatch):
    from xdsl.dialects import func

    from merlin.transforms import unrolling
    from merlin.verify import compiler_transforms as formal
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    original = unrolling.unroll_ordered_loops

    def wrong_producer(module, *args, **kwargs):
        # Second bulk tick reads the distinct same-typed %left, not captured %right.
        changed = original(module, *args, **kwargs)
        forward = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "forward")
        ticks = [op for op in forward.walk() if type(op) is func.CallOp and op.callee.root_reference.data == "tick"]
        assert changed and len(ticks) == 3
        ticks[1].operands = (ticks[1].operands[0], forward.body.block.args[0])
        module.verify()
        return changed

    monkeypatch.setattr(unrolling, "unroll_ordered_loops", wrong_producer)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=900)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.UNROLL}))
    try:
        with pytest.raises(GateFailure):
            with pass_validation.validating(selected):
                formal.apply(SOURCE, tmp_path / "compiler", options=formal.Options(unroll_factor=2))
    finally:
        worker.close()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "not_established"
    assert worker.verdicts[0]["reason"].startswith("generic loop query left generated obligations unsolved")
    proofs = list(worker.directory.glob("*/generic-loop-proof"))
    assert len(proofs) == 1
    proof = proofs[0]
    assert all(json.loads(path.read_text())["returncode"] == 0 for path in proof.glob("import-*.process.json"))
    assert json.loads((proof / "query.process.json").read_text())["returncode"] == 1
    assert ": error: unsolved goals" in (proof / "query.stdout").read_text()
    assert not (proof / "REPORT.json").exists()
    coverage = json.loads((worker.directory / "coverage.json").read_text())
    assert coverage["compiler_completed"] is False and coverage["verified_boundaries"] == 0
