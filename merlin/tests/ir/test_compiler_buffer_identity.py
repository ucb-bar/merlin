"""Identity prefixes retain complete memory proofs and precise coverage reports."""

import json
from pathlib import Path

import pytest
from buffer_reuse_qualification_cases import BufferCase

from merlin.transforms import loops, normalization, planning
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.verify import buffer_reuse_gate as gate
from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare_chain
from merlin.verify.lean_graph_loops_shape_capture import export
from merlin.verify.lean_graph_loops_shape_projection import Decoder


def chain(*, normalized, epochs):
    module = parse_ir(BufferCase("identity-prefix", 2, 2, 2, epochs=epochs).source)
    if normalized:
        normalization.normalize_shapes(module)
    buffers = [ir_text(module)]
    normalization.normalize_shapes(module)
    buffers.append(ir_text(module))
    planning.reuse_storage(module, planning.plan_storage(module))
    buffers.append(ir_text(module))
    loops.segment_loops(module, (2,), index_bits=64)
    buffers.append(ir_text(module))
    return tuple(buffers)


@pytest.mark.parametrize("normalized,epochs", [(True, 2), (False, 1), (True, 1)])
def test_actual_identity_prefixes_prepare(normalized, epochs):
    buffers = chain(normalized=normalized, epochs=epochs)
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    assert (buffers[0] == buffers[1]) == normalized
    assert (buffers[1] == buffers[2]) == (epochs == 1)
    identity_roots = [root for root in candidate.roots if ".Identity." in root]
    assert len(identity_roots) == (4 if normalized else 0)
    assert prepare_chain(buffers, allocator=ALLOCATOR) == candidate


def test_standalone_positive_normalization_still_refuses_identity():
    raw = chain(normalized=True, epochs=2)[0]
    decoded = export(parse_ir(raw))
    with pytest.raises(ValueError, match="identity/declined"):
        Decoder(decoded, decoded, index_bits=64, max_nodes=5000).project()
    projection = Decoder(decoded, decoded, index_bits=64, max_nodes=5000).project(allow_identity=True)
    assert projection["changed_sites"] == [] and projection["identity_normalization"] is True
    assert projection["source_binding"] == projection["target_binding"]


def test_identity_flag_cannot_hide_changed_caller():
    buffers = list(chain(normalized=True, epochs=2))
    buffers[1] = buffers[1].replace("arith.addi", "arith.subi", 1)
    with pytest.raises(ValueError, match="changed|different|actual|mismatch"):
        prepare_chain(tuple(buffers), allocator=ALLOCATOR)


@pytest.mark.parametrize("normalized,epochs", [(True, 2), (False, 1), (True, 1)])
def test_identity_prefixes_check_and_deliver(tmp_path, pass_gate_tools, normalized, epochs):
    buffers = chain(normalized=normalized, epochs=epochs)
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    result = gate.check(
        candidate, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "proof", timeout=300
    )
    assert result.status == "checked", result.reason
    delivered = []
    gate.deliver(result.completion, candidate, delivered.append)
    assert delivered == [buffers[-1]]
    report = json.loads(Path(result.completion.report).read_text())
    assert set(report["audit"]["axioms_used"]) <= gate.STD3
    assert report["relation"] == "equal_possible_successful_public_observations"


def test_compiler_gate_on_already_normalized_single_epoch(tmp_path, pass_gate_tools):
    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms as transforms
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    raw = chain(normalized=True, epochs=1)[0]
    options = transforms.Options(tile_size=2, buffer_allocator=ALLOCATOR)
    baseline = lower.lower_model(raw, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1800)
    validation = pass_validation.PassValidation(
        worker, pass_gate_tools, frozenset({transforms.BUFFER, transforms.TILE})
    )
    checked = lower.lower_model(
        raw, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=validation
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    summary = checked.stats["formal_transforms"]
    assert summary["changed"] == [transforms.TILE]
    assert summary["unchanged"] == ["shape_view_normalization", transforms.BUFFER]
    assert summary["selected_checks"] == [transforms.TILE]
    assert worker.verdicts[0]["detail"]["full_error_or_world_equivalence"] is False
